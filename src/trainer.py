# src/trainer.py

import os
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm

# NEW AMP API (no FutureWarning)
from torch.amp import autocast, GradScaler

from src.losses import (
    rpgan_d_loss,
    rpgan_g_loss,
    r1_penalty,
    r2_penalty,
    conservative_physical_loss,
)

# Road utilities
from src.utils import load_road_tensors, denorm_xy_torch, road_loss_from_distmap


# -------------------------------------------------------------
#  EMA UPDATE
# -------------------------------------------------------------
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    ema = decay * ema + (1 - decay) * model
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_((1 - decay) * param.data)


# -------------------------------------------------------------
#  CHECKPOINT
# -------------------------------------------------------------
def save_checkpoint(G, D, G_ema, opt_G, opt_D, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_ema": G_ema.state_dict() if G_ema is not None else None,
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "step": step,
    }
    path = os.path.join(out_dir, f"checkpoint_step_{step}.pt")
    torch.save(ckpt, path)


# -------------------------------------------------------------
#  ROAD LAMBDA SCHEDULE (WARMUP)
# -------------------------------------------------------------
def linear_warmup(step: int, start: int, ramp: int, max_val: float) -> float:
    """
    step < start       -> 0
    start..start+ramp  -> linear 0..max_val
    > start+ramp       -> max_val
    """
    if max_val <= 0.0:
        return 0.0
    if step < start:
        return 0.0
    if ramp <= 0:
        return float(max_val)
    t = (step - start) / float(ramp)
    t = max(0.0, min(1.0, t))
    return float(max_val) * t


# -------------------------------------------------------------
#  TRAIN LOOP
# -------------------------------------------------------------
def train_ultra_stable_traffic_gan(
    G,
    D,
    dataloader,
    *,
    device,
    latent_dim,
    n_critic,
    gamma_r1r2,
    lambda_phys=0.0,

    # ROAD
    lambda_road=0.0,          # target weight (MAX)
    data_dir=None,            # DATA_DIR con road_dist.npy + normalization_params.pkl
    road_warmup_start=5000,   # step da cui inizia la road loss
    road_warmup_ramp=15000,   # step per arrivare a lambda_road
    road_power=2.0,           # power per road_loss_from_distmap
    road_norm_by_valid=True,  # normalizza per #valid steps (consigliato)

    # EMA
    use_ema=False,
    ema_decay=0.999,

    # LR
    g_lr=2e-4,
    d_lr=1e-4,

    # run
    epochs=10,
    out_dir="outputs/checkpoints",
    sample_callback=None,

    # Noise injection
    noise_std=0.0,
    noise_decay=0.0,
    noise_min=0.0,
    noise_on_real=True,
    noise_on_fake=True,

    r1r2_every=2,
):
    """
    dataloader -> real_X (B,T,4), S (B,4), lengths (B,)
    pad_mask: True dove t >= L

    Rumore: SOLO agli input del Discriminator e SOLO sui timestep validi.
    Road-loss: applicata SOLO al generatore, su XY denormalizzati in metri.
    """

    # --- sanity checks road ---
    if lambda_road > 0.0 and data_dir is None:
        raise ValueError(
            "Se lambda_road > 0 devi passare data_dir=DATA_DIR "
            "(contiene road_dist.npy e normalization_params.pkl)."
        )

    G = G.to(device)
    D = D.to(device)
    G.train()
    D.train()

    # ---------------------------
    # Load ROAD MAP + NORM PARAMS (UNA VOLTA)
    # ---------------------------
    road_dist = None
    road_meta = None
    dyn_min_t = None
    dyn_max_t = None

    if lambda_road > 0.0:
        road_dist, road_meta = load_road_tensors(data_dir, device)

        norm_path = os.path.join(data_dir, "normalization_params.pkl")
        with open(norm_path, "rb") as f:
            params = pickle.load(f)

        if "dynamic_min" not in params or "dynamic_max" not in params:
            raise KeyError("normalization_params.pkl deve contenere 'dynamic_min' e 'dynamic_max'.")

        dyn_min_t = torch.tensor(params["dynamic_min"], dtype=torch.float32, device=device)
        dyn_max_t = torch.tensor(params["dynamic_max"], dtype=torch.float32, device=device)

    # ---------------------------
    # EMA
    # ---------------------------
    G_ema = None
    if use_ema:
        import copy
        G_ema = copy.deepcopy(G).to(device)
        for p in G_ema.parameters():
            p.requires_grad_(False)

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=d_lr, betas=(0.0, 0.9))

    history = {
        "d_loss": [],
        "g_loss": [],
        "r1r2": [],
        "phys": [],
        "road": [],
        "lam_road": [],
        "sigma": [],
    }

    scaler = GradScaler("cuda") if (device.type == "cuda") else GradScaler()
    global_step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for real_X, S, lengths in loop:
            global_step += 1

            real_X = real_X.to(device)       # (B,T,4)
            S = S.to(device)                 # (B,4)
            lengths = lengths.to(device)     # (B,)
            B, T, _ = real_X.shape

            # pad_mask True dove padding
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            pad_mask = t_idx >= lengths.unsqueeze(1)  # (B,T) bool
            valid_mask_float = (~pad_mask).unsqueeze(-1).float()  # (B,T,1)

            # sigma noise (decay per epoca)
            if noise_decay and noise_decay > 0.0:
                sigma = max(noise_min, float(noise_std) * (float(noise_decay) ** epoch))
            else:
                sigma = float(noise_std)

            # R1/R2 schedule
            do_r1r2 = (r1r2_every is not None) and (r1r2_every > 0) and (global_step % r1r2_every == 0)

            r1 = torch.zeros((), device=device)
            r2 = torch.zeros((), device=device)

            # ---------------------------
            # Train Critic (D)
            # ---------------------------
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)
                fake_X = G(z, S, pad_mask=pad_mask).detach()

                # Noise injection only to D inputs, only on valid timesteps
                if sigma > 0.0 and (noise_on_real or noise_on_fake):
                    real_in = real_X + (sigma * torch.randn_like(real_X) * valid_mask_float) if noise_on_real else real_X
                    fake_in = fake_X + (sigma * torch.randn_like(fake_X) * valid_mask_float) if noise_on_fake else fake_X
                else:
                    real_in = real_X
                    fake_in = fake_X

                opt_D.zero_grad(set_to_none=True)

                with autocast(device_type=device.type):
                    real_scores = D(real_in, S, pad_mask=pad_mask)
                    fake_scores = D(fake_in, S, pad_mask=pad_mask)
                    loss_D = rpgan_d_loss(real_scores, fake_scores)

                # R1/R2 in FP32 on "clean" inputs
                if do_r1r2:
                    with autocast(device_type=device.type, enabled=False):
                        r1 = r1_penalty(D, real_X.float(), S.float(), pad_mask=pad_mask)
                        r2 = r2_penalty(D, fake_X.float(), S.float(), pad_mask=pad_mask)

                total_D = loss_D + gamma_r1r2 * (r1 + r2)

                scaler.scale(total_D).backward()
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                scaler.step(opt_D)
                scaler.update()

            # ---------------------------
            # Train Generator (G)
            # ---------------------------
            z = torch.randn(B, latent_dim, device=device)
            opt_G.zero_grad(set_to_none=True)

            phys_loss = torch.zeros((), device=device)
            road_loss = torch.zeros((), device=device)

            # warmup lambda_road
            lam_road = linear_warmup(
                step=global_step,
                start=int(road_warmup_start),
                ramp=int(road_warmup_ramp),
                max_val=float(lambda_road),
            )

            with autocast(device_type=device.type):
                fake_X = G(z, S, pad_mask=pad_mask)          # (B,T,4) norm
                fake_scores = D(fake_X, S, pad_mask=pad_mask)
                real_scores_ref = D(real_X, S, pad_mask=pad_mask).detach()

                g_loss = rpgan_g_loss(real_scores_ref, fake_scores)

                total_G = g_loss

                # phys loss
                if lambda_phys and lambda_phys > 0.0:
                    phys_loss = conservative_physical_loss(fake_X, S, lengths)
                    total_G = total_G + float(lambda_phys) * phys_loss

                # ROAD loss (solo se warmup attivo e road disponibile)
                if (lam_road > 0.0) and (road_dist is not None):
                    fake_xy_norm = fake_X[:, :, :2]  # (B,T,2) norm
                    fake_xy_den = denorm_xy_torch(fake_xy_norm, dyn_min_t, dyn_max_t)  # meters

                    road_loss = road_loss_from_distmap(
                        fake_xy_den=fake_xy_den,
                        pad_mask=pad_mask,
                        road_dist=road_dist,
                        meta=road_meta,
                        power=float(road_power),
                    )

                    # NORMALIZZAZIONE: per non far esplodere ROAD con sequenze lunghe
                    if road_norm_by_valid:
                        valid_count = (~pad_mask).sum().clamp(min=1).float()  # scalare
                        road_loss = road_loss / valid_count

                    total_G = total_G + lam_road * road_loss

            scaler.scale(total_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt_G)
            scaler.update()

            # EMA update
            if use_ema and G_ema is not None:
                update_ema(G_ema, G, ema_decay)

            # Logging
            history["d_loss"].append(float(loss_D.item()))
            history["g_loss"].append(float(g_loss.item()))
            history["r1r2"].append(float((r1 + r2).item()))
            history["phys"].append(float(phys_loss.item()))
            history["road"].append(float(road_loss.item()))
            history["lam_road"].append(float(lam_road))
            history["sigma"].append(float(sigma))

            # Sampling callback
            if sample_callback and global_step % 1000 == 0:
                G_eval = G_ema if (use_ema and G_ema is not None) else G
                sample_callback(G_eval, epoch, global_step)

            loop.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "R1R2": f"{(r1 + r2).item():.3f}",
                "PHYS": f"{phys_loss.item():.3f}",
                "ROAD": f"{road_loss.item():.3f}",
                "lamR": f"{lam_road:.4f}",
                "sig": f"{sigma:.4f}",
            })

        save_checkpoint(G, D, G_ema, opt_G, opt_D, global_step, out_dir)

    return history, (G_ema if (use_ema and G_ema is not None) else G)
