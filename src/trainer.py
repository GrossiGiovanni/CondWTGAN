import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.losses import (
    rpgan_d_loss,
    rpgan_g_loss,
    r1_penalty,
    r2_penalty,
    conservative_physical_loss,
)

# -------------------------------------------------------------
#  EMA UPDATE
# -------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Aggiorna il modello EMA:
    ema = decay * ema + (1 - decay) * model
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_((1 - decay) * param.data)


# -------------------------------------------------------------
#  SALVATAGGIO CHECKPOINT
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
#  TRAINING LOOP
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
    lambda_phys,
    use_ema,
    ema_decay,
    g_lr,
    d_lr,
    epochs,
    out_dir,
    sample_callback,
    # ---- Noise injection (NEW) ----
    noise_std,       # tipico: 0.005 - 0.02 in scala [0,1]
    noise_decay,      # 0.0 = nessun decay; es: 0.95 per epoca
    noise_min,        # floor del rumore se usi decay
    noise_on_real=True,
    noise_on_fake=True,
    r1r2_every=3,
):
    """
    Training loop per GAN condizionata su traiettoria iniziale/finale.

    - dataloader produce: real_X (B,120,4), S (B,4), lengths (B,)
    - pad_mask: True dove t >= L (padding)
    - Rumore gaussiano: applicato SOLO agli input del Discriminator e SOLO su timestep validi.
    """

    G = G.to(device)
    D = D.to(device)

    G.train()
    D.train()

    # Duplicate EMA generator
    G_ema = None
    if use_ema:
        import copy
        G_ema = copy.deepcopy(G).to(device)
        for p in G_ema.parameters():
            p.requires_grad_(False)

    # Ottimizzatori
    opt_G = optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=d_lr, betas=(0.0, 0.9))

    history = {
        "d_loss": [],
        "g_loss": [],
        "r1r2": [],
        "phys": [],
        "sigma": [],   # log utile
    }

    scaler = GradScaler()
    global_step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for real_X, S, lengths in loop:
            global_step += 1

            real_X = real_X.to(device)       # (B,T,4)
            S = S.to(device)                 # (B,4)
            lengths = lengths.to(device)     # (B,)
            B, T, _ = real_X.shape

            # pad_mask: True dove t >= L (padding)
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            pad_mask = t_idx >= lengths.unsqueeze(1)   # (B,T) bool

            # mask float per applicare rumore solo ai timestep validi
            valid_mask_float = (~pad_mask).unsqueeze(-1).float()  # (B,T,1)

            # calcolo sigma (opzionale decay per epoca)
            if noise_decay and noise_decay > 0.0:
                sigma = max(noise_min, float(noise_std) * (float(noise_decay) ** epoch))
            else:
                sigma = float(noise_std)

             # Decidi se calcolare R1/R2 in questo step
            do_r1r2 = (r1r2_every is not None) and (r1r2_every > 0) and (global_step % r1r2_every == 0)

            # default r1/r2 (così esistono sempre per logging/postfix)
            r1 = torch.zeros((), device=device)
            r2 = torch.zeros((), device=device)
            # ---------------------------
            #  Train Critic (D)
            # ---------------------------
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)
                fake_X = G(z, S, pad_mask=pad_mask).detach()

                # Noise injection SOLO per input del D
                if sigma > 0.0 and (noise_on_real or noise_on_fake):
                    if noise_on_real:
                        real_in = real_X + sigma * torch.randn_like(real_X) * valid_mask_float
                    else:
                        real_in = real_X

                    if noise_on_fake:
                        fake_in = fake_X + sigma * torch.randn_like(fake_X) * valid_mask_float
                    else:
                        fake_in = fake_X
                else:
                    real_in = real_X
                    fake_in = fake_X

                opt_D.zero_grad(set_to_none=True)

                with autocast():
                    real_scores = D(real_in, S, pad_mask=pad_mask)
                    fake_scores = D(fake_in, S, pad_mask=pad_mask)
                    loss_D = rpgan_d_loss(real_scores, fake_scores)


                if do_r1r2:
                    with autocast(False):
                        r1 = r1_penalty(D, real_X.float(), S.float(), pad_mask=pad_mask)
                        r2 = r2_penalty(D, fake_X.float(), S.float(), pad_mask=pad_mask)
                else:
                    # lascia r1,r2 = 0
                    pass
                # R1/R2 meglio in FP32 (second-order autograd più stabile)
                # (calcolate su input "puliti", non rumorosi)
               
                total_D = loss_D + gamma_r1r2 * (r1 + r2)

                scaler.scale(total_D).backward()

                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)

                scaler.step(opt_D)
                scaler.update()

            # ---------------------------
            #  Train Generator (G)
            # ---------------------------
            z = torch.randn(B, latent_dim, device=device)
            opt_G.zero_grad(set_to_none=True)

            with autocast():
                fake_X = G(z, S, pad_mask=pad_mask)
                fake_scores = D(fake_X, S, pad_mask=pad_mask)

                # real come riferimento, senza grad
                real_scores_ref = D(real_X, S, pad_mask=pad_mask).detach()

                g_loss = rpgan_g_loss(real_scores_ref, fake_scores)

                # Phys loss su output pulito (NO rumore)
                phys_loss = conservative_physical_loss(fake_X, S, lengths)
                total_G = g_loss + lambda_phys * phys_loss

            scaler.scale(total_G).backward()

            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)

            scaler.step(opt_G)
            scaler.update()

            # EMA Update
            if use_ema and G_ema is not None:
                update_ema(G_ema, G, ema_decay)

            # Logging
            history["d_loss"].append(float(loss_D.item()))
            history["g_loss"].append(float(g_loss.item()))
            history["r1r2"].append(float((r1 + r2).item()))
            history["phys"].append(float(phys_loss.item()))
            history["sigma"].append(float(sigma))

            # Callback di sampling ogni N step
            if sample_callback and global_step % 1000 == 0:
                G_eval = G_ema if (use_ema and G_ema is not None) else G
                sample_callback(G_eval, epoch, global_step)

            loop.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "R1R2": f"{(r1 + r2).item():.3f}",
                "PHYS": f"{phys_loss.item():.3f}",
                "sig": f"{sigma:.4f}",
            })

        save_checkpoint(G, D, G_ema, opt_G, opt_D, global_step, out_dir)

    return history, (G_ema if (use_ema and G_ema is not None) else G)