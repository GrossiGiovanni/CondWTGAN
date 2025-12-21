import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.losses import (
    wasserstein_d_loss,
    wasserstein_g_loss,
    robust_gradient_penalty,
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
#  TRAINING LOOP (adattato al nuovo dataset)
# -------------------------------------------------------------

def train_ultra_stable_traffic_gan(
    G,
    D,
    dataloader,
    *,
    device,
    latent_dim=64,
    n_critic=3,
    lambda_gp=15.0,
    lambda_phys=0.1,
    use_ema=True,
    ema_decay=0.999,
    g_lr=2e-4,
    d_lr=7e-5,
    epochs=50,
    out_dir="outputs",
    sample_callback=None,
):
    """
    Training loop per GAN condizionata su traiettoria iniziale/finale.

    STRUTTURA DATI (NUOVA):
    ------------------------
    - dal dataloader arrivano:
        real_X: (B, 120, 4)   -> [x, y, speed, angle] normalizzati [0,1]
        S:      (B, 4)        -> [x_init, y_init, x_final, y_final] normalizzati [0,1]

    - G(z, S) produce fake_X con shape (B, 120, 4)
    - D(traj, S) produce score scalare (B, 1)

    OBIETTIVO:
    ----------
    - D: imparare a dare score alto a traiettorie reali e basso alle fake
      → WGAN + gradient penalty

    - G: generare traiettorie realistiche e fisicamente coerenti con S
      → WGAN loss + conservative_physical_loss(fake_X, S)
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
        "gp": [],
        "phys": [],
    }

    scaler = GradScaler()

    global_step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for real_X, S in loop:
            global_step += 1

            real_X = real_X.to(device)   # (B, 120, 4)
            S = S.to(device)             # (B, 4)
            B = real_X.size(0)

            # ---------------------------
            #  Train Critic (D)
            # ---------------------------
            for _ in range(n_critic):
                # Generiamo rumore z e poi fake
                z = torch.randn(B, latent_dim, device=device)
                fake_X = G(z, S).detach()    # (B, 120, 4) - staccato dal grafo di G

                opt_D.zero_grad(set_to_none=True)

                with autocast():
                    real_scores = D(real_X, S)
                    fake_scores = D(fake_X, S)
                    loss_D = wasserstein_d_loss(real_scores, fake_scores)
                    gp = robust_gradient_penalty(D, real_X, fake_X, S, device, lambda_gp)
                    total_D = loss_D + gp

                # backward in mixed precision
                scaler.scale(total_D).backward()

                # unscale + clip grad
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)

                # step + update scaler
                scaler.step(opt_D)
                scaler.update()

            # ---------------------------
            #  Train Generator (G)
            # ---------------------------
            z = torch.randn(B, latent_dim, device=device)

            opt_G.zero_grad(set_to_none=True)

            with autocast():
                fake_X = G(z, S)                 # (B, 120, 4)
                fake_scores = D(fake_X, S)
                g_loss = wasserstein_g_loss(fake_scores)
                phys_loss = conservative_physical_loss(fake_X, S)
                total_G = g_loss + lambda_phys * phys_loss

            # backward in mixed precision
            scaler.scale(total_G).backward()

            # unscale + clip grad
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)

            # step + update scaler
            scaler.step(opt_G)
            scaler.update()

            # EMA Update
            if use_ema and G_ema is not None:
                update_ema(G_ema, G, ema_decay)

            # Logging (usa gli ultimi valori di loss_D/gp/g_loss/phys_loss)
            history["d_loss"].append(float(loss_D.item()))
            history["g_loss"].append(float(g_loss.item()))
            history["gp"].append(float(gp.item()))
            history["phys"].append(float(phys_loss.item()))

            # Callback di sampling ogni N step
            if sample_callback and global_step % 200 == 0:
                G_eval = G_ema if (use_ema and G_ema is not None) else G
                sample_callback(G_eval, epoch, global_step)

            loop.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "GP": f"{gp.item():.3f}",
                "PHYS": f"{phys_loss.item():.3f}",
            })

        # Checkpoint a fine epoch
        save_checkpoint(G, D, G_ema, opt_G, opt_D, global_step, out_dir)

    return history, (G_ema if (use_ema and G_ema is not None) else G)
