import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, FEATURE_DIM, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G, N_LAYERS_D,
    BATCH_SIZE, EPOCHS, N_CRITIC, LAMBDA_GP,
    G_LR, D_LR
)

def plot_training_curves(history, out_path):
    plt.figure(figsize=(10, 6))
    for key in history:
        plt.plot(history[key], label=key)

    plt.title("Training History")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_sample_trajectory(
    G,
    S_sample,
    out_dir,
    epoch,
    step,
    device,
    latent_dim,
    mins=None,
    maxs=None,
    L_sample=None,
    plot_start_gen=True,      # NEW: mostra start generato (utile)
):
    """
    Genera una traiettoria esempio e salva un plot.

    - Plotta SEMPRE solo la parte reale: t = 0 .. L-1
    - Non plottare Start (cond) (come richiesto)
    """

    G_was_training = G.training
    G.eval()

    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        fake = G(z, S_sample.to(device)).cpu().numpy()[0]  # (SEQ_LEN, 4)

    # Denormalizzazione opzionale
    if mins is not None and maxs is not None:
        fake = fake * (maxs - mins) + mins
        S_plot = S_sample.cpu().numpy() * (maxs - mins) + mins
    else:
        S_plot = S_sample.cpu().numpy()

    # Determina l'indice finale reale
    if L_sample is None:
        end_i = fake.shape[0] - 1
    else:
        end_i = int(L_sample) - 1
        end_i = max(0, min(end_i, fake.shape[0] - 1))

    # Traiettoria reale (0..end_i)
    traj = fake[:end_i + 1]   # (L,4)
    xs = traj[:, 0]
    ys = traj[:, 1]

    # Condizionamento
    x0_cond, y0_cond, xT_cond, yT_cond = S_plot[0]

    # Start/End generati
    x0_gen, y0_gen = xs[0], ys[0]
    xT_gen, yT_gen = xs[-1], ys[-1]

    # FDE (end gen vs end cond)
    fde = float(np.sqrt((xT_gen - xT_cond) ** 2 + (yT_gen - yT_cond) ** 2))

    # Plot
    plt.figure(figsize=(6, 6))

    plt.plot(xs, ys, marker="o", markersize=2, label=f"Generated (L={end_i+1})")

    # (RIMOSSO) Start condizionato: non lo plottiamo
    # plt.scatter([x0_cond], [y0_cond], ...)

    # End condizionato
    plt.scatter([xT_cond], [yT_cond], s=80, marker="x", label="End (cond)")

    # End generato
    plt.scatter([xT_gen], [yT_gen], s=80, marker="+", label="End (gen)")

    # Opzionale: start generato (per verificare se parte coerente)
    if plot_start_gen:
        plt.scatter([x0_gen], [y0_gen], s=60, marker="o", label="Start (gen)")

    plt.title(f"Generated Trajectory - epoch {epoch}, step {step}\nFDE = {fde:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"traj_e{epoch}_s{step}.png")
    plt.savefig(path)
    plt.close()

    if G_was_training:
        G.train()

    return path, fde


def make_sample_callback(out_dir, device, latent_dim, dataset, plot_start_gen=True):
    """
    Callback per trainer.
    Dataset atteso: (traj, S, L) oppure (traj, S)
    """

    def callback(G, epoch, step):
        idx = np.random.randint(len(dataset))

        sample = dataset[idx]
        if len(sample) == 3:
            _, S_sample, L_sample = sample
            L_sample = int(L_sample)
        else:
            _, S_sample = sample
            L_sample = None

        if S_sample.dim() == 1:
            S_sample_batch = S_sample.unsqueeze(0)
        else:
            S_sample_batch = S_sample

        save_sample_trajectory(
            G,
            S_sample_batch,
            out_dir,
            epoch,
            step,
            device,
            latent_dim,
            L_sample=L_sample,
            plot_start_gen=plot_start_gen
        )

    return callback
