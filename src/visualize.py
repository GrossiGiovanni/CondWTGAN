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


# -------------------------------------------------------------
#  SALVATAGGIO PLOT DELLE LOSS
# -------------------------------------------------------------

def plot_training_curves(history, out_path):
    """
    Crea e salva i grafici delle curve di addestramento.
    history: dict con liste di valori (d_loss, g_loss, gp, phys, ...)
    """

    plt.figure(figsize=(10, 6))

    for key in history:
        plt.plot(history[key], label=key)

    plt.title("Training History (WGAN-GP Trajectories)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------
#  GENERAZIONE E SALVATAGGIO SAMPLE DURANTE IL TRAINING
# -------------------------------------------------------------

def save_sample_trajectory(
    G,
    S_sample,
    out_dir,
    epoch,
    step,
    device,
    latent_dim,
    mins=None,
    maxs=None
):
    """
    Genera una traiettoria esempio e la salva come grafico.

    Parametri:
    ----------
    G         : generatore
    S_sample  : tensor (1, 4) -> [x_init, y_init, x_final, y_final] (normalizzato)
    mins/maxs : opzionali, per denormalizzare le coordinate

    Ritorna:
    --------
    path, fde
    path : percorso del file png salvato
    fde  : final displacement error (scalare float)
    """

    G_was_training = G.training
    G.eval()

    with torch.no_grad():
        # z ~ N(0, I)
        z = torch.randn(1, latent_dim, device=device)
        # fake: shape tipica (T, feature_dim) oppure (T, 4) nel tuo caso
        fake = G(z, S_sample.to(device)).cpu().numpy()[0]  # (SEQ_LEN, feat)

    # --- Denormalizzazione opzionale ---
    if mins is not None and maxs is not None:
        fake = fake * (maxs - mins) + mins
        S_plot = S_sample.cpu().numpy() * (maxs - mins) + mins
    else:
        S_plot = S_sample.cpu().numpy()

    # Coordinate (x, y) dalla traiettoria generata
    xs = fake[:, 0]
    ys = fake[:, 1]

    # Condizionamento (start/end) - NOTA: uso S_plot per coerenza con eventuale denorm
    x0, y0, xT_cond, yT_cond = S_plot[0]

    # Punto finale generato
    xT_gen = xs[-1]
    yT_gen = ys[-1]

    # Final Displacement Error (tra punto finale generato e target condizionato)
    fde = float(np.sqrt((xT_gen - xT_cond) ** 2 + (yT_gen - yT_cond) ** 2))

    # --- Plot ---
    plt.figure(figsize=(6, 6))

    # traiettoria generata
    plt.plot(xs, ys, marker="o", markersize=2, label="Generated Trajectory")

    # Start condizionato
    plt.scatter([x0], [y0], s=80, marker="o", label="Start (cond)")

    # End condizionato (target)
    plt.scatter([xT_cond], [yT_cond], s=80, marker="x", label="End (cond)")

    # End generato
    plt.scatter([xT_gen], [yT_gen], s=80, marker="+", label="End (gen)")

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

    # Ripristino stato di training del generatore
    if G_was_training:
        G.train()

    return path, fde


# -------------------------------------------------------------
#  WRAPPER PER IL TRAINER (CALLBACK)
# -------------------------------------------------------------

def make_sample_callback(out_dir, device, latent_dim, dataset):
    """
    Ritorna una funzione da passare al trainer (sample_callback)
    che genera e salva una traiettoria ogni N step.

    Usa S REALI dal dataset (soluzione scientificamente corretta).
    """

    def callback(G, epoch, step):
        
       

        # Campiona un indice casuale nel dataset
        idx = np.random.randint(len(dataset))
        traj_sample, S_sample = dataset[idx]   # traiettoria reale, condizione

        # Assicuriamoci che S_sample abbia shape (1, 4)
        if S_sample.dim() == 1:
            S_sample_batch = S_sample.unsqueeze(0)
        else:
            S_sample_batch = S_sample

        path, fde = save_sample_trajectory(
            G,
            S_sample_batch,
            out_dir,
            epoch,
            step,
            device,
            latent_dim
        )

    return callback
