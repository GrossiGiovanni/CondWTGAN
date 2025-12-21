import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, FEATURE_DIM, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G
)

from src.model import TransformerGenerator

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

N_SAMPLES = 50
T_CUT = 20   # <-- quanti timestep escludere all'inizio
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")
SAVE_DIR = os.path.join(OUTPUT_DIR, "final_samples_20")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X_path = os.path.join(DATA_DIR, "X_train.npy")
S_path = os.path.join(DATA_DIR, "S_train.npy")

X = np.load(X_path)
S = np.load(S_path)

print("Loaded X:", X.shape)
print("Loaded S:", S.shape)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

G = TransformerGenerator(
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    nhead=N_HEADS,
    num_layers=N_LAYERS_G,
    ff_dim=FF_DIM,
).to(DEVICE)

G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

print("✅ Generator loaded.")

# ------------------------------------------------------------
# SAMPLE & PLOT
# ------------------------------------------------------------

indices = np.random.choice(len(S), N_SAMPLES, replace=False)

all_trajs = []

with torch.no_grad():
    for i, idx in enumerate(indices):

        S_sample = torch.tensor(S[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z = torch.randn(1, LATENT_DIM, device=DEVICE)

        fake = G(z, S_sample).cpu().numpy()[0]  # (120, 4)

        # ---- TAGLIA I PRIMI T_CUT TIMESTEP ----
        fake_trimmed = fake[T_CUT:]      # (120 - T_CUT, 4)

        xs = fake_trimmed[:, 0]
        ys = fake_trimmed[:, 1]

        # Nuovo start/end basati sulla parte tagliata
        x0, y0 = xs[0], ys[0]
        xT, yT = xs[-1], ys[-1]

        # ---- PLOT SINGOLO ----
        plt.figure(figsize=(5, 5))
        plt.plot(xs, ys, marker="o", markersize=2, label=f"Generated (trimmed, T_CUT={T_CUT})")
        plt.scatter([x0], [y0], s=80, marker="o", label="Start")
        plt.scatter([xT], [yT], s=80, marker="x", label="End")

        plt.title(f"Generated Trajectory #{i} (T_CUT = {T_CUT})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        path = os.path.join(SAVE_DIR, f"traj_{i}.png")
        plt.savefig(path)
        plt.close()

        print(f"Saved: {path}")

        # ---- SALVA TRAIETTORIA TRIMMED NEL PLOT COMPLESSIVO ----
        all_trajs.append(fake_trimmed[:, :2])   # <-- CORRETTO!

# ------------------------------------------------------------
# PLOT TUTTE LE TRAIETTORIE TAGLIATE
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))

for traj in all_trajs:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.7)

plt.title(f"{N_SAMPLES} Generated Trajectories (T_CUT = {T_CUT})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)

path_all = os.path.join(SAVE_DIR, "all_trajectories_trimmed.png")
plt.savefig(path_all)
plt.close()

print("✅ Plot complessivo salvato:", path_all)
