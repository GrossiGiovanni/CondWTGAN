import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G
)
from src.model import TransformerGenerator

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

N_SAMPLES = 20
MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")
SAVE_DIR = os.path.join(OUTPUT_DIR, "SAMPLES")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X_path = os.path.join(DATA_DIR, "X_train.npy")
S_path = os.path.join(DATA_DIR, "S_train.npy")
L_path = os.path.join(DATA_DIR, "L_train.npy")

X = np.load(X_path)  # (N,120,4)
S = np.load(S_path)  # (N,4)
L = np.load(L_path)  # (N,)

print("Loaded X:", X.shape)
print("Loaded S:", S.shape)
print("Loaded L:", L.shape, "min/med/max:", int(L.min()), float(np.median(L)), int(L.max()))

# scegli solo sequenze con L >= 2 (per sicurezza)
valid_idx = np.where(L >= 2)[0]
if len(valid_idx) < N_SAMPLES:
    raise ValueError(f"Not enough valid sequences. Available={len(valid_idx)}")

indices = np.random.choice(valid_idx, N_SAMPLES, replace=False)

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

print("Generator loaded.")

# ------------------------------------------------------------
# METRIC HELPERS
# ------------------------------------------------------------

def pos_mae(real_seg, fake_seg):
    # real_seg, fake_seg: (K,2)
    return float(np.mean(np.abs(real_seg - fake_seg)))

def end_error(fake_full, end_xy, Li):
    gen_end = fake_full[Li - 1, :2]
    return float(np.linalg.norm(gen_end - end_xy))

# ------------------------------------------------------------
# SAMPLE, OVERLAY REAL VS FAKE, SAVE
# ------------------------------------------------------------

all_real = []
all_fake = []

with torch.no_grad():
    for i, idx in enumerate(indices):
        Li = int(L[idx])
        S_i = S[idx]  # (4,)
        x0_cond, y0_cond, xT_cond, yT_cond = S_i
        end_cond = np.array([xT_cond, yT_cond], dtype=np.float32)

        # Real: solo parte valida 0..L-1
        real_full = X[idx]                # (120,4)
        real_seg = real_full[:Li, :2]     # (L,2)

        # Fake full
        S_sample = torch.tensor(S_i, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        fake_full = G(z, S_sample).cpu().numpy()[0]   # (120,4)
        fake_seg = fake_full[:Li, :2]                 # (L,2)

        # Metrics on valid segment
        mae_xy = pos_mae(real_seg, fake_seg)
        e_end = end_error(fake_full, end_cond, Li)

        # Plot overlay
        plt.figure(figsize=(6, 6))

        plt.plot(real_seg[:, 0], real_seg[:, 1], linewidth=2, label="Real")
        plt.plot(fake_seg[:, 0], fake_seg[:, 1], linewidth=2, label="Fake")

        
        plt.scatter([xT_cond], [yT_cond], s=80, marker="x", label="End (cond)")

        gen_end = fake_full[Li - 1, :2]
        plt.scatter([gen_end[0]], [gen_end[1]], s=80, marker="+", label="End (gen)")

        plt.title(f"Overlay Real vs Fake #{i} | L={Li}\nMAE_xy={mae_xy:.4f} | EndErr={e_end:.4f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(SAVE_DIR, f"overlay_{i}_idx{idx}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print("Saved:", out_path)

        all_real.append(real_seg)
        all_fake.append(fake_seg)

# ------------------------------------------------------------
# SUMMARY PLOTS
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))
for r in all_real:
    plt.plot(r[:, 0], r[:, 1], alpha=0.35)
plt.title(f"Real trajectories (N={N_SAMPLES})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
path_all_real = os.path.join(SAVE_DIR, "all_real.png")
plt.savefig(path_all_real, dpi=200)
plt.close()

plt.figure(figsize=(7, 7))
for f in all_fake:
    plt.plot(f[:, 0], f[:, 1], alpha=0.35)
plt.title(f"Fake trajectories (N={N_SAMPLES})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
path_all_fake = os.path.join(SAVE_DIR, "all_fake.png")
plt.savefig(path_all_fake, dpi=200)
plt.close()

print("Saved:", path_all_real)
print("Saved:", path_all_fake)
