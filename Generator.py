"""
Generator.py - COERENTE con train_with_road_constraint.py
=========================================================
Carica il modello salvato da:
outputs/transformer_G_final_with_road.pt

Usa gli stessi parametri di src.config e chiama:
G(z, S, pad_mask=pad_mask)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G,
)

from src.model import TransformerGenerator


# ============================================================
# CONFIG
# ============================================================
N_SAMPLES = 20
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")
SAVE_DIR = os.path.join(OUTPUT_DIR, "SAMPLES_generated")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
S = np.load(os.path.join(DATA_DIR, "S_train_fixed.npy"))
L = np.load(os.path.join(DATA_DIR, "L_train.npy"))

print(f"Loaded X: {X.shape}")
print(f"Loaded S: {S.shape}")
print(f"Loaded L: {L.shape} min/med/max: {int(L.min())} {float(np.median(L))} {int(L.max())}")

valid_idx = np.where(L >= 2)[0]
if len(valid_idx) < N_SAMPLES:
    raise ValueError(f"Not enough valid sequences. Available={len(valid_idx)}")
indices = np.random.choice(valid_idx, N_SAMPLES, replace=False)


# ============================================================
# BUILD MODEL (IDENTICO al training)
# ============================================================
print(f"\n📂 Loading model from: {MODEL_PATH}")

G = TransformerGenerator(
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    nhead=N_HEADS,
    num_layers=N_LAYERS_G,
    ff_dim=FF_DIM,
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)

# questo file è un puro state_dict (dal tuo train.py)
G.load_state_dict(state, strict=True)
G.eval()

print("✅ Generator loaded successfully!")


# ============================================================
# HELPERS
# ============================================================
def build_pad_mask(lengths, T, device):
    """pad_mask: (B,T) True dove padding, come nel training."""
    B = lengths.shape[0]
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    return t_idx >= lengths.unsqueeze(1)

def pos_mae(real_seg, fake_seg):
    return float(np.mean(np.abs(real_seg - fake_seg)))

def end_error(fake_full, end_xy, Li):
    gen_end = fake_full[Li - 1, :2]
    return float(np.linalg.norm(gen_end - end_xy))

def start_error(fake_full, start_xy):
    gen_start = fake_full[0, :2]
    return float(np.linalg.norm(gen_start - start_xy))


# ============================================================
# GENERATE AND VISUALIZE
# ============================================================
all_real = []
all_fake = []
all_start_errors = []
all_end_errors = []

print(f"\n🎯 Generating {N_SAMPLES} samples...")

with torch.no_grad():
    for i, idx in enumerate(indices):
        Li = int(L[idx])
        S_i = S[idx].astype(np.float32)

        x0_cond, y0_cond, xT_cond, yT_cond = S_i
        start_cond = np.array([x0_cond, y0_cond], dtype=np.float32)
        end_cond = np.array([xT_cond, yT_cond], dtype=np.float32)

        real_full = X[idx].astype(np.float32)
        if i < 5:
            real_start = real_full[0, :2]
            real_end   = real_full[Li-1, :2]

            s_start = S_i[:2]
            s_end   = S_i[2:4]

            print("\n==== CHECK SAMPLE", i, "idx", idx, "L", Li, "====")
            print("real_start:", real_start, " | S_start:", s_start, " | diff:", np.abs(real_start - s_start))
            print("real_end  :", real_end,   " | S_end  :", s_end,   " | diff:", np.abs(real_end - s_end))

            print("real[0] xy:", real_full[0, :2])
            print("real[1] xy:", real_full[1, :2])
            print("real[2] xy:", real_full[2, :2])
            print("S_start  :", S_i[:2])
        real_seg = real_full[:Li, :2]

        S_sample = torch.tensor(S_i, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        L_sample = torch.tensor([Li], dtype=torch.long).to(DEVICE)
        pad_mask = build_pad_mask(L_sample, SEQ_LEN, DEVICE)

        z = torch.randn(1, LATENT_DIM, device=DEVICE)

        fake_full = G(z, S_sample, pad_mask=pad_mask).cpu().numpy()[0].astype(np.float32)
        fake_seg = fake_full[:Li, :2]

        mae_xy = pos_mae(real_seg, fake_seg)
        e_start = start_error(fake_full, start_cond)
        e_end = end_error(fake_full, end_cond, Li)

        all_start_errors.append(e_start)
        all_end_errors.append(e_end)

        # Plot overlay
        plt.figure(figsize=(8, 8))
        plt.plot(real_seg[:, 0], real_seg[:, 1], 'b-', linewidth=2, label="Real", alpha=0.7)
        plt.plot(fake_seg[:, 0], fake_seg[:, 1], 'r-', linewidth=2, label="Fake", alpha=0.7)

        plt.scatter([x0_cond], [y0_cond], s=100, c='green', marker='o', label="Start (cond)", zorder=5)
        plt.scatter([fake_full[0, 0]], [fake_full[0, 1]], s=80, c='lime', marker='s', label="Start (gen)", zorder=5)

        plt.scatter([xT_cond], [yT_cond], s=100, c='blue', marker='x', label="End (cond)", zorder=5)
        gen_end = fake_full[Li - 1, :2]
        plt.scatter([gen_end[0]], [gen_end[1]], s=80, c='red', marker='+', label="End (gen)", zorder=5)

        plt.title(
            f"Overlay Real vs Fake #{i} | L={Li}\n"
            f"MAE_xy={mae_xy:.4f} | StartErr={e_start:.4f} | EndErr={e_end:.4f}"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        out_path = os.path.join(SAVE_DIR, f"overlay_{i}_idx{idx}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   [{i+1}/{N_SAMPLES}] L={Li}, StartErr={e_start:.6f}, EndErr={e_end:.6f}")

        all_real.append(real_seg)
        all_fake.append(fake_seg)

print("\n" + "=" * 50)
print("📊 SUMMARY STATISTICS")
print("=" * 50)
print(f"Start Error - Mean: {np.mean(all_start_errors):.6f}, Max: {np.max(all_start_errors):.6f}")
print(f"End Error   - Mean: {np.mean(all_end_errors):.6f}, Max: {np.max(all_end_errors):.6f}")

print(f"\n✅ Saved plots to: {SAVE_DIR}")
