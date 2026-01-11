"""
GENERATOR.PY - Aggiornato per il nuovo modello
===============================================
Assicurati che i parametri corrispondano a quelli usati nel training!
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# IMPORTANTE: Usa gli stessi parametri del training!
# Se hai trainato con config_test.py (leggero), usa questi:
# ============================================================

# Parametri TEST (config_test.py) - 3 layer
USE_TEST_CONFIG = False  # <<< Cambia a False se usi il modello full

if USE_TEST_CONFIG:
    # Configurazione TEST (leggera)
    LATENT_DIM = 64
    D_MODEL = 128
    FF_DIM = 256
    N_HEADS = 4
    N_LAYERS_G = 3
    print("⚙️ Usando configurazione TEST (3 layers)")
else:
    # Configurazione FULL (completa)
    LATENT_DIM = 128
    D_MODEL = 256
    FF_DIM = 1024
    N_HEADS = 4
    N_LAYERS_G = 5
    print("⚙️ Usando configurazione FULL (5 layers)")

# Altri parametri
SEQ_LEN = 120
COND_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

if USE_TEST_CONFIG:
    OUTPUT_DIR = os.path.join(ROOT, "outputs_test")  # Cartella test
else:
    OUTPUT_DIR = os.path.join(ROOT, "outputs")

MODEL_PATH = os.path.join(OUTPUT_DIR,  "transformer_G_final_improved.pt")
SAVE_DIR = os.path.join(OUTPUT_DIR, "SAMPLES_generated")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Import del modello migliorato
# ============================================================

from src.model import ImprovedTransformerGenerator

# ============================================================
# CONFIG
# ============================================================

N_SAMPLES = 20

# ============================================================
# LOAD DATA
# ============================================================

X_path = os.path.join(DATA_DIR, "X_train.npy")
S_path = os.path.join(DATA_DIR, "S_train.npy")
L_path = os.path.join(DATA_DIR, "L_train.npy")

X = np.load(X_path)
S = np.load(S_path)
L = np.load(L_path)

print(f"Loaded X: {X.shape}")
print(f"Loaded S: {S.shape}")
print(f"Loaded L: {L.shape} min/med/max: {int(L.min())} {float(np.median(L))} {int(L.max())}")

# Scegli solo sequenze con L >= 2
valid_idx = np.where(L >= 2)[0]
if len(valid_idx) < N_SAMPLES:
    raise ValueError(f"Not enough valid sequences. Available={len(valid_idx)}")

indices = np.random.choice(valid_idx, N_SAMPLES, replace=False)

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\n📂 Loading model from: {MODEL_PATH}")

G = ImprovedTransformerGenerator(
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    d_model=D_MODEL,
    nhead=N_HEADS,
    num_layers=N_LAYERS_G,
    ff_dim=FF_DIM,
    hard_start=True,
    soft_end_blend=True,
).to(DEVICE)

# Carica checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Il checkpoint può contenere diversi formati
if isinstance(checkpoint, dict):
    if "G_ema" in checkpoint and checkpoint["G_ema"] is not None:
        print("   Loading G_ema weights")
        G.load_state_dict(checkpoint["G_ema"])
    elif "G" in checkpoint:
        print("   Loading G weights")
        G.load_state_dict(checkpoint["G"])
    else:
        # Prova a caricare direttamente
        G.load_state_dict(checkpoint)
else:
    G.load_state_dict(checkpoint)

G.eval()
print("✅ Generator loaded successfully!")

# ============================================================
# METRIC HELPERS
# ============================================================

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
        S_i = S[idx]
        x0_cond, y0_cond, xT_cond, yT_cond = S_i
        start_cond = np.array([x0_cond, y0_cond], dtype=np.float32)
        end_cond = np.array([xT_cond, yT_cond], dtype=np.float32)

        # Real trajectory
        real_full = X[idx]
        real_seg = real_full[:Li, :2]

        # Generate fake
        S_sample = torch.tensor(S_i, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        L_sample = torch.tensor([Li], dtype=torch.long).to(DEVICE)
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        
        # IMPORTANTE: passa lengths al generator!
        fake_full = G(z, S_sample, lengths=L_sample).cpu().numpy()[0]
        fake_seg = fake_full[:Li, :2]

        # Metrics
        mae_xy = pos_mae(real_seg, fake_seg)
        e_start = start_error(fake_full, start_cond)
        e_end = end_error(fake_full, end_cond, Li)
        
        all_start_errors.append(e_start)
        all_end_errors.append(e_end)

        # Plot overlay
        plt.figure(figsize=(8, 8))

        plt.plot(real_seg[:, 0], real_seg[:, 1], 'b-', linewidth=2, label="Real", alpha=0.7)
        plt.plot(fake_seg[:, 0], fake_seg[:, 1], 'r-', linewidth=2, label="Fake", alpha=0.7)

        # Start points
        plt.scatter([x0_cond], [y0_cond], s=100, c='green', marker='o', label="Start (cond)", zorder=5)
        plt.scatter([fake_full[0, 0]], [fake_full[0, 1]], s=80, c='lime', marker='s', label="Start (gen)", zorder=5)
        
        # End points
        plt.scatter([xT_cond], [yT_cond], s=100, c='blue', marker='x', label="End (cond)", zorder=5)
        gen_end = fake_full[Li - 1, :2]
        plt.scatter([gen_end[0]], [gen_end[1]], s=80, c='red', marker='+', label="End (gen)", zorder=5)

        plt.title(f"Overlay Real vs Fake #{i} | L={Li}\n"
                  f"MAE_xy={mae_xy:.4f} | StartErr={e_start:.4f} | EndErr={e_end:.4f}")
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

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 50)
print("📊 SUMMARY STATISTICS")
print("=" * 50)
print(f"Start Error - Mean: {np.mean(all_start_errors):.6f}, Max: {np.max(all_start_errors):.6f}")
print(f"End Error   - Mean: {np.mean(all_end_errors):.6f}, Max: {np.max(all_end_errors):.6f}")

# ============================================================
# SUMMARY PLOTS
# ============================================================

# All real trajectories
plt.figure(figsize=(10, 10))
for r in all_real:
    plt.plot(r[:, 0], r[:, 1], 'b-', alpha=0.4, linewidth=1)
plt.title(f"Real Trajectories (N={N_SAMPLES})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
path_all_real = os.path.join(SAVE_DIR, "all_real.png")
plt.savefig(path_all_real, dpi=150, bbox_inches='tight')
plt.close()

# All fake trajectories
plt.figure(figsize=(10, 10))
for f in all_fake:
    plt.plot(f[:, 0], f[:, 1], 'r-', alpha=0.4, linewidth=1)
plt.title(f"Generated Trajectories (N={N_SAMPLES})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
path_all_fake = os.path.join(SAVE_DIR, "all_fake.png")
plt.savefig(path_all_fake, dpi=150, bbox_inches='tight')
plt.close()

# Overlay all
plt.figure(figsize=(10, 10))
for r in all_real:
    plt.plot(r[:, 0], r[:, 1], 'b-', alpha=0.3, linewidth=1)
for f in all_fake:
    plt.plot(f[:, 0], f[:, 1], 'r-', alpha=0.3, linewidth=1)
plt.title(f"Real (blue) vs Generated (red) - N={N_SAMPLES}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)
path_overlay = os.path.join(SAVE_DIR, "all_overlay.png")
plt.savefig(path_overlay, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved plots to: {SAVE_DIR}")
print(f"   - {path_all_real}")
print(f"   - {path_all_fake}")
print(f"   - {path_overlay}")