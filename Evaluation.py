import os, json, pickle
import numpy as np
import torch

from src.model import TransformerGenerator
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    LATENT_DIM, SEQ_LEN, D_MODEL, FF_DIM, N_HEADS, N_LAYERS_G
)

# ===== Units (modifica se serve) =====
UNITS = {"pos": "m", "speed": "m/s", "angle": "deg"}  # se angle è rad -> "rad"

# ============================================================
#  Load min/max from normalization_params.pkl (YOUR FORMAT)
# ============================================================
def load_minmax_params():
    pkl_path = os.path.join(DATA_DIR, "normalization_params.pkl")
    with open(pkl_path, "rb") as f:
        p = pickle.load(f)

    feats = p["features_dynamic"]
    dmin = np.array(p["dynamic_min"], dtype=np.float32).reshape(-1)
    dmax = np.array(p["dynamic_max"], dtype=np.float32).reshape(-1)

    if len(feats) != len(dmin) or len(feats) != len(dmax):
        raise ValueError("features_dynamic e dynamic_min/max non hanno la stessa lunghezza.")

    # match robusto per nomi feature
    def find_idx(candidates):
        fl = [str(x).lower() for x in feats]
        for cand in candidates:
            cand = cand.lower()
            for i, name in enumerate(fl):
                # match "contains" + qualche protezione base
                if cand in name:
                    return i, feats[i]
        return None, None

    # prova pattern comuni (adatta se hai nomi strani)
    ix, nx = find_idx(["x", "pos_x", "px"])
    iy, ny = find_idx(["y", "pos_y", "py"])
    iv, nv = find_idx(["speed", "vel", "velocity", "v"])
    ia, na = find_idx(["angle", "heading", "yaw", "theta", "dir", "direction"])

    missing = []
    if ix is None: missing.append("x")
    if iy is None: missing.append("y")
    if iv is None: missing.append("speed")
    if ia is None: missing.append("angle")

    if missing:
        raise ValueError(
            "Non riesco a matchare queste feature in features_dynamic: "
            + ", ".join(missing)
            + "\nfeatures_dynamic = "
            + str(feats)
        )

    # X_min/max (4,) nell'ordine [x,y,speed,angle]
    X_min = np.array([dmin[ix], dmin[iy], dmin[iv], dmin[ia]], dtype=np.float32)
    X_max = np.array([dmax[ix], dmax[iy], dmax[iv], dmax[ia]], dtype=np.float32)
    X_rng = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min)).astype(np.float32)

    # S non ha min/max nel pickle -> fallback su x,y
    S_min = np.array([X_min[0], X_min[1], X_min[0], X_min[1]], dtype=np.float32)
    S_max = np.array([X_max[0], X_max[1], X_max[0], X_max[1]], dtype=np.float32)
    S_rng = np.where((S_max - S_min) == 0, 1.0, (S_max - S_min)).astype(np.float32)

    print("[norm] Matched dynamic features:")
    print(f"  x     -> {nx}")
    print(f"  y     -> {ny}")
    print(f"  speed -> {nv}")
    print(f"  angle -> {na}")

    return X_min, X_rng, S_min, S_rng


def inv_minmax(x, mn, rng):
    return x * rng + mn

# ============================================================
#  Mask-aware metrics (compatte)
# ============================================================
def masked_mae(a, b, m):
    m3 = m[..., None]
    denom = max(m.sum() * a.shape[-1], 1.0)
    return float((np.abs(a - b) * m3).sum() / denom)

def masked_mean(x, m):
    m3 = m[..., None]
    denom = max(m.sum(), 1.0)
    return (x * m3).sum(axis=(0, 1)) / denom

def masked_std_feat(x, m):
    xf = x.reshape(-1, x.shape[-1])
    mf = m.reshape(-1)
    v = xf[mf]
    if v.shape[0] < 2:
        return np.zeros((x.shape[-1],), dtype=np.float32)
    return v.std(axis=0).astype(np.float32)

# ============================================================
#  Main
# ============================================================
def run_eval(T_CUT=0, N_SAMPLES=300):
    print("\nGAN EVALUATION (compact, denormalized)")
    print("--------------------------------------")

    Xn = np.load(os.path.join(DATA_DIR, "X_train.npy"))  # (N,120,4) in [0,1]
    Sn = np.load(os.path.join(DATA_DIR, "S_train.npy"))  # (N,4) in [0,1]
    L  = np.load(os.path.join(DATA_DIR, "L_train.npy"))  # (N,)

    X_min, X_rng, S_min, S_rng = load_minmax_params()

    valid = np.where(L > (T_CUT + 1))[0]
    if len(valid) < N_SAMPLES:
        raise ValueError(f"Poche sequenze valide: {len(valid)} < {N_SAMPLES}")
    idx = np.random.choice(valid, N_SAMPLES, replace=False)

    real_n = Xn[idx]
    S_sub_n = Sn[idx]
    L_sub = L[idx].astype(int)

    # Load model
    G = TransformerGenerator(
        latent_dim=LATENT_DIM, cond_dim=4, seq_len=SEQ_LEN,
        d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS_G, ff_dim=FF_DIM
    ).to(DEVICE)
    G.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "transformer_G_final.pt"), map_location=DEVICE))
    G.eval()

    # Generate fake (NORMALIZED)
    with torch.no_grad():
        z = torch.randn(N_SAMPLES, LATENT_DIM, device=DEVICE)
        S_t = torch.tensor(S_sub_n, dtype=torch.float32, device=DEVICE)
        fake_n = G(z, S_t).cpu().numpy()

    # DENORMALIZE
    real = inv_minmax(real_n, X_min, X_rng).astype(np.float32)
    fake = inv_minmax(fake_n, X_min, X_rng).astype(np.float32)
    S_den = inv_minmax(S_sub_n, S_min, S_rng).astype(np.float32)

    # mask valid timesteps post T_CUT
    T = real.shape[1]
    t = np.arange(T)[None, :]
    m = (t < L_sub[:, None]) & (t >= T_CUT)

    # metrics
    metrics = {}
    metrics["pos_error"]   = masked_mae(real[:, :, :2],  fake[:, :, :2],  m)
    metrics["speed_error"] = masked_mae(real[:, :, 2:3], fake[:, :, 2:3], m)
    metrics["angle_error"] = masked_mae(real[:, :, 3:4], fake[:, :, 3:4], m)

    # diversity (separata, sensata)
    stdf = masked_std_feat(fake, m)  # (4,)
    metrics["div_pos"]   = float(stdf[:2].mean())
    metrics["div_speed"] = float(stdf[2])
    metrics["div_angle"] = float(stdf[3])

    # realism score (dimensionless, normalizzato per range)
    rmu = masked_mean(real, m)
    fmu = masked_mean(fake, m)
    rel = np.abs(rmu - fmu) / np.maximum(X_rng, 1e-12)
    metrics["realism_score"] = float(1.0 / (1.0 + rel.mean()))

    # boundary errors su start/end veri
    start_fake = fake[:, 0, :2]
    start_real = S_den[:, :2]
    end_fake = np.vstack([fake[i, L_sub[i]-1, :2] for i in range(N_SAMPLES)])
    end_real = S_den[:, 2:4]
    metrics["start_boundary_error"] = float(np.mean(np.abs(start_fake - start_real)))
    metrics["end_boundary_error"]   = float(np.mean(np.abs(end_fake - end_real)))

    # extra: km/h se speed è m/s
    if UNITS["speed"] == "m/s":
        metrics["speed_error_kmh"] = metrics["speed_error"] * 3.6
        metrics["div_speed_kmh"]   = metrics["div_speed"] * 3.6

    # units in json
    metrics["units"] = {
        "pos_error": UNITS["pos"],
        "speed_error": UNITS["speed"],
        "angle_error": UNITS["angle"],
        "div_pos": UNITS["pos"],
        "div_speed": UNITS["speed"],
        "div_angle": UNITS["angle"],
        "start_boundary_error": UNITS["pos"],
        "end_boundary_error": UNITS["pos"],
        "realism_score": "dimensionless",
        **({"speed_error_kmh": "km/h", "div_speed_kmh": "km/h"} if UNITS["speed"] == "m/s" else {})
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics:")
    for k, v in metrics.items():
        if k != "units":
            print(f"  - {k}: {v:.6f}")
    print("\nSaved: metrics.json")

if __name__ == "__main__":
    run_eval(T_CUT=0, N_SAMPLES=300)
