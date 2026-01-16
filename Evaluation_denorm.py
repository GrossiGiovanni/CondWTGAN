import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from src.model import TransformerGenerator
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    LATENT_DIM, SEQ_LEN, D_MODEL, FF_DIM, N_HEADS, N_LAYERS_G,
    COND_DIM
)

# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = 2000
SEED = 42

# "Carreggiata" come corridoio attorno alla traiettoria reale
CORRIDOR_RADIUS_M = 2.0  # metri

# ============================================================
# IO + NORMALIZATION
# ============================================================

def load_norm_params(pkl_path):
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)

    if "dynamic_min" not in params or "dynamic_max" not in params:
        raise KeyError(
            "normalization_params.pkl deve contenere 'dynamic_min' e 'dynamic_max'. "
            f"Chiavi trovate: {list(params.keys())}"
        )

    dyn_min = np.asarray(params["dynamic_min"], dtype=np.float64)  # (4,)
    dyn_max = np.asarray(params["dynamic_max"], dtype=np.float64)  # (4,)

    if dyn_min.shape != (4,) or dyn_max.shape != (4,):
        raise ValueError(f"dynamic_min/max devono essere shape (4,), trovati {dyn_min.shape} {dyn_max.shape}")

    return dyn_min, dyn_max, params


def denorm_X(X_norm, dyn_min, dyn_max):
    # X_norm: (..., 4) in [0,1]
    return X_norm * (dyn_max - dyn_min) + dyn_min


def denorm_S(S_norm, dyn_min, dyn_max):
    # S_norm: (..., 4) = [x0,y0,xT,yT] normalizzati usando min/max di x,y
    min_x, min_y = dyn_min[0], dyn_min[1]
    max_x, max_y = dyn_max[0], dyn_max[1]

    den_x = (max_x - min_x) if (max_x - min_x) != 0 else 1.0
    den_y = (max_y - min_y) if (max_y - min_y) != 0 else 1.0

    S = np.array(S_norm, dtype=np.float64, copy=True)
    S[:, 0] = S[:, 0] * den_x + min_x
    S[:, 1] = S[:, 1] * den_y + min_y
    S[:, 2] = S[:, 2] * den_x + min_x
    S[:, 3] = S[:, 3] * den_y + min_y
    return S


# ============================================================
# PADDING (MUST match training semantics: True = padding)
# ============================================================

def build_pad_mask(lengths_torch, T, device):
    """
    lengths_torch: (B,) long
    returns pad_mask: (B,T) bool, True where padding
    """
    B = lengths_torch.shape[0]
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    return t_idx >= lengths_torch.unsqueeze(1)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_fake(G, S_norm, L, device, latent_dim, T):
    """
    Generate with correct pad_mask, matching training.
    S_norm: (N,4)
    L: (N,) lengths
    """
    G.eval()
    N = len(S_norm)

    S_t = torch.tensor(S_norm, dtype=torch.float32, device=device)  # (N,4)
    L_t = torch.tensor(L, dtype=torch.long, device=device)          # (N,)
    pad_mask = build_pad_mask(L_t, T, device)                       # (N,T)

    z = torch.randn(N, latent_dim, device=device)
    fake = G(z, S_t, pad_mask=pad_mask).detach().cpu().numpy()      # (N,T,4) norm
    return fake


# ============================================================
# GEOMETRY: point-to-polyline distance (corridor)
# ============================================================

def point_to_polyline_min_dist(points, polyline):
    """
    points:   (P,2)  generated points
    polyline: (L,2)  real trajectory points (polyline vertices)
    returns:  (P,)   min distance from each point to polyline segments

    Uses point-to-segment distance, vectorized.
    """
    P = points.shape[0]
    L = polyline.shape[0]
    if L < 2:
        # fallback: distance to single point
        d = np.linalg.norm(points - polyline[0:1], axis=1)
        return d

    a = polyline[:-1]  # (S,2)
    b = polyline[1:]   # (S,2)
    v = b - a          # (S,2)
    vv = np.sum(v * v, axis=1)  # (S,)

    # Broadcast points vs segments
    p = points[:, None, :]      # (P,1,2)
    a2 = a[None, :, :]          # (1,S,2)
    v2 = v[None, :, :]          # (1,S,2)

    # projection t = dot(p-a, v) / dot(v,v)
    w = p - a2                  # (P,S,2)
    t = np.sum(w * v2, axis=2) / (vv[None, :] + 1e-12)  # (P,S)
    t = np.clip(t, 0.0, 1.0)

    proj = a2 + t[:, :, None] * v2  # (P,S,2)
    d2 = np.sum((p - proj) ** 2, axis=2)  # (P,S)
    return np.sqrt(np.min(d2, axis=1))    # (P,)


def corridor_metrics(real_den, fake_den, L, radius_m):
    """
    real_den, fake_den: (N,T,4) denormalizzati
    L: (N,) lunghezze valide
    radius_m: corridoio (metri)

    returns dict of corridor metrics on generated xy vs real polyline.
    """
    N, T, _ = real_den.shape
    thr = float(radius_m)

    all_viol = []
    all_excess = []
    seq_viol = 0

    for i in range(N):
        Li = int(L[i])
        Li = max(2, min(Li, T))

        real_xy = real_den[i, :Li, :2]
        fake_xy = fake_den[i, :Li, :2]

        d = point_to_polyline_min_dist(fake_xy, real_xy)  # (Li,)
        viol = d > thr
        all_viol.append(viol)

        if np.any(viol):
            seq_viol += 1
            excess = d[viol] - thr
            all_excess.append(excess)

    all_viol = np.concatenate(all_viol) if len(all_viol) else np.array([], dtype=bool)

    metrics = {}
    metrics["corridor_radius_m"] = thr
    metrics["corridor_violation_rate"] = float(np.mean(all_viol)) if all_viol.size else 0.0
    metrics["corridor_seq_violation_rate"] = float(seq_viol / max(N, 1))

    if len(all_excess) == 0:
        metrics["corridor_violation_mean_excess_m"] = 0.0
        metrics["corridor_violation_p95_excess_m"] = 0.0
    else:
        ex = np.concatenate(all_excess)
        metrics["corridor_violation_mean_excess_m"] = float(np.mean(ex))
        metrics["corridor_violation_p95_excess_m"] = float(np.percentile(ex, 95))

    return metrics


# ============================================================
# METRICS
# ============================================================

def wrap_angle_diff(dtheta):
    # Se angoli sono in radianti.
    return (dtheta + np.pi) % (2 * np.pi) - np.pi


def masked_metrics(real_den, fake_den, S_den, L):
    """
    Metriche su parte valida (Li) per sample.
    """
    N, T, _ = real_den.shape

    sde_real = []
    fde_real = []
    sde_cond = []
    fde_cond = []
    path_ratio_list = []

    real_speeds = []
    fake_speeds = []
    real_turn = []
    fake_turn = []

    for i in range(N):
        Li = int(L[i])
        Li = max(2, min(Li, T))

        r = real_den[i, :Li]
        f = fake_den[i, :Li]

        real_start = r[0, 0:2]
        real_end   = r[Li - 1, 0:2]

        cond_start = S_den[i, 0:2]
        cond_end   = S_den[i, 2:4]

        gen_start = f[0, 0:2]
        gen_end   = f[Li - 1, 0:2]

        sde_real.append(float(np.linalg.norm(gen_start - real_start)))
        fde_real.append(float(np.linalg.norm(gen_end - real_end)))

        sde_cond.append(float(np.linalg.norm(gen_start - cond_start)))  # debug
        fde_cond.append(float(np.linalg.norm(gen_end - cond_end)))      # debug

        dif = f[1:, 0:2] - f[:-1, 0:2]
        path_len = float(np.sum(np.linalg.norm(dif, axis=1)))
        direct = float(np.linalg.norm(cond_end - cond_start) + 1e-9)
        path_ratio_list.append(path_len / direct)

        real_speeds.append(r[:, 2])
        fake_speeds.append(f[:, 2])

        rt = wrap_angle_diff(np.diff(r[:, 3]))
        ft = wrap_angle_diff(np.diff(f[:, 3]))
        real_turn.append(rt)
        fake_turn.append(ft)

    real_speeds = np.concatenate(real_speeds)
    fake_speeds = np.concatenate(fake_speeds)
    real_turn = np.concatenate(real_turn)
    fake_turn = np.concatenate(fake_turn)

    metrics = {
        "SDE_real_mean": float(np.mean(sde_real)),
        "FDE_real_mean": float(np.mean(fde_real)),
        "SDE_cond_mean": float(np.mean(sde_cond)),
        "FDE_cond_mean": float(np.mean(fde_cond)),
        "path_ratio_mean": float(np.mean(path_ratio_list)),
        "path_ratio_p90": float(np.percentile(path_ratio_list, 90)),
        "W1_speed": float(wasserstein_distance(real_speeds, fake_speeds)),
        "W1_turnrate": float(wasserstein_distance(real_turn, fake_turn)),
    }
    return metrics


def build_summary_features(X_den, S_den, L):
    N, T, _ = X_den.shape
    feats = []

    for i in range(N):
        Li = int(L[i])
        Li = max(2, min(Li, T))
        x = X_den[i, :Li]

        pos = x[:, :2]
        speed = x[:, 2]
        ang = x[:, 3]

        dif = pos[1:] - pos[:-1]
        step = np.linalg.norm(dif, axis=1)
        tr = wrap_angle_diff(np.diff(ang))

        f = [
            np.mean(step), np.std(step),
            np.mean(speed), np.std(speed),
            np.mean(tr), np.std(tr),
            np.linalg.norm(pos[-1] - pos[0]),
            Li
        ]
        f += [S_den[i, 0], S_den[i, 1], S_den[i, 2], S_den[i, 3]]
        feats.append(f)

    return np.asarray(feats, dtype=np.float64)


def two_sample_1nn_accuracy(real_feats, fake_feats, k=1, seed=42):
    X = np.concatenate([real_feats, fake_feats], axis=0)
    y = np.array([0] * len(real_feats) + [1] * len(fake_feats))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def plot_overlays(real_den, fake_den, L, out_path, n_plot=20, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(L), size=min(n_plot, len(L)), replace=False)

    plt.figure(figsize=(7, 7))
    for i in idxs:
        Li = int(L[i])
        Li = max(2, min(Li, real_den.shape[1]))

        r = real_den[i, :Li, :2]
        f = fake_den[i, :Li, :2]

        plt.plot(r[:, 0], r[:, 1], alpha=0.6, linewidth=1.7)
        plt.plot(f[:, 0], f[:, 1], alpha=0.6, linewidth=1.7, linestyle="--")

    plt.title(f"Overlays Real (solid) vs Fake (dashed) - {len(idxs)} samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    # Paths
    X_path = os.path.join(DATA_DIR, "X_train.npy")
    S_path = os.path.join(DATA_DIR, "S_train_fixed.npy")   # <-- cambia in S_train_fixed.npy se lo usi
    L_path = os.path.join(DATA_DIR, "L_train.npy")
    norm_path = os.path.join(DATA_DIR, "normalization_params.pkl")
    model_path = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")  # modello appena addestrato

    # Load arrays (normalized)
    X_norm = np.load(X_path)  # (N,120,4)
    S_norm = np.load(S_path)  # (N,4)
    L = np.load(L_path)       # (N,)

    # Load normalization params
    dyn_min, dyn_max, _ = load_norm_params(norm_path)

    # Sample subset for evaluation
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(L), size=min(N_SAMPLES, len(L)), replace=False)

    Xn = X_norm[idx]
    Sn = S_norm[idx]
    Ln = L[idx]

    # Load model
    G = TransformerGenerator(
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS_G,
        ff_dim=FF_DIM
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    G.load_state_dict(state, strict=True)
    G.eval()

    # Generate fake (normalized) with padding
    fake_norm = generate_fake(G, Sn, Ln, DEVICE, LATENT_DIM, SEQ_LEN)

    # Denormalize
    X_den = denorm_X(Xn, dyn_min, dyn_max)
    fake_den = denorm_X(fake_norm, dyn_min, dyn_max)
    S_den = denorm_S(Sn, dyn_min, dyn_max)

    # Metrics
    metrics = {}
    metrics.update(masked_metrics(X_den, fake_den, S_den, Ln))

    # 1-NN two-sample test
    real_feats = build_summary_features(X_den, S_den, Ln)
    fake_feats = build_summary_features(fake_den, S_den, Ln)
    metrics["1NN_acc"] = two_sample_1nn_accuracy(real_feats, fake_feats, k=1)

    # Corridor / "road" metric (2m around REAL polyline)
    metrics.update(corridor_metrics(X_den, fake_den, Ln, radius_m=CORRIDOR_RADIUS_M))

    # Save overlay plot
    overlay_path = os.path.join(OUTPUT_DIR, "eval_overlays_real_vs_fake.png")
    plot_overlays(X_den, fake_den, Ln, overlay_path, n_plot=20, seed=1)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_denorm.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION METRICS (denormalized)")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\nSaved:", metrics_path)
    print("Saved:", overlay_path)


if __name__ == "__main__":
    main()
