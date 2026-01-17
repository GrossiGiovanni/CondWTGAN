import os
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance

# opzionali (li abiliti con flag sotto)
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

# "Carreggiata" come corridoio attorno alla traiettoria reale (metriche extra)
CORRIDOR_RADIUS_M = 2.0  # metri

# Toggle: se vuoi evaluation più snella
RUN_1NN = False
RUN_OVERLAY_PLOT = True
RUN_CORRIDOR = True

# Road-map based metric (questa è quella che ti serve)
RUN_ROADMAP_METRICS = True


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

    return dyn_min, dyn_max


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
# ROAD MAP METRICS (precise in/out carriageway)
# ============================================================

def load_road_map(data_dir):
    road_dist_path = os.path.join(data_dir, "road_dist.npy")
    road_meta_path = os.path.join(data_dir, "road_meta.json")

    road_dist = np.load(road_dist_path)  # (H,W)
    with open(road_meta_path, "r") as f:
        meta = json.load(f)

    # sanity
    H, W = int(meta["H"]), int(meta["W"])
    if road_dist.shape != (H, W):
        raise ValueError(f"road_dist shape {road_dist.shape} != meta (H,W)=({H},{W})")

    return road_dist, meta


def roadmap_point_stats(fake_den, L, road_dist, meta, eps=1e-9):
    """
    fake_den: (N,T,4) in METERS
    L: (N,) valid lengths
    road_dist: (H,W) where 0=on-road, >0 off-road (meters beyond radius)
    meta: has x_min,x_max,y_min,y_max,res_m,H,W,radius_m

    Returns:
      - percent points on/off road (valid timesteps only)
      - percent out-of-bounds (counted as off-road)
      - percent sequences with >=1 off-road point
      - mean/p95 off-road excess meters (only in-bounds off-road)
    """
    N, T, _ = fake_den.shape
    H, W = int(meta["H"]), int(meta["W"])
    x_min, x_max = float(meta["x_min"]), float(meta["x_max"])
    y_min, y_max = float(meta["y_min"]), float(meta["y_max"])
    res = float(meta["res_m"])

    # valid mask (N,T)
    t = np.arange(T)[None, :]
    Lc = np.clip(L.astype(np.int64), 0, T)[:, None]
    valid = t < Lc  # True for valid points

    xy = fake_den[:, :, :2]  # (N,T,2)
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    # in-bounds (only for valid points)
    inb = valid & (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)

    # indices for all points (we will only read where inb is True)
    col = np.floor((x - x_min) / res).astype(np.int64)
    row = np.floor((y - y_min) / res).astype(np.int64)
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    total_valid = int(np.sum(valid))
    if total_valid == 0:
        return {
            "road_total_points": 0,
            "road_on_points": 0,
            "road_off_points": 0,
            "road_oob_points": 0,
            "road_on_pct": 0.0,
            "road_off_pct": 0.0,
            "road_oob_pct": 0.0,
            "road_seq_off_pct": 0.0,
            "road_off_mean_excess_m": 0.0,
            "road_off_p95_excess_m": 0.0,
            "road_radius_m": float(meta.get("radius_m", np.nan)),
        }

    # out-of-bounds valid points => off-road
    oob = valid & (~inb)
    oob_points = int(np.sum(oob))

    # sample road_dist only for in-bounds points
    # (use flattened indexing for speed)
    inb_idx = np.where(inb)
    d = road_dist[row[inb_idx], col[inb_idx]]  # distances for in-bounds valid points

    on = d <= eps
    off = ~on

    on_points = int(np.sum(on))
    off_inb_points = int(np.sum(off))
    off_points = off_inb_points + oob_points

    # sequence-level: at least one off-road (including OOB)
    # compute per-seq boolean efficiently
    off_map = np.zeros((N, T), dtype=bool)
    off_map[inb_idx] = off  # mark off-road in-bounds
    seq_off = np.any(off_map | oob, axis=1)
    seq_off_pct = float(np.mean(seq_off))

    # excess meters: only in-bounds off-road points
    if off_inb_points > 0:
        ex = d[off]  # already "meters beyond radius"
        mean_ex = float(np.mean(ex))
        p95_ex = float(np.percentile(ex, 95))
    else:
        mean_ex = 0.0
        p95_ex = 0.0

    return {
        "road_radius_m": float(meta.get("radius_m", np.nan)),
        "road_total_points": total_valid,
        "road_on_points": on_points,
        "road_off_points": off_points,
        "road_oob_points": oob_points,
        "road_on_pct": float(on_points / total_valid),
        "road_off_pct": float(off_points / total_valid),
        "road_oob_pct": float(oob_points / total_valid),
        "road_seq_off_pct": seq_off_pct,
        "road_off_mean_excess_m": mean_ex,
        "road_off_p95_excess_m": p95_ex,
        "road_eps": float(eps),
    }


# ============================================================
# GEOMETRY: point-to-polyline distance (corridor)
# ============================================================

def point_to_polyline_min_dist(points, polyline):
    """
    points:   (P,2)  generated points
    polyline: (L,2)  real trajectory points (polyline vertices)
    returns:  (P,)   min distance from each point to polyline segments
    """
    P = points.shape[0]
    L = polyline.shape[0]
    if L < 2:
        return np.linalg.norm(points - polyline[0:1], axis=1)

    a = polyline[:-1]  # (S,2)
    b = polyline[1:]   # (S,2)
    v = b - a          # (S,2)
    vv = np.sum(v * v, axis=1)  # (S,)

    p = points[:, None, :]      # (P,1,2)
    a2 = a[None, :, :]          # (1,S,2)
    v2 = v[None, :, :]          # (1,S,2)

    w = p - a2
    t = np.sum(w * v2, axis=2) / (vv[None, :] + 1e-12)
    t = np.clip(t, 0.0, 1.0)

    proj = a2 + t[:, :, None] * v2
    d2 = np.sum((p - proj) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def corridor_metrics(real_den, fake_den, L, radius_m):
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

        d = point_to_polyline_min_dist(fake_xy, real_xy)
        viol = d > thr
        all_viol.append(viol)

        if np.any(viol):
            seq_viol += 1
            all_excess.append(d[viol] - thr)

    all_viol = np.concatenate(all_viol) if len(all_viol) else np.array([], dtype=bool)

    metrics = {
        "corridor_radius_m": thr,
        "corridor_violation_rate": float(np.mean(all_viol)) if all_viol.size else 0.0,
        "corridor_seq_violation_rate": float(seq_viol / max(N, 1)),
    }

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

        # debug: quanto rispetta la condizione start/end
        sde_cond.append(float(np.linalg.norm(gen_start - cond_start)))
        fde_cond.append(float(np.linalg.norm(gen_end - cond_end)))

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

    return {
        "SDE_real_mean": float(np.mean(sde_real)),
        "FDE_real_mean": float(np.mean(fde_real)),
        "SDE_cond_mean": float(np.mean(sde_cond)),
        "FDE_cond_mean": float(np.mean(fde_cond)),
        "path_ratio_mean": float(np.mean(path_ratio_list)),
        "path_ratio_p90": float(np.percentile(path_ratio_list, 90)),
        "W1_speed": float(wasserstein_distance(real_speeds, fake_speeds)),
        "W1_turnrate": float(wasserstein_distance(real_turn, fake_turn)),
    }


# ============================================================
# OPTIONAL: 1NN two-sample test
# ============================================================

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
            Li,
            S_den[i, 0], S_den[i, 1], S_den[i, 2], S_den[i, 3]
        ]
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


# ============================================================
# OPTIONAL: plots
# ============================================================

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
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
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
    S_path = os.path.join(DATA_DIR, "S_train_fixed.npy")
    L_path = os.path.join(DATA_DIR, "L_train.npy")
    norm_path = os.path.join(DATA_DIR, "normalization_params.pkl")
    model_path = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")

    # Load arrays (normalized)
    X_norm = np.load(X_path)  # (N,T,4)
    S_norm = np.load(S_path)  # (N,4)
    L = np.load(L_path)       # (N,)

    # Load normalization params
    dyn_min, dyn_max = load_norm_params(norm_path)

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

    # Road-map metrics: % punti dentro/fuori carreggiata (preciso)
    if RUN_ROADMAP_METRICS:
        road_dist, road_meta = load_road_map(DATA_DIR)
        metrics.update(roadmap_point_stats(fake_den, Ln, road_dist, road_meta))

    # 1-NN two-sample test (opzionale)
    if RUN_1NN:
        real_feats = build_summary_features(X_den, S_den, Ln)
        fake_feats = build_summary_features(fake_den, S_den, Ln)
        metrics["1NN_acc"] = two_sample_1nn_accuracy(real_feats, fake_feats, k=1, seed=SEED)

    # Corridor metric (opzionale)
    if RUN_CORRIDOR:
        metrics.update(corridor_metrics(X_den, fake_den, Ln, radius_m=CORRIDOR_RADIUS_M))

    # Save overlay plot (opzionale)
    if RUN_OVERLAY_PLOT:
        overlay_path = os.path.join(OUTPUT_DIR, "eval_overlays_real_vs_fake.png")
        plot_overlays(X_den, fake_den, Ln, overlay_path, n_plot=20, seed=1)
    else:
        overlay_path = None

    # Save metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    if overlay_path is not None:
        print("Saved:", overlay_path)


if __name__ == "__main__":
    main()
