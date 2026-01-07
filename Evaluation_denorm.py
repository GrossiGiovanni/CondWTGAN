import os
import json
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from src.model import TransformerGenerator
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    LATENT_DIM, SEQ_LEN, D_MODEL, FF_DIM, N_HEADS, N_LAYERS_G,
    COND_DIM
)

# ============================================================
#  NORMALIZATION / DENORMALIZATION
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

    if dyn_min.shape[0] != 4 or dyn_max.shape[0] != 4:
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
#  MODEL LOADING (robusto: state_dict o checkpoint dict)
# ============================================================

def _strip_module_prefix(state_dict):
    # gestisce modelli salvati con DataParallel (chiavi "module.xxx")
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_generator_weights(G, model_path, device, prefer_ema=True):
    obj = torch.load(model_path, map_location=device)

    # caso 1: puro state_dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # caso checkpoint: cerca chiavi note
        cand_keys = []
        if prefer_ema:
            cand_keys += ["G_ema"]
        cand_keys += ["G", "generator", "state_dict", "model"]

        state_dict = None
        for k in cand_keys:
            if k in obj and obj[k] is not None and isinstance(obj[k], dict):
                state_dict = obj[k]
                break

        # se non trovato, prova a interpretare "obj" stesso come state_dict
        if state_dict is None:
            state_dict = obj

        state_dict = _strip_module_prefix(state_dict)
        G.load_state_dict(state_dict, strict=True)
        return

    raise ValueError(f"Formato file modello non riconosciuto: {type(obj)}")

# ============================================================
#  GENERATION
# ============================================================

@torch.no_grad()
def generate_fake(G, S_norm, device, latent_dim):
    G.eval()
    S_t = torch.tensor(S_norm, dtype=torch.float32, device=device)  # (N,4)
    z = torch.randn(len(S_t), latent_dim, device=device)
    fake = G(z, S_t).detach().cpu().numpy()  # (N, T, 4) norm
    return fake

# ============================================================
#  METRICS BASE (già presenti + extra)
# ============================================================

def masked_metrics(real_den, fake_den, S_den, L, dt=0.5):
    """
    real_den, fake_den: (N, T, 4) denormalizzati
    S_den: (N, 4) denorm
    L: (N,) int lunghezze reali
    dt: timestep in secondi (se 120 step = 60s -> dt=0.5)
    """
    N, T, _ = real_den.shape

    sde_list = []
    fde_list = []
    path_ratio_list = []

    real_speeds = []
    fake_speeds = []
    real_turn = []
    fake_turn = []

    real_acc = []
    fake_acc = []
    real_jerk = []
    fake_jerk = []

    stop_thr = 0.2  # m/s (aggiusta se serve)
    real_stop_ratio = []
    fake_stop_ratio = []

    for i in range(N):
        Li = int(L[i])
        Li = max(2, min(Li, T))

        r = real_den[i, :Li]
        f = fake_den[i, :Li]

        # start/end
        start_cond = S_den[i, 0:2]
        end_cond = S_den[i, 2:4]

        start_gen = f[0, 0:2]
        end_gen = f[Li - 1, 0:2]

        sde = float(np.linalg.norm(start_gen - start_cond))
        fde = float(np.linalg.norm(end_gen - end_cond))

        sde_list.append(sde)
        fde_list.append(fde)

        # path length ratio (fake)
        dif = f[1:, 0:2] - f[:-1, 0:2]
        path_len = float(np.sum(np.linalg.norm(dif, axis=1)))
        direct = float(np.linalg.norm(end_cond - start_cond) + 1e-9)
        path_ratio_list.append(path_len / direct)

        # speed + turnrate
        real_speeds.append(r[:, 2])
        fake_speeds.append(f[:, 2])

        real_turn.append(np.diff(r[:, 3]))
        fake_turn.append(np.diff(f[:, 3]))

        # acceleration / jerk (da speed)
        ra = np.diff(r[:, 2]) / dt
        fa = np.diff(f[:, 2]) / dt
        real_acc.append(ra)
        fake_acc.append(fa)

        if len(ra) >= 2:
            real_jerk.append(np.diff(ra) / dt)
            fake_jerk.append(np.diff(fa) / dt)

        # stop ratio
        real_stop_ratio.append(float(np.mean(r[:, 2] < stop_thr)))
        fake_stop_ratio.append(float(np.mean(f[:, 2] < stop_thr)))

    real_speeds = np.concatenate(real_speeds)
    fake_speeds = np.concatenate(fake_speeds)
    real_turn = np.concatenate(real_turn) if len(real_turn) else np.array([0.0])
    fake_turn = np.concatenate(fake_turn) if len(fake_turn) else np.array([0.0])

    real_acc = np.concatenate(real_acc) if len(real_acc) else np.array([0.0])
    fake_acc = np.concatenate(fake_acc) if len(fake_acc) else np.array([0.0])
    real_jerk = np.concatenate(real_jerk) if len(real_jerk) else np.array([0.0])
    fake_jerk = np.concatenate(fake_jerk) if len(fake_jerk) else np.array([0.0])

    metrics = {}
    metrics["SDE_mean"] = float(np.mean(sde_list))
    metrics["SDE_p50"] = float(np.percentile(sde_list, 50))
    metrics["SDE_p90"] = float(np.percentile(sde_list, 90))
    metrics["SDE_p95"] = float(np.percentile(sde_list, 95))

    metrics["FDE_mean"] = float(np.mean(fde_list))
    metrics["FDE_p50"] = float(np.percentile(fde_list, 50))
    metrics["FDE_p90"] = float(np.percentile(fde_list, 90))
    metrics["FDE_p95"] = float(np.percentile(fde_list, 95))

    metrics["path_ratio_mean"] = float(np.mean(path_ratio_list))
    metrics["path_ratio_p90"] = float(np.percentile(path_ratio_list, 90))

    # Wasserstein distances
    metrics["W1_speed"] = float(wasserstein_distance(real_speeds, fake_speeds))
    metrics["W1_turnrate"] = float(wasserstein_distance(real_turn, fake_turn))
    metrics["W1_acc"] = float(wasserstein_distance(real_acc, fake_acc))
    metrics["W1_jerk"] = float(wasserstein_distance(real_jerk, fake_jerk))

    # stop ratio
    metrics["stop_ratio_real_mean"] = float(np.mean(real_stop_ratio))
    metrics["stop_ratio_fake_mean"] = float(np.mean(fake_stop_ratio))

    return metrics


def build_summary_features(X_den, S_den, L, dt=0.5):
    """
    Feature summary per 1-NN two-sample test (real vs fake)
    """
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

        tr = np.diff(ang)
        acc = np.diff(speed) / dt if Li >= 2 else np.array([0.0])

        f = [
            np.mean(step), np.std(step),
            np.mean(speed), np.std(speed),
            np.mean(tr), np.std(tr),
            np.mean(acc), np.std(acc),
            np.linalg.norm(pos[-1] - pos[0]),  # displacement
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

# ============================================================
#  ROAD MASK (data-driven "carreggiata")
# ============================================================

def collect_valid_xy(X_den, L):
    """Ritorna array (M,2) con solo punti validi (no padding)."""
    xs = []
    for i in range(len(L)):
        Li = int(L[i])
        Li = max(1, min(Li, X_den.shape[1]))
        xs.append(X_den[i, :Li, :2])
    return np.concatenate(xs, axis=0)


def build_road_mask_from_real(real_xy, bounds, res=512, smooth_sigma=1.2, count_thresh=3, dilate_iter=2):
    """
    Crea una maschera binaria (res x res) della "carreggiata" usando occupancy dei punti reali.
    bounds = (minx, maxx, miny, maxy)
    """
    minx, maxx, miny, maxy = bounds

    # histogram2d: x su asse 0, y su asse 1 (attenzione poi a indexing immagine)
    H, xedges, yedges = np.histogram2d(
        real_xy[:, 0], real_xy[:, 1],
        bins=res,
        range=[[minx, maxx], [miny, maxy]]
    )

    # smoothing per riempire buchini
    Hs = gaussian_filter(H, sigma=smooth_sigma)

    # soglia: presenza minima
    mask = Hs >= count_thresh

    # dilatazione per robustezza (bordo carreggiata)
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)

    # distance transform per distanza fuori carreggiata:
    # dt fuori strada = distanza dal "True" più vicino
    # distance_transform_edt calcola distanza dai pixel 0, quindi invertiamo
    dist_to_road = distance_transform_edt(~mask)

    return mask, dist_to_road


def points_in_mask(points_xy, mask, bounds):
    """
    points_xy: (M,2) in coordinate mondo
    mask: (res,res) boolean
    bounds: (minx,maxx,miny,maxy)
    ritorna boolean (M,) se punto cade in mask=True
    """
    minx, maxx, miny, maxy = bounds
    resx, resy = mask.shape

    # normalizza su [0, res-1]
    u = (points_xy[:, 0] - minx) / (maxx - minx + 1e-12)
    v = (points_xy[:, 1] - miny) / (maxy - miny + 1e-12)

    ix = np.clip((u * (resx - 1)).astype(int), 0, resx - 1)
    iy = np.clip((v * (resy - 1)).astype(int), 0, resy - 1)

    # NB: mask è indicizzata [ix, iy] perché histogram2d produce H[xbin, ybin]
    return mask[ix, iy], ix, iy


def road_metrics(real_den, fake_den, L, bounds, res=512, smooth_sigma=1.2, count_thresh=3, dilate_iter=2):
    """
    Calcola:
    - % punti in carreggiata per traiettoria (real e fake)
    - distanza media/p90 dei punti fake fuori carreggiata
    Restituisce dict + mask e dist_to_road per plotting.
    """
    real_xy_all = collect_valid_xy(real_den, L)
    mask, dist_to_road = build_road_mask_from_real(
        real_xy_all, bounds, res=res,
        smooth_sigma=smooth_sigma, count_thresh=count_thresh, dilate_iter=dilate_iter
    )

    # world units per pixel (approssimazione)
    minx, maxx, miny, maxy = bounds
    px = (maxx - minx) / res
    py = (maxy - miny) / res
    pix_size = float(np.mean([px, py]))

    real_pct = []
    fake_pct = []
    fake_offroad_dist = []

    for i in range(len(L)):
        Li = int(L[i])
        Li = max(1, min(Li, real_den.shape[1]))

        rxy = real_den[i, :Li, :2]
        fxy = fake_den[i, :Li, :2]

        rin, _, _ = points_in_mask(rxy, mask, bounds)
        fin, fix, fiy = points_in_mask(fxy, mask, bounds)

        real_pct.append(float(np.mean(rin)))
        fake_pct.append(float(np.mean(fin)))

        # distanza solo per punti fake fuori
        out = ~fin
        if np.any(out):
            dpx = dist_to_road[fix[out], fiy[out]]
            fake_offroad_dist.append(dpx * pix_size)

    fake_offroad_dist = np.concatenate(fake_offroad_dist) if len(fake_offroad_dist) else np.array([0.0])

    out = {
        "road_in_pct_real_mean": float(np.mean(real_pct)),
        "road_in_pct_fake_mean": float(np.mean(fake_pct)),
        "road_in_pct_fake_p10": float(np.percentile(fake_pct, 10)),
        "road_in_pct_fake_p50": float(np.percentile(fake_pct, 50)),
        "road_in_pct_fake_p90": float(np.percentile(fake_pct, 90)),
        "traj_fake_over_95pct_inroad": float(np.mean(np.array(fake_pct) >= 0.95)),
        "fake_offroad_dist_mean": float(np.mean(fake_offroad_dist)),
        "fake_offroad_dist_p90": float(np.percentile(fake_offroad_dist, 90)),
    }
    return out, mask, dist_to_road, np.array(real_pct), np.array(fake_pct)

# ============================================================
#  PLOTS
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
        plt.plot(r[:, 0], r[:, 1], alpha=0.5, linewidth=1.5)
        plt.plot(f[:, 0], f[:, 1], alpha=0.5, linewidth=1.5, linestyle="--")

    plt.title(f"Overlays Real (solid) vs Fake (dashed) - {len(idxs)} samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_histogram(a, b, title, xlabel, out_path, bins=30):
    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=bins, alpha=0.6, label="Real")
    plt.hist(b, bins=bins, alpha=0.6, label="Fake")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_roadmask_and_samples(mask, bounds, real_den, fake_den, L, out_path, n_plot=30, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(L), size=min(n_plot, len(L)), replace=False)

    # imshow: attenzione, qui mostriamo mask trasposta per allineare assi in modo “visivo”
    # Per semplicità: usiamo extent e transponiamo.
    minx, maxx, miny, maxy = bounds

    plt.figure(figsize=(8, 7))
    plt.imshow(
        mask.T, origin="lower",
        extent=[minx, maxx, miny, maxy],
        alpha=0.35,
        aspect="auto"
    )

    for i in idxs:
        Li = int(L[i])
        Li = max(2, min(Li, real_den.shape[1]))
        r = real_den[i, :Li, :2]
        f = fake_den[i, :Li, :2]
        plt.plot(r[:, 0], r[:, 1], alpha=0.5, linewidth=1.2)
        plt.plot(f[:, 0], f[:, 1], alpha=0.5, linewidth=1.2, linestyle="--")

    plt.title("Road mask (data-driven) + samples Real/Fake")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=os.path.join(OUTPUT_DIR, "eval_plus"))
    parser.add_argument("--model_path", type=str, default=os.path.join(OUTPUT_DIR, "transformer_G_final.pt"))

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--mask_build_samples", type=int, default=6000)  # più alto = mask più stabile
    parser.add_argument("--dt", type=float, default=0.5)

    # road mask params
    parser.add_argument("--mask_res", type=int, default=512)
    parser.add_argument("--mask_smooth_sigma", type=float, default=1.2)
    parser.add_argument("--mask_count_thresh", type=float, default=3.0)
    parser.add_argument("--mask_dilate_iter", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Paths
    X_path = os.path.join(args.data_dir, "X_train.npy")
    S_path = os.path.join(args.data_dir, "S_train.npy")
    L_path = os.path.join(args.data_dir, "L_train.npy")
    norm_path = os.path.join(args.data_dir, "normalization_params.pkl")

    # Load arrays (normalized)
    X_norm = np.load(X_path)  # (N,120,4)
    S_norm = np.load(S_path)  # (N,4)
    L = np.load(L_path)       # (N,)

    # Norm params
    dyn_min, dyn_max, _ = load_norm_params(norm_path)
    bounds = (float(dyn_min[0]), float(dyn_max[0]), float(dyn_min[1]), float(dyn_max[1]))

    rng = np.random.default_rng(42)

    # subset per evaluation
    idx_eval = rng.choice(len(L), size=min(args.n_samples, len(L)), replace=False)
    Xn = X_norm[idx_eval]
    Sn = S_norm[idx_eval]
    Ln = L[idx_eval]

    # subset per road mask (usa più dati, così la mask è più “carreggiata” e meno “rumore”)
    idx_mask = rng.choice(len(L), size=min(args.mask_build_samples, len(L)), replace=False)
    X_mask_den = denorm_X(X_norm[idx_mask], dyn_min, dyn_max)

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

    load_generator_weights(G, args.model_path, DEVICE, prefer_ema=True)
    G.eval()

    # Generate fake (normalized)
    fake_norm = generate_fake(G, Sn, DEVICE, LATENT_DIM)

    # Denormalize
    X_den = denorm_X(Xn, dyn_min, dyn_max)
    fake_den = denorm_X(fake_norm, dyn_min, dyn_max)
    S_den = denorm_S(Sn, dyn_min, dyn_max)

    # BASE metrics
    metrics = masked_metrics(X_den, fake_den, S_den, Ln, dt=args.dt)

    # 1-NN two-sample test
    real_feats = build_summary_features(X_den, S_den, Ln, dt=args.dt)
    fake_feats = build_summary_features(fake_den, S_den, Ln, dt=args.dt)
    metrics["1NN_acc"] = two_sample_1nn_accuracy(real_feats, fake_feats, k=1)

    # ROAD metrics (carreggiata) - mask costruita dai real (più sample)
    road_m, road_mask, dist_to_road, real_pct, fake_pct = road_metrics(
        X_mask_den,  # usa più dati real per costruire mask
        denorm_X(generate_fake(G, S_norm[idx_mask], DEVICE, LATENT_DIM), dyn_min, dyn_max) * 0 + X_mask_den,  # dummy, non usato qui
        L[idx_mask],
        bounds=bounds,
        res=args.mask_res,
        smooth_sigma=args.mask_smooth_sigma,
        count_thresh=args.mask_count_thresh,
        dilate_iter=args.mask_dilate_iter
    )
    # NB: road_metrics sopra costruisce mask dai real; per i fake usiamo i fake_den della eval
    # quindi ricalcolo percentuali usando la mask appena creata:
    # (così le metriche sono coerenti: mask robusta + fake eval)
    # real eval %
    real_pct_eval = []
    fake_pct_eval = []
    fake_off = []
    minx, maxx, miny, maxy = bounds
    px = (maxx - minx) / args.mask_res
    py = (maxy - miny) / args.mask_res
    pix_size = float(np.mean([px, py]))

    for i in range(len(Ln)):
        Li = int(Ln[i])
        Li = max(1, min(Li, X_den.shape[1]))
        rxy = X_den[i, :Li, :2]
        fxy = fake_den[i, :Li, :2]

        rin, _, _ = points_in_mask(rxy, road_mask, bounds)
        fin, fix, fiy = points_in_mask(fxy, road_mask, bounds)

        real_pct_eval.append(float(np.mean(rin)))
        fake_pct_eval.append(float(np.mean(fin)))

        out = ~fin
        if np.any(out):
            dpx = dist_to_road[fix[out], fiy[out]]
            fake_off.append(dpx * pix_size)

    fake_off = np.concatenate(fake_off) if len(fake_off) else np.array([0.0])

    metrics["road_in_pct_real_mean"] = float(np.mean(real_pct_eval))
    metrics["road_in_pct_fake_mean"] = float(np.mean(fake_pct_eval))
    metrics["road_in_pct_fake_p10"] = float(np.percentile(fake_pct_eval, 10))
    metrics["road_in_pct_fake_p50"] = float(np.percentile(fake_pct_eval, 50))
    metrics["road_in_pct_fake_p90"] = float(np.percentile(fake_pct_eval, 90))
    metrics["traj_fake_over_95pct_inroad"] = float(np.mean(np.array(fake_pct_eval) >= 0.95))
    metrics["fake_offroad_dist_mean"] = float(np.mean(fake_off))
    metrics["fake_offroad_dist_p90"] = float(np.percentile(fake_off, 90))

    # Plots
    overlay_path = os.path.join(args.output_dir, "eval_overlays_real_vs_fake.png")
    plot_overlays(X_den, fake_den, Ln, overlay_path, n_plot=25, seed=1)

    hist_road_path = os.path.join(args.output_dir, "hist_inroad_pct.png")
    plot_histogram(
        np.array(real_pct_eval), np.array(fake_pct_eval),
        title="Percentuale punti in carreggiata (mask data-driven)",
        xlabel="in-road ratio per traiettoria",
        out_path=hist_road_path,
        bins=30
    )

    # Speed / turnrate histogram (utile per capire se “imita” davvero o bara)
    hist_speed_path = os.path.join(args.output_dir, "hist_speed.png")
    # flatten valid speeds
    def flat_speed(Xd, Larr):
        out = []
        for i in range(len(Larr)):
            Li = int(Larr[i])
            Li = max(1, min(Li, Xd.shape[1]))
            out.append(Xd[i, :Li, 2])
        return np.concatenate(out)

    hist_turn_path = os.path.join(args.output_dir, "hist_turnrate.png")
    def flat_turn(Xd, Larr):
        out = []
        for i in range(len(Larr)):
            Li = int(Larr[i])
            Li = max(2, min(Li, Xd.shape[1]))
            out.append(np.diff(Xd[i, :Li, 3]))
        return np.concatenate(out) if len(out) else np.array([0.0])

    plot_histogram(
        flat_speed(X_den, Ln), flat_speed(fake_den, Ln),
        title="Speed distribution (denorm)",
        xlabel="speed",
        out_path=hist_speed_path,
        bins=40
    )
    plot_histogram(
        flat_turn(X_den, Ln), flat_turn(fake_den, Ln),
        title="Turn-rate distribution (denorm)",
        xlabel="delta angle per step",
        out_path=hist_turn_path,
        bins=40
    )

    roadmask_plot_path = os.path.join(args.output_dir, "roadmask_and_samples.png")
    plot_roadmask_and_samples(road_mask, bounds, X_den, fake_den, Ln, roadmask_plot_path, n_plot=35, seed=2)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics_denorm_plus.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("\n===== METRICS (DENORM PLUS) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\nSaved:", metrics_path)
    print("Saved:", overlay_path)
    print("Saved:", hist_road_path)
    print("Saved:", hist_speed_path)
    print("Saved:", hist_turn_path)
    print("Saved:", roadmask_plot_path)


if __name__ == "__main__":
    main()
