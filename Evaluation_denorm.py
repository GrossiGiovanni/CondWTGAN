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


@torch.no_grad()
def generate_fake(G, S_norm, device, latent_dim):
    G.eval()
    S_t = torch.tensor(S_norm, dtype=torch.float32, device=device)  # (N,4)
    z = torch.randn(len(S_t), latent_dim, device=device)
    fake = G(z, S_t).detach().cpu().numpy()  # (N, T, 4) norm
    return fake


def masked_metrics(real_den, fake_den, S_den, L):
    """
    real_den, fake_den: (N, T, 4) denormalizzati
    S_den: (N, 4) denorm
    L: (N,) int lunghezze reali
    """
    N, T, _ = real_den.shape

    sde_list = []
    fde_list = []
    path_ratio_list = []

    # per distribuzioni
    real_speeds = []
    fake_speeds = []
    real_turn = []
    fake_turn = []

    for i in range(N):
        Li = int(L[i])
        Li = max(2, min(Li, T))  # almeno 2 per differenze

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

        # speed distributions (feature 2)
        real_speeds.append(r[:, 2])
        fake_speeds.append(f[:, 2])

        # turn-rate distributions (delta angle)
        real_turn.append(np.diff(r[:, 3]))
        fake_turn.append(np.diff(f[:, 3]))

    real_speeds = np.concatenate(real_speeds)
    fake_speeds = np.concatenate(fake_speeds)
    real_turn = np.concatenate(real_turn)
    fake_turn = np.concatenate(fake_turn)

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

    # distanze distribuzionali (real vs fake)
    metrics["W1_speed"] = float(wasserstein_distance(real_speeds, fake_speeds))
    metrics["W1_turnrate"] = float(wasserstein_distance(real_turn, fake_turn))

    return metrics


def build_summary_features(X_den, S_den, L):
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

        # path
        dif = pos[1:] - pos[:-1]
        step = np.linalg.norm(dif, axis=1)

        # turn rate
        tr = np.diff(ang)

        f = [
            np.mean(step), np.std(step),
            np.mean(speed), np.std(speed),
            np.mean(tr), np.std(tr),
            np.linalg.norm(pos[-1] - pos[0]),  # displacement
            Li
        ]

        # aggiungo info condizione (denorm)
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
    acc = float(clf.score(X_test, y_test))
    return acc


def plot_overlays(real_den, fake_den, S_den, L, out_path, n_plot=20, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(L), size=min(n_plot, len(L)), replace=False)

    plt.figure(figsize=(7, 7))
    for j, i in enumerate(idxs):
        Li = int(L[i])
        Li = max(2, min(Li, real_den.shape[1]))

        r = real_den[i, :Li, :2]
        f = fake_den[i, :Li, :2]

        # real: linea piena
        plt.plot(r[:, 0], r[:, 1], alpha=0.5, linewidth=1.5)
        # fake: tratteggiata
        plt.plot(f[:, 0], f[:, 1], alpha=0.5, linewidth=1.5, linestyle="--")

    plt.title(f"Overlays Real (solid) vs Fake (dashed) - {len(idxs)} samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    # Paths
    X_path = os.path.join(DATA_DIR, "X_train.npy")
    S_path = os.path.join(DATA_DIR, "S_train.npy")
    L_path = os.path.join(DATA_DIR, "L_train.npy")
    norm_path = os.path.join(DATA_DIR, "normalization_params.pkl")
    model_path = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")

    # Load arrays (normalized)
    X_norm = np.load(X_path)  # (N,120,4)
    S_norm = np.load(S_path)  # (N,4)
    L = np.load(L_path)       # (N,)

    # Load normalization params
    dyn_min, dyn_max, _ = load_norm_params(norm_path)

    # Sample subset for evaluation
    N_SAMPLES = 1000
    rng = np.random.default_rng(42)
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

    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()

    # Generate fake (normalized)
    fake_norm = generate_fake(G, Sn, DEVICE, LATENT_DIM)

    # Denormalize
    X_den = denorm_X(Xn, dyn_min, dyn_max)
    fake_den = denorm_X(fake_norm, dyn_min, dyn_max)
    S_den = denorm_S(Sn, dyn_min, dyn_max)

    # Metrics (mask-aware)
    metrics = masked_metrics(X_den, fake_den, S_den, Ln)

    # 1-NN two-sample test
    real_feats = build_summary_features(X_den, S_den, Ln)
    fake_feats = build_summary_features(fake_den, S_den, Ln)
    metrics["1NN_acc"] = two_sample_1nn_accuracy(real_feats, fake_feats, k=1)

    # Save overlay plot
    overlay_path = os.path.join(OUTPUT_DIR, "eval_overlays_real_vs_fake.png")
    plot_overlays(X_den, fake_den, S_den, Ln, overlay_path, n_plot=20, seed=1)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_denorm.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("Saved:", metrics_path)
    print("Saved:", overlay_path)


if __name__ == "__main__":
    main()
