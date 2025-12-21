import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import linalg

from src.model import TransformerGenerator
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    LATENT_DIM, SEQ_LEN, D_MODEL, FF_DIM, N_HEADS, N_LAYERS_G
)


# ============================================================
#  TRAJECTORY GENERATION
# ============================================================

def generate_trajectories(G, S, latent_dim, device, T_CUT=20):
    """
    Generate trajectories using the trained generator.
    S: (N, 4) conditioning vectors
    Returns fake trajectories shape = (N, T-T_CUT, 4)
    """
    G.eval()

    if isinstance(S, np.ndarray):
        S = torch.tensor(S, dtype=torch.float32)

    S = S.to(device)

    N = len(S)
    z = torch.randn(N, latent_dim, device=device)

    with torch.no_grad():
        fake = G(z, S).cpu().numpy()  # (N, 120, 4)

    if T_CUT > 0:
        fake = fake[:, T_CUT:, :]

    return fake


# ============================================================
#  FID
# ============================================================

def compute_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean))


# ============================================================
#  GAN PERFORMANCE METRICS
# ============================================================

def evaluate_performance(real, fake, S):
    """
    real: (N, T, 4)
    fake: (N, T, 4)
    S:    (N, 4)
    """
    metrics = {}

    # Position error
    metrics["pos_error"] = float(np.mean(np.abs(real[:, :, :2] - fake[:, :, :2])))

    # Speed error
    metrics["speed_error"] = float(np.mean(np.abs(real[:, :, 2] - fake[:, :, 2])))

    # Angle error
    metrics["angle_error"] = float(np.mean(np.abs(real[:, :, 3] - fake[:, :, 3])))

    # Diversity
    metrics["diversity"] = float(np.mean(np.std(fake, axis=0)))

    # Realism score
    real_mean = real.mean(axis=(0,1))
    fake_mean = fake.mean(axis=(0,1))
    metrics["realism_score"] = float(1.0 / (1.0 + np.mean(np.abs(real_mean - fake_mean))))

    # Boundary errors
    start_fake = fake[:, 0, :2]
    end_fake = fake[:, -1, :2]
    start_real = S[:, :2]
    end_real = S[:, 2:]

    metrics["start_boundary_error"] = float(np.mean(np.abs(start_fake - start_real)))
    metrics["end_boundary_error"] = float(np.mean(np.abs(end_fake - end_real)))

    return metrics


# ============================================================
#  FID + t-SNE Visualization
# ============================================================

def compute_fid_and_tsne(real, fake, out_path="tsne_plot.png"):
    real_flat = real.reshape(real.shape[0], -1)
    fake_flat = fake.reshape(fake.shape[0], -1)

    mu_r, mu_f = real_flat.mean(axis=0), fake_flat.mean(axis=0)
    sigma_r = np.cov(real_flat, rowvar=False)
    sigma_f = np.cov(fake_flat, rowvar=False)

    fid = compute_fid(mu_r, sigma_r, mu_f, sigma_f)

    # t-SNE
    size = min(len(real_flat), len(fake_flat))
    real_small = real_flat[:size]
    fake_small = fake_flat[:size]

    feats = np.concatenate([real_small, fake_small], axis=0)
    labels = np.array([0]*size + [1]*size)

    print("🌀 Running t-SNE...")
    tsne = TSNE(
        n_components=2, perplexity=30, learning_rate=200,
        n_iter=1500, random_state=42
    )
    tsne_results = tsne.fit_transform(feats)

    plt.figure(figsize=(9,7))
    plt.scatter(tsne_results[labels==0,0], tsne_results[labels==0,1], alpha=0.6, label="Real")
    plt.scatter(tsne_results[labels==1,0], tsne_results[labels==1,1], alpha=0.6, label="Fake")
    plt.legend()
    plt.title(f"Real vs Fake Trajectories (t-SNE)\nFID = {fid:.2f}")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("📁 t-SNE saved:", out_path)
    return fid, tsne_results


# ============================================================
#  MASTER PIPELINE
# ============================================================

def run_full_evaluation(T_CUT=20, N_SAMPLES=200):
    """
    Full pipeline:
    1. Load data & model
    2. Generate fake samples
    3. Compute metrics
    4. Compute FID + t-SNE
    5. Save everything
    """

    print("\n🚀 FULL TRANSFORMER GAN EVALUATION")
    print("-----------------------------------")

    # Load data
    X = np.load(os.path.join(DATA_DIR, "X_train.npy"))   # (N,120,4)
    S = np.load(os.path.join(DATA_DIR, "S_train.npy"))   # (N,4)

    # Sample subset
    idx = np.random.choice(len(X), size=N_SAMPLES, replace=False)
    real = X[idx]
    S_subset = S[idx]

    # Trim real
    if T_CUT > 0:
        real = real[:, T_CUT:, :]

    # Load model
    G = TransformerGenerator(
        latent_dim=LATENT_DIM,
        cond_dim=4,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS_G,
        ff_dim=FF_DIM
    ).to(DEVICE)

    G.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "transformer_G_final.pt"), map_location=DEVICE))
    G.eval()

    # Generate fake
    fake = generate_trajectories(G, S_subset, LATENT_DIM, DEVICE, T_CUT=T_CUT)

    # METRICS
    metrics = evaluate_performance(real, fake, S_subset)

    # FID + t-SNE
    fid, _ = compute_fid_and_tsne(real, fake, out_path="tsne_plot.png")
    metrics["fid"] = fid

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n📊 Evaluation Metrics:")
    for k,v in metrics.items():
        print(f"  • {k}: {v:.4f}")

    print("\n📁 Metrics saved to metrics.json")
    print("✅ Done!")



# ============================================================
#  RUN IF CALLED DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_full_evaluation(T_CUT=20, N_SAMPLES=300)
