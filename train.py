import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, FEATURE_DIM, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G, N_LAYERS_D,
    BATCH_SIZE, EPOCHS, N_CRITIC, GAMMA_R1R2,
    G_LR, D_LR ,NOISE_STD, NOISE_DECAY, NOISE_MIN,
    DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
    DEVICE,
    LATENT_DIM,
    N_CRITIC, GAMMA_R1R2,
    G_LR, D_LR,
    EPOCHS,
    NOISE_STD, NOISE_DECAY, NOISE_MIN,
    USE_EMA, EMA_DECAY,LAMBDA_PHYS,LAMBDA_ROAD,R1R2_EVERY,
    ROAD_WARMUP_START, ROAD_WARMUP_RAMP
)

from src.model import TransformerGenerator, TransformerDiscriminator
from src.utils import load_numpy_triplet, ensure_dir
from src.trainer import train_ultra_stable_traffic_gan
from src.visualize import make_sample_callback
from src.utils import load_road_tensors

def main():

    print("🚦 TRAINING TRANSFORMER GAN (NUOVO DATASET)")

    # ---------------------------------------------------------------
    # 1. Load dataset (GIÀ NORMALIZZATO 0–1)
    # ---------------------------------------------------------------

    X_path = os.path.join(DATA_DIR, "X_train.npy")
    S_path = os.path.join(DATA_DIR, "S_train_fixed.npy")
    L_path = os.path.join(DATA_DIR, "L_train.npy")

    X, S, L = load_numpy_triplet(X_path, S_path, L_path)
    print("Loaded X:", X.shape)  # (B,120,4)
    print("Loaded S:", S.shape)  # (B,4)
    print("Loaded L:", L.shape, "min/med/max:", L.min(), np.median(L), L.max())

    assert X.shape[1] == SEQ_LEN, "SEQ_LEN mismatch"
    assert X.shape[2] == FEATURE_DIM, "FEATURE_DIM mismatch"
    assert S.shape[1] == COND_DIM, "COND_DIM mismatch"
    

    road_dist, road_meta = load_road_tensors(DATA_DIR, DEVICE)

    # ---------------------------------------------------------------
    # 2. Create DataLoader
    # ---------------------------------------------------------------

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(S, dtype=torch.float32),
        torch.tensor(L, dtype=torch.long),
    )

    dataloader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,             
        persistent_workers=True,
        prefetch_factor=4
    )


    # ---------------------------------------------------------------
    # 3. Build Transformer Models
    # ---------------------------------------------------------------

    print("⚙️ Building Transformer Generator + Discriminator")

    G = TransformerGenerator(
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS_G,
        ff_dim=FF_DIM,
    ).to(DEVICE)

    D = TransformerDiscriminator(
        cond_dim=COND_DIM,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS_D,
        ff_dim=FF_DIM,
    ).to(DEVICE)

    # ---------------------------------------------------------------
    # 4. Sample callback
    # ---------------------------------------------------------------

    sample_dir = ensure_dir(os.path.join(OUTPUT_DIR, "samples_transformer"))
    sample_callback = make_sample_callback(
        out_dir=sample_dir,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        dataset=ds,
        
    )

    # ---------------------------------------------------------------
    # 5. TRAINING
    # ---------------------------------------------------------------

    print(" " \
    "" \
    "" \
    "Starting Transformer GAN training" \
    "" \
    "" \
    "" \
    "")

    history, G_final = train_ultra_stable_traffic_gan(
        G, D, dataloader,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        n_critic=N_CRITIC,
        gamma_r1r2=GAMMA_R1R2,
        lambda_phys=LAMBDA_PHYS,
        lambda_road=LAMBDA_ROAD,
        data_dir=DATA_DIR,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        g_lr=G_LR,
        d_lr=D_LR,
        epochs=EPOCHS,
        out_dir=CHECKPOINT_DIR,
        sample_callback=sample_callback,
        noise_std=NOISE_STD,
        noise_decay=NOISE_DECAY,
        noise_min=NOISE_MIN,
        r1r2_every=R1R2_EVERY,         # target
        road_warmup_start=ROAD_WARMUP_START,
        road_warmup_ramp=ROAD_WARMUP_RAMP,
        road_norm_by_valid=True,
        road_power=2.0,
        
    )

    # ---------------------------------------------------------------
    # 6. Save final generator
    # ---------------------------------------------------------------

    model_path = os.path.join(OUTPUT_DIR, "transformer_G_final.pt")
    torch.save(G_final.state_dict(), model_path)

    print(f"✅ DONE! Model saved at: {model_path}")


if __name__ == "__main__":
    main()