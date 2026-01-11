"""
train_with_road_constraint.py

VERSIONE MODIFICATA del tuo train.py con:
1. Costruzione automatica della road mask dai dati di training
2. Loss fisica ribilanciata (meno peso su endpoint, più su smoothness)
3. Penalità off-road differenziabile
4. lambda_phys aumentato

ISTRUZIONI:
1. Sostituisci il tuo train.py con questo file
2. Assicurati che normalization_params.pkl sia nella cartella data/
3. Esegui: python train_with_road_constraint.py
"""

import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt

from src.config import (
    DATA_DIR, OUTPUT_DIR, DEVICE,
    SEQ_LEN, FEATURE_DIM, COND_DIM,
    LATENT_DIM, D_MODEL, FF_DIM, N_HEADS,
    N_LAYERS_G, N_LAYERS_D,
    BATCH_SIZE, EPOCHS, N_CRITIC, GAMMA_R1R2,
    G_LR, D_LR, NOISE_STD, NOISE_DECAY, NOISE_MIN
)

from src.model import TransformerGenerator, TransformerDiscriminator
from src.utils import load_numpy_triplet, ensure_dir
from src.visualize import make_sample_callback


# =============================================================================
#  NUOVE FUNZIONI PER ROAD CONSTRAINT
# =============================================================================

def load_normalization_params(pkl_path):
    """Carica i parametri di normalizzazione dal pickle."""
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
    
    dyn_min = np.array(params["dynamic_min"], dtype=np.float32)
    dyn_max = np.array(params["dynamic_max"], dtype=np.float32)
    
    return dyn_min, dyn_max


def denormalize_X(X_norm, dyn_min, dyn_max):
    """Denormalizza X da [0,1] a coordinate mondo."""
    return X_norm * (dyn_max - dyn_min) + dyn_min


def build_road_mask_from_data(X_denorm, L, bounds, resolution=256, sigma=1.5, thresh=2, dilate=4):
    """
    Costruisce una maschera binaria della carreggiata dai dati reali.
    
    X_denorm: (N, T, 4) traiettorie denormalizzate
    L: (N,) lunghezze
    bounds: (minx, maxx, miny, maxy)
    
    Ritorna: mask (H, W) booleana
    """
    N, T, _ = X_denorm.shape
    minx, maxx, miny, maxy = bounds
    
    # Raccogli tutti i punti xy validi
    all_xy = []
    for i in range(N):
        Li = int(L[i])
        Li = max(1, min(Li, T))
        all_xy.append(X_denorm[i, :Li, :2])
    
    all_xy = np.concatenate(all_xy, axis=0)
    
    # Istogramma 2D
    H, _, _ = np.histogram2d(
        all_xy[:, 0], all_xy[:, 1],
        bins=resolution,
        range=[[minx, maxx], [miny, maxy]]
    )
    
    # Smoothing e soglia
    H_smooth = gaussian_filter(H, sigma=sigma)
    mask = H_smooth >= thresh
    
    # Dilatazione per tolleranza
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    
    return mask.astype(bool)


def build_sdf_from_mask(road_mask, bounds):
    """
    Costruisce Signed Distance Field dalla maschera.
    Positivo = fuori strada, Negativo = dentro strada.
    """
    resolution = road_mask.shape[0]
    minx, maxx, miny, maxy = bounds
    
    # Distance transform
    dist_outside = distance_transform_edt(~road_mask).astype(np.float32)
    dist_inside = distance_transform_edt(road_mask).astype(np.float32)
    sdf = dist_outside - dist_inside
    
    # Converti pixel in metri
    pixel_size = ((maxx - minx) / resolution + (maxy - miny) / resolution) / 2
    sdf_meters = sdf * pixel_size
    
    return torch.tensor(sdf_meters, dtype=torch.float32)


def sample_sdf(sdf_tensor, xy_world, bounds, device):
    """
    Campiona il SDF alle coordinate mondo usando interpolazione bilineare.
    
    sdf_tensor: (H, W)
    xy_world: (B, T, 2) coordinate mondo
    bounds: (minx, maxx, miny, maxy)
    
    Ritorna: (B, T) valori SDF
    """
    minx, maxx, miny, maxy = bounds
    B, T, _ = xy_world.shape
    H, W = sdf_tensor.shape
    
    # Normalizza a [-1, 1] per grid_sample
    x_norm = 2 * (xy_world[..., 0] - minx) / (maxx - minx) - 1
    y_norm = 2 * (xy_world[..., 1] - miny) / (maxy - miny) - 1
    
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B, T, 1, 2)
    
    sdf_batched = sdf_tensor.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).to(device)
    
    sampled = F.grid_sample(
        sdf_batched, grid.to(device),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return sampled[:, 0, :, 0]  # (B, T)


# =============================================================================
#  LOSS FISICA MIGLIORATA CON ROAD CONSTRAINT
# =============================================================================

def improved_physical_loss_with_road(
    generated_trajectories,  # (B, T, 4) normalizzato [0,1]
    s_conditions,            # (B, 4)
    lengths,                 # (B,)
    sdf_tensor,              # (H, W) SDF precomputato
    bounds,                  # (minx, maxx, miny, maxy)
    dyn_min,                 # (4,) numpy
    dyn_max,                 # (4,) numpy
    device,
    # Pesi (NUOVI VALORI BILANCIATI)
    w_start=5.0,             # Ridotto da 30
    w_end=5.0,               # Ridotto da 30
    w_smooth=2.0,            # Aumentato da 0.05
    w_speed_smooth=1.0,      # Aumentato da 0.1
    w_speed_range=0.5,
    w_road=10.0,             # NUOVO - peso penalità off-road
    w_direction=1.0,         # NUOVO - consistenza direzione
    road_margin=0.5,         # Tolleranza in metri prima della penalità
):
    """
    Loss fisica migliorata con:
    - Pesi ribilanciati (meno endpoint, più smoothness)
    - Penalità off-road differenziabile
    - Consistenza della direzione
    """
    B, T, _ = generated_trajectories.shape
    
    # Maschere di validità
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid_t = t_idx < lengths.unsqueeze(1)
    valid_pairs = valid_t[:, 1:] & valid_t[:, :-1]
    valid_triplets = valid_t[:, 2:] & valid_t[:, 1:-1] & valid_t[:, :-2]
    
    def masked_mean(x, mask):
        mask_f = mask.float()
        return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    
    losses = {}
    
    # 1) Start loss
    start_pos = generated_trajectories[:, 0, :2]
    target_start = s_conditions[:, 0:2]
    losses['start'] = w_start * F.mse_loss(start_pos, target_start)
    
    # 2) End loss
    end_idx = (lengths - 1).clamp(min=0).long()
    batch_idx = torch.arange(B, device=device)
    end_pos = generated_trajectories[batch_idx, end_idx, :2]
    target_end = s_conditions[:, 2:4]
    losses['end'] = w_end * F.mse_loss(end_pos, target_end)
    
    # 3) Smoothness (displacement)
    pos = generated_trajectories[:, :, :2]
    pos_diff = pos[:, 1:, :] - pos[:, :-1, :]
    step_norm = torch.norm(pos_diff, dim=-1)
    losses['smooth'] = w_smooth * masked_mean(step_norm, valid_pairs)
    
    # 4) Speed smoothness
    speeds = generated_trajectories[:, :, 2]
    speed_diff = torch.abs(speeds[:, 1:] - speeds[:, :-1])
    losses['speed_smooth'] = w_speed_smooth * masked_mean(speed_diff, valid_pairs)
    
    # 5) Speed range
    speed_min_pen = torch.relu(0.02 - speeds)
    speed_max_pen = torch.relu(speeds - 0.98)
    losses['speed_range'] = w_speed_range * masked_mean(speed_min_pen + speed_max_pen, valid_t)
    
    # 6) Direction consistency (NUOVO)
    if valid_triplets.any():
        direction = pos_diff / (torch.norm(pos_diff, dim=-1, keepdim=True) + 1e-8)
        dir_change = direction[:, 1:, :] - direction[:, :-1, :]
        dir_change_norm = torch.norm(dir_change, dim=-1)
        losses['direction'] = w_direction * masked_mean(dir_change_norm, valid_triplets)
    else:
        losses['direction'] = torch.tensor(0.0, device=device)
    
    # 7) ROAD CONSTRAINT (NUOVO - la parte più importante!)
    if sdf_tensor is not None:
        # Denormalizza xy a coordinate mondo
        xy_norm = generated_trajectories[:, :, :2]
        min_xy = torch.tensor(dyn_min[:2], device=device, dtype=torch.float32)
        max_xy = torch.tensor(dyn_max[:2], device=device, dtype=torch.float32)
        xy_world = xy_norm * (max_xy - min_xy) + min_xy
        
        # Campiona SDF
        sdf_values = sample_sdf(sdf_tensor, xy_world, bounds, device)
        
        # Penalità: solo per punti fuori strada (SDF > margin)
        offroad_dist = torch.relu(sdf_values - road_margin)
        offroad_penalty = offroad_dist ** 2  # Penalità quadratica
        
        losses['road'] = w_road * masked_mean(offroad_penalty, valid_t)
    else:
        losses['road'] = torch.tensor(0.0, device=device)
    
    # Totale
    total = sum(losses.values())
    
    return total, losses


# =============================================================================
#  TRAINING LOOP MODIFICATO
# =============================================================================

def train_with_road_constraint(
    G, D, dataloader,
    device, latent_dim,
    sdf_tensor, bounds, dyn_min, dyn_max,  # NUOVI PARAMETRI
    n_critic=3,
    gamma_r1r2=15.0,
    lambda_phys=1.0,      # AUMENTATO da 0.1
    w_road=10.0,          # Peso road constraint
    use_ema=False,
    ema_decay=0.999,
    g_lr=3e-4,
    d_lr=1e-4,
    epochs=120,
    out_dir="outputs",
    sample_callback=None,
    noise_std=0.0005,
    noise_decay=0.985,
    noise_min=0.00001,
    r1r2_every=3,
):
    """Training loop con road constraint."""
    
    from src.losses import rpgan_d_loss, rpgan_g_loss, r1_penalty, r2_penalty
    from torch.cuda.amp import autocast, GradScaler
    from tqdm import tqdm
    import copy
    
    G = G.to(device)
    D = D.to(device)
    G.train()
    D.train()
    
    # EMA
    G_ema = None
    if use_ema:
        G_ema = copy.deepcopy(G).to(device)
        for p in G_ema.parameters():
            p.requires_grad_(False)
    
    # Ottimizzatori
    opt_G = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.0, 0.9))
    
    # Sposta SDF su device
    if sdf_tensor is not None:
        sdf_tensor = sdf_tensor.to(device)
    
    history = {
        "d_loss": [], "g_loss": [], "r1r2": [],
        "phys_total": [], "phys_road": [], "phys_start": [], "phys_end": []
    }
    
    scaler = GradScaler()
    global_step = 0
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real_X, S, lengths in loop:
            global_step += 1
            
            real_X = real_X.to(device)
            S = S.to(device)
            lengths = lengths.to(device)
            B, T, _ = real_X.shape
            
            # Padding mask
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            pad_mask = t_idx >= lengths.unsqueeze(1)
            valid_mask_float = (~pad_mask).unsqueeze(-1).float()
            
            # Noise
            if noise_decay > 0:
                sigma = max(noise_min, noise_std * (noise_decay ** epoch))
            else:
                sigma = noise_std
            
            do_r1r2 = r1r2_every > 0 and global_step % r1r2_every == 0
            r1 = r2 = torch.zeros((), device=device)
            
            # ===== TRAIN DISCRIMINATOR =====
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)
                fake_X = G(z, S, pad_mask=pad_mask).detach()
                
                if sigma > 0:
                    real_in = real_X + sigma * torch.randn_like(real_X) * valid_mask_float
                    fake_in = fake_X + sigma * torch.randn_like(fake_X) * valid_mask_float
                else:
                    real_in, fake_in = real_X, fake_X
                
                opt_D.zero_grad(set_to_none=True)
                
                with autocast():
                    real_scores = D(real_in, S, pad_mask=pad_mask)
                    fake_scores = D(fake_in, S, pad_mask=pad_mask)
                    loss_D = rpgan_d_loss(real_scores, fake_scores)
                
                if do_r1r2:
                    r1 = r1_penalty(D, real_X.float(), S.float(), pad_mask=pad_mask)
                    r2 = r2_penalty(D, fake_X.float(), S.float(), pad_mask=pad_mask)
                
                total_D = loss_D + gamma_r1r2 * (r1 + r2)
                
                scaler.scale(total_D).backward()
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                scaler.step(opt_D)
                scaler.update()
            
            # ===== TRAIN GENERATOR =====
            z = torch.randn(B, latent_dim, device=device)
            opt_G.zero_grad(set_to_none=True)
            
            with autocast():
                fake_X = G(z, S, pad_mask=pad_mask)
                fake_scores = D(fake_X, S, pad_mask=pad_mask)
                real_scores_ref = D(real_X, S, pad_mask=pad_mask).detach()
                
                g_loss = rpgan_g_loss(real_scores_ref, fake_scores)
                
                # NUOVA LOSS FISICA CON ROAD CONSTRAINT
                phys_total, phys_breakdown = improved_physical_loss_with_road(
                    fake_X, S, lengths,
                    sdf_tensor=sdf_tensor,
                    bounds=bounds,
                    dyn_min=dyn_min,
                    dyn_max=dyn_max,
                    device=device,
                    w_start=5.0,      # Ridotto
                    w_end=5.0,        # Ridotto
                    w_smooth=2.0,     # Aumentato
                    w_speed_smooth=1.0,
                    w_speed_range=0.5,
                    w_road=w_road,    # Penalità off-road
                    w_direction=1.0,
                    road_margin=0.5,
                )
                
                total_G = g_loss + lambda_phys * phys_total
            
            scaler.scale(total_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt_G)
            scaler.update()
            
            # EMA
            if use_ema and G_ema is not None:
                for ema_p, p in zip(G_ema.parameters(), G.parameters()):
                    ema_p.data.mul_(ema_decay).add_((1 - ema_decay) * p.data)
            
            # Logging
            history["d_loss"].append(float(loss_D.item()))
            history["g_loss"].append(float(g_loss.item()))
            history["r1r2"].append(float((r1 + r2).item()))
            history["phys_total"].append(float(phys_total.item()))
            history["phys_road"].append(float(phys_breakdown['road'].item()))
            history["phys_start"].append(float(phys_breakdown['start'].item()))
            history["phys_end"].append(float(phys_breakdown['end'].item()))
            
            # Callback
            if sample_callback and global_step % 1000 == 0:
                G_eval = G_ema if (use_ema and G_ema is not None) else G
                sample_callback(G_eval, epoch, global_step)
            
            loop.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "Road": f"{phys_breakdown['road'].item():.3f}",
                "Phys": f"{phys_total.item():.3f}",
            })
        
        # Save checkpoint
        os.makedirs(out_dir, exist_ok=True)
        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "G_ema": G_ema.state_dict() if G_ema else None,
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "step": global_step,
        }
        torch.save(ckpt, os.path.join(out_dir, f"checkpoint_step_{global_step}.pt"))
    
    return history, (G_ema if use_ema and G_ema else G)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🚦 TRAINING CON ROAD CONSTRAINT")
    print("=" * 60)
    
    # ---------------------------------------------------------------
    # 1. Carica dataset
    # ---------------------------------------------------------------
    X_path = os.path.join(DATA_DIR, "X_train.npy")
    S_path = os.path.join(DATA_DIR, "S_train.npy")
    L_path = os.path.join(DATA_DIR, "L_train.npy")
    norm_path = os.path.join(DATA_DIR, "normalization_params.pkl")
    
    X, S, L = load_numpy_triplet(X_path, S_path, L_path)
    print(f"✓ Loaded X: {X.shape}")
    print(f"✓ Loaded S: {S.shape}")
    print(f"✓ Loaded L: {L.shape}, min={L.min()}, max={L.max()}")
    
    # ---------------------------------------------------------------
    # 2. Carica parametri normalizzazione
    # ---------------------------------------------------------------
    dyn_min, dyn_max = load_normalization_params(norm_path)
    print(f"✓ dyn_min: {dyn_min}")
    print(f"✓ dyn_max: {dyn_max}")
    
    # Bounds per la road mask
    bounds = (float(dyn_min[0]), float(dyn_max[0]), 
              float(dyn_min[1]), float(dyn_max[1]))
    print(f"✓ bounds (x,y): {bounds}")
    
    # ---------------------------------------------------------------
    # 3. Costruisci road mask e SDF
    # ---------------------------------------------------------------
    print("\n📍 Costruzione road mask dai dati di training...")
    
    # Denormalizza X per costruire la mask
    X_denorm = denormalize_X(X, dyn_min, dyn_max)
    
    road_mask = build_road_mask_from_data(
        X_denorm, L, bounds,
        resolution=256,    # Risoluzione della griglia
        sigma=1.5,         # Smoothing
        thresh=2,          # Soglia minima di passaggi
        dilate=4           # Dilatazione (più alto = più tolleranza)
    )
    print(f"✓ Road mask shape: {road_mask.shape}")
    print(f"✓ Road coverage: {road_mask.mean()*100:.1f}% dell'area")
    
    # Costruisci SDF
    sdf_tensor = build_sdf_from_mask(road_mask, bounds)
    print(f"✓ SDF tensor shape: {sdf_tensor.shape}")
    print(f"✓ SDF range: [{sdf_tensor.min():.2f}, {sdf_tensor.max():.2f}] metri")
    
    # ---------------------------------------------------------------
    # 4. Crea DataLoader
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
        num_workers=4,
    )
    
    # ---------------------------------------------------------------
    # 5. Costruisci modelli
    # ---------------------------------------------------------------
    print("\n⚙️ Building models...")
    
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
    
    print(f"✓ Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"✓ Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    
    # ---------------------------------------------------------------
    # 6. Callback per sampling
    # ---------------------------------------------------------------
    sample_dir = ensure_dir(os.path.join(OUTPUT_DIR, "samples_with_road"))
    sample_callback = make_sample_callback(
        out_dir=sample_dir,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        dataset=ds,
    )
    
    # ---------------------------------------------------------------
    # 7. TRAINING
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    print(f"   lambda_phys = 1.0 (era 0.1)")
    print(f"   w_road = 10.0 (NUOVO)")
    print(f"   w_start/end = 5.0 (era 30.0)")
    print(f"   w_smooth = 2.0 (era 0.05)")
    print("=" * 60 + "\n")
    
    history, G_final = train_with_road_constraint(
        G=G,
        D=D,
        dataloader=dataloader,
        device=DEVICE,
        latent_dim=LATENT_DIM,
        # NUOVI PARAMETRI ROAD
        sdf_tensor=sdf_tensor,
        bounds=bounds,
        dyn_min=dyn_min,
        dyn_max=dyn_max,
        # Parametri training
        n_critic=N_CRITIC,
        gamma_r1r2=GAMMA_R1R2,
        lambda_phys=1.0,      # AUMENTATO da 0.1
        w_road=10.0,          # Peso penalità off-road
        use_ema=False,
        ema_decay=0.999,
        g_lr=G_LR,
        d_lr=D_LR,
        epochs=EPOCHS,
        out_dir=OUTPUT_DIR,
        sample_callback=sample_callback,
        noise_std=NOISE_STD,
        noise_decay=NOISE_DECAY,
        noise_min=NOISE_MIN,
        r1r2_every=3,
    )
    
    # ---------------------------------------------------------------
    # 8. Salva modello finale
    # ---------------------------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, "transformer_G_final_with_road.pt")
    torch.save(G_final.state_dict(), model_path)
    print(f"\n✅ DONE! Model saved at: {model_path}")
    
    # Salva anche road mask e SDF per uso futuro
    np.save(os.path.join(OUTPUT_DIR, "road_mask.npy"), road_mask)
    torch.save(sdf_tensor, os.path.join(OUTPUT_DIR, "sdf_tensor.pt"))
    print(f"✅ Road mask saved at: {os.path.join(OUTPUT_DIR, 'road_mask.npy')}")


if __name__ == "__main__":
    main()