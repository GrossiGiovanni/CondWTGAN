"""
trainer_improved.py - Enhanced training loop with road geometry constraints

Key changes from original:
1. Pre-builds road SDF from real data (or from SUMO network)
2. Uses improved_physical_loss with road constraint
3. Option for multi-sample selection during validation
4. Better loss weight scheduling
"""

import os
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation

# Import improved losses
from losses import (
    rpgan_d_loss,
    rpgan_g_loss,
    r1_penalty,
    r2_penalty,
    improved_physical_loss,
    build_road_sdf,
    local_coherence_loss,
)


# =============================================================================
#  ROAD MASK CONSTRUCTION (from real trajectories)
# =============================================================================

def build_road_mask_from_trajectories(
    X_train,           # (N, T, 4) training trajectories (denormalized!)
    L_train,           # (N,) sequence lengths
    bounds,            # (minx, maxx, miny, maxy)
    resolution=512,
    smooth_sigma=1.2,
    count_thresh=2,
    dilate_iter=3,     # More dilation = more tolerance
):
    """
    Build road mask from training trajectory distribution.
    
    This is a data-driven approach: where real vehicles drove is "road".
    """
    N, T, _ = X_train.shape
    minx, maxx, miny, maxy = bounds
    
    # Collect all valid (x, y) points
    all_xy = []
    for i in range(N):
        Li = int(L_train[i])
        Li = max(1, min(Li, T))
        all_xy.append(X_train[i, :Li, :2])
    
    all_xy = np.concatenate(all_xy, axis=0)  # (M, 2)
    
    # Build 2D histogram
    H, xedges, yedges = np.histogram2d(
        all_xy[:, 0], all_xy[:, 1],
        bins=resolution,
        range=[[minx, maxx], [miny, maxy]]
    )
    
    # Smooth to fill small gaps
    H_smooth = gaussian_filter(H, sigma=smooth_sigma)
    
    # Threshold to binary mask
    mask = H_smooth >= count_thresh
    
    # Dilate to add tolerance
    if dilate_iter > 0:
        mask = binary_dilation(mask, iterations=dilate_iter)
    
    return mask.astype(bool)


# =============================================================================
#  EMA UPDATE
# =============================================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_((1 - decay) * param.data)


# =============================================================================
#  CHECKPOINT SAVING
# =============================================================================

def save_checkpoint(G, D, G_ema, opt_G, opt_D, step, out_dir, extra_info=None):
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_ema": G_ema.state_dict() if G_ema is not None else None,
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "step": step,
    }
    if extra_info:
        ckpt.update(extra_info)
    
    path = os.path.join(out_dir, f"checkpoint_step_{step}.pt")
    torch.save(ckpt, path)
    return path


# =============================================================================
#  MAIN TRAINING LOOP (IMPROVED)
# =============================================================================

def train_traffic_gan_with_road_constraint(
    G,
    D,
    dataloader,
    *,
    device,
    latent_dim,
    # GAN parameters
    n_critic=3,
    gamma_r1r2=10.0,
    # Physical loss parameters
    lambda_phys=1.0,        # Increased from 0.1!
    # Road constraint parameters
    road_mask=None,         # (H, W) boolean mask, or None to build from data
    sdf_tensor=None,        # Pre-built SDF tensor
    bounds=None,            # (minx, maxx, miny, maxy)
    dyn_min=None,           # (4,) normalization min
    dyn_max=None,           # (4,) normalization max
    # Road constraint weights
    w_road=10.0,            # Road constraint weight
    w_road_start=5.0,       # Start weight (lower for road focus)
    w_road_end=5.0,         # End weight (lower for road focus)
    road_margin=0.5,        # Tolerance in meters
    # Optimizer parameters
    g_lr=2e-4,
    d_lr=1e-4,
    # Training parameters
    epochs=100,
    use_ema=True,
    ema_decay=0.999,
    # Noise injection
    noise_std=0.005,
    noise_decay=0.98,
    noise_min=0.0001,
    noise_on_real=True,
    noise_on_fake=True,
    # Regularization
    r1r2_every=4,
    # Output
    out_dir="outputs",
    sample_callback=None,
    # Advanced options
    use_local_coherence=True,
    lambda_local=0.5,
):
    """
    Enhanced training loop with explicit road geometry constraints.
    
    Key differences from original:
    1. Road SDF loss is computed and applied every step
    2. Physical loss weights favor road-following over endpoint matching
    3. Optional local coherence loss for receding-horizon style training
    """
    
    G = G.to(device)
    D = D.to(device)
    G.train()
    D.train()
    
    # EMA generator
    G_ema = None
    if use_ema:
        G_ema = copy.deepcopy(G).to(device)
        for p in G_ema.parameters():
            p.requires_grad_(False)
    
    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=d_lr, betas=(0.0, 0.9))
    
    # Build SDF if not provided
    if sdf_tensor is None and road_mask is not None and bounds is not None:
        print("Building road SDF from mask...")
        sdf_tensor, pixel_size = build_road_sdf(road_mask, bounds, resolution=road_mask.shape[0])
        print(f"  SDF shape: {sdf_tensor.shape}, pixel size: {pixel_size:.3f}m")
    
    if sdf_tensor is not None:
        sdf_tensor = sdf_tensor.to(device)
    
    # History tracking
    history = {
        "d_loss": [],
        "g_loss": [],
        "r1r2": [],
        "phys_total": [],
        "phys_road": [],
        "phys_start": [],
        "phys_end": [],
        "phys_smooth": [],
        "local_coherence": [],
        "sigma": [],
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
            
            # Noise scheduling
            if noise_decay > 0:
                sigma = max(noise_min, noise_std * (noise_decay ** epoch))
            else:
                sigma = noise_std
            
            # R1/R2 scheduling
            do_r1r2 = r1r2_every > 0 and global_step % r1r2_every == 0
            
            r1 = torch.zeros((), device=device)
            r2 = torch.zeros((), device=device)
            
            # -----------------------------------------------------------------
            #  TRAIN DISCRIMINATOR
            # -----------------------------------------------------------------
            for _ in range(n_critic):
                z = torch.randn(B, latent_dim, device=device)
                fake_X = G(z, S, pad_mask=pad_mask).detach()
                
                # Noise injection
                if sigma > 0:
                    noise_real = sigma * torch.randn_like(real_X) * valid_mask_float if noise_on_real else 0
                    noise_fake = sigma * torch.randn_like(fake_X) * valid_mask_float if noise_on_fake else 0
                    real_in = real_X + noise_real
                    fake_in = fake_X + noise_fake
                else:
                    real_in = real_X
                    fake_in = fake_X
                
                opt_D.zero_grad(set_to_none=True)
                
                with autocast():
                    real_scores = D(real_in, S, pad_mask=pad_mask)
                    fake_scores = D(fake_in, S, pad_mask=pad_mask)
                    loss_D = rpgan_d_loss(real_scores, fake_scores)
                
                # R1/R2 penalties (outside autocast for stability)
                if do_r1r2:
                    r1 = r1_penalty(D, real_X.float(), S.float(), pad_mask=pad_mask)
                    r2 = r2_penalty(D, fake_X.float(), S.float(), pad_mask=pad_mask)
                
                total_D = loss_D + gamma_r1r2 * (r1 + r2)
                
                scaler.scale(total_D).backward()
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                scaler.step(opt_D)
                scaler.update()
            
            # -----------------------------------------------------------------
            #  TRAIN GENERATOR
            # -----------------------------------------------------------------
            z = torch.randn(B, latent_dim, device=device)
            opt_G.zero_grad(set_to_none=True)
            
            with autocast():
                fake_X = G(z, S, pad_mask=pad_mask)
                fake_scores = D(fake_X, S, pad_mask=pad_mask)
                real_scores_ref = D(real_X, S, pad_mask=pad_mask).detach()
                
                g_loss = rpgan_g_loss(real_scores_ref, fake_scores)
                
                # Improved physical loss with road constraint
                phys_total, phys_breakdown = improved_physical_loss(
                    fake_X, S, lengths,
                    sdf_tensor=sdf_tensor,
                    bounds=bounds,
                    dyn_min=dyn_min,
                    dyn_max=dyn_max,
                    w_start=w_road_start,
                    w_end=w_road_end,
                    w_smooth_xy=2.0,
                    w_smooth_speed=1.0,
                    w_speed_range=0.5,
                    w_road=w_road,
                    w_curvature=0.5,
                    road_margin=road_margin,
                )
                
                # Optional local coherence loss
                local_loss = torch.tensor(0.0, device=device)
                if use_local_coherence:
                    local_loss = local_coherence_loss(fake_X, lengths)
                
                total_G = g_loss + lambda_phys * phys_total + lambda_local * local_loss
            
            scaler.scale(total_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt_G)
            scaler.update()
            
            # EMA update
            if use_ema and G_ema is not None:
                update_ema(G_ema, G, ema_decay)
            
            # Logging
            history["d_loss"].append(float(loss_D.item()))
            history["g_loss"].append(float(g_loss.item()))
            history["r1r2"].append(float((r1 + r2).item()))
            history["phys_total"].append(float(phys_total.item()))
            history["phys_road"].append(float(phys_breakdown.get('road', torch.tensor(0.0)).item()))
            history["phys_start"].append(float(phys_breakdown.get('start', torch.tensor(0.0)).item()))
            history["phys_end"].append(float(phys_breakdown.get('end', torch.tensor(0.0)).item()))
            history["phys_smooth"].append(float(phys_breakdown.get('smooth_xy', torch.tensor(0.0)).item()))
            history["local_coherence"].append(float(local_loss.item()))
            history["sigma"].append(float(sigma))
            
            # Sample callback
            if sample_callback and global_step % 1000 == 0:
                G_eval = G_ema if (use_ema and G_ema is not None) else G
                sample_callback(G_eval, epoch, global_step)
            
            # Progress bar
            loop.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{g_loss.item():.3f}",
                "Road": f"{phys_breakdown.get('road', torch.tensor(0.0)).item():.3f}",
                "Phys": f"{phys_total.item():.3f}",
                "σ": f"{sigma:.4f}",
            })
        
        # Save checkpoint each epoch
        save_checkpoint(G, D, G_ema, opt_G, opt_D, global_step, out_dir)
    
    return history, (G_ema if use_ema and G_ema is not None else G)


# =============================================================================
#  MULTI-SAMPLE SELECTION (Inference-time fix)
# =============================================================================

@torch.no_grad()
def generate_with_road_selection(
    G,
    S,                     # (B, 4) conditions
    latent_dim,
    device,
    n_candidates=10,
    sdf_tensor=None,
    bounds=None,
    dyn_min=None,
    dyn_max=None,
    lengths=None,          # (B,) or None for full sequence
):
    """
    Generate multiple candidate trajectories and select the one
    that stays most on-road.
    
    This is your "receding horizon" inference strategy:
    generate multiple futures, pick the most realistic one.
    """
    G.eval()
    B = S.size(0)
    T = 120  # Assuming fixed sequence length
    
    if lengths is None:
        lengths = torch.full((B,), T, device=device, dtype=torch.long)
    
    best_trajs = []
    best_scores = []
    
    for b in range(B):
        s_single = S[b:b+1]  # (1, 4)
        L_single = lengths[b:b+1]
        
        candidates = []
        scores = []
        
        for _ in range(n_candidates):
            z = torch.randn(1, latent_dim, device=device)
            traj = G(z, s_single)  # (1, T, 4)
            candidates.append(traj)
            
            # Score: percentage of points on road
            if sdf_tensor is not None:
                # Denormalize
                xy_norm = traj[0, :, :2]
                min_xy = torch.tensor(dyn_min[:2], device=device)
                max_xy = torch.tensor(dyn_max[:2], device=device)
                xy_world = xy_norm * (max_xy - min_xy) + min_xy
                
                # Sample SDF
                from losses import sample_sdf_bilinear
                sdf_vals = sample_sdf_bilinear(
                    sdf_tensor.unsqueeze(0).unsqueeze(0),
                    xy_world.unsqueeze(0),
                    bounds
                )[0]  # (T,)
                
                # Count on-road points (SDF <= 0)
                Li = int(L_single.item())
                on_road = (sdf_vals[:Li] <= 0.5).float().mean().item()
                scores.append(on_road)
            else:
                scores.append(1.0)  # No SDF, accept all
        
        # Select best
        best_idx = np.argmax(scores)
        best_trajs.append(candidates[best_idx])
        best_scores.append(scores[best_idx])
    
    return torch.cat(best_trajs, dim=0), best_scores
