"""
losses_improved.py - Enhanced loss functions with road geometry constraints

Key improvements:
1. Differentiable road SDF loss
2. Rebalanced physical loss weights
3. Local trajectory coherence loss
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# =============================================================================
#  ROAD SDF (Signed Distance Field) CONSTRUCTION
# =============================================================================

def build_road_sdf(road_mask, bounds, resolution=512):
    """
    Build signed distance field from road mask.
    
    Args:
        road_mask: (H, W) boolean array, True = on road
        bounds: (minx, maxx, miny, maxy) world coordinates
        resolution: mask resolution
        
    Returns:
        sdf_tensor: (H, W) tensor, positive = outside road
        pixel_size: meters per pixel
    """
    # distance_transform_edt gives distance to nearest True pixel
    dist_outside = distance_transform_edt(~road_mask).astype(np.float32)
    dist_inside = distance_transform_edt(road_mask).astype(np.float32)
    sdf = dist_outside - dist_inside  # Positive outside, negative inside
    
    minx, maxx, miny, maxy = bounds
    pixel_size_x = (maxx - minx) / resolution
    pixel_size_y = (maxy - miny) / resolution
    pixel_size = (pixel_size_x + pixel_size_y) / 2
    
    # Convert pixel distances to world distances
    sdf_meters = sdf * pixel_size
    
    return torch.tensor(sdf_meters, dtype=torch.float32), pixel_size


def sample_sdf_bilinear(sdf_tensor, xy_world, bounds):
    """
    Sample SDF at world coordinates using bilinear interpolation.
    
    Args:
        sdf_tensor: (H, W) SDF values
        xy_world: (B, T, 2) world coordinates
        bounds: (minx, maxx, miny, maxy)
        
    Returns:
        (B, T) SDF values at each point
    """
    minx, maxx, miny, maxy = bounds
    B, T, _ = xy_world.shape
    H, W = sdf_tensor.shape
    
    # Normalize to [-1, 1] for grid_sample (note: x maps to width, y to height)
    x_norm = 2 * (xy_world[..., 0] - minx) / (maxx - minx) - 1
    y_norm = 2 * (xy_world[..., 1] - miny) / (maxy - miny) - 1
    
    # grid_sample expects (B, H_out, W_out, 2) with (x, y) in [-1, 1]
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, T, 2)
    grid = grid.unsqueeze(2)  # (B, T, 1, 2)
    
    # SDF needs to be (B, 1, H, W)
    sdf_batched = sdf_tensor.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    sdf_batched = sdf_batched.to(xy_world.device)
    
    # Sample with bilinear interpolation
    sampled = F.grid_sample(
        sdf_batched, 
        grid, 
        mode='bilinear', 
        padding_mode='border',  # Extrapolate outside bounds
        align_corners=True
    )  # (B, 1, T, 1)
    
    return sampled[:, 0, :, 0]  # (B, T)


# =============================================================================
#  ROAD CONSTRAINT LOSS (DIFFERENTIABLE)
# =============================================================================

def road_sdf_loss(
    trajectories,      # (B, T, 4) normalized [0,1]
    lengths,           # (B,) int
    sdf_tensor,        # (H, W) SDF
    bounds,            # (minx, maxx, miny, maxy)
    dyn_min,           # (4,) numpy - min values for denormalization
    dyn_max,           # (4,) numpy - max values for denormalization
    margin=0.5,        # Allow small margin inside road before penalty starts
    exponent=2.0       # Quadratic penalty
):
    """
    Penalize trajectory points outside the drivable road corridor.
    
    This is the KEY NEW LOSS that enforces road-following behavior.
    
    Args:
        trajectories: (B, T, 4) normalized trajectories
        lengths: (B,) sequence lengths
        sdf_tensor: precomputed signed distance field
        bounds: world coordinate bounds
        dyn_min, dyn_max: denormalization parameters
        margin: distance from road edge before penalty kicks in (meters)
        exponent: penalty exponent (2 = quadratic)
        
    Returns:
        Scalar loss value
    """
    B, T, _ = trajectories.shape
    device = trajectories.device
    
    # Denormalize x, y to world coordinates
    xy_norm = trajectories[:, :, :2]  # (B, T, 2)
    
    # Convert to torch tensors for computation
    min_xy = torch.tensor(dyn_min[:2], device=device, dtype=torch.float32)
    max_xy = torch.tensor(dyn_max[:2], device=device, dtype=torch.float32)
    range_xy = max_xy - min_xy
    
    xy_world = xy_norm * range_xy + min_xy  # (B, T, 2)
    
    # Sample SDF at trajectory points
    sdf_values = sample_sdf_bilinear(sdf_tensor, xy_world, bounds)  # (B, T)
    
    # Build validity mask (True = valid timestep, not padding)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid_mask = t_idx < lengths.unsqueeze(1)  # (B, T)
    
    # Penalty: only for positive SDF (outside road) beyond margin
    # sdf_values > margin means point is outside road by more than margin meters
    offroad_dist = torch.relu(sdf_values - margin)  # (B, T)
    
    # Apply penalty function
    if exponent == 2.0:
        penalty = offroad_dist ** 2
    else:
        penalty = offroad_dist ** exponent
    
    # Mask padding and compute mean over valid points
    penalty = penalty * valid_mask.float()
    num_valid = valid_mask.sum().clamp(min=1.0)
    
    return penalty.sum() / num_valid


# =============================================================================
#  IMPROVED PHYSICAL CONSISTENCY LOSS
# =============================================================================

def improved_physical_loss(
    generated_trajectories,  # (B, T, 4)
    s_conditions,            # (B, 4) [x_init, y_init, x_final, y_final]
    lengths,                 # (B,)
    # --- Optional road constraint ---
    sdf_tensor=None,
    bounds=None,
    dyn_min=None,
    dyn_max=None,
    # --- Weight configuration ---
    w_start=5.0,            # Reduced from 30 - let distribution handle this
    w_end=5.0,              # Reduced from 30
    w_smooth_xy=2.0,        # Increased from 0.05 - critical for realism
    w_smooth_speed=1.0,     # Increased from 0.1
    w_speed_range=0.5,      # Kept similar
    w_road=10.0,            # NEW - dominant road constraint
    w_curvature=0.5,        # NEW - penalize sharp turns
    road_margin=0.5,        # Meters of tolerance before penalty
):
    """
    Enhanced physical loss with road constraints and better balance.
    
    The key insight: smoothness and road-following are MORE important
    than exact endpoint matching for realistic trajectory generation.
    """
    B, T, _ = generated_trajectories.shape
    device = generated_trajectories.device
    
    # Build validity masks
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid_t = t_idx < lengths.unsqueeze(1)  # (B, T)
    valid_pairs = valid_t[:, 1:] & valid_t[:, :-1]  # (B, T-1)
    valid_triplets = valid_t[:, 2:] & valid_t[:, 1:-1] & valid_t[:, :-2]  # (B, T-2)
    
    def masked_mean(x, mask):
        mask_f = mask.to(x.dtype)
        return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    
    losses = {}
    
    # ----- 1. Start position loss -----
    start_pos = generated_trajectories[:, 0, :2]  # (B, 2)
    target_start = s_conditions[:, 0:2]
    losses['start'] = w_start * F.mse_loss(start_pos, target_start)
    
    # ----- 2. End position loss -----
    end_idx = (lengths - 1).clamp(min=0).long()
    batch_idx = torch.arange(B, device=device)
    end_pos = generated_trajectories[batch_idx, end_idx, :2]
    target_end = s_conditions[:, 2:4]
    losses['end'] = w_end * F.mse_loss(end_pos, target_end)
    
    # ----- 3. Spatial smoothness (step size regularity) -----
    pos = generated_trajectories[:, :, :2]  # (B, T, 2)
    pos_diff = pos[:, 1:, :] - pos[:, :-1, :]  # (B, T-1, 2)
    step_norm = torch.norm(pos_diff, dim=-1)  # (B, T-1)
    
    # Penalize both large jumps AND sudden changes in step size
    step_variance = (step_norm[:, 1:] - step_norm[:, :-1]) ** 2  # (B, T-2)
    
    losses['smooth_xy'] = w_smooth_xy * masked_mean(step_norm, valid_pairs)
    
    # ----- 4. Speed smoothness -----
    speeds = generated_trajectories[:, :, 2]  # (B, T)
    speed_diff = torch.abs(speeds[:, 1:] - speeds[:, :-1])  # (B, T-1)
    losses['smooth_speed'] = w_smooth_speed * masked_mean(speed_diff, valid_pairs)
    
    # ----- 5. Speed range constraint -----
    speed_min_penalty = torch.relu(0.02 - speeds)  # Too slow
    speed_max_penalty = torch.relu(speeds - 0.98)  # Too fast
    losses['speed_range'] = w_speed_range * masked_mean(
        speed_min_penalty + speed_max_penalty, valid_t
    )
    
    # ----- 6. Curvature smoothness (penalize sharp turns) -----
    if valid_triplets.any() and w_curvature > 0:
        # Compute direction changes
        directions = pos_diff / (torch.norm(pos_diff, dim=-1, keepdim=True) + 1e-8)
        direction_change = directions[:, 1:, :] - directions[:, :-1, :]  # (B, T-2, 2)
        curvature = torch.norm(direction_change, dim=-1)  # (B, T-2)
        losses['curvature'] = w_curvature * masked_mean(curvature, valid_triplets)
    else:
        losses['curvature'] = torch.tensor(0.0, device=device)
    
    # ----- 7. Road SDF loss (if provided) -----
    if sdf_tensor is not None and bounds is not None:
        losses['road'] = w_road * road_sdf_loss(
            generated_trajectories,
            lengths,
            sdf_tensor,
            bounds,
            dyn_min,
            dyn_max,
            margin=road_margin
        )
    else:
        losses['road'] = torch.tensor(0.0, device=device)
    
    # Combine all losses
    total = sum(losses.values())
    
    return total, losses


# =============================================================================
#  LOCAL COHERENCE LOSS (receding horizon perspective)
# =============================================================================

def local_coherence_loss(
    trajectories,      # (B, T, 4)
    lengths,           # (B,)
    window_size=10,    # Look-ahead/behind window
):
    """
    Penalize local trajectory inconsistencies - each segment should be
    locally smooth and coherent, regardless of global endpoint.
    
    This implements your "receding horizon" idea: focus on local realism.
    """
    B, T, _ = trajectories.shape
    device = trajectories.device
    
    pos = trajectories[:, :, :2]  # (B, T, 2)
    
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid = t_idx < lengths.unsqueeze(1)
    
    total_loss = torch.tensor(0.0, device=device)
    
    # For each window, compute local direction consistency
    for start in range(0, T - window_size, window_size // 2):
        end = min(start + window_size, T)
        
        window_pos = pos[:, start:end, :]  # (B, W, 2)
        window_valid = valid[:, start:end]  # (B, W)
        
        if window_pos.size(1) < 3:
            continue
            
        # Local displacement vectors
        disp = window_pos[:, 1:, :] - window_pos[:, :-1, :]  # (B, W-1, 2)
        
        # Direction consistency within window
        disp_norm = disp / (torch.norm(disp, dim=-1, keepdim=True) + 1e-8)
        
        # Consecutive direction similarity (dot product)
        dir_sim = (disp_norm[:, 1:, :] * disp_norm[:, :-1, :]).sum(dim=-1)  # (B, W-2)
        
        # Penalize direction changes (1 - cos_sim)
        dir_change_penalty = 1 - dir_sim  # 0 = same direction, 2 = opposite
        
        # Mask and accumulate
        pair_valid = window_valid[:, 2:end-start] & window_valid[:, 1:end-start-1]
        if pair_valid.any():
            total_loss = total_loss + (dir_change_penalty * pair_valid.float()).sum() / pair_valid.sum().clamp(min=1)
    
    return total_loss


# =============================================================================
#  ORIGINAL LOSSES (kept for reference)
# =============================================================================

def wasserstein_d_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()

def wasserstein_g_loss(fake_scores):
    return -fake_scores.mean()

def rpgan_d_loss(real_scores, fake_scores):
    real_scores = real_scores.view(-1)
    fake_scores = fake_scores.view(-1)
    return F.softplus(-(real_scores - fake_scores)).mean()

def rpgan_g_loss(real_scores, fake_scores):
    real_scores = real_scores.view(-1)
    fake_scores = fake_scores.view(-1)
    return F.softplus(real_scores - fake_scores).mean()

def robust_gradient_penalty(D, real_data, fake_data, s_conditions, device, lambda_gp=15.0, pad_mask=None):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = alpha * real_data + (1.0 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    critic_interpolated = D(interpolated, s_conditions, pad_mask=pad_mask)
    grad_outputs = torch.ones_like(critic_interpolated, device=device)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
    return lambda_gp * ((gradient_norm - 1.0) ** 2).mean()

def r1_penalty(D, real_X, S, pad_mask=None):
    x = real_X.detach().requires_grad_(True)
    scores = D(x, S, pad_mask=pad_mask).view(-1)
    
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    if pad_mask is not None:
        grad = grad.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        valid_t = (~pad_mask).sum(dim=1).clamp(min=1).float()
        feat_dim = grad.size(-1)
        denom = valid_t * feat_dim
        grad2 = (grad ** 2).sum(dim=(1, 2)) / denom
    else:
        grad2 = (grad ** 2).sum(dim=(1, 2))
    
    return 0.5 * grad2.mean()

def r2_penalty(D, fake_X, S, pad_mask=None):
    x = fake_X.detach().requires_grad_(True)
    scores = D(x, S, pad_mask=pad_mask).view(-1)
    
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    if pad_mask is not None:
        grad = grad.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        valid_t = (~pad_mask).sum(dim=1).clamp(min=1).float()
        feat_dim = grad.size(-1)
        denom = valid_t * feat_dim
        grad2 = (grad ** 2).sum(dim=(1, 2)) / denom
    else:
        grad2 = (grad ** 2).sum(dim=(1, 2))
    
    return 0.5 * grad2.mean()