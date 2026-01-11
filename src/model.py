"""
model_improved.py - Enhanced Generator with stronger conditioning

Key improvements:
1. Cross-attention conditioning (instead of simple additive)
2. Residual trajectory formulation (predict deviations from baseline)
3. Hard start constraint enforcement
4. Option for iterative refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
#  POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# =============================================================================
#  CROSS-ATTENTION CONDITIONING MODULE
# =============================================================================

class CrossAttentionConditioner(nn.Module):
    """
    Condition trajectory generation via cross-attention to start/end points.
    
    This is MUCH stronger than additive conditioning because:
    - Each timestep can "look at" the condition explicitly
    - Attention weights show which parts of the trajectory attend to which endpoints
    - Gradients flow directly through attention to condition embedding
    """
    
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        
        # Project 4D condition to d_model
        self.cond_proj = nn.Linear(4, d_model)
        
        # Expand condition to multiple "tokens" for richer conditioning
        # Token 0: start position
        # Token 1: end position
        # Token 2: displacement vector
        self.start_embed = nn.Linear(2, d_model)
        self.end_embed = nn.Linear(2, d_model)
        self.displacement_embed = nn.Linear(2, d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, cond):
        """
        x: (B, T, d_model) trajectory representation
        cond: (B, 4) [x_start, y_start, x_end, y_end]
        
        Returns: (B, T, d_model) conditioned representation
        """
        B = x.size(0)
        
        # Build condition key-value tokens
        start_kv = self.start_embed(cond[:, :2]).unsqueeze(1)  # (B, 1, d_model)
        end_kv = self.end_embed(cond[:, 2:4]).unsqueeze(1)    # (B, 1, d_model)
        
        displacement = cond[:, 2:4] - cond[:, :2]  # Direction from start to end
        disp_kv = self.displacement_embed(displacement).unsqueeze(1)  # (B, 1, d_model)
        
        cond_kv = torch.cat([start_kv, end_kv, disp_kv], dim=1)  # (B, 3, d_model)
        
        # Cross-attention: trajectory queries attend to condition keys
        attended, _ = self.cross_attn(
            query=x,
            key=cond_kv,
            value=cond_kv
        )
        
        # Residual + norm
        return self.norm(x + attended)


# =============================================================================
#  IMPROVED GENERATOR (with residual formulation)
# =============================================================================

class TransformerGenerator(nn.Module):
    """
    Improved generator with:
    1. Cross-attention conditioning at every layer
    2. Residual trajectory formulation (predict delta from baseline)
    3. Hard start constraint
    4. Separate noise injection per timestep
    """
    
    def __init__(
        self,
        latent_dim=128,
        cond_dim=4,
        seq_len=120,
        d_model=256,
        nhead=4,
        num_layers=5,
        ff_dim=1024,
        dropout=0.1,
        use_residual_trajectory=True,  # Predict deviations from linear baseline
        enforce_start=True,            # Hard constraint on start position
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_residual_trajectory = use_residual_trajectory
        self.enforce_start = enforce_start
        
        # Input projection: latent + condition
        self.input_proj = nn.Linear(latent_dim + cond_dim, d_model)
        
        # Per-timestep noise injection (more diversity than shared noise)
        self.timestep_noise = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer layers with cross-attention conditioning
        self.layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # Self-attention + feedforward
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.layers.append(encoder_layer)
            
            # Cross-attention to condition
            self.cond_layers.append(CrossAttentionConditioner(d_model, nhead, dropout))
        
        # Causal mask for autoregressive-style generation
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        
        # Output projection: predict (x, y, speed, angle) or residuals
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4)
        )
        
        # For residual formulation: baseline trajectory predictor
        if use_residual_trajectory:
            # Simple linear interpolation baseline: weight per timestep
            t_weights = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
            self.register_buffer("t_weights", t_weights)  # (1, T, 1)
    
    def _compute_baseline(self, cond):
        """
        Compute baseline trajectory: linear interpolation from start to end.
        
        cond: (B, 4) [x_start, y_start, x_end, y_end]
        Returns: (B, T, 2) baseline xy positions
        """
        start_xy = cond[:, :2].unsqueeze(1)  # (B, 1, 2)
        end_xy = cond[:, 2:4].unsqueeze(1)   # (B, 1, 2)
        
        # Linear interpolation
        baseline = start_xy + self.t_weights * (end_xy - start_xy)  # (B, T, 2)
        return baseline
    
    def forward(self, z, cond, pad_mask=None):
        """
        z: (B, latent_dim)
        cond: (B, 4)
        pad_mask: (B, T) boolean, True = padding
        
        Returns: (B, T, 4) generated trajectory
        """
        B = z.size(0)
        
        # Initial token embedding
        token = torch.cat([z, cond], dim=1)  # (B, latent + 4)
        x = self.input_proj(token)           # (B, d_model)
        
        # Expand to sequence with per-timestep variation
        x = x.unsqueeze(1).expand(B, self.seq_len, self.d_model)
        x = x + self.timestep_noise.unsqueeze(0)  # Add learned per-timestep noise
        x = self.pos_encoder(x)
        
        # Process through transformer layers
        for layer, cond_layer in zip(self.layers, self.cond_layers):
            # Self-attention with causal mask
            x = layer(x, src_mask=self.causal_mask, src_key_padding_mask=pad_mask)
            
            # Cross-attention to condition
            x = cond_layer(x, cond)
        
        # Output projection
        output = self.output_fc(x)  # (B, T, 4)
        
        # Apply residual formulation for xy if enabled
        if self.use_residual_trajectory:
            baseline = self._compute_baseline(cond)  # (B, T, 2)
            
            # Output[:, :, :2] is DELTA from baseline
            # Scale residuals to prevent huge deviations
            xy_delta = torch.tanh(output[:, :, :2]) * 0.3  # Max ~0.3 deviation
            output = torch.cat([
                baseline + xy_delta,  # Actual xy = baseline + delta
                output[:, :, 2:4]     # Speed and angle direct
            ], dim=-1)
        
        # Enforce hard start constraint
        if self.enforce_start:
            output = output.clone()
            output[:, 0, :2] = cond[:, :2]  # First position = condition start
        
        return output


# =============================================================================
#  IMPROVED DISCRIMINATOR (with trajectory encoding)
# =============================================================================

class TransformerDiscriminator(nn.Module):
    """
    Improved discriminator with:
    1. Cross-attention conditioning
    2. Trajectory-aware features (velocity, curvature)
    3. Multi-scale discrimination
    """
    
    def __init__(
        self,
        cond_dim=4,
        seq_len=120,
        d_model=256,
        nhead=4,
        num_layers=3,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        
        # Input: 4 trajectory features + derived features
        # Derived: velocity magnitude, direction change, etc.
        self.input_fc = nn.Linear(4 + 4, d_model)  # +4 for derived features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Condition cross-attention
        self.cond_cross_attn = CrossAttentionConditioner(d_model, nhead, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale pooling
        self.pool_1 = nn.AdaptiveAvgPool1d(1)    # Global
        self.pool_4 = nn.AdaptiveAvgPool1d(4)    # 4 segments
        self.pool_8 = nn.AdaptiveAvgPool1d(8)    # 8 segments
        
        # Final classifier
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * (1 + 4 + 8), d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
    
    def _compute_derived_features(self, traj, pad_mask=None):
        """
        Compute additional trajectory features for better discrimination.
        
        traj: (B, T, 4) [x, y, speed, angle]
        Returns: (B, T, 4) [vel_mag, vel_dir, accel, curvature]
        """
        B, T, _ = traj.shape
        device = traj.device
        
        pos = traj[:, :, :2]  # (B, T, 2)
        
        # Velocity vectors
        vel = torch.zeros_like(pos)
        vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
        
        # Velocity magnitude
        vel_mag = torch.norm(vel, dim=-1, keepdim=True)  # (B, T, 1)
        
        # Velocity direction (angle)
        vel_dir = torch.atan2(vel[:, :, 1:2], vel[:, :, 0:1] + 1e-8)  # (B, T, 1)
        
        # Acceleration (change in velocity magnitude)
        accel = torch.zeros(B, T, 1, device=device)
        accel[:, 1:, 0] = vel_mag[:, 1:, 0] - vel_mag[:, :-1, 0]
        
        # Curvature (change in direction)
        curv = torch.zeros(B, T, 1, device=device)
        curv[:, 1:, 0] = vel_dir[:, 1:, 0] - vel_dir[:, :-1, 0]
        # Normalize angle differences to [-pi, pi]
        curv = torch.atan2(torch.sin(curv), torch.cos(curv))
        
        derived = torch.cat([vel_mag, vel_dir, accel, curv], dim=-1)  # (B, T, 4)
        
        return derived
    
    def forward(self, traj, cond, pad_mask=None):
        """
        traj: (B, T, 4)
        cond: (B, 4)
        pad_mask: (B, T) boolean, True = padding
        
        Returns: (B, 1) critic score
        """
        B = traj.size(0)
        
        # Compute derived features
        derived = self._compute_derived_features(traj, pad_mask)
        
        # Concatenate original + derived
        traj_aug = torch.cat([traj, derived], dim=-1)  # (B, T, 8)
        
        # Project to d_model
        x = self.input_fc(traj_aug)  # (B, T, d_model)
        x = self.pos_encoder(x)
        
        # Cross-attention to condition
        x = self.cond_cross_attn(x, cond)
        
        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        # Multi-scale pooling
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        
        p1 = self.pool_1(x_t).squeeze(-1)                    # (B, d_model)
        p4 = self.pool_4(x_t).view(B, -1)                    # (B, d_model * 4)
        p8 = self.pool_8(x_t).view(B, -1)                    # (B, d_model * 8)
        
        pooled = torch.cat([p1, p4, p8], dim=-1)  # (B, d_model * 13)
        
        return self.fc_out(pooled)


# =============================================================================
#  ITERATIVE REFINEMENT GENERATOR (Diffusion-style)
# =============================================================================

class RefinementGenerator(nn.Module):
    """
    Alternative architecture: iteratively refine trajectory.
    
    Start with rough trajectory, refine it over N steps.
    Each step predicts residuals to improve the trajectory.
    
    This can help with:
    - Correcting off-road points
    - Smoothing inconsistencies
    - Better endpoint matching
    """
    
    def __init__(
        self,
        latent_dim=128,
        cond_dim=4,
        seq_len=120,
        d_model=256,
        nhead=4,
        num_refine_steps=3,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_refine_steps = num_refine_steps
        
        # Initial trajectory generator (simple)
        self.init_generator = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len * 4),
        )
        
        # Refinement network (shared across steps)
        self.refine_encoder = nn.Linear(4 + cond_dim, d_model)  # Current traj + condition
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )
        self.refine_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output residual
        self.refine_output = nn.Linear(d_model, 4)
        
        # Step embedding (so network knows which refinement step)
        self.step_embed = nn.Embedding(num_refine_steps, d_model)
    
    def forward(self, z, cond, pad_mask=None):
        B = z.size(0)
        device = z.device
        
        # Initial trajectory
        init_input = torch.cat([z, cond], dim=1)
        traj = self.init_generator(init_input).view(B, self.seq_len, 4)
        
        # Iterative refinement
        for step in range(self.num_refine_steps):
            # Encode current trajectory + condition
            cond_expanded = cond.unsqueeze(1).expand(B, self.seq_len, -1)
            refine_input = torch.cat([traj, cond_expanded], dim=-1)  # (B, T, 8)
            
            x = self.refine_encoder(refine_input)
            x = self.pos_encoder(x)
            
            # Add step embedding
            step_emb = self.step_embed(torch.tensor(step, device=device))
            x = x + step_emb.unsqueeze(0).unsqueeze(0)
            
            # Transformer refinement
            x = self.refine_transformer(x, src_key_padding_mask=pad_mask)
            
            # Predict residual
            residual = self.refine_output(x)  # (B, T, 4)
            
            # Apply residual with decreasing weight
            alpha = 1.0 / (step + 1)  # Decreasing step size
            traj = traj + alpha * residual
        
        # Enforce start constraint
        traj = traj.clone()
        traj[:, 0, :2] = cond[:, :2]
        
        return traj


# =============================================================================
#  FACTORY FUNCTION
# =============================================================================

def create_generator(
    model_type="improved",  # "improved", "refinement", "original"
    **kwargs
):
    """
    Factory function to create generator.
    """
    if model_type == "improved":
        return TransformerGenerator(**kwargs)
    elif model_type == "refinement":
        return RefinementGenerator(**kwargs)
    else:
        from model import TransformerGenerator
        return TransformerGenerator(**kwargs)


def create_discriminator(
    model_type="improved",
    **kwargs
):
    if model_type == "improved":
        return TransformerDiscriminator(**kwargs)
    else:
        from model import TransformerDiscriminator
        return TransformerDiscriminator(**kwargs)