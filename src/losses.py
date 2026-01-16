import torch
import torch.nn.functional as F


# -------------------------------------------------------------
#  WASSERSTEIN LOSSES (INVARIATE)
# -------------------------------------------------------------

def wasserstein_d_loss(real_scores, fake_scores):
    """
    Critic loss per WGAN:
      maximize D(real) - D(fake)
    (ritorniamo il negativo perché PyTorch minimizza)
    """
    return fake_scores.mean() - real_scores.mean()


def wasserstein_g_loss(fake_scores):
    """Il generatore vuole massimizzare D(fake)."""
    return -fake_scores.mean()


# -------------------------------------------------------------
#  GRADIENT PENALTY (COERENTE CON TRAIETTORIE 4D)
# -------------------------------------------------------------

def robust_gradient_penalty(
    D,
    real_data,
    fake_data,
    s_conditions,
    device,
    lambda_gp=15.0,
    pad_mask=None
):
    """
    real_data, fake_data: (B,T,4)
    s_conditions: (B,4)
    pad_mask: (B,T) True=padding
    """
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


# -------------------------------------------------------------
#  PHYSICAL CONSISTENCY LOSS (NUOVA VERSIONE)
# -------------------------------------------------------------

def conservative_physical_loss(generated_trajectories, s_conditions, lengths):
    """
    generated_trajectories: (B,T,4)
    s_conditions: (B,4) [x_init,y_init,x_final,y_final]
    lengths: (B,) timestep reali (1..T)
    """
    B, T, _ = generated_trajectories.shape
    device = generated_trajectories.device

    # mask valid: True dove t < L
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid_t = t_idx < lengths.unsqueeze(1)         # (B,T) True=valido
    valid_pairs = valid_t[:, 1:] & valid_t[:, :-1] # (B,T-1)

    def masked_mean(x, mask):
        mask_f = mask.to(x.dtype)
        return (x * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    # 1) start (t=0)
    start_pos = generated_trajectories[:, 0, :2]
    target_start = s_conditions[:, 0:2]
    start_loss = F.mse_loss(start_pos, target_start)

    # 2) end (t = L-1, non -1!)
    end_idx = (lengths - 1).clamp(min=0)  # (B,)
    end_pos = generated_trajectories[torch.arange(B, device=device), end_idx, :2]
    target_end = s_conditions[:, 2:4]
    end_loss = F.mse_loss(end_pos, target_end)

    # 3) smoothness x,y solo su coppie valide
    pos = generated_trajectories[:, :, :2]           # (B,T,2)
    pos_diff = pos[:, 1:, :] - pos[:, :-1, :]        # (B,T-1,2)
    step_norm = torch.norm(pos_diff, dim=-1)         # (B,T-1)
    smoothness_loss = masked_mean(step_norm, valid_pairs)

    # 4) smoothness speed solo su coppie valide
    speeds = generated_trajectories[:, :, 2]         # (B,T)
    speed_diff = speeds[:, 1:] - speeds[:, :-1]      # (B,T-1)
    speed_smoothness = masked_mean(torch.abs(speed_diff), valid_pairs)

    # 5) speed range solo su timestep validi
    speed_min_penalty = torch.relu(0.02 - speeds)    # (B,T)
    speed_max_penalty = torch.relu(speeds - 0.98)    # (B,T)
    speed_range_loss = masked_mean(speed_min_penalty + speed_max_penalty, valid_t)

    total_phys = (
        2.0 * start_loss +
        2.0 * end_loss +
        0.1 * smoothness_loss +
        0.1 * speed_smoothness +
        0.2 * speed_range_loss
    )

    return total_phys

# -------------------------------------------------------------
#  RELATIVISTIC PAIRING GAN (RpGAN) + R1/R2 (0-GP)
# -------------------------------------------------------------
# RpGAN: L = E_{z,x}[ f(D(G(z)) - D(x)) ]  (Eq. 2 del paper)
# con f(t) = softplus(-t) => -log(sigmoid(t)) (logistic)
# R1/R2: gamma/2 * E[ ||∇_x D(x)||^2 ] su real e fake (Eq. 3)

def rpgan_d_loss(real_scores, fake_scores):
    """
    Discriminator (critic) objective for RpGAN logistic:
      maximize log(sigmoid(D(real)-D(fake)))
    In minimization form:
      softplus(-(D(real)-D(fake)))
    """
    # real_scores, fake_scores: (B,) oppure (B,1)
    real_scores = real_scores.view(-1)
    fake_scores = fake_scores.view(-1)
    return F.softplus(-(real_scores - fake_scores)).mean()


def rpgan_g_loss(real_scores, fake_scores):
    """
    Generator objective: flip the ordering so that D(fake) > D(real).
    Minimization form:
      softplus(D(real)-D(fake))
    """
    real_scores = real_scores.view(-1)
    fake_scores = fake_scores.view(-1)
    return F.softplus(real_scores - fake_scores).mean()


def _masked_grad_norm2(grad, pad_mask=None):
    """
    grad: (B,T,F)
    pad_mask: (B,T) True=padding
    ritorna: norma^2 per sample, normalizzata sui timestep validi (se mask presente)
    """
    if pad_mask is None:
        return (grad ** 2).sum(dim=(1, 2))

    # azzera gradienti sui timestep padding
    grad = grad.masked_fill(pad_mask.unsqueeze(-1), 0.0)

    # normalizza per #valid * F (evita che sequenze più lunghe pesino di più)
    valid_t = (~pad_mask).sum(dim=1).clamp(min=1).float()  # (B,)
    feat_dim = grad.size(-1)
    denom = valid_t * feat_dim
    return (grad ** 2).sum(dim=(1, 2)) / denom


def r1_penalty(D, real_X, S, pad_mask=None):
    """
    R1 = (gamma/2) * E_x~pD [ ||∇_x D(x)||^2 ]
    Qui ritorniamo solo: (1/2)*E[||∇||^2]
    (così nel trainer fai: gamma * (r1 + r2))
    """
    x = real_X.detach().requires_grad_(True)
    scores = D(x, S, pad_mask=pad_mask).view(-1)

    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B,T,F)

    grad2 = _masked_grad_norm2(grad, pad_mask=pad_mask)  # (B,)
    return 0.5 * grad2.mean()


def r2_penalty(D, fake_X, S, pad_mask=None):
    """
    R2 = (gamma/2) * E_x~pθ [ ||∇_x D(x)||^2 ]
    Ritorniamo (1/2)*E[||∇||^2]
    """
    x = fake_X.detach().requires_grad_(True)
    scores = D(x, S, pad_mask=pad_mask).view(-1)

    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad2 = _masked_grad_norm2(grad, pad_mask=pad_mask)
    return 0.5 * grad2.mean()
