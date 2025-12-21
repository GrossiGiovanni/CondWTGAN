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
    lambda_gp=15.0
):
    """
    WGAN-GP robusta per traiettorie:
    real_data, fake_data: (B,120,4)
    s_conditions: (B,4)
    """

    batch_size = real_data.size(0)

    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = alpha * real_data + (1.0 - alpha) * fake_data
    interpolated.requires_grad_(True)

    critic_interpolated = D(interpolated, s_conditions)

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

def conservative_physical_loss(generated_trajectories, s_conditions):
    """
    Vincoli fisici SOFT coerenti con il nuovo dataset.

    generated_trajectories: (B,120,4)
        [:,:,0] = x
        [:,:,1] = y
        [:,:,2] = speed   (normalizzata 0–1)
        [:,:,3] = angle   (normalizzato 0–1)

    s_conditions: (B,4)
        [x_init, y_init, x_final, y_final] (normalizzati 0–1)

    Vincoli applicati:
    1) Traiettoria deve partire da (x_init, y_init)
    2) Traiettoria deve arrivare a (x_final, y_final)
    3) Continuità spaziale (smoothness)
    4) Continuità della velocità
    5) Penalizzazione velocità estreme (0 o 1)
    """

    # -----------------------------
    # 1) VINCOLO POSIZIONE INIZIALE
    # -----------------------------
    start_pos = generated_trajectories[:, 0, :2]        # (B,2)
    target_start = s_conditions[:, 0:2]                 # (B,2)

    start_loss = F.mse_loss(start_pos, target_start)

    # -----------------------------
    # 2) VINCOLO POSIZIONE FINALE
    # -----------------------------
    end_pos = generated_trajectories[:, -1, :2]         # (B,2)
    target_end = s_conditions[:, 2:4]                   # (B,2)

    end_loss = F.mse_loss(end_pos, target_end)

    # -----------------------------
    # 3) SMOOTHNESS SPAZIALE (x,y)
    # -----------------------------
    pos = generated_trajectories[:, :, :2]
    pos_diff = torch.diff(pos, dim=1)
    smoothness_loss = torch.norm(pos_diff, dim=-1).mean()

    # -----------------------------
    # 4) CONTINUITÀ DELLA VELOCITÀ
    # -----------------------------
    speeds = generated_trajectories[:, :, 2]
    speed_diff = torch.diff(speeds, dim=1)
    speed_smoothness = torch.abs(speed_diff).mean()

    # -----------------------------
    # 5) VELOCITÀ FISICAMENTE SENSATA
    # (evita speed ~0 o ~1 costanti)
    # -----------------------------
    speed_min_penalty = torch.relu(0.02 - speeds).mean()
    speed_max_penalty = torch.relu(speeds - 0.98).mean()
    speed_range_loss = speed_min_penalty + speed_max_penalty

    # -----------------------------
    #  LOSS FINALE PESATA
    # -----------------------------
    total_phys = (
        2.0 * start_loss +          # vincolo inizio forte
        2.0 * end_loss +            # vincolo fine forte
        0.1 * smoothness_loss +     # regolarità traiettoria
        0.1 * speed_smoothness +    # regolarità velocità
        0.2 * speed_range_loss      # velocità sensata
    )

    return total_phys
