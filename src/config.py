"""
IMPROVED CONFIG.PY FOR CONDWTGAN
=================================

Key changes:
1. Higher physical loss weight (lambda_phys)
2. Added boundary loss parameters
3. Road penalty configuration
4. Curriculum learning settings
5. Better optimizer parameters
"""

import os
import torch

# ============================================================
#  PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
NORM_DIR = os.path.join(ROOT, "data_norm")

# Ensure dirs exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(NORM_DIR, exist_ok=True)

# ============================================================
#  DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#  DATA SHAPES
# ============================================================

SEQ_LEN = 120              # 60s / 0.5s
FEATURE_DIM = 4            # [x, y, speed, angle]
COND_DIM = 4               # [x_init, y_init, x_final, y_final]

# ============================================================
#  TRANSFORMER MODEL DIMENSIONS
# ============================================================

LATENT_DIM = 128
D_MODEL = 256
FF_DIM = 1024
N_HEADS = 4

N_LAYERS_G = 6
N_LAYERS_D = 2

# ============================================================
#  TRAINING HYPERPARAMETERS - IMPROVED
# ============================================================

BATCH_SIZE = 128
EPOCHS = 10

# Critic iterations per generator step
N_CRITIC = 1

# Learning rates (slightly lower for stability)
G_LR = 3e-4          # was 3e-4
D_LR = 1e-5         # was 1e-4

# ============================================================
#  LOSS WEIGHTS - CRITICAL CHANGES
# ============================================================

# Gradient penalty (for WGAN-GP, not used with RpGAN+R1R2)
LAMBDA_GP = 20.0

# R1/R2 regularization
GAMMA_R1R2 = 20.0    # was 15.0, slightly reduced
R1R2_EVERY = 2  # Compute R1/R2 every N steps (was 3)

# ============================================================
#  PHYSICAL / BOUNDARY LOSS WEIGHTS - KEY IMPROVEMENTS
# ============================================================

# Main physical loss multiplier
LAMBDA_PHYS = 3.0    # <<< INCREASED from 0.1 (10x increase!)

# Boundary loss configuration
LAMBDA_BOUNDARY = 2.0        # Additional boundary loss weight
BOUNDARY_BASE_WEIGHT = 30.0  # Base weight for boundary MSE
BOUNDARY_MAX_WEIGHT = 100.0  # Max weight at end of training (progressive)
USE_HUBER_BOUNDARY = True    # Use Huber loss (more robust to outliers)
HUBER_DELTA = 0.05           # Huber loss delta parameter

# Individual physical loss components (inside conservative_physical_loss)
PHYS_W_START = 50.0          # was 30.0 - Start position weight
PHYS_W_END = 50.0            # was 30.0 - End position weight
PHYS_W_SMOOTHNESS = 0.05     # Position smoothness
PHYS_W_SPEED_SMOOTH = 0.1    # Speed smoothness
PHYS_W_SPEED_RANGE = 0.2     # Speed bounds penalty

# ============================================================
#  ROAD PENALTY (NEW)
# ============================================================

USE_ROAD_PENALTY = True      # Enable road mask penalty during training
LAMBDA_ROAD = 50.0           # Road penalty weight
ROAD_MASK_RESOLUTION = 256   # Resolution of road mask
ROAD_MASK_SIGMA = 1.5        # Gaussian smoothing for road mask
ROAD_MASK_THRESHOLD = 0.03  # Density threshold for "on road"

# ============================================================
#  GENERATOR CONSTRAINTS (NEW)
# ============================================================

# Hard constraint: force output[0,:2] = condition start
HARD_START_CONSTRAINT = True

# Soft end blending: interpolate trajectory to reach endpoint
SOFT_END_BLEND = False

# ============================================================
#  EMA (Exponential Moving Average)
# ============================================================

USE_EMA = True               # Enable EMA (recommended)
EMA_DECAY = 0.999

# ============================================================
#  NOISE INJECTION (for discriminator inputs)
# ============================================================

# Previous values were too small to matter (0.0005)
# Either use meaningful values or disable completely
USE_NOISE_INJECTION = False  # Disabled - was not helping
NOISE_STD = 0.02            # If enabled, use this
NOISE_DECAY = 0.99
NOISE_MIN = 0.005

# ============================================================
#  CURRICULUM LEARNING (NEW - OPTIONAL)
# ============================================================

USE_CURRICULUM = False       # Start with shorter sequences
CURRICULUM_START_LEN = 30    # Initial max sequence length
CURRICULUM_END_EPOCH = 50    # Epoch at which full length is used

# ============================================================
#  LOGGING / CHECKPOINTING
# ============================================================

SAMPLE_EVERY = 500           # Generate samples every N steps
CHECKPOINT_EVERY = 2        # Save checkpoint every N epochs
LOG_EVERY = 100              # Print detailed logs every N steps

# ============================================================
#  VALIDATION
# ============================================================

VAL_RATIO = 0.1              # Fraction of data for validation
VAL_EVERY = 5                # Validate every N epochs