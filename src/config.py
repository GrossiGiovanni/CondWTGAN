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
#  DATA SHAPES (NUOVO DATASET)
# ============================================================

SEQ_LEN = 120              # 60s / 0.5s
FEATURE_DIM = 4            # [x, y, speed, angle]
COND_DIM = 4               # [x_init, y_init, x_final, y_final]

# ============================================================
#  TRANSFORMER MODEL DIMENSIONS
# ============================================================

LATENT_DIM = 128
D_MODEL = 256               # aumentata perché 128 è un po’ stretta per 120 step
FF_DIM = 512
N_HEADS = 4

N_LAYERS_G = 4
N_LAYERS_D = 3

# ============================================================
#  TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 50

N_CRITIC = 5
LAMBDA_GP = 20.0

G_LR = 2e-4
D_LR = 1.5e-4

USE_EMA = False
EMA_DECAY = 0.995
