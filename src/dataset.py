import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# -------------------------------------------------------------
#  DATASET CLASSE PRINCIPALE (NUOVA STRUTTURA)
# -------------------------------------------------------------

class TrafficDataset(Dataset):
    
    """
    Dataset per traffico (solo geometria + cinematica).
    
    X = sequenze dinamiche normalizzate (B, 120, 4)
        [x, y, speed, angle]
        
    S = condizioni statiche normalizzate (B, 4)
        [x_init, y_init, x_final, y_final]
    """

    def __init__(self, X, S):
        """
        X: numpy array (B, 120, 4)
        S: numpy array (B, 4)
        """
        assert len(X) == len(S), "❌ Mismatch tra X e S"

        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]


# -------------------------------------------------------------
#  GENERATORE BASE DI DATALOADER
# -------------------------------------------------------------

def create_dataloader(X, S, batch_size=32, shuffle=True, num_workers=2):
    """
    Ritorna un DataLoader stabile per GAN.
    """
    dataset = TrafficDataset(X, S)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True   # importante per stabilità GAN
    )


# -------------------------------------------------------------
#  CREAZIONE TRAIN / VALIDATION SPLIT
# -------------------------------------------------------------

def create_train_val_split(X, S, val_ratio=0.1, seed=42):
    """
    Split pulito train / validation.
    """
    np.random.seed(seed)

    N = len(X)
    indices = np.random.permutation(N)

    n_val = int(N * val_ratio)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        X[train_idx], S[train_idx],
        X[val_idx], S[val_idx]
    )