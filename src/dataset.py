"""
CondWTGAN v2 - Dataset & DataLoader
===================================
PyTorch Dataset per traiettorie.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import json


class TrajectoryDataset(Dataset):
    """
    Dataset per traiettorie di traffico.
    
    Ogni sample contiene:
    - trajectory: (seq_len, 4) [x, y, vx, vy]
    - conditioning: (4,) [x_start, y_start, x_end, y_end]
    """
    
    def __init__(
        self,
        X: np.ndarray,
        S: np.ndarray,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            X: (N, seq_len, 4) traiettorie
            S: (N, 4) conditioning
            transform: trasformazione opzionale
        """
        assert len(X) == len(S), "X e S devono avere lo stesso numero di samples"
        
        self.X = torch.from_numpy(X).float()
        self.S = torch.from_numpy(S).float()
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.X[idx]
        conditioning = self.S[idx]
        
        if self.transform is not None:
            trajectory = self.transform(trajectory)
        
        return trajectory, conditioning


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Crea DataLoader per training e validation.
    
    Args:
        data_dir: directory con i dati processati
        batch_size: dimensione batch
        num_workers: workers per caricamento
        pin_memory: pin memory per GPU
        
    Returns:
        train_loader, val_loader, stats
    """
    data_dir = Path(data_dir)
    
    # Carica dati
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    S_train = np.load(data_dir / "S_train.npy")
    S_val = np.load(data_dir / "S_val.npy")
    
    with open(data_dir / "stats.json", "r") as f:
        stats = json.load(f)
    
    # Crea dataset
    train_dataset = TrajectoryDataset(X_train, S_train)
    val_dataset = TrajectoryDataset(X_val, S_val)
    
    # Crea dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Importante per batch normalization / gradient penalties
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader, stats


def get_sample_batch(
    data_dir: str,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ottiene un batch di esempio per testing/visualization.
    
    Returns:
        X: (batch_size, seq_len, 4)
        S: (batch_size, 4)
    """
    data_dir = Path(data_dir)
    
    X = np.load(data_dir / "X_train.npy")[:batch_size]
    S = np.load(data_dir / "S_train.npy")[:batch_size]
    
    return (
        torch.from_numpy(X).float().to(device),
        torch.from_numpy(S).float().to(device),
    )


if __name__ == "__main__":
    # Test
    from data_preprocessing import create_synthetic_data, save_processed_data
    
    # Crea dati sintetici
    X, S = create_synthetic_data(500, 60)
    
    stats = {
        'x_min': 0.0, 'x_max': 1.0,
        'y_min': 0.0, 'y_max': 1.0,
        'vx_mean': 0.0, 'vx_std': 1.0,
        'vy_mean': 0.0, 'vy_std': 1.0,
        'seq_len': 60,
    }
    
    save_processed_data("data", X, S, stats)
    
    # Test dataloader
    train_loader, val_loader, stats = create_dataloaders("data", batch_size=32)
    
    for X_batch, S_batch in train_loader:
        print(f"Batch shapes: X={X_batch.shape}, S={S_batch.shape}")
        break
    
    print("✅ Dataset test passed!")