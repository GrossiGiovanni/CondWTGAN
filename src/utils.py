import os
import json
import numpy as np


# -------------------------------------------------------------
#  NORMALIZZAZIONE (COERENTE CON X: (B,120,4))
# -------------------------------------------------------------

def compute_min_max(X):
    """
    Calcola min e max per ogni feature dinamica.
    
    X shape: (B, 120, 4)
    Feature: [x, y, speed, angle]
    
    Ritorna:
        mins: (4,)
        maxs: (4,)
    """
    X_flat = X.reshape(-1, X.shape[-1])  # (B*120, 4)
    mins = X_flat.min(axis=0)
    maxs = X_flat.max(axis=0)
    return mins, maxs


def normalize_X(X, mins, maxs):
    """
    Normalizzazione min-max con epsilon.
    Usata SOLO se lavori con X grezzo non ancora normalizzato.
    """
    eps = 1e-8
    return (X - mins) / (maxs - mins + eps)


def denormalize_X(X_norm, mins, maxs):
    """
    Riporta le traiettorie dallo spazio [0,1]
    allo spazio reale SUMO.
    """
    return X_norm * (maxs - mins) + mins


# -------------------------------------------------------------
#  NORMALIZZAZIONE CONDIZIONI STATICHE S = (B,4)
# -------------------------------------------------------------

def compute_min_max_S(S):
    """
    S shape: (B, 4)
    Feature: [x_init, y_init, x_final, y_final]
    """
    mins = S.min(axis=0)
    maxs = S.max(axis=0)
    return mins, maxs


def normalize_S(S, mins, maxs):
    eps = 1e-8
    return (S - mins) / (maxs - mins + eps)


def denormalize_S(S_norm, mins, maxs):
    return S_norm * (maxs - mins) + mins


# -------------------------------------------------------------
#  SALVATAGGIO / CARICAMENTO PARAMETRI MIN-MAX
# -------------------------------------------------------------

def save_min_max(mins, maxs, path):
    """
    Salva min/max in JSON leggibile.
    """
    data = {
        "mins": mins.tolist(),
        "maxs": maxs.tolist()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_min_max(path):
    """
    Ritorna mins e maxs come array numpy.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["mins"]), np.array(data["maxs"])


# -------------------------------------------------------------
#  CARICAMENTO DATASET NUMPY
# -------------------------------------------------------------

def load_numpy_pair(X_path, S_path):
    """
    Carica:
        X: (B, 120, 4)
        S: (B, 4)
    """
    X = np.load(X_path)
    S = np.load(S_path)

    assert X.ndim == 3 and X.shape[2] == 4, "X deve essere (B,120,4)"
    assert S.ndim == 2 and S.shape[1] == 4, "S deve essere (B,4)"

    return X, S


# -------------------------------------------------------------
#  AIUTO FILESYSTEM
# -------------------------------------------------------------

def ensure_dir(path):
    """
    Crea cartella ricorsivamente se non esiste.
    """
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------------------------------------------
#  STORICO TRAINING
# -------------------------------------------------------------

def save_training_history(history_dict, out_path):
    """
    Salva JSON con andamento training:
    - d_loss
    - g_loss
    - gp
    - phys
    """
    with open(out_path, "w") as f:
        json.dump(history_dict, f, indent=4)
