import os
import json
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt

# ============== PARAMETRI ==============
RADIUS_M = 1.5          # raggio carreggiata (3m totale)
RES_M = 0.20            # risoluzione griglia (metri per pixel). 0.2m = buon compromesso
MARGIN_M = 2.0          # margine extra fuori dai bounds
MAX_SAMPLES = 50000      # puoi mettere es. 50000 se vuoi limitarlo

DATA_DIR = "data"       # cambia se serve

# ============== IO + DENORM ==============
def load_norm_params(pkl_path):
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
    dyn_min = np.asarray(params["dynamic_min"], dtype=np.float64)  # (4,)
    dyn_max = np.asarray(params["dynamic_max"], dtype=np.float64)  # (4,)
    return dyn_min, dyn_max

def denorm_X(X_norm, dyn_min, dyn_max):
    return X_norm * (dyn_max - dyn_min) + dyn_min

def rasterize_polyline_corridor(occ, poly, x_min, y_min, res):
    """
    occ: (H,W) bool, True = road
    poly: (L,2) in meters
    Disegna su griglia i punti della polilinea; poi allargamento si fa via distance transform
    """
    H, W = occ.shape
    # campionamento denso per non “bucare” segmenti lunghi
    # step ~ res/2
    step = res * 0.5

    for k in range(len(poly) - 1):
        a = poly[k]
        b = poly[k+1]
        seg = b - a
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            pts = a[None, :]
        else:
            n = int(np.ceil(seg_len / step)) + 1
            t = np.linspace(0, 1, n)
            pts = a[None,:] + t[:,None] * seg[None,:]

        # to pixel
        xs = ((pts[:,0] - x_min) / res).astype(np.int32)
        ys = ((pts[:,1] - y_min) / res).astype(np.int32)
        xs = np.clip(xs, 0, W-1)
        ys = np.clip(ys, 0, H-1)
        occ[ys, xs] = True

def main():
    X_path = os.path.join(DATA_DIR, "X_train.npy")
    L_path = os.path.join(DATA_DIR, "L_train.npy")
    norm_path = os.path.join(DATA_DIR, "normalization_params.pkl")

    X_norm = np.load(X_path)   # (N,T,4) norm [0,1]
    L = np.load(L_path)        # (N,)
    dyn_min, dyn_max = load_norm_params(norm_path)

    if MAX_SAMPLES is not None and MAX_SAMPLES < len(L):
        idx = np.random.default_rng(42).choice(len(L), size=MAX_SAMPLES, replace=False)
        X_norm = X_norm[idx]
        L = L[idx]

    X_den = denorm_X(X_norm, dyn_min, dyn_max)  # (N,T,4) in meters

    # bounds globali su xy usando solo parte valida
    xs = []
    ys = []
    for i in range(len(L)):
        Li = int(L[i])
        Li = max(2, min(Li, X_den.shape[1]))
        xs.append(X_den[i, :Li, 0])
        ys.append(X_den[i, :Li, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    x_min = float(xs.min() - (MARGIN_M + RADIUS_M))
    x_max = float(xs.max() + (MARGIN_M + RADIUS_M))
    y_min = float(ys.min() - (MARGIN_M + RADIUS_M))
    y_max = float(ys.max() + (MARGIN_M + RADIUS_M))

    W = int(np.ceil((x_max - x_min) / RES_M)) + 1
    H = int(np.ceil((y_max - y_min) / RES_M)) + 1

    print("Grid size HxW:", H, W, "  res:", RES_M, "m/px")

    occ = np.zeros((H, W), dtype=bool)

    # rasterizza tutte le polilinee
    for i in range(len(L)):
        Li = int(L[i])
        Li = max(2, min(Li, X_den.shape[1]))
        poly = X_den[i, :Li, :2]
        rasterize_polyline_corridor(occ, poly, x_min, y_min, RES_M)

        if (i+1) % 5000 == 0:
            print(f"rasterized {i+1}/{len(L)}")

    # Distance transform: distanza (in pixel) dal "road" (True).
    # distance_transform_edt calcola distanza dai pixel ZERO, quindi invertiamo.
    # invert: non-road = True -> distanza a nearest road
    dist_px = distance_transform_edt(~occ)
    dist_m = dist_px * RES_M

    # Converti in "excess fuori carreggiata": dentro raggio -> 0, fuori -> (dist - RADIUS)
    road_excess = np.maximum(0.0, dist_m - RADIUS_M).astype(np.float32)

    out_dist = os.path.join(DATA_DIR, "road_dist.npy")
    np.save(out_dist, road_excess)

    meta = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "res_m": RES_M,
        "radius_m": RADIUS_M,
        "H": int(H), "W": int(W)
    }
    out_meta = os.path.join(DATA_DIR, "road_meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_dist)
    print("Saved:", out_meta)

if __name__ == "__main__":
    main()
