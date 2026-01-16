import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import pickle
import torch.nn.functional as F


import torch

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

def load_numpy_triplet(X_path, S_path, L_path):
    """
    Carica:
        X: (N, 120, 4)
        S: (N, 4)
        L: (N,)  lunghezze reali (int)
    """
    X = np.load(X_path)
    S = np.load(S_path)
    L = np.load(L_path)

    assert X.ndim == 3 and X.shape[2] == 4, "X deve essere (N,120,4)"
    assert S.ndim == 2 and S.shape[1] == 4, "S deve essere (N,4)"
    assert L.ndim == 1 and len(L) == len(X), "L deve essere (N,) e allineato a X"
    assert np.all((L >= 1) & (L <= X.shape[1])), "L fuori range [1, T]"

    return X, S, L

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


def _parse_lane_shape(shape_str: str):
    """
    shape_str formato SUMO: "x,y x,y x,y ..."
    ritorna lista di (x,y)
    """
    pts = []
    for token in shape_str.strip().split():
        x_str, y_str = token.split(",")
        pts.append((float(x_str), float(y_str)))
    return pts


def load_lanes_from_net(net_xml_path: str):
    """
    Estrae lane centerline e width dal file rete.net.xml.
    Ritorna lista di dict: {lane_id, width, shape[(x,y),...]}
    """
    if not os.path.exists(net_xml_path):
        raise FileNotFoundError(f"rete.net.xml non trovato: {net_xml_path}")

    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    lanes = []
    for lane in root.iter("lane"):
        lane_id = lane.get("id")
        shape_str = lane.get("shape")
        if not lane_id or not shape_str:
            continue

        width_attr = lane.get("width")
        width = float(width_attr) if width_attr is not None else None

        shape = _parse_lane_shape(shape_str)
        if len(shape) < 2:
            continue

        lanes.append({
            "lane_id": lane_id,
            "width": width,
            "shape": shape
        })

    if len(lanes) == 0:
        raise RuntimeError("Non ho trovato lane con attributo 'shape' nel net.xml.")

    return lanes


def build_road_geometry(net_xml_path: str, lane_width_override: float = 3.0, default_lane_width: float = 3.0):
    """
    Crea una geometria 'carreggiata' unendo tutte le corsie.
    - Se lane_width_override è not None: usa SEMPRE quella (es. 3.0m)
    - altrimenti usa width della lane se presente, altrimenti default_lane_width

    Ritorna:
      road_union (shapely geometry)
      road_prepared (prepared geometry per contains() veloce)
    """
    lanes = load_lanes_from_net(net_xml_path)

    polys = []
    for ln in lanes:
        shape = ln["shape"]
        ls = LineString(shape)

        if lane_width_override is not None:
            w = float(lane_width_override)
        else:
            w = float(ln["width"]) if (ln["width"] is not None and ln["width"] > 0) else float(default_lane_width)

        # buffer di metà larghezza corsia
        poly = ls.buffer(w / 2.0, cap_style=2, join_style=2)
        polys.append(poly)

    road_union = unary_union(polys)
    road_prepared = prep(road_union)
    return road_union, road_prepared


def trajectory_inside_mask(X_xy: np.ndarray, road_prepared):
    """
    X_xy: (T,2) in metri
    Ritorna mask bool (T,) True se punto è dentro carreggiata
    """
    T = X_xy.shape[0]
    inside = np.zeros(T, dtype=bool)
    for t in range(T):
        inside[t] = road_prepared.contains(Point(float(X_xy[t, 0]), float(X_xy[t, 1])))
    return inside


def compute_offroad_metrics_for_batch(X_den: np.ndarray, L: np.ndarray, road_prepared):
    """
    X_den: (N,T,4) denormalizzato (metri)
    L: (N,) lunghezze reali
    Ritorna:
      per_traj: lista dict per ogni traiettoria
      summary: dict aggregato
    """
    N, T, _ = X_den.shape
    per_traj = []

    inside_rates = []
    ever_off = []
    final_inside = []
    max_off_runs = []

    for i in range(N):
        Li = int(L[i])
        Li = max(1, min(Li, T))

        xy = X_den[i, :Li, :2]
        inside = trajectory_inside_mask(xy, road_prepared)

        inside_rate = float(inside.mean())
        off = ~inside

        # max run consecutiva offroad
        run = 0
        max_run = 0
        for v in off:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0

        d = {
            "inside_rate": inside_rate,
            "offroad_rate": float(1.0 - inside_rate),
            "ever_offroad": bool(off.any()),
            "final_inside": bool(inside[-1]),
            "max_offroad_run": int(max_run),
            "L": int(Li),
        }
        per_traj.append(d)

        inside_rates.append(inside_rate)
        ever_off.append(d["ever_offroad"])
        final_inside.append(d["final_inside"])
        max_off_runs.append(max_run)

    inside_rates = np.array(inside_rates, dtype=np.float64)
    max_off_runs = np.array(max_off_runs, dtype=np.float64)

    summary = {
        "inside_rate_mean": float(np.mean(inside_rates)),
        "inside_rate_p50": float(np.percentile(inside_rates, 50)),
        "inside_rate_p90": float(np.percentile(inside_rates, 90)),
        "inside_rate_p95": float(np.percentile(inside_rates, 95)),
        "ever_offroad_frac": float(np.mean(np.array(ever_off, dtype=np.float64))),
        "final_inside_frac": float(np.mean(np.array(final_inside, dtype=np.float64))),
        "max_offroad_run_mean": float(np.mean(max_off_runs)),
        "max_offroad_run_p90": float(np.percentile(max_off_runs, 90)),
    }

    return per_traj, summary


def plot_road_on_ax(ax, net_xml_path: str, lane_width_override: float = 3.0, default_lane_width: float = 3.0,
                    draw_polygons=True, draw_centerlines=False, alpha=0.15, linewidth=0.8):
    """
    Disegna la carreggiata sul plot.
    - draw_polygons=True: disegna i bordi delle lane (buffer)
    - draw_centerlines=True: disegna le centerline

    Consiglio: draw_polygons=True e alpha basso.
    """
    lanes = load_lanes_from_net(net_xml_path)

    if draw_polygons:
        for ln in lanes:
            shape = ln["shape"]
            ls = LineString(shape)

            if lane_width_override is not None:
                w = float(lane_width_override)
            else:
                w = float(ln["width"]) if (ln["width"] is not None and ln["width"] > 0) else float(default_lane_width)

            poly = ls.buffer(w / 2.0, cap_style=2, join_style=2)

            if isinstance(poly, Polygon):
                x, y = poly.exterior.xy
                ax.plot(x, y, linewidth=linewidth)
                ax.fill(x, y, alpha=alpha)
            else:
                # MultiPolygon: disegna ogni parte
                for p in poly.geoms:
                    x, y = p.exterior.xy
                    ax.plot(x, y, linewidth=linewidth)
                    ax.fill(x, y, alpha=alpha)

    if draw_centerlines:
        for ln in lanes:
            pts = np.array(ln["shape"], dtype=np.float64)
            ax.plot(pts[:, 0], pts[:, 1], linewidth=0.6, alpha=0.6)


import torch

def enforce_start_end_xy(xy: torch.Tensor, cond: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Enforce x,y to start at (x0,y0) and end at (x1,y1) on the LAST VALID timestep for each sample.
    Uses baseline linear interpolation + windowed residual.
    
    Args:
        xy:       (B, T, 2) predicted x,y
        cond:     (B, 4) [x0,y0,x1,y1] same normalization as xy
        pad_mask: (B, T) bool, True = padding, False = valid. Can be None.

    Returns:
        (B, T, 2) corrected xy
    """
    B, T, _ = xy.shape
    device = xy.device

    start = cond[:, 0:2]  # (B,2)
    end   = cond[:, 2:4]  # (B,2)

    if pad_mask is None:
        lengths = torch.full((B,), T, device=device, dtype=torch.long)
    else:
        # number of valid steps per sample
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)  # (B,)
    last_idx = lengths - 1  # (B,)

    # Build per-sample normalized time tau in [0,1] on valid part
    idx = torch.arange(T, device=device).view(1, T).expand(B, T)  # (B,T)
    denom = last_idx.clamp(min=1).unsqueeze(1).float()            # (B,1)
    tau = (idx.float() / denom).clamp(0.0, 1.0).unsqueeze(-1)     # (B,T,1)

    # Baseline linear interpolation between start and end
    base = (1.0 - tau) * start.unsqueeze(1) + tau * end.unsqueeze(1)  # (B,T,2)

    # Window: 0 at endpoints, 1 in the middle (valid region)
    w = 4.0 * tau * (1.0 - tau)  # (B,T,1)

    # Residual around baseline
    residual = xy - base
    xy_out = base + w * residual

    # If padding exists: keep padded timesteps as-is (or set them to last valid; your choice)
    if pad_mask is not None:
        idx_b = torch.arange(B, device=device)
        last_xy = xy_out[idx_b, last_idx, :].unsqueeze(1)  # (B,1,2)
        xy_out = torch.where((~pad_mask).unsqueeze(-1), xy_out, last_xy)

    # Hard guarantee exact endpoints on valid start and valid end
    xy_out[:, 0, :] = start
    xy_out[torch.arange(B, device=device), last_idx, :] = end

    return xy_out
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

def load_road_tensors(data_dir: str, device: torch.device):
    dist_path = os.path.join(data_dir, "road_dist.npy")
    meta_path = os.path.join(data_dir, "road_meta.json")

    road_dist_np = np.load(dist_path)  # (H,W) float32
    road_dist = torch.tensor(road_dist_np, dtype=torch.float32, device=device)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return road_dist, meta


def denorm_xy_torch(xy_norm: torch.Tensor, dyn_min: torch.Tensor, dyn_max: torch.Tensor):
    """
    xy_norm: (B,T,2) in [0,1]
    dyn_min/max: (4,) torch
    returns (B,T,2) in meters
    """
    min_xy = dyn_min[:2].view(1, 1, 2)
    max_xy = dyn_max[:2].view(1, 1, 2)
    return xy_norm * (max_xy - min_xy) + min_xy


def road_loss_from_distmap(
    fake_xy_den: torch.Tensor,
    pad_mask: torch.Tensor | None,
    road_dist: torch.Tensor,
    meta: dict,
    power: float = 2.0,
):
    """
    fake_xy_den: (B,T,2) meters
    pad_mask: (B,T) bool True=padding
    road_dist: (H,W) meters-excess (0 inside, >0 outside)
    """
    B, T, _ = fake_xy_den.shape
    H, W = road_dist.shape

    x_min = float(meta["x_min"])
    x_max = float(meta["x_max"])
    y_min = float(meta["y_min"])
    y_max = float(meta["y_max"])

    dx = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
    dy = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

    x = fake_xy_den[..., 0]  # (B,T)
    y = fake_xy_den[..., 1]  # (B,T)

    # Normalized grid coords for grid_sample: [-1,1]
    gx = ((x - x_min) / dx) * 2.0 - 1.0
    gy = ((y - y_min) / dy) * 2.0 - 1.0

    # grid_sample expects grid (N, H_out, W_out, 2)
    # We'll sample (T x 1) "image"
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B,T,1,2)

    img = road_dist.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    img = img.expand(B, 1, H, W)                       # (B,1,H,W)

    sampled = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B,1,T,1)

    d = sampled.squeeze(1).squeeze(-1)  # (B,T) meters-excess (>=0)

    if pad_mask is not None:
        valid = (~pad_mask).float()
        d = d * valid
        denom = valid.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(B * T, device=fake_xy_den.device, dtype=torch.float32)

    if power == 1.0:
        return d.sum() / denom
    else:
        return (d ** power).sum() / denom


def linear_warmup(step: int, start: int, ramp: int, max_val: float) -> float:
    """
    step < start       -> 0
    start..start+ramp  -> linear 0..max_val
    > start+ramp       -> max_val
    """
    if ramp <= 0:
        return float(max_val) if step >= start else 0.0
    if step < start:
        return 0.0
    t = (step - start) / float(ramp)
    t = max(0.0, min(1.0, t))
    return float(max_val) * t
