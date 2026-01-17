import json, numpy as np, matplotlib.pyplot as plt

DATA_DIR="data"
road_dist = np.load(f"{DATA_DIR}/road_dist.npy")
meta = json.load(open(f"{DATA_DIR}/road_meta.json"))

extent=[meta["x_min"], meta["x_max"], meta["y_min"], meta["y_max"]]
radius=meta["radius_m"]

# “carreggiata”: dentro raggio (qui road_dist è già “oltre raggio”)
onroad = (road_dist <= 1e-9)

plt.figure()
plt.imshow(onroad.astype(float), origin="lower", extent=extent, aspect="equal")
plt.title(f"Road area (within radius {radius} m)")
plt.xlabel("x [m]"); plt.ylabel("y [m]")

# contorno netto della carreggiata
plt.contour(onroad.astype(int), levels=[0.5], origin="lower", extent=extent, linewidths=1.0)
plt.savefig(f"{DATA_DIR}/road_dist_heatmap.png", dpi=200, bbox_inches="tight")


plt.show()


