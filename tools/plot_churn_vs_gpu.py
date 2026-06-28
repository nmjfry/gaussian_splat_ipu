#!/usr/bin/env python3
# Slide figure: IPU tile-churn (per-frame re-route work) scales with motion,
# while the GPU re-sorts 100% of the scene every frame regardless of velocity.
# Reads the real churn_results.csv so the IPU curve + error bars are exact.
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "churn_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "slides_figures/figures/03_churn_velocity_sweep_gpu.png"

# --- gather orbit sweep means/stds, exactly like plot_churn.py ---
rows = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["trajectory"].startswith("orbit_") and int(r["frame"]) > 0:
            rows.setdefault(r["trajectory"], []).append(float(r["churn_rate"]))

omegas, churns, stds = [], [], []
for name, data in rows.items():
    omegas.append(float(name.split("_")[1].replace("deg", "")))
    churns.append(np.mean(data))
    stds.append(np.std(data))
order = np.argsort(omegas)
omegas = np.array(omegas)[order]
churns = np.array(churns)[order]
stds = np.array(stds)[order]

IPU = "#DD8452"   # orange, matches existing figure
GPU = "#4C72B0"   # blue

fig, ax = plt.subplots(figsize=(7, 4.5))

# Shade the work the IPU avoids (between its curve and the GPU's constant 1.0).
# Neutral gray + diagonal hatch: shows on a projector even when light fills wash
# out (hatch lines survive the contrast crush that kills a pale tint).
ax.fill_between(omegas, churns, 1.0, facecolor="#b5b5b5", alpha=0.40,
                hatch="//", edgecolor="#5f5f5f", linewidth=0.0, zorder=0,
                label="Work the IPU avoids")

# GPU: global re-sort every frame -> 100% of the scene, independent of motion.
ax.plot(omegas, np.ones_like(omegas), "s--", color=GPU, lw=1.8, markersize=6,
        label="GPU: global re-sort every frame (100%)")

# IPU: only the churned Gaussians need re-routing.
ax.errorbar(omegas, churns, yerr=stds, fmt="o-", color=IPU, capsize=4,
            markersize=6, lw=1.8, label="IPU: NEWS routing (scales with motion)")

ax.set_xscale("log")
ax.set_xlabel("Orbit Angular Velocity (deg/frame)")
ax.set_ylabel("Mean Tile-Churn Rate")
ax.set_title("IPU scales with motion, GPU is always worst case")
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
print("IPU churn:", dict(zip(omegas.tolist(), np.round(churns, 4).tolist())))
