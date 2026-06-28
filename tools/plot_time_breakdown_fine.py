#!/usr/bin/env python3
# Slide figure: single horizontal bar, fine-grained per-frame time breakdown of
# the IPU NEWS algorithm, averaged over the 4 test scenes (orbit benchmark).
#
# The measured frame time splits into Route (host wall) + Blend + Exchange.
# Route is further decomposed into Projection / Routing / Sort using the on-tile
# cycle-counter proportions (max across tiles = the BSP-barrier-limiting tile,
# whose per-phase maxima sum to ~the measured route time). Segments sum to the
# measured frame time.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "slides_figures/figures/02b_time_breakdown_static.png"

# Means over the 4 scenes — updated static-pose benchmark (current code).
# route_ms = total - blend - exchange = 37.67 - 14.95 - 0.07.
route_ms, blend_ms, exch_ms = 22.65, 14.95, 0.07
proj_max, rout_max, sort_max = 7.81, 10.10, 2.33     # on-tile cycle maxima (ms)

# Split the measured route time by the cycle-max proportions.
denom = proj_max + rout_max + sort_max
proj = route_ms * proj_max / denom
rout = route_ms * rout_max / denom
sort = route_ms * sort_max / denom

segs = [
    ("Projection", proj, "#4C72B0"),
    ("Receive & Route", rout, "#DD8452"),
    ("Sort", sort, "#55A868"),
    ("Blend (rasterize)", blend_ms, "#C44E52"),
    ("Exchange", exch_ms, "#8172B3"),
]
total = sum(v for _, v, _ in segs)

fig, ax = plt.subplots(figsize=(11, 3.6))
left = 0.0
for name, val, col in segs:
    ax.barh(0, val, left=left, color=col, height=0.6,
            label=f"{name}: {val:.1f} ms ({100*val/total:.0f}%)")
    if val / total > 0.06:   # in-bar label for the big segments
        ax.text(left + val / 2, 0, f"{100*val/total:.0f}%", ha="center",
                va="center", color="white", fontsize=16, fontweight="bold")
    left += val

ax.set_xlim(0, total)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Per-frame time (ms)", labelpad=8, fontsize=15)
ax.tick_params(axis="x", labelsize=13)
ax.set_title(f"IPU per-frame time breakdown — mean over 4 scenes "
             f"({total:.0f} ms, {1000/total:.0f} fps)", fontsize=17, pad=14)
ax.spines[["top", "right", "left"]].set_visible(False)

# Axis label stays attached to the bar; legend goes at the figure bottom with a
# clear gap below "Per-frame time (ms)".
fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.46)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
           handletextpad=0.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
for name, val, _ in segs:
    print(f"  {name:<18} {val:6.2f} ms  {100*val/total:5.1f}%")
