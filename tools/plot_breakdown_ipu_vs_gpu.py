#!/usr/bin/env python3
# Slide figure: IPU (NEWS) vs GPU (RTX 4090) per-frame pipeline breakdown,
# two stacked horizontal bars normalised to 100% so the proportions compare
# directly. Same 4 stages, same colours as the individual breakdown figures.
#
# The story: the GPU spends ~26% of its pipeline on a GLOBAL radix sort; the
# IPU's local per-tile sort is ~7%. Projection inverts the other way (GPU 3%
# vs IPU 23%) because the GPU has far more parallelism per Gaussian.
#
# IPU numbers: plot_time_breakdown_fine.py (mean over 4 scenes, NEWS).
#   route folds Receive&Route + Exchange into "data movement / routing".
# GPU numbers: plot_time_breakdown_gpu.py (nsys kernel sum, mean over 4 scenes).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "slides_figures/figures/02e_breakdown_ipu_vs_gpu.png"

# Stage order + colours (shared with the single-bar figures).
STAGES = [
    ("Projection",            "#4C72B0"),
    ("Data movement / routing", "#DD8452"),
    ("Sorting",               "#55A868"),
    ("Alpha blending",        "#C44E52"),
]

# Per-frame ms, mean over chairs/pringles/salad/sloth.
# IPU NEWS: proj / (receive&route + exchange) / sort / blend.
ipu_ms = np.array([8.74, 11.30 + 0.07, 2.61, 14.95])
# GPU RTX 4090: preprocess / (duplicate+bin+scan) / radix sort / render.
gpu_ms = np.array([0.0049, 0.0406, 0.0477, 0.0891])

ipu_pct = 100 * ipu_ms / ipu_ms.sum()
gpu_pct = 100 * gpu_ms / gpu_ms.sum()

rows = [
    (f"Graphcore IPU C600\n(NEWS, on-chip)\n{ipu_ms.sum():.1f} ms · {1000/ipu_ms.sum():.0f} fps", ipu_pct),
    (f"NVIDIA RTX 4090\n(diff-gaussian-raster.)\n{gpu_ms.sum()*1000:.0f} µs busy · ~{1000/gpu_ms.sum():.0f} fps", gpu_pct),
]

fig, ax = plt.subplots(figsize=(12, 4.6))
ypos = [1.0, 0.0]
bar_h = 0.5

for (label, pct), y in zip(rows, ypos):
    left = 0.0
    for (sname, col), p in zip(STAGES, pct):
        ax.barh(y, p, left=left, height=bar_h, color=col,
                edgecolor="white", linewidth=1.2)
        if p > 4.5:
            ax.text(left + p / 2, y, f"{p:.0f}%", ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold")
        left += p

ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=12)
ax.set_xlim(0, 100)
ax.set_ylim(-0.45, 1.45)
ax.set_xlabel("Share of per-frame pipeline time (%)", fontsize=14, labelpad=8)
ax.tick_params(axis="x", labelsize=12)
ax.spines[["top", "right"]].set_visible(False)

# Title + subtitle carry the sort-inversion story (no in-plot callouts to
# avoid colliding with the bars / axis).
fig.suptitle("Where the frame goes: IPU local routing + sort vs GPU global sort",
             fontsize=16, fontweight="bold", y=0.98)
fig.text(0.5, 0.88,
         "The GPU spends 26% of its frame on a global radix sort (green); "
         "the IPU's local per-tile sort is just 7%.",
         ha="center", fontsize=11.5, color="#2f6b40")

# Legend.
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in STAGES]
fig.legend(handles, [s for s, _ in STAGES], loc="lower center", ncol=4,
           frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.01))

fig.subplots_adjust(left=0.21, right=0.98, top=0.80, bottom=0.24)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(f"Saved {OUT}")
print(f"{'stage':<24}{'IPU %':>8}{'GPU %':>8}")
for (s, _), ip, gp in zip(STAGES, ipu_pct, gpu_pct):
    print(f"{s:<24}{ip:8.1f}{gp:8.1f}")
