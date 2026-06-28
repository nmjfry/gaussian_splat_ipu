#!/usr/bin/env python3
# Slide figure: single horizontal bar, fine-grained per-frame KERNEL breakdown
# of the GPU (RTX 4090) 3D Gaussian Splatting pipeline (diff-gaussian-
# rasterization), averaged over the same 4 test scenes as the IPU figures.
#
# Numbers are GPU-busy time (sum of CUDA kernel durations) from nsys, mapped to
# the DGR pipeline stages:
#   Projection     <- preprocessCUDA
#   Data movement  <- duplicateWithKeys + identifyTileRanges + cub DeviceScan
#   Sorting        <- cub::DeviceRadixSort* kernels
#   Alpha blending <- renderCUDA
# Stage colours match plot_time_breakdown_fine.py so the IPU and GPU bars line
# up stage-for-stage. NOTE: these scenes are tiny, so the GPU is launch-bound —
# wall-clock per frame (~0.4-0.8 ms) is larger than the 0.18 ms of actual
# compute because of kernel-launch gaps. This bar shows the compute, not gaps.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "slides_figures/figures/02d_time_breakdown_gpu.png"

# Mean over chairs/pringles/salad/sloth (nsys, 200 frames each), per frame.
proj_ms     = 0.0049   # preprocessCUDA
datamove_ms = 0.0406   # duplicateWithKeys + identifyTileRanges + DeviceScan
sort_ms     = 0.0477   # cub DeviceRadixSort*
blend_ms    = 0.0891   # renderCUDA

segs = [
    ("Projection\n(preprocess)",      proj_ms,     "#4C72B0"),
    ("Data movement\n(duplicate+bin)", datamove_ms, "#DD8452"),
    ("Sorting\n(global radix)",        sort_ms,     "#55A868"),
    ("Alpha blending\n(render)",       blend_ms,    "#C44E52"),
]
total = sum(v for _, v, _ in segs)

fig, ax = plt.subplots(figsize=(11, 3.6))
left = 0.0
for name, val, col in segs:
    pct = 100 * val / total
    us = val * 1000.0
    ax.barh(0, val, left=left, color=col, height=0.6,
            label=f"{name.replace(chr(10), ' ')}: {us:.0f} µs ({pct:.0f}%)")
    if pct > 6:
        ax.text(left + val / 2, 0, f"{pct:.0f}%", ha="center",
                va="center", color="white", fontsize=16, fontweight="bold")
    left += val

ax.set_xlim(0, total)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Per-frame GPU-busy time (ms)", labelpad=8, fontsize=15)
ax.tick_params(axis="x", labelsize=13)
ax.set_title(f"GPU (RTX 4090) per-frame kernel breakdown — mean over 4 scenes "
             f"({total*1000:.0f} µs busy, ~{1000/total:.0f} fps)", fontsize=16, pad=14)
ax.spines[["top", "right", "left"]].set_visible(False)

fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.46)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
           handletextpad=0.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
for name, val, _ in segs:
    print(f"  {name.replace(chr(10), ' '):<26} {val*1000:6.1f} us  {100*val/total:5.1f}%")
print(f"  {'TOTAL (GPU-busy)':<26} {total*1000:6.1f} us  ~{1000/total:.0f} fps")
