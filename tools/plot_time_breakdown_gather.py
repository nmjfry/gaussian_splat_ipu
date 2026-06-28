#!/usr/bin/env python3
# Slide figure: single horizontal bar, fine-grained per-frame time breakdown of
# the IPU multiSlice gather algorithm, averaged over the 4 test scenes.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "slides_figures/figures/02c_time_breakdown_gather.png"

# Means over the 4 scenes — static-pose benchmark (current code).
# Total = discovery + stream + run + readback = 69.87 ms
disc_ms   = 29.06   # host: project all Gaussians, assign to tiles
stream_ms =  0.77   # host→device: upload offsets + counts
run_ms    = 39.11   # device: multiSlice gather + project + blend
rb_ms     =  0.92   # device→host: read framebuffer

segs = [
    ("Discovery\n(host CPU)",    disc_ms,   "#C44E52"),
    ("Stream\n(host→device)",    stream_ms, "#9C9C9C"),
    ("Gather + Project\n+ Blend (device)", run_ms, "#55A868"),
    ("Readback\n(device→host)",  rb_ms,     "#8172B3"),
]
total = sum(v for _, v, _ in segs)

fig, ax = plt.subplots(figsize=(11, 3.6))
left = 0.0
for name, val, col in segs:
    pct = 100 * val / total
    ax.barh(0, val, left=left, color=col, height=0.6,
            label=f"{name.replace(chr(10), ' ')}: {val:.1f} ms ({pct:.0f}%)")
    if pct > 6:
        ax.text(left + val / 2, 0, f"{pct:.0f}%", ha="center",
                va="center", color="white", fontsize=16, fontweight="bold")
    left += val

ax.set_xlim(0, total)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("Per-frame time (ms)", labelpad=8, fontsize=15)
ax.tick_params(axis="x", labelsize=13)
ax.set_title(f"multiSlice gather per-frame breakdown — mean over 4 scenes "
             f"({total:.0f} ms, {1000/total:.0f} fps)", fontsize=17, pad=14)
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
    print(f"  {name.replace(chr(10), ' '):<30} {val:6.2f} ms  {100*val/total:5.1f}%")
