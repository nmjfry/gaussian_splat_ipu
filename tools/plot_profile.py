#!/usr/bin/env python3
"""Plot per-frame timing from benchmark_profile.csv.

Usage:
    python3 tools/plot_profile.py benchmark_profile.csv
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_profile.csv"

    frames, zooms, frame_ms = [], [], []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            zooms.append(float(row["zoom"]))
            frame_ms.append(float(row["frame_ms"]))

    frames = np.array(frames)
    zooms = np.array(zooms)
    frame_ms = np.array(frame_ms)

    zoom_changes = np.where(np.diff(zooms) != 0)[0] + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: Per-frame time
    ax = axes[0]
    ax.plot(frames, frame_ms, "o-", markersize=2, label="Frame time")
    for zc in zoom_changes:
        ax.axvline(x=frames[zc], color="red", linestyle="--", alpha=0.5,
                   label="Zoom change" if zc == zoom_changes[0] else "")
    ax.set_ylabel("Frame time (ms)")
    ax.set_title("Per-frame time (expect spike after each zoom change, then settle)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: FPS
    ax = axes[1]
    fps = 1000.0 / frame_ms
    ax.plot(frames, fps, "o-", markersize=2, color="tab:green", label="FPS")
    for zc in zoom_changes:
        ax.axvline(x=frames[zc], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("FPS")
    ax.set_xlabel("Frame")
    ax.set_title("Frames per second")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.close()

    # Print summary table
    unique_zooms = sorted(set(zooms))
    print(f"\n{'Zoom':<8} {'Mean ms':>10} {'Std ms':>10} {'FPS':>8} {'Frame 0 ms':>12} {'Settled ms':>12}")
    print("-" * 65)
    for z in unique_zooms:
        mask = zooms == z
        ms = frame_ms[mask]
        settled = ms[min(20, len(ms)):]  # skip first 20 frames
        print(f"{z:<8.2f} {ms.mean():>10.3f} {ms.std():>10.3f} {1000/ms.mean():>8.1f} "
              f"{ms[0]:>12.3f} {settled.mean():>12.3f}" if len(settled) > 0 else "")

if __name__ == "__main__":
    main()
