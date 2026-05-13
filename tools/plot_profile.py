#!/usr/bin/env python3
"""Plot per-frame phase timing from benchmark_profile.csv.

Usage:
    python3 tools/plot_profile.py benchmark_profile.csv
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_profile.csv"

    frames, zooms = [], []
    mvp, compute, exchange, readback, total = [], [], [], [], []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            zooms.append(float(row["zoom"]))
            mvp.append(float(row["mvp_ms"]))
            compute.append(float(row["compute_ms"]))
            exchange.append(float(row["exchange_ms"]))
            readback.append(float(row["readback_ms"]))
            total.append(float(row["total_ms"]))

    frames = np.array(frames)
    zooms = np.array(zooms)
    compute = np.array(compute)
    exchange = np.array(exchange)
    readback = np.array(readback)
    mvp_arr = np.array(mvp)
    total = np.array(total)

    zoom_changes = np.where(np.diff(zooms) != 0)[0] + 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Total frame time with phase breakdown
    ax = axes[0]
    ax.fill_between(frames, 0, compute, alpha=0.7, label="GSplat Compute")
    ax.fill_between(frames, compute, compute + exchange, alpha=0.7, label="NEWS Exchange")
    ax.fill_between(frames, compute + exchange, compute + exchange + readback,
                    alpha=0.7, label="Readback")
    ax.fill_between(frames, compute + exchange + readback, total,
                    alpha=0.5, label="MVP Broadcast")
    for zc in zoom_changes:
        ax.axvline(x=frames[zc], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-frame phase breakdown")
    ax.legend(loc="upper right")

    # Plot 2: Exchange time (routing cost) — the spike we're looking for
    ax = axes[1]
    ax.plot(frames, exchange, "o-", markersize=2, label="NEWS Exchange")
    for zc in zoom_changes:
        ax.axvline(x=frames[zc], color="red", linestyle="--", alpha=0.5,
                   label="Zoom change" if zc == zoom_changes[0] else "")
    ax.set_ylabel("Exchange time (ms)")
    ax.set_title("Routing cost per frame (expect spike after view change)")
    ax.legend(loc="upper right")

    # Plot 3: Compute time
    ax = axes[2]
    ax.plot(frames, compute, "o-", markersize=2, color="tab:orange", label="GSplat Compute")
    for zc in zoom_changes:
        ax.axvline(x=frames[zc], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Compute time (ms)")
    ax.set_xlabel("Frame")
    ax.set_title("Compute cost per frame")
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.close()

    # Print summary table
    unique_zooms = sorted(set(zooms))
    print(f"\n{'Zoom':<8} {'Compute':>10} {'Exchange':>10} {'Readback':>10} {'MVP':>10} {'Total':>10} {'FPS':>8}")
    print("-" * 70)
    for z in unique_zooms:
        mask = zooms == z
        c = compute[mask].mean()
        e = exchange[mask].mean()
        r = readback[mask].mean()
        m = mvp_arr[mask].mean()
        t = total[mask].mean()
        print(f"{z:<8.2f} {c:>10.3f} {e:>10.3f} {r:>10.3f} {m:>10.3f} {t:>10.3f} {1000/t:>8.1f}")

if __name__ == "__main__":
    main()
