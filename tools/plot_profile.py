#!/usr/bin/env python3
"""Plot per-phase timing and convergence from benchmark_profile.csv.

Usage:
    python3 tools/plot_profile.py benchmark_profile.csv
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_profile.csv"

    data = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if "substep" not in fields and "frame" in fields:
            print(f"Detected old CSV format ({','.join(fields)}).")
            print("Rebuild with the new benchmark code and re-run --benchmark.")
            sys.exit(1)
        has_cycles = "routing_mean" in fields
        for key in fields:
            data[key] = []
        for row in reader:
            for key in fields:
                data[key].append(float(row[key]))

    for key in data:
        data[key] = np.array(data[key])

    zooms = data["zoom"]
    substeps = data["substep"].astype(int)
    total_visible = data["total_visible"].astype(int)

    unique_zooms = sorted(set(zooms))
    n_zooms = len(unique_zooms)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_zooms, 2)))

    n_plots = 4 if has_cycles else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))

    # Plot 1: Host-side per-phase timing
    ax = axes[0]
    for i, z in enumerate(unique_zooms):
        mask = zooms == z
        s = substeps[mask]
        ax.plot(s, data["route_ms"][mask], '-o', markersize=3, color=colors[i],
                label=f'route (z={z:.1f})')
        ax.plot(s, data["blend_ms"][mask], '--s', markersize=3, color=colors[i],
                label=f'blend (z={z:.1f})', alpha=0.7)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Host-measured compute set timing")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Total substep time
    ax = axes[1]
    for i, z in enumerate(unique_zooms):
        mask = zooms == z
        s = substeps[mask]
        ax.plot(s, data["total_ms"][mask], '-o', markersize=3, color=colors[i],
                label=f'zoom {z:.1f}')
    ax.set_ylabel("Total substep time (ms)")
    ax.set_title("Total substep time (route + blend + exchange)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 3: On-tile cycle breakdown with min/max shading
    if has_cycles:
        ax = axes[2]
        phase_keys = [
            ("routing", "Routing"),
            ("proj", "Projection"),
            ("sort", "Sort"),
        ]
        linestyles = ['-o', '--s', ':^']
        for i, z in enumerate(unique_zooms):
            mask = zooms == z
            s = substeps[mask]
            for j, (key, label) in enumerate(phase_keys):
                mean = data[f"{key}_mean"][mask]
                lo = data[f"{key}_min"][mask]
                hi = data[f"{key}_max"][mask]
                line = ax.plot(s, mean, linestyles[j], markersize=3, color=colors[i],
                        label=f'{label} (z={z:.1f})', alpha=0.8 - 0.2*j)
                ax.fill_between(s, lo, hi, color=colors[i], alpha=0.08)
        ax.set_ylabel("Time (ms, from cycle counter)")
        ax.set_title("On-tile cycle breakdown — line=mean, shading=min..max across 1440 tiles")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Plot N: Convergence
    ax = axes[-1]
    for i, z in enumerate(unique_zooms):
        mask = zooms == z
        s = substeps[mask]
        ax.plot(s, total_visible[mask], '-o', markersize=3, color=colors[i],
                label=f'zoom {z:.1f}')
    ax.set_ylabel("Total visible Gaussians")
    ax.set_xlabel("Substep")
    ax.set_title("Convergence: visible Gaussians vs routing substeps")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")
    plt.close()

    # Summary table
    print(f"\n{'Zoom':<6} {'Route':>8} {'Blend':>8} {'Total':>8} {'FPS':>6}"
          f" | {'Routing':>22s} | {'Projection':>22s} | {'Sort':>22s}"
          f" | {'Visible':>8} {'Sttl':>5}")
    print(f"{'':6} {'':>8} {'':>8} {'':>8} {'':>6}"
          f" | {'min/mean/max':>22s} | {'min/mean/max':>22s} | {'min/mean/max':>22s}"
          f" | {'':>8} {'':>5}")
    print("-" * 140)
    for z in unique_zooms:
        mask = zooms == z
        t = data["total_ms"][mask]
        v = total_visible[mask]

        settled = -1
        for j in range(1, len(v)):
            if v[j] == v[j-1]:
                settled = j
                break

        # Use last substep values for the cycle stats
        last = mask.nonzero()[0][-1]
        line = (f"{z:<6.2f} {data['route_ms'][last]:>8.2f} {data['blend_ms'][last]:>8.2f} "
                f"{data['total_ms'][last]:>8.2f} {1000/t.mean():>6.1f}")
        if has_cycles:
            for key in ["routing", "proj", "sort"]:
                line += (f" | {data[f'{key}_min'][last]:>6.2f}/"
                         f"{data[f'{key}_mean'][last]:>6.2f}/"
                         f"{data[f'{key}_max'][last]:>6.2f}")
        line += f" | {v[-1]:>8d} {settled:>5d}"
        print(line)

if __name__ == "__main__":
    main()
