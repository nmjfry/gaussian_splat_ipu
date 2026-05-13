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

    zooms, substeps = [], []
    route_ms, blend_ms, exchange_ms, total_ms, total_visible = [], [], [], [], []
    cyc_clear, cyc_routing, cyc_proj, cyc_sort = [], [], [], []

    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if "substep" not in fields and "frame" in fields:
            print(f"Detected old CSV format ({','.join(fields)}).")
            print("Rebuild with the new benchmark code and re-run --benchmark.")
            sys.exit(1)
        has_cycles = "clear_cyc_ms" in fields
        for row in reader:
            zooms.append(float(row["zoom"]))
            substeps.append(int(row["substep"]))
            route_ms.append(float(row["route_ms"]))
            blend_ms.append(float(row["blend_ms"]))
            exchange_ms.append(float(row["exchange_ms"]))
            total_ms.append(float(row["total_ms"]))
            total_visible.append(int(row["total_visible"]))
            if has_cycles:
                cyc_clear.append(float(row["clear_cyc_ms"]))
                cyc_routing.append(float(row["routing_cyc_ms"]))
                cyc_proj.append(float(row["proj_cyc_ms"]))
                cyc_sort.append(float(row["sort_cyc_ms"]))

    zooms = np.array(zooms)
    substeps = np.array(substeps)
    route_ms = np.array(route_ms)
    blend_ms = np.array(blend_ms)
    exchange_ms = np.array(exchange_ms)
    total_ms = np.array(total_ms)
    total_visible = np.array(total_visible)

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
        ax.plot(s, route_ms[mask], '-o', markersize=3, color=colors[i],
                label=f'route (z={z:.1f})')
        ax.plot(s, blend_ms[mask], '--s', markersize=3, color=colors[i],
                label=f'blend (z={z:.1f})', alpha=0.7)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Host-measured phase timing (route = single CS, blend = single CS)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Total substep time
    ax = axes[1]
    for i, z in enumerate(unique_zooms):
        mask = zooms == z
        s = substeps[mask]
        ax.plot(s, total_ms[mask], '-o', markersize=3, color=colors[i],
                label=f'zoom {z:.1f}')
    ax.set_ylabel("Total substep time (ms)")
    ax.set_title("Total substep time (route + blend + exchange)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 3: On-tile cycle breakdown (if available)
    if has_cycles:
        cyc_clear = np.array(cyc_clear)
        cyc_routing = np.array(cyc_routing)
        cyc_proj = np.array(cyc_proj)
        cyc_sort = np.array(cyc_sort)

        ax = axes[2]
        for i, z in enumerate(unique_zooms):
            mask = zooms == z
            s = substeps[mask]
            ax.plot(s, cyc_routing[mask], '-o', markersize=3, color=colors[i],
                    label=f'routing (z={z:.1f})')
            ax.plot(s, cyc_proj[mask], '--s', markersize=3, color=colors[i],
                    label=f'project (z={z:.1f})', alpha=0.7)
            ax.plot(s, cyc_sort[mask], ':^', markersize=3, color=colors[i],
                    label=f'sort (z={z:.1f})', alpha=0.5)
        ax.set_ylabel("Time (ms, from cycle counter)")
        ax.set_title("On-tile cycle breakdown (mean across 1440 tiles, @1.85 GHz)")
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
    hdr = (f"{'Zoom':<8} {'Route ms':>10} {'Blend ms':>10} {'Exch ms':>10} "
           f"{'Total ms':>10} {'FPS':>8}")
    if has_cycles:
        hdr += f" {'Clear':>8} {'Routing':>8} {'Project':>8} {'Sort':>8}"
    hdr += f" {'Visible':>10} {'Settled':>8}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for z in unique_zooms:
        mask = zooms == z
        r = route_ms[mask]
        b = blend_ms[mask]
        e = exchange_ms[mask]
        t = total_ms[mask]
        v = total_visible[mask]

        settled = -1
        for j in range(1, len(v)):
            if v[j] == v[j-1]:
                settled = j
                break

        line = (f"{z:<8.2f} {r.mean():>10.3f} {b.mean():>10.3f} {e.mean():>10.3f} "
                f"{t.mean():>10.3f} {1000/t.mean():>8.1f}")
        if has_cycles:
            cr = cyc_routing[mask]
            cp = cyc_proj[mask]
            cs = cyc_sort[mask]
            cc = cyc_clear[mask]
            line += f" {cc.mean():>8.3f} {cr.mean():>8.3f} {cp.mean():>8.3f} {cs.mean():>8.3f}"
        line += f" {v[-1]:>10d} {settled:>8d}"
        print(line)

if __name__ == "__main__":
    main()
