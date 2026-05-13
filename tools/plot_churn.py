#!/usr/bin/env python3
"""Plot tile-churn experiment results for EGSR rebuttal.

Usage:
    python3 tools/plot_churn.py --input churn_results.csv --outdir churn_plots/
"""

import argparse
import os
import csv
import numpy as np
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — skipping plots, printing summary only")


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame"] = int(row["frame"])
            for k in ("n_visible", "n_stable", "n_moved", "n_entered", "n_exited"):
                row[k] = int(row[k])
            for k in ("churn_rate", "work_ratio", "mean_hops", "p95_hops"):
                row[k] = float(row[k])
            rows.append(row)
    return rows


def group_by_trajectory(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["trajectory"]].append(r)
    return dict(groups)


def plot_bar_chart(groups, outdir):
    """Plot 1: Mean incremental work ratio by trajectory type."""
    names, means, stds = [], [], []
    for name in sorted(groups.keys()):
        data = [r["work_ratio"] for r in groups[name] if r["frame"] > 0]
        if data:
            names.append(name.replace("_", "\n"))
            means.append(np.mean(data))
            stds.append(np.std(data))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Incremental Work Ratio")
    ax.set_title("Fraction of Gaussians Requiring Routing per Frame")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="Cold-start baseline")
    ax.legend(fontsize=8)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "work_ratio_bar.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved work_ratio_bar.png")


def plot_velocity_sweep(groups, outdir):
    """Plot 2: Churn rate vs orbit angular velocity."""
    omegas, churns, stds = [], [], []
    for name in groups.keys():
        if name.startswith("orbit_"):
            omega = float(name.split("_")[1].replace("deg", ""))
            data = [r["churn_rate"] for r in groups[name] if r["frame"] > 0]
            if data:
                omegas.append(omega)
                churns.append(np.mean(data))
                stds.append(np.std(data))
    # Sort by omega (not alphabetically)
    order = np.argsort(omegas)
    omegas = [omegas[i] for i in order]
    churns = [churns[i] for i in order]
    stds = [stds[i] for i in order]

    if not omegas:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(omegas, churns, yerr=stds, fmt="o-", color="#DD8452",
                capsize=4, markersize=6, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Orbit Angular Velocity (deg/frame)")
    ax.set_ylabel("Mean Tile-Churn Rate")
    ax.set_title("Churn Rate Scales with Motion Magnitude")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "velocity_sweep.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved velocity_sweep.png")


def plot_hop_histogram(groups, outdir):
    """Plot 3: Hop distance distribution for the slowest orbit (SLAM-like)."""
    # Use orbit_0.5deg as a representative smooth trajectory
    target = None
    for name in ("orbit_0.5deg", "orbit_0.1deg", "pure_translation"):
        if name in groups:
            target = name
            break
    if not target:
        return

    hops = [r["mean_hops"] for r in groups[target] if r["frame"] > 0 and r["mean_hops"] > 0]
    if not hops:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hops, bins=range(0, max(int(max(hops)) + 3, 6)), color="#55A868",
            alpha=0.85, edgecolor="white", rwidth=0.85)
    ax.set_xlabel("Mean Hop Distance (Manhattan, tile units)")
    ax.set_ylabel("Frame Count")
    ax.set_title(f"Hop Distribution — {target}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hop_histogram.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved hop_histogram.png")


def plot_timeline(groups, outdir):
    """Plot 4: Per-frame churn rate timeline, one line per trajectory."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for i, name in enumerate(sorted(groups.keys())):
        frames = [r["frame"] for r in groups[name] if r["frame"] > 0]
        churns = [r["churn_rate"] for r in groups[name] if r["frame"] > 0]
        if frames:
            ax.plot(frames, churns, label=name, color=cmap(i % 10), alpha=0.7, lw=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Tile-Churn Rate")
    ax.set_title("Per-Frame Churn Rate by Trajectory")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "churn_timeline.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved churn_timeline.png")


def main():
    parser = argparse.ArgumentParser(description="Plot churn experiment results")
    parser.add_argument("--input", default="churn_results.csv")
    parser.add_argument("--outdir", default="churn_plots")
    args = parser.parse_args()

    rows = load_csv(args.input)
    groups = group_by_trajectory(rows)

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Trajectory':<25} {'MeanChurn':>10} {'MeanWork':>10} {'MeanHops':>10}")
    for name in sorted(groups.keys()):
        data = [r for r in groups[name] if r["frame"] > 0]
        mc = np.mean([r["churn_rate"] for r in data]) if data else 0
        mw = np.mean([r["work_ratio"] for r in data]) if data else 0
        mh = np.mean([r["mean_hops"] for r in data]) if data else 0
        print(f"{name:<25} {mc:>10.4f} {mw:>10.4f} {mh:>10.2f}")

    if not HAS_MPL:
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nGenerating plots in {args.outdir}/")
    plot_bar_chart(groups, args.outdir)
    plot_velocity_sweep(groups, args.outdir)
    plot_hop_histogram(groups, args.outdir)
    plot_timeline(groups, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
