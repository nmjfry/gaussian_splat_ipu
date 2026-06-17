#!/usr/bin/env python3
"""Analyse a teleport-mode benchmark CSV: group frames into per-pose blocks
of length K and show the settling curve.

Usage:
    python3 tools/analyse_teleport.py <bench_results_dir> --hold-frames 30
"""

import argparse
import csv
import math
import os
import statistics
import sys


def load_floats(path, col):
    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                v = float(row[col])
            except (ValueError, KeyError):
                continue
            if not math.isnan(v):
                out.append(v)
    return out


def per_block_stats(total_ms, hold):
    """Group frames into consecutive blocks of length `hold`. For each
    position-within-block (0..hold-1), return the mean across all blocks.
    The first block is included; if you want to exclude it as global warm-up,
    drop it before calling.
    """
    n_blocks = len(total_ms) // hold
    if n_blocks < 2:
        return None
    by_pos = [[] for _ in range(hold)]
    for b in range(n_blocks):
        for p in range(hold):
            by_pos[p].append(total_ms[b * hold + p])
    return [statistics.mean(xs) for xs in by_pos], n_blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_dir', help='bench_results_* dir containing per-scene subdirs')
    ap.add_argument('--hold-frames', type=int, default=30,
                    help='Frames-per-pose used when generating the teleport traj (default 30)')
    ap.add_argument('--report-positions', nargs='+', type=int,
                    default=[0, 1, 2, 5, 10, 20, 29],
                    help='Which frame-within-block positions to show in the table')
    ap.add_argument('--drop-first-block', action='store_true',
                    help='Skip the very first block (global IPU warm-up)')
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.results_dir)
                    if os.path.isdir(os.path.join(args.results_dir, d)))

    print(f"Teleport analysis  (hold = {args.hold_frames} frames/pose)")
    print()

    header_positions = " ".join(f"f={p:<3d}" for p in args.report_positions)
    fmt_h = "{:>10s}  {:>7s}  " + " ".join(f"{{:>{max(len(str(p))+3, 6)}s}}" for p in args.report_positions) + "  {:>9s}  {:>10s}"
    fmt_r = "{:>10s}  {:>7d}  " + " ".join(["{:>6.2f}"] * len(args.report_positions)) + "  {:>9.2f}  {:>10.2f}"

    pos_headers = [f"f={p}" for p in args.report_positions]
    print(fmt_h.format("scene", "blocks", *pos_headers, "settled", "tele-cost"))
    print("-" * (28 + 8 * len(args.report_positions) + 24))

    rows = []
    for scene in scenes:
        prof = os.path.join(args.results_dir, scene, "benchmark_profile.csv")
        if not os.path.isfile(prof):
            continue
        total_ms = load_floats(prof, "total_ms")
        if args.drop_first_block:
            total_ms = total_ms[args.hold_frames:]
        result = per_block_stats(total_ms, args.hold_frames)
        if result is None:
            print(f"  {scene}: not enough frames")
            continue
        means_by_pos, n_blocks = result

        # Settled = last position in block; teleport cost = pos0 - settled
        settled = means_by_pos[-1]
        tele_cost = means_by_pos[0] - settled
        vals = [means_by_pos[p] for p in args.report_positions if p < len(means_by_pos)]
        print(fmt_r.format(scene, n_blocks, *vals, settled, tele_cost))
        rows.append({
            "scene": scene,
            "n_blocks": n_blocks,
            "means_by_pos": means_by_pos,
            "settled": settled,
            "tele_cost": tele_cost,
        })

    # Cross-scene average per position
    if rows:
        avg_by_pos = [statistics.mean(r["means_by_pos"][p] for r in rows)
                      for p in range(args.hold_frames)]
        avg_settled = avg_by_pos[-1]
        avg_tele = avg_by_pos[0] - avg_settled
        avg_vals = [avg_by_pos[p] for p in args.report_positions if p < args.hold_frames]
        print("-" * (28 + 8 * len(args.report_positions) + 24))
        print(fmt_r.format("avg", 0, *avg_vals, avg_settled, avg_tele))

    print()
    print("  f=0       : first frame at the new pose (channels fully scrambled)")
    print("  f=hold-1  : last frame at the same pose (steady state, comparable to static)")
    print("  tele-cost : ms cost of one teleport = mean(f=0) - mean(settled)")


if __name__ == '__main__':
    main()
