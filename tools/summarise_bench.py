#!/usr/bin/env python3
"""Summarise benchmark results across scenes: mean/std ms-per-frame and power.

Reads bench_results/<scene>/benchmark_profile.csv and power_samples.csv,
prints a table, and writes bench_results/summary.csv.
"""

import csv
import os
import sys
import math
import statistics


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


def stats(xs):
    if not xs:
        return None, None, None, None
    return min(xs), statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, max(xs)


def main():
    if len(sys.argv) < 2:
        print("usage: summarise_bench.py <bench_results_dir>")
        sys.exit(1)
    root = sys.argv[1]
    scenes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    rows = []
    for scene in scenes:
        d = os.path.join(root, scene)
        prof = os.path.join(d, "benchmark_profile.csv")
        pwr = os.path.join(d, "power_samples.csv")
        if not os.path.isfile(prof):
            continue

        total_ms = load_floats(prof, "total_ms")
        route_ms = load_floats(prof, "route_ms")
        blend_ms = load_floats(prof, "blend_ms")
        visible = load_floats(prof, "total_visible")
        powers = load_floats(pwr, "power_w") if os.path.isfile(pwr) else []
        # Drop the first power sample of each scene as warmup
        powers = [p for p in powers if p > 1.0]

        # Drop first 30 frames of timing to ignore the channel-saturation transient.
        warmup = 30
        tail_total = total_ms[warmup:] if len(total_ms) > warmup else total_ms
        tail_route = route_ms[warmup:] if len(route_ms) > warmup else route_ms
        tail_blend = blend_ms[warmup:] if len(blend_ms) > warmup else blend_ms

        rows.append({
            "scene": scene,
            "frames": len(total_ms),
            "total_ms_mean": statistics.mean(tail_total) if tail_total else float("nan"),
            "total_ms_std": statistics.stdev(tail_total) if len(tail_total) > 1 else 0.0,
            "fps_mean": (1000.0 / statistics.mean(tail_total)) if tail_total else float("nan"),
            "route_ms_mean": statistics.mean(tail_route) if tail_route else float("nan"),
            "blend_ms_mean": statistics.mean(tail_blend) if tail_blend else float("nan"),
            "visible_mean": statistics.mean(visible) if visible else float("nan"),
            "power_w_mean": statistics.mean(powers) if powers else float("nan"),
            "power_w_max": max(powers) if powers else float("nan"),
            "power_samples": len(powers),
        })

    # Print
    fmt = "{:>10s}  {:>6s}  {:>10s}  {:>9s}  {:>7s}  {:>9s}  {:>9s}  {:>10s}  {:>9s}  {:>9s}"
    print(fmt.format("scene", "frames", "total_ms", "fps", "±std", "route_ms", "blend_ms",
                     "visible", "power_W", "peak_W"))
    print("-" * 105)
    rfmt = "{:>10s}  {:>6d}  {:>10.2f}  {:>9.2f}  {:>7.2f}  {:>9.2f}  {:>9.2f}  {:>10.0f}  {:>9.2f}  {:>9.2f}"
    for r in rows:
        print(rfmt.format(r["scene"], r["frames"], r["total_ms_mean"], r["fps_mean"],
                          r["total_ms_std"], r["route_ms_mean"], r["blend_ms_mean"],
                          r["visible_mean"], r["power_w_mean"], r["power_w_max"]))

    # Average across scenes
    if rows:
        def avg(k): return statistics.mean(r[k] for r in rows if not math.isnan(r[k]))
        print("-" * 105)
        print(rfmt.format("avg", 0, avg("total_ms_mean"), avg("fps_mean"), 0.0,
                          avg("route_ms_mean"), avg("blend_ms_mean"),
                          avg("visible_mean"), avg("power_w_mean"), avg("power_w_max")))

    # Write summary CSV
    out_csv = os.path.join(root, "summary.csv")
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
