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
        exchange_ms = load_floats(prof, "exchange_ms")
        # Per-tile cycle counter min/mean/max (over 1440 tiles) for the
        # subphases inside route_ms. route_ms wall-clock is dominated by the
        # slowest tile; mean shows work balance; min shows the lightest tile.
        routing_min = load_floats(prof, "routing_min")
        proj_min = load_floats(prof, "proj_min")
        sort_min = load_floats(prof, "sort_min")
        routing_mean_cyc = load_floats(prof, "routing_mean")
        proj_mean_cyc = load_floats(prof, "proj_mean")
        sort_mean_cyc = load_floats(prof, "sort_mean")
        routing_max = load_floats(prof, "routing_max")
        proj_max = load_floats(prof, "proj_max")
        sort_max = load_floats(prof, "sort_max")
        visible = load_floats(prof, "total_visible")
        powers = load_floats(pwr, "power_w") if os.path.isfile(pwr) else []
        powers = [p for p in powers if p > 1.0]

        # Drop first 30 frames of timing to ignore the channel-saturation transient.
        warmup = 30
        def tail(xs): return xs[warmup:] if len(xs) > warmup else xs

        rows.append({
            "scene": scene,
            "frames": len(total_ms),
            "total_ms_mean": statistics.mean(tail(total_ms)) if tail(total_ms) else float("nan"),
            "total_ms_std": statistics.stdev(tail(total_ms)) if len(tail(total_ms)) > 1 else 0.0,
            "fps_mean": (1000.0 / statistics.mean(tail(total_ms))) if tail(total_ms) else float("nan"),
            "route_ms_mean":    statistics.mean(tail(route_ms)) if tail(route_ms) else float("nan"),
            "blend_ms_mean":    statistics.mean(tail(blend_ms)) if tail(blend_ms) else float("nan"),
            "exchange_ms_mean": statistics.mean(tail(exchange_ms)) if tail(exchange_ms) else float("nan"),
            "routing_min": statistics.mean(tail(routing_min)) if tail(routing_min) else float("nan"),
            "proj_min":    statistics.mean(tail(proj_min)) if tail(proj_min) else float("nan"),
            "sort_min":    statistics.mean(tail(sort_min)) if tail(sort_min) else float("nan"),
            "routing_mean": statistics.mean(tail(routing_mean_cyc)) if tail(routing_mean_cyc) else float("nan"),
            "proj_mean":    statistics.mean(tail(proj_mean_cyc)) if tail(proj_mean_cyc) else float("nan"),
            "sort_mean":    statistics.mean(tail(sort_mean_cyc)) if tail(sort_mean_cyc) else float("nan"),
            "routing_max": statistics.mean(tail(routing_max)) if tail(routing_max) else float("nan"),
            "proj_max":    statistics.mean(tail(proj_max)) if tail(proj_max) else float("nan"),
            "sort_max":    statistics.mean(tail(sort_max)) if tail(sort_max) else float("nan"),
            "visible_mean":     statistics.mean(visible) if visible else float("nan"),
            "power_w_mean":     statistics.mean(powers) if powers else float("nan"),
            "power_w_max":      max(powers) if powers else float("nan"),
            "power_samples":    len(powers),
        })

    # Two-line table: wall-clock phases (host-timed) + per-tile subphase
    # min / mean / max across 1440 tiles. route_ms wall-clock is dominated by
    # the slowest tile; max is the bottleneck view, mean shows balance,
    # min shows the lightest tile.
    fmt = ("{:>10s}  {:>6s}  {:>9s}  {:>7s}  {:>7s}  | {:>9s}  {:>9s}  {:>9s}  "
           "| {:>16s}  {:>16s}  {:>16s}  | {:>10s}  {:>8s}  {:>7s}")
    rfmt = ("{:>10s}  {:>6d}  {:>9.2f}  {:>7.2f}  {:>7.2f}  | {:>9.2f}  {:>9.2f}  {:>9.2f}  "
            "| {:>16s}  {:>16s}  {:>16s}  | {:>10.0f}  {:>8.2f}  {:>7.2f}")

    def mmm(mn, mean, mx):
        return f"{mn:.2f}/{mean:.2f}/{mx:.2f}"

    print(fmt.format("scene", "frames", "total_ms", "fps", "±std",
                     "route_ms", "blend_ms", "exch_ms",
                     "rout min/mn/max", "proj min/mn/max", "sort min/mn/max",
                     "visible", "power_W", "peak_W"))
    print("-" * 195)
    for r in rows:
        print(rfmt.format(r["scene"], r["frames"], r["total_ms_mean"], r["fps_mean"], r["total_ms_std"],
                          r["route_ms_mean"], r["blend_ms_mean"], r["exchange_ms_mean"],
                          mmm(r["routing_min"], r["routing_mean"], r["routing_max"]),
                          mmm(r["proj_min"],    r["proj_mean"],    r["proj_max"]),
                          mmm(r["sort_min"],    r["sort_mean"],    r["sort_max"]),
                          r["visible_mean"], r["power_w_mean"], r["power_w_max"]))

    if rows:
        def avg(k): return statistics.mean(r[k] for r in rows if not math.isnan(r[k]))
        print("-" * 195)
        print(rfmt.format("avg", 0, avg("total_ms_mean"), avg("fps_mean"), 0.0,
                          avg("route_ms_mean"), avg("blend_ms_mean"), avg("exchange_ms_mean"),
                          mmm(avg("routing_min"), avg("routing_mean"), avg("routing_max")),
                          mmm(avg("proj_min"),    avg("proj_mean"),    avg("proj_max")),
                          mmm(avg("sort_min"),    avg("sort_mean"),    avg("sort_max")),
                          avg("visible_mean"), avg("power_w_mean"), avg("power_w_max")))
        print()
        print("  route_ms/blend_ms/exch_ms: wall-clock per phase (host-timed).")
        print("  rout/proj/sort min/mean/max: per-tile cycle-counter ms across 1440 tiles.")
        print("  route_ms ≤ max(rout)+max(proj)+max(sort)  (slowest tile may differ per subphase).")

        # Compact per-scene summary (paper-quote-friendly).
        print()
        print("=" * 130)
        def compact_lines(label, r):
            print(f"{label}")
            print(f"  total_ms:{r['total_ms_mean']:6.2f}   fps:{r['fps_mean']:6.2f}   exchange:{r['exchange_ms_mean']:5.2f}")
            print(f"  Blend:{r['blend_ms_mean']:6.2f}   "
                  f"Routing(min/mean/max):{r['routing_min']:5.2f}/{r['routing_mean']:5.2f}/{r['routing_max']:6.2f}   "
                  f"Projection(min/mean/max):{r['proj_min']:5.2f}/{r['proj_mean']:5.2f}/{r['proj_max']:6.2f}   "
                  f"Sort(min/mean/max):{r['sort_min']:5.2f}/{r['sort_mean']:5.2f}/{r['sort_max']:5.2f}")
            print(f"  power_W:{r['power_w_mean']:5.2f}   peak_W:{r['power_w_max']:5.2f}")
            print()
        for r in rows:
            compact_lines(r["scene"] + ":", r)
        # Cross-scene mean
        avg_row = {k: avg(k) for k in
                   ("total_ms_mean", "fps_mean", "exchange_ms_mean", "blend_ms_mean",
                    "routing_min", "routing_mean", "routing_max",
                    "proj_min", "proj_mean", "proj_max",
                    "sort_min", "sort_mean", "sort_max",
                    "power_w_mean", "power_w_max")}
        compact_lines("average:", avg_row)

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
