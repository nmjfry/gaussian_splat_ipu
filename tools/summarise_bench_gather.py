#!/usr/bin/env python3
"""Summarise multiSlice-gather benchmark results across scenes.

Reads <root>/<scene>/benchmark_profile.csv (gather columns:
discovery_ms, stream_ms, mvp_ms, gather_ms, project_ms, blend_ms,
readback_ms, total_ms, assigned) plus optional
power_samples.csv, prints a table, and writes <root>/summary.csv.
"""
import csv, os, sys, math, statistics


def load_floats(path, col):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[col])
            except (ValueError, KeyError):
                continue
            if not math.isnan(v):
                out.append(v)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: summarise_bench_gather.py <bench_results_dir>")
        sys.exit(1)
    root = sys.argv[1]
    scenes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    warmup = 30

    def tail(xs):
        return xs[warmup:] if len(xs) > warmup else xs

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    rows = []
    for scene in scenes:
        d = os.path.join(root, scene)
        prof = os.path.join(d, "benchmark_profile.csv")
        if not os.path.isfile(prof):
            continue
        disc = tail(load_floats(prof, "discovery_ms"))
        strm = tail(load_floats(prof, "stream_ms"))
        mvp  = tail(load_floats(prof, "mvp_ms"))
        gath = tail(load_floats(prof, "gather_ms"))
        proj = tail(load_floats(prof, "project_ms"))
        blen = tail(load_floats(prof, "blend_ms"))
        rb   = tail(load_floats(prof, "readback_ms"))
        tot  = tail(load_floats(prof, "total_ms"))
        asg  = tail(load_floats(prof, "assigned"))
        powers = [p for p in load_floats(os.path.join(d, "power_samples.csv"), "power_w") if p > 1.0]
        rows.append({
            "scene": scene,
            "frames": len(load_floats(prof, "total_ms")),
            "total_ms_mean": mean(tot),
            "total_ms_std": statistics.stdev(tot) if len(tot) > 1 else 0.0,
            "fps_mean": (1000.0 / mean(tot)) if tot else float("nan"),
            "discovery_ms_mean": mean(disc),
            "stream_ms_mean": mean(strm),
            "mvp_ms_mean": mean(mvp),
            "gather_ms_mean": mean(gath),
            "project_ms_mean": mean(proj),
            "blend_ms_mean": mean(blen),
            "readback_ms_mean": mean(rb),
            "assigned_mean": mean(asg),
            "power_w_mean": mean(powers) if powers else float("nan"),
            "power_w_max": max(powers) if powers else float("nan"),
            "power_samples": len(powers),
        })

    hdr = ("{:>10s}  {:>6s}  {:>9s}  {:>6s}  {:>10s}  {:>8s}  {:>6s}  {:>8s}  {:>8s}  {:>7s}  {:>9s}  {:>9s}  {:>8s}")
    rfmt = ("{:>10s}  {:>6d}  {:>9.2f}  {:>6.2f}  {:>10.2f}  {:>8.2f}  {:>6.2f}  {:>8.2f}  {:>8.2f}  {:>7.2f}  {:>9.2f}  {:>9.0f}  {:>8.2f}")
    print(hdr.format("scene", "frames", "total_ms", "fps", "discovery", "stream", "mvp",
                     "gather", "project", "blend", "readback", "assigned", "power_W"))
    print("-" * 140)
    for r in rows:
        print(rfmt.format(r["scene"], r["frames"], r["total_ms_mean"], r["fps_mean"],
                          r["discovery_ms_mean"], r["stream_ms_mean"], r["mvp_ms_mean"],
                          r["gather_ms_mean"], r["project_ms_mean"], r["blend_ms_mean"],
                          r["readback_ms_mean"], r["assigned_mean"], r["power_w_mean"]))
    if rows:
        def avg(k):
            vs = [r[k] for r in rows if not math.isnan(r[k])]
            return statistics.mean(vs) if vs else float("nan")
        print("-" * 140)
        print(rfmt.format("avg", 0, avg("total_ms_mean"), avg("fps_mean"),
                          avg("discovery_ms_mean"), avg("stream_ms_mean"), avg("mvp_ms_mean"),
                          avg("gather_ms_mean"), avg("project_ms_mean"), avg("blend_ms_mean"),
                          avg("readback_ms_mean"), avg("assigned_mean"), avg("power_w_mean")))
        out_csv = os.path.join(root, "summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
