#!/usr/bin/env python3
"""Parse nsys kernel CSVs into the 4-stage GPU 3DGS breakdown, mean over scenes.

Buckets (DGR pipeline):
  Projection     <- preprocessCUDA
  Data movement  <- duplicateWithKeys + identifyTileRanges + cub DeviceScan (offsets)
  Sorting        <- cub DeviceRadixSort* kernels
  Alpha blending <- renderCUDA

Framework/wrapper kernels (torch fills, clamp, cublas gemv for the per-frame
projection-matrix matmul) are reported separately as 'framework' and excluded
from the pipeline breakdown — they are PyTorch-wrapper artifacts, not the
rasteriser. Per-frame = total kernel time / (#renderCUDA launches).
"""
import csv, subprocess, sys, os

SCENES = ["chairs", "pringles", "salad", "sloth"]
SCRATCH = os.path.dirname(os.path.abspath(__file__))


def classify(name):
    n = name.lower()
    if "rendercuda" in n:
        return "blend"
    if "preprocesscuda" in n:
        return "projection"
    if "duplicatewithkeys" in n or "identifytilerange" in n:
        return "datamove"
    if "devicescan" in n:               # InclusiveSum offsets for duplication
        return "datamove"
    if "radixsort" in n:
        return "sort"
    # framework / wrapper noise
    if "elementwise" in n or "gemm" in n or "gemv" in n or "fill" in n or "clamp" in n:
        return "framework"
    return "other"


def parse_scene(scene):
    rep = os.path.join(SCRATCH, f"{scene}_prof.nsys-rep")
    out = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", rep],
        capture_output=True, text=True)
    lines = out.stdout.splitlines()
    # find the header row
    start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
    reader = csv.DictReader(lines[start:])
    buckets = {"projection": 0.0, "datamove": 0.0, "sort": 0.0, "blend": 0.0,
               "framework": 0.0, "other": 0.0}
    frames = None
    for row in reader:
        name = row["Name"]
        total_ns = float(row["Total Time (ns)"])
        inst = int(row["Instances"])
        b = classify(name)
        buckets[b] += total_ns
        if "rendercuda" in name.lower():
            frames = inst
    # per-frame ms
    per_frame = {k: (v / frames) / 1e6 for k, v in buckets.items()}
    return per_frame, frames


def main():
    print(f"{'scene':>10} {'proj':>7} {'datamv':>7} {'sort':>7} {'blend':>7} "
          f"{'pipe_tot':>9} {'fps':>6}  | {'fw':>6} frames")
    agg = {"projection": [], "datamove": [], "sort": [], "blend": [], "framework": []}
    for s in SCENES:
        pf, frames = parse_scene(s)
        pipe = pf["projection"] + pf["datamove"] + pf["sort"] + pf["blend"]
        fps = 1000.0 / pipe
        print(f"{s:>10} {pf['projection']:7.4f} {pf['datamove']:7.4f} "
              f"{pf['sort']:7.4f} {pf['blend']:7.4f} {pipe:9.4f} {fps:6.0f}  | "
              f"{pf['framework']:6.4f} {frames}")
        for k in agg:
            agg[k].append(pf[k])
    print("-" * 78)
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    pipe = mean["projection"] + mean["datamove"] + mean["sort"] + mean["blend"]
    print(f"{'MEAN':>10} {mean['projection']:7.4f} {mean['datamove']:7.4f} "
          f"{mean['sort']:7.4f} {mean['blend']:7.4f} {pipe:9.4f} {1000/pipe:6.0f}  | "
          f"{mean['framework']:6.4f}")
    print()
    print("Mean per-frame (ms) and % of pipeline:")
    for k, label in [("projection", "Projection"), ("datamove", "Data movement"),
                     ("sort", "Sorting"), ("blend", "Alpha blending")]:
        print(f"  {label:<16} {mean[k]*1000:7.1f} us   {100*mean[k]/pipe:5.1f}%")
    print(f"  {'PIPELINE TOTAL':<16} {pipe*1000:7.1f} us")
    print(f"  (framework/wrapper overhead, excluded: {mean['framework']*1000:.1f} us/frame)")

    # emit python-literal means for the plot script
    print("\n# for plot:")
    print(f"proj_ms={mean['projection']:.4f}; datamove_ms={mean['datamove']:.4f}; "
          f"sort_ms={mean['sort']:.4f}; blend_ms={mean['blend']:.4f}")


if __name__ == "__main__":
    main()
