#!/usr/bin/env python3
"""Join IPU and GPU benchmark CSVs into a FPS/W comparison table.

Usage:
    python3 tools/make_fpsw_table.py \
        --ipu ipu_benchmark_results.csv \
        --gpu gpu_benchmark_results.csv \
        --output fpsw_table.txt
"""

import argparse, csv, os


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            scene = os.path.basename(row["scene"])
            rows[scene] = row
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ipu", default="ipu_benchmark_results.csv")
    p.add_argument("--gpu", default="gpu_benchmark_results.csv")
    p.add_argument("--output", default="fpsw_table.txt")
    args = p.parse_args()

    ipu = load_csv(args.ipu) if os.path.exists(args.ipu) else {}
    gpu = load_csv(args.gpu) if os.path.exists(args.gpu) else {}

    scenes = sorted(set(list(ipu.keys()) + list(gpu.keys())))

    header = f"{'Scene':<25} {'IPU FPS':>8} {'IPU W':>7} {'IPU FPS/W':>10}  " \
             f"{'GPU FPS':>8} {'GPU W':>7} {'GPU FPS/W':>10}  {'Ratio':>7}"
    sep = "-" * len(header)

    lines = [header, sep]
    for s in scenes:
        i = ipu.get(s, {})
        g = gpu.get(s, {})
        ifps = i.get("fps", "-")
        iw   = i.get("watts", "-")
        ifpw = i.get("fps_per_watt", "-")
        gfps = g.get("fps", "-")
        gw   = g.get("watts", "-")
        gfpw = g.get("fps_per_watt", "-")

        ratio = "-"
        try:
            ratio = f"{float(ifpw) / float(gfpw):.2f}x"
        except (ValueError, ZeroDivisionError):
            pass

        lines.append(
            f"{s:<25} {ifps:>8} {iw:>7} {ifpw:>10}  "
            f"{gfps:>8} {gw:>7} {gfpw:>10}  {ratio:>7}"
        )

    table = "\n".join(lines)
    print(table)

    with open(args.output, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
