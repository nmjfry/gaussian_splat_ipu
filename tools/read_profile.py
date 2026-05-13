#!/usr/bin/env python3
"""Extract per-step cycle counts from a Poplar profile.

Usage:
    python3 tools/read_profile.py profile/ipu_utils_engine/profile.pop
"""

import sys
import pva

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile/ipu_utils_engine/profile.pop"
    print(f"Loading {path}...")
    report = pva.openReport(path)

    target = report.compilation.target
    clock_hz = target.clockFrequency
    print(f"Target: {target.numTiles} tiles, {clock_hz/1e6:.0f} MHz tile clock")

    # Per-step cycle breakdown
    print(f"\n{'Step':>4} {'Type':<25} {'Max Cycles':>12} {'ms':>10} {'Mean Cycles':>12} {'ms':>10}")
    print("-" * 80)

    total_max = 0
    for i, step in enumerate(report.execution.steps):
        prog_type = type(step.program).__name__
        cycles = list(step.cyclesByTile)
        if not cycles:
            continue
        max_c = max(cycles)
        mean_c = sum(cycles) / len(cycles) if cycles else 0
        total_max += max_c
        max_ms = max_c / clock_hz * 1000
        mean_ms = mean_c / clock_hz * 1000

        # Try to get program name
        prog_name = ""
        try:
            prog_name = step.program.name
        except:
            pass
        label = f"{prog_type}"
        if prog_name:
            label = f"{prog_type}({prog_name})"
        if len(label) > 25:
            label = label[:22] + "..."

        print(f"{i:>4} {label:<25} {max_c:>12,} {max_ms:>10.4f} {mean_c:>12,.0f} {mean_ms:>10.4f}")

    print(f"\n{'Total (sum of max)':>30}: {total_max:>12,} {total_max/clock_hz*1000:>10.4f} ms")

    # Per-run breakdown (each run = one frame in benchmark)
    print(f"\n=== Runs (frames) ===")
    for i, run in enumerate(report.execution.runs):
        try:
            ipu_cycles = list(run.cyclesByIpu)
            if ipu_cycles:
                max_c = max(ipu_cycles)
                print(f"  Run {i}: {max_c:,} cycles ({max_c/clock_hz*1000:.4f} ms)")
        except Exception as e:
            print(f"  Run {i}: {e}")
        if i > 10:
            remaining = len(list(report.execution.runs)) - i - 1
            if remaining > 0:
                print(f"  ... ({remaining} more runs)")
            break

    # Compute set estimated cycles
    print(f"\n=== Compute Sets (estimated cycles) ===")
    for cs in report.compilation.computeSets:
        name = cs.name
        est = list(cs.estimatedCyclesByTile)
        if est:
            max_c = max(est)
            mean_c = sum(est) / len(est)
            print(f"  {name:<60} max={max_c:>10,} ({max_c/clock_hz*1000:.4f} ms)  mean={mean_c:>10,.0f}")
        else:
            print(f"  {name}: no estimated cycles")

    # Programs with estimated cycles
    print(f"\n=== Programs with estimated cycles ===")
    for prog in report.compilation.programs:
        name = prog.name
        if not name.strip():
            continue
        est = getattr(prog, 'estimatedCyclesByTile', None)
        if est is not None:
            est = list(est)
            if est and max(est) > 0:
                max_c = max(est)
                print(f"  {name:<60} max={max_c:>10,} ({max_c/clock_hz*1000:.4f} ms)")

if __name__ == "__main__":
    main()
