#!/usr/bin/env python3
"""Extract per-step cycle counts from a Poplar profile.

Usage:
    python3 tools/read_profile.py profile/ipu_utils_engine/profile.pop
"""

import sys
import pva

def classify_step(step):
    """Classify an execution step into a human-readable phase."""
    prog_type = type(step.program).__name__
    prog_name = ""
    try:
        prog_name = step.program.name
    except:
        pass

    if "OnTileExecute" in prog_type:
        if "project" in prog_name or "project" in str(prog_name):
            return "GSplat Compute"
        elif "Copy" in prog_name or "copy" in prog_name:
            return "On-tile Copy"
        else:
            return f"Compute({prog_name})"
    elif "DoExchange" in prog_type:
        return "NEWS Exchange"
    elif "StreamCopy" in prog_type:
        return "Stream I/O"
    elif prog_type == "Program":
        if "Sync" in prog_name:
            return "Sync"
        elif "Internal" in prog_name:
            return "Internal Sync"
        else:
            return f"Program({prog_name})"
    else:
        return prog_type

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile/ipu_utils_engine/profile.pop"
    print(f"Loading {path}...")
    report = pva.openReport(path)

    target = report.compilation.target
    clock_hz = target.clockFrequency
    print(f"Target: {target.numTiles} tiles, {clock_hz/1e6:.0f} MHz tile clock")

    steps = list(report.execution.steps)

    # Raw step table
    print(f"\n{'Step':>4} {'Type':<25} {'Max Cycles':>12} {'ms':>10} {'Mean Cycles':>12} {'ms':>10}")
    print("-" * 80)

    for i, step in enumerate(steps):
        prog_type = type(step.program).__name__
        cycles = list(step.cyclesByTile)
        if not cycles:
            continue
        max_c = max(cycles)
        mean_c = sum(cycles) / len(cycles)
        max_ms = max_c / clock_hz * 1000
        mean_ms = mean_c / clock_hz * 1000

        label = classify_step(step)
        if len(label) > 25:
            label = label[:22] + "..."

        print(f"{i:>4} {label:<25} {max_c:>12,} {max_ms:>10.4f} {mean_c:>12,.0f} {mean_ms:>10.4f}")

    # Identify frame boundaries: each frame starts with Stream I/O for mv/write
    # Group runs by looking at execution.runs
    runs = list(report.execution.runs)
    print(f"\n=== {len(runs)} Execution Runs ===")

    for ri, run in enumerate(runs):
        try:
            ipu_cycles = list(run.cyclesByIpu)
            if ipu_cycles:
                max_c = max(ipu_cycles)
                print(f"  Run {ri}: {max_c:,} cycles ({max_c/clock_hz*1000:.4f} ms)")
        except:
            pass

        run_steps = list(run.steps)
        # Aggregate by phase within this run
        phase_cycles = {}
        for step in run_steps:
            phase = classify_step(step)
            cycles = list(step.cyclesByTile)
            if not cycles:
                continue
            max_c = max(cycles)
            if phase not in phase_cycles:
                phase_cycles[phase] = 0
            phase_cycles[phase] += max_c

        if phase_cycles:
            total = sum(phase_cycles.values())
            print(f"  Phase breakdown (max tile cycles):")
            for phase, cyc in sorted(phase_cycles.items(), key=lambda x: -x[1]):
                ms = cyc / clock_hz * 1000
                pct = cyc / total * 100 if total > 0 else 0
                print(f"    {phase:<25} {cyc:>12,} ({ms:>8.4f} ms, {pct:>5.1f}%)")
            print(f"    {'TOTAL':<25} {total:>12,} ({total/clock_hz*1000:>8.4f} ms)")
        print()

    # Summary: compute-only breakdown across all runs
    print("=== On-Device Compute Summary (excluding stream I/O and sync) ===")
    compute_phases = {}
    for step in steps:
        phase = classify_step(step)
        if phase in ("Stream I/O", "Sync", "Internal Sync"):
            continue
        cycles = list(step.cyclesByTile)
        if not cycles:
            continue
        max_c = max(cycles)
        mean_c = sum(cycles) / len(cycles)
        if phase not in compute_phases:
            compute_phases[phase] = {"max_total": 0, "mean_total": 0, "count": 0}
        compute_phases[phase]["max_total"] += max_c
        compute_phases[phase]["mean_total"] += mean_c
        compute_phases[phase]["count"] += 1

    n_frames = max(1, compute_phases.get("GSplat Compute", {}).get("count", 1))
    print(f"  Frames profiled: {n_frames}")
    print(f"\n  {'Phase':<25} {'Total Max':>12} {'Per-frame Max':>14} {'Per-frame ms':>12}")
    print(f"  {'-'*65}")
    for phase, data in sorted(compute_phases.items(), key=lambda x: -x[1]["max_total"]):
        per_frame = data["max_total"] / n_frames
        per_frame_ms = per_frame / clock_hz * 1000
        print(f"  {phase:<25} {data['max_total']:>12,} {per_frame:>14,.0f} {per_frame_ms:>12.4f}")

if __name__ == "__main__":
    main()
