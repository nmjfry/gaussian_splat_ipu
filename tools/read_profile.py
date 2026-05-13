#!/usr/bin/env python3
"""Extract per-program-step cycle counts from a Poplar profile.

Usage:
    python3 tools/read_profile.py profile/ipu_utils_engine/profile.pop
"""

import sys
import pva

def dump_attrs(obj, prefix, depth=0):
    """Recursively print non-private attributes and their types."""
    if depth > 2:
        return
    for attr in sorted(dir(obj)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obj, attr)
            t = type(val).__name__
            if callable(val) and not isinstance(val, (int, float, str, bool)):
                print(f"  {'  '*depth}{prefix}.{attr} -> {t} (callable)")
            elif t in ('int', 'float', 'str', 'bool'):
                print(f"  {'  '*depth}{prefix}.{attr} = {val}")
            else:
                print(f"  {'  '*depth}{prefix}.{attr} -> {t}")
        except Exception as e:
            print(f"  {'  '*depth}{prefix}.{attr} -> ERROR: {e}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile/ipu_utils_engine/profile.pop"
    print(f"Loading {path}...")
    report = pva.openReport(path)

    target = report.compilation.target
    clock_hz = target.clockFrequency
    print(f"Target: {target.numTiles} tiles, {clock_hz/1e6:.0f} MHz tile clock")

    # Total execution cycles
    try:
        total = report.execution.totalCycles
        print(f"Total execution cycles: {total:,} ({total/clock_hz*1000:.4f} ms)")
    except Exception as e:
        print(f"(totalCycles unavailable: {e})")

    # Explore execution.steps
    print(f"\n=== Execution Steps ===")
    try:
        for i, step in enumerate(report.execution.steps):
            name = getattr(step, 'name', f'step_{i}')
            print(f"\n  Step {i}: {name}")
            for attr in sorted(dir(step)):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(step, attr)
                    t = type(val).__name__
                    if t in ('int', 'float', 'str', 'bool'):
                        if t == 'float' and 'cycle' in attr.lower():
                            print(f"    {attr} = {val:,.0f} ({val/clock_hz*1000:.4f} ms)")
                        else:
                            print(f"    {attr} = {val}")
                    else:
                        print(f"    {attr} -> {t}")
                except Exception as e:
                    print(f"    {attr} -> ERROR: {e}")
            if i > 20:
                print("  ... (truncated)")
                break
    except Exception as e:
        print(f"(steps unavailable: {e})")

    # Explore execution.runs
    print(f"\n=== Execution Runs ===")
    try:
        for i, run in enumerate(report.execution.runs):
            name = getattr(run, 'name', f'run_{i}')
            print(f"\n  Run {i}: {name}")
            for attr in sorted(dir(run)):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(run, attr)
                    t = type(val).__name__
                    if t in ('int', 'float', 'str', 'bool'):
                        print(f"    {attr} = {val}")
                    else:
                        print(f"    {attr} -> {t}")
                except Exception as e:
                    print(f"    {attr} -> ERROR: {e}")
            if i > 5:
                print("  ... (truncated)")
                break
    except Exception as e:
        print(f"(runs unavailable: {e})")

    # Explore execution.blocks
    print(f"\n=== Execution Blocks ===")
    try:
        for i, block in enumerate(report.execution.blocks):
            name = getattr(block, 'name', f'block_{i}')
            print(f"\n  Block {i}: {name}")
            for attr in sorted(dir(block)):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(block, attr)
                    t = type(val).__name__
                    if t in ('int', 'float', 'str', 'bool'):
                        if 'cycle' in attr.lower():
                            print(f"    {attr} = {val:,.0f} ({float(val)/clock_hz*1000:.4f} ms)")
                        else:
                            print(f"    {attr} = {val}")
                    else:
                        print(f"    {attr} -> {t}")
                except Exception as e:
                    print(f"    {attr} -> ERROR: {e}")
            if i > 20:
                print("  ... (truncated)")
                break
    except Exception as e:
        print(f"(blocks unavailable: {e})")

    # Compute sets with cycle info
    print(f"\n=== Compute Sets ===")
    try:
        for cs in report.compilation.computeSets:
            name = str(cs.name) if hasattr(cs, 'name') else str(cs)
            print(f"\n  {name}:")
            for attr in sorted(dir(cs)):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(cs, attr)
                    t = type(val).__name__
                    if t in ('int', 'float', 'str', 'bool'):
                        print(f"    {attr} = {val}")
                    else:
                        print(f"    {attr} -> {t}")
                except Exception as e:
                    print(f"    {attr} -> ERROR: {e}")
    except Exception as e:
        print(f"(compute sets unavailable: {e})")

    # Programs with cycle info
    print(f"\n=== Programs (with names) ===")
    try:
        for prog in report.compilation.programs:
            name = str(prog.name) if hasattr(prog, 'name') else str(prog)
            if not name.strip():
                continue
            print(f"\n  {name}:")
            for attr in sorted(dir(prog)):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(prog, attr)
                    t = type(val).__name__
                    if t in ('int', 'float', 'str', 'bool'):
                        print(f"    {attr} = {val}")
                    else:
                        print(f"    {attr} -> {t}")
                except Exception as e:
                    print(f"    {attr} -> ERROR: {e}")
    except Exception as e:
        print(f"(programs unavailable: {e})")

if __name__ == "__main__":
    main()
