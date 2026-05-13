#!/usr/bin/env python3
"""Extract per-program-step cycle counts from a Poplar profile.

Usage:
    python3 tools/read_profile.py profile/debug.cbor
"""

import sys
import pva

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile/ipu_utils_engine/profile.pop"
    print(f"Loading {path}...")
    report = pva.openReport(path)

    target = report.compilation.target
    print(f"Target: {target.numTiles} tiles, {target.clockFrequency/1e6:.0f} MHz tile clock")

    # Safely print compilation info
    try:
        print(f"Total tiles used: {report.compilation.tiles.numTiles}")
        mem = report.compilation.tiles
        print(f"Memory: max {mem.maxMemory.total.includingGaps/1024:.1f} KB/tile")
    except Exception as e:
        print(f"(compilation tile info unavailable: {e})")

    # Execution steps
    print(f"\n{'Program':<50} {'Cycles':>12} {'ms':>10}")
    print("-" * 75)

    try:
        for step in report.execution.runs:
            for prog in step.programs:
                name = str(prog.name) if hasattr(prog, 'name') else str(prog)
                try:
                    cycles = prog.cycles.total
                except:
                    cycles = 0
                ms = cycles / target.clockFrequency * 1000
                print(f"{name:<50} {cycles:>12,} {ms:>10.4f}")
    except Exception as e:
        print(f"(execution runs unavailable: {e})")

    # Try alternative: programs list
    print(f"\n--- All Programs ---")
    try:
        for prog in report.compilation.programs:
            name = str(prog.name) if hasattr(prog, 'name') else str(prog)
            print(f"  {name}")
    except Exception as e:
        print(f"(programs list unavailable: {e})")

    # Compute sets
    print(f"\n--- Compute Sets ---")
    try:
        for cs in report.compilation.computeSets:
            name = str(cs.name) if hasattr(cs, 'name') else str(cs)
            print(f"  {name}")
    except Exception as e:
        print(f"(compute sets unavailable: {e})")

    # Dump top-level attributes to discover API
    print(f"\n--- Report attributes ---")
    for attr in dir(report):
        if not attr.startswith('_'):
            print(f"  report.{attr}")
    print(f"\n--- Execution attributes ---")
    for attr in dir(report.execution):
        if not attr.startswith('_'):
            print(f"  execution.{attr}")

if __name__ == "__main__":
    main()
