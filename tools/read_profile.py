#!/usr/bin/env python3
"""Extract per-program-step cycle counts from a Poplar profile.

Usage:
    python3 tools/read_profile.py profile/debug.cbor
"""

import sys
import pva

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile/debug.cbor"
    print(f"Loading {path}...")
    report = pva.openReport(path)

    target = report.compilation.target
    print(f"Target: {target.numTiles} tiles, {target.clockFrequency/1e6:.0f} MHz tile clock")
    print(f"Total tiles used: {report.compilation.tiles.numTiles}")

    # Memory summary
    mem = report.compilation.tiles
    print(f"Memory: max {mem.maxMemory.total.includingGaps/(1024):.1f} KB/tile")

    print(f"\n{'Program Step':<45} {'Cycles':>12} {'ms':>10}")
    print("-" * 70)

    for step in report.execution.runs:
        for prog in step.programs:
            name = prog.name if hasattr(prog, 'name') else str(prog)
            cycles = prog.cycles.total if hasattr(prog.cycles, 'total') else 0
            ms = cycles / (target.clockFrequency) * 1000
            print(f"{name:<45} {cycles:>12,} {ms:>10.4f}")

    # Try to get per-tile breakdown for the compute set
    print("\n--- Compute Set Details ---")
    for cs in report.compilation.computeSets:
        name = cs.name if hasattr(cs, 'name') else str(cs)
        cycles_from = getattr(cs, 'cyclesFrom', None)
        cycles_to = getattr(cs, 'cyclesTo', None)
        if cycles_from is not None and cycles_to is not None:
            print(f"  {name}: {cycles_from}-{cycles_to} cycles")
        else:
            print(f"  {name}")

if __name__ == "__main__":
    main()
