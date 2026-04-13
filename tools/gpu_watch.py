#!/usr/bin/env python3
"""
Watch the IPU server's paired-shots directory and auto-produce a matching
GPU reference render (via diff-gaussian-rasterization) for every new sidecar.

Typical workflow:
  1. Start the IPU server in the container:
       ./build/src/main/splat --input data/salad.ply --ui-port 5000 \\
           --paired-shots-dir paired_shots
  2. Start this watcher on the host (where CUDA is available):
       source .venv-gpu/bin/activate
       python3 tools/gpu_watch.py
  3. Click Screenshot in the remote UI — within ~1 second you get:
       paired_shots/screenshot-YYYYMMDD-HHMMSS.png        (IPU render)
       paired_shots/screenshot-YYYYMMDD-HHMMSS.json       (pose metadata)
       paired_shots/screenshot-YYYYMMDD-HHMMSS_gpu.png    (GPU reference)

The server writes the JSON file AFTER the PNG to avoid races.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def render_pair(json_path: Path, renderer: Path, python: str) -> bool:
    """Run the GPU reference renderer for one sidecar. Returns True on success."""
    try:
        meta = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"  [!] could not read {json_path.name}: {exc}")
        return False

    try:
        vm = meta["view_matrix"]            # 16 floats, GLM column-major (COLMAP view)
        fov_half_rad = meta["fov_half_rad"] # server stores half-FOV in radians
        ply = meta["ply"]
    except KeyError as exc:
        print(f"  [!] {json_path.name} is missing {exc}")
        return False

    if len(vm) != 16:
        print(f"  [!] view_matrix should have 16 floats, got {len(vm)}")
        return False

    # Resolve the PLY path. Two common issues:
    #  (1) old sidecars stored relative paths (../data/...)
    #  (2) the server runs in a Docker container where /home/<user>/... is
    #      bind-mounted to /nethome/<user>/... on the host, so absolute
    #      container paths are invalid on the host where the watcher runs.
    import os
    ply_path = Path(ply)

    def rewrite_container_to_host(p: Path) -> Path:
        # /home/<user>/... -> /nethome/<user>/... (common IPU-container mount)
        parts = p.parts
        if len(parts) >= 3 and parts[0] == "/" and parts[1] == "home":
            return Path("/nethome", *parts[2:])
        return p

    if not ply_path.exists():
        candidates = []
        if ply_path.is_absolute():
            candidates.append(rewrite_container_to_host(ply_path))
        else:
            candidates += [
                renderer.parent.parent / ply,       # repo root (renderer in tools/)
                json_path.parent.parent / ply,      # one above sidecar dir
                json_path.parent / ply,             # next to sidecar
                Path.cwd() / ply,                   # cwd
            ]
        for c in candidates:
            try:
                c = c.resolve(strict=False)
            except Exception:
                continue
            if c.exists():
                ply = str(c)
                ply_path = c
                print(f"  resolved ply -> {ply}")
                break
        else:
            print(f"  [!] could not locate '{ply}'; tried: {[str(c) for c in candidates]}")
            return False

    fov_deg = fov_half_rad * 2.0 * 180.0 / math.pi
    vm_str = " ".join(f"{v:.10g}" for v in vm)
    out_png = json_path.with_name(json_path.stem + "_gpu.png")

    cmd = [
        python, str(renderer),
        "--ply", ply,
        "--out", str(out_png),
        "--fov-deg", f"{fov_deg:.4f}",
        "--view-matrix", vm_str,
    ]
    print(f"  running: fov={fov_deg:.2f}° ply={ply}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"  [!] renderer timed out")
        return False
    if r.returncode != 0:
        print(f"  [!] renderer failed ({r.returncode}):")
        print(r.stderr.strip()[-1000:])
        return False
    print(f"  -> {out_png.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="paired_shots",
                    help="Directory to watch (must match the server's --paired-shots-dir).")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Poll interval in seconds.")
    ap.add_argument("--renderer", default="tools/render_gpu_dgr.py",
                    help="Path to the GPU reference renderer script.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current).")
    ap.add_argument("--once", action="store_true",
                    help="Render everything present and exit (no polling).")
    args = ap.parse_args()

    watched = Path(args.dir).resolve()
    renderer = Path(args.renderer).resolve()
    if not renderer.exists():
        print(f"renderer not found: {renderer}", file=sys.stderr)
        return 1

    watched.mkdir(parents=True, exist_ok=True)
    print(f"gpu_watch: watching {watched}")
    print(f"gpu_watch: using renderer {renderer}")

    processed = set(p for p in watched.glob("screenshot-*_gpu.png"))
    # Consider already-processed any sidecar whose _gpu.png sibling exists.
    seen = {p.with_suffix(".json") for p in processed}

    while True:
        for j in sorted(watched.glob("screenshot-*.json")):
            if j in seen:
                continue
            out_png = j.with_name(j.stem + "_gpu.png")
            if out_png.exists():
                seen.add(j)
                continue
            print(f"[{j.name}]")
            if render_pair(j, renderer, args.python):
                seen.add(j)
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
