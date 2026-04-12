#!/usr/bin/env python3
"""
Download Dylan Ebert's tiny 3DGS dataset from Hugging Face and convert each
sample to a PLY file compatible with the IPU splat renderer.

The IPU renderer only consumes the DC spherical-harmonic coefficients
(f_dc_0/1/2) — all higher-order terms (f_rest_*) are stripped at conversion
time so the output is as small as possible.

Usage:
    python3 fetch_dylanebert_3dgs.py \\
        --output-dir data/tiny_3dgs \\
        [--limit 5] [--split train]

Dependencies:
    pip install datasets huggingface_hub numpy plyfile
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing 'datasets' — install with `pip install datasets`.")

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    sys.exit("Missing 'plyfile' — install with `pip install plyfile`.")


# The renderer's file_io.cpp loads exactly these fields from the PLY:
REQUIRED_FIELDS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


def _load_input_ply(path: str) -> PlyData:
    """Read a PLY file from disk. Raises on malformed input."""
    return PlyData.read(path)


def _strip_to_dc_only(ply: PlyData) -> PlyData:
    """Return a new PlyData with only the 14 fields the IPU renderer needs."""
    vertex = ply["vertex"]
    data = vertex.data

    missing = [f for f in REQUIRED_FIELDS if f not in data.dtype.names]
    if missing:
        raise RuntimeError(f"Input PLY is missing required fields: {missing}")

    # Build a new structured array with only the fields we want.
    dtype = [(name, "f4") for name in REQUIRED_FIELDS]
    out = np.empty(len(data), dtype=dtype)
    for name in REQUIRED_FIELDS:
        out[name] = data[name].astype(np.float32)

    element = PlyElement.describe(out, "vertex")
    return PlyData([element], text=False)


def _resolve_ply_path(sample, cache_dir: str) -> str | None:
    """
    Different HF datasets use different column names for the gaussian data.
    Try the obvious ones and return a filesystem path to a PLY.
    """
    # Common shapes:
    #   - {"file": "<local path>"}
    #   - {"ply": "<local path>"}
    #   - {"file": {"path": "<local path>"}}
    #   - The `datasets` library may auto-download into a cache; paths then
    #     point at something under ~/.cache/huggingface/
    for key in ("file", "ply", "path"):
        if key in sample:
            val = sample[key]
            if isinstance(val, dict) and "path" in val:
                return val["path"]
            if isinstance(val, str) and val.endswith(".ply"):
                return val
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True,
                   help="Where to write the converted PLY files.")
    p.add_argument("--dataset", default="dylanebert/3dgs",
                   help="HF dataset id.")
    p.add_argument("--split", default="train",
                   help="Dataset split to pull from.")
    p.add_argument("--limit", type=int, default=0,
                   help="Max number of samples to convert (0 = all).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}' [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  {len(ds)} samples available.")

    written = 0
    for i, sample in enumerate(ds):
        if args.limit and written >= args.limit:
            break

        src_path = _resolve_ply_path(sample, cache_dir="")
        if src_path is None:
            # Dump the sample structure once so the user can see what key to use.
            if i == 0:
                print("WARN: no obvious 'file'/'ply' key in sample. Keys:",
                      list(sample.keys()))
            continue

        try:
            ply_in = _load_input_ply(src_path)
            ply_out = _strip_to_dc_only(ply_in)
        except Exception as exc:  # noqa: BLE001 - intentionally broad
            print(f"  [{i:04d}] skip {src_path}: {exc}")
            continue

        # Name the output after the sample id if one is available; otherwise
        # use the source filename.
        name = sample.get("id") or sample.get("name") \
               or Path(src_path).stem
        dst = out_dir / f"{name}.ply"
        ply_out.write(str(dst))
        n_gauss = len(ply_out["vertex"].data)
        print(f"  [{i:04d}] {src_path} -> {dst}  ({n_gauss} gaussians, "
              f"{dst.stat().st_size / 1024:.1f} KB)")
        written += 1

    print(f"Wrote {written} PLY file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
