#!/usr/bin/env python3
"""
Render a 3DGS PLY on GPU using `gsplat`, matching the IPU renderer's math
(DC-only, identity view for zero pose). Intended for producing matched
baseline images alongside the IPU output for paper figures.

Example:
    # Single image at a fixed pose:
    python3 tools/render_gpu.py \\
        --ply data/tiny_3dgs/bonsai_7000.ply \\
        --out figures/bonsai_gpu.png \\
        --fov-deg 60 --width 1280 --height 720

    # Render every camera in the HF cameras.json:
    python3 tools/render_gpu.py \\
        --ply data/tiny_3dgs/bonsai_30000.ply \\
        --cameras data/tiny_3dgs/bonsai_cameras.json \\
        --out figures/bonsai_gpu/

    # Render at the same view matrix the IPU is currently using:
    #   (paste the "Dynamic view matrix" rows from the server log)
    python3 tools/render_gpu.py \\
        --ply data/tiny_3dgs/bonsai_7000.ply \\
        --out figures/bonsai_gpu_match.png \\
        --view-matrix " -1  0  0  0 \\
                         0  1  0  0 \\
                         0  0 -1  0 \\
                         0  0 -10  1"

Requires: torch (with CUDA), gsplat, plyfile, numpy.
    pip install torch gsplat plyfile numpy
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _activation(f_dc: np.ndarray,
                opacity_raw: np.ndarray,
                scale_raw: np.ndarray) -> tuple:
    """Apply the activations 3DGS applies at render time."""
    SH_C0 = 0.28209479177387814
    rgb = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    scale = np.exp(scale_raw)
    return rgb, opacity, scale


def _load_ply(path: str):
    """Load a 3DGS PLY and return torch tensors (on CUDA)."""
    from plyfile import PlyData
    import torch

    data = PlyData.read(path)["vertex"].data
    needed = ["x", "y", "z",
              "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity",
              "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [f for f in needed if f not in data.dtype.names]
    if missing:
        raise RuntimeError(f"PLY missing fields: {missing}")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1).astype(np.float32)
    opac = np.asarray(data["opacity"], dtype=np.float32)
    sc = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1).astype(np.float32)
    # 3DGS PLY convention is (w, x, y, z); gsplat expects (w, x, y, z) too.
    rot = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1).astype(np.float32)

    rgb, opacity, scale = _activation(f_dc, opac, sc)

    print(f"  loaded {len(xyz)} gaussians from {path}")
    dev = "cuda"
    return dict(
        means   = torch.from_numpy(xyz).to(dev),
        quats   = torch.from_numpy(rot).to(dev),
        scales  = torch.from_numpy(scale).to(dev),
        opacity = torch.from_numpy(opacity).to(dev),
        colors  = torch.from_numpy(rgb).to(dev),
    )


def _intrinsics(fov_y_deg: float, w: int, h: int) -> np.ndarray:
    """Pinhole K from vertical FOV + image size."""
    fy = 0.5 * h / math.tan(math.radians(fov_y_deg) / 2.0)
    fx = fy  # same pixel-space focal (aspect handled via horizontal fov elsewhere)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def _look_at(eye, target, up) -> np.ndarray:
    """Right-handed OpenGL-convention view matrix (world -> camera, -Z forward)."""
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float32), (eye, target, up))
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.stack([right, true_up, -fwd], axis=0)  # rows
    t = -R @ eye
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3,  3] = t
    return V


def _view_from_cameras_json(cam: dict) -> np.ndarray:
    """COLMAP-style camera entry from the HF dataset -> 4x4 world->camera."""
    R = np.asarray(cam["rotation"], dtype=np.float32)  # 3x3
    t = np.asarray(cam["position"], dtype=np.float32)  # 3
    # cameras.json stores camera-to-world; we want world-to-camera.
    # (Same convention as getWorld2View in the original 3DGS.)
    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, :3] = R
    C2W[:3,  3] = t
    V = np.linalg.inv(C2W)
    return V


def _render_one(gaussians, V: np.ndarray, K: np.ndarray, w: int, h: int,
                out_path: str):
    """Call gsplat and save a PNG."""
    import torch
    from gsplat import rasterization

    Vt = torch.from_numpy(V).unsqueeze(0).to("cuda")  # (1, 4, 4)
    Kt = torch.from_numpy(K).unsqueeze(0).to("cuda")  # (1, 3, 3)

    colors, _, _ = rasterization(
        means     = gaussians["means"],
        quats     = gaussians["quats"],
        scales    = gaussians["scales"],
        opacities = gaussians["opacity"],
        colors    = gaussians["colors"],
        viewmats  = Vt,
        Ks        = Kt,
        width     = w,
        height    = h,
        sh_degree = None,     # colours are already RGB — skip SH evaluation
        render_mode = "RGB",
    )
    img = (colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    from PIL import Image
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"  -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True, help="3DGS PLY file.")
    p.add_argument("--out", required=True,
                   help="Output PNG (single) or directory (with --cameras).")
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov-deg", type=float, default=60.0,
                   help="Vertical FOV in degrees (ignored with --cameras).")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--eye",    nargs=3, type=float, metavar=("X","Y","Z"),
                       help="Camera position (world space).")
    group.add_argument("--view-matrix",
                       help="4x4 world->camera matrix, row-major, 16 numbers.")
    group.add_argument("--cameras",
                       help="cameras.json from the HF dataset — render every entry.")

    p.add_argument("--target", nargs=3, type=float, metavar=("X","Y","Z"),
                   default=[0, 0, 0], help="Look-at point (with --eye).")
    p.add_argument("--up",     nargs=3, type=float, metavar=("X","Y","Z"),
                   default=[0, 1, 0], help="World up (with --eye).")
    p.add_argument("--limit", type=int, default=0,
                   help="Max cameras to render (with --cameras).")
    args = p.parse_args()

    print(f"Loading {args.ply} ...")
    gaussians = _load_ply(args.ply)

    if args.cameras:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.cameras) as f:
            cams = json.load(f)
        print(f"Rendering {len(cams)} cameras from {args.cameras} ...")
        for i, cam in enumerate(cams):
            if args.limit and i >= args.limit:
                break
            V = _view_from_cameras_json(cam)
            w = int(cam.get("width",  args.width))
            h = int(cam.get("height", args.height))
            fx = float(cam.get("fx", _intrinsics(args.fov_deg, w, h)[0, 0]))
            fy = float(cam.get("fy", fx))
            K = np.array([[fx, 0, w/2.0], [0, fy, h/2.0], [0, 0, 1]],
                         dtype=np.float32)
            name = cam.get("img_name", f"view_{i:04d}")
            _render_one(gaussians, V, K, w, h, str(out_dir / f"{name}.png"))
    else:
        if args.view_matrix:
            nums = [float(x) for x in args.view_matrix.split()]
            if len(nums) != 16:
                raise SystemExit("--view-matrix needs 16 numbers")
            V = np.asarray(nums, dtype=np.float32).reshape(4, 4)
        elif args.eye:
            V = _look_at(args.eye, args.target, args.up)
        else:
            # Default: 2× scene-diagonal behind the centroid, looking at centroid.
            import torch
            means = gaussians["means"]
            lo = means.amin(0).cpu().numpy()
            hi = means.amax(0).cpu().numpy()
            centroid = 0.5 * (lo + hi)
            diag = float(np.linalg.norm(hi - lo))
            eye = centroid + np.array([0, 0, diag])
            V = _look_at(eye, centroid, [0, 1, 0])
            print(f"  auto camera: eye={eye.tolist()}  centroid={centroid.tolist()}")
        K = _intrinsics(args.fov_deg, args.width, args.height)
        _render_one(gaussians, V, K, args.width, args.height, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
