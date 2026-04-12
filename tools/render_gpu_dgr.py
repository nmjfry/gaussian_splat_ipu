#!/usr/bin/env python3
"""GPU reference renderer using the official diff-gaussian-rasterization.
This is the exact rasterizer from the original 3DGS paper — the authoritative
baseline for paper comparisons.

Requires: torch + the inria diff-gaussian-rasterization extension
    TORCH_CUDA_ARCH_LIST="6.1" pip install \\
        git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import torch


def _activations(f_dc, opacity_raw, scale_raw):
    SH_C0 = 0.28209479177387814
    rgb = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    scale = np.exp(scale_raw)
    return rgb, opacity, scale


def load_ply(path):
    from plyfile import PlyData
    data = PlyData.read(path)["vertex"].data
    fields = ["x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity",
              "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    miss = [f for f in fields if f not in data.dtype.names]
    if miss: raise RuntimeError(f"PLY missing: {miss}")

    xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], -1).astype(np.float32)
    opac = np.asarray(data["opacity"], dtype=np.float32)
    sc = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], -1).astype(np.float32)
    rot = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], -1).astype(np.float32)
    rgb, opacity, scale = _activations(f_dc, opac, sc)
    print(f"  {len(xyz)} gaussians from {path}")

    return dict(
        means3D   = torch.from_numpy(xyz).cuda(),
        rotations = torch.from_numpy(rot).cuda(),
        scales    = torch.from_numpy(scale).cuda(),
        opacity   = torch.from_numpy(opacity).unsqueeze(-1).cuda(),
        colors    = torch.from_numpy(rgb).cuda(),
    )


def look_at(eye, target, up):
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float32), (eye, target, up))
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.stack([right, true_up, -fwd], axis=0)
    t = -R @ eye
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = t
    return V, eye


def make_projection(fov_y_deg, aspect, znear=0.01, zfar=100.0):
    """Original 3DGS projection (z_sign = +1, Python row-major)."""
    fovY = math.radians(fov_y_deg)
    fovX = 2.0 * math.atan(math.tan(fovY / 2.0) * aspect)
    tanY = math.tan(fovY / 2.0); tanX = math.tan(fovX / 2.0)
    top = tanY * znear; bottom = -top
    right = tanX * znear; left = -right
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2 * znear / (right - left)
    P[1, 1] = 2 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P, fovX, fovY


def render(g, V_np, width, height, fov_y_deg, out_path):
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )

    # Original 3DGS transposes both matrices before handing them to CUDA.
    aspect = width / float(height)
    P_np, fovX, fovY = make_projection(fov_y_deg, aspect)

    V = torch.from_numpy(V_np).cuda()
    P = torch.from_numpy(P_np).cuda()
    world_view_T = V.transpose(0, 1)
    full_proj_T  = (world_view_T.unsqueeze(0) @ P.transpose(0,1).unsqueeze(0)).squeeze(0)

    # Camera position in world space = inverse(V) * origin.
    cam_center = torch.from_numpy(np.linalg.inv(V_np)[:3, 3].copy()).cuda()

    bg = torch.zeros(3, device="cuda", dtype=torch.float32)
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(fovX / 2.0), tanfovy=math.tan(fovY / 2.0),
        bg=bg, scale_modifier=1.0,
        viewmatrix=world_view_T, projmatrix=full_proj_T,
        sh_degree=0, campos=cam_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)

    # Gradient buffer (unused for inference, but API requires it):
    screenspace = torch.zeros_like(g["means3D"], requires_grad=True)

    rendered_image, _radii = rasterizer(
        means3D=g["means3D"],
        means2D=screenspace,
        shs=None,
        colors_precomp=g["colors"],
        opacities=g["opacity"],
        scales=g["scales"],
        rotations=g["rotations"],
        cov3D_precomp=None,
    )
    img = (rendered_image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    from PIL import Image
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"  -> {out_path}  ({width}x{height})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov-deg", type=float, default=60.0)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--eye",  nargs=3, type=float)
    grp.add_argument("--view-matrix",
                     help="16 numbers in the order the IPU server logs them "
                          "(GLM column-major: 4 numbers per column, 4 columns).")
    p.add_argument("--target", nargs=3, type=float, default=[0, 0, 0])
    p.add_argument("--up",     nargs=3, type=float, default=[0, 1, 0])
    args = p.parse_args()

    print(f"Loading {args.ply} ...")
    g = load_ply(args.ply)

    if args.view_matrix:
        nums = [float(x) for x in args.view_matrix.split()]
        if len(nums) != 16: raise SystemExit("--view-matrix needs 16 numbers")
        # GLM is column-major: the 16 numbers are 4 columns of 4 entries each.
        # numpy.reshape(4,4) treats the flat array as row-major, so transpose
        # to convert column-major flat data into the (row, col) matrix used
        # everywhere else in this script.
        V = np.asarray(nums, dtype=np.float32).reshape(4, 4).T
    elif args.eye:
        V, _ = look_at(args.eye, args.target, args.up)
    else:
        means = g["means3D"]
        lo, hi = means.amin(0).cpu().numpy(), means.amax(0).cpu().numpy()
        centroid = 0.5 * (lo + hi)
        diag = float(np.linalg.norm(hi - lo))
        eye = centroid + np.array([0, 0, diag])
        V, _ = look_at(eye, centroid, [0, 1, 0])
        print(f"  auto camera: eye={eye.tolist()}")

    render(g, V, args.width, args.height, args.fov_deg, args.out)


if __name__ == "__main__":
    main()
