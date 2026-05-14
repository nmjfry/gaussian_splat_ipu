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

    # The IPU server re-centres the scene on load (subtracts the bounding-box
    # centroid so everything sits at world origin). If we don't do the same
    # here, the view matrix logged by the IPU assumes a scene centred at
    # origin but the raw PLY might be offset, producing a shifted render.
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    centroid = 0.5 * (lo + hi)
    xyz = xyz - centroid
    print(f"  {len(xyz)} gaussians from {path}  (centred, shift={centroid.tolist()})")

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
    """One-shot render: rasterise + save PNG. Returns nothing.

    For trajectory benchmarks, use `render_step` instead — it skips the PNG
    write and exposes a precomputed tensor render path."""
    img_np = _rasterise(g, V_np, width, height, fov_y_deg).cpu().numpy()
    img = (img_np * 255).astype(np.uint8)
    from PIL import Image
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"  -> {out_path}  ({width}x{height})")


def _rasterise(g, V_np, width, height, fov_y_deg):
    """Run DGR for a single view; return the rendered (H, W, 3) tensor on CPU.

    Centralised so the benchmark loop can reuse the exact same call path as
    the screenshot helper — no skew between bench numbers and the saved PNG.
    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )
    aspect = width / float(height)
    P_np, fovX, fovY = make_projection(fov_y_deg, aspect)

    V = torch.from_numpy(V_np).cuda()
    P = torch.from_numpy(P_np).cuda()
    world_view_T = V.transpose(0, 1)
    full_proj_T = (world_view_T.unsqueeze(0) @ P.transpose(0, 1).unsqueeze(0)).squeeze(0)
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
    screenspace = torch.zeros_like(g["means3D"], requires_grad=True)
    rendered_image, _radii = rasterizer(
        means3D=g["means3D"], means2D=screenspace,
        shs=None, colors_precomp=g["colors"],
        opacities=g["opacity"], scales=g["scales"],
        rotations=g["rotations"], cov3D_precomp=None,
    )
    return rendered_image.clamp(0, 1).permute(1, 2, 0).detach()


def parse_traj(path):
    """Read a benchmark_traj_*.traj file. Returns (poses, fov_half_rad, ply).

    poses is a list of (4, 4) numpy view matrices in row-major mathematical
    form (i.e. each "v" line's 16 GLM column-major floats reshaped to (4, 4)
    and transposed, exactly the same conversion the existing --view-matrix
    flag does).
    """
    poses = []
    fov_half = None
    ply = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("fov_half_rad:"):
                fov_half = float(s.split(":", 1)[1])
            elif s.startswith("ply:"):
                ply = s.split(":", 1)[1].strip()
            elif s.startswith("v "):
                nums = [float(x) for x in s.split()[1:]]
                if len(nums) == 16:
                    V = np.asarray(nums, dtype=np.float32).reshape(4, 4).T
                    poses.append(V)
    if not poses:
        raise SystemExit(f"No 'v <16 floats>' lines in {path}")
    if fov_half is None:
        raise SystemExit(f"No fov_half_rad in {path}")
    return poses, fov_half, ply


def run_static_benchmark(g, V_np, fov_y_deg, width, height,
                          num_frames, out_csv, last_png):
    """Render `num_frames` frames at a single fixed view. Same CSV schema as
    run_trajectory_benchmark so static and orbit results can be diffed."""
    return run_trajectory_benchmark(g, [V_np] * 1, fov_y_deg, width, height,
                                    num_frames, out_csv, last_png)


def run_trajectory_benchmark(g, poses, fov_y_deg, width, height,
                              spins, out_csv, last_png):
    """Iterate `spins` full loops of the trajectory, render each frame, log
    per-frame timing to CSV. Last frame is saved as a sanity-check PNG.

    Timing uses cudaEvent for the rasterisation step (matches the IPU's
    device-side compute_ms — excludes Python/host overhead) and a wall clock
    for end-to-end including any Python glue."""
    import time, csv as csv_mod
    num_poses = len(poses)
    total_frames = spins * num_poses

    # Warm-up: lazy CUDA init + DGR JIT compile + cache warmup
    for _ in range(5):
        _rasterise(g, poses[0], width, height, fov_y_deg)
    torch.cuda.synchronize()

    rows = []
    print(f"GPU trajectory benchmark: {total_frames} frames ({spins} spins x {num_poses} poses)")
    bench_start = time.perf_counter()
    last_img = None
    for f in range(total_frames):
        V = poses[f % num_poses]
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        ev0.record()
        img = _rasterise(g, V, width, height, fov_y_deg)
        ev1.record()
        torch.cuda.synchronize()
        wall1 = time.perf_counter()
        gpu_ms = ev0.elapsed_time(ev1)        # device-side ms
        wall_ms = (wall1 - wall0) * 1000.0    # host wall ms incl. Python
        rows.append((f, f % num_poses, gpu_ms, wall_ms))
        if f % 60 == 0 or f + 1 == total_frames:
            print(f"  frame {f+1:4d}/{total_frames}  pose={f % num_poses:4d}  gpu={gpu_ms:6.2f}ms  wall={wall_ms:6.2f}ms")
        if f + 1 == total_frames:
            last_img = img
    bench_end = time.perf_counter()
    bench_secs = bench_end - bench_start
    print(f"GPU benchmark done: {total_frames} frames in {bench_secs:.2f}s ({total_frames/bench_secs:.1f} FPS)")

    # CSV with column names compatible with the IPU summariser. The columns
    # not measured on GPU (per-phase) are left as 0 so the same script reads both.
    with open(out_csv, "w", newline="") as fh:
        w = csv_mod.writer(fh)
        w.writerow(["frame", "pose_idx", "wall_ms", "route_ms", "blend_ms",
                    "exchange_ms", "total_ms", "mvp_ms",
                    "clear_min", "clear_mean", "clear_max",
                    "routing_min", "routing_mean", "routing_max",
                    "proj_min", "proj_mean", "proj_max",
                    "sort_min", "sort_mean", "sort_max",
                    "total_cyc_min", "total_cyc_mean", "total_cyc_max",
                    "total_visible"])
        for frame, pose_idx, gpu_ms, wall_ms in rows:
            w.writerow([frame, pose_idx, f"{wall_ms:.4f}",
                        0, 0, 0, f"{gpu_ms:.4f}", 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0])
    print(f"Wrote {out_csv}")

    if last_png and last_img is not None:
        from PIL import Image
        img_np = (last_img.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(last_png)
        print(f"Wrote {last_png}")


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
    p.add_argument("--benchmark", type=int, default=0,
                   help="Render N frames at the (single) given pose and report mean FPS")
    p.add_argument("--play-path", default=None,
                   help="Path to a .traj file from tools/sample_orbit_path.py. "
                        "Renders every pose in order for --spins full loops; "
                        "skips --view-matrix/--eye and uses the trajectory's fov.")
    p.add_argument("--spins", type=int, default=2,
                   help="Number of trajectory loops for --play-path mode (default 2)")
    p.add_argument("--bench-static", type=int, default=0,
                   help="Render N frames at a single fixed pose and write per-frame "
                        "CSV (same schema as --play-path). Requires --pose-json, "
                        "--view-matrix, or --eye.")
    p.add_argument("--pose-json", default=None,
                   help="Load view matrix + fov from a benchmark_pose_*.json file "
                        "(same format the IPU server's Screenshot button writes).")
    p.add_argument("--out-csv", default=None,
                   help="--play-path / --bench-static: per-frame timing CSV "
                        "(default: alongside --out)")
    args = p.parse_args()

    print(f"Loading {args.ply} ...")
    g = load_ply(args.ply)

    if args.play_path:
        poses, fov_half, _ = parse_traj(args.play_path)
        fov_y_deg = math.degrees(fov_half) * 2.0
        out_csv = args.out_csv or str(Path(args.out).with_suffix(".csv"))
        run_trajectory_benchmark(g, poses, fov_y_deg, args.width, args.height,
                                 args.spins, out_csv, args.out)
        return

    if args.bench_static > 0:
        # Need a view matrix + fov from somewhere
        fov_y_deg = args.fov_deg
        if args.pose_json:
            with open(args.pose_json) as f:
                pose = json.load(f)
            nums = pose["view_matrix"]
            V = np.asarray(nums, dtype=np.float32).reshape(4, 4).T
            if "fov_half_rad" in pose:
                fov_y_deg = math.degrees(float(pose["fov_half_rad"])) * 2.0
        elif args.view_matrix:
            nums = [float(x) for x in args.view_matrix.split()]
            V = np.asarray(nums, dtype=np.float32).reshape(4, 4).T
        elif args.eye:
            V, _ = look_at(args.eye, args.target, args.up)
        else:
            raise SystemExit("--bench-static needs --pose-json, --view-matrix, or --eye")
        out_csv = args.out_csv or str(Path(args.out).with_suffix(".csv"))
        run_static_benchmark(g, V, fov_y_deg, args.width, args.height,
                             args.bench_static, out_csv, args.out)
        return

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

    if args.benchmark > 0:
        import time
        # Warm-up
        for _ in range(5):
            render(g, V, args.width, args.height, args.fov_deg, args.out)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.benchmark):
            render(g, V, args.width, args.height, args.fov_deg, args.out)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        secs = t1 - t0
        fps = args.benchmark / secs
        print(f"BENCHMARK: frames={args.benchmark} total_sec={secs:.4f} "
              f"fps={fps:.2f} ms_per_frame={1000*secs/args.benchmark:.3f}")
    else:
        render(g, V, args.width, args.height, args.fov_deg, args.out)


if __name__ == "__main__":
    main()
