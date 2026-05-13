#!/usr/bin/env python3
"""Interactive 3D viewer showing camera trajectories relative to the scene.

Usage:
    python3 tools/view_trajectories.py --ply data/point_cloud_10.ply
    python3 tools/view_trajectories.py --ply data/point_cloud_10.ply --trajectory orbit_2.0deg
    python3 tools/view_trajectories.py --ply data/point_cloud_10.ply --all
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_ply_positions(path, max_points=5000):
    """Load PLY vertex positions, subsampled for display."""
    try:
        from plyfile import PlyData
        data = PlyData.read(path)["vertex"].data
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
    except ImportError:
        import struct
        # Minimal PLY parser for vertex x,y,z
        with open(path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            # Parse header for vertex count
            n_verts = 0
            for line in header.decode().split("\n"):
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
            # Read binary (assume float32 x,y,z as first 3 properties)
            # This is a rough fallback — use plyfile for reliability
            props = []
            in_vertex = False
            for line in header.decode().split("\n"):
                if "element vertex" in line:
                    in_vertex = True
                elif line.startswith("element"):
                    in_vertex = False
                elif in_vertex and line.startswith("property"):
                    props.append(line)
            n_props = len(props)
            raw = np.frombuffer(f.read(n_verts * n_props * 4), dtype=np.float32)
            raw = raw.reshape(n_verts, n_props)
            xyz = raw[:, :3]

    # Centre (same as IPU pipeline)
    xyz = xyz.copy()
    lo, hi = xyz.min(0), xyz.max(0)
    centroid = 0.5 * (lo + hi)
    xyz -= centroid

    # Subsample for display
    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]

    return xyz


def look_at(eye, target, up):
    """GLM-style lookAt → 4×4 view matrix (OpenGL convention)."""
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = true_up
    R[2, :3] = -fwd
    T = np.eye(4)
    T[:3, 3] = -eye
    return R @ T


FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


def cam_pos_from_view(V):
    """Extract camera world-space position from a 4×4 view matrix."""
    Vinv = np.linalg.inv(V)
    return Vinv[:3, 3]


def cam_forward_from_view(V):
    """Extract camera forward direction (what the camera looks at) in world space."""
    Vinv = np.linalg.inv(V)
    # In COLMAP convention (+Z forward), the forward vector is the 3rd column of inv(V)
    return Vinv[:3, 2]


def make_frustum_lines(V, scale=0.3):
    """Return line segments for a small camera frustum wireframe."""
    Vinv = np.linalg.inv(V)
    origin = Vinv[:3, 3]
    right = Vinv[:3, 0] * scale * 0.6
    up = Vinv[:3, 1] * scale * 0.4
    fwd = Vinv[:3, 2] * scale

    corners = [
        origin + fwd + right + up,
        origin + fwd - right + up,
        origin + fwd - right - up,
        origin + fwd + right - up,
    ]
    lines = []
    for c in corners:
        lines.append([origin, c])
    for i in range(4):
        lines.append([corners[i], corners[(i + 1) % 4]])
    return lines


def generate_trajectories(bb_min, bb_max):
    """Replicate the C++ trajectory generators."""
    centre = 0.5 * (bb_min + bb_max)
    diag = bb_max - bb_min
    radius = np.linalg.norm(diag) * 0.5
    up = np.array([0, 1, 0], dtype=np.float64)

    trajectories = {}

    # Static
    eye = centre + np.array([0, 0, 2 * radius])
    V = look_at(eye, centre, up)
    trajectories["static"] = [(FLIP_YZ @ V, 0)]

    # Orbits
    for omega in [0.1, 0.5, 2.0, 5.0, 20.0]:
        frames = []
        for f in range(300):
            angle = np.radians(omega * f)
            eye = centre + np.array([
                np.sin(angle) * 2 * radius,
                0,
                np.cos(angle) * 2 * radius
            ])
            V = look_at(eye, centre, up)
            frames.append((FLIP_YZ @ V, f))
        trajectories[f"orbit_{omega:.1f}deg"] = frames

    # Pure rotation
    frames = []
    cam_pos = centre + np.array([0, 0, 2 * radius])
    for f in range(300):
        angle = np.radians(1.0 * f)
        target = cam_pos + np.array([np.sin(angle), 0, -np.cos(angle)])
        V = look_at(cam_pos, target, up)
        frames.append((FLIP_YZ @ V, f))
    trajectories["pure_rotation_1deg"] = frames

    # Pure translation
    frames = []
    for f in range(300):
        step = radius * 0.001 * f
        eye = centre + np.array([0, 0, 2 * radius]) - np.array([0, 0, step])
        V = look_at(eye, centre, up)
        frames.append((FLIP_YZ @ V, f))
    trajectories["pure_translation"] = frames

    # Random teleport
    rng = np.random.default_rng(42)
    frames = []
    for f in range(100):
        rng2 = np.random.default_rng(42 + f)
        d = rng2.uniform(-1, 1, 3)
        d /= np.linalg.norm(d)
        eye = centre + d * 2 * radius
        V = look_at(eye, centre, up)
        frames.append((FLIP_YZ @ V, f))
    trajectories["random_teleport"] = frames

    return trajectories


def plot_trajectory(ax, traj_frames, name, color, frustum_every=20, frustum_scale=0.3):
    """Plot a single trajectory as a path with frustum markers."""
    positions = np.array([cam_pos_from_view(V) for V, _ in traj_frames])

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color=color, linewidth=1.5, label=name, alpha=0.8)

    # Start marker
    ax.scatter(*positions[0], color=color, s=60, marker="o", edgecolors="black", zorder=5)

    # Draw frustums at intervals
    for i in range(0, len(traj_frames), frustum_every):
        V, _ = traj_frames[i]
        lines = make_frustum_lines(V, scale=frustum_scale)
        for seg in lines:
            pts = np.array(seg)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=color, linewidth=0.5, alpha=0.4)


def main():
    parser = argparse.ArgumentParser(description="View camera trajectories in 3D")
    parser.add_argument("--ply", required=True, help="PLY scene file")
    parser.add_argument("--trajectory", "-t", default=None,
                        help="Show only this trajectory (e.g. orbit_2.0deg)")
    parser.add_argument("--all", action="store_true",
                        help="Show all trajectories at once")
    parser.add_argument("--max-points", type=int, default=3000,
                        help="Max scene points to display")
    parser.add_argument("--save", default=None,
                        help="Save to file instead of showing interactively")
    parser.add_argument("--list", action="store_true",
                        help="List available trajectory names and exit")
    args = parser.parse_args()

    print(f"Loading {args.ply}...")
    pts = load_ply_positions(args.ply, max_points=args.max_points)
    bb_min, bb_max = pts.min(0), pts.max(0)
    print(f"  {len(pts)} points (subsampled), BB: {bb_min} to {bb_max}")

    trajectories = generate_trajectories(bb_min, bb_max)

    if args.list:
        print("\nAvailable trajectories:")
        for name, frames in trajectories.items():
            print(f"  {name} ({len(frames)} frames)")
        return

    # Choose which trajectories to show
    if args.trajectory:
        if args.trajectory not in trajectories:
            print(f"Unknown trajectory '{args.trajectory}'. Available:")
            for n in trajectories:
                print(f"  {n}")
            return
        show = {args.trajectory: trajectories[args.trajectory]}
    elif args.all:
        show = trajectories
    else:
        # Default: show a representative subset
        show = {}
        for name in ["orbit_0.5deg", "orbit_5.0deg", "pure_rotation_1deg",
                      "pure_translation", "random_teleport"]:
            if name in trajectories:
                show[name] = trajectories[name]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(show), 1)))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw scene points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=0.3, c="gray", alpha=0.15, rasterized=True)

    # Draw bounding box wireframe
    lo, hi = bb_min, bb_max
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="black", linewidth=0.3, alpha=0.3)

    # Draw trajectories
    diag = np.linalg.norm(bb_max - bb_min)
    frustum_scale = diag * 0.08
    for (name, frames), color in zip(show.items(), colors):
        n_frames = len(frames)
        every = max(1, n_frames // 15)
        plot_trajectory(ax, frames, name, color,
                       frustum_every=every, frustum_scale=frustum_scale)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Camera Trajectories vs Scene")

    # Set equal aspect ratio
    max_range = diag * 1.5
    mid = 0.5 * (bb_min + bb_max)
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved to {args.save}")
    else:
        print("Showing interactive 3D viewer (close window to exit)...")
        plt.show()


if __name__ == "__main__":
    main()
