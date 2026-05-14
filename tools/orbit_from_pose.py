#!/usr/bin/env python3
"""Infer an orbit description (benchmark_path_*.json) from a single calibrated
pose (benchmark_pose_*.json). Useful when you only have one anchor.

Heuristic:
  - Orbit axis: snapped to +Y by default.
  - look_at: the closest point on the camera's forward ray to the world origin
    (since scenes are centred at load, this approximates the visual focus).
  - camera_y: the anchor camera's Y position.
  - radius: planar distance from anchor camera to look_at (perpendicular to axis).
  - anchor_a azimuth: atan2(planar.z, planar.x).

You can tweak look_at after the fact by editing the JSON.
"""

import argparse
import json
import os
import sys
import numpy as np


def vm_to_M(flat):
    return np.array(flat, dtype=np.float64).reshape(4, 4).T


def dense_centre_from_ply(ply_path, flip_scene=True, bins=30, kernel=5):
    """Return the centre of the densest region of the scene in the same
    coordinate frame the renderer uses (post --flip-scene, post bbox-centring).

    Density is the 5x5x5 box-sum of a 30x30x30 histogram of vertex positions.
    """
    from plyfile import PlyData
    p = PlyData.read(ply_path)
    v = p['vertex']
    pts = np.stack([np.asarray(v['x']), np.asarray(v['y']),
                    np.asarray(v['z'])], axis=1).astype(np.float64)
    # splat.cpp: bbCentre on raw coords, then v.p -= bbCentre, then --flip-scene
    # applies diag(1,-1,-1) effectively.
    bbc = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    pts -= bbc
    if flip_scene:
        pts[:, 1] *= -1
        pts[:, 2] *= -1
    H, edges = np.histogramdd(pts, bins=bins)
    K = kernel
    H_pad = np.pad(H, K // 2, mode='constant')
    sm = np.zeros_like(H)
    for dx in range(K):
        for dy in range(K):
            for dz in range(K):
                sm += H_pad[dx:dx + H.shape[0], dy:dy + H.shape[1], dz:dz + H.shape[2]]
    idx = np.unravel_index(int(sm.argmax()), H.shape)
    return np.array([0.5 * (edges[d][i] + edges[d][i + 1]) for d, i in enumerate(idx)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pose_json', help='Calibrated pose JSON (one anchor)')
    ap.add_argument('--out', required=True, help='Output orbit description JSON')
    ap.add_argument('--axis', default='0,1,0', help='Orbit axis (default +Y)')
    ap.add_argument('--world-up', default='0,-1,0',
                    help='Scene up direction. Default -Y matches --flip-scene scenes.')
    ap.add_argument('--rotation-direction', type=int, default=-1)
    ap.add_argument('--look-at', default=None,
                    help='Override look_at point as "x,y,z" (in --flip-scene post-centre coords)')
    ap.add_argument('--dense-from-ply', default=None,
                    help='Compute look_at as the densest region of the given .ply '
                         '(applies --flip-scene + bbox-centring like the server). '
                         'Implies the orbit will be centred on the dense mass of the scene.')
    args = ap.parse_args()

    with open(args.pose_json) as f:
        pose = json.load(f)

    M = vm_to_M(pose['view_matrix'])
    R = M[:3, :3]
    t = M[:3, 3]
    eye = -R.T @ t
    forward = R[2, :]
    fov = float(pose['fov_half_rad'])
    ply = pose.get('ply')

    axis = np.array([float(x) for x in args.axis.split(',')], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    world_up = np.array([float(x) for x in args.world_up.split(',')], dtype=np.float64)
    world_up /= np.linalg.norm(world_up)

    look_at_source = "closest-point-on-ray-to-origin"
    if args.look_at is not None:
        look_at = np.array([float(x) for x in args.look_at.split(',')], dtype=np.float64)
        look_at_source = f"override {args.look_at}"
        s = float(np.dot(look_at - eye, forward))
    elif args.dense_from_ply is not None:
        look_at = dense_centre_from_ply(args.dense_from_ply)
        look_at_source = f"density peak from {args.dense_from_ply}"
        s = float(np.dot(look_at - eye, forward))
    else:
        # look_at = closest point on view ray (eye + s*forward) to origin.
        s = float(-np.dot(forward, eye))
        look_at = eye + s * forward

    camera_y = float(np.dot(eye, axis))

    d = eye - look_at
    d_planar = d - np.dot(d, axis) * axis
    radius = float(np.linalg.norm(d_planar))
    az = float(np.degrees(np.arctan2(d_planar[2], d_planar[0])))

    print(f"eye      = {eye}")
    print(f"forward  = {forward}")
    print(f"look_at  = {look_at}  (|origin->look_at| = {np.linalg.norm(look_at):.3f}, s_along_ray = {s:.3f})")
    print(f"camera_y = {camera_y:.4f}")
    print(f"radius   = {radius:.4f}   az = {az:+.2f} deg")

    out = {
        "type": "orbit",
        "look_at": [float(x) for x in look_at],
        "axis": [float(x) for x in axis],
        "world_up": [float(x) for x in world_up],
        "camera_y": camera_y,
        "radius": radius,
        "fov_half_rad": fov,
        "ply": ply,
        "anchor_a": {
            "pose": os.path.relpath(args.pose_json),
            "azimuth_deg": az,
        },
        "rotation_direction": args.rotation_direction,
        "notes": (
            f"Orbit inferred from a single calibrated pose by orbit_from_pose.py. "
            f"look_at source: {look_at_source}. "
            f"camera_y / radius come from the anchor pose; axis snapped to +Y."
        )
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
