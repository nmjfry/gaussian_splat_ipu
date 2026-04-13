# CLAUDE.md

Project-specific guidance for AI assistants working in this repository. Read
this first — it captures the non-obvious conventions and constraints that
differ from a generic 3DGS project.

---

## What this is

An IPU (Graphcore Intelligence Processing Unit) implementation of 3D Gaussian
Splatting rendering. The work targets the **EGSR 2026** paper on in-SRAM
radiance field rendering. The point is **not** to beat GPU performance — it is
to study what happens to 3DGS when you remove global memory and force all
scene data to live in tile-local SRAM connected only by explicit inter-tile
channels.

Scope: **forward pass only**. No training. Loads pre-trained `.ply` files
(3DGS or Gaussian-Splatting-SLAM output), renders novel views, streams them
over TCP to a remote UI client.

---

## Machine topology

There are three places code runs, and files move between them via bind mounts
and git:

| Role | Where | Environment |
|---|---|---|
| **IPU server** | Docker container on `ipu-machine` | Poplar SDK 3.3, no CUDA, no GPU |
| **GPU reference** | Host `ipu-machine` (outside container) | Python venv `.venv-gpu`, CUDA 11.7 toolkit, GTX 1080, `diff-gaussian-rasterization` |
| **Remote UI** | User's laptop | `remote-ui` (nanogui) or `remote-ui-imgui` |

Bind mount inside container: `/home/nf20/` ⇔ host `/nethome/nf20/`. Same files,
different absolute paths — **write paths that work on both sides** (e.g.
`/nethome/$USER/workspace/...` on host, `/home/$USER/workspace/...` in
container). Never use `/home/nicholas/...` — that's a third machine entirely
(the user's laptop) and is not visible from either the container or
`ipu-machine`.

Git: `github.com:Nmjfry/gaussian_splat_ipu`, branch `cleanup`. The two ends
sync via push/pull.

---

## Directory layout

```
gaussian_splat_ipu/
├── src/
│   ├── main/splat.cpp            # server entry: PLY load, UI server, main loop
│   └── splat/
│       ├── camera.cpp            # lookAtBoundingBox (initial view)
│       ├── geometry.cpp          # fitFrustumToBoundingBox (projection)
│       ├── ipu_rasteriser.cpp    # Poplar graph build + tile mapping
│       ├── cpu_rasteriser.cpp    # point-only CPU fallback (NOT a full Gaussian renderer)
│       ├── file_io.cpp           # .ply loading via happly
│       └── edge_builder.cpp      # NEWS channel wiring
├── include/
│   ├── splat/ipu_geometry.hpp    # Gaussian3D / Gaussian2D, ComputeCov3D/2D (the maths)
│   ├── splat/viewport.hpp        # clipSpaceToViewport (matches DGR's ndc2Pix)
│   ├── splat/ipu_rasteriser.hpp
│   ├── remote_ui/InterfaceServer.hpp  # packet schema + State struct
│   └── tileMapping/tile_config.hpp    # IPU_TILEWIDTH/HEIGHT, TiledFramebuffer
├── codelets/
│   └── splat/codelets.cpp        # IPU tile-local code: route, bloom, sort, composite
├── tools/
│   ├── fetch_dylanebert_3dgs.py  # download scenes from HF (strips higher SH)
│   └── render_gpu_dgr.py         # CUDA reference renderer (host only)
├── tests/
│   ├── test.cpp
│   ├── test_comparison.cpp       # unit test: IPU math == original 3DGS math
│   └── test_pipeline.cpp         # end-to-end Cov3D→Cov2D→compositing test
├── data/                         # .ply files (not in git)
└── build/                        # ninja build dir
```

Remote UI lives in a sibling repo, `nf20_splatting/remote_render_ui/`.

---

## Coordinate conventions (IMPORTANT)

The IPU pipeline now uses **COLMAP / 3DGS convention end-to-end**, matching
`graphdeco-inria/diff-gaussian-rasterization` exactly:

- **World**: Y-up (arbitrary but consistent).
- **Camera / view space**: +X right, **−Y up** (image Y is down), **+Z forward** (into the scene).
- **Visible Gaussians**: `view_z > 0`.
- **Near-plane cull**: `view_z ≤ 0.2`.
- **Projection**: 3DGS `z_sign = +1`.
- **Front-to-back sort**: *ascending* `view_z` (smallest positive first).
- **NDC→pixel**: `((ndc + 1) * S − 1) * 0.5` (DGR's `ndc2Pix`; half-pixel offset from the naïve `(ndc + 1) * S / 2`).

How this is achieved internally:

1. `lookAtBoundingBox` builds a standard OpenGL view matrix via `glm::lookAt`.
2. Client sends WASD offsets in what it thinks is "world space" (works because
   the initial viewMatrix's rotation is still identity-ish).
3. Server composes `dynamicView = R_pitch * R_yaw * T_off * viewMatrix` in
   OpenGL convention.
4. **Final step in `splat.cpp`**: pre-multiply by `diag(1, −1, −1, 1)` to flip
   the Y and Z rows. This is the OpenGL→COLMAP conversion. The view matrix
   streamed to the codelet is COLMAP.
5. Projection matrix is built directly in 3DGS style in `geometry.cpp`
   (`fitFrustumToBoundingBox`). No `glm::perspective` here — that would give
   z_sign = −1.

The log line `Dynamic view matrix: <4 numbers>` prints GLM **column-major**
data: each log line is one column, 4 columns total = 16 numbers. Feed these
directly into `tools/render_gpu_dgr.py --view-matrix "..."`; the script does
the `.reshape(4, 4).T` for you.

Historical note: the `--flip-up` server flag exists for scenes whose world Y
is the opposite direction from the norm. After the COLMAP conversion most
scenes don't need it any more.

---

## Gaussian struct layout

`Gaussian3D` is packed to **60 bytes**, 4-byte aligned, with `gid` (uint32 as
float) as the last element (required by the `insert()`/`evict()` helpers which
locate `gid` at offset `sizeof(g) − sizeof(float)`):

```cpp
class Gaussian3D {
    ivec3 mean;   // world-space, 12 B (mean.w was always 1.0 — dropped)
    ivec4 colour; // RGB + pre-activated opacity (sigmoid applied at load),  16 B
    ivec4 rot;    // quaternion (real, i, j, k), 16 B
    ivec3 scale;  // log-space per-axis (exp() applied inside ComputeCov3D),  12 B
    float gid;    //  4 B  ← MUST stay last
};                // total = 60 B
```

`Gaussian2D` stores the precomputed **conic** (3 floats, inverse of 2D
covariance) rather than the covariance itself — avoids a per-pixel 2×2
inversion in the inner alpha-blend loop. `Gaussian2D::BoundingBoxFromCov()` is
the static helper for the 3σ-radius bounding box.

---

## Codelet / tile architecture

Single `GSplat` MultiVertex per IPU tile. Framebuffer is 1280×720, tiled into
1440 pieces of 32×20 pixels, one per IPU tile. Each tile owns:

- A slice of the framebuffer (8-bit RGBX, 4 B/pixel, **always written as
  32-bit words** via `memcpy` — individual byte writes on IPU SRAM can race
  between hardware threads).
- An input buffer of Gaussians that are "anchored" to this screen region.
- NEWS (north-east-west-south) channels of capacity `numPoints = 360`
  Gaussians each, for routing and bloom.
- An overflow/storage buffer.

Per frame, a tile runs:

1. **readInput**: pull from N/E/W/S `in` channels. If the anchor (projected
   mean) is on this tile, store locally; otherwise forward toward the
   anchor via Manhattan-distance routing.
2. **renderInternal**: project all locally-held Gaussians; if their 2D
   bounding box straddles this tile, write a copy to the outgoing direction
   (bloom). Collect the visible subset into `gaus2D`.
3. **renderTile**: sort `gaus2D` ascending by `view_z`, iterate pixels
   (distributed across 6 workers by row), alpha-blend in front-to-back order.

All compute and all scene data stay on-tile. The only host traffic each frame
is the view/projection matrices (down) and the assembled framebuffer (up).

---

## Architectural differences from `diff-gaussian-rasterization`

These **cannot be eliminated** — they're the paper's subject matter.

| DGR | IPU |
|---|---|
| Global radix sort over `(tile_id, depth)` in DRAM | No DRAM → per-tile local sort only |
| Every tile sees every visible Gaussian via shared memory | Tiles only see what was routed + bloomed to them |
| Saturation-free: unlimited Gaussians per tile | NEWS channel capacity → drops under saturation |
| Background blend at end (`C += T·bg`) | Clears to black (no bg blend) |

When IPU output differs visibly from DGR at the same pose, expect:
- **Tile-boundary-aligned holes** (channel saturation in dense regions)
- **Blooming** (Gaussians take 1 hop/frame to propagate)
- **Minor per-tile sort-order differences** (local vs global sort)

The pixel-rasterisation math is otherwise identical and verified by
`tests/test_pipeline.cpp`.

---

## Build

```bash
# Inside the Docker container
cd ~/workspace/gaussian_splat_ipu/build
ninja -j11
```

The first `cmake ..` needs to detect Poplar / Boost / OpenCV. External deps
are vendored under `external/` (packetcomms, videolib, glm). Popart/Poplar
comes from the container image.

### Tests (CPU-only, sanity checks on the maths)

```bash
g++ -std=c++17 -I include -I external/glm tests/test_pipeline.cpp -o test_pipeline
./test_pipeline
# → "IPU pipeline is NUMERICALLY EQUIVALENT to the original 3DGS"
```

---

## Run

```bash
# Server (in container)
./build/src/main/splat --input data/bonsai-7k-mini.ply --ui-port 5000

# Remote UI (on user laptop)
./remote-ui-imgui --host ipu-machine --port 5000
```

Useful server flags:
- `--input <path>` (required)
- `--ui-port 5000`
- `--device cpu|ipu` (default `cpu`; CPU is point-only, for debugging initial
  scene placement)
- `--flip-up` — for scenes whose world Y is inverted; rarely needed now
- `--no-amp` — disables optimised AMP codelets (default: disabled)
- `--paired-shots-dir <dir>` — where Screenshot-button paired renders go
  (default `paired_shots/`).

### Paired IPU + GPU screenshots (one click → two images)

When the client's **Screenshot** button is pressed it (1) saves the decoded
video frame locally, and (2) sends a `screenshot` packet to the server, which
writes its own framebuffer + a sidecar `.json` describing the pose into
`--paired-shots-dir`. A watcher script on the host polls that dir and runs
`render_gpu_dgr.py` at exactly the same pose, producing
`screenshot-YYYYMMDD-HHMMSS_gpu.png` alongside the IPU `screenshot-...png`.

Run order:

```bash
# HOST (needs CUDA / venv .venv-gpu with diff-gaussian-rasterization):
./tools/start_watcher.sh                 # leaves a watcher tailing the dir

# CONTAINER (IPU):
./build/src/main/splat --input data/salad.ply --ui-port 5000

# Laptop: start the remote UI as usual, click Screenshot.
```

The `paired_shots/` dir is bind-mounted (container `/home/$USER/...` =
host `/nethome/$USER/...`) so the watcher sees what the server writes without
any extra plumbing.

The server logs the current `Dynamic view matrix` (4 lines = 4 GLM columns)
and `fov` (HALF-FOV in radians) every few seconds. These are what you feed the
GPU reference renderer to reproduce a pose exactly.

---

## GPU reference renderer

On the host (not in the container), for paper figure comparisons:

```bash
cd /nethome/$USER/workspace/gaussian_splat_ipu
source .venv-gpu/bin/activate

python3 tools/render_gpu_dgr.py \
    --ply data/bonsai-7k-mini.ply \
    --out /nethome/$USER/workspace/gaussian_splat_ipu/gpu_figures/bonsai_pose.png \
    --fov-deg 60 \
    --view-matrix "<16 numbers from server log, space-separated>"
```

`--fov-deg` is **full FOV in degrees**. Server log shows `fov` in radians
**half-FOV**. Convert: `full_deg = half_rad * 2 * 180/π`.

The script centres the scene (subtracts centroid from all means) to match
what the server does on load. SLAM-trained `.ply` files often have a centroid
well away from origin; without this the GPU render ends up offset.

---

## Remote UI (client-side)

Two GUIs coexist in `nf20_splatting/remote_render_ui/`:

- `remote-ui` (nanogui, stable baseline, default target)
- `remote-ui-imgui` (ImGui, game-style controls; `cmake -DBUILD_IMGUI_UI=ON`,
  ImGui vendored under `external/imgui/`)

Controls:

| Key / input | Action |
|---|---|
| WASD / arrows | translate (forward/back/strafe) |
| Q / E | down / up in camera frame |
| Left-click-drag viewport | mouse-look (pitch/yaw, Y-inverted by default) |
| Shift | sprint (4× speed) |
| Space | reset pose |
| Y | toggle pitch inversion (nanogui only) |

Packet protocol (see `include/remote_ui/InterfaceServer.hpp`): `X Y Z
env_rotation env_rotation_2 fov device stop detach render_preview
render_time tile_histogram ready`. Client sends world-space WASD offsets +
yaw/pitch degrees; server composes the full view matrix. Server also pushes
`render_time` (ms) for telemetry display in both clients.

Both clients have a **Screenshot** button that writes a timestamped PNG of
the current decoded frame using `cv::imwrite`. Note `rgbBuffer` and
`bufferMutex` in `VideoTexture.hpp` are `mutable` so the `const` save method
can take a brief lock.

---

## When making changes

- Do **not** flip conventions in just one place. View, projection, cull,
  sort, and the viewport mapping must all agree (see the table above). Use
  `tests/test_pipeline.cpp` after any maths change — it compares against
  CUDA 3DGS cell-by-cell.
- The `Gaussian3D` struct is byte-critical. `gid` **must** stay last. Size
  must be a multiple of 4. If you need more fields, shrink something else.
- On-tile memory writes: use `memcpy` of full 32-bit words, not individual
  byte stores, or workers on the same tile will race.
- The IPU codelet has no DRAM, no dynamic allocation, no `stdlib` beyond a
  small subset. No `std::function`, no virtual calls, no recursion. Keep
  everything flat.
- Keep the routing NEWS topology fixed at compile time. JIT Dynamic Lookup
  exists as an experimental Poplar feature but is out of scope for this
  project.
- Host-side code can use OpenCV, GLM, Boost freely. Codelet code has a
  drastically smaller stdlib surface.

---

## Paper & related docs

The paper is in a separate repo (`report/EGSR2026/`) outside this workspace.
Related reference material on GPU 3DGS lives at
`https://github.com/graphdeco-inria/gaussian-splatting` and the CUDA
rasterizer at `https://github.com/graphdeco-inria/diff-gaussian-rasterization`
(version pinned to the commit used by the reference renderer; see
`tools/render_gpu_dgr.py`).

The HF dataset `dylanebert/3dgs` has small scenes suitable for the IPU
(bicycle, bonsai, stump at 7k and 30k iterations, plus a toy "luigi").
`tools/fetch_dylanebert_3dgs.py` downloads and strips higher-order SH to
produce DC-only `.ply`s that fit the IPU loader.
