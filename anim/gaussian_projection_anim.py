"""
Side-by-side animation for slides: a 3D Gaussian + camera frustum (left) and
the projected 2D Gaussian on the image plane (right).

The camera makes small smooth motions in three phases
(translation -> rotation -> both) to show that the screen-space
Gaussian barely moves for small camera motion.

Projection follows EWA splatting: Sigma2D = J W Sigma W^T J^T.

Usage:
    python3 gaussian_projection_anim.py            # renders MP4
    python3 gaussian_projection_anim.py --preview  # saves 3 preview PNGs only
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Figtree",
    "font.size": 15,
})
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------- scene ----
MU = np.array([0.0, 0.0, 0.0])                      # gaussian mean (world)

# anisotropic covariance: principal std devs, rotated in world space.
# GAUSS_SCALE shrinks/grows the gaussian; != 1 adds a filename suffix so
# variants don't overwrite the originals
SCALE = float(os.environ.get("GAUSS_SCALE", "1.0"))
SUFFIX = "" if SCALE == 1.0 else "_tight"
_stds = SCALE * np.array([0.95, 0.55, 0.32])

# TILES=N splits image space into an NxN grid, highlights the tile the
# gaussian centre lands on, and enlarges the camera motion so the splat
# reaches the edge tiles
TILES = int(os.environ.get("TILES", "0"))
if TILES:
    SUFFIX += "_tiled"
def _rot(axis, deg):
    a = np.deg2rad(deg)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])
_Rg = _rot(np.array([0.3, 1.0, 0.5]), 35.0)
SIGMA = _Rg @ np.diag(_stds**2) @ _Rg.T             # 3x3 world covariance

C0 = np.array([-4.2, -3.4, 1.3])                    # base camera centre
WORLD_UP = np.array([0.0, 0.0, 1.0])
FOCAL = 1.0                                         # focal length (normalised)
SENSOR = 0.45                                       # image half-extent (u,v)
D_PLANE = 1.0                                       # frustum image-plane depth

FPS = 30
T_PHASE = 8.0                                       # seconds per phase
N_PHASE = int(FPS * T_PHASE)
N_FRAMES = 3 * N_PHASE
PHASES = ["Translation", "Rotation", "Translation + rotation"]

A_TRANS = (0.65, 0.45)                              # translation amplitudes
A_ROT = (6.0, 4.5)                                  # yaw, pitch amplitude (deg)
G_BOTH = 0.7                                        # combined-phase damping
if TILES:                                           # reach the edge tiles
    A_TRANS = (1.6, 1.5)
    A_ROT = (16.0, 15.0)
    G_BOTH = 0.45
if os.environ.get("GENTLE"):                        # cross tile boundaries
    A_TRANS = tuple(0.62 * a for a in A_TRANS)      # without going as deep
    A_ROT = tuple(0.62 * a for a in A_ROT)
    SUFFIX += "_gentle"

COL_GAUSS = "#d53e4f"                               # same red in 3D and 2D
COL_SPLAT = "#d53e4f"
COL_TRAIL = "#2c4b8f"
COL_CAM = "#333333"

# ---------------------------------------------------------------- camera ----
def look_at(c, target):
    z = target - c
    z = z / np.linalg.norm(z)
    x = np.cross(z, WORLD_UP)
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)                              # v points up in the image
    return np.stack([x, y, z])                      # rows: world->cam rotation

R0 = look_at(C0, MU)

def camera_pose(frame):
    """Return (c, R, phase index) for a frame. Each phase starts/ends at the
    base pose so the three phases join smoothly."""
    k = min(frame // N_PHASE, 2)
    p = (frame - k * N_PHASE) / N_PHASE             # local progress in [0,1)
    # plus-sign profile: full left/right sweep in the first half of the
    # phase, full up/down sweep in the second half
    if p < 0.5:
        h, v = np.sin(4 * np.pi * p), 0.0
    else:
        h, v = 0.0, np.sin(4 * np.pi * (p - 0.5))

    do_t = k in (0, 2)
    do_r = k in (1, 2)
    g = G_BOTH if k == 2 else 1.0                   # tone down combined phase

    c = C0.copy()
    if do_t:
        c = c + g * (A_TRANS[0] * h * R0[0] + A_TRANS[1] * v * R0[1])

    R = R0
    if do_r:
        yaw = np.deg2rad(g * A_ROT[0] * h)
        pitch = np.deg2rad(g * A_ROT[1] * v)
        # small rotations about the camera's own y (yaw) and x (pitch) axes
        Ry = _rot(R0[1], np.rad2deg(yaw))
        Rx = _rot(R0[0], np.rad2deg(pitch))
        R = R0 @ (Ry @ Rx).T
    return c, R, k

# ------------------------------------------------------------- projection ---
def project(c, R):
    """EWA projection of the gaussian: returns 2D mean and 2x2 covariance."""
    t = R @ (MU - c)                                # mean in camera coords
    tx, ty, tz = t
    mu2 = FOCAL * np.array([tx / tz, ty / tz])
    J = FOCAL * np.array([[1 / tz, 0, -tx / tz**2],
                          [0, 1 / tz, -ty / tz**2]])
    cov2 = J @ R @ SIGMA @ R.T @ J.T
    return mu2, cov2

def ellipse_pts(mu2, cov2, k=1.0, n=80):
    vals, vecs = np.linalg.eigh(cov2)
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(th), np.sin(th)])
    return mu2[:, None] + k * vecs @ (np.sqrt(vals)[:, None] * circ)

def plane_to_world(c, R, uv):
    """Map image coords (2,N) onto the frustum's image plane in world space."""
    return (c[:, None]
            + D_PLANE * ((uv[0] / FOCAL) * R[0][:, None]
                         + (uv[1] / FOCAL) * R[1][:, None]
                         + R[2][:, None]))

# ------------------------------------------------------- static resources ---
_u = np.linspace(0, 2 * np.pi, 40)
_v = np.linspace(0, np.pi, 24)
_sx = np.outer(np.cos(_u), np.sin(_v))
_sy = np.outer(np.sin(_u), np.sin(_v))
_sz = np.outer(np.ones_like(_u), np.cos(_v))
_A = _Rg @ np.diag(_stds)                           # unit sphere -> 1 sigma
ELL = [MU[i] + _A[i, 0] * _sx + _A[i, 1] * _sy + _A[i, 2] * _sz
       for i in range(3)]

GX, GY = np.meshgrid(np.linspace(-SENSOR, SENSOR, 220),
                     np.linspace(-SENSOR, SENSOR, 220))
def density(gx, gy, mu2, cov2):
    d = np.stack([gx - mu2[0], gy - mu2[1]], axis=-1)
    P = np.linalg.inv(cov2)
    return np.exp(-0.5 * np.einsum("...i,ij,...j", d, P, d))

MU2_0, COV2_0 = project(C0, R0)                     # frame-0 reference
E1_0 = ellipse_pts(MU2_0, COV2_0, 1.0)

# white -> splat colour colormap for the density image
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("splat", ["#ffffff", COL_SPLAT])

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(12.8, 6.4), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0],
                      left=0.0, right=0.97, top=0.88, bottom=0.07, wspace=0.04)
ax3 = fig.add_subplot(gs[0], projection="3d")
ax2 = fig.add_subplot(gs[1])

# panel titles as figure text so they sit at exactly the same height;
# underline by hand (matplotlib has no underline) using the rendered extent
_titles = []
for _ax, _label in ((ax3, "World space"), (ax2, "Image space")):
    _p = _ax.get_position()
    _titles.append(fig.text((_p.x0 + _p.x1) / 2, 0.908, _label, ha="center",
                            va="bottom", fontsize=19, fontweight="bold"))
fig.canvas.draw()
for _t in _titles:
    (_x0, _y0), (_x1, _) = fig.transFigure.inverted().transform(
        _t.get_window_extent())
    fig.add_artist(plt.Line2D([_x0, _x1], [_y0 - 0.012] * 2, color="black",
                              lw=1.6, transform=fig.transFigure))

# phase label, centred under the 3D panel; updated every frame in draw()
_p3 = ax3.get_position()
phase_text = fig.text((_p3.x0 + _p3.x1) / 2, 0.07, "", ha="center",
                      va="bottom", fontsize=20)

trail = []

def draw_world(ax3, c, R, mu2, cov2):
    """Render the 3D world-space panel onto ax3."""
    ax3.cla()
    ax3.set_axis_off()
    ax3.set_box_aspect((1, 1, 0.62))
    if TILES:                                       # larger motion envelope
        ax3.set_xlim(-6.3, 1.4)
        ax3.set_ylim(-5.4, 1.5)
        ax3.set_zlim(-2.4, 3.1)
    else:
        ax3.set_xlim(-5.3, 1.3)
        ax3.set_ylim(-4.4, 1.4)
        ax3.set_zlim(-1.5, 1.9)
    ax3.view_init(elev=16, azim=-118)
    ax3.dist = 4.5 if TILES else 4.0                # zoom in on the scene

    # gaussian (1 sigma ellipsoid)
    ax3.plot_surface(ELL[0], ELL[1], ELL[2], rstride=2, cstride=2,
                     color=COL_GAUSS, alpha=0.30, linewidth=0, shade=True)
    ax3.plot_wireframe(ELL[0], ELL[1], ELL[2], rstride=8, cstride=6,
                       color=COL_GAUSS, alpha=0.25, linewidth=0.5)

    # frustum
    corners = np.array([[-SENSOR, -SENSOR], [SENSOR, -SENSOR],
                        [SENSOR, SENSOR], [-SENSOR, SENSOR]]).T
    cw = plane_to_world(c, R, corners)
    for i in range(4):
        ax3.plot(*np.stack([c, cw[:, i]]).T, color=COL_CAM, lw=1.0)
    loop = np.concatenate([cw, cw[:, :1]], axis=1)
    ax3.plot(*loop, color=COL_CAM, lw=1.2)
    ax3.add_collection3d(Poly3DCollection([cw.T], facecolor="#888888",
                                          alpha=0.08))
    ax3.scatter(*c, color=COL_CAM, s=25)

    # ray camera -> gaussian mean, and the projected splat on the plane:
    # the same density + 1 sigma outline as the 2D panel, with the density
    # mapped to opacity (nudged toward the camera to avoid z-fighting)
    ax3.plot(*np.stack([c, MU]).T, color="#999999", lw=0.9, ls="--")
    # nested ellipse layers whose cumulative opacity follows the gaussian
    # falloff, approximating the 2D panel's density image without the mesh
    # artifacts of a semi-transparent plot_surface
    levels = np.linspace(3.2, 0.05, 25)
    opacity = 0.9 * np.exp(-0.5 * levels**2)
    prev = 0.0
    for k_lvl, f_lvl in zip(levels, opacity):
        a = (f_lvl - prev) / (1 - prev)
        prev = f_lvl
        e = plane_to_world(c, R, ellipse_pts(mu2, cov2, k_lvl))
        ax3.add_collection3d(Poly3DCollection([e.T], facecolor=COL_SPLAT,
                                              edgecolor="none", alpha=a))
    e = plane_to_world(c, R, ellipse_pts(mu2, cov2, 1.0))
    ax3.plot(*e, color=COL_SPLAT, lw=1.5)
    pm = plane_to_world(c, R, mu2[:, None])
    ax3.scatter(*pm[:, 0], color=COL_SPLAT, s=12)


def draw_image(ax2, mu2, cov2):
    """Render the 2D image-plane panel onto ax2."""
    ax2.cla()
    dens = density(GX, GY, mu2, cov2)
    ax2.imshow(dens, extent=[-SENSOR, SENSOR, -SENSOR, SENSOR],
               origin="lower", cmap=CMAP, vmin=0, vmax=1, zorder=0)

    if TILES:
        t = 2 * SENSOR / TILES
        # light up every tile the gaussian visibly renders on
        for ix in range(TILES):
            for iy in range(TILES):
                x0, y0 = -SENSOR + ix * t, -SENSOR + iy * t
                m = ((GX >= x0) & (GX < x0 + t) &
                     (GY >= y0) & (GY < y0 + t))
                if dens[m].max() > 0.05:
                    ax2.add_patch(Rectangle((x0, y0), t, t,
                                            facecolor="#ffd54f",
                                            edgecolor="#f0a020",
                                            lw=2.0, alpha=0.35, zorder=1))
        for i in range(1, TILES):
            x = -SENSOR + i * t
            ax2.axvline(x, color="#bbbbbb", lw=1.0, zorder=1)
            ax2.axhline(x, color="#bbbbbb", lw=1.0, zorder=1)

    ax2.plot(*E1_0, color="#888888", lw=1.4, ls="--", zorder=2,
             label="initial pose (1$\\sigma$)")
    e1 = ellipse_pts(mu2, cov2, 1.0)
    ax2.plot(*e1, color=COL_SPLAT, lw=2.0, zorder=3, label="current (1$\\sigma$)")
    if len(trail) > 1:
        tr = np.array(trail).T
        ax2.plot(*tr, color=COL_TRAIL, lw=1.4, alpha=0.8, zorder=4)
    ax2.plot(*mu2, "o", color=COL_TRAIL, ms=6, zorder=5)

    ax2.set_xlim(-SENSOR, SENSOR)
    ax2.set_ylim(-SENSOR, SENSOR)
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_linewidth(1.4)
    ax2.legend(loc="lower right", fontsize=12, framealpha=0.9)

    # centre displacement, as % of image width
    shift = 100 * np.linalg.norm(mu2 - MU2_0) / (2 * SENSOR)
    ax2.text(0.02, 0.03, f"centre shift: {shift:4.1f}% of image",
             transform=ax2.transAxes, fontsize=14, color="#444444")


def draw(frame):
    c, R, k = camera_pose(frame)
    mu2, cov2 = project(c, R)
    trail.append(mu2)
    draw_world(ax3, c, R, mu2, cov2)
    draw_image(ax2, mu2, cov2)
    phase_text.set_text(f"Camera motion: {PHASES[k]}")
    ax2.apply_aspect()                              # resolve the square box
    phase_text.set_y(ax2.get_position().y0)        # align with its bottom edge
    return []

# -------------------------------------------------------------------- run ---
if __name__ == "__main__":
    if "--preview" in sys.argv:
        # frames at the motion extremes, where clipping would show first
        for name, fr in [("p1_translation", N_PHASE // 8),
                         ("p2_rotation", N_PHASE + N_PHASE // 8),
                         ("p3_both", 2 * N_PHASE + 5 * N_PHASE // 8)]:
            trail.clear()
            for f in range(fr + 1):                 # build trail honestly
                c, R, _ = camera_pose(f)
                trail.append(project(c, R)[0])
            trail.pop()
            draw(fr)
            fig.savefig(f"/home/nicholas/splatting/anim/preview_{name}.png")
            print("saved preview", name)
    else:
        anim = animation.FuncAnimation(fig, draw, frames=N_FRAMES)
        out = f"/home/nicholas/splatting/anim/gaussian_projection{SUFFIX}.mp4"
        anim.save(out, writer=animation.FFMpegWriter(
            fps=FPS, bitrate=10000,
            extra_args=["-pix_fmt", "yuv420p"]))    # plays in PowerPoint/Keynote
        print("saved", out)
