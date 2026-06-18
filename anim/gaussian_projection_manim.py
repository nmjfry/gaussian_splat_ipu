"""
Manim version of the side-by-side 3DGS projection animation.

Two scenes, rendered separately and composited side by side (see
render_manim.sh):

    WorldScene  - 3D camera frustum + Gaussian, with the projected splat
                  drawn on the image plane (left)
    ImageScene  - the 2D image plane: projected Gaussian density, 1-sigma
                  ellipse, and the plus-sign trail of the centre (right)

The camera makes the same three-phase plus-sign motion as the matplotlib
version (translation -> rotation -> both); the projection is EWA splatting,
Sigma2D = J R Sigma R^T J^T.  Math is copied verbatim from
gaussian_projection_anim.py so the two stay numerically identical.

Render:
    .venv-manim/bin/manim -qh -r 1680,1400 gaussian_projection_manim.py WorldScene
    .venv-manim/bin/manim -qh -r 1400,1400 gaussian_projection_manim.py ImageScene
"""
import numpy as np
from manim import *

# ============================================================ scene math ====
MU = np.array([0.0, 0.0, 0.0])
_stds = np.array([0.95, 0.55, 0.32])


def _rot(axis, deg):
    a = np.deg2rad(deg)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])


_Rg = _rot(np.array([0.3, 1.0, 0.5]), 35.0)
SIGMA = _Rg @ np.diag(_stds**2) @ _Rg.T
_A = _Rg @ np.diag(_stds)                           # unit sphere -> 1 sigma

C0 = np.array([-4.2, -3.4, 1.3])
WORLD_UP = np.array([0.0, 0.0, 1.0])
FOCAL = 1.0
SENSOR = 0.45
D_PLANE = 1.0

FPS = 30
T_PHASE = 8.0
N_PHASE = int(FPS * T_PHASE)
N_FRAMES = 3 * N_PHASE
PHASES = ["Translation", "Rotation", "Translation + rotation"]

A_TRANS = (0.65, 0.45)
A_ROT = (6.0, 4.5)
G_BOTH = 0.7

COL = "#d53e4f"                                      # gaussian / splat red
COL_TRAIL = "#2c4b8f"
COL_CAM = "#333333"
COL_GREY = "#888888"


def look_at(c, target):
    z = target - c
    z = z / np.linalg.norm(z)
    x = np.cross(z, WORLD_UP)
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)
    return np.stack([x, y, z])


R0 = look_at(C0, MU)


def camera_pose(frame):
    k = min(frame // N_PHASE, 2)
    p = (frame - k * N_PHASE) / N_PHASE
    if p < 0.5:
        h, v = np.sin(4 * np.pi * p), 0.0
    else:
        h, v = 0.0, np.sin(4 * np.pi * (p - 0.5))
    do_t = k in (0, 2)
    do_r = k in (1, 2)
    g = G_BOTH if k == 2 else 1.0
    c = C0.copy()
    if do_t:
        c = c + g * (A_TRANS[0] * h * R0[0] + A_TRANS[1] * v * R0[1])
    R = R0
    if do_r:
        Ry = _rot(R0[1], g * A_ROT[0] * h)
        Rx = _rot(R0[0], g * A_ROT[1] * v)
        R = R0 @ (Ry @ Rx).T
    return c, R, k


def project(c, R):
    t = R @ (MU - c)
    tx, ty, tz = t
    mu2 = FOCAL * np.array([tx / tz, ty / tz])
    J = FOCAL * np.array([[1 / tz, 0, -tx / tz**2],
                          [0, 1 / tz, -ty / tz**2]])
    cov2 = J @ R @ SIGMA @ R.T @ J.T
    return mu2, cov2


def ellipse_pts(mu2, cov2, k=1.0, n=64):
    vals, vecs = np.linalg.eigh(cov2)
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(th), np.sin(th)])
    return mu2[:, None] + k * vecs @ (np.sqrt(vals)[:, None] * circ)


def plane_to_world(c, R, uv):
    return (c[:, None]
            + D_PLANE * ((uv[0] / FOCAL) * R[0][:, None]
                         + (uv[1] / FOCAL) * R[1][:, None]
                         + R[2][:, None]))


GX, GY = np.meshgrid(np.linspace(-SENSOR, SENSOR, 300),
                     np.linspace(-SENSOR, SENSOR, 300))


def density(gx, gy, mu2, cov2):
    d = np.stack([gx - mu2[0], gy - mu2[1]], axis=-1)
    P = np.linalg.inv(cov2)
    return np.exp(-0.5 * np.einsum("...i,ij,...j", d, P, d))


MU2_0, COV2_0 = project(C0, R0)

# ============================================================ world view ====
# map true world coords into manim frame: recentre then scale
CENTER = np.array([-1.7, -1.2, 0.35])
WS = 1.6


def T(p):
    return (np.asarray(p) - CENTER) * WS


def frame_index(tr):
    return int(round(tr.get_value()))


class WorldScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_camera_orientation(phi=72 * DEGREES, theta=-105 * DEGREES)
        tr = ValueTracker(0.0)

        # --- static gaussian ellipsoid ---
        def ell_func(u, v):
            p = MU + _A @ np.array([np.cos(u) * np.sin(v),
                                    np.sin(u) * np.sin(v), np.cos(v)])
            return T(p)

        gauss = Surface(ell_func, u_range=[0, TAU], v_range=[0, PI],
                        resolution=(28, 18), checkerboard_colors=False)
        gauss.set_fill(COL, opacity=0.33)
        gauss.set_stroke(COL, width=0.3, opacity=0.25)
        self.add(gauss)

        def pose():
            return camera_pose(frame_index(tr))

        # --- camera frustum ---
        def build_frustum():
            c, R, _ = pose()
            corners = np.array([[-SENSOR, -SENSOR], [SENSOR, -SENSOR],
                                [SENSOR, SENSOR], [-SENSOR, SENSOR]]).T
            cw = plane_to_world(c, R, corners)
            cwt = [T(cw[:, i]) for i in range(4)]
            ct = T(c)
            g = VGroup()
            for i in range(4):
                g.add(Line(ct, cwt[i], stroke_color=COL_CAM, stroke_width=2))
            g.add(Polygon(*cwt, stroke_color=COL_CAM, stroke_width=2.5,
                          fill_color=COL_GREY, fill_opacity=0.07))
            g.add(Dot3D(ct, color=COL_CAM, radius=0.05))
            return g

        # --- ray camera -> mean ---
        def build_ray():
            c, _, _ = pose()
            return DashedLine(T(c), T(MU), stroke_color=COL_GREY,
                              stroke_width=1.5, dash_length=0.07)

        # --- projected splat on the image plane ---
        def build_splat():
            c, R, _ = pose()
            mu2, cov2 = project(c, R)
            g = VGroup()
            levels = np.linspace(3.0, 0.08, 14)
            opacity = 0.92 * np.exp(-0.5 * levels**2)
            prev = 0.0
            for kk, fl in zip(levels, opacity):
                a = (fl - prev) / (1 - prev)
                prev = fl
                pts = plane_to_world(c, R, ellipse_pts(mu2, cov2, kk, n=44))
                g.add(Polygon(*[T(pts[:, i]) for i in range(pts.shape[1])],
                              stroke_width=0, fill_color=COL,
                              fill_opacity=float(a)))
            e = plane_to_world(c, R, ellipse_pts(mu2, cov2, 1.0, n=64))
            g.add(Polygon(*[T(e[:, i]) for i in range(e.shape[1])],
                          stroke_color=COL, stroke_width=2.5, fill_opacity=0))
            pm = plane_to_world(c, R, mu2[:, None])[:, 0]
            g.add(Dot3D(T(pm), color=COL, radius=0.035))
            return g

        ray = always_redraw(build_ray)
        splat = always_redraw(build_splat)
        frustum = always_redraw(build_frustum)
        self.add(ray, splat, frustum)

        # --- fixed-in-frame title + phase caption ---
        title = Text("World space", font="Figtree", weight=BOLD,
                     color=BLACK).scale(0.95).to_edge(UP, buff=0.3)
        ul = Underline(title, color=BLACK, buff=0.08)
        self.add_fixed_in_frame_mobjects(title, ul)

        for k in range(3):
            cap = Text(f"Camera motion: {PHASES[k]}", font="Figtree",
                       color=BLACK).scale(0.8).to_edge(DOWN, buff=0.45)
            self.add_fixed_in_frame_mobjects(cap)
            self.play(tr.animate.set_value((k + 1) * N_PHASE),
                      run_time=T_PHASE, rate_func=linear)
            self.remove(cap)


# ============================================================ image view ====
BOXC = np.array([0.0, -0.35, 0.0])
HALF = 3.25
IS = HALF / SENSOR


def to_screen(p2):
    return BOXC + np.array([p2[0] * IS, p2[1] * IS, 0.0])


def mk_ellipse(mu2, cov2, k):
    vals, vecs = np.linalg.eigh(cov2)
    ang = np.arctan2(vecs[1, 0], vecs[0, 0])
    e = Ellipse(width=2 * k * np.sqrt(vals[0]) * IS,
                height=2 * k * np.sqrt(vals[1]) * IS)
    e.rotate(ang)
    e.move_to(to_screen(mu2))
    return e


class ImageScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tr = ValueTracker(0.0)

        def pose():
            return camera_pose(frame_index(tr))

        def proj():
            c, R, _ = pose()
            return project(c, R)

        # --- density image (recomputed each frame, clipped to the box) ---
        red = np.array([213.0, 62.0, 79.0])

        def make_dens():
            mu2, cov2 = proj()
            dens = density(GX, GY, mu2, cov2)
            rgb = (1 - dens)[..., None] * 255.0 + dens[..., None] * red
            arr = np.dstack([rgb, np.full_like(dens, 255.0)]).astype(np.uint8)
            im = ImageMobject(np.flipud(arr))
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
            im.stretch_to_fit_width(2 * HALF)
            im.stretch_to_fit_height(2 * HALF)
            im.move_to(BOXC)
            return im

        dens_im = always_redraw(make_dens)
        box = Square(side_length=2 * HALF, color=BLACK,
                     stroke_width=3).move_to(BOXC)

        # --- initial-pose reference (dashed) + current 1-sigma ellipse ---
        init = DashedVMobject(
            mk_ellipse(MU2_0, COV2_0, 1.0).set_stroke(COL_GREY, 2.5),
            num_dashes=40)
        cur = always_redraw(
            lambda: mk_ellipse(*proj(), 1.0).set_stroke(COL, 3.5))

        # --- centre dot + traced plus-sign path ---
        dot = Dot(color=COL_TRAIL, radius=0.07)
        dot.add_updater(lambda m: m.move_to(to_screen(proj()[0])))
        trail = TracedPath(dot.get_center, stroke_color=COL_TRAIL,
                           stroke_width=3.5)

        # --- centre-shift readout (bottom-left, inside the box) ---
        def make_shift():
            mu2, _ = proj()
            s = 100 * np.linalg.norm(mu2 - MU2_0) / (2 * SENSOR)
            t = Text(f"centre shift: {s:4.1f}% of image", font="Figtree",
                     color="#444444").scale(0.46)
            t.move_to(box.get_corner(DL) + np.array([0.12, 0.12, 0]) +
                      np.array([t.width / 2, t.height / 2, 0]))
            return t

        shift = always_redraw(make_shift)

        # --- legend (bottom-right, inside the box) ---
        l1 = DashedLine(ORIGIN, RIGHT * 0.45, color=COL_GREY, stroke_width=2.5)
        t1 = Text("initial 1σ", font="Figtree", color=BLACK).scale(0.42)
        l2 = Line(ORIGIN, RIGHT * 0.45, color=COL, stroke_width=3.5)
        t2 = Text("current 1σ", font="Figtree", color=BLACK).scale(0.42)
        legend = VGroup(VGroup(l1, t1).arrange(RIGHT, buff=0.12),
                        VGroup(l2, t2).arrange(RIGHT, buff=0.12)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1)
        legbg = SurroundingRectangle(legend, color=COL_GREY, buff=0.1,
                                     fill_color=WHITE, fill_opacity=0.85,
                                     stroke_width=1.0)
        legend = VGroup(legbg, legend)
        legend.move_to(box.get_corner(DR) + np.array([-0.14, 0.14, 0]) -
                       np.array([legend.width / 2, -legend.height / 2, 0]))

        # shift up so this title lands at the same fraction-of-height as the
        # World-space title (the two scenes have different frame heights)
        title = Text("Image space", font="Figtree", weight=BOLD,
                     color=BLACK).scale(0.95).to_edge(UP, buff=0.3).shift(UP * 0.78)
        ul = Underline(title, color=BLACK, buff=0.08)

        self.add(dens_im, box, init, cur, trail, dot, shift, legend, title, ul)
        self.play(tr.animate.set_value(N_FRAMES),
                  run_time=3 * T_PHASE, rate_func=linear)
