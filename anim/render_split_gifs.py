"""
Render the two panels of gaussian_projection_anim as separate high-res
animations, each with the "Camera motion: ..." label underneath:

    world_space.mp4 / .gif   - 3D camera frustum + gaussian
    image_space.mp4 / .gif   - 2D image-plane panel

Usage:
    python3 render_split_gifs.py            # renders both MP4s
    python3 render_split_gifs.py --preview  # one PNG per panel only
"""
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

import gaussian_projection_anim as g

OUT = "/home/nicholas/splatting/anim"

# ---- left panel: 3D world space -------------------------------------------
fig1 = plt.figure(figsize=(7.0, 6.6), dpi=240)
# nudged below the figure bottom so the scene sits centred above the caption
ax1 = fig1.add_axes([0.0, -0.20, 1.0, 1.10], projection="3d")
txt1 = fig1.text(0.5, 0.03, "", ha="center", va="bottom", fontsize=20)

def draw_left(frame):
    c, R, k = g.camera_pose(frame)
    mu2, cov2 = g.project(c, R)
    g.draw_world(ax1, c, R, mu2, cov2)
    txt1.set_text(f"Camera motion: {g.PHASES[k]}")
    return []

# ---- right panel: 2D image plane -------------------------------------------
fig2 = plt.figure(figsize=(6.6, 6.6), dpi=240)
ax2 = fig2.add_axes([0.05, 0.10, 0.90, 0.88])
txt2 = fig2.text(0.5, 0.025, "", ha="center", va="bottom", fontsize=20)

def draw_right(frame):
    if frame == 0:
        g.trail.clear()
    c, R, k = g.camera_pose(frame)
    mu2, cov2 = g.project(c, R)
    g.trail.append(mu2)
    g.draw_image(ax2, mu2, cov2)
    txt2.set_text(f"Camera motion: {g.PHASES[k]}")
    return []

# -------------------------------------------------------------------- run ---
if __name__ == "__main__":
    if "--preview" in sys.argv:
        fr = g.N_PHASE // 8                         # translation extreme
        draw_left(fr)
        fig1.savefig(f"{OUT}/preview_world_space.png")
        for f in range(fr + 1):
            draw_right(f)
        fig2.savefig(f"{OUT}/preview_image_space.png")
        print("saved previews")
    else:
        for fig, drawfn, name in [(fig1, draw_left, "world_space"),
                                  (fig2, draw_right, "image_space")]:
            anim = animation.FuncAnimation(fig, drawfn, frames=g.N_FRAMES)
            anim.save(f"{OUT}/{name}{g.SUFFIX}.mp4", writer=animation.FFMpegWriter(
                fps=g.FPS, bitrate=12000, extra_args=["-pix_fmt", "yuv420p"]))
            print("saved", name)
