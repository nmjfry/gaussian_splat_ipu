# Projection animations (presentation figures)

Scripts that generate the camera-projection animations for the EGSR 2026 talk:
a 3D camera frustum + Gaussian on the left (world space) and the resulting 2D
splat on the right (image space), illustrating that the projected Gaussian
barely moves as the camera makes small motions. The math is EWA splatting,
`Σ2D = J·R·Σ·Rᵀ·Jᵀ` with the perspective Jacobian `J`.

> Note: output paths inside the scripts are hardcoded to
> `/home/nicholas/splatting/anim/`. Edit those constants (or run from there) to
> render elsewhere. The scripts are committed as the exact generators that
> produced the slide media.

## Files

| File | What it makes |
|------|---------------|
| `gaussian_projection_anim.py` | Core math + the **combined side-by-side** MP4 (`gaussian_projection*.mp4`). Imported by the others. |
| `render_split_gifs.py` | Renders the two panels **separately** (`world_space*.mp4`, `image_space*.mp4`), each with a "Camera motion: …" caption. |
| `gaussian_projection_manim.py` | Self-contained **Manim** version (`WorldScene` + `ImageScene`). |
| `render_manim.sh` | Renders both Manim scenes, hstacks them, makes a GIF. |

## Dependencies

- `numpy`, `matplotlib`, `ffmpeg` on PATH (Agg backend + FFMpegWriter).
- **Figtree** font installed (downloaded from Google Fonts; the static
  `Figtree-Bold.ttf` is needed for bold weights).
- Manim CE 0.19.1 in a venv (`anim/.venv-manim/`, not committed).

## Rendering

### matplotlib — combined + split panels

```bash
# combined side-by-side MP4
python3 gaussian_projection_anim.py

# the two panels separately, as MP4s
python3 render_split_gifs.py
```

Variants are selected by environment variables (each appends a filename suffix
so variants don't overwrite each other):

| Env var | Effect | Suffix |
|---------|--------|--------|
| `GAUSS_SCALE=0.6` | shrink the Gaussian (tighter covariance) | `_tight` |
| `TILES=3` | 3×3 image-space grid, highlight the hit tile, enlarge motion to reach edge tiles | `_tiled` |
| `GENTLE=1` | scale motion amplitudes ×0.62 | `_gentle` |

```bash
# e.g. the tight + tiled combined animation used in the deck
GAUSS_SCALE=0.6 TILES=3 python3 gaussian_projection_anim.py
GAUSS_SCALE=0.6 TILES=3 python3 render_split_gifs.py
```

### Manim

```bash
./render_manim.sh   # renders WorldScene + ImageScene, hstacks, GIF
```

## GIFs for Google Slides

Full-frame gradient GIFs can fail to import into Google Slides. A slimmed
two-pass-palette encode (≈720px / 8 fps, ~3–4 MB) imports reliably:

```bash
SRC=image_space_tight_tiled.mp4
ffmpeg -i "$SRC" -vf "fps=8,scale=720:-1:flags=lanczos,palettegen=stats_mode=full" /tmp/pal.png
ffmpeg -i "$SRC" -i /tmp/pal.png \
  -lavfi "fps=8,scale=720:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4" \
  image_space_tiled_slides.gif
```

For the 2:1 side-by-side combined video, widen the scale to keep both panels'
text legible (e.g. `scale=1280:-1`).

> Do **not** run ImageMagick `-coalesce`/`+map` on these — it strips inter-frame
> optimization and makes the file much larger.
