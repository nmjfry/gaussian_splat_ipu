#!/usr/bin/env bash
# Render the two Manim scenes and composite them side by side (+ a GIF).
# The build-time prefix is only needed if extensions must recompile; at
# runtime manim loads the system pango/cairo .so, so no special env is needed.
set -e
cd /home/nicholas/splatting/anim
MANIM=.venv-manim/bin/manim
MEDIA=/home/nicholas/splatting/anim/media_manim
SRC=gaussian_projection_manim.py

# --- render both scenes (square-ish, 30fps, white bg already set in scene) ---
$MANIM -r 1680,1400 --fps 30 --format mp4 --media_dir "$MEDIA" \
       --disable_caching -o WorldScene "$SRC" WorldScene
$MANIM -r 1400,1400 --fps 30 --format mp4 --media_dir "$MEDIA" \
       --disable_caching -o ImageScene "$SRC" ImageScene

WS=$(find "$MEDIA" -name WorldScene.mp4 | head -1)
IS=$(find "$MEDIA" -name ImageScene.mp4 | head -1)
echo "world: $WS"
echo "image: $IS"

# --- composite side by side ---
OUT=/home/nicholas/splatting/anim/gaussian_projection_manim.mp4
ffmpeg -y -loglevel error -i "$WS" -i "$IS" \
  -filter_complex "[0:v][1:v]hstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -pix_fmt yuv420p -c:v libx264 -crf 17 "$OUT"
echo "composite: $OUT"

# --- gif ---
GIF=/home/nicholas/splatting/anim/gaussian_projection_manim.gif
ffmpeg -y -loglevel error -i "$OUT" \
  -vf "fps=15,scale=1600:-1:flags=lanczos,palettegen=stats_mode=diff" /tmp/pal_manim.png
ffmpeg -y -loglevel error -i "$OUT" -i /tmp/pal_manim.png \
  -lavfi "fps=15,scale=1600:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4" "$GIF"
echo "gif: $GIF"
echo ALL_DONE
