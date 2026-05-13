#!/bin/bash
# GPU benchmark: run the DGR reference renderer in benchmark mode for each scene,
# sample power via nvidia-smi, and produce a CSV of (scene, FPS, watts, FPS/W).
#
# Usage:
#   ./tools/benchmark_gpu.sh [output.csv]
#
# Run on the HOST machine (not inside the Docker container) with CUDA available.
# Requires: python3 with torch, diff-gaussian-rasterization, plyfile installed.

set -euo pipefail

OUTPUT="${1:-gpu_benchmark_results.csv}"
FRAMES=200
POWER_LOG="/tmp/gpu_power_$$.csv"

# Scenes (same as IPU benchmark — adjust paths for host filesystem)
SCENES=(
  data/point_cloud_8.ply
  data/point_cloud_9.ply
  data/point_cloud_10.ply
  data/point_cloud_12.ply
)

echo "scene,fps,watts,fps_per_watt" > "$OUTPUT"

for scene in "${SCENES[@]}"; do
  if [ ! -f "$scene" ]; then
    echo "SKIP: $scene not found"
    continue
  fi

  echo "Benchmarking: $scene ($FRAMES frames)"

  # Start nvidia-smi power logging in background
  nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1 > "$POWER_LOG" 2>/dev/null &
  SMI_PID=$!
  sleep 2

  # Run GPU benchmark
  RESULT=$(python3 tools/render_gpu_dgr.py \
    --ply "$scene" --out /tmp/gpu_bench_frame.png \
    --benchmark "$FRAMES" 2>&1 | grep "^BENCHMARK:" || true)

  # Stop nvidia-smi
  kill "$SMI_PID" 2>/dev/null || true
  wait "$SMI_PID" 2>/dev/null || true

  if [ -z "$RESULT" ]; then
    echo "  ERROR: no benchmark output for $scene"
    continue
  fi

  FPS=$(echo "$RESULT" | grep -oP 'fps=\K[0-9.]+')
  echo "  FPS: $FPS"

  # Parse mean power from nvidia-smi log
  AVG_POWER=$(awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}' "$POWER_LOG" 2>/dev/null || echo "0")
  echo "  Avg power: ${AVG_POWER}W"

  if [ "$AVG_POWER" != "0" ] && [ -n "$AVG_POWER" ]; then
    FPSW=$(python3 -c "print(f'{$FPS / $AVG_POWER:.4f}')")
  else
    FPSW="N/A"
  fi

  echo "$scene,$FPS,$AVG_POWER,$FPSW" >> "$OUTPUT"
  rm -f "$POWER_LOG"
done

echo ""
echo "Results saved to: $OUTPUT"
cat "$OUTPUT"
