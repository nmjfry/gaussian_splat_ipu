#!/bin/bash
# IPU benchmark: run the splat server in benchmark mode for each scene,
# sample power via gc-monitor, and produce a CSV of (scene, FPS, watts, FPS/W).
#
# Usage:
#   ./tools/benchmark_ipu.sh [output.csv]
#
# Assumes the binary is at build/src/main/splat (inside the Docker container).
# Place your .ply files in data/ or adjust SCENES below.

set -euo pipefail

OUTPUT="${1:-ipu_benchmark_results.csv}"
FRAMES=200
BINARY="./build/src/main/splat"
POWER_LOG="/tmp/gc_power_$$.csv"

# List of scenes to benchmark (adjust paths as needed):
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

  # Start gc-monitor in background (if available)
  if command -v gc-monitor &>/dev/null; then
    gc-monitor --csv > "$POWER_LOG" 2>/dev/null &
    GC_PID=$!
    sleep 2
  else
    echo "  WARNING: gc-monitor not found — power will be reported as 0"
    GC_PID=""
  fi

  # Run benchmark
  RESULT=$("$BINARY" --input "$scene" --benchmark "$FRAMES" 2>&1 | grep "^BENCHMARK:" || true)

  # Stop gc-monitor
  if [ -n "$GC_PID" ]; then
    kill "$GC_PID" 2>/dev/null || true
    wait "$GC_PID" 2>/dev/null || true
  fi

  if [ -z "$RESULT" ]; then
    echo "  ERROR: no benchmark output for $scene"
    continue
  fi

  FPS=$(echo "$RESULT" | grep -oP 'fps=\K[0-9.]+')
  echo "  FPS: $FPS"

  # Parse gc-monitor CSV for average board power (column depends on device)
  # Adjust the awk column ($COL) based on your gc-monitor output format.
  AVG_POWER="0"
  if [ -f "$POWER_LOG" ] && [ -s "$POWER_LOG" ]; then
    # Try to find a "Board Power" or "power" column. Fallback: column 2.
    HEADER=$(head -1 "$POWER_LOG" 2>/dev/null || echo "")
    # Simple approach: take the last numeric column as power
    AVG_POWER=$(awk -F',' 'NR>1 && NF>1 {sum+=$NF; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}' "$POWER_LOG" 2>/dev/null || echo "0")
    echo "  Avg power: ${AVG_POWER}W"
  fi

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
