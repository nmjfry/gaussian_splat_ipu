#!/bin/bash
# Run a benchmark for every scene and collect per-scene CSVs.
#
# Two modes:
#   orbit  (default): each scene does N full loops of its --play trajectory
#   static          : each scene renders N frames at the fixed --from-pose
#
# Usage (from the build/ directory in the container):
#   bash ../tools/bench_all_scenes.sh                          # orbit, 2 spins, all scenes
#   bash ../tools/bench_all_scenes.sh 3                        # orbit, 3 spins
#   bash ../tools/bench_all_scenes.sh static 2000              # static, 2000 frames
#   bash ../tools/bench_all_scenes.sh static 2000 chairs salad # only these scenes
#   bash ../tools/bench_all_scenes.sh orbit 2 pringles sloth   # explicit orbit form

set -e

MODE="orbit"
if [ "${1:-}" = "static" ] || [ "${1:-}" = "orbit" ] || \
   [ "${1:-}" = "teleport" ] || [ "${1:-}" = "slow-orbit" ]; then
  MODE="$1"
  shift
fi

# Mode-specific defaults / outputs
if [ "$MODE" = "static" ]; then
  COUNT="${1:-2000}"
  OUTDIR="bench_results_static"
  shift || true
elif [ "$MODE" = "teleport" ]; then
  NUM_POSES="${1:-40}"
  HOLD_FRAMES="${2:-30}"
  OUTDIR="bench_results_teleport"
  shift 2 || true
elif [ "$MODE" = "slow-orbit" ]; then
  FRAMES_PER_ORBIT="${1:-2880}"
  SPINS="${2:-1}"
  OUTDIR="bench_results_orbit_slow"
  shift 2 || true
else
  COUNT="${1:-2}"
  OUTDIR="bench_results"
  shift || true
fi

if [ $# -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=(pringles sloth chairs salad)
fi

mkdir -p "$OUTDIR"
if [ "$MODE" = "static" ]; then
  echo "Mode: static (${COUNT} frames per scene)"
elif [ "$MODE" = "teleport" ]; then
  echo "Mode: teleport (${NUM_POSES} random poses x ${HOLD_FRAMES} held frames = $((NUM_POSES * HOLD_FRAMES)) frames per scene)"
elif [ "$MODE" = "slow-orbit" ]; then
  ANGULAR_VEL=$(python3 -c "print(360.0/$FRAMES_PER_ORBIT)")
  echo "Mode: slow-orbit (${FRAMES_PER_ORBIT} frames per 360-deg orbit = ${ANGULAR_VEL} deg/frame, ${SPINS} spins per scene)"
else
  echo "Mode: orbit (${COUNT} spins per scene)"
fi
echo "Scenes: ${SCENES[*]}"
echo "Output dir: $OUTDIR"
echo ""

for scene in "${SCENES[@]}"; do
  echo "============================================================"
  echo "  Scene: $scene"
  echo "============================================================"
  SCENE_DIR="$OUTDIR/$scene"
  mkdir -p "$SCENE_DIR"

  PLY_PATH="../data/${scene}.ply"
  if [ ! -f "$PLY_PATH" ]; then
    echo "  ! $PLY_PATH not found, skipping"
    continue
  fi

  POWER_CSV="$SCENE_DIR/power_samples.csv"
  echo "timestamp,power_w,die_temp_c,board_temp_c,ipu_util" > "$POWER_CSV"

  # Power sampler in the background
  (
    while true; do
      line=$(gc-monitor -s --csv-output --no-headers 2>/dev/null | head -1)
      if [ -n "$line" ]; then
        power=$(echo "$line"   | cut -d',' -f19 | tr -d ' W')
        die_temp=$(echo "$line" | cut -d',' -f17 | tr -d ' C')
        brd_temp=$(echo "$line" | cut -d',' -f18 | tr -d ' C')
        ipu_util=$(echo "$line" | cut -d',' -f26 | tr -d ' %')
        ts=$(date +%s.%N)
        echo "$ts,$power,$die_temp,$brd_temp,$ipu_util" >> "$POWER_CSV"
      fi
      sleep 1
    done
  ) &
  SAMPLE_PID=$!

  # Run the benchmark
  if [ "$MODE" = "static" ]; then
    POSE_JSON="../tools/benchmark_pose_${scene}.json"
    if [ ! -f "$POSE_JSON" ]; then
      echo "  ! $POSE_JSON not found, skipping"
      kill $SAMPLE_PID 2>/dev/null || true
      continue
    fi
    GCDA_MONITOR=1 ./src/main/splat \
        --input "$PLY_PATH" \
        --flip-scene \
        --device ipu \
        --from-pose "$POSE_JSON" \
        --bench-static "$COUNT" 2>&1 | tee "$SCENE_DIR/stdout.log"
  elif [ "$MODE" = "teleport" ]; then
    TELEPORT_TRAJ="../tools/benchmark_traj_teleport_${scene}.traj"
    ORBIT_JSON="../tools/benchmark_path_${scene}.json"
    if [ ! -f "$TELEPORT_TRAJ" ]; then
      if [ ! -f "$ORBIT_JSON" ]; then
        echo "  ! $ORBIT_JSON not found, skipping"
        kill $SAMPLE_PID 2>/dev/null || true
        continue
      fi
      echo "  Generating teleport trajectory: $TELEPORT_TRAJ"
      python3 ../tools/sample_teleport_path.py "$ORBIT_JSON" \
          --out "$TELEPORT_TRAJ" \
          --num-poses "$NUM_POSES" \
          --hold-frames "$HOLD_FRAMES"
    fi
    GCDA_MONITOR=1 ./src/main/splat \
        --input "$PLY_PATH" \
        --flip-scene \
        --device ipu \
        --play-path "$TELEPORT_TRAJ" \
        --benchmark 1 2>&1 | tee "$SCENE_DIR/stdout.log"
  elif [ "$MODE" = "slow-orbit" ]; then
    SLOW_TRAJ="../tools/benchmark_traj_${scene}_slow${FRAMES_PER_ORBIT}.traj"
    ORBIT_JSON="../tools/benchmark_path_${scene}.json"
    if [ ! -f "$ORBIT_JSON" ]; then
      echo "  ! $ORBIT_JSON not found, skipping"
      kill $SAMPLE_PID 2>/dev/null || true
      continue
    fi
    echo "  Generating slow-orbit trajectory ($FRAMES_PER_ORBIT frames/orbit): $SLOW_TRAJ"
    python3 ../tools/sample_orbit_path.py "$ORBIT_JSON" \
        --out "$SLOW_TRAJ" --frames "$FRAMES_PER_ORBIT" --span-deg 360
    GCDA_MONITOR=1 ./src/main/splat \
        --input "$PLY_PATH" \
        --flip-scene \
        --device ipu \
        --play-path "$SLOW_TRAJ" \
        --benchmark "$SPINS" 2>&1 | tee "$SCENE_DIR/stdout.log"
  else
    GCDA_MONITOR=1 ./src/main/splat \
        --input "$PLY_PATH" \
        --flip-scene \
        --device ipu \
        --play \
        --benchmark "$COUNT" 2>&1 | tee "$SCENE_DIR/stdout.log"
  fi

  # Stop sampler
  kill $SAMPLE_PID 2>/dev/null || true
  wait $SAMPLE_PID 2>/dev/null || true

  # Move per-run outputs into the scene dir
  [ -f benchmark_profile.csv ] && mv benchmark_profile.csv "$SCENE_DIR/"
  [ -f benchmark_last_frame.png ] && mv benchmark_last_frame.png "$SCENE_DIR/"

  echo ""
done

echo "============================================================"
echo "All scenes done. Summarising..."
echo "============================================================"
python3 ../tools/summarise_bench.py "$OUTDIR"
