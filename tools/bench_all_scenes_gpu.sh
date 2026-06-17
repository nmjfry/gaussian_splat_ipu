#!/bin/bash
# GPU counterpart to bench_all_scenes.sh — runs the same trajectories on the
# GPU via diff-gaussian-rasterization, sampling nvidia-smi for power.
#
# Run on the HOST (NOT inside the IPU docker container — the container doesn't
# have CUDA). Before running:
#   cd /nethome/$USER/workspace/gaussian_splat_ipu
#   source .venv-gpu/bin/activate
#   bash tools/bench_all_scenes_gpu.sh [SPINS] [SCENE1 SCENE2 ...]
#
# Defaults: SPINS=2, scenes = pringles sloth chairs salad.
#
# Outputs (in cwd):
#   bench_results_gpu/<scene>/benchmark_profile.csv
#   bench_results_gpu/<scene>/power_samples.csv
#   bench_results_gpu/<scene>/benchmark_last_frame.png
#   bench_results_gpu/<scene>/stdout.log
#
# After completion the summariser is invoked so the table can be compared
# directly against the IPU's bench_results/.

set -e

MODE="orbit"
if [ "${1:-}" = "static" ] || [ "${1:-}" = "orbit" ] || \
   [ "${1:-}" = "teleport" ] || [ "${1:-}" = "slow-orbit" ]; then
  MODE="$1"
  shift
fi

if [ "$MODE" = "static" ]; then
  COUNT="${1:-2000}"
  OUTDIR="bench_results_gpu_static"
  shift || true
elif [ "$MODE" = "teleport" ]; then
  NUM_POSES="${1:-40}"
  HOLD_FRAMES="${2:-30}"
  OUTDIR="bench_results_gpu_teleport"
  shift 2 || true
elif [ "$MODE" = "slow-orbit" ]; then
  FRAMES_PER_ORBIT="${1:-2880}"
  SPINS="${2:-1}"
  OUTDIR="bench_results_gpu_orbit_slow"
  shift 2 || true
else
  COUNT="${1:-2}"
  OUTDIR="bench_results_gpu"
  shift || true
fi

if [ $# -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=(pringles sloth chairs salad)
fi

mkdir -p "$OUTDIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU: ${GPU_NAME:-unknown}"
if [ "$MODE" = "static" ]; then
  echo "Mode: static (${COUNT} frames per scene)"
elif [ "$MODE" = "teleport" ]; then
  echo "Mode: teleport (${NUM_POSES} poses x ${HOLD_FRAMES} held = $((NUM_POSES * HOLD_FRAMES)) frames per scene)"
elif [ "$MODE" = "slow-orbit" ]; then
  echo "Mode: slow-orbit (${FRAMES_PER_ORBIT} frames per 360-deg orbit, ${SPINS} spins per scene)"
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

  PLY_PATH="data/${scene}.ply"
  TRAJ_PATH="tools/benchmark_traj_${scene}.traj"
  if [ ! -f "$PLY_PATH" ]; then
    echo "  ! $PLY_PATH not found, skipping"
    continue
  fi
  if [ ! -f "$TRAJ_PATH" ]; then
    echo "  ! $TRAJ_PATH not found, skipping"
    continue
  fi

  POWER_CSV="$SCENE_DIR/power_samples.csv"
  echo "timestamp,power_w,gpu_util,mem_util,temp_c" > "$POWER_CSV"

  (
    while true; do
      line=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,temperature.gpu \
                        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [ -n "$line" ]; then
        ts=$(date +%s.%N)
        echo "$ts,$line" >> "$POWER_CSV"
      fi
      sleep 1
    done
  ) &
  SAMPLE_PID=$!

  if [ "$MODE" = "static" ]; then
    POSE_JSON="tools/benchmark_pose_${scene}.json"
    if [ ! -f "$POSE_JSON" ]; then
      echo "  ! $POSE_JSON not found, skipping"
      kill $SAMPLE_PID 2>/dev/null || true
      continue
    fi
    python3 tools/render_gpu_dgr.py \
      --ply "$PLY_PATH" \
      --out "$SCENE_DIR/benchmark_last_frame.png" \
      --out-csv "$SCENE_DIR/benchmark_profile.csv" \
      --pose-json "$POSE_JSON" \
      --bench-static "$COUNT" 2>&1 | tee "$SCENE_DIR/stdout.log"
  elif [ "$MODE" = "teleport" ]; then
    TELEPORT_TRAJ="tools/benchmark_traj_teleport_${scene}.traj"
    ORBIT_JSON="tools/benchmark_path_${scene}.json"
    if [ ! -f "$TELEPORT_TRAJ" ]; then
      if [ ! -f "$ORBIT_JSON" ]; then
        echo "  ! $ORBIT_JSON not found, skipping"
        kill $SAMPLE_PID 2>/dev/null || true
        continue
      fi
      echo "  Generating teleport trajectory: $TELEPORT_TRAJ"
      python3 tools/sample_teleport_path.py "$ORBIT_JSON" \
          --out "$TELEPORT_TRAJ" \
          --num-poses "$NUM_POSES" \
          --hold-frames "$HOLD_FRAMES"
    fi
    python3 tools/render_gpu_dgr.py \
      --ply "$PLY_PATH" \
      --out "$SCENE_DIR/benchmark_last_frame.png" \
      --out-csv "$SCENE_DIR/benchmark_profile.csv" \
      --play-path "$TELEPORT_TRAJ" \
      --spins 1 2>&1 | tee "$SCENE_DIR/stdout.log"
  elif [ "$MODE" = "slow-orbit" ]; then
    SLOW_TRAJ="tools/benchmark_traj_${scene}_slow${FRAMES_PER_ORBIT}.traj"
    ORBIT_JSON="tools/benchmark_path_${scene}.json"
    if [ ! -f "$ORBIT_JSON" ]; then
      echo "  ! $ORBIT_JSON not found, skipping"
      kill $SAMPLE_PID 2>/dev/null || true
      continue
    fi
    echo "  Generating slow-orbit trajectory ($FRAMES_PER_ORBIT frames/orbit): $SLOW_TRAJ"
    python3 tools/sample_orbit_path.py "$ORBIT_JSON" \
        --out "$SLOW_TRAJ" --frames "$FRAMES_PER_ORBIT" --span-deg 360
    python3 tools/render_gpu_dgr.py \
      --ply "$PLY_PATH" \
      --out "$SCENE_DIR/benchmark_last_frame.png" \
      --out-csv "$SCENE_DIR/benchmark_profile.csv" \
      --play-path "$SLOW_TRAJ" \
      --spins "$SPINS" 2>&1 | tee "$SCENE_DIR/stdout.log"
  else
    python3 tools/render_gpu_dgr.py \
      --ply "$PLY_PATH" \
      --out "$SCENE_DIR/benchmark_last_frame.png" \
      --out-csv "$SCENE_DIR/benchmark_profile.csv" \
      --play-path "$TRAJ_PATH" \
      --spins "$COUNT" 2>&1 | tee "$SCENE_DIR/stdout.log"
  fi

  kill $SAMPLE_PID 2>/dev/null || true
  wait $SAMPLE_PID 2>/dev/null || true
  echo ""
done

echo "============================================================"
echo "All scenes done. Summarising..."
echo "============================================================"
python3 tools/summarise_bench.py "$OUTDIR"
