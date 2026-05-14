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

SPINS="${1:-2}"
shift || true
if [ $# -gt 0 ]; then
  SCENES=("$@")
else
  SCENES=(pringles sloth chairs salad)
fi

OUTDIR="bench_results_gpu"
mkdir -p "$OUTDIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "GPU: ${GPU_NAME:-unknown}"
echo "Running ${#SCENES[@]} scenes x ${SPINS} spins."
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

  python3 tools/render_gpu_dgr.py \
    --ply "$PLY_PATH" \
    --out "$SCENE_DIR/benchmark_last_frame.png" \
    --out-csv "$SCENE_DIR/benchmark_profile.csv" \
    --play-path "$TRAJ_PATH" \
    --spins "$SPINS" 2>&1 | tee "$SCENE_DIR/stdout.log"

  kill $SAMPLE_PID 2>/dev/null || true
  wait $SAMPLE_PID 2>/dev/null || true
  echo ""
done

echo "============================================================"
echo "All scenes done. Summarising..."
echo "============================================================"
python3 tools/summarise_bench.py "$OUTDIR"
