#!/bin/bash
# Compare NEWS vs multiSlice-gather on the same scenes at a fixed pose.
# For each scene, renders COUNT static frames at benchmark_pose_<scene>.json with
# both --gather-mode news and --gather-mode multislice, sampling power, then
# summarises each into its own summary.csv.
#
# Usage (from build/ in the container):
#   bash ../tools/bench_compare_gather.sh                 # 2000 frames, 4 scenes
#   bash ../tools/bench_compare_gather.sh 1000 pringles   # 1000 frames, one scene
#
# Set FLIP=1 to add --flip-scene (match the original orbit benchmark); default
# off to match the interactive command. Check */benchmark_last_frame.png to
# confirm the scene rendered right.
set -e

COUNT="${1:-2000}"; shift || true
if [ $# -gt 0 ]; then SCENES=("$@"); else SCENES=(pringles sloth chairs salad); fi
FLIP_ARG=""; [ "${FLIP:-0}" = "1" ] && FLIP_ARG="--flip-scene"

run_mode() {  # $1=mode(news|multislice) $2=outdir
  local mode="$1" outdir="$2"
  mkdir -p "$outdir"
  for scene in "${SCENES[@]}"; do
    local ply="../data/${scene}.ply" pose="../tools/benchmark_pose_${scene}.json"
    [ -f "$ply" ]  || { echo "  ! $ply missing, skip";  continue; }
    [ -f "$pose" ] || { echo "  ! $pose missing, skip"; continue; }
    local sdir="$outdir/$scene"; mkdir -p "$sdir"
    local pcsv="$sdir/power_samples.csv"
    echo "timestamp,power_w,die_temp_c,board_temp_c,ipu_util" > "$pcsv"
    ( while true; do
        line=$(gc-monitor -s --csv-output --no-headers 2>/dev/null | head -1)
        [ -n "$line" ] && echo "$(date +%s.%N),$(echo "$line"|cut -d',' -f19|tr -d ' W'),\
$(echo "$line"|cut -d',' -f17|tr -d ' C'),$(echo "$line"|cut -d',' -f18|tr -d ' C'),\
$(echo "$line"|cut -d',' -f26|tr -d ' %')" >> "$pcsv"
        sleep 1
      done ) & local spid=$!
    echo "=== $mode / $scene ($COUNT frames) ==="
    GCDA_MONITOR=1 ./src/main/splat --input "$ply" $FLIP_ARG --device ipu \
        --from-pose "$pose" --bench-static "$COUNT" --gather-mode "$mode" \
        2>&1 | tee "$sdir/stdout.log"
    kill $spid 2>/dev/null || true; wait $spid 2>/dev/null || true
    [ -f benchmark_profile.csv ]   && mv benchmark_profile.csv   "$sdir/"
    [ -f benchmark_last_frame.png ] && mv benchmark_last_frame.png "$sdir/"
  done
}

run_mode news       bench_results_news
run_mode multislice bench_results_gather

echo "============================================================"
echo "NEWS summary:"
python3 ../tools/summarise_bench.py bench_results_news
echo "============================================================"
echo "multiSlice gather summary:"
python3 ../tools/summarise_bench_gather.py bench_results_gather
