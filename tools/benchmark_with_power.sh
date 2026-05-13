#!/bin/bash
# Run benchmark with IPU power sampling.
# Usage: ./tools/benchmark_with_power.sh <splat args...>
# Example: ./tools/benchmark_with_power.sh --input ../data/point_cloud_9.ply --flip-scene --from-pose ../tools/benchmark_pose.json --benchmark 70

set -e

POWER_CSV="power_samples.csv"
echo "timestamp,power_w,die_temp_c,board_temp_c,ipu_util" > "$POWER_CSV"

# Start power sampling in background (1 sample/sec)
(
  while true; do
    line=$(gc-monitor -s --csv-output --no-headers 2>/dev/null | head -1)
    if [ -n "$line" ]; then
      power=$(echo "$line" | cut -d',' -f19 | tr -d ' W')
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

# Run the benchmark with sensor monitoring enabled
GCDA_MONITOR=1 ./src/main/splat "$@"

# Stop sampling
kill $SAMPLE_PID 2>/dev/null
wait $SAMPLE_PID 2>/dev/null || true

# Print summary
echo ""
echo "=== Power Summary ==="
awk -F',' 'NR>1 && $2+0 > 0 {
  n++; sum+=$2;
  if(NR==2 || $2+0 < min) min=$2+0;
  if(NR==2 || $2+0 > max) max=$2+0;
} END {
  if(n>0) printf "Samples: %d\nMean power: %.1f W\nMin power: %.1f W\nMax power: %.1f W\n", n, sum/n, min, max
}' "$POWER_CSV"
echo "Raw samples saved to $POWER_CSV"
