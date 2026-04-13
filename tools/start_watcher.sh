#!/usr/bin/env bash
# Launch the GPU watcher in the background so screenshots taken via the
# remote UI automatically produce matching GPU reference renders.
#
# Run this on the HOST (not inside the container) — the watcher needs CUDA
# and lives in the .venv-gpu virtual-environment. The IPU server, meanwhile,
# runs in the Docker container and writes sidecar files into a bind-mounted
# path that both sides can see.
#
# Usage:
#   ./tools/start_watcher.sh              # defaults to paired_shots/
#   ./tools/start_watcher.sh foo/         # custom dir
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${1:-${REPO}/paired_shots}"
VENV="${REPO}/.venv-gpu/bin/activate"

if [[ ! -f "$VENV" ]]; then
  echo "No venv at $VENV; create it with 'python3 -m venv .venv-gpu' and" >&2
  echo "pip install torch diff-gaussian-rasterization plyfile Pillow numpy" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

mkdir -p "$DIR"
LOG="$DIR/gpu_watch.log"
echo "gpu_watch: writing to $DIR, log at $LOG"
exec python3 "$REPO/tools/gpu_watch.py" --dir "$DIR" 2>&1 | tee -a "$LOG"
