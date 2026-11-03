#!/usr/bin/env bash
# The variants and their scores as a pipeline: the CPU builds the variants in order (variants.sh) while the GPUs score
# each one as soon as it is written (kld.sh), so a variant's round trip overlaps the previous one's KLD runs.
#   (the environment of variants.sh and kld.sh) [WAIT_PID=<pid of a kld.sh run still on the GPUs>] bash all.sh FORMAT...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/variants.sh" "$@" &
gen=$!
[ -n "${WAIT_PID:-}" ] && while kill -0 "$WAIT_PID" 2> /dev/null; do sleep 30; done
for f in "$@"; do
    until [ -f "$WORK/variants/$f/done" ]; do
        kill -0 "$gen" 2> /dev/null || [ -f "$WORK/variants/$f/done" ] || { echo "variants.sh stopped before $f"; exit 1; }
        sleep 30
    done
    bash "$HERE/kld.sh" "$MODEL" "$WORK/variants/$f"
done
wait "$gen"
