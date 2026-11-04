#!/usr/bin/env bash
# Q895: the compute 6.1 build of agentionai's fork checked on a newer NVIDIA card through its PTX (the driver compiles
# the 6.1 PTX for the card, and the fork's host code takes the 6.1 paths: ggml_cuda_highest_compiled_arch), in Docker:
#   1. test-backend-ops (test mode) against the CPU for every case of the trellis types (TQ2_T, TQK6, TQK7, TQP5-7)
#      in MUL_MAT, MUL_MAT_ID, MUL_MAT_ID_FUSION, MUL_MAT_VEC_FUSION and GET_ROWS of the fork's test list
#   2. p106_tq.sh with PARTS="check perf" as a smoke run of the P106 run file: its cases run and parse. The times of a
#      newer card running Pascal code are no measurement of a P106 and go into no table
#   (env: BIN OUT GPU [IMAGE] [MOUNT] [DOCKER_ARGS] [OPS] [TYPES] [PARAMS] [TEST_ENV] [SMOKE]) bash ptx_check.sh
# BIN and OUT must lie under MOUNT (default: the folder above BIN), which the container mounts at the same path.
# OPS and TYPES narrow step 1 (a space-separated op list; a regex of type names), PARAMS adds a regex the case's
# parameters must also match, TEST_ENV sets variables for step 1 (e.g. GGML_CUDA_TQ_MMVQ_ROWS=0: a newer card takes
# the one-token layout a P106 takes), SMOKE=0 skips step 2. The trellis encoder makes MUL_MAT_ID the slow step: ~110
# cases a type, ~8 a minute on a loaded 12-core host.
set -euo pipefail
: "${BIN:?}" "${OUT:?}" "${GPU:?}"
OPS=${OPS:-MUL_MAT MUL_MAT_ID MUL_MAT_ID_FUSION MUL_MAT_VEC_FUSION GET_ROWS}
TYPES=${TYPES:-tq2_t|tqk6|tqk7|tqp5|tqp6|tqp7}
SMOKE=${SMOKE:-1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MOUNT=${MOUNT:-$(dirname "$(dirname "$BIN")")}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cp "$HERE/p106_tq.sh" "$OUT/"
docker run --rm --label pascal-quants=1 ${DOCKER_ARGS:---cpu-shares=256} --name pq-gyro-p106-ptx \
    --gpus "\"device=$GPU\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_CACHE_PATH="$OUT/jitcache" \
    -e CUDA_CACHE_MAXSIZE=4294967296 -e BIN="$BIN" -e OUT="$OUT" -e OPS="$OPS" -e TYPES="$TYPES" -e SMOKE="$SMOKE" \
    -e PARAMS="${PARAMS:-}" -e TEST_ENV="${TEST_ENV:-}" -v "$MOUNT:$MOUNT" "$IMAGE" bash -euo pipefail -c '
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/device.txt"
    re="($TYPES)"
    [ -z "$PARAMS" ] || re="$re.*$PARAMS"
    echo "params regex: $re; env: ${TEST_ENV:-none}" | tee -a "$OUT/summary.txt"
    for op in $OPS; do
        start=$(date +%s)
        env $TEST_ENV nice -n 19 "$BIN/test-backend-ops" test -b CUDA0 -o $op -p "$re" > "$OUT/test_$op.log" 2>&1 || true
        echo "$op: $(grep -E "tests passed" "$OUT/test_$op.log" | tail -n 1) ($(( $(date +%s) - start )) s)" | tee -a "$OUT/summary.txt"
    done
    grep -hE "FAIL" "$OUT"/test_*.log | head -n 40 | tee -a "$OUT/summary.txt" || true
    [ "$SMOKE" = 1 ] || exit 0
    start=$(date +%s)
    PARTS="check perf" CARD=0 OUT="$OUT/p106_tq" nice -n 19 bash "$OUT/p106_tq.sh" > "$OUT/p106_tq.out" 2>&1 || true
    echo "p106_tq.sh smoke: $(tail -n 1 "$OUT/p106_tq/run.log") ($(( $(date +%s) - start )) s)" | tee -a "$OUT/summary.txt"
'
touch "$OUT/done"
