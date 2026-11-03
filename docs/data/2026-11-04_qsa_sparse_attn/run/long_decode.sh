#!/usr/bin/env bash
# long_decode.sh: Q230's long-context diagnostic, run ON the rig and detached (a dropped ssh connection cannot stop it).
# Per arm one llama-server with the same build and flags, GGML_CUDA_FATTN_SPARSE_PASCAL=0 (off: the dense kernels read
# every cell) or 1 (on), and long_requests.py (beside this script) against it: decode after fresh prompts of DEPTHS
# tokens. With PROFILE=1 the servers also write GGML_CUDA_PROFILE traces (prof-<arm>.csv; scripts/prof_summary.py
# gives FLASH_ATTN_EXT's time per token); use one depth then, so the decode part of the trace has a single context.
# LLAMA_QSA_INPUT_STATS logs the k-pool inputs' host time and bytes per ubatch into the server log.
#   ssh $RIG "BIN=<build>/bin MODEL=<first shard> OUT=<work dir> setsid -f nohup bash long_decode.sh > <work dir>/long.out 2>&1 < /dev/null"
# Env: BIN, MODEL, OUT; ARMS (default "off on"), DEPTHS (default 8192,32768), CTX (65536), ARGS (default the rig's
#      nine-card flags), PORT (8096), PROFILE (0)
# Output: OUT/long.csv (one line per request), OUT/server-<arm>.log, OUT/qsa_inputs-<arm>.txt, OUT/prof-<arm>.csv, OUT/done
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARMS=${ARMS:-off on}
DEPTHS=${DEPTHS:-8192,32768}
CTX=${CTX:-65536}
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
PORT=${PORT:-8096}
mkdir -p "$OUT"
rm -f "$OUT/done"
first=1
for arm in $ARMS; do
    case $arm in off) v=0 ;; on) v=1 ;; *) echo "unknown arm $arm"; continue ;; esac
    envs="GGML_CUDA_FATTN_SPARSE_PASCAL=$v LLAMA_QSA_INPUT_STATS=16"
    [ "${PROFILE:-0}" = 1 ] && envs="$envs GGML_CUDA_PROFILE=$OUT/prof-$arm.csv"
    LOG=$OUT/server-$arm.log
    echo "== $arm: $envs $(date -Is)"
    pkill -x llama-server; sleep 2
    env $envs "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" $ARGS -np 1 --cache-ram 0 --no-webui \
        --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
    srv=$!
    t0=$(date +%s)
    for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    echo "load $(( $(date +%s) - t0 )) s; cards $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    free -m | sed -n 2p
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        if [ $first = 1 ]; then
            python3 -I "$HERE/long_requests.py" "$PORT" "$arm" "$DEPTHS" | tee "$OUT/long.csv"
            first=0
        else
            python3 -I "$HERE/long_requests.py" "$PORT" "$arm" "$DEPTHS" | tail -n +2 | tee -a "$OUT/long.csv"
        fi
    else
        echo "server did not start"; tail -20 "$LOG"
    fi
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server
    grep -h "QSA k-pool inputs" "$LOG" | sed 's/^.*QSA k-pool inputs/QSA k-pool inputs/' > "$OUT/qsa_inputs-$arm.txt"
    grep -E "out of memory|failed|error" "$LOG" | grep -v "unused tensor" | head -5
done
echo "== end $(date -Is)"
touch "$OUT/done"
