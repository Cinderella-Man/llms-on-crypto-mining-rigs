#!/usr/bin/env bash
# Q550: what pipelined speculation costs each card for Qwen3.8-Flash-Next. llama-server loads the model and
# --pipe-depth D, and is stopped as soon as it listens. Its log gives per device the recurrent-state buffer (once
# before llama_pipe_init: one group; again after it: its 2 D + 3 state groups for 2 (D + 1) pipe slots and a stash) and
# the pipe slots' compute buffers (one-token graphs over the whole context), and whether pipelining came up or failed
# to allocate ("failed to set up the pipe slots"). Read once it listens: nvidia-smi's memory per card (gpu-d<D>.txt)
# and the server's host memory (/proc/<pid>/status: status-d<D>.txt). Nothing is timed. memory_table.py turns the
# logs into one table.
#
# On the rig (nine cards): run it there, on the real file or on header-only sparse copies of it (the first MB of each
# shard, `truncate -s` to the full size): the buffers are the same and zero weights load faster. On a machine with
# fewer GPUs, GGML_CUDA_DEVICES=9 gives nine virtual devices round-robin over the real ones (the sizes are per
# device), but nvidia-smi then shows the real GPUs.
#
# Usage (environment): BIN=<dir with llama-server> MODEL=<gguf, first shard> OUT=<dir> [DEPTHS='0 1 2 3']
#   [ARGS='-c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fa on -fit off -lm dio -ngl 99'] [PORT=18555] [WAIT=1200]
#   bash memory.sh
# e.g. from another machine: ssh $RIG "BIN=<build>/bin MODEL=<gguf> OUT=<dir> bash -s" < memory.sh
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}" "${OUT:?set OUT}"
DEPTHS=${DEPTHS:-0 1 2 3}
ARGS=${ARGS:--c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fa on -fit off -lm dio -ngl 99}
PORT=${PORT:-18555}
mkdir -p "$OUT"
for D in $DEPTHS; do
    log=$OUT/server-d$D.log
    echo "=== depth $D $(date +%T)"
    # -lv 5: llama-server's default verbosity (3) hides libllama's buffer lines, which are what this reads (R31 ran
    # without it and got only nvidia-smi and the host's memory)
    # shellcheck disable=SC2086
    "$BIN/llama-server" -m "$MODEL" $ARGS -lv 5 -np 1 -bs --pipe-depth "$D" --no-webui --host 127.0.0.1 \
        --port "$PORT" > "$log" 2>&1 < /dev/null &
    pid=$!
    for _ in $(seq 1 "${WAIT:-1200}"); do
        grep -q "listening on" "$log" && break
        kill -0 $pid 2> /dev/null || break
        sleep 1
    done
    # memory once loaded (pinned buffers are anonymous memory; the model's mapped pages are RssFile)
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits > "$OUT/gpu-d$D.txt"
    fi
    grep -E "^(VmRSS|RssAnon|RssFile|RssShmem|VmLck|VmPin):" /proc/$pid/status > "$OUT/status-d$D.txt" 2> /dev/null
    kill $pid 2> /dev/null; wait $pid 2> /dev/null
    grep -E "pipelined speculation|failed to set up the pipe|state groups|pipe slots|RS buffer|compute buffer size|listening on|out of memory|failed to allocate" \
        "$log" | tail -n 40
    [ -s "$OUT/gpu-d$D.txt" ] && sed 's/^/nvidia-smi: /' "$OUT/gpu-d$D.txt"
done
echo "=== memory ended $(date +%T)"
