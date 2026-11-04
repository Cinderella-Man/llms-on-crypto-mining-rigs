#!/usr/bin/env bash
# kreal.sh: T(k) on real text (C271 / Q827). One llama-server load per arm; per k one request whose prompt (the first
# N tokens of TEXT, the KV cache not reused) runs in micro-batches of exactly k tokens through the -taper branch's
# per-request "ubatch_taper": "short=1000000:k,min=1". A micro-batch of k consecutive real tokens is a k-token pass, as a
# verify of k drafted tokens is: T(k) = prompt_ms / (N / k). With several cards llama.cpp pipelines a call's micro-batches
# across them (the next one starts on card 0 while the last runs on card 1), which a verify of one micro-batch never
# does; any tensor override turns that off (llama_context: !has_tensor_overrides), so the server gets NOPIPE, by
# default the token embeddings on the CPU, where they are anyway. Runs ON the rig, detached; ends with $OUT/DONE.
# Arms (ARMS): def (every switch at its default), s32k (GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768), prof (s32k under
# GGML_CUDA_PROFILE_MODE=group, k = PROF_KS on the first PROF_N tokens, for the parts).
# Env: BIN, MODEL (first shard), TEXT (a text file), OUT; optional N (1680: every k of 1-8 divides it), KS (1..8),
# PROF_N (340), PROF_KS ("1 5"), PORT (8097), TS (6,6,6,6,5,6,5,6,3), CTX (8192), ARMS ("def s32k prof"), NOPIPE
# ("-ot token_embd.weight=CPU"; empty keeps the pipelining: the prompt throughput in micro-batches of k instead).
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
N=${N:-1680}; KS=${KS:-1 2 3 4 5 6 7 8}; PROF_N=${PROF_N:-340}; PROF_KS=${PROF_KS:-1 5}
PORT=${PORT:-8097}; TS=${TS:-6,6,6,6,5,6,5,6,3}; CTX=${CTX:-8192}; ARMS=${ARMS:-def s32k prof}
NOPIPE=${NOPIPE-"-ot token_embd.weight=CPU"}
mkdir -p "$OUT"; rm -f "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
for arm in $ARMS; do
    envs=(); ks=$KS; n=$N
    case $arm in
        def)  ;;
        s32k) envs=(GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768) ;;
        prof) envs=(GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_PROFILE="$OUT/prof-group.csv" GGML_CUDA_PROFILE_MODE=group); ks=$PROF_KS; n=$PROF_N ;;
        *) log "unknown arm $arm"; continue ;;
    esac
    pkill -x llama-server; sleep 2; quiet
    env "${envs[@]+"${envs[@]}"}" "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -np 1 --cache-ram 0 --no-warmup \
        -b 2048 -ub 512 -ts "$TS" -fit off -lm dio --no-host --no-webui $NOPIPE --host 127.0.0.1 --port "$PORT" \
        > "$OUT/server-$arm.log" 2>&1 < /dev/null &
    srv=$!
    for i in $(seq 1 450); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then log "$arm: server did not start"; tail -5 "$OUT/server-$arm.log"; continue; fi
    log "$arm: server up (${envs[*]:-defaults})"
    python3 -I "$HERE/kreal_req.py" --port "$PORT" --text "$TEXT" --n "$n" --ks $ks --arm "$arm" --out "$OUT/requests.tsv" >> "$OUT/status.txt" 2>&1
    log "$arm: requests exit $?"
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server
done
quiet
touch "$OUT/DONE"
