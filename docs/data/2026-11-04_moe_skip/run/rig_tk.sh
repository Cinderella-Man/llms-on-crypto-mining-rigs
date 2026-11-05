#!/usr/bin/env bash
# Q660 (C120) on the rig: what skipping the routed experts that LLAMA_MOE_TAIL drops does to k-token passes of Swift
# Qwen3.8-Flash-Next on nine P106-090s. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
#   check  test-backend-ops on one P106 (gpu_checks.sh MODE=native: the skipped-slot cases on the Pascal kernels, -mtok,
#          llama.cpp's kernels, the tail op and the fused router chain); the PTX run covered the rest
#   tk     T(k) on real agent text at k = 1-8, C271's method (2026-11-04_kreal_p106: one llama-server load per arm, per k
#          one request whose 1,680 prompt tokens run in micro-batches of exactly k, the pipelining off; T(k) = prompt_ms
#          / (1,680 / k)), one arm per setting of the switch (ARMS, in this order):
#            off     LLAMA_MOE_TAIL unset (this build as -kshape runs)
#            p090r   p=0.9,renorm=1: 8.77 of 10 experts kept on the agent text (C122), the dropped ones not read
#            p080r   p=0.8,renorm=1: 7.27 of 10 kept
#            p090rr  p=0.9,renorm=1,repeat=1: the same cut, the dropped slots reading their token's last kept expert
#                    again at weight 0 instead of being skipped (Q661: whether the P106's 1.5 MiB L2 serves the repeats)
#            off2    as off, a second load at the end (the spread between loads)
#          every arm with hist=0 (the histogram reads the graph back after each micro-batch)
#   prof   the first 340 tokens at k = 1 and 5 under GGML_CUDA_PROFILE_MODE=group, arms PROF_ARMS (off p090r): the
#          routed experts' part of the pass (C271's parts)
# Usage: BIN=<bin/ of the unpacked moeskip tarball, the extras in it> MODEL=<Swift Q2_0 first shard> TEXT=<the agent
#   text file of C271> OUT=<dir> [STEPS='check tk prof'] [ARMS=...] [PROF_ARMS=...] [CARD=0] [REQ=<kreal_req.py>]
#   [PORT=8097] [TS=6,6,6,6,5,6,5,6,3] [CTX=8192] [N=1680] [KS='1 2 3 4 5 6 7 8'] bash rig_tk.sh
# Output: OUT/check/console.txt, OUT/requests.tsv (arm, k, prompt_n, prompt_ms, micro-batches, ms per micro-batch),
#   OUT/prof-<arm>.csv, OUT/server-<arm>.log, OUT/status.txt
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
STEPS=${STEPS:-check tk prof}
ARMS=${ARMS:-off p090r p080r p090rr}
PROF_ARMS=${PROF_ARMS:-off p090r}
CARD=${CARD:-0}
REQ=${REQ:-$HERE/../../2026-11-04_kreal_p106/run/kreal_req.py}
N=${N:-1680}; KS=${KS:-1 2 3 4 5 6 7 8}; PROF_N=${PROF_N:-340}; PROF_KS=${PROF_KS:-1 5}
PORT=${PORT:-8097}; TS=${TS:-6,6,6,6,5,6,5,6,3}; CTX=${CTX:-8192}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
step() { [[ " $STEPS " == *" $1 "* ]]; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
tail_of() { # the LLAMA_MOE_TAIL of an arm
    case $1 in
        off|off2) echo "" ;;  # off2: the switch off again, a second load (the spread between loads)
        p090r)  echo "p=0.9,renorm=1,hist=0" ;;
        p080r)  echo "p=0.8,renorm=1,hist=0" ;;
        p090rr) echo "p=0.9,renorm=1,hist=0,repeat=1" ;;
        *)      echo "?" ;;
    esac
}

if step check; then
    log "check start"
    MODE=native BIN="$BIN" OUT="$OUT/check" DEV="$CARD" RUNS="skip skip-mmvq skip-mtok0 tail" bash "$HERE/gpu_checks.sh" \
        > "$OUT/check.out" 2>&1
    log "check exit $? ($(grep -c ' 0 FAIL ' "$OUT/check/console.txt" 2>/dev/null) runs without FAIL lines)"
fi

# one llama-server load per arm and step; <arm> <step> <env...>
serve_and_send() {
    local arm=$1 st=$2 n=$3 ks=$4
    shift 4
    local mt
    mt=$(tail_of "$arm")
    [ "$mt" = "?" ] && { log "unknown arm $arm"; return; }
    pkill -x llama-server; sleep 2; quiet
    # the token embeddings on the CPU (where they are anyway) turn the pipelining off, as a verify pass runs
    env ${mt:+LLAMA_MOE_TAIL=$mt} "$@" "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -np 1 --cache-ram 0 \
        --no-warmup -b 2048 -ub 512 -ts "$TS" -fit off -lm dio --no-host --no-webui -ot token_embd.weight=CPU \
        --host 127.0.0.1 --port "$PORT" > "$OUT/server-$st-$arm.log" 2>&1 < /dev/null &
    local srv=$!
    for i in $(seq 1 450); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then log "$st $arm: server did not start"; tail -5 "$OUT/server-$st-$arm.log"; return; fi
    log "$st $arm: server up (LLAMA_MOE_TAIL=${mt:-unset} $*)"
    # shellcheck disable=SC2086
    python3 -I "$REQ" --port "$PORT" --text "$TEXT" --n "$n" --ks $ks --arm "$st-$arm" --out "$OUT/requests.tsv" \
        >> "$OUT/status.txt" 2>&1
    log "$st $arm: requests exit $?"
    kill -INT $srv 2> /dev/null; sleep 8; pkill -9 -x llama-server
}

if step tk; then
    : "${MODEL:?}" "${TEXT:?}"
    for arm in $ARMS; do
        serve_and_send "$arm" tk "$N" "$KS"
    done
fi

if step prof; then
    : "${MODEL:?}" "${TEXT:?}"
    for arm in $PROF_ARMS; do
        serve_and_send "$arm" prof "$PROF_N" "$PROF_KS" GGML_CUDA_PROFILE="$OUT/prof-$arm.csv" GGML_CUDA_PROFILE_MODE=group
    done
fi

quiet
touch "$OUT/DONE"
