#!/usr/bin/env bash
# R41 (Q681's profiler on the release line): where a k-token pass of the Miner Llama v3 preset spends its time on nine
# P106-090s, at k = 1 and k = 4. C120r's method (2026-11-04_moe_skip/run/rig_tk.sh, C271's requests): per k one request
# whose prompt (the first N tokens of real agent text, the KV cache not reused) runs in micro-batches of exactly k
# tokens ("ubatch_taper": "short=1000000:k,min=1"); T(k) = the request's prompt time / (N / k).
# Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# Every load is the preset's line (PRESET_ENV, PRESET_ARGS) without the MTP draft and its arguments, plus EXTRA. With
# DRAFT set the draft is loaded as the preset loads it, and then no planned micro-batch holds fewer than
# --spec-draft-n-max + 2 tokens (the trailing tokens that carry the rollback snapshots stay in one micro-batch), so a
# request for k = 1 or 4 runs in passes of 5 tokens.
#   --no-warmup  keeps the load's warm-up evaluations out of the trace
#   -ot token_embd.weight=CPU  the token embeddings on the CPU (where they are anyway): any tensor override of the
#       model turns llama.cpp's pipelining of a call's micro-batches across the cards off, so that each micro-batch
#       runs alone, as a decode or verify pass does
# Loads (LOADS, in this order):
#   warm  unprofiled, N tokens at each k: warms the cards (the first load after an idle hour runs slow, C556); its
#         pass times are kept as the first load's
#   prof  under GGML_CUDA_PROFILE=$OUT/prof-release.csv GGML_CUDA_PROFILE_MODE=group: the first PROF_N tokens at each k
#   wall  unprofiled, N tokens at each k (the pass times), then the profile's own PROF_N tokens at each k
# Usage: BIN=<bin/ of the release build> MODEL=<Swift Q2_0 first shard> PRESET=<settings.env> TEXT=<the agent text file
#   of C271> OUT=<dir> [DRAFT=<the MTP draft>] [REQ=<kreal_req.py>] [LOADS='warm prof wall'] [KS='1 4'] [N=1680]
#   [PROF_N=340] [PORT=8097] [EXTRA=...] bash release_prof.sh
# Output: OUT/requests.tsv (arm, k, prompt_n, prompt_ms, micro-batches, ms per micro-batch; arm = the load, "wall340"
#   for the wall load's PROF_N-token requests), OUT/prof-release.csv, OUT/server-<load>.log, OUT/cards.txt (memory,
#   temperature and clock per card before each load, once loaded and after its requests), OUT/status.txt (with the
#   epoch time around every request, to place it in the trace)
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${PRESET:?}" "${TEXT:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
REQ=${REQ:-$HERE/../../2026-11-04_kreal_p106/run/kreal_req.py}
LOADS=${LOADS:-warm prof wall}
KS=${KS:-1 4}; N=${N:-1680}; PROF_N=${PROF_N:-340}; PORT=${PORT:-8097}
EXTRA=${EXTRA---no-warmup --no-webui -ot token_embd.weight=CPU}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
# the server's arguments: the preset's, with the draft (DRAFT set) or without it and its arguments
args=()
if [ -n "${DRAFT:-}" ]; then
    # shellcheck disable=SC2206
    args=($PRESET_ARGS -md "$DRAFT")
else
    # shellcheck disable=SC2086
    set -- $PRESET_ARGS
    while [ $# -gt 0 ]; do
        case $1 in --spec-*|-otd|-md) shift 2 ;; *) args+=("$1"); shift ;; esac
    done
fi
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
cards() { # <label>: memory used, temperature and clock per card
    { echo "# $(date -Is) $1"; nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.gr --format=csv,noheader; } >> "$OUT/cards.txt"
}
send() { # <arm> <n> : one request per k, the epoch time around each
    local arm=$1 n=$2 k
    for k in $KS; do
        log "$arm k=$k n=$n: request start $(date +%s.%3N)"
        python3 -I "$REQ" --port "$PORT" --text "$TEXT" --n "$n" --ks "$k" --arm "$arm" --out "$OUT/requests.tsv" \
            >> "$OUT/status.txt" 2>&1
        log "$arm k=$k n=$n: request exit $? end $(date +%s.%3N)"
    done
}

for load in $LOADS; do
    envs=()
    case $load in
        warm|wall) ;;
        prof) envs=(GGML_CUDA_PROFILE="$OUT/prof-release.csv" GGML_CUDA_PROFILE_MODE=group) ;;
        *) log "unknown load $load"; continue ;;
    esac
    pkill -x llama-server; sleep 2; quiet; cards "$load: before the load"
    log "$load: server start $(date +%s.%3N)"
    # shellcheck disable=SC2086
    env $PRESET_ENV "${envs[@]+"${envs[@]}"}" "$BIN/llama-server" -m "$MODEL" "${args[@]}" $EXTRA \
        --host 127.0.0.1 --port "$PORT" > "$OUT/server-$load.log" 2>&1 < /dev/null &
    srv=$!
    for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; kill -0 $srv 2> /dev/null || break; sleep 2; done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then log "$load: server did not start"; tail -n 5 "$OUT/server-$load.log"; pkill -9 -x llama-server; continue; fi
    log "$load: server up ($PRESET_ENV ${envs[*]:-}; ${args[*]} $EXTRA)"
    cards "$load: loaded"
    case $load in
        warm) send warm "$N" ;;
        prof) send prof "$PROF_N" ;;
        wall) send wall "$N"; send "wall$PROF_N" "$PROF_N" ;;
    esac
    cards "$load: after the requests"
    # a clean exit, so that the trace is whole
    kill -INT $srv 2> /dev/null
    for i in $(seq 1 90); do kill -0 $srv 2> /dev/null || break; sleep 1; done
    log "$load: server stopped after ${i} s"
    pkill -9 -x llama-server
done
sleep 2; quiet; cards "end"
touch "$OUT/DONE"
