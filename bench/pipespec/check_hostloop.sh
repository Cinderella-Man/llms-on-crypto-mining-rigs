#!/usr/bin/env bash
# Checks the host-side changes of the pipelined decode loop for exactness (packed input uploads, Q120; tokens fed on the
# devices, Q121). Run it with the layers split over two or more GPUs (or virtual devices, GGML_CUDA_DEVICES=N on one
# GPU), so that every input upload and the card-to-card token copy are exercised:
#   upload-<m>    check_identity.sh with LLAMA_PIPE_UPLOAD=m for m in 0 1 2 (0: one synchronous copy per host input,
#                 as llama_decode does; 1: one asynchronous copy per input; 2: each card's inputs in one asynchronous
#                 copy from pinned staging memory, the default), greedy sampling on the host: every pipelined run
#                 equals plain decoding bit for bit;
#   embd-gpu      the same with LLAMA_INPUT_ON_GPU=1 (the embedding lookup on the first card); embd_same reports whether
#                 its plain run's logits equal those of upload-2, where the CPU does the lookup;
#   feed-greedy   LLAMA_INPUT_ON_GPU=1, greedy sampling on the device and FEED=1: the fed run (each next token copied
#                 from the last card to the first, llama_pipe_submit_fed) and the other pipelined runs equal plain
#                 decoding;
#   feed-temp     the same with the temperature chain on the device (seeded, each draw keyed by its position);
#   streams-<N>   check_streams.sh with N conversations for N in NS, LLAMA_INPUT_ON_GPU=1, the temperature chain on the
#                 device and FEED=1: every conversation of every job, the fed ones included, equals its plain run;
#   streams-host  check_streams.sh with 4 conversations and host sampling (the packed uploads without a device sampler);
#   feed-refused  a fed run with the lookup on the CPU stops with an error instead of decoding a wrong token.
# Upload mode 2 is the default in every check but upload-0 and upload-1.
#
# Usage (environment variables):
#   BIN=<dir with llama-pipespec> MODEL=<gguf> [STREAMS_MODEL=<gguf, default MODEL>] [ARGS='-ngl 99 -fa on']
#   [NS='1 4 8'] [N=256] [NTOK=128] [OUT=/tmp/pipespec_hostloop] [TIMEOUT=1800] bench/pipespec/check_hostloop.sh
# It prints verdicts only, no timings. Exit status 0 when every check passes.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=${ARGS:--ngl 99 -fa on}
STREAMS_MODEL=${STREAMS_MODEL:-$MODEL}
NS=${NS:-1 4 8}
N=${N:-256}
NTOK=${NTOK:-128}
OUT=${OUT:-/tmp/pipespec_hostloop}
TIMEOUT=${TIMEOUT:-1800}
TEMP='--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42'
mkdir -p "$OUT"

summary=()
fail=0
# check <name> <script> [VAR=value ...]
check() {
    local name=$1 script=$2
    shift 2
    echo "== $name"
    if env "$@" BIN="$BIN" OUT="$OUT/$name" SPEED=0 TIMEOUT="$TIMEOUT" bash "$HERE/$script"; then
        summary+=("$name pass")
    else
        summary+=("$name FAIL")
        fail=1
    fi
}

ident=(DEPTHS="1 3" MISSES="0 0.5" N="$N" MODEL="$MODEL")
for m in 0 1 2; do
    check "upload-$m" check_identity.sh "${ident[@]}" LLAMA_PIPE_UPLOAD=$m ARGS="$ARGS -c 4096" SAMPLING="--temp 0"
done
check embd-gpu check_identity.sh "${ident[@]}" LLAMA_INPUT_ON_GPU=1 ARGS="$ARGS -c 4096" SAMPLING="--temp 0"
check feed-greedy check_identity.sh "${ident[@]}" LLAMA_INPUT_ON_GPU=1 FEED=1 ARGS="$ARGS -c 4096 --pipe-backend-greedy" \
    SAMPLING="--temp 0"
check feed-temp check_identity.sh "${ident[@]}" LLAMA_INPUT_ON_GPU=1 FEED=1 ARGS="$ARGS -c 4096 --pipe-backend-sample" \
    SAMPLING="$TEMP"
for n in $NS; do
    check "streams-$n" check_streams.sh MODEL="$STREAMS_MODEL" N="$n" NTOK="$NTOK" CTX=4096 LLAMA_INPUT_ON_GPU=1 FEED=1 \
        ARGS="$ARGS --pipe-backend-sample" SAMPLING="$TEMP"
done
check streams-host check_streams.sh MODEL="$STREAMS_MODEL" N=4 NTOK="$NTOK" CTX=4096 ARGS="$ARGS" SAMPLING="$TEMP"

# with the embedding lookup on the CPU the CPU reads the token ids, so a fed submit must refuse
echo "== feed-refused"
mkdir -p "$OUT/feed-refused"
echo "name=fed mode=pipe depth=0 guess=none feed=1 prompt=$HERE/prompts/short.txt n=8 chat=1" > "$OUT/feed-refused/jobs.txt"
# shellcheck disable=SC2086
if env -u LLAMA_INPUT_ON_GPU timeout "$TIMEOUT" "$BIN/llama-pipespec" -m "$MODEL" $ARGS -c 4096 --temp 0 \
        --pipe-backend-greedy --pipe-jobs "$OUT/feed-refused/jobs.txt" > "$OUT/feed-refused/run.log" 2>&1; then
    summary+=("feed-refused FAIL (a fed run went through with the lookup on the CPU)")
    fail=1
elif grep -q "a fed token needs" "$OUT/feed-refused/run.log"; then
    summary+=("feed-refused pass")
else
    summary+=("feed-refused FAIL (another error)")
    fail=1
fi

# the plain runs do not use the pipe API: the same in every upload mode
field() { echo " $1" | grep -o " $2=[^ ]*" | cut -d= -f2; }
plain() { field "$(grep 'name=plain ' "$OUT/$1/results.txt" 2>/dev/null)" logits; }
p0=$(plain upload-0)
for m in 1 2; do
    if [ -n "$p0" ] && [ "$(plain "upload-$m")" = "$p0" ]; then
        summary+=("plain-upload-$m same as upload-0")
    else
        summary+=("plain-upload-$m DIFF from upload-0")
        fail=1
    fi
done
pe=$(plain embd-gpu)
if [ -n "$pe" ] && [ "$pe" = "$(plain upload-2)" ]; then embd=yes; else embd=no; fi
summary+=("embd_same $embd (plain logits with the lookup on the first card vs on the CPU; informational)")

echo "== summary"
printf '%s\n' "${summary[@]}"
exit $fail
