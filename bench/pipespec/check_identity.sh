#!/usr/bin/env bash
# Checks that pipelined speculation reproduces plain decoding exactly. For every pipelined run it compares with the
# plain run:
#   tokens  the generated tokens;
#   logits  the logits every token was sampled from, bit for bit (a hash per token);
#   state   the sequence's KV cache after generation (llama_state_seq_get_data);
#   probe   the logits of a few tokens decoded afterwards with llama_decode, which read every KV cell that is still
#           visible, so a cell left behind by a dropped guess would show.
# Pipelined runs, for each depth in DEPTHS:
#   - an oracle that replays the plain run's tokens with a share MISSES of the guesses made wrong on purpose (exercises
#     dropping guesses and rolling back their KV cells at a known rate);
#   - the copy lookup (match >= 2), copy >= 4 else continuation counts (cont), ngram-simple (4/16) and ngram-map-k
#     (4/8);
# plus depth 0 (the pipelined API without guesses). One process runs all of them with one model load; a timeout catches
# deadlocks.
#
# Usage (environment variables):
#   BIN=<dir with llama-pipespec> MODEL=<gguf> [ARGS='-ngl 99 -fa on -c 4096'] [PROMPT=prompts/rewrite.txt] [N=256]
#   [DEPTHS='1 2 3'] [MISSES='0 0.1 0.5'] [SAMPLING='--temp 0'] [PROBE=8] [OUT=/tmp/pipespec_check] [TIMEOUT=1800]
#   [SPEED=1] bench/pipespec/check_identity.sh
# SAMPLING='--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42' checks sampling with a fixed seed: the plain run
# then also fixes what the oracle replays. SPEED=0 leaves the tokens/s column out.
# RACE=K (1 = a single sample timed to the stop text) checks a race of K samples (seeds seed..seed+K-1) until one emits RACE_STOP (default "</think>"), after which
# the winner continues alone: the plain run decodes the samples round robin, the pipelined runs keep one token of each
# in flight and use the guessers on the winner. Also compared: the winner and every sample's tokens and logits (race).
# Use it with temperature sampling.
# FEED=1 adds pipe-d0-fed: depth 0 with every next token fed on the devices (llama_pipe_submit_fed), which needs a
# sampler on the device (--pipe-backend-greedy or --pipe-backend-sample in ARGS) and LLAMA_INPUT_ON_GPU=1.
# Environment variables of llama.cpp pass through (e.g. LLAMA_PIPE_UPLOAD=0|1|2, LLAMA_INPUT_ON_GPU=1).
# With the layers split over several GPUs (or over virtual devices, GGML_CUDA_DEVICES=N on one GPU) every hop between
# stages is exercised.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=${ARGS:--ngl 99 -fa on -c 4096}
PROMPT=${PROMPT:-$HERE/prompts/rewrite.txt}
N=${N:-256}
DEPTHS=${DEPTHS:-1 2 3}
MISSES=${MISSES:-0 0.1 0.5}
SAMPLING=${SAMPLING:---temp 0}
PROBE=${PROBE:-8}
OUT=${OUT:-/tmp/pipespec_check}
TIMEOUT=${TIMEOUT:-1800}
SPEED=${SPEED:-1}
RACE=${RACE:-0}
RACE_STOP=${RACE_STOP:-</think>}
FEED=${FEED:-0}
mkdir -p "$OUT"

jobs=$OUT/jobs.txt
common="prompt=$PROMPT n=$N probe=$PROBE chat=1"
[ "$RACE" -ge 1 ] && common="$common race=$RACE race_stop=$RACE_STOP"
{
    echo "name=plain mode=plain $common out=$OUT/plain.tok"
    echo "name=pipe-d0 mode=pipe depth=0 guess=none $common"
    [ "$FEED" = 1 ] && echo "name=pipe-d0-fed mode=pipe depth=0 guess=none feed=1 $common"
    for D in $DEPTHS; do
        for M in $MISSES; do
            echo "name=oracle-d$D-m$M mode=pipe depth=$D guess=oracle oracle=@plain miss=$M $common"
        done
        echo "name=copy2-d$D mode=pipe depth=$D guess=copy copy_min=2 $common"
        echo "name=cont4-d$D mode=pipe depth=$D guess=cont copy_min=4 $common"
        echo "name=simple-d$D mode=pipe depth=$D guess=ngram-simple $common"
        echo "name=mapk-d$D mode=pipe depth=$D guess=spec spec=ngram-map-k $common"
    done
} > "$jobs"

# shellcheck disable=SC2086
timeout "$TIMEOUT" "$BIN/llama-pipespec" -m "$MODEL" $ARGS $SAMPLING \
    --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 16 \
    --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 8 --pipe-jobs "$jobs" > "$OUT/run.log" 2>&1
rc=$?
[ $rc -eq 124 ] && echo "TIMEOUT after $TIMEOUT s (deadlock?)"
[ $rc -ne 0 ] && { echo "llama-pipespec exit $rc"; grep -E "error|failed|ERR" "$OUT/run.log" | tail -5; }
grep -o 'PIPESPEC.*' "$OUT/run.log" > "$OUT/results.txt"

field() { echo " $1" | grep -o " $2=[^ ]*" | cut -d= -f2; }
ref=$(grep 'name=plain ' "$OUT/results.txt")
[ -n "$ref" ] || { echo "no plain run"; exit 1; }
printf '%-16s %6s %6s %6s %6s %7s %7s  %-6s %-6s %-6s %-6s %-6s %-6s  %s\n' run n_gen guess hits misses wasted tok/s tokens logits state probe winner race verdict
fail=0
while read -r line; do
    verdict=same
    cols=()
    for k in tokens logits state probe winner race_hash; do
        if [ "$(field "$line" $k)" = "$(field "$ref" $k)" ]; then cols+=(same); else cols+=(DIFF); verdict=DIFF; fi
    done
    [ "$verdict" = DIFF ] && fail=1
    speed=-; [ "$SPEED" = 1 ] && speed=$(field "$line" tok_s)
    printf '%-16s %6s %6s %6s %6s %7s %7s  %-6s %-6s %-6s %-6s %-6s %-6s  %s\n' "$(field "$line" name)" "$(field "$line" n_gen)" \
        "$(field "$line" guesses)" "$(field "$line" hits)" "$(field "$line" misses)" "$(field "$line" wasted)" "$speed" \
        "${cols[@]}" "$verdict"
done < "$OUT/results.txt"
n_jobs=$(wc -l < "$jobs"); n_res=$(wc -l < "$OUT/results.txt")
[ "$n_res" -eq "$n_jobs" ] || { echo "only $n_res of $n_jobs runs finished"; fail=1; }
exit $fail
