#!/usr/bin/env bash
# Checks llama-pipespec's streams mode (several conversations staggered through the cards) against separate plain runs:
# conversation c of every streams job must give the same tokens, bit-identical logits for every token (or the same
# sampled ids with sampling on the device), the same KV state and the same probe logits as a plain run of its prompt
# with seed + c and the same prompt chunk size.
#
# Jobs, in one process (one model load; the context is split into N KV streams):
#   ref-c<c>       plain decoding of conversation c alone, in its sequence c (its prompt in chunks of CHUNK tokens)
#   all            N conversations, all joining at once, no guesses
#   arrive         conversation c joins after ARRIVE * c generated tokens: prompt chunks interleave with other
#                  conversations' tokens
#   slack          guesses (copy >= 4, else continuation counts, depth 2) in a budget of N + 2 items in flight
#   nobudget       the same guesser with a budget of one item in flight: a guess never fits, so there is none
# A timeout catches deadlocks.
#
# Usage (environment variables):
#   BIN=<dir with llama-pipespec> MODEL=<gguf> [ARGS='-ngl 99 -fa on'] [N=3] [CTX=4096 per conversation]
#   [PROMPTS='prompts/chat.txt,prompts/rewrite.txt,prompts/code.txt'] [NTOK=128] [CHUNK=64] [ARRIVE=16]
#   [SAMPLING='--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42'] [PROBE=8] [OUT=/tmp/pipespec_streams]
#   [TIMEOUT=1800] [SPEED=1] bench/pipespec/check_streams.sh
# Sampling on the device: ARGS='... --pipe-backend-sample' (or --pipe-backend-greedy with --temp 0).
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
HERE=$(cd "$(dirname "$0")" && pwd)
ARGS=${ARGS:--ngl 99 -fa on}
N=${N:-3}
CTX=${CTX:-4096}
PROMPTS=${PROMPTS:-$HERE/prompts/chat.txt,$HERE/prompts/rewrite.txt,$HERE/prompts/code.txt}
NTOK=${NTOK:-128}
CHUNK=${CHUNK:-64}
ARRIVE=${ARRIVE:-16}
SAMPLING=${SAMPLING:---temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42}
PROBE=${PROBE:-8}
OUT=${OUT:-/tmp/pipespec_streams}
TIMEOUT=${TIMEOUT:-1800}
SPEED=${SPEED:-1}
mkdir -p "$OUT"

seed=$(echo "$SAMPLING" | grep -o -- '--seed [0-9]*' | awk '{print $2}')
seed=${seed:-42}
IFS=',' read -r -a files <<< "$PROMPTS"

jobs=$OUT/jobs.txt
common="n=$NTOK probe=$PROBE chat=1 chunk=$CHUNK"
{
    for ((c = 0; c < N; c++)); do
        echo "name=ref-c$c mode=plain prompt=${files[c % ${#files[@]}]} seed=$((seed + c)) seq=$c $common"
    done
    echo "name=all streams=$N prompts=$PROMPTS seed=$seed depth=0 guess=none $common"
    echo "name=arrive streams=$N prompts=$PROMPTS seed=$seed depth=0 guess=none arrive=$ARRIVE $common"
    echo "name=slack streams=$N prompts=$PROMPTS seed=$seed depth=2 guess=cont copy_min=4 budget=$((N + 2)) arrive=$ARRIVE $common"
    echo "name=nobudget streams=$N prompts=$PROMPTS seed=$seed depth=2 guess=cont copy_min=4 budget=1 arrive=$ARRIVE $common"
} > "$jobs"

# shellcheck disable=SC2086
timeout "$TIMEOUT" "$BIN/llama-pipespec" -m "$MODEL" $ARGS -c $((N * CTX)) -np "$N" $SAMPLING --pipe-jobs "$jobs" \
    > "$OUT/run.log" 2>&1
rc=$?
[ $rc -eq 124 ] && echo "TIMEOUT after $TIMEOUT s (deadlock?)"
[ $rc -ne 0 ] && { echo "llama-pipespec exit $rc"; grep -E "error|failed|ERR" "$OUT/run.log" | tail -5; }
grep -o 'PIPESPEC.*' "$OUT/run.log" > "$OUT/results.txt"

field() { echo " $1" | grep -o " $2=[^ ]*" | cut -d= -f2; }
printf '%-12s %6s %7s %6s %6s %6s %7s  %-6s %-6s %-6s %-6s  %s\n' run n_gen chunks guess hits misses tok/s tokens logits state probe verdict
fail=0
for j in all arrive slack nobudget; do
    for ((c = 0; c < N; c++)); do
        line=$(grep "name=$j-c$c " "$OUT/results.txt")
        ref=$(grep "name=ref-c$c " "$OUT/results.txt")
        if [ -z "$line" ] || [ -z "$ref" ]; then
            echo "$j-c$c: missing"; fail=1; continue
        fi
        verdict=same
        cols=()
        for k in tokens logits state probe; do
            if [ "$(field "$line" $k)" = "$(field "$ref" $k)" ]; then cols+=(same); else cols+=(DIFF); verdict=DIFF; fi
        done
        [ "$verdict" = DIFF ] && fail=1
        speed=-; [ "$SPEED" = 1 ] && speed=$(field "$line" tok_s)
        printf '%-12s %6s %7s %6s %6s %6s %7s  %-6s %-6s %-6s %-6s  %s\n' "$j-c$c" "$(field "$line" n_gen)" \
            "$(field "$line" chunks)" "$(field "$line" guesses)" "$(field "$line" hits)" "$(field "$line" misses)" "$speed" \
            "${cols[@]}" "$verdict"
    done
done
nb_guesses=$(grep "name=nobudget-c" "$OUT/results.txt" | grep -o " guesses=[0-9]*" | cut -d= -f2 | awk '{s += $1} END {print s + 0}')
echo "guesses with a budget of one item: ${nb_guesses:-?} (must be 0)"
[ "${nb_guesses:-1}" = 0 ] || fail=1
exit $fail
