#!/usr/bin/env bash
# C880 (2026-11-21): the tools and tests that the batch-API change touches, run with one build's CPU binaries on fixed
# random models; run.sh runs it once with the base build and once with the head build, and compare.py compares the two.
# Nothing here is timed. Every run's exit code goes to $OUT/rc.tsv.
#   pipespec        bench/pipespec/check_identity.sh (plain decoding against pipelined decoding, one model load):
#                   ps-greedy (depths 1 and 3, misses 0 and 0.5), ps-chunk7 (the prompt in calls of 7 tokens:
#                   --pipe-chunk 7), ps-seeded (temperature 1, seed 42), ps-race2 (a race of two samples, plain mode
#                   decodes them round robin as sequences 0 and 1); bench/pipespec/check_streams.sh (ps-streams: two
#                   conversations in sequences 0 and 1, their plain references in calls of 16 prompt tokens)
#   prc             llama-pipe-recurrent-check: greedy, plain against three pipe slots with forced rollbacks
#   screen          llama-screen-check on the model with a screened head (add_screen_head.py): greedy, and sampled
#   moe-ids         test-moe-ids on the MoE model with 32 experts, 8 used
#   hop-cast        test-hop-cast --unit-only, and -m with the arms off and q8_0 (LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1)
#   kpool-dev, pool-mask-dev, kpool-stage
#                   C100's, C719's and C720's tests: per-sequence and unified KV, the model's top-k and 32, micro-batches
#                   of 16 and 512 tokens, seed 1, STEPS steps; every context's logits equal bit for bit
# Usage (inside a container with the build's libraries next to its binaries):
#   BIN=<build bin/> MD=<models: qwen4exp-moe.gguf, qwen4exp-moe-screen.gguf, e32/qwen4exp-moe.gguf, prompts.txt>
#   PS=<bench/pipespec> OUT=<dir> [THREADS=6] [STEPS=240] bash tools.sh
set -uo pipefail
: "${BIN:?BIN=<bin dir>}" "${MD:?MD=<models dir>}" "${PS:?PS=<bench/pipespec>}" "${OUT:?OUT=<out dir>}"
THREADS=${THREADS:-6}
STEPS=${STEPS:-240}
M=$MD/qwen4exp-moe.gguf
S=$MD/qwen4exp-moe-screen.gguf
E32=$MD/e32/qwen4exp-moe.gguf
mkdir -p "$OUT"
: > "$OUT/rc.tsv"
rec() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/rc.tsv"; echo "$(date +%T) $1 rc=$2"; }
RARGS="-ngl 0 -t $THREADS -c 4096 -fa on"
SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"

ci() { # <name> <env ...>: check_identity.sh into $OUT/<name>/ and $OUT/<name>.log
    local name=$1
    shift
    env "$@" BIN="$BIN" MODEL="$M" CHAT=0 SPEED=0 TIMEOUT=1800 PROMPT="$PS/prompts/rewrite.txt" OUT="$OUT/$name" \
        bash "$PS/check_identity.sh" > "$OUT/$name.log" 2>&1
    rec "$name" $?
}
ci ps-greedy ARGS="$RARGS" N=96 DEPTHS="1 3" MISSES="0 0.5"
ci ps-chunk7 ARGS="$RARGS --pipe-chunk 7" N=96 DEPTHS="1" MISSES="0.5"
ci ps-seeded ARGS="$RARGS" N=96 DEPTHS="1 3" MISSES="0.5" SAMPLING="$SAMPLING"
ci ps-race2  ARGS="$RARGS" N=64 DEPTHS="1" MISSES="0.5" RACE=2 SAMPLING="$SAMPLING"
env BIN="$BIN" MODEL="$M" ARGS="-ngl 0 -t $THREADS -fa on" N=2 CTX=2048 NTOK=48 CHUNK=16 ARRIVE=8 CHUNKS=2 PROBE=4 \
    SPEED=0 TIMEOUT=1800 SAMPLING="$SAMPLING" PROMPTS="$PS/prompts/chat.txt,$PS/prompts/rewrite.txt" OUT="$OUT/ps-streams" \
    bash "$PS/check_streams.sh" > "$OUT/ps-streams.log" 2>&1
rec ps-streams $?

timeout 900 "$BIN/llama-pipe-recurrent-check" -m "$M" -f "$PS/prompts/rewrite.txt" -n 96 $RARGS > "$OUT/prc.log" 2>&1
rec prc $?

timeout 900 "$BIN/llama-screen-check" -m "$S" -f "$MD/prompts.txt" -n 32 -ngl 0 -t "$THREADS" -c 1024 --temp 0 \
    > "$OUT/screen-greedy.log" 2>&1
rec screen-greedy $?
timeout 900 "$BIN/llama-screen-check" -m "$S" -f "$MD/prompts.txt" -n 32 -ngl 0 -t "$THREADS" -c 1024 \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234 > "$OUT/screen-sampled.log" 2>&1
rec screen-sampled $?

timeout 900 "$BIN/test-moe-ids" -m "$E32" -t "$THREADS" > "$OUT/moe-ids.log" 2>&1
rec moe-ids $?

timeout 900 "$BIN/test-hop-cast" --unit-only > "$OUT/hop-cast-unit.log" 2>&1
rec hop-cast-unit $?
timeout 900 "$BIN/test-hop-cast" -m "$M" -ngl 0 -t "$THREADS" --arm off --arm "q8_0 LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1" \
    --tsv "$OUT/hop-cast.tsv" > "$OUT/hop-cast.log" 2>&1
rec hop-cast-model $?

for t in kpool-dev pool-mask-dev kpool-stage; do
    for kv in seq unified; do
        for topk in 0 32; do
            for ub in 16 512; do
                arm=$t-$kv-k$topk-u$ub
                # shellcheck disable=SC2046
                timeout 1800 "$BIN/test-$t" -m "$M" -t "$THREADS" -s "$STEPS" -u "$ub" --top-k "$topk" --seed 1 \
                    $([ $kv = unified ] && echo --unified) > "$OUT/$arm.log" 2>&1
                rec "$arm" $?
            done
        done
    done
done
echo "=== tools done $(date -Is)"
