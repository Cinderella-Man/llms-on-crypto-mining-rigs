#!/usr/bin/env bash
# C957 (2026-11-23): C880's tools.sh (docs/data/2026-11-21_c880_batch_api/run/tools.sh) on the cards: the tools and
# tests that the batch-API change touches, run with one rig build's binaries on fixed random models, the model's
# layers split over the visible cards. run.sh runs it once per build and setting, and compare.py compares the runs.
# Nothing here is timed. Every run's exit code goes to $OUT/rc.tsv. Parts (PARTS, any of):
#   ps      bench/pipespec/check_identity.sh (plain decoding against pipelined decoding, one model load): ps-greedy
#           (depths 1 and 3, misses 0 and 0.5), ps-chunk7 (the prompt in calls of 7 tokens), ps-seeded (temperature 1,
#           seed 42), ps-race2 (a race of two samples); bench/pipespec/check_streams.sh (ps-streams: two conversations
#           in sequences 0 and 1 against their plain references)
#   prc     llama-pipe-recurrent-check: greedy, plain against three pipe slots with forced rollbacks
#   screen  llama-screen-check on the model with a screened head ($SCREEN): greedy, and sampled
#   moe     test-moe-ids on the MoE model with 32 experts, 8 used ($E32), every layer on the cards: each MUL_MAT_ID's ids
#           read back after it ran
#   hop     test-hop-cast --unit-only (round trips through ggml_cast on the CPU and on each card, each card against
#           the CPU), and -m with every layer on the cards split -ts $TS, four arms: off (the reference, run twice),
#           q8_0 (LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1), staged (exact, GGML_CUDA_HOP_STAGED=2: every hop through a
#           pinned host ring) and mailbox (exact, GGML_CUDA_HOP_MAILBOX=2: every hop through mapped host memory)
#   pool    test-kpool-dev, test-pool-mask-dev, test-kpool-stage: per-sequence and unified KV, the model's top-k and 32,
#           micro-batches of 16 and 512 tokens, seed 1, STEPS steps (these three tests pin their model to the CPU
#           backend by design, as on C880's run; here they run from the CUDA build)
# Usage (inside a container with the cards and the build's bin/):
#   BIN=<build bin/> M=<model> [E32=<MoE model, 32 experts>] [SCREEN=<model with a screened head>] [PROMPTS=<file>]
#   PS=<bench/pipespec> OUT=<dir> [PARTS="ps prc screen moe hop pool"] [TS=1,1,1] [THREADS=6] [STEPS=240] bash gpu_tools.sh
set -uo pipefail
: "${BIN:?BIN=<bin dir>}" "${M:?M=<model>}" "${PS:?PS=<bench/pipespec>}" "${OUT:?OUT=<out dir>}"
PARTS=${PARTS:-ps prc screen moe hop pool}
TS=${TS:-1,1,1}
THREADS=${THREADS:-6}
STEPS=${STEPS:-240}
mkdir -p "$OUT"
: > "$OUT/rc.tsv"
rec() { printf '%s\t%s\n' "$1" "$2" >> "$OUT/rc.tsv"; echo "$(date +%T) $1 rc=$2"; }
part() { [[ " $PARTS " == *" $1 "* ]]; }
GARGS="-ngl 99 -ts $TS -t $THREADS"
RARGS="$GARGS -c 4096 -fa on"
SAMPLING="--temp 1 --top-k 20 --top-p 0.95 --min-p 0 --seed 42"

ci() { # <name> <env ...>: check_identity.sh into $OUT/<name>/ and $OUT/<name>.log
    local name=$1
    shift
    env "$@" BIN="$BIN" MODEL="$M" CHAT=0 SPEED=0 TIMEOUT=1800 PROMPT="$PS/prompts/rewrite.txt" OUT="$OUT/$name" \
        bash "$PS/check_identity.sh" > "$OUT/$name.log" 2>&1
    rec "$name" $?
}
if part ps; then
    ci ps-greedy ARGS="$RARGS" N=96 DEPTHS="1 3" MISSES="0 0.5"
    ci ps-chunk7 ARGS="$RARGS --pipe-chunk 7" N=96 DEPTHS="1" MISSES="0.5"
    ci ps-seeded ARGS="$RARGS" N=96 DEPTHS="1 3" MISSES="0.5" SAMPLING="$SAMPLING"
    ci ps-race2  ARGS="$RARGS" N=64 DEPTHS="1" MISSES="0.5" RACE=2 SAMPLING="$SAMPLING"
    env BIN="$BIN" MODEL="$M" ARGS="$GARGS -fa on" N=2 CTX=2048 NTOK=48 CHUNK=16 ARRIVE=8 CHUNKS=2 PROBE=4 \
        SPEED=0 TIMEOUT=1800 SAMPLING="$SAMPLING" PROMPTS="$PS/prompts/chat.txt,$PS/prompts/rewrite.txt" OUT="$OUT/ps-streams" \
        bash "$PS/check_streams.sh" > "$OUT/ps-streams.log" 2>&1
    rec ps-streams $?
fi

if part prc; then
    timeout 900 "$BIN/llama-pipe-recurrent-check" -m "$M" -f "$PS/prompts/rewrite.txt" -n 96 $RARGS > "$OUT/prc.log" 2>&1
    rec prc $?
fi

if part screen; then
    : "${SCREEN:?SCREEN=<model with a screened head>}" "${PROMPTS:?PROMPTS=<prompts file>}"
    timeout 900 "$BIN/llama-screen-check" -m "$SCREEN" -f "$PROMPTS" -n 32 $GARGS -c 1024 --temp 0 \
        > "$OUT/screen-greedy.log" 2>&1
    rec screen-greedy $?
    timeout 900 "$BIN/llama-screen-check" -m "$SCREEN" -f "$PROMPTS" -n 32 $GARGS -c 1024 \
        --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234 > "$OUT/screen-sampled.log" 2>&1
    rec screen-sampled $?
fi

if part moe; then
    : "${E32:?E32=<MoE model with 32 experts>}"
    timeout 900 "$BIN/test-moe-ids" -m "$E32" -ngl 99 -t "$THREADS" > "$OUT/moe-ids.log" 2>&1
    rec moe-ids $?
fi

if part hop; then
    timeout 900 "$BIN/test-hop-cast" --unit-only > "$OUT/hop-cast-unit.log" 2>&1
    rec hop-cast-unit $?
    timeout 900 "$BIN/test-hop-cast" -m "$M" -ngl 99 -ts "$TS" -t "$THREADS" --arm off \
        --arm "q8_0 LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1" --arm "staged exact GGML_CUDA_HOP_STAGED=2" \
        --arm "mailbox exact GGML_CUDA_HOP_MAILBOX=2" --tsv "$OUT/hop-cast.tsv" > "$OUT/hop-cast.log" 2>&1
    rec hop-cast-model $?
fi

if part pool; then
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
fi
echo "=== tools done $(date -Is)"
