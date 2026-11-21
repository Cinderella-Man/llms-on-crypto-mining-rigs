#!/usr/bin/env bash
# C932's P106 check (2026-11-21): pipelined decoding against plain decoding on the rig's cards, with the fix's build
# (NEW, fork branch pascal-p106-v5-pipe-logits) and its base (REF, pascal-p106-v5). Runs on the rig (ssh "$RIG" ... bash -s
# < p106_check.sh); nothing is timed. Every run is bench/pipespec/check_identity.sh (tokens, every token's logits, the
# KV state and a probe, each pipelined job against the plain one), under timeout.
#   random  on card 0 only (CUDA_VISIBLE_DEVICES=0): random Flash-Next (qwen4exp) and Flint models made by NEW's
#           test-llama-archs (seed 1, a 4,096-token vocabulary), 96 tokens, depths 1 and 3, 0 and 50 % wrong guesses,
#           with REF, REF with LLAMA_GDN_FUSE=0 (the unfused conv window, the control) and NEW. Expected: Flint the same
#           everywhere; Flash-Next DIFF with REF if the cards show the fault, the same with the control and with NEW
#   flint   Flint's release file on all cards with the v5 preset's split and batch sizes, 4,096-token context, its chat
#           template, greedy, 256 tokens, depth 4 (the preset's), 0 and 50 % wrong guesses and the four guessers, with
#           NEW and REF: every job the same as plain on both, and NEW's plain hashes REF's
#   fn      (when FN is set) Swift Flash-Next's real file as Q550 ran it, 128 tokens, depths 1 and 3, with REF and NEW
# Usage: NEW=<NEW's bin/> REF=<REF's bin/> PS=<bench/pipespec on the rig> OUT=<rig dir> FLINT=<Flint's file>
#   [FN=<Swift Flash-Next shard 1>] [STEPS='random flint fn'] bash p106_check.sh
# Output: OUT/<step>-<model>-<build>/ (check_identity's folder) and .log (its table), OUT/summary.tsv (step, model,
#   build, exit code, pipelined jobs, jobs differing, jobs with other tokens, plain's logits hash); OUT/done at the end.
set -uo pipefail
: "${NEW:?NEW=<bin dir>}" "${REF:?REF=<bin dir>}" "${PS:?PS=<bench/pipespec>}" "${OUT:?OUT=<rig dir>}"
STEPS=${STEPS:-random flint fn}
FLINT_ARGS=${FLINT_ARGS:--ngl 99 -fa on -c 4096 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6}
FN_ARGS=${FN_ARGS:--ngl 99 -fa on -c 4096 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
mkdir -p "$OUT/models"
rm -f "$OUT/done"
[ -f "$OUT/summary.tsv" ] || printf 'step\tmodel\tbuild\trc\tpipelined jobs\tjobs differing\tjobs with other tokens\tplain logits hash\n' > "$OUT/summary.tsv"
step() { [[ " $STEPS " == *" $1 "* ]]; }
ci() { # <step> <model tag> <build tag> <bin> <model> <env ...> -- check_identity.sh's variables
    local s=$1 tag=$2 b=$3 bin=$4 m=$5; shift 5
    local o=$OUT/$s-$tag-$b
    env "$@" BIN="$bin" MODEL="$m" SPEED=0 TIMEOUT=1800 OUT="$o" bash "$PS/check_identity.sh" > "$o.log" 2>&1
    local rc=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$tag" "$b" "$rc" \
        "$(grep -vc '^plain ' <(grep -E ' (same|DIFF)$' "$o.log"))" "$(grep -c ' DIFF$' "$o.log")" \
        "$(awk '$1 != "plain" && $1 != "run" && $8 == "DIFF"' "$o.log" | wc -l)" \
        "$(grep -o 'name=plain .*' "$o/results.txt" 2> /dev/null | grep -o ' logits=[^ ]*' | cut -d= -f2)" | tee -a "$OUT/summary.tsv"
}
echo "=== start $(date -Is): NEW=$NEW REF=$REF steps: $STEPS"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

if step random; then
    for a in qwen4exp flint; do
        mkdir -p "$OUT/models/$a"
        (cd "$OUT/models/$a" && LLAMA_TEST_ARCHS_N_VOCAB=4096 CUDA_VISIBLE_DEVICES=0 timeout 600 "$NEW/test-llama-archs" -a "^$a\$" -s 1 -o . > gen.log 2>&1)
        m=$(ls "$OUT/models/$a/$a"*.gguf | head -n 1)
        md5sum "$m"
        for b in ref ref-fuse0 new; do
            bin=$NEW; [ "$b" = new ] || bin=$REF
            extra=""; [ "$b" = ref-fuse0 ] && extra="LLAMA_GDN_FUSE=0"
            ci random "$a" "$b" "$bin" "$m" CUDA_VISIBLE_DEVICES=0 $extra CHAT=0 PROMPT="$PS/prompts/rewrite.txt" N=96 \
                DEPTHS="1 3" MISSES="0 0.5" ARGS="-ngl 99 -fa on -c 2048"
        done
    done
fi

if step flint; then
    for b in new ref; do
        bin=$NEW; [ "$b" = ref ] && bin=$REF
        ci flint flint-q8 "$b" "$bin" "$FLINT" CHAT=1 PROMPT="$PS/prompts/code.txt" N=256 DEPTHS=4 MISSES="0 0.5" \
            ARGS="$FLINT_ARGS"
    done
fi

if step fn && [ -n "${FN:-}" ]; then
    for b in ref new; do
        bin=$NEW; [ "$b" = ref ] && bin=$REF
        ci fn swift-q2 "$b" "$bin" "$FN" CHAT=1 PROMPT="$PS/prompts/rewrite.txt" N=128 DEPTHS="1 3" MISSES="0 0.5" \
            ARGS="$FN_ARGS"
    done
fi
date -Is > "$OUT/done"
echo "=== done $(date -Is)"
cat "$OUT/summary.tsv"
