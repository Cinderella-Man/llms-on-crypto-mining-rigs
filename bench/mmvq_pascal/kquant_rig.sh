#!/usr/bin/env bash
# The Pascal Q4_K and Q6_K matrix-vector kernels on the rig (Q149, Q101): the fork branch pascal-p106-v2-kquant.
# Runs on the rig; results and logs go to $OUT.
#
#   gates  bench/mmvq_pascal on one card: q4_K and q6_K, with q5_K and q8_0 as references, all shapes and variants.
#          Gates: q6_K >= 130 GB/s on the dense K = 2048 shapes (Q101); q4_K >= 120 GB/s on the dense K = 2048 shapes
#          and the expert shapes (Q5_K, the same design with a fifth bit, measured 120-128).
#   tbo    test-backend-ops on one card, native sm_61 code: every Q4_K and Q6_K product of 1-4 tokens (products,
#          expert products, fused gate/up, large activations, routed + shared expert).
#   qwen   Qwen3.6-35B-A3B Q4_K_M, tg128 on $QWEN_GPUS: the build as it is, without the Q4_K kernel, without the Q6_K
#          kernel, without both (GGML_CUDA_MMVQ_PASCAL_SKIP), and with every Pascal kernel off (GGML_CUDA_MMVQ_PASCAL=0);
#          then the first and the fourth again, to show drift between the first and the last runs.
#   flint  Flint UD-Q4_K_XL, tg128 on $FLINT_GPUS: the build as it is against GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K,
#          alternating, two rounds. UD-Q4_K_XL reads only ~32 MB of Q4_K and Q6_K per token, so expect at most ~1 %.
#   text   greedy text, 64 tokens, Qwen3.6 with the kernels as built against GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K
#          (not in the default steps; the same check ran on another GPU through the compute_61 PTX).
#
# Usage:
#   BIN=/opt/llama-builds/<build>/bin MMVQ=<mmvq_pascal binary> QWEN=<Qwen3.6 Q4_K_M gguf> FLINT=<UD-Q4_K_XL gguf> \
#       bash kquant_rig.sh [steps]          (default steps: gates tbo qwen flint)
# Takes ~20 minutes, plus model loads from slow storage. Nothing else may run on the cards meanwhile.
set -u
STEPS=${*:-gates tbo qwen flint}
BIN=${BIN:?set BIN to the build bin/ directory}
OUT=${OUT:-/tmp/q149}
QWEN_GPUS=${QWEN_GPUS:-0,1,2,3,4}
FLINT_GPUS=${FLINT_GPUS:-0,1,2,3,4,5,6,7,8}
TG="-ngl 99 -fa on -p 0 -n 128 -r 3 -o md"
mkdir -p "$OUT"
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
ncards() { echo "$1" | tr ',' '\n' | wc -l; }

# one llama-bench run: label, GPUs, model, env assignments...
tg() {
    local label=$1 gpus=$2 model=$3
    shift 3
    echo "== $label, $(ncards "$gpus") cards $(date +%T) ${*:-(defaults)}"
    env CUDA_VISIBLE_DEVICES="$gpus" "$@" "$BIN/llama-bench" -m "$model" $TG > "$OUT/tg-$label.md" 2> "$OUT/tg-$label.log"
    grep -E "tg128" "$OUT/tg-$label.md"
    grep -iE "error|fail" "$OUT/tg-$label.log" | head -3
}

nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 1000 > "$OUT/smi.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT

if has gates; then
    MMVQ=${MMVQ:?set MMVQ to the mmvq_pascal binary}
    echo "== gates: mmvq_pascal on GPU 0 $(date +%T)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$MMVQ" --types q4_K,q6_K,q5_K,q8_0 --csv "$OUT/gates.csv" \
        > "$OUT/gates.txt" 2>&1
    echo "exit $?"
    grep -E "correctness|graph node|streaming" "$OUT/gates.txt"
    # fastest correct variant per type and shape (GB/s is column 15 of the CSV)
    awk -F, 'NR > 1 && $3 != "TOKEN_TOTAL" && $23 == "PASS" && ($9 == "q4_K" || $9 == "q6_K") {
                 k = $9 " " $3; if ($15 > best[k]) { best[k] = $15; v[k] = $10 }
                 if ($10 == "llama") ref[k] = $15 }
             END { for (k in best) printf "  %-20s %6.1f GB/s  %-8s (llama.cpp %5.1f)\n", k, best[k], v[k], ref[k] }' \
        "$OUT/gates.csv" | sort
    grep -E "^(q4_K|q6_K) " "$OUT/gates.txt" | head -20
fi

if has tbo; then
    echo "== tbo: test-backend-ops on GPU 0, Q4_K and Q6_K, 1-4 tokens $(date +%T)"
    for op in "MUL_MAT,MUL_MAT_ADD|^type_a=(q4_K|q6_K),.*,n=[1-4]," \
              "MUL_MAT_ID|^type_a=(q4_K|q6_K),.*,n=[1-4]," \
              "MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION|^type=(q4_K|q6_K),.*,m=[1-4]," \
              "MUL_MAT_ID_SHARED|^type=(q4_K|q6_K),n=[1-4]\$"; do
        IFS='|' read -r ops filter <<< "$op"
        CUDA_VISIBLE_DEVICES=0 "$BIN/test-backend-ops" -b CUDA0 -o "$ops" -p "$filter" > "$OUT/tbo-${ops%%,*}.log" 2>&1
        rc=$?
        echo "  $ops: $(grep -aE "tests passed" "$OUT/tbo-${ops%%,*}.log" | tail -1 | xargs), exit $rc"
    done
fi

if has qwen; then
    QWEN=${QWEN:?set QWEN to the Qwen3.6-35B-A3B Q4_K_M file}
    tg qwen-on      "$QWEN_GPUS" "$QWEN"
    tg qwen-noq4k   "$QWEN_GPUS" "$QWEN" GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K
    tg qwen-noq6k   "$QWEN_GPUS" "$QWEN" GGML_CUDA_MMVQ_PASCAL_SKIP=q6_K
    tg qwen-noboth  "$QWEN_GPUS" "$QWEN" GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K
    tg qwen-alloff  "$QWEN_GPUS" "$QWEN" GGML_CUDA_MMVQ_PASCAL=0
    tg qwen-on2     "$QWEN_GPUS" "$QWEN"
    tg qwen-noboth2 "$QWEN_GPUS" "$QWEN" GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K
fi

if has flint; then
    FLINT=${FLINT:?set FLINT to the Flint UD-Q4_K_XL file}
    for round in 1 2; do
        tg flint-on$round     "$FLINT_GPUS" "$FLINT" LLAMA_NO_MOE_DOWN_RESCALE=tg
        tg flint-noboth$round "$FLINT_GPUS" "$FLINT" LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K
    done
fi

if has text; then
    QWEN=${QWEN:?set QWEN to the Qwen3.6-35B-A3B Q4_K_M file}
    prompt="The three most important causes of the French Revolution were"
    for arm in on noboth; do
        skip=; [ $arm = noboth ] && skip=q4_K,q6_K
        CUDA_VISIBLE_DEVICES="$QWEN_GPUS" GGML_CUDA_MMVQ_PASCAL_SKIP=$skip "$BIN/llama-completion" -m "$QWEN" -ngl 99 \
            -fa on -p "$prompt" -n 64 --temp 0 --seed 1 -no-cnv --ignore-eos --no-display-prompt \
            > "$OUT/text-$arm.txt" 2> "$OUT/text-$arm.log"
    done
    cmp -s "$OUT/text-on.txt" "$OUT/text-noboth.txt" && echo "== text: identical" || echo "== text: differs (see $OUT)"
fi

echo "== done $(date +%T); Xid lines: $(journalctl -k --since '-2h' --no-pager 2>/dev/null | grep -ci xid)"
