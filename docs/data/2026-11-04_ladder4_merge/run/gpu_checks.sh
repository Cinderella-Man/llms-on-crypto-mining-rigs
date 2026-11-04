#!/usr/bin/env bash
# The -ladder4 merge on one GPU: correctness of the merged rig tarball, in one container that sees one card.
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX, the fork's kernel paths forced on (=2) with a
#                P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as for -kshape and -moeskip
#   MODE=native  on a P106-090
# Parts (PARTS, default all, in this order):
#   moeskip  -moeskip's run/gpu_checks.sh, every run: the skipped slots through every expert kernel, the tail op alone
#            and fused into the router kernel, and test-llama-archs on the card (the switch off against BASE_BIN, the
#            -kshape tarball: bit for bit; LLAMA_MOE_TAIL as the op against the graph-level ops: within rounding)
#   combo    -moeskip with -coldtier: checks.sh's sharp random model (32 experts, 8 used) and its tiered copy (hot 24)
#            on the card, at -ub 128 and 1, with and without LLAMA_MOE_TAIL=$CUT: the tiered model with the op as asked
#            must equal it with the graph-level cut bit for bit (the same graph: a tiered layer takes the graph's cut);
#            the tiered model must match the untiered one, and the op the graph-level cut, to rounding (KLD <= 1e-4);
#            the cut must change the output (KLD > 1e-3)
#   moeids   test-moe-ids on the card: the tiered model with the cut, every MUL_MAT_ID's ids distinct and in range
#   dense    test-backend-ops MUL_MAT at 1 to 8 tokens with the k-column setting the rig runs (DENSE_ENV)
# Usage (inside a container of the build image with the card): BIN=<bin/ of the unpacked rig tarball, extras in it> \
#   OUT=<dir> CHECKS=<checks.sh's WORK/checks> TEXT=<text file> [BASE_BIN=<-kshape's tarball bin/>] [MODE=ptx|native] \
#   [DEV=0] [CUT=p=0.8,renorm=1] [PARTS='moeskip combo moeids dense'] [MOESKIP_RUN=<-moeskip's run/>] bash gpu_checks.sh
# Output: OUT/console.txt (one line per check), OUT/moeskip/ (its console.txt and logs), OUT/combo/, OUT/done
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(realpath "${BIN:?BIN=<bin/ of the unpacked tarball>}")
OUT=${OUT:?OUT=<dir>}
CHECKS=${CHECKS:?CHECKS=<checks.sh WORK/checks>}
TEXT=${TEXT:?TEXT=<text file>}
MODE=${MODE:-ptx}
DEV=${DEV:-0}
CUT=${CUT:-p=0.8,renorm=1}
PARTS=${PARTS:-moeskip combo moeids dense}
MOESKIP_RUN=${MOESKIP_RUN:-$HERE/../../2026-11-04_moe_skip/run}
DENSE_ENV=${DENSE_ENV:-GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0}
mkdir -p "$OUT/combo"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$MODE" = ptx ]; then
    ON="CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
        GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
else
    ON=""
fi
C=$OUT/console.txt
want() { [[ " $PARTS " == *" $1 "* ]]; }
say() { echo "$*" | tee -a "$C"; }
line() { printf '%-26s %-4s %s\n' "$1" "$2" "$3" | tee -a "$C"; }
say "== $(date -Is) gpu_checks (ladder4) MODE=$MODE DEV=$DEV BIN=$BIN"
say "   $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i "$DEV" 2>/dev/null)"

if want moeskip; then
    MODE=$MODE DEV=$DEV BIN=$BIN BASE_BIN=${BASE_BIN:-} OUT=$OUT/moeskip bash "$MOESKIP_RUN/gpu_checks.sh" > "$OUT/moeskip.out" 2>&1
    say "-- moeskip's gpu_checks.sh (exit $?): $OUT/moeskip/console.txt"
    sed -n '3,$p' "$OUT/moeskip/console.txt" | grep -v '^== ' | tee -a "$C"
fi

UM=$CHECKS/coldtier/random/qwen4exp-moe-sharp.gguf
TM=$CHECKS/coldtier/checks/random/qwen4exp-moe-sharp-h24/qwen4exp-moe-sharp.gguf
K=$OUT/combo
# <tag> <model> <LLAMA_MOE_TAIL or ""> <ubatch> [<base tag>]: llama-perplexity on the card; with a base tag, the KLD
# against that run's logits
ppl() {
    local tag=$1 m=$2 cut=$3 ub=$4 base=${5:-}
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $ON ${cut:+LLAMA_MOE_TAIL=$cut} timeout 1800 "$BIN/llama-perplexity" -m "$m" \
        -f "$TEXT" -c 128 -b 128 -ub "$ub" --chunks 4 -ngl 99 \
        $([ -n "$base" ] && echo "--kl-divergence-base $K/$base.kld --kl-divergence" || echo "--kl-divergence-base $K/$tag.kld") \
        > "$K/$tag.log" 2>&1
    echo "exit $?" >> "$K/$tag.log"
}
kld() { grep -E '^Mean    KLD' "$K/$1.log" | awk '{print $3}'; }
le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a != "" && a + 0 <= b + 0) }'; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a != "" && a + 0 > b + 0) }'; }
if want combo; then
    if [ ! -f "$UM" ] || [ ! -f "$TM" ]; then
        line combo FAIL "no random models (checks.sh's coldtier step): $UM $TM"
    else
        for ub in 128 1; do
            ppl u0-$ub  "$UM" ""            $ub
            ppl ug-$ub  "$UM" "$CUT,graph=1" $ub
            ppl tg-$ub  "$TM" "$CUT,graph=1" $ub
            ppl to-$ub  "$TM" "$CUT"         $ub
            ppl t0v-$ub "$TM" ""             $ub u0-$ub   # tiered against untiered, no cut
            ppl uov-$ub "$UM" "$CUT"         $ub ug-$ub   # the op against the graph-level cut, untiered
            ppl tgv-$ub "$TM" "$CUT,graph=1" $ub ug-$ub   # tiered against untiered, graph-level cut
            ppl ugv-$ub "$UM" "$CUT,graph=1" $ub u0-$ub   # the cut against no cut
            same=no; [ -s "$K/to-$ub.kld" ] && cmp -s "$K/to-$ub.kld" "$K/tg-$ub.kld" && same=yes
            a=$(kld t0v-$ub) b=$(kld uov-$ub) c=$(kld tgv-$ub) d=$(kld ugv-$ub)
            ok=OK
            [ $same = yes ] && le "$a" 1e-4 && le "$b" 1e-4 && le "$c" 1e-4 && gt "$d" 1e-3 || ok=FAIL
            line "combo ub $ub" $ok "tiered op = tiered graph bit for bit: $same; KLD tiered/untiered $a, op/graph $b, tiered/untiered with the cut $c, the cut $d ($CUT)"
        done
        rm -f "$K"/*.kld
    fi
fi

if want moeids; then
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $ON LLAMA_MOE_TAIL="$CUT" timeout 1800 "$BIN/test-moe-ids" -m "$TM" -ngl 99 > "$K/moeids.log" 2>&1
    rc=$?
    line "moeids tiered + cut" "$([ $rc = 0 ] && echo OK || echo FAIL)" "exit $rc: $(grep -E '^test-moe-ids:|checks,' "$K/moeids.log" | tr '\n' ' ')"
fi

if want dense; then
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $ON $DENSE_ENV timeout 7200 "$BIN/test-backend-ops" -b CUDA0 -o MUL_MAT -p 'n=[1-8],' \
        > "$OUT/dense.log" 2>&1
    rc=$?
    s=$(grep -a -E 'tests passed' "$OUT/dense.log" | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' ' ')
    nf=$(grep -a -c 'FAIL' "$OUT/dense.log")
    line "dense ($DENSE_ENV)" "$([ $rc = 0 ] && [ "$nf" = 0 ] && [ -n "$s" ] && echo OK || echo FAIL)" "exit $rc, $nf FAIL lines: ${s:-NO CASES RAN}"
fi

date -Is > "$OUT/done"
say "== $(date -Is) done"
