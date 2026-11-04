#!/usr/bin/env bash
# Q660 (C120): GPU correctness of fork branch pascal-p106-v3-b12076-moeskip: the expert kernels with skipped slots
# (id -1), the routed experts' tail as one op (alone and fused into the router kernel), and the logits of whole models.
#   MODE=native  on a P106-090: the kernels as the rig runs them (compute 6.1 turns the fork's kernels on)
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX, the fork's kernel paths forced on (=2) with a
#                P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as for -kshape
# Correctness only, no timings: test-backend-ops compares the card with the CPU backend; test-llama-archs compares
# logits between two runs (LLAMA_ARCHS_LOGITS: the first run writes them, the next ones compare, nmse <= 1e-8).
# Runs (RUNS, default all, in this order):
#   skip-mmq    the skipped-slot cases past 8 tokens (MMQ, MMF, the host fallback: the ids copy), first
#   skip        every skipped-slot case (MUL_MAT_ID, the fused gate/up + SwiGLU), defaults: the Pascal one- and
#               multi-token kernels and -mtok for Q2_0 and the IQ types, MMQ, MMF and the host fallback for the others
#   skip-mmvq   GGML_CUDA_MMVQ_PASCAL=0: llama.cpp's mul_mat_vec_q and mul_mat_vec_q_moe for every type
#   skip-mtok0  GGML_CUDA_MMVQ_PASCAL_MTOK=0: Q2_0's 2 to 4 tokens on the earlier multi-token kernel
#   tail        MOE_TAIL (its own kernel), TOPK_MOE_TAIL (the router chain, fused), GET_ROWS (I32 rows, row -1), ADD_ID
#   experts     every expert case without skipped slots and the router chain (MUL_MAT_ID, MUL_MAT_VEC_FUSION,
#               MUL_MAT_ID_SHARED, TOPK_MOE): what the switch off runs, unchanged
#   archs-off   test-llama-archs over every architecture on the card, the switch off: BASE_BIN's build writes the
#               logits, this build's must equal them (bit for bit expected: the kernels changed only where an id is -1)
#   archs-tail  the same with LLAMA_MOE_TAIL=p=0.8,renorm=1: the graph-level ops write, the op (-1 ids, the fused router
#               kernel), the op with skip=0 (every slot's own expert) and with repeat=1 (the last kept expert) compare,
#               within rounding
# Usage: BIN=<bin/ of the unpacked rig tarball, the extras in it> OUT=<dir> [BASE_BIN=<the -kshape tarball's bin/>] \
#   [MODE=native|ptx] [DEV=<card>] [RUNS='skip tail ...'] bash gpu_checks.sh
#   DEV   the card in CUDA_DEVICE_ORDER=PCI_BUS_ID numbering (default: the first of CUDA_VISIBLE_DEVICES, else 0)
# Output: OUT/console.txt (one line per run: the run, its switch, the summary, FAIL lines, exit status, what passing
#   means), OUT/<run>.log, OUT/done
set -uo pipefail
BIN=$(realpath "${BIN:?BIN=<bin/ of the unpacked tarball>}")
OUT=${OUT:?OUT=<dir>}
MODE=${MODE:-native}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEV=${DEV:-${CUDA_VISIBLE_DEVICES%%,*}}
else
    DEV=${DEV:-0}
fi
if [ "$MODE" = ptx ]; then
    ON="CUDA_FORCE_PTX_JIT=${PTX_JIT:-0} GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
        GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
else
    ON=""
fi
C=$OUT/console.txt
say() { echo "$*" | tee -a "$C"; }
wanted() { [ -z "${RUNS:-}" ] || [[ " $RUNS " == *" $1 "* ]]; }
summary() { # test-backend-ops' own summary lines, or test-llama-archs' comparisons
    if grep -a -q 'logits against' "$1"; then
        echo "$(grep -a -c 'logits against' "$1") compared, $(grep -a 'logits against' "$1" | grep -a -c -v ', 0 of ') with a difference, max nmse $(grep -a -o -E 'nmse [0-9.e+-]+' "$1" | sort -g -k2 | tail -n 1 | cut -d' ' -f2)"
        return
    fi
    grep -a -E 'tests passed|^Summary|failed +: [0-9]+|passed +: [0-9]+' "$1" | sed 's/\x1b\[[0-9;]*m//g' | sed -E 's/^ +//' \
        | sort | uniq -c | sed -E 's/^ +//' | tr '\n' ';' | cut -c1-140
}
run() { # <run> <the switch it tests> <what passing means> <env (K=V ...)> <command...>
    local tag=$1 what=$2 expect=$3 envs=$4 log
    shift 4
    log=$OUT/$tag.log
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" $ON $envs timeout 7200 "$@" > "$log" 2>&1
    local rc=$?
    echo "exit $rc" >> "$log"
    local warn=""
    case "$*" in *test-backend-ops*)
        grep -a -q 'tests passed' "$log" || warn="NO CASES RAN; "
        grep -a -q 'CUDA0' "$log" || warn="${warn}NO CUDA0; " ;;
    esac
    printf '%-12s %-44s %-72s %4s FAIL  exit %-3s %s\n' "$tag" "$what" "$warn$(summary "$log")" \
        "$(grep -a -c 'FAIL' "$log")" "$rc" "$expect" | tee -a "$C"
}
T="$BIN/test-backend-ops -b CUDA0"
PASS="N/N tests passed, 0 FAIL lines, exit 0"
say "== $(date -Is) gpu_checks (moeskip) MODE=$MODE DEV=$DEV BIN=$BIN"
say "   $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$DEV" 2>/dev/null)"

MO=MUL_MAT_ID,MUL_MAT_VEC_FUSION
# first the skipped slots past 8 tokens (MMQ, MMF and the host fallback on a copy of the ids), where a repeated expert faults
wanted skip-mmq   && run skip-mmq   "skipped slots past 8 tokens (MMQ)"   "$PASS" ""                                $T -o $MO -p 'n=(17|33|129),.*skip=1'
wanted skip       && run skip       "defaults, skipped slots"             "$PASS" ""                                $T -o $MO -p 'skip=1'
wanted skip-mmvq  && run skip-mmvq  "MMVQ_PASCAL=0, skipped slots"        "$PASS" "GGML_CUDA_MMVQ_PASCAL=0"         $T -o $MO -p 'skip=1'
wanted skip-mtok0 && run skip-mtok0 "MTOK=0, skipped slots of Q2_0"       "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=0"    $T -o $MO -p '(type_a|type)=q2_0,.*skip=1'
wanted tail       && run tail       "the tail op, the fused router chain" "$PASS" ""                                $T -o MOE_TAIL,TOPK_MOE_TAIL,GET_ROWS,ADD_ID
wanted experts    && run experts    "the expert cases, nothing skipped"   "$PASS" ""                                $T -o MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED,TOPK_MOE -p 'skip=0|^((?!skip).)*$'

A() { # <run> <what> <expect> <env> <bin dir> <logits prefix>: test-llama-archs over every architecture on the card
    local tag=$1 what=$2 expect=$3 envs=$4 bin=$5 prefix=$6
    run "$tag" "$what" "$expect" "$envs LLAMA_ARCHS_LOGITS=$prefix" "$bin/test-llama-archs" -b CUDA0 -s 1
}
if wanted archs-off && [ -n "${BASE_BIN:-}" ]; then
    rm -f "$OUT"/off-*.f32
    A archs-off-base "the switch off, BASE_BIN writes"     "logits written"                     "" "$BASE_BIN" "$OUT/off-"
    A archs-off      "the switch off, this build compares" "0 with a difference, exit 0"        "" "$BIN"      "$OUT/off-"
    rm -f "$OUT"/off-*.f32
fi
if wanted archs-tail; then
    rm -f "$OUT"/tail-*.f32
    A archs-tail-graph "p=0.8 renorm, graph-level ops write" "logits written"           "LLAMA_MOE_TAIL=p=0.8,renorm=1,hist=0,graph=1" "$BIN" "$OUT/tail-"
    A archs-tail-op    "the same as the op compares"         "max nmse <= 1e-8, exit 0" "LLAMA_MOE_TAIL=p=0.8,renorm=1,hist=0"         "$BIN" "$OUT/tail-"
    A archs-tail-op0   "the op, skip=0, compares"            "max nmse <= 1e-8, exit 0" "LLAMA_MOE_TAIL=p=0.8,renorm=1,hist=0,skip=0"  "$BIN" "$OUT/tail-"
    A archs-tail-rep   "the op, repeat=1, compares"          "max nmse <= 1e-8, exit 0" "LLAMA_MOE_TAIL=p=0.8,renorm=1,hist=0,repeat=1" "$BIN" "$OUT/tail-"
    rm -f "$OUT"/tail-*.f32
fi

date -Is > "$OUT/done"
say "== $(date -Is) done"
