#!/usr/bin/env bash
# C86 (Q551, Q591, Q603): GPU correctness of every feature merged into fork branch pascal-p106-v3-b12076-ladder2, each
# switch on and off, so that a failing switch is found by its run's name. One NVIDIA GPU, run natively (no container):
#   MODE=native  on a P106-090: the kernels as the rig runs them, every switch at its default (compute 6.1 turns the
#                fork's kernels on);
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX (CUDA_FORCE_PTX_JIT=1), every fork path forced
#                on with =2 and the P106's launch shapes (NSM=5): the kernels a P106 runs, on another card. Wrap the script
#                in the CUDA 12.9 container with one GPU.
# Correctness only, no timings. test-backend-ops compares the GPU with the CPU backend.
#
# Usage: BIN=<bin/ of the unpacked rig tarball, with test-llama-archs and test-llama-context from the extra tarball> \
#   OUT=<dir> [MODE=native|ptx] [MODELS=<dir with test-llama-archs -o models: qwen4exp, qwen35, mamba>] \
#   [DEV=0] [DEVS2=0,1] [RUNS="<tag> ..."] [SKIP="<tag> ..."] [KCOL_BENCH=<mmvq_kcol>] [MTOK_BENCH=<mmvq_pascal>] \
#   bash gpu_checks.sh
#   DEV     the card for one-card runs (CUDA_DEVICE_ORDER=PCI_BUS_ID numbering); DEVS2 two cards for the store cases
#           (empty: skipped; MODE=ptx defaults to empty)
#   RUNS    only these runs (default: all; MODE=native leaves out the whole-suite sweeps all-on and all-off, which take
#           hours against the rig's CPU)
# Output: OUT/<tag>.log per run (its last line "exit <status>"), OUT/console.txt (one line per run: tag, the switch it
# tests, the summary, FAIL lines, exit status, what passing means), OUT/done
set -uo pipefail
BIN=$(realpath "${BIN:?BIN=<bin/ of the unpacked tarball>}")
OUT=${OUT:?OUT=<dir>}
MODE=${MODE:-native}
DEV=${DEV:-0}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# every fork path off, and (MODE=ptx) forced on with the P106's launch shapes
OFF="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_MMVF_PASCAL=0 GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0
     GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_FATTN_TILE_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0
     GGML_CUDA_NORM_Q8=0 GGML_CUDA_GDN_FUSE=0 LLAMA_GDN_FUSE=0"
if [ "$MODE" = ptx ]; then
    ON="CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2
        GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2
        GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
    OFF="CUDA_FORCE_PTX_JIT=1 $OFF"
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
    DEVS2=${DEVS2:-}
    DEFAULT_SKIP=""
else
    ON=""
    DEVS2=${DEVS2-0,1}
    DEFAULT_SKIP="all-on all-off"
fi
SKIP=${SKIP-$DEFAULT_SKIP}

wanted() {
    case " ${SKIP} " in *" $1 "*) return 1 ;; esac
    [ -z "${RUNS:-}" ] && return 0
    case " $RUNS " in *" $1 "*) return 0 ;; esac
    return 1
}
summary() { # <log>: test-backend-ops' or test-llama-archs' or test-llama-context's own summary line
    local n
    n=$(grep -a -c 'identical to the host mask' "$1")
    [ "$n" -gt 0 ] && printf 'masks identical to the host mask: %s;' "$n"
    grep -a -E 'tests passed|^Summary|failed +: [0-9]+|correctness|persistent MoE kernel ran' "$1" | sed 's/\x1b\[[0-9;]*m//g' \
        | sed -E 's/^ +//; s/persistent MoE kernel ran ([0-9]+) layers, ([0-9]+) compared with the per-op path, ([0-9]+) differed.*/moe: \1 layers, \2 compared, \3 differed/' \
        | sort | uniq -c | sed -E 's/^ +//' | tr '\n' ';' | cut -c1-160
}
run() { # <tag> <the switch it tests> <what passing means> <devices> <env (K=V ...)> <command...>
    local tag=$1 what=$2 expect=$3 devs=$4 envs=$5
    shift 5
    wanted "$tag" || return 0
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$devs" $envs timeout 7200 "$@" > "$OUT/$tag.log" 2>&1
    local rc=$?
    echo "exit $rc" >> "$OUT/$tag.log"
    # without a CUDA device the test tools pass on the CPU alone (test-backend-ops -b CUDA0 runs no case at all)
    local warn=""
    case "$*" in
        *test-backend-ops*) grep -a -q 'tests passed' "$OUT/$tag.log" || warn="NO CASES RAN; " ;;
    esac
    case "$*" in
        *test-backend-ops*|*test-llama-archs*|*test-llama-context*) grep -a -q 'CUDA0' "$OUT/$tag.log" || warn="${warn}NO CUDA0; " ;;
    esac
    printf '%-16s %-34s %-60s %4s FAIL  exit %-3s %s\n' "$tag" "$what" "$warn$(summary "$OUT/$tag.log")" \
        "$(grep -a -c 'FAIL' "$OUT/$tag.log")" "$rc" "$expect" | tee -a "$OUT/console.txt"
}

T="$BIN/test-backend-ops -b CUDA0"
A="$BIN/test-llama-archs"
MM="MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED"
PASS="every case passes"
echo "== $(date -Is) gpu_checks MODE=$MODE DEV=$DEV BIN=$BIN" | tee "$OUT/console.txt"

# -ladder: C54's BF16 kernel (1-8 tokens with -kcol), C3's Q2_0 kernel, the HC_STREAM_MEAN test
run bf16        "-bf16mv (GGML_CUDA_MMVF_PASCAL)"  "$PASS" "$DEV" "$ON" $T -o MUL_MAT,MUL_MAT_VEC_FUSION -p 'type_a=bf16|type=bf16'
run bf16-g8r8   "MMVF_PASCAL_SHAPE=8,8"            "$PASS" "$DEV" "$ON GGML_CUDA_MMVF_PASCAL_SHAPE=8,8" $T -o MUL_MAT -p 'type_a=bf16'
run bf16-g128r8 "MMVF_PASCAL_SHAPE=128,8"          "$PASS" "$DEV" "$ON GGML_CUDA_MMVF_PASCAL_SHAPE=128,8" $T -o MUL_MAT -p 'type_a=bf16'
run bf16-off    "BF16 kernel off (stock)"          "$PASS" "$DEV" "$ON GGML_CUDA_MMVF_PASCAL=0" $T -o MUL_MAT,MUL_MAT_VEC_FUSION -p 'type_a=bf16|type=bf16'
run q2_0        "-q2_0mv alone (KCOL=0, MTOK=0)"   "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0" $T -o $MM -p 'type_a=q2_0|type=q2_0'
run q2_0-stack  "Q2_0 with -kcol and -mtok on"     "$PASS" "$DEV" "$ON" $T -o $MM -p 'type_a=q2_0|type=q2_0'
run q2_0-off    "Q2_0 to llama.cpp's kernels"      "all but Q2_0's large-activation cases (they need the Pascal kernel)" "$DEV" \
    "$ON GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q2_0" $T -o $MM -p 'type_a=q2_0|type=q2_0'
run hcsm        "HC_STREAM_MEAN"                   "$PASS" "$DEV" "$ON" $T -o HC_STREAM_MEAN

# -iqmmvq (C79): the grid types' Pascal kernels beside Q2_0's tile staging
IQ='type_a=iq2_s|type=iq2_s|type_a=iq3_xxs|type=iq3_xxs'
run iq          "-iqmmvq (IQ2_S, IQ3_XXS)"         "$PASS" "$DEV" "$ON" $T -o $MM -p "$IQ"
run iq-off      "grid types to llama.cpp"          "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs" $T -o $MM -p "$IQ"

# -kcol (C70): the k-column dense products, 1-8 tokens
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs),(type_b=f32,)?m=[0-9]+,n=[1-8],'
run kcol        "-kcol default (KCOL=1)"           "$PASS" "$DEV" "$ON" $T -o MUL_MAT -p "$KT"
run kcol-all    "KCOL=2 (every 1-8 col product)"   "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=2" $T -o MUL_MAT -p "$KT"
run kcol-r1     "KCOL=2, KCOL_R=1"                 "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=1" $T -o MUL_MAT -p "$KT"
run kcol-r2     "KCOL=2, KCOL_R=2"                 "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=2" $T -o MUL_MAT -p "$KT"
run kcol-itt1   "KCOL=2, KCOL_ITT=1 (K tiles)"     "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_ITT=1" $T -o MUL_MAT -p "$KT"
run kcol-off    "KCOL=0 (the ladder's kernels)"    "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_KCOL=0" $T -o MUL_MAT -p "$KT"

# -mtok (R7): Q2_0 expert products of 2-8 tokens, each distinct expert read once
MP='type_a=q2_0|type=q2_0'
MO=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
run mtok        "-mtok default"                    "$PASS" "$DEV" "$ON" $T -o $MO -p "$MP"
run mtok-cap1   "MTOK=1,1 (groups of one)"         "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=1,1" $T -o $MO -p "$MP"
run mtok-cap3   "MTOK=1,3"                         "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=1,3" $T -o $MO -p "$MP"
run mtok-rb1    "MTOK=1,0,1 (one row block)"       "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1" $T -o $MO -p "$MP"
run mtok-rb8    "MTOK=1,0,8"                       "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,8" $T -o $MO -p "$MP"
run mtok-gdir   "MTOK=1,0,0,1 (direct gate/up)"    "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,1" $T -o $MO -p "$MP"
run mtok-off    "MTOK=0 (Pascal mc / llama.cpp)"   "$PASS" "$DEV" "$ON GGML_CUDA_MMVQ_PASCAL_MTOK=0" $T -o $MO -p "$MP"

# -gdnfuse (C19/C20): the recurrent layers' fusions, on the ops and on a random qwen4exp against the CPU
GO=GATED_DELTA_NET_STATE_ROWS,GDN_CONV,GDN_CONV_CACHE_FUSION,GATED_RMS_NORM,GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,SSM_CONV,SSM_CONV_BIAS_SILU,RMS_NORM,L2_NORM
run gdn         "-gdnfuse CUDA fusions"            "$PASS" "$DEV" "$ON" $T -o $GO
run gdn-off     "GGML_CUDA_GDN_FUSE=0"             "$PASS" "$DEV" "$ON GGML_CUDA_GDN_FUSE=0" $T -o $GO
for rs in 0 3; do
    run gdn-archs-rs$rs     "qwen4exp fused, n_rs_seq $rs"   "the qwen4exp rows OK (nmse < 1e-4)" "$DEV" \
        "$ON LLAMA_ARCHS_N_RS_SEQ=$rs" $A -a '^qwen4exp$' -b CUDA0 -s 1
    run gdn-archs-rs$rs-off "qwen4exp unfused, n_rs_seq $rs" "the qwen4exp rows OK" "$DEV" \
        "$ON LLAMA_ARCHS_N_RS_SEQ=$rs LLAMA_GDN_FUSE=0 GGML_CUDA_GDN_FUSE=0" $A -a '^qwen4exp$' -b CUDA0 -s 1
done

# the persistent MoE kernel's self-check (Flint-shaped, one token: every stage bit for bit against the per-op path), and
# its negative control, which must report differences
run moe-check   "persistent MoE self-check"        "0 differed" "$DEV" "$ON GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1" $T -o MOE_LAYER
run moe-fault   "self-check's negative control"    "differences reported" "$DEV" \
    "$ON GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1 GGML_CUDA_PERSISTENT_MOE_FAULT=2" $T -o MOE_LAYER

# random tiny models of every architecture, the card against the CPU: defaults, every fork path off, the card-built
# attention mask for every micro-batch with its check (each step compared with the host mask, aborts on a difference)
run archs       "test-llama-archs, defaults"       "Summary: all passed" "$DEV" "$ON" $A
run archs-off   "test-llama-archs, fork paths off" "Summary: all passed" "$DEV" "$OFF" $A
run archs-mask  "LLAMA_GPU_KQ_MASK_CHECK=1"        "all passed, masks identical to the host mask" "$DEV" \
    "$ON LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1" $A -v 4

# -ckpt (C22): device stores of a sequence state on random qwen4exp, qwen35, mamba models, one card and two
if [ -n "${MODELS:-}" ]; then
    for m in qwen4exp-moe qwen35-dense mamba-dense; do
        f=$(ls "$MODELS"/${m}*.gguf 2>/dev/null | head -1)
        [ -n "$f" ] || { echo "no $m model in $MODELS" | tee -a "$OUT/console.txt"; continue; }
        run "ctx-$m"   "-ckpt stores, one card"  "store, store_p OK; failed : 0" "$DEV"   "$ON" "$BIN/test-llama-context" -m "$f" -ngl 99
        [ -n "$DEVS2" ] && run "ctx2-$m" "-ckpt stores, two cards" "store, store_p OK; failed : 0" "$DEVS2" "$ON" \
            "$BIN/test-llama-context" -m "$f" -ngl 99
    done
fi

# the whole suite, every fork path on and off (MODE=ptx by default)
run all-on      "every case, fork paths on"        "$PASS" "$DEV" "$ON" $T
run all-off     "every case, fork paths off"       "all but Q2_0's large-activation cases" "$DEV" "$OFF" $T

# the standalone benches' own correctness checks (C70's mmvq_kcol, R7's --q2_0-mtok), when given
NSM=$([ "$MODE" = ptx ] && echo "--nsm 5")
[ -n "${KCOL_BENCH:-}" ] && run kcol-bench "mmvq_kcol --check-only" "all variants match the CPU reference" "$DEV" \
    "$([ "$MODE" = ptx ] && echo CUDA_FORCE_PTX_JIT=1)" "$KCOL_BENCH" --check-only $NSM
[ -n "${MTOK_BENCH:-}" ] && run mtok-bench "mmvq_pascal --q2_0-mtok --check-only" "every variant matches" "$DEV" \
    "$([ "$MODE" = ptx ] && echo CUDA_FORCE_PTX_JIT=1)" "$MTOK_BENCH" --q2_0-mtok --check-only

date -Is > "$OUT/done"
echo "== $(date -Is) done" | tee -a "$OUT/console.txt"
