#!/usr/bin/env bash
# C267-C269 (Q823-Q825): GPU correctness of fork branch pascal-p106-v3-b12076-kshape, one test-backend-ops run per
# switch setting of the changed kernels, and the bench's CPU-reference check of every launch-shape variant.
#   MODE=native  on a P106-090: the kernels as the rig runs them (compute 6.1 turns the fork's kernels on)
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX, the fork's kernel paths forced on (=2) with a
#                P106's launch shapes (5 SMs, 96 KB of shared memory per SM). The rig build holds only sm_61 code and
#                compute_61 PTX, so the driver JIT-compiles that PTX for the card. The registers the JIT gives a kernel
#                differ from the P106's, so where the launch shape depends on them (Q823) a run here may pick another
#                tile than a P106 would; every shape gives the same results (the design note in
#                mmvq-pascal-kcol-impl.cuh), and the forced-shape runs cover the alternatives.
# Correctness only, no timings: test-backend-ops compares the card with the CPU backend.
# Runs (RUNS, default all, in this order):
#   kcol         the k-column products of 1-8 tokens (every k-column type), defaults: Q823's shapes, Q824's table
#   kcol-table0  GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0: the k-column kernels wherever they take the product
#   kcol-all     KCOL=2: every product of 1-8 tokens of these types on the k-column kernels (no table)
#   kcol-old     KCOL=2, KCOL_SMEM=49152: the launch shapes before Q823
#   kcol-s32k    KCOL=2, KCOL_SMEM=32768
#   kcol-r1      KCOL=2, KCOL_R=1
#   kcol-r2      KCOL=2, KCOL_R=2
#   kcol-itt1    KCOL=2, KCOL_ITT=1
#   kcol-off     KCOL=0: the other kernels
#   stock        every large-activation case (activation blocks that sum past FP16's range), defaults: Q916's FP32 sums
#   stock-skip   the same with KCOL=0 and Q4_0, Q4_1, Q5_0 on llama.cpp's mul_mat_vec_q at every token count
#   mtok         the Q2_0 experts of 1-8 tokens (MUL_MAT_ID, the fused gate/up, the routed + shared launch), defaults:
#                since Q825 the down projection's 180-byte rows on the multi-token kernel's warp tiles
#   mtok-direct  MTOK=1,0,0,0,0: every product without a gate that would take tiles loads its weights directly
#   mtok-gdir    MTOK=1,0,0,1: the fused gate/up's direct loads with the down projection's tiles
#   mtok-cap1    MTOK=1,1: groups of one entry
#   mtok-rb1     MTOK=1,0,1: one row block per block
#   mtok-off     MTOK=0: the earlier kernels
#   mtok-iq      the grid types' and IQ1_M's experts on the multi-token kernels (the switch's parser changed)
#   bench        mmvq_kcol --check-only (every variant, every type: pascal = this build's shape, old = before Q823)
# Usage: BIN=<bin/ of the unpacked rig tarball> OUT=<dir> [MODE=native|ptx] [DEV=<card>] [RUNS='kcol mtok ...'] \
#   [KCOL_BENCH=<bench/mmvq_pascal/build/mmvq_kcol>] bash gpu_checks.sh
#   DEV   the card in CUDA_DEVICE_ORDER=PCI_BUS_ID numbering (default: the first of CUDA_VISIBLE_DEVICES, else 0)
# Output: OUT/console.txt (one line per run: the run, its switch, test-backend-ops' summary, FAIL lines, exit status,
#   what passing means), OUT/<run>.log, OUT/done
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
    BA="--check-only --nsm 5 --smem-sm 98304"
else
    ON=""
    BA="--check-only"
fi
C=$OUT/console.txt
say() { echo "$*" | tee -a "$C"; }
wanted() { [ -z "${RUNS:-}" ] || [[ " $RUNS " == *" $1 "* ]]; }
summary() { # test-backend-ops' own summary lines
    grep -a -E 'tests passed|^Summary|failed +: [0-9]+|passed +: [0-9]+' "$1" | sed 's/\x1b\[[0-9;]*m//g' | sed -E 's/^ +//' \
        | sort | uniq -c | sed -E 's/^ +//' | tr '\n' ';' | cut -c1-140
}
run() { # <run> <the switch it tests> <what passing means> <env (K=V ...)> <command...>
    local tag=$1 what=$2 expect=$3 envs=$4 log
    shift 4
    wanted "$tag" || return 0
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
    printf '%-12s %-44s %-64s %4s FAIL  exit %-3s %s\n' "$tag" "$what" "$warn$(summary "$log")" \
        "$(grep -a -c 'FAIL' "$log")" "$rc" "$expect" | tee -a "$C"
}
T="$BIN/test-backend-ops -b CUDA0"
PASS="N/N tests passed, 0 FAIL lines, exit 0"
say "== $(date -Is) gpu_checks (kshape) MODE=$MODE DEV=$DEV BIN=$BIN"
say "   $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$DEV" 2>/dev/null)"

# the k-column products (C267/Q823, C268/Q824)
KT='type_a=(q8_0|q2_0|q3_K|q4_K|q5_K|q6_K|iq4_xs|q4_0|q5_0|iq4_nl|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s|iq1_m),(type_b=f32,)?m=[0-9]+,n=[1-8],'
K2="GGML_CUDA_MMVQ_PASCAL_KCOL=2"
run kcol        "defaults (Q823 shapes, Q824 table)"     "$PASS" ""                                           $T -o MUL_MAT -p "$KT"
run kcol-table0 "KCOL_TABLE=0"                           "$PASS" "GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"         $T -o MUL_MAT -p "$KT"
run kcol-all    "KCOL=2 (every 1-8 column product)"      "$PASS" "$K2"                                        $T -o MUL_MAT -p "$KT"
run kcol-old    "KCOL=2, KCOL_SMEM=49152 (before Q823)"  "$PASS" "$K2 GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=49152"  $T -o MUL_MAT -p "$KT"
run kcol-s32k   "KCOL=2, KCOL_SMEM=32768"                "$PASS" "$K2 GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768"  $T -o MUL_MAT -p "$KT"
run kcol-r1     "KCOL=2, KCOL_R=1"                       "$PASS" "$K2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=1"         $T -o MUL_MAT -p "$KT"
run kcol-r2     "KCOL=2, KCOL_R=2"                       "$PASS" "$K2 GGML_CUDA_MMVQ_PASCAL_KCOL_R=2"         $T -o MUL_MAT -p "$KT"
run kcol-itt1   "KCOL=2, KCOL_ITT=1"                     "$PASS" "$K2 GGML_CUDA_MMVQ_PASCAL_KCOL_ITT=1"       $T -o MUL_MAT -p "$KT"
run kcol-off    "KCOL=0 (the other kernels)"             "$PASS" "GGML_CUDA_MMVQ_PASCAL_KCOL=0"               $T -o MUL_MAT -p "$KT"

# llama.cpp's mul_mat_vec_q with FP32 activation sums (Q916): every large-activation case (activation blocks summing past
# FP16's range), at the defaults and with Q4_0, Q4_1 and Q5_0 on llama.cpp's kernel at every token count
run stock       "large activations, defaults"            "$PASS" ""                                           $T -o MUL_MAT -p 'big='
run stock-skip  "large act., KCOL=0, PASCAL_SKIP=q4_0,q4_1,q5_0" "$PASS" "GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_SKIP=q4_0,q4_1,q5_0" \
    $T -o MUL_MAT -p 'big='

# the Q2_0 experts (C269/Q825)
MO=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
MP='type_a=q2_0|type=q2_0'
run mtok        "defaults (down projection on tiles)"    "$PASS" ""                                        $T -o $MO -p "$MP"
run mtok-direct "MTOK=1,0,0,0,0 (direct loads, no gate)" "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,0,0"    $T -o $MO -p "$MP"
run mtok-gdir   "MTOK=1,0,0,1 (direct gate/up)"          "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,1"      $T -o $MO -p "$MP"
run mtok-cap1   "MTOK=1,1 (groups of one)"               "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,1"          $T -o $MO -p "$MP"
run mtok-rb1    "MTOK=1,0,1 (one row block)"             "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1"        $T -o $MO -p "$MP"
run mtok-off    "MTOK=0 (the earlier kernels)"           "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=0"            $T -o $MO -p "$MP"
run mtok-iq     "defaults, the grid types and IQ1_M"     "$PASS" ""                                        $T -o $MO -p 'type_a=iq(2_xxs|2_xs|2_s|3_xxs|3_s|1_m)|type=iq(2_xxs|2_xs|2_s|3_xxs|3_s|1_m)'

# the bench: every launch-shape variant against its CPU reference
if [ -n "${KCOL_BENCH:-}" ] && wanted bench; then
    # shellcheck disable=SC2086
    run bench "mmvq_kcol $BA, every variant and type" "correctness: all variants match the CPU reference" "" \
        "$KCOL_BENCH" $BA --tokens 1,2,3,4,5,6,7,8 --csv "$OUT/bench.csv"
    say "bench detail: $(grep -a -c ' PASS' "$OUT/bench.log") PASS lines, $(grep -a -c 'FAIL' "$OUT/bench.log") FAIL, $(grep -a 'correctness:' "$OUT/bench.log")"
else
    wanted bench && say "bench        SKIPPED: KCOL_BENCH not given"
fi

date -Is > "$OUT/done"
say "== $(date -Is) done"
