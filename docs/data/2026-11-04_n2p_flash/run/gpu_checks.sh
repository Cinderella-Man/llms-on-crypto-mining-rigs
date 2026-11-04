#!/usr/bin/env bash
# Q928/Q220: GPU correctness of fork branch pascal-p106-v3-b12076-n2p (N2P, N3P, N2P-128, N3P-128 on the Pascal kernels),
# one test-backend-ops run per switch setting, and regression runs of the kernels the branch touched.
#   MODE=native  on a P106-090: the kernels as the rig runs them (compute 6.1 turns the fork's kernels on), MMQ included
#   MODE=ptx     on a newer NVIDIA GPU through the build's compute_61 PTX, the fork's kernel paths forced on (=2) with a
#                P106's launch shapes (5 SMs). The rig build holds only sm_61 code and compute_61 PTX, so the driver
#                JIT-compiles that PTX for the card. MMQ (9 or more tokens) can't be checked this way: the host picks the
#                card's own tile configuration while the compute 6.1 code holds Pascal's, so these runs keep to 1-8 tokens.
# Correctness only, no timings: test-backend-ops compares the card with the CPU backend.
# Runs (RUNS, default all, in this order):
#   np           the four types' cases (every op, 1-8 tokens), defaults: the one-token kernels (warp tiles for the
#                640-wide rows), the multi-token expert kernel at 2-8 tokens, the multi-token kernel of mmvq-pascal.cu
#                for plain products
#   np-mtok-off  MTOK=0: experts of 2-8 tokens on mul_mat_vec_q_pascal_mc (2-4) and llama.cpp's kernels (5-8)
#   np-cap1      MTOK=1,1: groups of one entry
#   np-rb1       MTOK=1,0,1: one row block per block
#   np-notile    MTOK=1,0,0,0,0: no warp tiles in the multi-token kernel (N3P-128 at K = 640 then goes to the other kernels)
#   np-skip      GGML_CUDA_MMVQ_PASCAL_SKIP=<the four>: llama.cpp's matrix-vector kernel at every token count
#   np-mmq       (native only) the prompt batches, MMQ
#   reg-half     regression: the other half-word types (Q6_K, the grid types) and Q2_0 (whose tile staging the
#                128-weight types now share), every product op, 1-8 tokens
#   reg-rows     regression: GET_ROWS of every type (get_rows_cuda_kq takes the block size now)
# Usage: BIN=<bin/ of the unpacked rig tarball> OUT=<dir> [MODE=native|ptx] [DEV=<card>] [RUNS='np reg-half ...'] \
#   bash gpu_checks.sh
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
    ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304
        GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5"
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$OUT/nvcache} CUDA_CACHE_MAXSIZE=4294967296
    NT='n=[1-8],'
else
    ON=""
    NT='n=[0-9]+,'
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
say "== $(date -Is) gpu_checks (n2p) MODE=$MODE DEV=$DEV BIN=$BIN"
say "   $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$DEV" 2>/dev/null)"

# the four types: products (n tokens as the mode allows), the fused gate/up and the shared expert (their own n field)
OPS=MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED,GET_ROWS
NP="type(_a)?=(n2p|n3p|n2p_128|n3p_128),.*$NT|type=(n2p|n3p|n2p_128|n3p_128),"
[ "$MODE" = ptx ] && NP="type_a=(n2p|n3p|n2p_128|n3p_128),.*$NT|type=(n2p|n3p|n2p_128|n3p_128),"
run np          "defaults"                               "$PASS" ""                                        $T -o $OPS -p "$NP"
run np-mtok-off "MTOK=0 (the earlier kernels)"           "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=0"            $T -o $OPS -p "$NP"
run np-cap1     "MTOK=1,1 (groups of one)"               "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,1"          $T -o $OPS -p "$NP"
run np-rb1      "MTOK=1,0,1 (one row block)"             "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1"        $T -o $OPS -p "$NP"
run np-notile   "MTOK=1,0,0,0,0 (no multi-token tiles)"  "$PASS" "GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,0,0"    $T -o $OPS -p "$NP"
run np-skip     "PASCAL_SKIP=the four (llama.cpp's)"     "$PASS" "GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p,n2p_128,n3p_128" \
    $T -o $OPS -p "$NP"
if [ "$MODE" = native ]; then
    run np-mmq  "prompt batches (MMQ)"                   "$PASS" ""                                        $T -o MUL_MAT,MUL_MAT_ID -p 'type_a=(n2p|n3p|n2p_128|n3p_128),.*n=(9|33|128),'
fi

# regression of the kernels the branch touched
RH="type(_a)?=(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s),.*$NT|type=(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s),"
[ "$MODE" = ptx ] && RH="type_a=(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s),.*$NT|type=(q2_0|q6_K|iq2_xxs|iq2_xs|iq2_s|iq3_xxs|iq3_s),"
run reg-half    "Q2_0, Q6_K, grid types, defaults"       "$PASS" ""                                        $T -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED -p "$RH"
run reg-rows    "GET_ROWS, every type"                   "$PASS" ""                                        $T -o GET_ROWS

date -Is > "$OUT/done"
say "== $(date -Is) done"
