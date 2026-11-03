#!/usr/bin/env bash
# Correctness of Miner Llama's Pascal Q2_0 matrix-vector kernel (the Q2_0 path of ggml/src/ggml-cuda/mmvq-pascal.cu)
# with test-backend-ops on one NVIDIA GPU, in the CUDA 12.9 container. A build with sm_61 code only runs on a newer GPU
# through its compute_61 PTX, JIT-compiled once into $WORK/nvcache, so this checks the code a P106-090 runs. No timings.
#
# Usage: BIN=<build dir> WORK=<dir> [GPU=n] [RUNS="tag ..."] [TAG=prefix] [DOCKER_ARGS=...] check.sh
#   BIN   an unpacked tarball of scripts/build_llamacpp_docker.sh built with TESTS=1 (holds bin/test-backend-ops)
#   RUNS  the runs to do (default all, in the order below); TAG prefixes the log names (e.g. mut- for a mutation build)
# Runs, each logged to $WORK/logs/q2-<tag>.log with "exit <status>" as its last line:
#   off      GGML_CUDA_MMVQ_PASCAL=0: the Q2_0 cases on llama.cpp's own kernels
#   on       GGML_CUDA_MMVQ_PASCAL=2 (the Pascal kernels on any GPU): the Q2_0 cases at the fork's launch shapes
#   all_mm   =2: every MUL_MAT case of every weight type (the other types' paths are unchanged)
#   all_id, all_sh, all_fu
#            =2: every MUL_MAT_ID, MUL_MAT_ID_SHARED and MUL_MAT_VEC_FUSION case of every weight type with 1 to 4
#            tokens, the batches the Pascal kernels take
#   big0     =0: the MUL_MAT_VEC_FUSION cases with more tokens, where test-backend-ops aborts on a newer GPU with "CUDA
#            error: invalid argument" at a launch of llama.cpp's own kernels with the Pascal kernels off as well
#            (registry Q556: the host picks the launch for the GPU's compute capability, the code is compiled for 6.1)
#   sa       =2 with GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1: the Q2_0 cases, 2 to 4 tokens with staged activations
#   sG_P_T   =2 with GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T: the Q2_0 cases at G rows per lane group with a fused gate, P
#            without, and tile staging T (0 off, 1 where segments can't be staged, 2 wherever it fits)
# The Q2_0 cases are the MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION and MUL_MAT_ID_SHARED cases with Q2_0 weights. Then
# prints one line per run: its tag, the "tests passed" line, the number of failed cases and the exit status.
set -uo pipefail
BIN=${BIN:?BIN=<build dir>}
WORK=${WORK:?WORK=<work dir>}
BIN=$(realpath "$BIN")
mkdir -p "$WORK/logs" "$WORK/nvcache"

OPS='MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED'
Q2='type_a=q2_0|type=q2_0'
SHAPES=""
for G in 1 2; do
    for P in 1 2; do
        for T in 0 1 2; do
            SHAPES="$SHAPES s${G}_${P}_${T}"
        done
    done
done
ALL="off on all_mm all_id all_sh all_fu big0 sa$SHAPES"
RUNS=${RUNS:-$ALL}

wanted() {
    case " $RUNS " in *" $1 "*) return 0 ;; esac
    return 1
}

run() { # tag, extra env (space-separated K=V), test-backend-ops args...
    local tag=$1 envs=$2
    shift 2
    wanted "$tag" || return 0
    tag=${TAG:-}$tag
    local e=()
    for kv in $envs; do e+=(-e "$kv"); done
    docker run --rm --name "tbo-q2-$tag" --cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-} \
        --gpus "\"device=${GPU:-0}\"" -v "$WORK:$WORK" -v "$BIN:$BIN:ro" \
        -e CUDA_CACHE_PATH="$WORK/nvcache" -e CUDA_CACHE_MAXSIZE=4294967296 "${e[@]}" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 3600 stdbuf -oL "$BIN/bin/test-backend-ops" -b CUDA0 "$@" \
        > "$WORK/logs/q2-$tag.log" 2>&1
    echo "exit $?" >> "$WORK/logs/q2-$tag.log"
}

run off "GGML_CUDA_MMVQ_PASCAL=0" -o "$OPS" -p "$Q2"
run on  "GGML_CUDA_MMVQ_PASCAL=2" -o "$OPS" -p "$Q2"
run all_mm "GGML_CUDA_MMVQ_PASCAL=2" -o MUL_MAT
run all_id "GGML_CUDA_MMVQ_PASCAL=2" -o MUL_MAT_ID -p ',n=[1-4],'
run all_sh "GGML_CUDA_MMVQ_PASCAL=2" -o MUL_MAT_ID_SHARED -p 'n=[1-4]$'
run all_fu "GGML_CUDA_MMVQ_PASCAL=2" -o MUL_MAT_VEC_FUSION -p ',m=[1-4],'
run big0   "GGML_CUDA_MMVQ_PASCAL=0" -o MUL_MAT_VEC_FUSION -p ',m=([5-9]|[1-9][0-9]+),'
run sa  "GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1" -o "$OPS" -p "$Q2"
for s in $SHAPES; do
    v=${s#s}
    run "$s" "GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_Q2_0=${v//_/,}" -o "$OPS" -p "$Q2"
done

for tag in $RUNS; do
    f=$WORK/logs/q2-${TAG:-}$tag.log
    printf '%-12s %-28s %5s failed  %s\n' "${TAG:-}$tag" "$(grep -m1 'tests passed' "$f" | sed 's/^ *//')" \
        "$(grep -c 'FAIL' "$f")" "$(tail -n 1 "$f")"
done
