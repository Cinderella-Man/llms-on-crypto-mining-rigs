#!/usr/bin/env bash
# Correctness of Miner Llama's Pascal BF16 matrix-vector kernel (ggml/src/ggml-cuda/mmvf-pascal.cu) with
# test-backend-ops on one NVIDIA GPU, in the CUDA 12.9 container. A build with sm_61 code only runs on a newer GPU
# through its compute_61 PTX, JIT-compiled once into $WORK/nvcache, so this checks the code a P106-090 runs. No timings.
#
# Usage: BIN=<build dir> WORK=<dir> [GPU=n] [RUNS="tag ..."] [TAG=prefix] [DOCKER_ARGS=...] check.sh
#   BIN   an unpacked tarball of scripts/build_llamacpp_docker.sh built with TESTS=1 (holds bin/test-backend-ops)
#   RUNS  the runs to do (default all, in the order below); TAG prefixes the log names (e.g. mut- for a mutation build)
# Runs, each logged to $WORK/logs/bf16-<tag>.log with "exit <status>" as its last line:
#   off      GGML_CUDA_MMVF_PASCAL=0: the BF16 cases on llama.cpp's own kernels
#   on       GGML_CUDA_MMVF_PASCAL=2 (any GPU): every MUL_MAT case of every type, launch shapes for this GPU
#   nsm5     =2 with GGML_CUDA_MMVF_PASCAL_NSM=5: the BF16 cases at the launch shapes a 5-SM P106-090 picks
#   gG_rR    =2 with GGML_CUDA_MMVF_PASCAL_SHAPE=G,R, for each of the nine launch shapes: the BF16 cases
# The BF16 cases are the MUL_MAT and MUL_MAT_VEC_FUSION cases with BF16 weights. Then prints one line per run: its tag,
# the "tests passed" line, the number of failed cases and the exit status.
set -uo pipefail
BIN=${BIN:?BIN=<build dir>}
WORK=${WORK:?WORK=<work dir>}
BIN=$(realpath "$BIN")
mkdir -p "$WORK/logs" "$WORK/nvcache"

OPS='MUL_MAT,MUL_MAT_VEC_FUSION'
BF16='type_a=bf16|type=bf16'
ALL="off on nsm5 g8_r1 g8_r2 g8_r4 g32_r1 g32_r2 g32_r4 g128_r1 g128_r2 g128_r4"
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
    docker run --rm --name "tbo-bf16-$tag" --cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-} \
        --gpus "\"device=${GPU:-0}\"" -v "$WORK:$WORK" -v "$BIN:$BIN:ro" \
        -e CUDA_CACHE_PATH="$WORK/nvcache" -e CUDA_CACHE_MAXSIZE=4294967296 "${e[@]}" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 1800 "$BIN/bin/test-backend-ops" -b CUDA0 "$@" \
        > "$WORK/logs/bf16-$tag.log" 2>&1
    echo "exit $?" >> "$WORK/logs/bf16-$tag.log"
}

run off  "GGML_CUDA_MMVF_PASCAL=0" -o "$OPS" -p "$BF16"
run on   "GGML_CUDA_MMVF_PASCAL=2" -o MUL_MAT
run nsm5 "GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVF_PASCAL_NSM=5" -o "$OPS" -p "$BF16"
for G in 8 32 128; do
    for R in 1 2 4; do
        run "g${G}_r${R}" "GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVF_PASCAL_SHAPE=$G,$R" -o "$OPS" -p "$BF16"
    done
done

for tag in $RUNS; do
    f=$WORK/logs/bf16-${TAG:-}$tag.log
    printf '%-12s %-28s %5s failed  %s\n' "${TAG:-}$tag" "$(grep -m1 'tests passed' "$f" | sed 's/^ *//')" \
        "$(grep -c 'FAIL' "$f")" "$(tail -n 1 "$f")"
done
