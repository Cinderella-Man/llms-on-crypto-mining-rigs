#!/usr/bin/env bash
# Correctness of Miner Llama v3 rebased onto upstream llama.cpp b12076 (fork branch pascal-p106-v3-b12076) on one NVIDIA
# GPU, in the CUDA 12.9 container. The rig build holds sm_61 code only, so a newer GPU runs its compute_61 PTX,
# JIT-compiled once into $WORK/nvcache: these are the kernels a P106-090 runs. No timings.
#
# Usage: BIN=<unpacked rig tarball built with TESTS=1> ARCHS=<test-llama-archs binary> WORK=<dir> [GPU=n] [RUNS="tag ..."]
#        [DOCKER_ARGS=...] check.sh
# Runs, each logged to $WORK/logs/<tag>.log with "exit <status>" as its last line:
#   tbo-off     test-backend-ops, every case, with the fork's CUDA paths off: the Pascal matrix-vector kernels, the
#               persistent MoE kernel, both Pascal attention kernels, the two-launch attention block, the norm's Q8_1 copy
#   tbo-on      the same with all of them on (=2 forces them onto any NVIDIA GPU; the norm's Q8_1 copy is on by default)
#   moe-check   MOE_LAYER with the persistent kernel's self-check: every stage bit for bit against the per-op path
#   moe-fault   the same with a deliberate routing error (GGML_CUDA_PERSISTENT_MOE_FAULT=2), which must be reported
#   norm-q8     ADD_RMS_NORM_MUL_Q8 with the stock matrix-vector kernels (which read the Q8_1 block sums), fused and not
#   archs       test-llama-archs: random tiny models of every architecture, each device against the CPU
#   archs-on    the same with the fork's CUDA paths forced on
#   archs-mask  the same with the card-built attention mask for every micro-batch and its check (LLAMA_GPU_KQ_MASK=1
#               LLAMA_GPU_KQ_MASK_CHECK=1): each step compares the mask the card copies give with the host mask and
#               aborts on any difference
# Then prints one line per run: its tag, the summary line, the failure count and the exit status.
set -uo pipefail
BIN=${BIN:?BIN=<unpacked rig tarball>}
ARCHS=${ARCHS:?ARCHS=<test-llama-archs binary>}
WORK=${WORK:?WORK=<work dir>}
BIN=$(realpath "$BIN")
ARCHS=$(realpath "$ARCHS")
mkdir -p "$WORK/logs" "$WORK/nvcache"

OFF="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_FATTN_PASCAL=0 GGML_CUDA_FATTN_TILE_PASCAL=0 GGML_CUDA_ATTN_BLOCK=0 GGML_CUDA_NORM_Q8=0"
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_NORM_Q8=1"
ALL="tbo-off tbo-on moe-check moe-fault norm-q8 archs archs-on archs-mask"
RUNS=${RUNS:-$ALL}

wanted() {
    case " $RUNS " in *" $1 "*) return 0 ;; esac
    return 1
}

run() { # tag, extra env (space-separated K=V), binary, args...
    local tag=$1 envs=$2
    shift 2
    wanted "$tag" || return 0
    local e=()
    for kv in $envs; do e+=(-e "$kv"); done
    docker run --rm --name "pq-check-$tag" --cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-} \
        --gpus "\"device=${GPU:-0}\"" -v "$WORK:$WORK" -v "$BIN:$BIN:ro" -v "$ARCHS:$ARCHS:ro" \
        -e CUDA_CACHE_PATH="$WORK/nvcache" -e CUDA_CACHE_MAXSIZE=4294967296 "${e[@]}" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 7200 "$@" > "$WORK/logs/$tag.log" 2>&1
    echo "exit $?" >> "$WORK/logs/$tag.log"
}

T="$BIN/bin/test-backend-ops -b CUDA0"
run tbo-off   "$OFF" $T
run tbo-on    "$ON"  $T
run moe-check "$ON GGML_CUDA_PERSISTENT_MOE_CHECK=1" $T -o MOE_LAYER
run moe-fault "$ON GGML_CUDA_PERSISTENT_MOE_CHECK=1 GGML_CUDA_PERSISTENT_MOE_FAULT=2" $T -o MOE_LAYER
run norm-q8   "GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0 GGML_CUDA_NORM_Q8=1 GGML_CUDA_NORM_Q8_DEBUG=1" $T -o ADD_RMS_NORM_MUL_Q8
run archs      "" "$ARCHS"
run archs-on   "$ON" "$ARCHS"
run archs-mask "LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1" "$ARCHS" -v 4

for tag in $ALL; do
    wanted "$tag" || continue
    f=$WORK/logs/$tag.log
    case $tag in
        tbo-*|norm-q8) summary=$(grep -a -m1 'tests passed' "$f" | sed 's/^ *//') ;;
        moe-*) summary=$(grep -a -oE 'ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed' "$f" | sort | uniq -c | tr -s ' \n' ' ') ;;
        archs*) summary="$(grep -a -c 'OK' "$f") OK lines; masks identical to the host mask: $(grep -a -c 'identical to the host mask' "$f")" ;;
    esac
    printf '%-11s %-60s %5s FAIL  %s\n' "$tag" "$summary" "$(grep -a -c 'FAIL' "$f")" "$(tail -n 1 "$f")"
done
