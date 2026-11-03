#!/usr/bin/env bash
# Correctness of the ops behind upstream llama.cpp #30927 on Miner Llama's v3 engine: qwen4exp (Qwen3.8-Flash-Next)
# gives a DFlash drafter the mean of its hyper-connection streams as each tapped layer's input, built from strided row
# views of the [n_embd, hc, T] residual, a chain of ADDs and a SCALE. test-backend-ops compares the GPU with the CPU,
# one container per run, in the CUDA 12.9 container. A build with sm_61 code only runs on a newer GPU through its
# compute_61 PTX, JIT-compiled once into $WORK/nvcache, so this checks the code a P106-090 runs. No timings.
#
# Usage: BIN=<build dir> WORK=<dir> [GPU=n] [RUNS="tag ..."] [DOCKER_ARGS=...] check.sh
#   BIN   an unpacked tarball of scripts/build_llamacpp_docker.sh built with TESTS=1 (holds bin/test-backend-ops)
# Runs, each logged to $WORK/logs/tbo-<tag>.log with "exit <status>" as its last line:
#   hc         HC_STREAM_MEAN: the fix's graph, run whole (so the backend's ADD fusions see the model's nodes), at
#              Flash-Next's shapes (n_embd 2560, 4 streams, 1/4/64/512 tokens) and two odd ones
#   hc-nofuse  the same with the CUDA backend's fusions off (GGML_CUDA_DISABLE_FUSION=1): one kernel per ADD
#   add        every ADD case (broadcasts, permuted and overlapping sources, fused chains of up to 16 ADDs)
#   addadd     ADD_ADD: two fused ADDs, with a non-contiguous first source among them
#   scale      every SCALE case
# Then prints one line per run: its tag, the "tests passed" line, the number of failed and unsupported cases and the
# exit status.
set -uo pipefail
BIN=${BIN:?BIN=<build dir>}
WORK=${WORK:?WORK=<work dir>}
BIN=$(realpath "$BIN")
mkdir -p "$WORK/logs" "$WORK/nvcache"

RUNS=${RUNS:-hc hc-nofuse add addadd scale}

wanted() {
    case " $RUNS " in *" $1 "*) return 0 ;; esac
    return 1
}

run() { # tag, extra env (space-separated K=V), test-backend-ops args...
    local tag=$1 envs=$2
    shift 2
    wanted "$tag" || return 0
    local e=()
    for kv in $envs; do e+=(-e "$kv"); done
    docker run --rm --name "tbo-c63-$tag" --cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-} \
        --gpus "\"device=${GPU:-0}\"" -v "$WORK:$WORK" -v "$BIN:$BIN:ro" \
        -e CUDA_CACHE_PATH="$WORK/nvcache" -e CUDA_CACHE_MAXSIZE=4294967296 "${e[@]}" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 timeout 1800 "$BIN/bin/test-backend-ops" -b CUDA0 "$@" \
        > "$WORK/logs/tbo-$tag.log" 2>&1
    echo "exit $?" >> "$WORK/logs/tbo-$tag.log"
}

run hc        ""                          -o HC_STREAM_MEAN
run hc-nofuse "GGML_CUDA_DISABLE_FUSION=1" -o HC_STREAM_MEAN
run add       ""                          -o ADD
run addadd    ""                          -o ADD_ADD
run scale     ""                          -o SCALE

for tag in $RUNS; do
    f=$WORK/logs/tbo-$tag.log
    printf '%-10s %-28s %5s failed %5s unsupported  %s\n' "$tag" "$(grep -m1 'tests passed' "$f" | sed 's/^ *//')" \
        "$(grep -c 'FAIL' "$f")" "$(grep -c 'not supported' "$f")" "$(tail -n 1 "$f")"
done
