#!/usr/bin/env bash
# Old against new on one card for a fast multi-token kernel of C1138 (IQ2_XXS, IQ2_S): bench/mmvq_pascal/verify's
# mtok_iq_fast_check (built with verify/build_mtok_iq_fast_check.sh against the branch's build tree) run with the fork's
# Pascal kernels forced (scripts/pascal_forced.sh: a newer card otherwise takes llama.cpp's own kernels) and
# GGML_CUDA_MTOK_IQ_FAST at 0 (the old kernel), at each value in ARMS, and at FAULT_ARM with the test fault
# GGML_CUDA_MTOK_IQ_FAST_FAULT=1. Each arm's outputs are compared byte for byte (cmp) and case by case (the hash lines)
# with arm 0's. A pass: every arm in ARMS byte-identical; the fault arm different in every fused and shared case of 2
# to 8 tokens and nowhere else.
# TYPE: the weights' type, the harness's second argument (default iq2_s; iq2_xxs reruns 28r1's cases).
#
# Env: WORK (output folder), CHECK (the mtok_iq_fast_check binary), REPO (a folder with pascal_forced.sh), GPU (the
# card, docker's device index), ARMS (default "1"), FAULT_ARM (default 1; empty: none), MOUNTS (folders to mount, at
# the same paths), JITCACHE (a PTX JIT cache folder), DOCKER_ARGS, IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04),
# PRELOAD (run/launch_count.c built as a .so: every arm then also lists the driver-API launches by kernel, which shows
# whether the hand-written PTX kernels ran)
set -uo pipefail
: "${WORK:?}" "${CHECK:?}" "${REPO:?}" "${GPU:?}"
ARMS=${ARMS:-1}
TYPE=${TYPE:-iq2_s}
FAULT_ARM=${FAULT_ARM-1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
# shellcheck disable=SC1091
. "$REPO/pascal_forced.sh"
mkdir -p "$WORK"
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m"); done
run_arm() {  # <tag> <env ...>
    local tag=$1; shift
    docker run --rm --gpus "device=$GPU" ${DOCKER_ARGS:-} "${VOL[@]}" ${JITCACHE:+-e CUDA_CACHE_PATH=$JITCACHE} \
        ${PRELOAD:+-e LD_PRELOAD=$PRELOAD} -e CUDA_CACHE_MAXSIZE=4294967296 "$IMAGE" bash -c \
        "env $PASCAL_FORCED $* '$CHECK' '$WORK/$tag.bin' '$TYPE' > '$WORK/$tag.txt' 2> '$WORK/$tag.err'; echo rc=\$?" | tail -n 1
}
echo "arm 0: $(run_arm off GGML_CUDA_MTOK_IQ_FAST=0) $(tail -n 1 "$WORK/off.txt")"
rc=0
for a in $ARMS; do
    echo "arm $a: $(run_arm "on$a" GGML_CUDA_MTOK_IQ_FAST="$a") $(tail -n 1 "$WORK/on$a.txt")"
    [ -n "${PRELOAD:-}" ] && grep -a '^launch_count:' "$WORK/on$a.err" | sed 's/^/  /'
    if cmp -s "$WORK/off.bin" "$WORK/on$a.bin"; then
        echo "  arm $a: outputs byte-identical to arm 0 ($(stat -c %s "$WORK/on$a.bin") bytes); case lines differing: $(diff "$WORK/off.txt" "$WORK/on$a.txt" | grep -c '^>')"
    else
        echo "  arm $a: OUTPUTS DIFFER from arm 0: $(diff "$WORK/off.txt" "$WORK/on$a.txt" | grep -c '^>') case lines"; rc=1
        diff "$WORK/off.txt" "$WORK/on$a.txt" | grep '^>' | head -n 20
    fi
done
if [ -n "$FAULT_ARM" ]; then
    echo "fault arm ($FAULT_ARM): $(run_arm fault GGML_CUDA_MTOK_IQ_FAST="$FAULT_ARM" GGML_CUDA_MTOK_IQ_FAST_FAULT=1)"
    # cases that must differ: fused and shared at 2 to 8 tokens
    paste -d'|' "$WORK/off.txt" "$WORK/fault.txt" | awk -F'|' '/^case/ {
        split($1, a, " "); n = a[4]; sub("n=", "", n); sub(":", "", n);
        must = (a[2] != "plain" && n + 0 >= 2); same = ($1 == $2);
        if (must && same) bad_same++; else if (!must && !same) bad_diff++; else if (must) ok++;
        tot++ } END { printf "  fault arm: %d of %d cases that must differ do; must differ but equal: %d; differ but must not: %d\n",
        ok, ok + bad_same, bad_same + 0, bad_diff + 0 }'
fi
exit $rc
