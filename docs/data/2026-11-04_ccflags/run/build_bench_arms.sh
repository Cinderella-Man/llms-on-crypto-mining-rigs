#!/usr/bin/env bash
# Q671 / C131: the bench/mmvq_pascal binaries (mmvq_pascal from src/main.cu, mmvq_kcol from src/kcol_main.cu), once per
# compiler-flag arm of arms.tsv, for a P106 timing. Same flags as bench/mmvq_pascal/build.sh, but sm_61 machine code
# only (the P106 runs nothing else) and the arm's extra nvcc flags. CPU only: nvcc in the CUDA 12.9 container.
#
# Usage: REPO=<this repository> OUT=<dir> ARMS='base dlcm_ca ...' [JOBS=3] [LOCK=<file>] [DOCKER_ARGS=...] build_bench_arms.sh
#   ARMS  arm names of arms.tsv whose files column is "." (flags for every file); the arms with a file pattern are
#         macros of one fork file and have no counterpart in the bench
# Output: $OUT/<arm>/{mmvq_pascal,mmvq_kcol,ptxas.log,res-usage.txt}, $OUT/BUILDINFO.txt
set -uo pipefail
REPO=$(realpath "${REPO:?REPO=<this repository>}")
OUT=$(realpath -m "${OUT:?OUT=<output dir>}")
ARMS=${ARMS:?ARMS=<arm names>}
JOBS=${JOBS:-3}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(dirname "$(realpath "$0")")
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
mkdir -p "$OUT"
: > "$OUT/jobs.txt"
for arm in $ARMS; do
    flags=$(awk -F'\t' -v a="$arm" '$1==a && $3=="." {print $2}' "$HERE/arms.tsv")
    [ -n "$flags" ] || { echo "arm $arm: no global arm of that name in arms.tsv" >&2; exit 1; }
    [ "$flags" = - ] && flags=""
    for t in mmvq_pascal:main mmvq_kcol:kcol_main; do printf '%s\t%s\t%s\n' "$arm" "$t" "$flags" >> "$OUT/jobs.txt"; done
done
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
cp "$HERE/bench_inner.sh" "$OUT/bench_inner.sh"
locked docker run --rm $D -u "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="$JOBS" -v "$REPO/bench/mmvq_pascal:/b:ro" -v "$OUT:/o" \
    "$IMAGE" bash /o/bench_inner.sh
