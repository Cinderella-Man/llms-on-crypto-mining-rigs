#!/usr/bin/env bash
# The CPU decoding check (bench/mmvq_pascal/verify/kcol_unit_check.sh) on a copy of the bench with three planted errors,
# one per type that Q630 adds to the k-column kernels: Q4_0's offset -7 instead of -8, Q5_0's fifth bits of the first
# block read without the 2-byte shift, IQ4_NL's code words 0/1 and 2/3 swapped. It must fail exactly q4_0, q5_0 and
# iq4_nl and pass the seven other types.
# Env: REPO (a checkout of this repository), WORK (a scratch folder for the copy), TRIALS (default 8), DOCKER_ARGS.
# Usage: REPO=<checkout> WORK=<dir> docs/data/2026-11-04_kcol_coverage/run/mutation_check.sh
set -euo pipefail
REPO=${REPO:?REPO=<this repository>}
WORK=${WORK:?WORK=<scratch folder>}
rm -rf "$WORK/mut" && mkdir -p "$WORK/mut"
cp -r "$REPO/bench/mmvq_pascal/src" "$REPO/bench/mmvq_pascal/verify" "$WORK/mut/"
H="$WORK/mut/src/kcol_mm.cuh"
perl -0pi -e 's/T == KQ4_0 \? -8 :/T == KQ4_0 ? -7 :/ or die "q4_0 edit"' "$H"
perl -0pi -e 's/qh   = prmt\(w\[0\], w\[1\], 0x5432\);/qh   = w[1];/ or die "q5_0 edit"' "$H"
perl -0pi -e 's/iq4nl_words\(cw\[k\], q\[i\] \+ 2 \* k\);/iq4nl_words(cw[k ^ 1], q[i] + 2 * k);/ or die "iq4_nl edit"' "$H"
diff "$REPO/bench/mmvq_pascal/src/kcol_mm.cuh" "$H" || true
cd "$WORK/mut"
mkdir -p build
docker run --rm ${DOCKER_ARGS:-} -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w "${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}" \
    bash -euo pipefail -c 'nvcc -std=c++17 -O2 -arch=sm_61 -Wno-deprecated-gpu-targets -o build/kcol_unit_check verify/kcol_unit_check.cu
                           build/kcol_unit_check "$1" || true' _ "${TRIALS:-8}"
