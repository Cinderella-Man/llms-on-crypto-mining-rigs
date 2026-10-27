#!/usr/bin/env bash
# Times the tuning variants of the grid types' Pascal kernels (run/tune_build.sh) on one P106 (card 0) with
# test-backend-ops perf at Ornith-1.5-122B's shapes: the fused expert gate/up 3072 -> 1024 and the expert down
# 1024 -> 3072 (8 of 64 experts), a dense 3072 -> 8192 product; 1, 2 and 4 tokens; at least 8 MB of weights per call,
# beyond the card's 1.5 MB L2. llama.cpp's own kernels (GGML_CUDA_MMVQ_PASCAL_SKIP) are timed once as the reference.
# GGML_TEST_RAW_QUANT=1 (fork commit 82661fcad) fills the weights with random bytes: quantizing 400 M random floats to
# IQ2_S per case takes the rig's 2-core Celeron many minutes, and perf mode doesn't check values.
# Run from a machine that reaches the rig (RIG) and the build machine (CLIENT), with nothing else on the rig:
#   CLIENT=<alias> WORK=<build folder> BUILDS=<rig folder> MD5=<md5 of tune.tar.gz> VARIANTS='...' bash run/tune_rig.sh
# The archive goes to $BUILDS/iq-tune and is removed at the end (KEEP=1 leaves it, e.g. for a correctness check).
set -uo pipefail
RIG=${RIG:-rig}
CLIENT=${CLIENT:?ssh alias of the build machine}
WORK=${WORK:?build folder on the build machine}
BUILDS=${BUILDS:?builds folder on the rig}
MD5=${MD5:?md5 of tune.tar.gz}
VARIANTS=${VARIANTS:?variant names}
R=$BUILDS/iq-tune
echo "== install $(date +%T)"
ssh "$CLIENT" "cat $WORK/out/tune.tar.gz" | ssh "$RIG" "mkdir -p $R && cat > $R/tune.tar.gz"
ssh "$RIG" "cd $R && echo '$MD5  tune.tar.gz' | md5sum -c && tar xzf tune.tar.gz && rm tune.tar.gz && ls tune" || exit 1
run() {  # run LABEL VARIANT ENV
  echo "-- $1"
  ssh "$RIG" "cd $R/tune/$2 && export CUDA_VISIBLE_DEVICES=0 GGML_TEST_RAW_QUANT=1 $3; for op in MUL_MAT_VEC_FUSION MUL_MAT_ID MUL_MAT; do
      ./test-backend-ops perf -b CUDA0 -o \$op -p '=(iq2_s|iq3_xxs),'; done" 2>&1 | grep -E "us/run" | sed -E 's/^ +//; s/ +/ /g'
}
for v in $VARIANTS; do
  run "$v" "$v" ""
done
run "llama.cpp's kernels" "${VARIANTS%% *}" "GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs"
[ -n "${KEEP:-}" ] || ssh "$RIG" "rm -rf $R"
echo "== done $(date +%T)"
