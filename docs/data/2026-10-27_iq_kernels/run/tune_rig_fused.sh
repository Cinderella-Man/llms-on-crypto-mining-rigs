#!/usr/bin/env bash
# Round 2, second pass: the fused expert gate/up (test-backend-ops' MUL_MAT_VEC_FUSION case, 8 of 64 experts of
# 1024 x 3072, IQ2_S and IQ3_XXS, 1, 2 and 4 tokens) per tuning variant. test-backend-ops perf can't time it alone: perf
# mode repeats a graph's last node, here the GLU. So each case runs once under GGML_CUDA_PROFILE, and per graph
# evaluation the launch groups that start with MUL_MAT_ID are summed (the fused gate/up and its GLU, or the two expert
# products where a kernel doesn't fuse); the median over the evaluations is the case's time. llama.cpp's kernels
# (GGML_CUDA_MMVQ_PASCAL_SKIP) are timed once as the reference, with the first variant's binary. Needs the variants
# tune_rig.sh left in BUILDS/iq-tune (KEEP=1), card 0, nothing else on the rig:
#   RIG=rig BUILDS=<rig builds folder> VARIANTS='...' bash run/tune_rig_fused.sh
# Output: one line per variant and case: variant, type, tokens, median microseconds, evaluations, the group's label.
# BUILDS/iq-tune is removed at the end (KEEP=1 leaves it).
set -uo pipefail
RIG=${RIG:-rig}
BUILDS=${BUILDS:?builds folder on the rig}
VARIANTS=${VARIANTS:?variant names}
R=$BUILDS/iq-tune
run() {  # run LABEL VARIANT ENV
  ssh "$RIG" "cd $R/tune/$2 && export CUDA_VISIBLE_DEVICES=0 GGML_TEST_RAW_QUANT=1 $3
    for t in iq2_s iq3_xxs; do for m in 1 2 4; do
      rm -f /tmp/tune-fused.csv
      GGML_CUDA_PROFILE=/tmp/tune-fused.csv ./test-backend-ops perf -b CUDA0 -o MUL_MAT_VEC_FUSION \
        -p \"type=\$t,glu_op=2,m=\$m,\" > /dev/null 2>&1
      med=\$(awk -F, '\$5 ~ /^MUL_MAT_ID/ { s[\$2] += \$6 } END { for (e in s) print s[e] }' /tmp/tune-fused.csv \
        | sort -n | awk '{ a[NR] = \$1 } END { printf \"%.2f %d\", a[int((NR + 1) / 2)], NR }')
      lab=\$(awk -F, '\$5 ~ /^MUL_MAT_ID/ { l[\$5] = 1 } END { for (k in l) printf \"%s; \", k }' /tmp/tune-fused.csv)
      echo \"$1 \$t \$m \$med \$lab\"
    done; done; rm -f /tmp/tune-fused.csv"
}
echo "== fused gate/up $(date +%T)"
for v in $VARIANTS; do
  run "$v" "$v" ""
done
run "llama.cpp" "${VARIANTS%% *}" "GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs"
[ -n "${KEEP:-}" ] || ssh "$RIG" "rm -rf $R"
echo "== done $(date +%T)"
