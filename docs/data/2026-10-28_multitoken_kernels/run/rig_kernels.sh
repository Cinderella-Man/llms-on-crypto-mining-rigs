#!/usr/bin/env bash
# The 2-4-token Pascal kernels (fork branch pascal-p106-v3-mtpk) on the rig, with nothing else on the rig; the machine
# that runs this script only relays over ssh.
#   1. install: the CUDA build's tarball from the quality machine to BUILDS on the rig, md5 check, unpack
#   2. correctness on card 0: test-backend-ops for the 2-4-token cases small enough for the rig's CPU to quantize
#      (MUL_MAT, MUL_MAT_ID, the fused gate/up), new kernel and the earlier one
#   3. kernel time on card 0: test-backend-ops perf at Ornith-1.5-122B's shapes (8 of 64 experts: expert gate
#      3072 -> 1024 with a broadcast input, down 1024 -> 3072; dense 3072 -> 1024, 1024 -> 3072, 3072 -> 8192; for Q8_0
#      and Q5_0 also 8192 -> 3072 and the 248,320-row output layer), 1 to 4 tokens, for n2p, n3p, iq2_s, iq3_xxs,
#      q8_0, q5_0, with three kernels: mul_mat_vec_q_pascal_mt (MT), the earlier mul_mat_vec_q_pascal_mc
#      (GGML_CUDA_MMVQ_PASCAL_MT=0) and llama.cpp's (GGML_CUDA_MMVQ_PASCAL_MC=0: the Pascal kernels for one token
#      only); the fused gate/up under GGML_CUDA_PROFILE (fused() below); then MT's row blocks per block and activation
#      budget on the expert down and gate products (GGML_CUDA_MMVQ_PASCAL_MT_RB, _MT_ACT_KB). Weights are random
#      bytes (GGML_TEST_RAW_QUANT=1): perf mode doesn't check values, and quantizing them on the rig's 2-core Celeron
#      takes many minutes; every expert call reads 8 random experts of 64 (25-50 M weights, beyond the 1.5 MB L2),
#      routed afresh for every repetition.
#   RIG=rig QM=<ssh alias> WORK=<work dir on QM> NAME=<build name> MD5=<tarball md5> [BUILDS=/opt/llama-builds] \
#     [STEPS="1 2 3"] bash rig_kernels.sh > OUT/rig_kernels.txt
set -uo pipefail
RIG=${RIG:-rig}
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?work dir on QM}
NAME=${NAME:?build name}
MD5=${MD5:?md5 of NAME.tar.gz}
BUILDS=${BUILDS:-/opt/llama-builds}
STEPS=${STEPS:-1 2 3}
TYPES=${TYPES:-n2p n3p iq2_s iq3_xxs q8_0 q5_0}
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has 1; then
  echo "== 1. install $NAME $(date +%T)"
  ssh "$QM" "cat $WORK/out/$NAME.tar.gz" | ssh "$RIG" "cat > $BUILDS/$NAME.tar.gz"
  ssh "$RIG" "cd $BUILDS && echo '$MD5  $NAME.tar.gz' | md5sum -c && rm -rf $NAME && tar xzf $NAME.tar.gz && rm $NAME.tar.gz && head -3 $NAME/BUILDINFO.txt" || exit 1
fi

if has 2; then
  echo "== 2. correctness, card 0 $(date +%T)"
  small='^(?!.*(k=3072|k=4096|k=2048|k=8192|k=14336|n_mats=32|n_mats=64|n_mats=128|m=3072|m=4097|m=8192|m=4096)).*n=[2-4],'
  for mt in 1 0; do
    for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
      echo "-- MT=$mt $op"
      ssh "$RIG" "cd $BUILDS/$NAME/bin && CUDA_VISIBLE_DEVICES=0 GGML_CUDA_MMVQ_PASCAL_MT=$mt ./test-backend-ops -b CUDA0 -o $op -p '$small'" 2>&1 \
        | grep -E "FAIL|tests passed|Backend .*: (OK|FAIL)|Abort|CUDA error" | sort | uniq -c
    done
  done
fi

perf() {  # perf LABEL ENV TYPES
  echo "-- $1"
  for t in $3; do
    ssh "$RIG" "cd $BUILDS/$NAME/bin && export CUDA_VISIBLE_DEVICES=0 GGML_TEST_RAW_QUANT=1 $2;
      ./test-backend-ops perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=$t,.*n_mats=64,n_used=8';
      ./test-backend-ops perf -b CUDA0 -o MUL_MAT -p 'type_a=$t,.*m=(1024|3072|8192|248320),n=[1-4],k=(1024|3072|8192),'" 2>&1 \
      | grep -E "us/run" | sed -E 's/^ +//; s/ +/ /g'
  done
}

# the fused gate/up: perf mode repeats a graph's last node (the GLU), so each case runs once under GGML_CUDA_PROFILE
# and per graph evaluation the launch groups that start with MUL_MAT_ID are summed (the fused gate/up and its GLU, or
# the two expert products where a kernel doesn't fuse); the median over the evaluations is the case's time
fused() {  # fused LABEL ENV TYPES; prints: type, tokens, median microseconds, evaluations, the groups' labels
  echo "-- $1, fused gate/up"
  ssh "$RIG" "cd $BUILDS/$NAME/bin && export CUDA_VISIBLE_DEVICES=0 GGML_TEST_RAW_QUANT=1 $2
    for t in $3; do for m in 1 2 3 4; do
      rm -f /tmp/mtpk-fused.csv
      GGML_CUDA_PROFILE=/tmp/mtpk-fused.csv ./test-backend-ops perf -b CUDA0 -o MUL_MAT_VEC_FUSION \
        -p \"type=\$t,glu_op=2,m=\$m,n=1024,k=3072,use_id=1,n_mats=64,\" > /dev/null 2>&1
      med=\$(awk -F, '\$5 ~ /^MUL_MAT_ID/ { s[\$2] += \$6 } END { for (e in s) print s[e] }' /tmp/mtpk-fused.csv \
        | sort -n | awk '{ a[NR] = \$1 } END { printf \"%.2f %d\", a[int((NR + 1) / 2)], NR }')
      lab=\$(awk -F, '\$5 ~ /^MUL_MAT_ID/ { l[\$5] = 1 } END { for (k in l) printf \"%s; \", k }' /tmp/mtpk-fused.csv)
      echo \"\$t \$m \$med \$lab\"
    done; done; rm -f /tmp/mtpk-fused.csv"
}

if has 3; then
  echo "== 3. kernels, card 0 $(date +%T)"
  perf "MT (new kernel)" "" "$TYPES"
  perf "MC (GGML_CUDA_MMVQ_PASCAL_MT=0)" "GGML_CUDA_MMVQ_PASCAL_MT=0" "$TYPES"
  perf "llama.cpp's kernel for 2-4 tokens (GGML_CUDA_MMVQ_PASCAL_MC=0)" "GGML_CUDA_MMVQ_PASCAL_MC=0" "$TYPES"
  fused "MT" "" "$TYPES"
  fused "MC" "GGML_CUDA_MMVQ_PASCAL_MT=0" "$TYPES"
  fused "llama.cpp" "GGML_CUDA_MMVQ_PASCAL_MC=0" "$TYPES"
  for v in "RB=1" "RB=2" "RB=4" "ACT_KB=8" "ACT_KB=32"; do
    perf "MT, GGML_CUDA_MMVQ_PASCAL_MT_$v" "GGML_CUDA_MMVQ_PASCAL_MT_$v" "n2p n3p iq3_xxs"
  done
fi
echo "== done $(date +%T)"
