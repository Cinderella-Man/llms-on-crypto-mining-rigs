#!/usr/bin/env bash
# N2P/N3P kernels on the rig (fork branch pascal-p106-v3-lowbit), with nothing else on the rig; the machine that runs this
# script only relays over ssh.
#   1. install: the CUDA build's tarball from the quality machine to BUILDS on the rig, md5 check, unpack
#   2. correctness on card 0: test-backend-ops (MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION, GET_ROWS) for n2p and n3p
#   3. kernels on card 0: test-backend-ops perf at Ornith-1.5-122B's shapes (8 of 64 experts: fused gate/up 3072 -> 1024,
#      down 1024 -> 3072; dense 3072 -> 8192; 1, 2, 4 tokens) for n2p, n3p, iq2_s, iq3_xxs, q2_K, q3_K, q4_0, q8_0 with
#      the Pascal kernels, then n2p/n3p on llama.cpp's kernel (GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p); the IQ and K types
#      run llama.cpp's kernel in this build
#   RIG=rig QM=<ssh alias> WORK=<work dir on QM, as for build.sh> NAME=<build name> MD5=<tarball md5> \
#     [BUILDS=/opt/llama-builds] [STEPS="1 2 3"] bash rig_kernels.sh > OUT/rig_kernels.txt
set -uo pipefail
RIG=${RIG:-rig}
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?work dir on QM}
NAME=${NAME:?build name}
MD5=${MD5:?md5 of NAME.tar.gz}
BUILDS=${BUILDS:-/opt/llama-builds}
STEPS=${STEPS:-1 2 3}
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has 1; then
  echo "== 1. install $NAME $(date +%T)"
  ssh "$QM" "cat $WORK/out/$NAME.tar.gz" | ssh "$RIG" "cat > $BUILDS/$NAME.tar.gz"
  ssh "$RIG" "cd $BUILDS && echo '$MD5  $NAME.tar.gz' | md5sum -c && rm -rf $NAME && tar xzf $NAME.tar.gz && rm $NAME.tar.gz && head -3 $NAME/BUILDINFO.txt" || exit 1
fi

if has 2; then
  echo "== 2. correctness, card 0 $(date +%T)"
  for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION GET_ROWS; do
    echo "-- $op"
    ssh "$RIG" "cd $BUILDS/$NAME/bin && CUDA_VISIBLE_DEVICES=0 ./test-backend-ops -b CUDA0 -o $op -p '=n[23]p,'" 2>&1 \
      | grep -E "FAIL|tests passed|Backend .*: (OK|FAIL)" | sort | uniq -c
  done
fi

perf() {  # perf LABEL ENV TYPES
  echo "-- $1"
  for t in $3; do
    ssh "$RIG" "cd $BUILDS/$NAME/bin && export CUDA_VISIBLE_DEVICES=0 $2;
      ./test-backend-ops perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=$t,.*n_mats=64,n_used=8';
      ./test-backend-ops perf -b CUDA0 -o MUL_MAT_VEC_FUSION -p '=$t,.*n=1024,k=3072,use_id=1,n_mats=64';
      ./test-backend-ops perf -b CUDA0 -o MUL_MAT -p 'type_a=$t,.*m=(1024|3072|8192),n=[124],k=(1024|3072),'" 2>&1 \
      | grep -E "us/run" | sed -E 's/^ +//; s/ +/ /g'
  done
}

if has 3; then
  echo "== 3. kernels, card 0 $(date +%T)"
  perf "Pascal kernels (llama.cpp's for the types the fork does not cover)" "" "n2p n3p iq2_s iq3_xxs q2_K q3_K q4_0 q8_0"
  perf "llama.cpp's kernel for n2p/n3p (GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p)" "GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p" "n2p n3p"
fi
echo "== done $(date +%T)"
