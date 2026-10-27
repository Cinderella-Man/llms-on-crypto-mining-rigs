#!/usr/bin/env bash
# Correctness suite for the types of the fork's Pascal matrix-vector kernels, through tbo.sh on one GPU: the plain
# products (MUL_MAT, MUL_MAT_ADD), the expert products (MUL_MAT_ID at 1-4 tokens and from 9 tokens on), the fused
# products and the shared-expert products of the types in TYPES. One summary line per part in
# $WORK/logs/<tag>-summary.txt (tests passed, FAIL lines, exit status). Used for Q149 (2026-10-24).
#
# Usage: [TYPES='(q4_K|q6_K)'] [GPU=n] [LOCK=file] [STAGE_ACT=1] [SKIP=...] WORK=<dir> run_suite.sh <tag> <GGML_CUDA_MMVQ_PASCAL> <build dir>
# Arms of a kernel check: 2 (forced), 2 with STAGE_ACT=1, 0 (off); a mutation build (verify/q4k-mut.diff) at 2 must
# fail exactly the cases the kernel takes (tbo_tally.py counts them), and at 2 with SKIP=q4_K must pass everything.
# 5-8-token expert products are left out: in builds with compute_61 PTX only they abort on newer GPUs with every
# kernel, llama.cpp's own included.
D=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:?WORK=<work dir>}
tag=$1; mode=$2; B=$3
T=${TYPES:-"(q4_K|q6_K)"}
"$D/tbo.sh" $tag-mm $mode $B -o MUL_MAT,MUL_MAT_ADD -p "^type_a=$T,"
"$D/tbo.sh" $tag-mmid $mode $B -o MUL_MAT_ID -p "^type_a=$T,.*,n=([1-4]|9|[1-9][0-9]+),"
"$D/tbo.sh" $tag-fus $mode $B -o MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION -p "^type=$T,.*,m=([1-4]|9|[1-9][0-9]+),"
"$D/tbo.sh" $tag-shared $mode $B -o MUL_MAT_ID_SHARED -p "^type=$T,n=[1-4]\$"
for p in mm mmid fus shared; do
    f=$WORK/logs/$tag-$p.log
    printf "%-22s %s | %s fail | %s\n" "$tag-$p" "$(grep -a -E "tests passed" $f | tail -1 | xargs)" "$(grep -a -c FAIL $f)" "$(tail -1 $f)"
done > $WORK/logs/$tag-summary.txt
cat $WORK/logs/$tag-summary.txt
