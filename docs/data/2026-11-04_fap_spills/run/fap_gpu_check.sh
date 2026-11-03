#!/usr/bin/env bash
# Correctness of the Pascal decode attention kernel without local memory (fork branch pascal-p106-v3-b12076-fap), on
# one CUDA GPU that runs the builds' compute_61 PTX (any GPU newer than Pascal; the tarballs carry sm_61 code and
# compute_61 PTX, CUDA_FORCE_PTX_JIT=1 makes the PTX path explicit). No timings. Each step in its own container:
#   1. test-backend-ops FLASH_ATTN_EXT, every case, the kernel forced (GGML_CUDA_FATTN_PASCAL=2: not a 6.1 card); the
#      kernel takes the Flint-shape cases (head 128, 8 query heads per KV head, F16 K/V), the stock kernels the rest
#   2. the kernel's faults on its own cases: 2 (sinks dropped) and 3 (each block's last key tile dropped) must fail
#      cases; 1 (output x 1.01) stays inside the 5e-4 NMSE limit, as it did in v2.0
#   3. the same build with the kernel off (=0) on its own cases: the stock kernels
#   4. llama-perplexity -ub 1 (every token a one-token step through the kernel) on a Flint file with the kernel forced:
#      the logits written by BEFORE twice and by AFTER once, byte for byte. BEFORE against itself shows the run is
#      deterministic; BEFORE against AFTER is the bit-identity of the change
# BEFORE, AFTER: unpacked tarball trees (bin/ inside) of the -ladder build (98bbd6a2f) and the -fap build; MODEL: a Flint
# GGUF (Q73's G3, 29.6 GB, was v2.0's check file: two 24 GB GPUs, GPU=a,b); TEXT: wikitext-2's wiki.test.raw; WORK: a
# folder for logs and the JIT cache. Every drive the paths are on must be mounted: MOUNTS="-v /x:/x ..." (same path).
# Usage: BEFORE=<dir> AFTER=<dir> MODEL=<gguf> TEXT=<file> WORK=<dir> GPU=<n|a,b> [MOUNTS=...] [DOCKER_ARGS=...] fap_gpu_check.sh
set -u
BEFORE=${BEFORE:?BEFORE=<tree of the -ladder build>}
AFTER=${AFTER:?AFTER=<tree of the -fap build>}
MODEL=${MODEL:?MODEL=<Flint gguf>}
TEXT=${TEXT:?TEXT=<wiki.test.raw>}
W=${WORK:?WORK=<work dir>}
GPU=${GPU:?GPU=<device or a,b>}
mkdir -p $W/logs $W/cuda-cache $W/kld
dock() { # <space-separated VAR=value list or -> <command...>
    local envs=() e
    [ "$1" != - ] && for e in $1; do envs+=(-e "$e"); done
    shift
    docker run --rm ${DOCKER_ARGS:-} --gpus "\"device=$GPU\"" -v "$W:$W" -v "$BEFORE:$BEFORE:ro" -v "$AFTER:$AFTER:ro" \
        ${MOUNTS:-} -e CUDA_FORCE_PTX_JIT=1 -e CUDA_CACHE_PATH=$W/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 "${envs[@]}" \
        nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@"
}
T="$AFTER/bin/test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT"
OWN='hsk=128,hsv=128,nh=(4|8),nr23=\[8,1\],kv=[0-9]+,nb=1,'
summ() { echo "$1: $(grep -oE '[0-9]+/[0-9]+ tests passed' $2 | tail -n 1), FAIL lines $(grep -c FAIL $2)"; }

echo "=== 1. all FLASH_ATTN_EXT cases, kernel forced  $(date +%T)"
dock "GGML_CUDA_FATTN_PASCAL=2" $T > $W/logs/t1-all-on.log 2>&1
summ "on, all" $W/logs/t1-all-on.log
echo "=== 2. faults on the kernel's own cases  $(date +%T)"
for f in 1 2 3; do
    dock "GGML_CUDA_FATTN_PASCAL=2 GGML_CUDA_FATTN_PASCAL_FAULT=$f" $T -p "$OWN" > $W/logs/t2-fault$f.log 2>&1
    summ "fault $f (1: passes, 2 and 3: must fail cases)" $W/logs/t2-fault$f.log
done
echo "=== 3. own cases, kernel off  $(date +%T)"
dock "GGML_CUDA_FATTN_PASCAL=0" $T -p "$OWN" > $W/logs/t3-own-off.log 2>&1
summ "off, own" $W/logs/t3-own-off.log

echo "=== 4. logits, kernel forced, -ub 1: before twice, after once  $(date +%T)"
ppl() { # <tree> <out name>
    dock "GGML_CUDA_FATTN_PASCAL=2" $1/bin/llama-perplexity -m "$MODEL" -f "$TEXT" -c 512 --chunks 4 -b 512 -ub 1 \
        -ngl 99 -fa on --kl-divergence-base $W/kld/$2.kld > $W/logs/t4-$2.log 2>&1 || echo "$2: exit status $?"
    echo "$2: $(grep -oE 'Final estimate.*' $W/logs/t4-$2.log), md5 $(md5sum < $W/kld/$2.kld | cut -c1-32)"
}
ppl "$BEFORE" before1
ppl "$BEFORE" before2
ppl "$AFTER" after
cmp -s $W/kld/before1.kld $W/kld/before2.kld && echo "before against before: identical" || echo "before against before: DIFFER"
cmp -s $W/kld/before1.kld $W/kld/after.kld && echo "before against after: identical" || echo "before against after: DIFFER"
echo "fap_gpu_check ended $(date +%T)"
