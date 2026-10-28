#!/usr/bin/env bash
# Q261's CPU check: test-llama-archs (random tiny models of every architecture: prompt, decode, mixed batches, a state
# round trip) from the fix branch's CPU build (build_cpu.sh TESTS=1), three ways: host mask only; the card-built mask
# for micro-batches of at most 16 tokens (LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16, the decode-only setting); for
# every micro-batch (LLAMA_GPU_KQ_MASK=1). With LLAMA_GPU_KQ_MASK_CHECK=1 each card-mask step's mask is compared with
# the host mask and the run aborts on any difference.
#   WORK=... BIN=$WORK/out-cpu-pp-t/build/bin bash mask_check_cpu.sh   (logs: $WORK/out/mask_cpu_<arm>.log)
set -uo pipefail
: "${WORK:?set WORK}"
BIN=${BIN:-$WORK/out-cpu-pp-t/build/bin}
mkdir -p "$WORK/out"
run() {  # arm env...
    local arm=$1; shift
    docker run --rm --cpu-shares=256 --label pascal-quants=1 --name "pq-rereads-mask-$arm" $(printf -- '-e %s ' "$@") \
        -v "$BIN:/bin-llama:ro" nvidia/cuda:12.9.2-devel-ubuntu24.04 /bin-llama/test-llama-archs -v 4 \
        > "$WORK/out/mask_cpu_$arm.log" 2>&1
    local rc=$?
    echo "$arm: exit $rc; $(grep -a "Summary" "$WORK/out/mask_cpu_$arm.log"); masks identical to the host mask: $(grep -a -c "identical to the host mask" "$WORK/out/mask_cpu_$arm.log"), by micro-batch size: $(grep -a -o "copy [0-9]*: [0-9]* tokens" "$WORK/out/mask_cpu_$arm.log" | awk '{print $3}' | sort -n | uniq -c | tr -s " \n" " ")"
}
run host LLAMA_GPU_KQ_MASK=0
run decode LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16 LLAMA_GPU_KQ_MASK_CHECK=1
run all LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1
