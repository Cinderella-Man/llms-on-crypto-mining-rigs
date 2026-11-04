#!/usr/bin/env bash
# A file's KL divergence against the bases of ../../2026-11-03_flash_next_quality (Swift-1.5 Q8_0's logits on
# wikitext-2 test 80 x 512, held-out agent sessions 12 x 8,192, the agent check's Exercism code 80 x 512) with the
# llama-perplexity of agentionai's fork (build.sh), in Docker, with that chain's flags (-ub 512 -fa on, every layer on
# the GPUs). The bases carry their own tokens, so the texts themselves are not read.
#   kld.sh <model> <name>     logs $OUT/<name>.<text>.log; a complete log is not run again, a failed text is reported
#                             and the others still run (exit status 1 at the end)
# BACKEND picks the build ($BIN/<backend>/bin/llama-perplexity): cuda runs in the CUDA image, vulkan in a small Ubuntu
# image with the Vulkan loader, the driver's Vulkan ICD mounted by the NVIDIA Container Toolkit (graphics capability).
# EXTRA adds llama-perplexity arguments (a tensor split, --ngram-on-disk).
#   (env: BIN BASES OUT GPUS MOUNTS [BACKEND] [TEXTS] [EXTRA] [TAG]) bash kld.sh <model> <name>
set -uo pipefail
: "${BIN:?}" "${BASES:?}" "${OUT:?}" "${GPUS:?}"
model=$1 name=$2
BACKEND=${BACKEND:-cuda} TAG=${TAG:-gyro}
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
ppl=$BIN/$BACKEND/bin/llama-perplexity
[ -x "$ppl" ] || { log "no $ppl"; exit 1; }
mounts="-v $BIN:$BIN -v $BASES:$BASES -v $OUT:$OUT"
for d in ${MOUNTS:-}; do mounts="$mounts -v $d:$d"; done
case $BACKEND in
    cuda) image=nvidia/cuda:12.9.2-devel-ubuntu24.04 caps="" ;;
    vulkan) image=pq-gyro-vulkan:24.04 caps="-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
        docker image inspect $image > /dev/null 2>&1 || printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && apt-get install -y -qq libvulkan1 libgomp1 > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
            | docker build -q --label pascal-quants=1 -t $image - > /dev/null || { log "image build failed"; exit 1; } ;;
    *) log "BACKEND is cuda or vulkan"; exit 1 ;;
esac
mkdir -p "$OUT"
failed=0
for t in ${TEXTS:-wiki agent code}; do
    case $t in
        wiki|code) c=512 n=80 ;;
        agent) c=8192 n=12 ;;
        *) log "unknown text $t"; failed=1; continue ;;
    esac
    base=$BASES/base.$t.kld out=$OUT/$name.$t.log
    [ -f "$base" ] || { log "no base $base"; failed=1; continue; }
    grep -q "^Mean    KLD" "$out" 2> /dev/null && continue
    log "$name $t: start ($BACKEND, GPUs $GPUS)"
    docker run --rm --label pascal-quants=1 --cpu-shares 256 --name "pq-$TAG-kld-$t" --gpus "\"device=$GPUS\"" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID $caps $mounts $image \
        "$ppl" -m "$model" -c $c -b $c -ub 512 --chunks $n -ngl 99 -fa on ${EXTRA:-} \
        --kl-divergence-base "$base" --kl-divergence > "$out" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$out"; then
        log "$name $t: $(grep -E "^(Mean    KLD|Same top p)" "$out" | tr -s ' ' | tr '\n' ' ')"
    else
        log "FAILED $name $t (exit $rc): $out"; failed=1
    fi
done
exit $failed
