#!/usr/bin/env bash
# KL divergence on the three texts of ../../2026-10-27_ornith_kld (wikitext-2 test 80 x 512, held-out agent sessions
# 12 x 8,192, the agent check's Exercism code 80 x 512) with llama-perplexity in Docker on CUDA GPUs, with the flags of
# ../../2026-11-03_table_bits/run/kld.sh (-ub 512 -fa on, every layer on the GPUs).
#   kld.sh base <first shard>          the reference's logits, $OUT/base.<text>.kld (written as .partial, renamed when
#                                      the run ends with its final estimate), log $OUT/base.<text>.log
#   kld.sh kld <first shard> <name>    a model against those logits, log $OUT/<name>.<text>.log
# EXTRA adds llama-perplexity arguments (the `-ub 256` control); $OUT/gpu_args (fn1b.sh's fit check: the tensor split
# the bases ran with) is added to every run. Bases and logs that are complete are not run again; a failed text is
# reported and the others still run (exit status 1 at the end).
#   (env: PPL OUT GPUS MOUNTS WIKI AGENT CODE TAG [TEXTS] [EXTRA]) bash kld.sh ...
set -uo pipefail
. "$(dirname "$0")/common.sh"
: "${PPL:?}" "${OUT:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}"
mode=$1 model=$2 name=${3:-base}
mkdir -p "$OUT"
bindir=$(dirname "$(dirname "$PPL")")
split=$(cat "$OUT/gpu_args" 2> /dev/null || true)
ngl="-ngl 99"; case "$split" in *"-fit on"*) ngl="";; esac
failed=0
for t in ${TEXTS:-wiki agent code}; do
    case $t in
        wiki) txt=$WIKI c=512 n=80 ;;
        agent) txt=$AGENT c=8192 n=12 ;;
        code) txt=$CODE c=512 n=80 ;;
        *) log "unknown text $t"; failed=1; continue ;;
    esac
    base=$OUT/base.$t.kld log=$OUT/$name.$t.log
    if [ "$mode" = base ]; then
        [ -f "$base" ] && continue
        args="--kl-divergence-base $base.partial"
    else
        [ -f "$base" ] || { log "no base $base"; failed=1; continue; }
        grep -q "^Mean    KLD" "$log" 2> /dev/null && continue
        args="--kl-divergence-base $base --kl-divergence ${EXTRA:-}"
    fi
    log "$name $t: start"
    docker run --rm $DOCKER_BASE --name "pq-$TAG-kld" --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -v "$bindir:$bindir" $MOUNT_ARGS "$IMAGE" \
        "$PPL" -m "$model" -f "$txt" -c $c -b $c -ub 512 --chunks $n $ngl -fa on $split $args > "$log" 2>&1
    rc=$?
    if [ "$mode" = base ]; then
        if [ $rc -eq 0 ] && grep -q "Final estimate" "$log"; then
            mv "$base.partial" "$base"
            log "base $t: $(grep "Final estimate" "$log")"
        else
            rm -f "$base.partial"; log "FAILED base $t (exit $rc): $log"; failed=1
        fi
    elif [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$log"; then
        log "$name $t: $(grep -E "^(Mean    KLD|Same top p)" "$log" | tr -s ' ' | tr '\n' ' ')"
    else
        log "FAILED $name $t (exit $rc): $log"; failed=1
    fi
done
exit $failed
