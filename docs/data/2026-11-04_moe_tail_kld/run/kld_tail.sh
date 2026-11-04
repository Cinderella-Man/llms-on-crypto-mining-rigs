#!/usr/bin/env bash
# Q618's quality by threshold: KL divergence of a model (MODEL, e.g. Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next)
# against the Q8_0 reference's logits of ../../2026-11-03_flash_next_quality (its kld.sh bases, the same three texts,
# flags and tensor split: -ub 512 -fa on, every layer on the GPUs, $BASES/gpu_args), once per LLAMA_MOE_TAIL arm, with
# llama-perplexity of the bytes-switches branch built for the GPUs at hand (PPL). The arms "off" (no switch) and "p1"
# (the switch on, p = 1: nothing dropped, the same KLD) are the controls; each arm's log ends with the histogram of
# experts kept (llama_context's "moe tail" lines, shown at -lv 4), which gives the bytes the cut would save.
#   (env: PPL BASES OUT MODEL GPUS MOUNTS WIKI AGENT CODE [TEXTS] [ARMS] [TAG] [STOP_AT] [EST]) bash kld_tail.sh
# ARMS: names from the list below (default: all, in order); TEXTS: wiki agent code (default: agent first).
# STOP_AT (epoch seconds): no run starts that would end after it, EST seconds a run (default 330), so the cards
# are free at a set time; STOP_IF (paths): no run starts once one exists. The script exits 2 then, and a later call
# with the same OUT picks up the runs left.
# Logs that are complete are not run again. Writes $OUT/<arm>.<text>.log; table: run/table.sh.
# The same script as ../../2026-11-04_bytes_switches/run/kld_tail.sh, plus STOP_AT and the done marker $OUT/DONE.
set -uo pipefail
: "${PPL:?}" "${BASES:?}" "${OUT:?}" "${MODEL:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}"
TAG=${TAG:-bytesw}
EST=${EST:-330}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

# arm name and its LLAMA_MOE_TAIL (empty: the switch off)
arm_env() {
    case $1 in
        off|offr) echo "" ;;  # offr: off again, the run-to-run spread
        p1)       echo "p=1" ;;
        p099r)    echo "p=0.99,renorm=1" ;;
        p095r)    echo "p=0.95,renorm=1" ;;
        p090r)    echo "p=0.9,renorm=1" ;;
        p080r)    echo "p=0.8,renorm=1" ;;
        p070r)    echo "p=0.7,renorm=1" ;;
        p095)     echo "p=0.95" ;;
        p090)     echo "p=0.9" ;;
        w002r)    echo "w=0.02,renorm=1" ;;
        w005r)    echo "w=0.05,renorm=1" ;;
        p090w003r) echo "p=0.9,w=0.03,renorm=1" ;;
        *)        return 1 ;;
    esac
}
# default: the controls, then the cuts from mildest to strongest
ARMS=${ARMS:-"off p1 p099r w002r p095r p095 w005r p090r p090 p090w003r p080r p070r"}

mount_args=""
for d in ${MOUNTS:-}; do mount_args="$mount_args -v $d:$d"; done
bindir=$(dirname "$(dirname "$PPL")")
split=$(cat "$BASES/gpu_args" 2> /dev/null || true)
failed=0

for arm in $ARMS; do
    tail_env=$(arm_env "$arm") || { log "unknown arm $arm"; failed=1; continue; }
    for t in ${TEXTS:-agent wiki code}; do
        case $t in
            wiki)  txt=$WIKI c=512 n=80 ;;
            agent) txt=$AGENT c=8192 n=12 ;;
            code)  txt=$CODE c=512 n=80 ;;
            *) log "unknown text $t"; failed=1; continue ;;
        esac
        base=$BASES/base.$t.kld lg=$OUT/$arm.$t.log
        [ -f "$base" ] || { log "no base $base"; failed=1; continue; }
        grep -q "^Mean    KLD" "$lg" 2> /dev/null && continue
        if [ -n "${STOP_AT:-}" ] && [ $(( $(date +%s) + EST )) -gt "$STOP_AT" ]; then
            log "STOPPED before $arm $t (STOP_AT $(date -d "@$STOP_AT" '+%F %T'))"; exit 2
        fi
        for f in ${STOP_IF:-}; do
            [ -e "$f" ] && { log "STOPPED before $arm $t ($f exists)"; exit 2; }
        done
        log "$arm $t: start (LLAMA_MOE_TAIL=${tail_env:-unset})"
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --name "pq-$TAG-kld" --gpus "\"device=$GPUS\"" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID ${tail_env:+-e LLAMA_MOE_TAIL=$tail_env} \
            -v "$bindir:$bindir" $mount_args "$IMAGE" \
            "$PPL" -m "$MODEL" -f "$txt" -c $c -b $c -ub 512 --chunks $n -ngl 99 -fa on $split \
            --kl-divergence-base "$base" --kl-divergence -lv 4 > "$lg" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$lg"; then
            log "$arm $t: $(grep -E "^(Mean    KLD|Same top p)" "$lg" | tr -s ' ' | tr '\n' ' ') $(grep -o 'experts kept: mean [0-9.]* of [0-9]*' "$lg" | tail -n 1)"
        else
            log "FAILED $arm $t (exit $rc): $lg"; failed=1
        fi
    done
done
[ $failed -eq 0 ] && log "DONE" && date '+%F %T' > "$OUT/DONE"
exit $failed
