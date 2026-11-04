#!/usr/bin/env bash
# Q860's quality by k: KL divergence of a model (MODEL, e.g. Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next) against the
# Q8_0 reference's logits of ../../2026-11-03_flash_next_quality (its kld.sh bases, the same three texts, flags and
# tensor split: -ub 512 -fa on, every layer on the GPUs, $BASES/gpu_args), once per LLAMA_MOE_ROUTE_ONCE arm, with
# llama-perplexity of the -routeonce branch built for the GPUs at hand (PPL, run/build_gpu.sh's cuda86 step).
# Route once groups each 512-token micro-batch into blocks of k consecutive positions, so k = 2-5 here prices what a
# k-token verify pass would see at every one of its positions. "off" (no switch) and "k1" (the masked path with every
# token its own block's first: the same experts) are the controls; every arm but off logs the histogram of distinct
# experts per block and layer (llama_context's "moe route once" lines, shown at -lv 4): read (10 when it routes) and
# under independent routing (what a k-token pass reads without it).
# First, with TEST (the cuda86 build's test-route-once) and RANDOM_MODELS (run/cpu_checks.sh's random/ folder), the
# CUDA check on the first GPU of GPUS: the same checks as on the CPU, the exact arms within rounding.
#   (env: PPL BASES OUT MODEL GPUS MOUNTS WIKI AGENT CODE [TEST RANDOM_MODELS] [TEXTS] [ARMS] [TAG]) bash kld_route_once.sh
# ARMS: names from the list below (default: off k1 k2 k3 k4 k5, in order); TEXTS: agent wiki code (default: all,
# agent first). Logs that are complete are not run again. Writes $OUT/<arm>.<text>.log and $OUT/cuda_unit*.log;
# table: run/table_route_once.sh.
set -uo pipefail
: "${PPL:?}" "${BASES:?}" "${OUT:?}" "${MODEL:?}" "${GPUS:?}" "${WIKI:?}" "${AGENT:?}" "${CODE:?}"
TAG=${TAG:-routeonce}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

# arm name and its LLAMA_MOE_ROUTE_ONCE (empty: the switch off)
arm_env() {
    case $1 in
        off)        echo "" ;;
        k1)         echo "1,hist=1" ;;
        k[2-9])     echo "${1#k},hist=1" ;;
        dry[2-9])   echo "${1#dry},dry=1,hist=1" ;;
        *)          return 1 ;;
    esac
}
ARMS=${ARMS:-"off k1 k2 k3 k4 k5"}

mount_args=""
for d in ${MOUNTS:-}; do mount_args="$mount_args -v $d:$d"; done
bindir=$(dirname "$(dirname "$PPL")")
split=$(cat "$BASES/gpu_args" 2> /dev/null || true)
failed=0

if [ -n "${TEST:-}" ] && [ -n "${RANDOM_MODELS:-}" ]; then
    gpu0=${GPUS%%,*}
    for m in qwen4exp-moe qwen4exp-moe-q5k; do
        lg=$OUT/cuda_unit-$m.log
        grep -q "checks, 0 failed" "$lg" 2> /dev/null && continue
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --name "pq-$TAG-unit" --gpus "\"device=$gpu0\"" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID -v "$(dirname "$(dirname "$TEST")"):$(dirname "$(dirname "$TEST")")" \
            -v "$RANDOM_MODELS:$RANDOM_MODELS" $mount_args "$IMAGE" \
            "$TEST" -m "$RANDOM_MODELS/$m.gguf" -d CUDA0 -n 16 --router-scale 30 > "$lg" 2>&1
        log "cuda unit $m: exit $?; $(tail -n 1 "$lg")"
        grep -q "checks, 0 failed" "$lg" || failed=1
    done
fi

for arm in $ARMS; do
    ro_env=$(arm_env "$arm") || { log "unknown arm $arm"; failed=1; continue; }
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
        log "$arm $t: start (LLAMA_MOE_ROUTE_ONCE=${ro_env:-unset})"
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --name "pq-$TAG-kld" --gpus "\"device=$GPUS\"" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID ${ro_env:+-e LLAMA_MOE_ROUTE_ONCE=$ro_env} \
            -v "$bindir:$bindir" $mount_args "$IMAGE" \
            "$PPL" -m "$MODEL" -f "$txt" -c $c -b $c -ub 512 --chunks $n -ngl 99 -fa on $split \
            --kl-divergence-base "$base" --kl-divergence -lv 4 > "$lg" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$lg"; then
            log "$arm $t: $(grep -E "^(Mean    KLD|Same top p)" "$lg" | tr -s ' ' | tr '\n' ' ') $(grep -o 'read mean [0-9.]*, independent routing mean [0-9.]*' "$lg" | tail -n 1)"
        else
            log "FAILED $arm $t (exit $rc): $lg"; failed=1
        fi
    done
done
exit $failed
