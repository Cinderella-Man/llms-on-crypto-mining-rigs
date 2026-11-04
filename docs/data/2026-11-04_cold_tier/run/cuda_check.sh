#!/usr/bin/env bash
# Q238's check on one CUDA card (~2 minutes, under 1 GB): the random qwen4exp models of run/cpu_checks.sh (its random
# step) through a CUDA build of -coldtier. The sharp random model (stdev 0.6) writes its logits on the card; its
# same-type tiered copy (hot 24 of 32) must match them to rounding (the matrix kernels may group a tier's tokens
# differently, so not bit for bit), the negative control (cold slabs of the wrong experts) must differ by far more,
# and the F32-hot / Q4_0-cold file of the stdev 0.1 model must run. Each at the default micro-batch (the batched matrix kernels) and at -ub 1 (the one-token
# kernels a decode uses).
# Env: WORK (cpu_checks.sh's WORK), BIN (the CUDA build's bin with llama-perplexity), TEXT, GPU (the card's UUID or
#      PCI address: the container sees that card alone, CUDA_VISIBLE_DEVICES=0 inside), IMAGE, CHUNKS (4)
# Writes WORK/checks/cuda/ and one verdict line per check in WORK/checks.tsv.
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${TEXT:?}" "${GPU:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CHUNKS=${CHUNKS:-4}
K=$WORK/checks/cuda R=$WORK/checks/random M=$WORK/random
mkdir -p "$K"
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/checks.tsv"; }
mounts="-v $WORK:$WORK -v $(dirname "$BIN"):$(dirname "$BIN") -v $(dirname "$TEXT"):$(dirname "$TEXT")"
# shellcheck disable=SC2086
ppl() { docker run --rm --label pascal-quants=1 --cpu-shares 256 --name pq-coldtier-cuda --gpus "\"device=$GPU\"" \
            -e CUDA_VISIBLE_DEVICES=0 $mounts "$IMAGE" "$BIN/llama-perplexity" -m "$1" -f "$TEXT" \
            -c 128 -b 128 --chunks "$CHUNKS" -ngl 99 "${@:2}"; }
# the card's Xid count in the kernel log, before and after
xids() { dmesg 2> /dev/null | grep -ci "xid"; }
x0=$(xids)
field() { grep -E "^$2" "$1" | awk '{print $'"$3"'}'; }
for ub in 128 1; do
    for m in qwen4exp-moe-sharp qwen4exp-moe; do
        ppl "$M/$m.gguf" -ub "$ub" --kl-divergence-base "$K/$m-ub$ub.kld" > "$K/$m-ub$ub.log" 2>&1
        grep -q "Final estimate" "$K/$m-ub$ub.log" || verdict "cuda ub $ub base $m" FAIL "$K/$m-ub$ub.log"
    done
    h24=""
    for arm in h24:qwen4exp-moe-sharp mismatch:qwen4exp-moe-sharp mixed:qwen4exp-moe; do
        name=${arm%%:*} m=${arm#*:}
        lg=$K/$name-ub$ub.log
        ppl "$R/$m-$name/$m.gguf" -ub "$ub" --kl-divergence-base "$K/$m-ub$ub.kld" --kl-divergence > "$lg" 2>&1
        kld=$(field "$lg" "Mean    KLD" 3) top=$(field "$lg" "Same top p" 4) mx=$(field "$lg" "Maximum KLD" 3)
        if [ -z "$kld" ]; then verdict "cuda ub $ub $name" FAIL "$lg"; continue; fi
        case $name in
            h24) h24=$kld
                 ok=$(awk -v k="$kld" -v m="$mx" 'BEGIN { print (k < 1e-4 && m < 1e-2) ? "OK" : "FAIL" }') ;;
            mismatch) ok=$(awk -v k="$kld" -v h="${h24:-1}" 'BEGIN { print (k > 1e-3 && k > 100 * h) ? "OK" : "FAIL" }') ;;
            mixed) ok=OK ;;
        esac
        verdict "cuda ub $ub $name" "$ok" "KLD $kld (max $mx), same top $top %"
    done
done
# the tiered models' MUL_MAT_IDs on the card: ids distinct per token (test-moe-ids, if the build has it)
if [ -x "$BIN/test-moe-ids" ]; then
    for f in "$R/qwen4exp-moe-sharp-h24/qwen4exp-moe-sharp.gguf" "$R/qwen4exp-moe-mixed/qwen4exp-moe.gguf"; do
        lg=$K/moeids-$(basename "$(dirname "$f")").log
        # shellcheck disable=SC2086
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --name pq-coldtier-cuda --gpus "\"device=$GPU\"" \
            -e CUDA_VISIBLE_DEVICES=0 $mounts "$IMAGE" "$BIN/test-moe-ids" -m "$f" -ngl 99 > "$lg" 2>&1
        rc=$?
        verdict "cuda moeids $(basename "$(dirname "$f")")" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" \
            "$(grep -E '^test-moe-ids:|checks,' "$lg" | tr '\n' ' ')"
    done
fi
x1=$(xids)
verdict "cuda xid" "$([ "$x1" = "$x0" ] && echo OK || echo FAIL)" "kernel log Xid lines: $x0 before, $x1 after"
