#!/usr/bin/env bash
# KLD of one GGUF against Flint BF16's logits on 80 x 512 tokens of wikitext-2 test, on the GPUs in $GPUS: the
# measurement of scripts/quality_kld.sh (same text, chunk count and reference file), run on a CUDA build of Miner Llama
# so that environment switches reach llama-perplexity (the 2026-10-23/24 quant runs used
# ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1). Every score in this folder and in quants/recipes/flint-mixes.tsv was made this way.
#
#   kld_gpu.sh <candidate.gguf> [tag]
#
# Appends one line to $WORK/scores.tsv: time, name+tag, PPL, PPL ratio to BF16, KLD ± SE, 99th-percentile KLD, same top
# token % ± SE, and the number of log lines that mention nan.
# Env: WORK (holds wikitext-2-raw/wiki.test.raw and flint-BF16.c512x80.kld, the reference logits that
#      scripts/quality_kld.sh's reference step writes), BIN (the build's bin/ directory), GPUS (e.g. 5,6), TS (optional
#      -ts split), ENVS (space-separated VAR=value pairs for the container), CAND_ARGS (extra llama-perplexity options).
set -euo pipefail
C=$(realpath "$1"); TAG=${2:-}
base=$WORK/flint-BF16.c512x80.kld
name=$(basename "$C" .gguf)$TAG
ts=""; [ -n "${TS:-}" ] && ts="-ts $TS"
envs=""; for e in ${ENVS:-}; do envs="$envs -e $e"; done
mkdir -p "$WORK/logs"; L=$WORK/logs/kld-$name.log
docker run --rm --gpus "\"device=$GPUS\"" -e CUDA_DEVICE_ORDER=PCI_BUS_ID $envs \
  -v "$WORK:$WORK" -v "$(dirname "$C"):$(dirname "$C"):ro" -v "$(dirname "$BIN"):$(dirname "$BIN"):ro" \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  "$BIN/llama-perplexity" -m "$C" -f "$WORK/wikitext-2-raw/wiki.test.raw" -c 512 --chunks 80 -ngl 99 $ts -fa on \
  --kl-divergence-base "$base" --kl-divergence ${CAND_ARGS:-} > "$L" 2>&1
ppl=$(grep -E "^Mean PPL\(Q\) +:" "$L" | awk '{print $4}')
pplr=$(grep -E "^Mean PPL\(Q\)/PPL\(base\)" "$L" | awk '{print $4}')
kld=$(grep -E "^Mean +KLD" "$L" | awk '{print $3"±"$5}')
kld99=$(grep -E "^99.0% +KLD" "$L" | awk '{print $3}')
top=$(grep -E "^Same top p" "$L" | awk '{print $4"±"$6}')
nan=$(grep -c -i nan "$L" || true)
printf "%s\t%s\tppl=%s\tppl_ratio=%s\tkld=%s\tkld99=%s\ttop=%s\tnan_lines=%s\n" "$(date +%H:%M)" "$name" "$ppl" "$pplr" \
  "$kld" "$kld99" "$top" "$nan" | tee -a "$WORK/scores.tsv"
