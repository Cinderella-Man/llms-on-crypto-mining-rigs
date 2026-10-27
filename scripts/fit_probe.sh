#!/usr/bin/env bash
# Projected memory per card without loading the weights: llama-server's --fit step at trace level (-lv 5) computes the
# model, context and compute buffers per device from the file's metadata (no allocation), then the server is stopped
# before the weights load, so each configuration takes seconds instead of a full load. Run from a machine that reaches
# the rig over ssh, with nothing else on the rig. Used for Ornith-1.5-122B (docs/data/2026-10-27_ornith/).
#   RIG=rig MODEL=<rig path of the GGUF> BUILDS=<rig folder holding <build>/bin/llama-server> OUT=fit_probe.md \
#   CONFIGS='label|build|ctx|args;label|build|ctx|args' bash scripts/fit_probe.sh
# e.g. CONFIGS='ub128-120k|llama-p106-v2.0|122880|-b 2048 -ub 128 -ts 5,6,5,6,5,5,6,5,7'. NGL (default "-ngl 99")
# replaces the layer option. The log lines kept per config: per-device projections ("projected memory use", the
# memory breakdown with model, context and compute MiB), the fit verdict, layer count and splits.
set -uo pipefail
RIG=${RIG:-rig}
M=${MODEL:?the GGUF on the rig}
BUILDS=${BUILDS:?folder on the rig holding the builds}
OUT=${OUT:?output file}
{
echo "# fit_probe $(basename "$M") $(date -Is)"
IFS=';' read -ra CFG <<< "${CONFIGS:?}"
for cfg in "${CFG[@]}"; do
  IFS='|' read -r label build ctx bargs <<< "$cfg"
  echo "== $label: $build -c $ctx $bargs"
  ssh "$RIG" "pkill -x llama-server; sleep 1; rm -f /tmp/fit_probe.log
    setsid nohup $BUILDS/$build/bin/llama-server -m $M ${NGL--ngl 99} -fa on -c $ctx $bargs -np 1 -fit on -lv 5 \
      --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port 8096 > /tmp/fit_probe.log 2>&1 < /dev/null &
    for i in \$(seq 1 60); do grep -qE 'successfully fit|failed to fit|encountered an error' /tmp/fit_probe.log && break; sleep 2; done
    sleep 2; pkill -9 -x llama-server
    grep -E 'projected|  - CUDA|  - Host|will leave|cannot meet|context size|failed|error|n_layer|split|ngl_per_device|successfully' /tmp/fit_probe.log | grep -v 'unused tensor' | cut -c1-200 | head -160
    rm -f /tmp/fit_probe.log" 2>&1 | sed "s/^/$label | /"
done
} | tee -a "$OUT"
