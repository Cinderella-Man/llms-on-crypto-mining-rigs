#!/usr/bin/env bash
# Q621 on one P106: lossless_bench over the weight sample, plain BF16 against x12 and x11 at 1 to 8 tokens, then the
# tables back to OUT and the rig cleaned. About 10 minutes with the copies; nothing else may run on the rig meanwhile.
# Env: RIG: ssh alias of the rig (root); BIN, SAMPLE: the bench binary (bench/mmvq_pascal/build_lossless.sh) and the
#      sample (analyze.sh), on this machine or, with FROM set, on the ssh host FROM (ssh arguments: options and the
#      host); MD5_BIN, MD5_SAMPLE: their md5s, checked on the rig before the run; OUT: where the tables go;
#      CARD: the PCI-order index of the card (default 0); ARGS: extra bench arguments (default none: all classes,
#      codes and token counts 1, 2, 3, 4, 5, 6, 8).
set -euo pipefail
: "${RIG:?}" "${BIN:?}" "${SAMPLE:?}" "${MD5_BIN:?}" "${MD5_SAMPLE:?}" "${OUT:?}"
CARD=${CARD:-0}
D=/tmp/lossless_q621
mkdir -p "$OUT"
# FROM may carry ssh options, so it is split into words
# shellcheck disable=SC2086
fetch() { if [ -n "${FROM:-}" ]; then ssh $FROM "cat '$1'"; else cat "$1"; fi; }

ssh "$RIG" "mkdir -p $D && nvidia-smi --query-gpu=index,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv" \
    > "$OUT/rig_gpus_before.csv"
fetch "$BIN" | ssh "$RIG" "cat > $D/lossless_bench && chmod +x $D/lossless_bench"
fetch "$SAMPLE" | ssh "$RIG" "cat > $D/sample.lbb"
ssh "$RIG" "cd $D && echo '$MD5_BIN  lossless_bench' > md5 && echo '$MD5_SAMPLE  sample.lbb' >> md5 && md5sum -c md5"

ssh "$RIG" "cd $D && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD ./lossless_bench --sample sample.lbb \
    --csv lossless.csv --summary lossless_summary.csv ${ARGS:-}" | tee "$OUT/lossless_rig.txt"
ssh "$RIG" "cat $D/lossless.csv" > "$OUT/lossless.csv"
ssh "$RIG" "cat $D/lossless_summary.csv" > "$OUT/lossless_summary.csv"
ssh "$RIG" "nvidia-smi --query-gpu=index,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv" \
    > "$OUT/rig_gpus_after.csv"
ssh "$RIG" "rm -rf $D"
grep -q "^all checks: PASS" "$OUT/lossless_rig.txt" && echo "rig run: PASS" || { echo "rig run: checks FAILED"; exit 1; }
