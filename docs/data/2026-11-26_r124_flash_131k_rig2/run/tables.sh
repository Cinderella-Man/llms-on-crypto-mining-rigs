#!/usr/bin/env bash
# R124's tables from the collected run, on any machine with python3 (standard library only). Run from the repository
# root: OUT (the collected results of run/r124.sh), PROBE (scripts/fit_probe.sh's output for the four lines of
# fit_probe.tsv), DEST (default this folder). The S38/R55 run of v3.0's 131,072-token line on the same prompts is
# docs/data/2026-11-10_deep_lane/ctx131k/ (its draft arm is the --second run of draft_pair.py).
set -euo pipefail
: "${OUT:?OUT=<collected results>}" "${PROBE:?PROBE=<fit_probe.sh output>}"
DEST=${DEST:-docs/data/2026-11-26_r124_flash_131k_rig2}
R=docs/data/2026-11-10_deep_lane/run
python3 -I bench/deep_lane/deep_table.py "$OUT" --control nodraft --out "$DEST/table.md" --tsv "$DEST/classes.tsv" > /dev/null
python3 -I "$R/draft_pair.py" "$OUT" --second docs/data/2026-11-10_deep_lane/ctx131k:draft \
  --second-name "v3.0's 131,072-token line (S38, R55)" \
  --yield "this run's sampler pass (one draw a depth):49152=3.37,65536=3.12,120000=3.71" \
  --yield "the preset's sampler off the rig (Q1172, Q1251):49152=2.876,65536=2.93,120000=2.69" --out "$DEST/draft_pair.md" --tsv "$DEST/draft_pair" > /dev/null
python3 -I "$R/host_memory.py" "$OUT" > "$DEST/host_memory.tsv"
python3 -I "$DEST/run/probe_table.py" "$PROBE" > "$DEST/fit_probe.tsv"
python3 -I "$DEST/run/card_fit.py" --probe "$PROBE" "$OUT" > "$DEST/card_fit.tsv"
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py "$OUT" --control-load nodraft1 \
  --loads draft1,nodraft1,nodraft2,draft2 > "$DEST/replies.tsv"
cp "$OUT/requests.csv" "$OUT/loads.tsv" "$DEST/"
