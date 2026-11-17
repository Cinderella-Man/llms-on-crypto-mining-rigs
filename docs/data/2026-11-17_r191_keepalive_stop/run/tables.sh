#!/usr/bin/env bash
# R191's tables from the deep lane's results folder (r191.sh's OUT), with R192's scripts unchanged: how every graph
# evaluation ran, pass by pass (graph_passes.py), the debug lines and the keep-alive's counts per request
# (server_graph_log.py), R121's first-request comparison (first_request.py, with R86's align_t0.py), then compare.py:
# this run (GGML_CUDA_KEEPALIVE_EAGER=0) against R192's published tables and profile (=1). CPU only, standard-library
# Python 3.
# Env: S (this repository's root), OUT (the deep lane's results: gp-k1.csv, hp-k1.csv, st-k1.jsonl, server-k1.log,
# requests-k1.csv, requests.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r191_keepalive_stop/run
R192=$S/docs/data/2026-11-17_r192_first_request_graphs
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
N="nice -n 19 python3 -I"
mkdir -p "$RES"
rm -f "$RES/DONE"
$N "$R192/run/graph_passes.py" "$OUT/gp-k1.csv" --names d4096p1,d4096p2 --tsv "$RES/passes.tsv" > "$RES/passes.md"
$N "$R192/run/server_graph_log.py" "$OUT/server-k1.log" --requests "$OUT/requests-k1.csv" --tsv "$RES/tasks.tsv" > "$RES/tasks.md"
$N "$AL" "$OUT/st-k1.jsonl" "$OUT/hp-k1.csv" > "$RES/align-k1.md"
t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/align-k1.md" | tail -n 1)
$N "$R121/first_request.py" --trace "$OUT/st-k1.jsonl" --requests "$OUT/requests-k1.csv" --profile "$OUT/hp-k1.csv" \
  --profile-t0 "$t0" --depth 4096 --tsv "$RES/steps-k1.tsv" --pairs-tsv "$RES/pairs-k1.tsv" > "$RES/first-k1.md"
$N "$D/compare.py" --a-passes "$R192/passes.tsv" --b-passes "$RES/passes.tsv" \
  --a-tasks "$R192/tasks.tsv" --b-tasks "$RES/tasks.tsv" \
  --a-requests "$R192/requests.csv" --b-requests "$OUT/requests.csv" \
  --a-gp "$R192/gp-g1.csv.gz" --b-gp "$OUT/gp-k1.csv" --a-name EAGER=1 --b-name EAGER=0 --a-load g1 --b-load k1 \
  --tsv "$RES/compare.tsv" > "$RES/compare.md"
touch "$RES/DONE"
