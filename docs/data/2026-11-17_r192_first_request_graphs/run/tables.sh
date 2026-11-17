#!/usr/bin/env bash
# R192's tables from the deep lane's results folder (r192.sh's OUT): how every graph evaluation ran, pass by pass,
# the first request against its repeat (graph_passes.py, from the graph-mode profile), the CUDA backend's debug lines
# and the keep-alive's counts per request (server_graph_log.py, from the server log), and R121's first-request
# comparison from the step trace and the host profile (R121's first_request.py, with R86's align_t0.py for the host
# profile's clock). CPU only, standard-library Python 3; runs on a 4 GB host.
# Env: S (this repository's root), OUT (the deep lane's results: gp-g1.csv, hp-g1.csv, st-g1.jsonl, server-g1.log,
# requests-g1.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
D=$S/docs/data/2026-11-17_r192_first_request_graphs/run
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
N="nice -n 19 python3 -I"
mkdir -p "$RES"
rm -f "$RES/DONE"
$N "$D/graph_passes.py" "$OUT/gp-g1.csv" --names d4096p1,d4096p2 --tsv "$RES/passes.tsv" > "$RES/passes.md"
$N "$D/server_graph_log.py" "$OUT/server-g1.log" --requests "$OUT/requests-g1.csv" --tsv "$RES/tasks.tsv" > "$RES/tasks.md"
$N "$AL" "$OUT/st-g1.jsonl" "$OUT/hp-g1.csv" > "$RES/align-g1.md"
t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/align-g1.md" | tail -n 1)
$N "$R121/first_request.py" --trace "$OUT/st-g1.jsonl" --requests "$OUT/requests-g1.csv" --profile "$OUT/hp-g1.csv" \
  --profile-t0 "$t0" --depth 4096 --tsv "$RES/steps-g1.tsv" --pairs-tsv "$RES/pairs-g1.tsv" > "$RES/first-g1.md"
touch "$RES/DONE"
