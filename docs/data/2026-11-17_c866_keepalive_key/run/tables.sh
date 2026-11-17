#!/usr/bin/env bash
# C866's tables from c866.sh's load (depths 4,096, 6,144 and 8,192), with R192's and R121's scripts unchanged: how every
# graph evaluation ran, pass by pass (graph_passes.py: names d<depth>p1 for the first request, p2 for its repeat), the
# debug lines and the keep-alive's counts per request (server_graph_log.py: "at new shapes" and "block stops" per
# request), and the first request against its repeat, step by step (first_request.py, with R86's align_t0.py).
# CPU only, standard-library Python 3. graph_passes.py was written for one depth (R192): check its passes.md against
# requests-k1.csv's times, and if it groups the three depths' requests wrongly, run it per depth on gp-k1.csv cut at
# the requests' times.
# The check (pass): at 6,144 the first request has 0 block stops (tasks.md) and its kernel-by-kernel passes are within
# 200 ms of the repeat's in total (passes.md); at 4,096 and 8,192 stops only before the passes whose picks changed
# (the README's table); nowhere an evaluation about 1 s slower than the same pass in the repeat (passes.md, first-k1.md
# step 0); the same reply in each pair (requests.csv).
# Env: S (this repository's root), DEPTHS (default "4096 6144 8192"), OUT (the deep lane's results: gp-k1.csv,
# hp-k1.csv, st-k1.jsonl, server-k1.log, requests-k1.csv, requests.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
DEPTHS=${DEPTHS:-4096 6144 8192}
R192=$S/docs/data/2026-11-17_r192_first_request_graphs
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
N="nice -n 19 python3 -I"
mkdir -p "$RES"
rm -f "$RES/DONE"
names=""
depth_args=""
for d in $DEPTHS; do
  names="$names${names:+,}d${d}p1,d${d}p2"
  depth_args="$depth_args --depth $d"
done
$N "$R192/run/graph_passes.py" "$OUT/gp-k1.csv" --names "$names" --tsv "$RES/passes.tsv" > "$RES/passes.md"
$N "$R192/run/server_graph_log.py" "$OUT/server-k1.log" --requests "$OUT/requests-k1.csv" --tsv "$RES/tasks.tsv" > "$RES/tasks.md"
$N "$AL" "$OUT/st-k1.jsonl" "$OUT/hp-k1.csv" > "$RES/align-k1.md"
t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/align-k1.md" | tail -n 1)
# shellcheck disable=SC2086
$N "$R121/first_request.py" --trace "$OUT/st-k1.jsonl" --requests "$OUT/requests-k1.csv" --profile "$OUT/hp-k1.csv" \
  --profile-t0 "$t0" $depth_args --tsv "$RES/steps-k1.tsv" --pairs-tsv "$RES/pairs-k1.tsv" > "$RES/first-k1.md"
touch "$RES/DONE"
