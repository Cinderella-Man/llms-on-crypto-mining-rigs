#!/usr/bin/env bash
# R195's tables from r195.sh's two timed loads (e1: GGML_CUDA_KEEPALIVE_EAGER=1, k1: =3; depths 4,096, 6,144 and
# 8,192), with R192's and R121's scripts unchanged: split_gp.py cuts each load's profile per depth, graph_passes.py
# shows how every evaluation of the first request and its repeat ran, pass by pass (passes-<load>-d<depth>.md),
# server_graph_log.py the keep-alive's counts per request ("at new shapes", "block stops": tasks-<load>.md),
# first_request.py the first request against its repeat, step by step (first-<load>.md, with R86's align_t0.py), and
# summary.py the answer: per arm and depth, the first request against its repeat, its passes paired by what they read
# (summary.md; with several depths in one load the first request's 60-token prompt micro-batch is written to the profile
# and listed apart).
# CPU only, standard-library Python 3.
# Env: S (this repository's root), DEPTHS (default "4096 6144 8192"), OUT (the deep lane's results: gp-*.csv, hp-*.csv,
# st-*.jsonl, server-*.log, requests-*.csv, requests.csv), RES (the tables). Ends with RES/DONE.
set -uo pipefail
: "${S:?}" "${OUT:?}" "${RES:?}"
DEPTHS=${DEPTHS:-4096 6144 8192}
HERE=$S/docs/data/2026-11-17_r195_first_request_fix/run
R192=$S/docs/data/2026-11-17_r192_first_request_graphs/run
R121=$S/docs/data/2026-11-17_r121_flash_step_map/run
AL=$S/docs/data/2026-11-11_r86_step_trace/run/align_t0.py
N="nice -n 19 python3 -I"
mkdir -p "$RES"
rm -f "$RES/DONE"
depth_args=""
for d in $DEPTHS; do depth_args="$depth_args --depth $d"; done
for L in e1 k1; do
  $N "$HERE/split_gp.py" "$OUT/gp-$L.csv" --depths "$(echo $DEPTHS | tr ' ' ,)" --prefix "$RES/gp-$L" > "$RES/split-$L.txt" || exit 1
  for d in $DEPTHS; do
    $N "$R192/graph_passes.py" "$RES/gp-$L-d$d.csv" --names "d${d}p1,d${d}p2" --tsv "$RES/passes-$L-d$d.tsv" > "$RES/passes-$L-d$d.md"
  done
  $N "$R192/server_graph_log.py" "$OUT/server-$L.log" --requests "$OUT/requests-$L.csv" --tsv "$RES/tasks-$L.tsv" > "$RES/tasks-$L.md"
  $N "$AL" "$OUT/st-$L.jsonl" "$OUT/hp-$L.csv" > "$RES/align-$L.md"
  t0=$(sed -n 's/^t0 = \([0-9.]*\).*/\1/p' "$RES/align-$L.md" | tail -n 1)
  # shellcheck disable=SC2086
  $N "$R121/first_request.py" --trace "$OUT/st-$L.jsonl" --requests "$OUT/requests-$L.csv" --profile "$OUT/hp-$L.csv" \
    --profile-t0 "$t0" $depth_args --tsv "$RES/steps-$L.tsv" --pairs-tsv "$RES/pairs-$L.tsv" > "$RES/first-$L.md"
done
# shellcheck disable=SC2086
$N "$HERE/summary.py" --res "$RES" --requests "$OUT/requests.csv" --depths "$(echo $DEPTHS | tr ' ' ,)" \
  --tsv "$RES/summary.tsv" > "$RES/summary.md"
rm -f "$RES"/gp-e1-d*.csv "$RES"/gp-k1-d*.csv
touch "$RES/DONE"
