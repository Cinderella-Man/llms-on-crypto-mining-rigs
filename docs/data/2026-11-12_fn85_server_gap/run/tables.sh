#!/usr/bin/env bash
# tables.sh: FN85's tables (Q1410) from R122's rig output (docs/data/2026-11-12_r122_host_ladder/run/r122_loads.sh),
# on any CPU; nothing here needs a GPU.
#   REPO=<a checkout of this repository> nice -n 19 bash tables.sh OUT TABLES
# OUT holds <load>/host_profile.csv and <load>/server-<load>.log for R122's fourteen loads.
set -euo pipefail
OUT=${1:?OUT=<R122 out/ folder>}
TB=${2:?TABLES=<the folder for the tables>}
: "${REPO:?REPO=<a checkout of this repository>}"
SS=$REPO/docs/data/2026-11-10_served_step_split/run
R=$REPO/docs/data/2026-11-12_fn85_server_gap/run
LOADS=${LOADS:-base,a,b,c,d,e,f,base-2,a-2,b-2,c-2,d-2,e-2,f-2}
mkdir -p "$TB/timeline"
# every request: the generation time outside the profiled steps, part by part; every wait for a keep-alive block
python3 -I "$R/fn85.py" "$OUT" --loads "$LOADS" --step-split "$SS" --tsv-dir "$TB" > "$TB/fn85.md"
# the draft's heavier evaluations on its embedding card (the second request's last step)
python3 -I "$R/card4_passes.py" "$OUT" --loads "$LOADS" --step-split "$SS" > "$TB/card4.md"
# the head and the tail of five requests, evaluation by evaluation
for x in "base greedy" "f greedy" "base long" "a long" "d-2 long"; do
    set -- $x
    python3 -I "$R/timeline.py" "$OUT" "$1" "$2" --step-split "$SS" > "$TB/timeline/${1}_$2.md"
done
# the two summary tables of the README
python3 -I "$R/summary.py" "$TB" > "$TB/summary.md"
