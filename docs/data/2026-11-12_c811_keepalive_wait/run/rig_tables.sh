#!/usr/bin/env bash
# rig_tables.sh: R131's tables (C811, Q1417) from the rig's out/ folder (run/rig_c811.sh), on any CPU; nothing here needs
# a GPU. From R122's run/tables.sh and FN85's run/tables.sh.
#   REPO=<a checkout of this repository> nice -n 19 bash rig_tables.sh OUT TABLES
# - Per load: S37's step_split.py on its host profile (both requests), the lowest MemAvailable from its vm.tsv, the
#   keep-alive counts after each reply (the server log's "keep-alive =" lines) and the graph reallocations per request.
# - S37's arms.py over the four loads with f0 and f0-2 (the switch at 0) as the reference: each load's step classed,
#   the server's timings, the replies against f0's.
# - FN85's fn85.py and card4_passes.py over the four loads: every wait for a keep-alive block, per request and part
#   (pass: none in generation with the switch at 1); timeline.py for both requests of f0 and f1.
# - The three test-cuda-keepalive gpu runs' lines.
set -uo pipefail
OUT=${1:?OUT=<the out/ folder of the rig run>}
TB=${2:?TABLES=<the folder for the tables>}
: "${REPO:?REPO=<a checkout of this repository>}"
P=$REPO/docs/data/2026-11-10_served_step_split/run
F=$REPO/docs/data/2026-11-12_fn85_server_gap/run
LOADS=${LOADS:-f0 f1 f0-2 f1-2}
REF=${REF:-f0,f0-2}   # the reference loads (the switch at 0)
TL=${TL:-f0 f1}       # the loads whose two requests get a timeline
csv() { local IFS=,; echo "$*"; }
mkdir -p "$TB/arms" "$TB/timeline"
for l in $LOADS; do
    python3 -I "$P/step_split.py" "$OUT/$l/host_profile.csv" --load "$l" --requests greedy,long \
        --tsv "$TB/steps_$l.tsv" > "$TB/split_$l.md"
    mkdir -p "$TB/arms/$l" && cp "$TB/steps_$l.tsv" "$TB/arms/$l/steps.tsv" && cp "$OUT/$l"/reply_*.json "$TB/arms/$l/"
    awk -F'\t' -v l="$l" 'NR == 2 { m = $2 } NR > 1 { if ($2 < m) m = $2 } END { printf "%s\t%d\n", l, m / 1024 }' "$OUT/$l/vm.tsv"
done | sed '1i load\tmin_mem_available_mib' > "$TB/vm.tsv"
for l in $LOADS; do
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/$l/server-$l.log" | grep -h -E 'keep-alive = ' | sed -E "s/.*keep-alive = /$l\t/"
done | sed '1i load\tcounts after each reply (since the server started)' > "$TB/keepalive_counts.tsv"
for l in $LOADS; do
    grep -h -E 'graph reallocs' "$OUT/$l/server-$l.log" \
        | sed -E "s/.*task +([0-9]+) .*= +([0-9]+) +\/ +([0-9]+).*/$l\t\1\t\2\t\3/"
done | sed '1i load\ttask\treallocs_same_size\treallocs_changed_size' > "$TB/reallocs.tsv"
# shellcheck disable=SC2086
python3 -I "$P/arms.py" "$TB/arms" --loads "$(csv $LOADS)" --ref "$REF" --requests greedy,long \
    --tsv "$TB/arms.tsv" > "$TB/arms.md"
# shellcheck disable=SC2086
python3 -I "$F/fn85.py" "$OUT" --loads "$(csv $LOADS)" --step-split "$P" --tsv-dir "$TB" > "$TB/fn85.md"
# shellcheck disable=SC2086
python3 -I "$F/card4_passes.py" "$OUT" --loads "$(csv $LOADS)" --step-split "$P" > "$TB/card4.md"
for l in $TL; do
    for r in greedy long; do
        python3 -I "$F/timeline.py" "$OUT" "$l" "$r" --step-split "$P" > "$TB/timeline/${l}_$r.md"
    done
done
grep -h -E '^gpu: |^test-cuda-keepalive gpu' "$OUT"/ka-gpu-*.log > "$TB/ka_gpu.txt" 2> /dev/null
grep -h 'test-cuda-keepalive gpu' "$OUT/status.txt" >> "$TB/ka_gpu.txt" 2> /dev/null
ls "$TB"
