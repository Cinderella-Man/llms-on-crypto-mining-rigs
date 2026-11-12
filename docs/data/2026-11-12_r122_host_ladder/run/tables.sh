#!/usr/bin/env bash
# tables.sh: R122's tables (Q1400) from the rig's out/ folder (run/r122_loads.sh), on any CPU; nothing here needs a GPU.
#   REPO=<a checkout of this repository> nice -n 19 bash tables.sh OUT TABLES
# Per load: S37's step_split.py on its host profile (both requests) and the lowest MemAvailable, the swap and the paging
# from its vm.tsv (C761's vm_sampler.sh, every 0.5 s); the switches' closing lines and the graph reallocations from
# its server log. Then S37's arms.py
#   - over all fourteen loads with base and base-2 as the reference (the server's timings, the greedy replies against
#     base's, each load against base);
#   - per rung against the rung before it (a against base, b against a, ... f against e), the two loads of the rung
#     before as the reference;
#   - the stack: f against base;
# and ladder.py's summary of the rungs (ladder.md, ladder.tsv).
set -uo pipefail
OUT=${1:?OUT=<the out/ folder of the rig run>}
TB=${2:?TABLES=<the folder for the tables>}
: "${REPO:?REPO=<a checkout of this repository>}"
P=$REPO/docs/data/2026-11-10_served_step_split/run
R=$REPO/docs/data/2026-11-12_r122_host_ladder/run
RUNGS=${RUNGS:-base a b c d e f}
LOADS=""
for r in $RUNGS; do LOADS="$LOADS $r"; done
for r in $RUNGS; do LOADS="$LOADS $r-2"; done
mkdir -p "$TB/arms"
for l in $LOADS; do
    python3 -I "$P/step_split.py" "$OUT/$l/host_profile.csv" --load "$l" --requests greedy,long \
        --tsv "$TB/steps_$l.tsv" > "$TB/split_$l.md"
    mkdir -p "$TB/arms/$l" && cp "$TB/steps_$l.tsv" "$TB/arms/$l/steps.tsv" && cp "$OUT/$l"/reply_*.json "$TB/arms/$l/"
    awk -F'\t' -v l="$l" 'NR == 2 { s0 = $4 - $5; f0 = $8; i0 = $6; o0 = $7; m = $2 }
        NR > 1 { if ($2 < m) m = $2; if ($4 - $5 > s) s = $4 - $5; f = $8; i = $6; o = $7 }
        END { printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\n", l, m / 1024, s0 / 1024, s / 1024, f - f0, i - i0, o - o0 }' \
        "$OUT/$l/vm.tsv"
done | sed '1i load\tmin_mem_available_mib\tswap_start_mib\tswap_max_mib\tmajor_faults\tpswpin\tpswpout' > "$TB/vm.tsv"
# the switches' closing lines in each server log (C615's prefetch counts, C738's prepared passes) and the graph
# reallocations per request (the server's print_timing line)
for l in $LOADS; do
    grep -h -o -E 'C[0-9]{3,4}: .*' "$OUT/$l/server-$l.log" | sed "s/^/$l\t/"
done | sed '1i load\tline' > "$TB/switch_lines.tsv"
for l in $LOADS; do
    grep -h -E 'graph reallocs' "$OUT/$l/server-$l.log" \
        | sed -E "s/.*task +([0-9]+) .*= +([0-9]+) +\/ +([0-9]+).*/$l\t\1\t\2\t\3/"
done | sed '1i load\ttask\treallocs_same_size\treallocs_changed_size' > "$TB/reallocs.tsv"
csv() { local IFS=,; echo "$*"; }
# shellcheck disable=SC2086
python3 -I "$P/arms.py" "$TB/arms" --loads "$(csv $LOADS)" --ref base,base-2 --requests greedy,long \
    --tsv "$TB/arms_all.tsv" > "$TB/arms_all.md"
prev=""
for r in $RUNGS; do
    if [ -n "$prev" ]; then
        python3 -I "$P/arms.py" "$TB/arms" --loads "$prev,$prev-2,$r,$r-2" --ref "$prev,$prev-2" --requests greedy,long \
            --tsv "$TB/rung_$r.tsv" > "$TB/rung_$r.md"
    fi
    prev=$r
done
first=${RUNGS%% *}
last=${RUNGS##* }
python3 -I "$P/arms.py" "$TB/arms" --loads "$first,$first-2,$last,$last-2" --ref "$first,$first-2" \
    --requests greedy,long --tsv "$TB/stack.tsv" > "$TB/stack.md"
python3 -I "$R/ladder.py" "$TB" --rungs "$(csv $RUNGS)" --tsv "$TB/ladder.tsv" > "$TB/ladder.md"
