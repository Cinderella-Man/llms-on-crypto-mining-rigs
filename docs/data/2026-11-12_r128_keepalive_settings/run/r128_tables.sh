#!/usr/bin/env bash
# r128_tables.sh: R128's tables (Q1419) from the rig's out/ folder (run/r128_loads.sh) and R122's (the reference loads
# e, e-2, f and f-2 of docs/data/2026-11-12_r122_host_ladder/: same rig, build and preset), on any CPU; nothing here
# needs a GPU.
#   REPO=<a checkout of this repository> REF_OUT=<R122's out/> nice -n 19 bash r128_tables.sh OUT TABLES
# - all/: one link per load, R128's six and R122's four, so that every script reads one folder.
# - Per profiled load (all but fnoprof): S37's step_split.py on its host profile (both requests).
# - Per load: the lowest MemAvailable, the swap and the paging (vm.tsv, from C761's vm_sampler.sh), each card's
#   largest memory.used in the load (gpu_mem.tsv, from rig_server.sh's gpu.csv every 10 s), the server's timings of
#   each reply (timings.tsv, from reply_*.json), the switches' closing lines and the graph reallocations.
# - S37's arms.py: fka250 and feager against f and f-2 (arms_f.md), e771, e744 and e742 against e and e-2 (arms_e.md).
# - FN85's fn85.py (1,000 ms timeout) over every profiled load but fka250, and with --timeout 250 --min-wait 100 over
#   fka250 (ka250/): every wait for a keep-alive block; FN85's summary.py on each; card4_passes.py; timeline.py for the
#   first request of each R128 load.
# - The second request's prompt: the server's progress line (prompt_progress.tsv) and each main pass's GPU time per card
#   (run/prompt_passes.py, prompt_passes.md).
set -uo pipefail
OUT=${1:?OUT=<the out/ folder of the rig run>}
TB=${2:?TABLES=<the folder for the tables>}
: "${REPO:?REPO=<a checkout of this repository>}" "${REF_OUT:?REF_OUT=<R122 out/>}"
P=$REPO/docs/data/2026-11-10_served_step_split/run
F=$REPO/docs/data/2026-11-12_fn85_server_gap/run
NEW=${NEW:-fka250 feager e771 e744 e742 fnoprof}
REFS=${REFS:-e e-2 f f-2}
NOPROF=${NOPROF:-fnoprof}
csv() { local IFS=,; echo "$*"; }
A=$(dirname "$TB")/all
mkdir -p "$TB/arms" "$TB/timeline" "$TB/ka250" "$A"
for l in $NEW; do ln -sfn "$(readlink -f "$OUT/$l")" "$A/$l"; done
for l in $REFS; do ln -sfn "$(readlink -f "$REF_OUT/$l")" "$A/$l"; done
ALL="$REFS $NEW"
PROF=""
for l in $ALL; do case " $NOPROF " in *" $l "*) ;; *) PROF="$PROF $l" ;; esac; done
for l in $PROF; do
    python3 -I "$P/step_split.py" "$A/$l/host_profile.csv" --load "$l" --requests greedy,long \
        --tsv "$TB/steps_$l.tsv" > "$TB/split_$l.md"
    mkdir -p "$TB/arms/$l" && cp "$TB/steps_$l.tsv" "$TB/arms/$l/steps.tsv" && cp "$A/$l"/reply_*.json "$TB/arms/$l/"
done
for l in $ALL; do
    awk -F'\t' -v l="$l" 'NR == 2 { s0 = $4 - $5; f0 = $8; i0 = $6; o0 = $7; m = $2 }
        NR > 1 { if ($2 < m) m = $2; if ($4 - $5 > s) s = $4 - $5; f = $8; i = $6; o = $7 }
        END { printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\n", l, m / 1024, s0 / 1024, s / 1024, f - f0, i - i0, o - o0 }' \
        "$A/$l/vm.tsv"
done | sed '1i load\tmin_mem_available_mib\tswap_start_mib\tswap_max_mib\tmajor_faults\tpswpin\tpswpout' > "$TB/vm.tsv"
# gpu.csv: timestamp, index, temperature, power, SM clock, memory.used, utilization (every 10 s while the server runs)
for l in $ALL; do
    awk -F', ' -v l="$l" '{ c = $2 + 0; m = $6 + 0; if (m > mx[c]) mx[c] = m; n[c]++ }
        END { printf "%s", l; t = 0; for (c = 0; c < 9; c++) { printf "\t%d", mx[c]; t += mx[c] } printf "\t%d\t%d\n", t, n[0] }' \
        "$A/$l/gpu.csv"
done | sed '1i load\tcard0\tcard1\tcard2\tcard3\tcard4\tcard5\tcard6\tcard7\tcard8\ttotal\tsamples_card0' > "$TB/gpu_mem.tsv"
for l in $ALL; do
    for r in greedy long; do
        python3 -I -c '
import json, sys
t = json.load(open(sys.argv[3])).get("timings", {})
print("\t".join([sys.argv[1], sys.argv[2]] + [str(t.get(k, "")) for k in
    ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")]))' "$l" "$r" "$A/$l/reply_$r.json"
    done
done | sed '1i load\trequest\tprompt_n\tprompt_ms\tpredicted_n\tpredicted_ms\tdraft_n\tdraft_n_accepted' > "$TB/timings.tsv"
for l in $ALL; do
    grep -h -o -E 'C[0-9]{3,4}: .*' "$A/$l/server-$l.log" | sed "s/^/$l\t/"
done | sed '1i load\tline' > "$TB/switch_lines.tsv"
for l in $ALL; do
    grep -h -E 'graph reallocs' "$A/$l/server-$l.log" \
        | sed -E "s/.*task +([0-9]+) .*= +([0-9]+) +\/ +([0-9]+).*/$l\t\1\t\2\t\3/"
done | sed '1i load\ttask\treallocs_same_size\treallocs_changed_size' > "$TB/reallocs.tsv"
python3 -I "$P/arms.py" "$TB/arms" --loads f,f-2,fka250,feager --ref f,f-2 --requests greedy,long \
    --tsv "$TB/arms_f.tsv" > "$TB/arms_f.md"
python3 -I "$P/arms.py" "$TB/arms" --loads e,e-2,e771,e744,e742 --ref e,e-2 --requests greedy,long \
    --tsv "$TB/arms_e.tsv" > "$TB/arms_e.md"
D1=""
for l in $PROF; do [ "$l" = fka250 ] || D1="$D1 $l"; done
# shellcheck disable=SC2086
python3 -I "$F/fn85.py" "$A" --loads "$(csv $D1)" --step-split "$P" --tsv-dir "$TB" > "$TB/fn85.md"
python3 -I "$F/summary.py" "$TB" > "$TB/summary.md"
python3 -I "$F/fn85.py" "$A" --loads fka250 --step-split "$P" --timeout 250 --min-wait 100 --tsv-dir "$TB/ka250" \
    > "$TB/ka250/fn85.md"
python3 -I "$F/summary.py" "$TB/ka250" > "$TB/ka250/summary.md"
# the second request's prompt: the server's progress line (the first 756 tokens) and each main pass's GPU time per card
for l in $ALL; do
    sed 's/\x1b\[[0-9;]*m//g' "$A/$l/server-$l.log" | grep -E 'prompt processing, n_tokens' \
        | sed -E "s/.*n_tokens = +([0-9]+), progress = ([0-9.]+), t = +([0-9.]+) s.*/$l\t\1\t\2\t\3/"
done | sed '1i load\tn_tokens\tprogress\tt_s' > "$TB/prompt_progress.tsv"
# shellcheck disable=SC2086
python3 -I "$REPO/docs/data/2026-11-12_r128_keepalive_settings/run/prompt_passes.py" "$A" --loads "$(csv $PROF)" \
    --step-split "$P" > "$TB/prompt_passes.md"
# shellcheck disable=SC2086
python3 -I "$F/card4_passes.py" "$A" --loads "$(csv $PROF)" --step-split "$P" > "$TB/card4.md"
for l in $NEW; do
    case " $NOPROF " in *" $l "*) continue ;; esac
    python3 -I "$F/timeline.py" "$A" "$l" greedy --step-split "$P" > "$TB/timeline/${l}_greedy.md"
done
ls "$TB"
