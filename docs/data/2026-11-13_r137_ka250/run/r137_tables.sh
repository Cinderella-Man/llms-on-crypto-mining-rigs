#!/usr/bin/env bash
# r137_tables.sh: R137's tables (Q1429) from the rig's out/ folder (run/r137_loads.sh), on any CPU; nothing here needs a
# GPU. From R128's run/r128_tables.sh (docs/data/2026-11-12_r128_keepalive_settings/), with the references in the same
# run (f1 and f1-2: GGML_CUDA_KEEPALIVE_EAGER=1 at the default 1,000 ms timeout).
#   REPO=<a checkout of this repository> nice -n 19 bash r137_tables.sh OUT TABLES
# - Per profiled load: S37's step_split.py on its host profile (both requests).
# - Per load: the lowest MemAvailable, the swap and the paging (vm.tsv), each card's largest memory.used (gpu_mem.tsv),
#   the server's timings of each reply (timings.tsv), the switches' closing lines and the graph reallocations.
# - S37's arms.py: f1ka250 and f0ka250 against f1 and f1-2 (arms.md; the class in arms_class.tsv, the greedy steps
#   paired in arms_paired.tsv).
# - FN85's fn85.py over f1 and f1-2 at the 1,000 ms timeout (ka1000/), and with --timeout 250 --min-wait 100 over the
#   _MS=250 loads (ka250/): every wait for a keep-alive block, in the prompts and in generation; FN85's summary.py on
#   each; timeline.py for the first request of each profiled load.
# - The second request's prompt: the server's progress lines and each main pass's GPU time per card (R128's
#   prompt_passes.py).
set -uo pipefail
OUT=${1:?OUT=<the out/ folder of the rig run>}
TB=${2:?TABLES=<the folder for the tables>}
: "${REPO:?REPO=<a checkout of this repository>}"
P=$REPO/docs/data/2026-11-10_served_step_split/run
F=$REPO/docs/data/2026-11-12_fn85_server_gap/run
PP=$REPO/docs/data/2026-11-12_r128_keepalive_settings/run/prompt_passes.py
ALL=${ALL:-f1 f1-2 f1np f1ka250 f1ka250-2 f1ka250np f0ka250 f0ka250-2 f0ka250np}
K1000=${K1000:-f1 f1-2}
K250=${K250:-f1ka250 f1ka250-2 f0ka250 f0ka250-2}
csv() { local IFS=,; echo "$*"; }
mkdir -p "$TB/arms" "$TB/timeline" "$TB/ka1000" "$TB/ka250"
PROF=""
for l in $ALL; do case $l in *np) ;; *) PROF="$PROF $l" ;; esac; done
for l in $PROF; do
    python3 -I "$P/step_split.py" "$OUT/$l/host_profile.csv" --load "$l" --requests greedy,long \
        --tsv "$TB/steps_$l.tsv" > "$TB/split_$l.md"
    mkdir -p "$TB/arms/$l" && cp "$TB/steps_$l.tsv" "$TB/arms/$l/steps.tsv" && cp "$OUT/$l"/reply_*.json "$TB/arms/$l/"
done
for l in $ALL; do
    awk -F'\t' -v l="$l" 'NR == 2 { s0 = $4 - $5; f0 = $8; i0 = $6; o0 = $7; m = $2 }
        NR > 1 { if ($2 < m) m = $2; if ($4 - $5 > s) s = $4 - $5; f = $8; i = $6; o = $7 }
        END { printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\n", l, m / 1024, s0 / 1024, s / 1024, f - f0, i - i0, o - o0 }' \
        "$OUT/$l/vm.tsv"
done | sed '1i load\tmin_mem_available_mib\tswap_start_mib\tswap_max_mib\tmajor_faults\tpswpin\tpswpout' > "$TB/vm.tsv"
for l in $ALL; do
    awk -F', ' -v l="$l" '{ c = $2 + 0; m = $6 + 0; if (m > mx[c]) mx[c] = m; n[c]++ }
        END { printf "%s", l; t = 0; for (c = 0; c < 9; c++) { printf "\t%d", mx[c]; t += mx[c] } printf "\t%d\t%d\n", t, n[0] }' \
        "$OUT/$l/gpu.csv"
done | sed '1i load\tcard0\tcard1\tcard2\tcard3\tcard4\tcard5\tcard6\tcard7\tcard8\ttotal\tsamples_card0' > "$TB/gpu_mem.tsv"
for l in $ALL; do
    for r in greedy long; do
        python3 -I -c '
import json, sys
d = json.load(open(sys.argv[3]))
t = d.get("timings", {})
c = d.get("choices", [{}])[0].get("message", {})
import hashlib
h = hashlib.md5(((c.get("reasoning_content") or "") + "\x00" + (c.get("content") or "")).encode()).hexdigest()[:12]
print("\t".join([sys.argv[1], sys.argv[2]] + [str(t.get(k, "")) for k in
    ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")] + [h]))' "$l" "$r" "$OUT/$l/reply_$r.json"
    done
done | sed '1i load\trequest\tprompt_n\tprompt_ms\tpredicted_n\tpredicted_ms\tdraft_n\tdraft_n_accepted\treply_md5' > "$TB/timings.tsv"
for l in $ALL; do
    grep -h -o -E 'C[0-9]{3,4}: .*' "$OUT/$l/server-$l.log" | sed "s/^/$l\t/"
done | sed '1i load\tline' > "$TB/switch_lines.tsv"
for l in $ALL; do
    grep -h -E 'graph reallocs' "$OUT/$l/server-$l.log" \
        | sed -E "s/.*task +([0-9]+) .*= +([0-9]+) +\/ +([0-9]+).*/$l\t\1\t\2\t\3/"
done | sed '1i load\ttask\treallocs_same_size\treallocs_changed_size' > "$TB/reallocs.tsv"
python3 -I "$P/arms.py" "$TB/arms" --loads "$(csv $PROF)" --ref f1,f1-2 --requests greedy,long \
    --tsv "$TB/arms.tsv" > "$TB/arms.md"
# shellcheck disable=SC2086
python3 -I "$F/fn85.py" "$OUT" --loads "$(csv $K1000)" --step-split "$P" --tsv-dir "$TB/ka1000" > "$TB/ka1000/fn85.md"
python3 -I "$F/summary.py" "$TB/ka1000" > "$TB/ka1000/summary.md"
# shellcheck disable=SC2086
python3 -I "$F/fn85.py" "$OUT" --loads "$(csv $K250)" --step-split "$P" --timeout 250 --min-wait 100 \
    --tsv-dir "$TB/ka250" > "$TB/ka250/fn85.md"
python3 -I "$F/summary.py" "$TB/ka250" > "$TB/ka250/summary.md"
for l in $ALL; do
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/$l/server-$l.log" | grep -E 'prompt processing, n_tokens' \
        | sed -E "s/.*n_tokens = +([0-9]+), progress = ([0-9.]+), t = +([0-9.]+) s.*/$l\t\1\t\2\t\3/"
done | sed '1i load\tn_tokens\tprogress\tt_s' > "$TB/prompt_progress.tsv"
# shellcheck disable=SC2086
python3 -I "$PP" "$OUT" --loads "$(csv $PROF)" --step-split "$P" > "$TB/prompt_passes.md"
for l in $PROF; do
    python3 -I "$F/timeline.py" "$OUT" "$l" greedy --step-split "$P" > "$TB/timeline/${l}_greedy.md"
done
ls "$TB"
