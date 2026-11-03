#!/usr/bin/env bash
# The tables of cpu_check.sh's random-model logs: one row per head-bound sampling arm, one per moe-tail arm.
#   OUT=<cpu_check.sh's OUT> bash table_random.sh   (writes $OUT/random/head_bound.tsv and $OUT/random/moe_tail.tsv)
set -uo pipefail
: "${OUT:?}"
M=$OUT/random
printf 'model\thead_rows\tsampling\ttokens_same\ttokens\tprobs_same\tfrom_shortlist_rows\tfrom_full_head\n' > "$M/head_bound.tsv"
printf 'model\tp\tw\trenorm\tlogits\tmax_abs_diff\tsame_top_pct\tpositions\texperts_kept_mean\tof\n' > "$M/moe_tail.tsv"
for m in qwen4exp-moe qwen4exp-moe-zero qwen4exp-moe-q5k qwen4exp-moe-q5k-zero; do
    case $m in *q5k*) h=q5_K ;; *) h=f32 ;; esac
    case $m in *zero) rows="zero from k_low" ;; *) rows="random" ;; esac
    grep -E '^head bound, ' "$M/$m.log" | sed -E 's/^head bound, (.*[^ ]) *: ([0-9]+) of ([0-9]+) tokens the same, probabilities (the same|DIFFER); ([0-9]+) from the shortlist.s rows, ([0-9]+) from the full head.s$/\1\t\2\t\3\t\4\t\5\t\6/' |
        awk -F'\t' -v h="$h" -v r="$rows" 'BEGIN{OFS="\t"} {print h, r, $1, $2, $3, ($4=="the same" ? "yes" : "NO"), $5, $6}' >> "$M/head_bound.tsv"
    [ "$rows" = random ] || continue
    grep -E '^moe tail p = ' "$M/$m.log" | sed -E 's/^moe tail p = ([0-9.]+), w = ([0-9.]+), renorm ([01]): (identical logits|logits differ), max \|diff\| ([0-9.e+-]+), same top token ([0-9.]+) % of ([0-9]+) positions; [0-9]+ token-layers, mean ([0-9.]+) of ([0-9]+) experts kept$/\1\t\2\t\3\t\4\t\5\t\6\t\7\t\8\t\9/' |
        awk -F'\t' -v h="$h" 'BEGIN{OFS="\t"} {print h, $0}' >> "$M/moe_tail.tsv"
done
cat "$M/head_bound.tsv" "$M/moe_tail.tsv"
