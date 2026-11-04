#!/usr/bin/env bash
# The table of kld_route_once.sh's logs (or cpu_checks.sh's real step, against the switch-off logits): per arm and
# text, the mean and median KLD, the same-top-token rate, the perplexity and its ratio to the base's, and the routed
# experts: the distinct experts per full block and MoE layer as read and under independent routing, and how many of a
# later token's own top-k lie in its block's first token's set.
#   OUT=<kld_route_once.sh's OUT> bash table_route_once.sh > kld_route_once.tsv
set -uo pipefail
: "${OUT:?}"
printf 'arm\ttext\tmean_kld\tmedian_kld\tsame_top_p_pct\tppl\tppl_ratio_to_base\tdistinct_read\tdistinct_independent\town_in_block_set\tfull_block_layers\n'
for lg in "$OUT"/*.log; do
    f=$(basename "$lg" .log)
    case $f in cuda_unit*) continue ;; esac
    arm=${f%.*} text=${f##*.}
    [ "$arm" = "$f" ] && text=-
    kld=$(awk '/^Mean    KLD/ {print $3}' "$lg" | tail -n 1)
    med=$(awk '/^Median  KLD/ {print $3}' "$lg" | tail -n 1)
    top=$(awk '/^Same top p/ {print $4}' "$lg" | tail -n 1)
    ppl=$(grep -o 'Final estimate: PPL = [0-9.]*' "$lg" | tail -n 1 | awk '{print $5}')
    [ -n "$ppl" ] || ppl=$(awk '/^Mean PPL\(Q\) +:/ {print $4}' "$lg" | tail -n 1)
    ratio=$(awk '/^Mean PPL\(Q\)\/PPL\(base\)/ {print $4}' "$lg" | tail -n 1)
    line=$(grep 'moe route once (k = ' "$lg" | tail -n 1)
    rd=$(echo "$line" | grep -o 'read mean [0-9.]*' | awk '{print $3}')
    ind=$(echo "$line" | grep -o 'independent routing mean [0-9.]*' | awk '{print $4}')
    own=$(echo "$line" | grep -o "first token's set: mean [0-9.]*" | awk '{print $5}')
    nb=$(echo "$line" | grep -o '[0-9]* full blocks' | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$text" "${kld:--}" "${med:--}" "${top:--}" "${ppl:--}" \
        "${ratio:--}" "${rd:--}" "${ind:--}" "${own:--}" "${nb:--}"
done
