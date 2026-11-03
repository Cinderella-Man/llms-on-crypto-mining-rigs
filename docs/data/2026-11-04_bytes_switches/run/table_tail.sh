#!/usr/bin/env bash
# The table of kld_tail.sh's logs: per arm and text, the mean KLD, the same-top-token rate and the routed experts
# kept (mean of the model's experts per token, and the share of the routed experts' reads that remains).
#   OUT=<kld_tail.sh's OUT> bash table_tail.sh > kld_tail.tsv
set -uo pipefail
: "${OUT:?}"
printf 'arm\ttext\tmean_kld\tsame_top_p_pct\texperts_kept_mean\trouted_reads_pct\n'
for lg in "$OUT"/*.log; do
    f=$(basename "$lg" .log)
    arm=${f%.*} text=${f##*.}
    kld=$(awk '/^Mean    KLD/ {print $3}' "$lg" | tail -n 1)
    top=$(awk '/^Same top p/ {print $4}' "$lg" | tail -n 1)
    kept=$(grep -o 'experts kept: mean [0-9.]* of [0-9]* ([0-9.]* %' "$lg" | tail -n 1)
    mean=$(echo "$kept" | awk '{print $4}')
    pct=$(echo "$kept" | awk '{gsub(/\(/, "", $7); print $7}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$text" "${kld:--}" "${top:--}" "${mean:--}" "${pct:--}"
done
