#!/usr/bin/env bash
# Q928: the KL divergence of the N2P Flash-Next files, one file at a time with ../../2026-11-04_pascal_mix/run/score_all.sh
# (score_direct.sh: FN2's kld.sh against FN1's bases with the bases' split, each file read where it lies), and beside
# each file the cards' memory, sampled every 5 s. The bases' split is `-fit on`, and llama-perplexity's log does not
# print where the layers went, so the peak per card is the check that a file was scored whole on the cards: the peaks'
# sum must reach the size of the file's shards outside the n-gram table (which stays in host memory).
# Resumable as score_all.sh is: a scored file has DONE/<name> and is skipped, a scored text is not run again.
#   (score_all.sh's and score_direct.sh's environment: ENV QUALITY BASES OUT HOT LINKS TABLE_FAST SCORE DONE [SCORE_TAG])
#   SCORE_ALL=<score_all.sh> GPUS=<the cards by NVML index, the same as ENV's GPUS_ALL> LIST="<name> <first shard> [<build marker>]
#   ..." setsid nohup bash kld.sh >> <log> 2>&1 < /dev/null &
# Output: OUT/<name>.<text>.log (llama-perplexity's), OUT/<name>.mem (time, card, MiB in use), OUT/mem_peak.tsv (name,
# card, peak MiB), DONE/<name> per scored file, and DONE/finished at the end (the exit status and the time).
set -uo pipefail
: "${SCORE_ALL:?}" "${SCORE:?}" "${DONE:?}" "${LIST:?}" "${OUT:?}" "${GPUS:?}"
mkdir -p "$OUT" "$DONE"
rm -f "$DONE/finished"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
peaks() { # <name>: "card:MiB ..." and their sum
    awk -F', ' '{if ($3 + 0 > m[$2]) m[$2] = $3 + 0} END {for (i in m) print i, m[i]}' "$OUT/$1.mem" | sort -n |
        awk '{printf "%s:%d ", $1, $2; s += $2} END {printf "sum %d", s}'
}
rc=0
while read -r name first ready; do
    [ -z "$name" ] && continue
    if [ -f "$DONE/$name" ]; then log "$name: scored before"; continue; fi
    nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 5 >> "$OUT/$name.mem" 2> /dev/null &
    smi=$!
    LIST="$name $first ${ready:-}" bash "$SCORE_ALL" || rc=1
    kill $smi 2> /dev/null
    wait $smi 2> /dev/null
    log "$name: peak MiB per card: $(peaks "$name")"
done <<< "$LIST"
{
    printf 'name\tcard\tpeak_mib\n'
    for f in "$OUT"/*.mem; do
        [ -e "$f" ] || continue
        n=$(basename "$f" .mem)
        awk -F', ' '{if ($3 + 0 > m[$2]) m[$2] = $3 + 0} END {for (i in m) print i, m[i]}' "$f" | sort -n |
            awk -v n="$n" '{printf "%s\t%s\t%d\n", n, $1, $2}'
    done
} > "$OUT/mem_peak.tsv"
echo "$rc $(date -Is)" > "$DONE/finished"
log "finished, exit $rc"
exit $rc
