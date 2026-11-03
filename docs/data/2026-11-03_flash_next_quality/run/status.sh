#!/usr/bin/env bash
# The chain's state on one screen: the chain log's last lines, the done and failed steps, the last line of each step
# still running, our containers, and every finished score (mean KLD and same top token per file and text).
#   WORK=<folder> bash status.sh
W=${WORK:?}
tail -n 3 "$W/logs/chain.log"
echo "done: $(ls "$W/done" 2> /dev/null | tr '\n' ' ')| failed: $(ls "$W/failed" 2> /dev/null | tr '\n' ' ')"
for f in "$W"/logs/*.log; do
    s=$(basename "$f" .log)
    [ "$s" = chain ] || [ -e "$W/done/$s" ] || [ -e "$W/failed/$s" ] || echo "$s: $(tail -n 1 "$f")"
done
docker ps --format '{{.Names}} {{.Status}}' | grep "pq-${TAG:-fn1c}-" || true
for f in "$W"/kld/*.log; do
    [ -f "$f" ] || continue
    k=$(grep -E "^Mean    KLD" "$f" | tr -s ' ' | cut -d' ' -f3-5)
    p=$(grep -E "^Same top p" "$f" | tr -s ' ' | cut -d' ' -f4-6)
    [ -n "$k" ] && echo "  $(basename "$f" .log): KLD $k, same top $p"
done
