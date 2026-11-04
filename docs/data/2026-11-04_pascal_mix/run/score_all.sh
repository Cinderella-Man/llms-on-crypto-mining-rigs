#!/usr/bin/env bash
# The mixes' KL divergence, one file at a time with ../../2026-11-04_gguf_compose/run/score.sh (FN2's kld.sh against
# FN1's bases, the bases' split), resumable: a file whose three texts are scored gets DONE/<name> and is skipped on a
# rerun (score.sh also skips texts already scored). A line's third field is the file's build marker: the file waits
# for it, and is skipped if the same path with /done/ replaced by /failed/ appears. `touch $STOP` stops the run before
# the next file; DONE/all marks the end.
#   (score.sh's environment: ENV QUALITY BASES OUT HOT TABLE_FAST [SCORE_TAG]) SCORE=<score.sh> DONE=<dir> \
#   [STOP=$DONE/STOP] LIST="<name> <first shard> [<build marker>]
#   ..." setsid nohup bash score_all.sh >> <log> 2>&1 < /dev/null &
set -uo pipefail
: "${SCORE:?}" "${DONE:?}" "${LIST:?}" "${OUT:?}"
STOP=${STOP:-$DONE/STOP}
mkdir -p "$DONE"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
rc=0
while read -r name first ready; do
    [ -z "$name" ] && continue
    [ -f "$DONE/$name" ] && { log "$name: scored before"; continue; }
    if [ -e "$STOP" ]; then log "STOP: stopping before $name"; exit 0; fi
    if [ -n "${ready:-}" ]; then
        n=0
        until [ -e "$ready" ]; do
            if [ -e "${ready/\/done\//\/failed\/}" ]; then log "$name: its build failed, skipped"; rc=1; continue 2; fi
            [ $((n % 10)) -eq 0 ] && log "$name: waiting for its build ($ready)"
            n=$((n + 1)); sleep 60
        done
    fi
    FILES="$name=$first" bash "$SCORE" || rc=1
    if grep -q "^Mean    KLD" "$OUT/$name".{wiki,agent,code}.log 2> /dev/null &&
        [ "$(grep -l "^Mean    KLD" "$OUT/$name".{wiki,agent,code}.log | wc -l)" -eq 3 ]; then
        date -Is > "$DONE/$name"
        log "$name: done"
    else
        log "$name: not all three texts scored"; rc=1
    fi
done <<< "$LIST"
date -Is > "$DONE/all"
exit $rc
