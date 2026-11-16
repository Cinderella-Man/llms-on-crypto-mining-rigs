#!/usr/bin/env bash
# FN93c (Q1487): the rig side, started detached by fn93c.sh (STEP=start) in WORK, which holds run.env, agent_text.py,
# ram_watchdog.sh and text/ (the recorded requests and their proxy log). It writes the plain text, then runs
# llama-perplexity once per arm, each its own load under scripts/ram_watchdog.sh:
#   base   writes the next-token distributions of the first CHUNKS chunks of CTX tokens (--kl-divergence-base)
#   taper  the same text and line with LLAMA_UBATCH_TAPER=$TAPER, scored against them (--kl-divergence)
#   rep*   the base's settings again, scored against them: the run-to-run floor
# -lv 4 makes the library's info lines visible, among them the taper's plan of every call it splits. Writes DONE at
# the end, also when an arm fails.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
. ./run.env
log() { echo "$(date -Is) $*" >> driver.log; }
log "start: ARMS=$ARMS CTX=$CTX CHUNKS=$CHUNKS TAPER=$TAPER"
if ! python3 -I agent_text.py text/proxy.jsonl.bodies.gz "$TAG" > text.txt 2> text.err; then
    log "agent_text failed"; date -Is > DONE; exit 1
fi
log "$(cat text.err); md5 $(md5sum < text.txt | cut -c1-32)"
for arm in $ARMS; do
    case $arm in
        base)  k="--kl-divergence-base $WORK/base.kld"; e="" ;;
        taper) k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="LLAMA_UBATCH_TAPER=$TAPER" ;;
        rep*)  k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="" ;;
        *) log "unknown arm $arm"; continue ;;
    esac
    LOG=/dev/shm/fn93c_rw.tsv PROC=llama-perplexity MIN_AVAIL_MB=200 setsid nohup chrt -f 10 bash ram_watchdog.sh \
        > /dev/null 2>&1 < /dev/null &
    wd=$!
    log "$arm: env $PRESET_ENV $e $BIN -m $MODEL $ARGS -f text.txt -c $CTX --chunks $CHUNKS -lv 4 $k"
    # shellcheck disable=SC2086
    env $PRESET_ENV $e timeout 2400 "$BIN" -m "$MODEL" $ARGS -f text.txt -c "$CTX" --chunks "$CHUNKS" -lv 4 $k \
        > "$arm.out" 2> "$arm.err"
    rc=$?
    kill "$wd" 2> /dev/null; sleep 1
    mv /dev/shm/fn93c_rw.tsv "rw-$arm.tsv" 2> /dev/null
    log "$arm: rc=$rc"
    if [ "$arm" = base ]; then
        log "base.kld $(stat -c %s base.kld 2> /dev/null) bytes"
        [ "$rc" -eq 0 ] || { log "no base: stop"; break; }
    fi
done
log "end"
date -Is > DONE
