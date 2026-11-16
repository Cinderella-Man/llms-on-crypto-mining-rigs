#!/usr/bin/env bash
# FN93c (Q1487, 2026-11-16): does the micro-batch taper's other arithmetic cost Ornith quality? llama-perplexity's KL
# divergence on a P106 rig: the v4 preset's micro-batch (one call of CTX tokens read as one micro-batch) writes the
# base, the same text with the taper (LLAMA_UBATCH_TAPER=short=600:64: each call of CTX <= 600 tokens read in
# micro-batches of 64) is scored against it, and a repeat of the base's settings gives the run-to-run floor.
# Run from the repository root on a machine that only runs ssh, with nothing else on the rig.
#
# Env: RIG (ssh alias), WORK (a rig folder with room for the base: CHUNKS x (CTX/2 - 1) x n_vocab x 2 bytes, 3.04 GB
#   for Ornith's 248,320-token vocabulary at the defaults), BIN (llama-perplexity on the rig), MODEL (on the rig),
#   TEXT_LOCAL (here: the agent check's proxy.jsonl.bodies.gz, its proxy.jsonl beside it), OUT (here), STEP:
#     dry      print each arm's command, touch nothing
#     start    copy the text and scripts to WORK and start run/driver.sh detached (returns at once)
#     wait     return when the driver has written DONE (checks once a minute)
#     collect  copy the logs to OUT, write OUT/kld.tsv, remove WORK, record the rig's state
#   optional TAG (the proxy tags to read, default ^t1swift/), CTX (512), CHUNKS (24), TAPER (short=600:64),
#   ARMS (base taper rep)
# The line is release/ornith-v4/settings.env's: PRESET_ENV as it is, and of PRESET_ARGS the flags llama-perplexity
# takes (-ngl, -fa, -ub, -ts, -fit), with -b CTX so that each chunk is one call of one sequence: with -b 2048
# llama-perplexity reads four chunks in one call as four sequences, and the taper acts only on one-sequence calls.
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${STEP:?}"
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
. "$repo/release/ornith-v4/settings.env"
TAG=${TAG:-^t1swift/}; CTX=${CTX:-512}; CHUNKS=${CHUNKS:-24}; TAPER=${TAPER:-short=600:64}; ARMS=${ARMS:-base taper rep}
BIN=${BIN:-}; MODEL=${MODEL:-}

# the preset's flags that llama-perplexity takes; a flag's value is the next word unless that starts with '-'
ARGS=""
set -- $PRESET_ARGS
while [ $# -gt 0 ]; do
    f=$1; v=""
    if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then v=$2; shift; fi
    shift
    case $f in -ngl|-fa|-ub|-ts|-fit) ARGS="$ARGS $f $v" ;; esac
done
ARGS="${ARGS# } -b $CTX"

case $STEP in
dry)
    for arm in $ARMS; do
        case $arm in base) k="--kl-divergence-base $WORK/base.kld"; e="" ;;
                     taper) k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="LLAMA_UBATCH_TAPER=$TAPER" ;;
                     *) k="--kl-divergence-base $WORK/base.kld --kl-divergence"; e="" ;; esac
        echo "$arm: env $PRESET_ENV $e $BIN -m $MODEL $ARGS -f $WORK/text.txt -c $CTX --chunks $CHUNKS -lv 4 $k"
    done ;;
start)
    : "${BIN:?}" "${MODEL:?}" "${TEXT_LOCAL:?}"
    ssh "$RIG" "test ! -e $WORK && mkdir -p $WORK/text"
    ssh "$RIG" "cat > $WORK/text/proxy.jsonl.bodies.gz" < "$TEXT_LOCAL"
    ssh "$RIG" "cat > $WORK/text/proxy.jsonl" < "${TEXT_LOCAL%.bodies.gz}"
    ssh "$RIG" "cat > $WORK/agent_text.py" < "$here/agent_text.py"
    ssh "$RIG" "cat > $WORK/driver.sh" < "$here/driver.sh"
    ssh "$RIG" "cat > $WORK/ram_watchdog.sh" < "$repo/scripts/ram_watchdog.sh"
    for v in WORK BIN MODEL TAG CTX CHUNKS ARGS PRESET_ENV TAPER ARMS; do printf '%s=%q\n' "$v" "${!v}"; done \
        | ssh "$RIG" "cat > $WORK/run.env"
    timeout 60 ssh "$RIG" "setsid nohup bash $WORK/driver.sh > $WORK/driver.console 2>&1 < /dev/null & echo started" ;;
wait)
    until ssh -o ConnectTimeout=20 "$RIG" "test -e $WORK/DONE" 2> /dev/null; do sleep 60; done
    ssh "$RIG" "cat $WORK/driver.log" ;;
collect)
    : "${OUT:?}"
    mkdir -p "$OUT"
    ssh "$RIG" "cd $WORK && ls -la . text && md5sum text.txt" > "$OUT/work_listing.txt"
    for f in driver.log driver.console text.err run.env; do ssh "$RIG" "cat $WORK/$f" > "$OUT/$f" || true; done
    for arm in $ARMS; do
        for f in "$arm.out" "$arm.err" "rw-$arm.tsv"; do ssh "$RIG" "cat $WORK/$f" > "$OUT/$f" || true; done
    done
    {
        printf 'arm\tagainst\tchunks\tmean KLD\tmedian KLD\t99.0%% KLD\tmaximum KLD\tsame top p %%\tPPL(Q)\tPPL(base)\tPPL(Q)/PPL(base)\tln(PPL(Q)/PPL(base))\ttaper lines\n'
        for arm in $ARMS; do
            o="$OUT/$arm.out"; e="$OUT/$arm.err"
            g() { grep -a -h -m1 "$1" "$o" "$e" 2> /dev/null | head -n 1 | cut -d: -f2- | xargs; }
            n=$(grep -a -c 'ubatch taper:' "$e" 2> /dev/null || true)
            if [ "$arm" = base ]; then
                ppl=$(grep -a -h -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$o" "$e" | sed 's/.*PPL = //' | tail -n 1)
                ch=$(grep -a -h -o 'calculating perplexity over [0-9]* chunks' "$e" "$o" | grep -o '[0-9]*' | head -n 1)
                printf '%s\t-\t%s\t-\t-\t-\t-\t-\t%s\t-\t-\t-\t%s\n' "$arm" "$ch" "$ppl" "$n"
            else
                ch=$(grep -a -h -o 'computing over [0-9]* chunks' "$e" "$o" | grep -o '[0-9]*' | head -n 1)
                printf '%s\tbase\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$ch" "$(g '^Mean    KLD:')" \
                    "$(g '^Median  KLD:')" "$(g '^99.0%   KLD:')" "$(g '^Maximum KLD:')" "$(g '^Same top p:')" \
                    "$(g '^Mean PPL(Q) ')" "$(g '^Mean PPL(base) ')" "$(g '^Mean PPL(Q)/PPL(base) ')" \
                    "$(g '^Mean ln(PPL(Q)/PPL(base)) ')" "$n"
            fi
        done
    } > "$OUT/kld.tsv"
    grep -a -h 'ubatch taper:' "$OUT"/*.err 2> /dev/null | sed 's/^[0-9.]* //' | sort | uniq -c > "$OUT/taper_plans.txt" || true
    ssh "$RIG" "rm -rf $WORK"
    ssh "$RIG" "date -Is; ps -eo comm | grep -c -x -E 'llama-perplexity|llama-server' || true;
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; df -h $(dirname "$WORK") | tail -n 1" \
        > "$OUT/rig_end.txt"
    cat "$OUT/kld.tsv" "$OUT/taper_plans.txt" "$OUT/rig_end.txt" ;;
*) echo "STEP: dry, start, wait or collect" >&2; exit 2 ;;
esac
