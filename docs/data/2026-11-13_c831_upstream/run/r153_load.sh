#!/usr/bin/env bash
# R153 part (3) (Q1463) on one rig: do the release files load and answer sanely on a new build (NEW) and on a reference
# build (REF)? Each model is loaded with its release line (the v3 presets in release/*-v3/settings.env, the file swapped
# for the v4 release file), every switch the line does not set at its default. Run from the repository root on a machine
# that only runs ssh. Per model, in this order:
#   load    one llama-server per build (NEW, then REF), untimed, under bench/agent_check/rig_server.sh (RAM watchdog,
#           monitors): memory per card (nvidia-smi) 20 s after it answers, the host's MemAvailable then, the load log;
#           then two greedy requests to the server (/completion, temperature 0, 256 tokens, no prompt cache): a coding
#           request and a prose paragraph. Flash-Next once more on NEW without --spec-draft-n-ubatch (label <new>-noub).
#   greedy  docs/data/2026-11-05_ladder5/run/rig_greedy.sh: llama-completion decodes the coding request greedily with
#           REF, then NEW, on the model's no-draft split, -c 4096, and compares the text byte for byte.
# Env: RIG (ssh alias), NEW, REF (build folders under /opt/llama-builds), FN (Flash-Next's first shard: Swift Q2_0, its
#   n-gram table as shard 3), DRAFT (Flash-Next's MTP draft), FLINT_Q5, FLINT_Q8, ORNITH (the release files), RIGDIR (a
#   folder on the rig for logs, on a drive with room), OUT (a local folder), PORT (8091), MODELS_RUN ("flashnext flintq8
#   flintq5 ornith"), STEPS ("load greedy"), LABEL_NEW (rc2), LABEL_REF (ref)
set -uo pipefail
: "${RIG:?}" "${NEW:?}" "${REF:?}" "${FN:?}" "${DRAFT:?}" "${FLINT_Q5:?}" "${FLINT_Q8:?}" "${ORNITH:?}" "${RIGDIR:?}" "${OUT:?}"
PORT=${PORT:-8091}
MODELS_RUN=${MODELS_RUN:-flashnext flintq8 flintq5 ornith}
STEPS=${STEPS:-load greedy}
LABEL_NEW=${LABEL_NEW:-rc2}; LABEL_REF=${LABEL_REF:-ref}
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$OUT"

# the release lines
# shellcheck disable=SC1091
. release/flash-next-v3/settings.env; FN_ENV=$PRESET_ENV; FN_ARGS="$PRESET_ARGS -md $DRAFT"
# shellcheck disable=SC1091
. release/flint-v3/settings.env; FL_ARGS=$PRESET_ARGS
# shellcheck disable=SC1091
. release/ornith-v3/settings.env; OR_ENV=$PRESET_ENV; OR_ARGS=$PRESET_ARGS
# Flint Pascal Q5 GGUF: the same line with layers by count (release/flint-v3/README.md, the 5-bit lane)
FL5_ARGS=${FL_ARGS/ -ts 6,6,5,5,6,5,5,6,6/}

model() {  # <name>: sets M_FILE M_ENV M_ARGS M_SPLIT
    case $1 in
        flashnext) M_FILE=$FN; M_ENV=$FN_ENV; M_ARGS=$FN_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio' ;;
        flintq8) M_FILE=$FLINT_Q8; M_ENV=''; M_ARGS=$FL_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,6,5,5,6,5,5,6,6' ;;
        flintq5) M_FILE=$FLINT_Q5; M_ENV=''; M_ARGS=$FL5_ARGS; M_SPLIT='-ngl 99 -fa on' ;;
        ornith) M_FILE=$ORNITH; M_ENV=$OR_ENV; M_ARGS=$OR_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,6,6,6,5,5,5,5,6 -fit off' ;;
        *) say "unknown model $1"; return 1 ;;
    esac
}

P_CODE='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
P_PROSE='Explain in one paragraph how a hash map handles collisions, then give one example in Python.'

load() {  # <model> <label> <build folder> <server arguments>
    local arm=$1-$2 d=$RIGDIR/load-$1-$2 o=$OUT/$1/load-$2 up=0 p body
    mkdir -p "$o"
    say "load $arm: $3"
    ssh "$RIG" "LOGDIR=$d MIN_AVAIL_MB=60 ACTION=start ARM=$arm BIN=/opt/llama-builds/$3/bin/llama-server MODEL=$M_FILE \
        PORT=$PORT SERVER_ENV='$M_ENV' SERVER_ARGS='$4 --metrics --no-webui' bash -s" \
        < bench/agent_check/rig_server.sh > "$o/console.txt" 2>&1 && up=1
    if [ $up = 1 ]; then
        sleep 20
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem.csv"
        ssh "$RIG" "grep -E '^(MemTotal|MemAvailable|CmaFree|SwapTotal|SwapFree):' /proc/meminfo" > "$o/meminfo.txt"
        for p in code prose; do
            if [ $p = code ]; then body=$P_CODE; else body=$P_PROSE; fi
            printf '{"prompt": "%s", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false}' "$body" |
                ssh "$RIG" "curl -s -m 900 -H 'Content-Type: application/json' -d @- http://127.0.0.1:$PORT/completion" \
                > "$o/reply-$p.json" 2> "$o/reply-$p.err"
            say "  $arm $p: $(wc -c < "$o/reply-$p.json") B of JSON"
        done
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem-after.csv"
    fi
    ssh "$RIG" "LOGDIR=$d ACTION=end PORT=$PORT bash -s" < bench/agent_check/rig_server.sh >> "$o/console.txt" 2>&1
    ssh "$RIG" "tar -C $d -cf - ." | tar -C "$o" -xf - && ssh "$RIG" "rm -rf $d"
    grep -aE 'buffer size|KV self size|compute buffer|graph|nodes|projected|failed|error|abort|assert|memory breakdown' \
        "$o/server-$arm.log" > "$o/buffers.txt"
    say "load $arm: $(grep -a 'up after\|exited\|timeout' "$o/console.txt" | head -n 1 | head -c 160)"
    [ $up = 1 ]
}

for m in $MODELS_RUN; do
    model "$m" || continue
    if has load; then
        load "$m" "$LABEL_NEW" "$NEW" "$M_ARGS"
        load "$m" "$LABEL_REF" "$REF" "$M_ARGS"
        if [ "$m" = flashnext ]; then
            load "$m" "$LABEL_NEW-noub" "$NEW" "${M_ARGS/ --spec-draft-n-ubatch 128/}"
        fi
    fi
    if has greedy; then
        say "greedy $m"
        RIG=$RIG MODEL=$M_FILE TAG=$m-code PROMPT=$P_CODE ARGS=$M_SPLIT DIR=$RIGDIR/greedy OUT=$OUT/$m/greedy \
            BUILDS="$REF $NEW" bash docs/data/2026-11-05_ladder5/run/rig_greedy.sh > "$OUT/$m/greedy.console" 2>&1
        say "$(tail -n 1 "$OUT/$m/greedy.console")"
    fi
done
ssh "$RIG" "rmdir $RIGDIR/greedy 2> /dev/null; ps -eo comm | grep -c '^llama-' ; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '" \
    > "$OUT/clean.txt"
say "done: llama processes and card MiB after: $(tr '\n' ' ' < "$OUT/clean.txt")"
