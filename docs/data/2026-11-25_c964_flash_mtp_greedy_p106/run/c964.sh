#!/usr/bin/env bash
# C964 (Q1694): Flash-Next's greedy replies with the MTP draft against plain decoding, on the P106 rig itself.
# Correctness only, nothing timed. One llama-server per arm, on the v5 preset's line (release/flash-next-v5/settings.env):
#   mtp    the preset as it is: -md <the MTP draft> and every --spec-* flag
#   plain  the same line without -md and without the --spec-* flags (plain decoding, one token a step)
# Each loaded arm answers run/session.py's requests (C951's three greedy requests, then the same three with n_probs 2),
# then the memory per card is read and the server is stopped.
# Run it on the rig, detached:  setsid -f nohup bash c964.sh > $WORK/driver.log 2>&1 < /dev/null
# Env:
#   WORK         a folder for the outputs
#   BIN          the build's bin/ folder (llama-server)
#   MODEL        the target's first shard (the other shards and the n-gram table's link beside it)
#   MODEL_DRAFT  the MTP draft
#   PRESET       release/flash-next-v5/settings.env
#   SESSION      run/session.py of this folder
#   TEXT         the deep prompt's text file (C951's: docs/switches.md at 6a9cf9d9)
#   ARMS ("mtp plain"), PORT (18964), DEPTH (8192), LOAD_TIMEOUT (1800 s)
# Output: WORK/<arm>/ with command.txt, server.log, server.rc, load.txt, quiet.txt, mem-load.csv, mem.csv, session.tsv,
#   session.out, deep-prompt.md5 and the replies; WORK/<arm>/done after each arm, WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${PRESET:?}" "${SESSION:?}" "${TEXT:?}"
ARMS=${ARMS:-mtp plain}
PORT=${PORT:-18964}
DEPTH=${DEPTH:-8192}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}

say() { echo "$(date +%H:%M:%S) $*"; }
mem() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits > "$1" 2>&1; }
mkdir -p "$WORK"
rm -f "$WORK/DONE"
md5sum "$BIN/llama-server" "$SESSION" "$TEXT" "$PRESET" > "$WORK/md5.txt" 2>&1
ls -l "$MODEL" "$MODEL_DRAFT" > "$WORK/files.txt" 2>&1

unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1090
. "$PRESET"
PLAIN_ARGS=$(sed -E 's/ --spec-[a-z-]+ [^ ]+//g' <<< " $PRESET_ARGS")
PLAIN_ARGS=${PLAIN_ARGS# }
[[ "$PLAIN_ARGS" != *"--spec-"* ]] || { say "a --spec- flag is left in the plain arm"; exit 1; }
say "start: arms '$ARMS', BIN=$BIN"

arm() {  # <arm> <draft args> <server args>
    local a=$1 md=$2 args=$3 o=$WORK/$1 up=0 t0 srv
    rm -rf "$o"; mkdir -p "$o"
    { date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli|llama-perplexity' || echo "no llama process"
      nvidia-smi --query-gpu=index,pstate,memory.used,clocks.mem --format=csv,noheader 2>&1
      grep -E 'MemAvailable' /proc/meminfo; } > "$o/quiet.txt"
    printf 'bin: %s\nenv: %s\nargs: -m %s %s %s --host 127.0.0.1 --port %s\n' "$BIN/llama-server" "$PRESET_ENV" \
        "$MODEL" "$md" "$args" "$PORT" > "$o/command.txt"
    say "$a: load"
    # shellcheck disable=SC2086
    ( env $PRESET_ENV "$BIN/llama-server" -m "$MODEL" $md $args --host 127.0.0.1 --port "$PORT" \
        > "$o/server.log" 2>&1 < /dev/null; echo "rc=$?" > "$o/server.rc" ) &
    srv=$!
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        kill -0 "$srv" 2> /dev/null || break
        sleep 5
    done
    echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
    if [ $up = 1 ]; then
        mem "$o/mem-load.csv"
        python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" "$TEXT" "$DEPTH" > "$o/session.out" 2>&1
        say "  $a: $(wc -l < "$o/session.tsv") requests, $(awk -F'\t' '$4 > 0' "$o/session.tsv" | wc -l) with tokens"
        mem "$o/mem.csv"
    fi
    pkill -TERM -x llama-server 2> /dev/null
    for _ in $(seq 60); do ps -eo comm | grep -qx llama-server || break; sleep 1; done
    pkill -KILL -x llama-server 2> /dev/null
    wait "$srv" 2> /dev/null
    say "$a: $(cat "$o/load.txt"), $(cat "$o/server.rc" 2> /dev/null)"
    date -Is > "$o/done"
}

for a in $ARMS; do
    case $a in
        mtp) arm mtp "-md $MODEL_DRAFT" "$PRESET_ARGS" ;;
        plain) arm plain "" "$PLAIN_ARGS" ;;
        *) say "unknown arm $a" ;;
    esac
done
say "done: llama processes left: $(ps -eo comm | grep -cx llama-server); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
date -Is > "$WORK/DONE"
