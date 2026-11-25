#!/usr/bin/env bash
# C936 (Q1675): Qwen3.8-Flash-Next with its MTP head as the guesser of llama-server's pipelined decoding, on the rig,
# nothing timed. One server load per configuration, in this order:
#   preset   the v5 preset as it is (release/flash-next-v5/settings.env: the MTP draft, three tokens a step)
#   plain    the preset with the draft off (no -md, no --spec-* flags): plain one-token decoding
#   plain_probs  plain again, with the top two tokens asked for on every reply (run after the others, CONFIGS=plain_probs)
#   d1 d2 d3 the preset plus --pipe-depth N --pipe-guess mtp
#   walk     the deepest dN that loaded, plus --pipe-mtp-walk 1000
#   grammar  the deepest dN that loaded, with LLAMA_PIPE_GRAMMAR=1
# Per load: each card's MiB after the load and at the end, then client.py's requests (a 256-token reply, a 512-token
# reply and a JSON-schema reply, all at temperature 0 with prompt caching off; the reference loads also ask for the
# 256-token reply's top two tokens), and the server log, from which tables.py reads the guesses right and wrong.
# Run on the rig from a folder holding this script, client.py and settings.env:
#   setsid -f nohup env BIN=<build>/bin MODELS=<folder with the shards and the draft> (or MODEL=<first shard> DRAFT=<draft>) OUT=<folder> \
#       [CONFIGS='preset plain d1 d2 d3 walk grammar'] [PORT=8290] [STOP_AT=21:50] bash pipe_mtp.sh > driver.log 2>&1 < /dev/null
# Each configuration writes OUT/<config>/{server.log,requests.jsonl,client.out,mib_load.csv,mib_end.csv,status} and
# then OUT/<config>.done. No load starts after STOP_AT minus 10 minutes, and a load still running at STOP_AT is stopped.
set -uo pipefail
: "${BIN:?set BIN}" "${OUT:?set OUT}"
HERE=$(cd "$(dirname "$0")" && pwd)
CONFIGS=${CONFIGS:-preset plain d1 d2 d3 walk grammar}
PORT=${PORT:-8290}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-900}
STOP_AT=${STOP_AT:-21:50}
deadline=$(date -d "$STOP_AT" +%s)
. "${PRESET:-$HERE/settings.env}"
MODEL=${MODEL:-${MODELS:?set MODELS, or MODEL and DRAFT}/$PRESET_MODEL}
DRAFT=${DRAFT:-$MODELS/$PRESET_DRAFT}
mkdir -p "$OUT"
DEEPEST=0

mib() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits > "$1"
        grep MemAvailable /proc/meminfo >> "$1"; }

healthy() { python3 -I -c "import urllib.request,sys; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5).read() else 1)" 2> /dev/null; }

stop_server() { # <pid>
    kill "$1" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$1" 2> /dev/null || break; sleep 1; done
    kill -9 "$1" 2> /dev/null
    for _ in $(seq 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')" -lt 100 ] && break
        sleep 1
    done
}

echo "$(date -Is) start: configs $CONFIGS, stop at $STOP_AT"
for cfg in $CONFIGS; do
    W=$OUT/$cfg; mkdir -p "$W"
    if [ "$(date +%s)" -gt $((deadline - 600)) ]; then
        echo "skipped: past $STOP_AT minus 10 minutes" > "$W/status"; touch "$OUT/$cfg.done"; continue
    fi
    env_extra=""; args="$PRESET_ARGS"; draft="-md $DRAFT"; probs=""
    case $cfg in
        preset) probs=probs ;;
        plain|plain_probs) probs=probs; [ $cfg = plain_probs ] && probs=probsall; draft=""
                args=$(echo "$args" | sed -E 's/--spec-type [^ ]+//; s/--spec-draft-n-max [^ ]+//; s/--spec-draft-n-ubatch [^ ]+//;
                                             s/--spec-draft-sampling [^ ]+//; s/--spec-draft-p-min [^ ]+//; s/  +/ /g') ;;
        d[0-9]) args="$args --pipe-depth ${cfg#d} --pipe-guess mtp" ;;
        walk|grammar)
            if [ "$DEEPEST" = 0 ]; then echo "skipped: no depth loaded" > "$W/status"; touch "$OUT/$cfg.done"; continue; fi
            args="$args --pipe-depth $DEEPEST --pipe-guess mtp"
            [ "$cfg" = walk ] && args="$args --pipe-mtp-walk 1000"
            [ "$cfg" = grammar ] && env_extra="LLAMA_PIPE_GRAMMAR=1" ;;
        *) echo "unknown config" > "$W/status"; touch "$OUT/$cfg.done"; continue ;;
    esac
    cmd="env $PRESET_ENV $env_extra $BIN/llama-server -m $MODEL $draft $args --metrics --no-webui --host 127.0.0.1 --port $PORT"
    echo "$cmd" > "$W/command.txt"
    echo "$(date -Is) $cfg: loading"
    date -Is > "$W/t_launch"
    $cmd > "$W/server.log" 2>&1 &
    pid=$!
    status=""
    t_end=$(( $(date +%s) + LOAD_TIMEOUT ))
    while :; do
        if ! kill -0 $pid 2> /dev/null; then status="did not load: the server exited during the load"; break; fi
        if healthy; then break; fi
        if [ "$(date +%s)" -gt "$t_end" ]; then status="did not load within $LOAD_TIMEOUT s"; break; fi
        if [ "$(date +%s)" -gt "$deadline" ]; then status="stopped at $STOP_AT during the load"; break; fi
        sleep 5
    done
    if [ -z "$status" ]; then
        date -Is > "$W/t_loaded"
        mib "$W/mib_load.csv"
        case $cfg in d[0-9]) DEEPEST=${cfg#d} ;; esac
        left=$(( deadline - $(date +%s) ))
        timeout "$left" python3 -I "$HERE/client.py" "$PORT" "$W/requests.jsonl" $probs > "$W/client.out" 2>&1
        rc=$?
        mib "$W/mib_end.csv"
        if [ $rc = 124 ]; then status="loaded; requests stopped at $STOP_AT"
        elif ! kill -0 $pid 2> /dev/null; then status="loaded; the server exited during the requests"
        else status="loaded; requests done (client exit $rc)"; fi
    fi
    echo "$status" > "$W/status"
    stop_server $pid
    date -Is > "$W/t_stopped"
    echo "$(date -Is) $cfg: $status"
    touch "$OUT/$cfg.done"
done
echo "$DEEPEST" > "$OUT/deepest"
echo "$(date -Is) end"
touch "$OUT/all.done"
