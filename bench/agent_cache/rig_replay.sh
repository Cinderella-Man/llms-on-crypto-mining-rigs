#!/usr/bin/env bash
# The agent cache (Q111) on the rig: replays multi-turn agent sessions against llama-server in several configurations
# and reports time to first token and prompt tokens read per turn (replay_sessions.py). Runs on the rig itself, client
# and server on the loopback.
#
#   BIN=<dir with llama-server> MODEL=<flint.gguf> SESSIONS=<sessions.jsonl> OUT=<dir>
#   [ARMS='stock stock-tail1 agent stock-drop agent-drop'] [STOCK_BIN=<dir>] [CTX=65536] [PORT=18480] bash rig_replay.sh
#
# Arms, one server each, the same flags apart from these:
#   stock       as stock llama.cpp serves an agent: a scheduler reserve per request (LLAMA_SAMPLER_RESERVE=1), two
#               checkpoints at each prompt's end (--checkpoint-tail2), no prefix files, no generation checkpoints
#   agent       --prefix-cache (an empty folder when the arm starts), one ring checkpoint at each prompt's end,
#               --checkpoint-gen
#   stock-tail1 the exactness control: stock behaviour with the agent cache's single tail checkpoint, so its prompts
#               split into the same micro-batches as the agent arm's and the agent arm's replies must equal its replies
#               (the cache is bit-exact; GPUs round by batch size, so stock's two tail checkpoints change the replies),
#               except after a generation-checkpoint restore, which reuses generated cells that stock-tail1 evaluates
#               again as a prompt
#   *-drop      the same, the client sending earlier replies back without their reasoning (most OpenAI-compatible
#               clients do; the server then reads each reply's visible part again)
#   v2next      STOCK_BIN's llama-server with the stock arm's flags (the build the agent cache started from)
# Common flags: -ngl 99 -fa on -c $CTX -b 2048 -ub 512 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
# --reasoning-format deepseek, the makers' sampling (temperature 1.0, top-p 0.95, top-k 20; a fixed seed per request,
# so exact servers give the same replies), LLAMA_NO_MOE_DOWN_RESCALE=tg. A watchdog stops llama-server below 400 MiB of
# available memory (the host has 4 GB). Results in OUT: replay.jsonl (one row per request), summary.md,
# server-<arm>.log, meminfo.log ("<epoch> <MemAvailable kB>" every 2 s).
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${SESSIONS:?}" "${OUT:?}"
ARMS=${ARMS:-stock stock-tail1 agent stock-drop agent-drop}
CTX=${CTX:-65536}
PORT=${PORT:-18480}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
COMMON="-ngl 99 -fa on -c $CTX -b 2048 -ub 512 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
--reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --host 127.0.0.1 --port $PORT"

(while true; do echo "$(date +%s) $(awk '/^MemAvailable/ {print $2}' /proc/meminfo)"; sleep 2; done) >> "$OUT/meminfo.log" &
MEMLOG=$!
trap 'kill $MEMLOG 2>/dev/null; [ -n "${SRV:-}" ] && kill $SRV 2>/dev/null' EXIT

start_server() { # arm bin env args
    local arm=$1 bin=$2 env=$3 args=$4
    echo "=== $(date -Is) $arm: $bin/llama-server $env $COMMON $args" >> "$OUT/server-$arm.log"
    # shellcheck disable=SC2086
    env LLAMA_NO_MOE_DOWN_RESCALE=tg $env "$bin/llama-server" -m "$MODEL" $COMMON $args >> "$OUT/server-$arm.log" 2>&1 &
    SRV=$!
    (while kill -0 $SRV 2>/dev/null; do
        a=$(awk '/^MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
        if [ "$a" -lt 400 ]; then echo "watchdog: MemAvailable $a MiB, killing llama-server" >> "$OUT/server-$arm.log"; kill -9 $SRV; break; fi
        sleep 0.5; done) &
    local t0; t0=$(date +%s)
    until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 $SRV 2>/dev/null || { echo "$arm: server exited while loading"; tail -20 "$OUT/server-$arm.log"; return 1; }
        [ $(($(date +%s) - t0)) -gt 1200 ] && { echo "$arm: load timeout"; kill $SRV; return 1; }
        sleep 2
    done
    echo "$arm: server up after $(($(date +%s) - t0)) s"
}
stop_server() {
    kill $SRV 2>/dev/null
    for _ in $(seq 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
    kill -9 $SRV 2>/dev/null
    SRV=
}

for arm in $ARMS; do
    hist=keep; base=$arm
    case $arm in *-drop) hist=drop; base=${arm%-drop};; esac
    case $base in
        stock)  bin=$BIN;       env="LLAMA_SAMPLER_RESERVE=1"; args="--checkpoint-tail2";;
        agent)  bin=$BIN;       env=""; rm -rf "$OUT/prefix-$arm"; mkdir -p "$OUT/prefix-$arm"
                args="--prefix-cache $OUT/prefix-$arm --checkpoint-gen";;
        stock-tail1) bin=$BIN;  env="LLAMA_SAMPLER_RESERVE=1"; args="";;
        v2next) bin=${STOCK_BIN:?STOCK_BIN for arm v2next}; env=""; args="";;
        *) echo "unknown arm $arm"; continue;;
    esac
    echo "== $arm $(date +%T)"
    start_server "$arm" "$bin" "$env" "$args" || continue
    python3 -I "$HERE/replay_sessions.py" run "$SESSIONS" --server "http://127.0.0.1:$PORT" --label "$arm" \
        --history $hist --out "$OUT/replay.jsonl" 2>&1 | tee -a "$OUT/replay-$arm.txt" | tail -3
    stop_server
    echo "$arm: $(grep -c 'sched_reserve: reserving' "$OUT/server-$arm.log") scheduler reserves, \
$(grep -c 'restored context checkpoint' "$OUT/server-$arm.log") checkpoint restores, \
$(grep -c 'forcing full prompt re-processing' "$OUT/server-$arm.log") full re-reads, \
$(grep -c 'prefix cache: restored' "$OUT/server-$arm.log") prefix restores" | tee -a "$OUT/arms.txt"
    du -sh "$OUT/prefix-$arm" 2>/dev/null | tee -a "$OUT/arms.txt"
done
python3 -I "$HERE/replay_sessions.py" summary "$OUT/replay.jsonl" --reference stock | tee "$OUT/summary.md"
case " $ARMS " in *" stock-tail1 "*)
    python3 -I "$HERE/replay_sessions.py" summary "$OUT/replay.jsonl" --reference stock-tail1 | tee -a "$OUT/summary.md";; esac
case " $ARMS " in *" stock-drop "*)
    python3 -I "$HERE/replay_sessions.py" summary "$OUT/replay.jsonl" --reference stock-drop | tee -a "$OUT/summary.md";; esac
echo RIG_REPLAY_DONE
