#!/usr/bin/env bash
# The agent cache (Q111) against a cold server on the rig: one llama-server with two slots, slot 0 with the agent cache
# (prefix files, ring checkpoints, generation checkpoints) and slot 1 cold (every request with cache_prompt off, so it
# processes the whole prompt as a server that has just started); check_exact.py compares every request's top
# log-probabilities bit for bit. Runs on the rig itself.
#
#   BIN=<dir with llama-server> MODEL=<flint.gguf> SESSIONS=<sessions.jsonl> OUT=<dir> [CTX=32768] [PORT=18481] bash rig_check.sh
#
# Expected, as on a V100: from-scratch prompts, the saved-prefix restore and a repeated request restored from its tail
# checkpoint are bit-identical (PASS). Paths that evaluate tokens in other batches than a cold server does differ in
# the last bits, as stock llama.cpp's own prompt cache does: reused generated tokens (turns/1-5, the live-window
# divergence) and a restore cut back into an earlier batch (diverge at 1500); for those the output gives the first
# token's largest log-probability difference and whether the top token is the same. Same server flags as rig_replay.sh
# with -np 2. Results in OUT: check.txt, server-check.log.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${SESSIONS:?}" "${OUT:?}"
CTX=${CTX:-32768}
PORT=${PORT:-18481}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
rm -rf "$OUT/prefix-check"; mkdir -p "$OUT/prefix-check"
LOG=$OUT/server-check.log

echo "=== $(date -Is) check: $BIN/llama-server" >> "$LOG"
env LLAMA_NO_MOE_DOWN_RESCALE=tg "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -b 2048 -ub 512 -np 2 \
    --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    --no-webui --host 127.0.0.1 --port "$PORT" --prefix-cache "$OUT/prefix-check" --checkpoint-gen >> "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
(while kill -0 $SRV 2>/dev/null; do
    a=$(awk '/^MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$a" -lt 400 ]; then echo "watchdog: MemAvailable $a MiB, killing llama-server" >> "$LOG"; kill -9 $SRV; break; fi
    sleep 0.5; done) &
t0=$(date +%s)
until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
    kill -0 $SRV 2>/dev/null || { echo "server exited while loading"; tail -20 "$LOG"; exit 1; }
    [ $(($(date +%s) - t0)) -gt 1200 ] && { echo "load timeout"; exit 1; }
    sleep 2
done
echo "server up after $(($(date +%s) - t0)) s"
U=http://127.0.0.1:$PORT
python3 -I "$HERE/check_exact.py" --test $U --cold $U --test-slot 0 --cold-slot 1 --sessions "$SESSIONS" \
    --scenarios prefix,turns,trim --expect-restore --expect-trim 2>&1 | tee "$OUT/check.txt"
echo "restores: $(grep -c 'prefix cache: restored' "$LOG") prefix, $(grep -c 'restored context checkpoint' "$LOG") checkpoint" \
    | tee -a "$OUT/check.txt"
echo RIG_CHECK_DONE
