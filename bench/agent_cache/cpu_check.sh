#!/usr/bin/env bash
# Exactness of the agent cache paths on a CPU (Q111), with the tiny random Flint (make_tiny_flint.py): a cold server
# against the server under test in two configurations, every request's top log-probabilities compared bit for bit
# (check_exact.py). Docker on the build machine, CPU only, low CPU share.
#
#   WORK=/path/with/build/model/sessions [SERVER_BIN=/ac/<other build>/llama-server] bench/agent_cache/cpu_check.sh
#
# WORK holds out-cpu/build/bin/llama-server, models/tiny-flint.gguf, sessions-e2-ud.jsonl and bench/agent_cache/;
# logs and the prefix files go to WORK/cpu-check/. CHECK_ARGS adds server flags to every server (-fa off: the CPU
# attention then sums every query's keys in the same order whatever the batch size). Exit status 1 if any check failed.
set -uo pipefail
WORK=${WORK:?}
OUT=$WORK/cpu-check
rm -rf "$OUT"; mkdir -p "$OUT/pfx"
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
BIN=${SERVER_BIN:-/ac/out-cpu/build/bin/llama-server}  # inside the containers: WORK is /ac
COMMON=(-m /ac/models/tiny-flint.gguf -c 16384 -b 2048 -ub 512 -np 1 -t 4 --host 127.0.0.1 --ctx-checkpoints 4
        --cache-ram 0 -lv 4 ${CHECK_ARGS:-})
TEST=pq-q111-test; COLD=pq-q111-cold
status=0

server() { # name port envfile args...
    local name=$1 port=$2; shift 2
    docker rm -f "$name" >/dev/null 2>&1
    docker run -d --name "$name" --cpu-shares=256 --label pascal-quants=1 --network host "${ENVS[@]}" \
        -v "$WORK:/ac" $IMG $BIN "${COMMON[@]}" --port "$port" "$@" >/dev/null
}
client() {
    docker run --rm --label pascal-quants=1 --network host -v "$WORK:/ac" -w /ac/bench/agent_cache python:3.12-slim \
        python3 check_exact.py --test http://127.0.0.1:18411 --cold http://127.0.0.1:18412 \
        --sessions /ac/sessions-e2-ud.jsonl "$@"
}
logs() { docker logs "$1" > "$OUT/$2.log" 2>&1; }
cleanup() { docker rm -f $TEST $COLD >/dev/null 2>&1; }
trap cleanup EXIT

ENVS=(); server $COLD 18412

# the generation-checkpoint token: one the gen scenario's greedy reply contains (the model is random)
ENVS=(); server $TEST 18411
client --scenarios pick | tee "$OUT/pick.txt"
GEN_AT=$(sed -n 's/^GEN_AT=\("[^"]*"\).*/\1/p' "$OUT/pick.txt" | head -1)
GEN_AT=${GEN_AT#\"}; GEN_AT=${GEN_AT%\"}
echo "gen-at: '$GEN_AT'"

# 1. the agent cache: prefix files, one tail checkpoint cut back on restore, generation checkpoints
NEW=(--prefix-cache /ac/cpu-check/pfx --checkpoint-gen --checkpoint-gen-at "$GEN_AT" --checkpoint-gen-min 100)
ENVS=(); server $TEST 18411 "${NEW[@]}"
client --scenarios prefix,turns,trim,gen --expect-restore --expect-trim | tee "$OUT/new.txt" || status=1
logs $TEST new-1
ENVS=(); server $TEST 18411 "${NEW[@]}"   # a new server process: the prefix comes from its file
client --scenarios prefix_again --expect-restore | tee "$OUT/new-restart.txt" || status=1
logs $TEST new-2

# 2. two tail checkpoints (--checkpoint-tail2) and a reserve per attached sampler, as stock llama.cpp
ENVS=(-e LLAMA_SAMPLER_RESERVE=1); server $TEST 18411 --checkpoint-tail2
client --scenarios turns,trim,gen | tee "$OUT/tail2.txt" || status=1
logs $TEST tail2

# 3. backend sampling (-bs): the same replies with and without a reserve per request; reserve counts from the logs
for mode in stock new; do
    if [ $mode = stock ]; then ENVS=(-e LLAMA_SAMPLER_RESERVE=1); else ENVS=(); fi
    server $TEST 18411 -bs
    client --scenarios bs | tee "$OUT/bs-$mode.txt" || status=1
    logs $TEST bs-$mode
    echo "bs-$mode: $(grep -c 'sched_reserve: reserving' "$OUT/bs-$mode.log") reserves"
done
if diff <(grep '^BS' "$OUT/bs-stock.txt" | sed 's/ cache_n=.*//') <(grep '^BS' "$OUT/bs-new.txt" | sed 's/ cache_n=.*//') >/dev/null; then
    echo "PASS bs: the same replies with one reserve"
else
    echo "FAIL bs: the replies differ"; status=1
fi
logs $COLD cold
grep -h "prefix cache\|restored context checkpoint\|created context checkpoint\|forcing full" "$OUT"/new-*.log | sed 's/^.*slot/slot/' | head -60 > "$OUT/cache-lines.txt"
echo "status $status"
exit $status
