#!/usr/bin/env bash
# replay.sh: Q228's run. Replays coding-agent sessions through Flash-Next's n-gram table hash and simulates the caches,
# on the CPU at the lowest priority. Two sets: the agent-check runs (RUNS) and a sessions JSONL (SESSIONS); each set's
# hot-row sidecar is also learned on the other.
#   MODEL=<shard 1 of a Flash-Next GGUF> TABLE=<the shard holding per_layer_token_embd> \
#   RUNS="<agent-check run folder>..." SESSIONS=<sessions.jsonl> \
#   TOOL=<repo>/bench/ngram_table BIN=<folder with ple_tok and ple_sim, from build.sh> WORK=<scratch> OUT=<tables> \
#   bash replay.sh
set -euo pipefail
: "${MODEL:?}" "${TABLE:?}" "${RUNS:?}" "${SESSIONS:?}" "${TOOL:?}" "${BIN:?}" "${WORK:?}" "${OUT:?}"
N="nice -n 19"
R="$N python3 -I $TOOL/replay.py"
for set in agent sessions; do
    mkdir -p "$WORK/$set" "$OUT/$set"
    if [ $set = agent ]; then
        # shellcheck disable=SC2086
        $R prepare --model "$MODEL" --table "$TABLE" --out "$WORK/$set" $RUNS
    else
        $R prepare --model "$MODEL" --table "$TABLE" --out "$WORK/$set" --sessions "$SESSIONS"
    fi
    $N "$BIN/ple_tok" "$MODEL" "$WORK/$set/texts.bin" "$WORK/$set/tokens.bin" 6 248044 248046
    $R stream --out "$WORK/$set" --ckpts 4
    $R check --out "$WORK/$set" --n 20000
done
$N "$BIN/ple_sim" "$WORK/agent" "$OUT/agent" "$WORK/sessions"
$N "$BIN/ple_sim" "$WORK/sessions" "$OUT/sessions" "$WORK/agent"
for set in agent sessions; do
    $R report --out "$WORK/$set" --tables "$OUT/$set"
    cp "$WORK/$set/params.txt" "$WORK/$set/traces.tsv" "$WORK/$set/requests.tsv" "$WORK/$set/render.tsv" "$OUT/$set/"
done
