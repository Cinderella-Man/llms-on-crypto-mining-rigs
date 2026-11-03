#!/usr/bin/env bash
# C22 / C28 (Q440, Q448, Q548): the CPU checks of the checkpoint branch, inside a container without a GPU.
#   1. random-weight models from test-llama-archs: qwen4exp (Flash-Next's architecture: Gated DeltaNet + attention with
#      the sparse-attention indexer), qwen35 (Gated DeltaNet + attention) and mamba (recurrent only)
#   2. test-llama-context on each, every case: the new store and store_p cases (numbered device stores against the
#      host path: identical logits after a restore, byte-identical host copies, also from another thread, two stores
#      apart, freed stores refused) and the existing state cases (host and device copies, rollback, replay)
#   3. server_check.sh on the qwen4exp model: llama-server's checkpoints on the devices and inside the prompt
# Usage: BIN=<dir with llama-server, test-llama-context, test-llama-archs> RUN=<this folder> OUT=<dir> [THREADS=8] \
#   bash checks.sh
# Output: OUT/models/, OUT/context-<model>.log, OUT/context.tsv (model, case, verdict), OUT/server/ (server_check.sh)
set -uo pipefail
BIN=${BIN:?}; RUN=${RUN:?}; OUT=${OUT:?}; THREADS=${THREADS:-8}
mkdir -p "$OUT/models"
fail=0

echo "=== random models $(date -Is)"
for a in qwen4exp qwen35 mamba; do
    "$BIN/test-llama-archs" -a "^$a\$" -s 1 -o "$OUT/models" > "$OUT/models/save-$a.log" 2>&1
    echo "$a: exit $?"
done
ls -la "$OUT/models"/*.gguf

echo "=== test-llama-context $(date -Is)"
printf 'model\tcase\tverdict\treason\n' > "$OUT/context.tsv"
for m in "$OUT"/models/*.gguf; do
    n=$(basename "$m" .gguf)
    "$BIN/test-llama-context" -m "$m" -t "$THREADS" > "$OUT/context-$n.log" 2>&1
    rc=$?
    echo "$n: exit $rc"
    [ $rc = 0 ] || fail=1
    # the runner prints one verdict line per case and run: "<case>  OK|SKIP|FAIL  [reason]" (the [rs] cases twice,
    # the second time against a cache pre-filled with a pattern)
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/context-$n.log" | \
        awk -v m="$n" '$2 ~ /^(OK|SKIP|FAIL)$/ { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; print m "\t" $1 "\t" $2 "\t" r }' \
        >> "$OUT/context.tsv"
    grep -q 'failed   : 0' "$OUT/context-$n.log" || fail=1
done

echo "=== server_check $(date -Is)"
q=$(ls "$OUT"/models/*qwen4exp*.gguf 2>/dev/null | head -1)
if [ -n "$q" ]; then
    BIN="$BIN" MODEL="$q" OUT="$OUT/server" THREADS="$THREADS" SARGS='--override-kv qwen4exp.context_length=int:16384' \
        bash "$RUN/server_check.sh" || fail=1
else
    echo "no qwen4exp model"; fail=1
fi

echo "=== checks ended $(date -Is) fail=$fail"
exit $fail
