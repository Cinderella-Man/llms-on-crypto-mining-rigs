#!/usr/bin/env bash
# C154 / C155 (Q704, Q706): the CPU checks of the reference-counted checkpoints and the fit fixes, inside a container
# without a GPU.
#   1. random-weight models from test-llama-archs: qwen4exp (Flash-Next's architecture), qwen35, mamba
#   2. test-llama-context on each, every case (C22's store and store_p cases included)
#   3. C22's server_check.sh on the qwen4exp model (the device stores and the spread: the six arms must still match),
#      once as published and once with --ctx-checkpoints-share on every arm
#   4. server_mem.sh on the qwen4exp model: the switch off and on, with and without a prompt cache, with the stores
#      and the spread, and the same requests on a build without the change (OLD_BIN)
#   5. server_mem.sh on BIG_MODEL, if given (a real hybrid model, for real-size checkpoints), with the native build
#   6. fit_check.sh: the random qwen4exp with itself as a draft, and MAIN with DRAFT (DFlash) and MTP, if given
# Usage: BIN=<dir with the build's llama-server> NBIN=<dir with a native CPU build: llama-server, test-llama-context,
#   test-llama-archs> RUN=<this folder> CKRUN=<the 2026-11-04_ckpt_cards run folder> OUT=<dir> [OLD_BIN=<dir>]
#   [BIG_MODEL=<gguf>] [MAIN=<gguf> DRAFT=<gguf> MTP=<gguf>] [THREADS=8] [PARTS='context server mem big fit'] bash checks.sh
# Output: OUT/models/, OUT/context-<model>.log, OUT/context.tsv, OUT/server{,_share}/ (server_check.sh),
#   OUT/mem/ and OUT/mem_big/ (server_mem.sh), OUT/fit/ (fit_check.sh)
set -uo pipefail
BIN=${BIN:?}; NBIN=${NBIN:?}; RUN=${RUN:?}; CKRUN=${CKRUN:?}; OUT=${OUT:?}; THREADS=${THREADS:-8}
OLD_BIN=${OLD_BIN:-}; BIG_MODEL=${BIG_MODEL:-}; MAIN=${MAIN:-}; DRAFT=${DRAFT:-}; MTP=${MTP:-}
PARTS=${PARTS:-context server mem big fit}
part() { [[ " $PARTS " == *" $1 "* ]]; }
mkdir -p "$OUT/models"
fail=0

echo "=== random models $(date -Is)"
for a in qwen4exp qwen35 mamba; do
    "$NBIN/test-llama-archs" -a "^$a\$" -s 1 -o "$OUT/models" > "$OUT/models/save-$a.log" 2>&1
    echo "$a: exit $?"
done
ls -la "$OUT/models"/*.gguf

if part context; then
echo "=== test-llama-context $(date -Is)"
printf 'model\tcase\tverdict\treason\n' > "$OUT/context.tsv"
for m in "$OUT"/models/*.gguf; do
    n=$(basename "$m" .gguf)
    "$NBIN/test-llama-context" -m "$m" -t "$THREADS" > "$OUT/context-$n.log" 2>&1
    rc=$?
    echo "$n: exit $rc"
    [ $rc = 0 ] || fail=1
    sed 's/\x1b\[[0-9;]*m//g' "$OUT/context-$n.log" | \
        awk -v m="$n" '$2 ~ /^(OK|SKIP|FAIL)$/ { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; print m "\t" $1 "\t" $2 "\t" r }' \
        >> "$OUT/context.tsv"
    grep -q 'failed   : 0' "$OUT/context-$n.log" || fail=1
done
fi

q=$(ls "$OUT"/models/*qwen4exp*.gguf 2>/dev/null | head -1)
[ -n "$q" ] || { echo "no qwen4exp model"; exit 1; }
CTXKV='--override-kv qwen4exp.context_length=int:16384'

if part server; then
echo "=== server_check (C22's arms) $(date -Is)"
BIN="$BIN" MODEL="$q" OUT="$OUT/server" THREADS="$THREADS" SARGS="$CTXKV" bash "$CKRUN/server_check.sh" || fail=1
echo "=== server_check, every arm with --ctx-checkpoints-share $(date -Is)"
BIN="$BIN" MODEL="$q" OUT="$OUT/server_share" THREADS="$THREADS" SARGS="$CTXKV --ctx-checkpoints-share" \
    bash "$CKRUN/server_check.sh" || fail=1
fi

if part mem; then
echo "=== server_mem, random qwen4exp $(date -Is)"
BIN="$BIN" OLD_BIN="$OLD_BIN" MODEL="$q" OUT="$OUT/mem" THREADS="$THREADS" SARGS="$CTXKV" bash "$RUN/server_mem.sh" || fail=1
fi

if part big && [ -n "$BIG_MODEL" ]; then
    echo "=== server_mem, $(basename "$BIG_MODEL") $(date -Is)"
    BIN="$NBIN" MODEL="$BIG_MODEL" OUT="$OUT/mem_big" THREADS="$THREADS" SARGS="-lm none" \
        ARMS="${BIG_ARMS:-c0_copy c0_share cr_copy cr_share cr2_copy cr2_share}" bash "$RUN/server_mem.sh" || fail=1
fi

if part fit; then
echo "=== fit_check $(date -Is)"
BIN="$BIN" OLD_BIN="$OLD_BIN" RMODEL="$q" MAIN="$MAIN" DRAFT="$DRAFT" MTP="$MTP" OUT="$OUT/fit" THREADS="$THREADS" \
    bash "$RUN/fit_check.sh" || fail=1
fi

echo "=== checks ended $(date -Is) fail=$fail"
exit $fail
