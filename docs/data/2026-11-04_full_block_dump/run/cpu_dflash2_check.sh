#!/usr/bin/env bash
# Q804: the full-block dump with the real DFlash2 drafter and the real target, on the CPU (no GPU): the one check that
# runs the second decode (llama-server's LLAMA_SPEC_DUMP_FULL_BLOCK, fork branch pascal-p106-v3-b12076-specdump2)
# through Inco AI's drafter, which no random-weight model can. Four servers on the CPU backend, the same greedy and
# sampled requests to each, drafting --spec-draft-n-max tokens (default 2, the rig's line):
#   plain  no dump                  dump   LLAMA_SPEC_DUMP
#   full   ... and FULL_BLOCK=1     full4  ... and FULL_BLOCK=4 (a second block of four tokens)
# All four must return the same tokens, and the dump's drafts, candidates and walks must be line for line the same with
# and without the second block (it is decoded after the draft is taken); the full dump must carry a second block of the
# trained block's width (8 tokens, 7 positions) and pass tree_replay.py --check, full4 one of 4 tokens (3 positions).
# A CPU run is slow (seconds per token on a 65 GB model) and is no speed measurement.
# Environment:
#   BIN    dir with a CPU build of llama-server (cpu_check.sh's $WORK/cpu/bin)   REPO   this repository (host path)
#   MODEL  the target GGUF (first shard)    DRAFT  the DFlash2 drafter GGUF      OUT    output dir (host path)
#   THREADS (default 8)  N_MAX (default 2)  PORT (default 18091)  MAXN (tokens per request, default 24)
#   MOUNTS docker -v arguments for MODEL and DRAFT's directories (read only)    IMAGE (default nvidia/cuda:12.9.2-devel-ubuntu24.04)
# Output: $OUT/<arm>.{server.log,dump.jsonl,R1.json,R2.json}, $OUT/console.txt, $OUT/summary.tsv; done when $OUT/DONE exists
set -uo pipefail
: "${BIN:?}" "${REPO:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}"
BIN=$(realpath "$BIN"); REPO=$(realpath "$REPO"); OUT=$(realpath -m "$OUT")
THREADS=${THREADS:-8}; N_MAX=${N_MAX:-2}; PORT=${PORT:-18091}; MAXN=${MAXN:-24}; MOUNTS=${MOUNTS:-}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT"/*.dump.jsonl "$OUT/summary.tsv"
log() { echo "$(date -Is) $*" | tee -a "$OUT/console.txt"; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$OUT/summary.tsv"; }
printf 'check\tverdict\tdetail\n' > "$OUT/summary.tsv"
TR="python3 -I $REPO/bench/tree_replay/tree_replay.py"

P1='def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
P2='The three laws of thermodynamics state that'

run_arm() { # <arm> [env...]
    local arm=$1; shift
    local name=fbcheck-$arm envs=() e
    for e in "$@"; do envs+=(-e "$e"); done
    docker rm -f "$name" > /dev/null 2>&1
    # shellcheck disable=SC2086
    docker run -d --name "$name" --label pascal-quants=1 --label fbcheck=1 --cpu-shares 256 --network host \
        -v "$BIN:$BIN:ro" -v "$OUT:$OUT" $MOUNTS "${envs[@]}" "$IMAGE" "$BIN/llama-server" \
        -m "$MODEL" -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$N_MAX" -ngl 0 -ngld 0 -t "$THREADS" \
        -c 4096 -np 1 -b 512 -ub 128 --ctx-checkpoints 4 --cache-ram 0 --no-webui --host 127.0.0.1 --port "$PORT" \
        > /dev/null || { log "$arm: docker run failed"; return 1; }
    log "$arm: started, waiting for /health"
    local t0=$SECONDS
    until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
        sleep 10
        [ "$(docker inspect -f '{{.State.Status}}' "$name" 2> /dev/null)" = running ] \
            || { docker logs "$name" 2>&1 | tail -20; log "$arm: server died while loading"; return 1; }
        [ $((SECONDS - t0)) -lt 3600 ] || { log "$arm: not up after an hour"; return 1; }
    done
    log "$arm: up after $((SECONDS - t0)) s"
    for r in R1 R2; do
        if [ $r = R1 ]; then body=$(jq -nc --arg p "$P1" --argjson n "$MAXN" '{prompt: $p, n_predict: $n, temperature: 0, return_tokens: true, cache_prompt: false}')
        else body=$(jq -nc --arg p "$P2" --argjson n "$MAXN" '{prompt: $p, n_predict: $n, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0, seed: 7, return_tokens: true, cache_prompt: false}'); fi
        curl -s --max-time 3600 -H 'Content-Type: application/json' --data "$body" "http://127.0.0.1:$PORT/completion" > "$OUT/$arm.$r.json"
        log "$arm $r: $(jq -c '{n: (.tokens | length), draft_n: .timings.draft_n, draft_n_accepted: .timings.draft_n_accepted}' "$OUT/$arm.$r.json")"
    done
    docker logs "$name" > "$OUT/$arm.server.log" 2>&1
    docker rm -f "$name" > /dev/null 2>&1
}

D=$OUT
run_arm plain || { verdict servers FAIL "plain did not run"; date -Is > "$OUT/DONE"; exit 1; }
run_arm dump LLAMA_SPEC_DUMP="$D/dump.dump.jsonl" LLAMA_SPEC_DUMP_K=8 LLAMA_SPEC_DUMP_TGT_K=8 LLAMA_SPEC_DUMP_NGRAM=16 || { verdict servers FAIL "dump did not run"; date -Is > "$OUT/DONE"; exit 1; }
run_arm full LLAMA_SPEC_DUMP="$D/full.dump.jsonl" LLAMA_SPEC_DUMP_K=8 LLAMA_SPEC_DUMP_TGT_K=8 LLAMA_SPEC_DUMP_NGRAM=16 LLAMA_SPEC_DUMP_FULL_BLOCK=1 || { verdict servers FAIL "full did not run"; date -Is > "$OUT/DONE"; exit 1; }
run_arm full4 LLAMA_SPEC_DUMP="$D/full4.dump.jsonl" LLAMA_SPEC_DUMP_K=8 LLAMA_SPEC_DUMP_TGT_K=8 LLAMA_SPEC_DUMP_NGRAM=16 LLAMA_SPEC_DUMP_FULL_BLOCK=4 || { verdict servers FAIL "full4 did not run"; date -Is > "$OUT/DONE"; exit 1; }

# the same tokens in every arm
same=0
for r in R1 R2; do
    if cmp -s <(jq -c .tokens "$OUT/plain.$r.json") <(jq -c .tokens "$OUT/dump.$r.json") &&
       cmp -s <(jq -c .tokens "$OUT/plain.$r.json") <(jq -c .tokens "$OUT/full.$r.json") &&
       cmp -s <(jq -c .tokens "$OUT/plain.$r.json") <(jq -c .tokens "$OUT/full4.$r.json"); then same=$((same + 1)); fi
done
verdict tokens "$([ $same = 2 ] && echo OK || echo FAIL)" "$same/2 requests return the same tokens in the four arms ($(jq -r '.tokens | length' "$OUT/plain.R1.json") and $(jq -r '.tokens | length' "$OUT/plain.R2.json") tokens)"

# the drafts and the draft's own lattice are the same with and without the second block
nd=$(wc -l < "$OUT/dump.dump.jsonl"); nf=$(wc -l < "$OUT/full.dump.jsonl"); n4=$(wc -l < "$OUT/full4.dump.jsonl")
if [ "$nd" = "$nf" ] && [ "$nd" = "$n4" ] && [ "$nd" -gt 0 ] &&
   cmp -s <(jq -c 'del(.run, .fb)' "$OUT/dump.dump.jsonl") <(jq -c 'del(.run, .fb)' "$OUT/full.dump.jsonl") &&
   cmp -s <(jq -c 'del(.run, .fb)' "$OUT/dump.dump.jsonl") <(jq -c 'del(.run, .fb)' "$OUT/full4.dump.jsonl"); then ok=OK; else ok=FAIL; fi
verdict drafts "$ok" "$nd / $nf / $n4 dump lines (dump / full / full4); every line equal but for the run and the second block: $ok"

# the second block
w8=$(jq -c 'select(.fb != null and .fb.w == 8 and (.fb.cand | length) == 7 and (.fb.pick | length) == 7 and all(.fb.cand[]; length >= 1))' "$OUT/full.dump.jsonl" | wc -l)
w4=$(jq -c 'select(.fb != null and .fb.w == 4 and (.fb.cand | length) == 3)' "$OUT/full4.dump.jsonl" | wc -l)
nofb=$(jq -c 'select(.fb != null)' "$OUT/dump.dump.jsonl" | wc -l)
verdict second_block "$([ "$w8" = "$nf" ] && [ "$w4" = "$n4" ] && [ "$nofb" = 0 ] && echo OK || echo FAIL)" \
    "$w8 of $nf lines of full carry a second block of 8 tokens, $w4 of $n4 of full4 one of 4, $nofb of dump's carry any"
$TR --check "$OUT/full.dump.jsonl" > "$OUT/full.check.txt" 2>&1; rc1=$?
$TR --check "$OUT/full4.dump.jsonl" > "$OUT/full4.check.txt" 2>&1; rc2=$?
verdict check "$([ $rc1 = 0 ] && [ $rc2 = 0 ] && echo OK || echo FAIL)" "full: $(head -n 1 "$OUT/full.check.txt"); full4: $(head -n 1 "$OUT/full4.check.txt")"
$TR "$OUT/full.dump.jsonl" --out "$OUT/tables-full" > /dev/null 2>&1; rc3=$?
verdict tables "$([ $rc3 = 0 ] && [ -s "$OUT/tables-full/block.csv" ] && echo OK || echo FAIL)" "tree_replay.py on the full dump: rc $rc3, $(wc -l < "$OUT/tables-full/block.csv" 2> /dev/null || echo 0) rows of block.csv"
date -Is > "$OUT/DONE"
log "done"
