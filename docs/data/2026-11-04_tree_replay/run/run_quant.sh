#!/usr/bin/env bash
# Q688 / FN29: the DFlash2 drafter's candidates and the target's tokens on recorded agent sessions, dumped by
# llama-server (LLAMA_SPEC_DUMP, fork branch pascal-p106-v3-b12076-specdump), then the tree-value replay
# (bench/tree_replay). On a GPU machine with Docker: one llama-server with Flash-Next and the drafter drafting the
# whole trained block (7 tokens, so the lattice covers every position the drafter predicts), the recorded requests
# of agent-check runs re-sent to it in order (bench/agent_check/replay.py: the same messages, tools, sampling,
# lengths and per-request seeds), every verify step written to the dump. Acceptance only: timings on a shared
# machine are never speed results, and the replay's speed-ups come from a cost model.
#
# Environment (paths as seen inside the server container unless noted; mount them with MOUNTS):
#   BIN      dir with llama-server (the branch's rig tarball runs through its compute_61 PTX on newer cards)
#   MODEL    the target GGUF (the first shard)          DRAFT    the DFlash2 drafter GGUF
#   OUT      output dir (host path, mounted at the same path)
#   LOG      recorded proxy.jsonl of an agent-check run (host path; its proxy.jsonl.bodies.gz beside it)
#   GPUS     device list (default 2,3,4)                PORT     llama-server's port on 127.0.0.1 (default 18190)
#   ARMS     the arms of LOG whose traces are replayed, all their traces (default "dflash2b mtp2 nodraft")
#   TRACES   instead of ARMS: a comma-separated trace list
#   N_MAX    drafted tokens per step (default 7: the drafter's block of 8 minus the anchor)
#   CTX      context (default 65536)                   SARGS    more server arguments (default "-fit off")
#   REPO     this repository (host path)               MOUNTS   docker -v arguments for BIN, MODEL, DRAFT
#   IMAGE    default nvidia/cuda:12.9.2-devel-ubuntu24.04   LABEL  label on the container (default pascal-quants=1)
#   CACHE    the CUDA JIT cache (host path, default $OUT/nvcache): the PTX is compiled once
# Steps: refuse if a GPU in GPUS has a process; start the server; a smoke request (greedy, S18's coding prompt) whose
# dump must pass tree_replay.py --check with DFlash2 lattices of N_MAX positions; the replay; the server stopped; the
# tables (markov and factorized) in OUT/tables-*/ from every OUT/replay-*.jsonl. Each start dumps to its own
# OUT/dump-<start>.jsonl, split into smoke-<start>.jsonl and replay-<start>.jsonl. Resumable: replay.py skips traces
# already replayed under the same label in OUT/replay.jsonl (a partly replayed trace runs again, so its first requests
# appear twice in the tables). Done when OUT/DONE exists; OUT/FAILED on a failure.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${OUT:?}" "${LOG:?}" "${REPO:?}"
OUT=$(realpath -m "$OUT")
REPO=$(realpath "$REPO")
GPUS=${GPUS:-2,3,4}; PORT=${PORT:-18190}; N_MAX=${N_MAX:-7}; CTX=${CTX:-65536}; SARGS=${SARGS:--fit off}
ARMS=${ARMS:-dflash2b mtp2 nodraft}; TRACES=${TRACES:-}; MOUNTS=${MOUNTS:-}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}; LABEL=${LABEL:-pascal-quants=1}
CACHE=${CACHE:-$OUT/nvcache}
STAMP=$(date +%Y%m%d-%H%M%S)
DUMP=$OUT/dump-$STAMP.jsonl
NAME=specdump-srv-${GPUS//,/}
mkdir -p "$OUT" "$CACHE"
rm -f "$OUT/DONE" "$OUT/FAILED"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
fail() { log "FAILED: $*"; echo "$*" > "$OUT/FAILED"; exit 1; }

# the drafter's last card: the container sees GPUS as CUDA0..n-1
n_gpu=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DEVD=CUDA$((n_gpu - 1))

for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$used" = 0 ] || fail "GPU $g has $used processes"
done

if [ -z "$TRACES" ]; then
    TRACES=$(python3 -I "$REPO/bench/agent_check/replay.py" list --log "$LOG" \
        | awk -F'|' -v arms=" $ARMS " 'NR > 2 { t = $2; gsub(/ /, "", t); split(t, p, "/"); if (index(arms, " " p[1] " ")) printf "%s%s", (n++ ? "," : ""), t }')
fi
[ -n "$TRACES" ] || fail "no traces for arms $ARMS in $LOG"
log "traces: $(echo "$TRACES" | tr ',' '\n' | wc -l) ($ARMS)"

cleanup() {
    docker logs "$NAME" > "$OUT/server.log" 2>&1
    docker rm -f "$NAME" > /dev/null 2>&1
    log "server stopped"
}
trap cleanup EXIT
docker rm -f "$NAME" > /dev/null 2>&1
# shellcheck disable=SC2086
docker run -d --name "$NAME" --label "$LABEL" --label specdump=1 --network host --gpus "\"device=$GPUS\"" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_CACHE_PATH=/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 \
    -e LLAMA_SPEC_DUMP="$DUMP" -e LLAMA_SPEC_DUMP_K=8 -e LLAMA_SPEC_DUMP_TGT_K=8 -e LLAMA_SPEC_DUMP_NGRAM=16 \
    -v "$OUT:$OUT" -v "$CACHE:/nvcache" $MOUNTS "$IMAGE" "$BIN/llama-server" \
    -m "$MODEL" -ngl 99 -fa on -np 1 -c "$CTX" -b 2048 -ub 512 --ctx-checkpoints 4 --cache-ram 0 \
    --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui \
    -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$N_MAX" --spec-draft-n-ubatch 64 -devd "$DEVD" -ngld 99 \
    $SARGS --host 127.0.0.1 --port "$PORT" > /dev/null || fail "server did not start"
log "server $NAME on GPUs $GPUS (drafter on $DEVD), port $PORT; waiting for /health"
until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
    sleep 10
    [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2> /dev/null)" = running ] \
        || { docker logs "$NAME" 2>&1 | tail -30; fail "server died while loading"; }
done
log "loaded; $(docker logs "$NAME" 2>&1 | grep -m 1 'LLAMA_SPEC_DUMP')"

# smoke: one greedy request, then the dump must be consistent and carry full DFlash2 lattices
SMOKE=$REPO/docs/data/2026-11-03_flash_next_dflash2/run/smoke_greedy.json
curl -s --max-time 1800 -H 'Content-Type: application/json' --data @"$SMOKE" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > "$OUT/smoke.json"
python3 -I "$REPO/bench/tree_replay/tree_replay.py" --check "$DUMP" > "$OUT/smoke_check.txt" 2>&1 \
    || { cat "$OUT/smoke_check.txt"; fail "the smoke dump does not pass --check"; }
python3 -I - "$DUMP" "$N_MAX" > "$OUT/smoke_lattice.txt" <<'EOF' || fail "the smoke dump has no full lattices"
import json, sys
steps = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = int(sys.argv[2])
full = [s for s in steps if s.get("type") == "draft-dflash" and s.get("lattice") and len(s.get("cand", [])) == n and all(len(c) == 8 for c in s["cand"])]
print(f"{len(steps)} steps, {len(full)} with an 8-candidate lattice of {n} positions, "
      f"{sum(s['n_acc'] for s in steps)} of {sum(len(s['draft']) for s in steps)} drafted tokens accepted")
sys.exit(0 if steps and len(full) == len(steps) else 1)
EOF
log "smoke: $(head -n 1 "$OUT/smoke_check.txt"); $(cat "$OUT/smoke_lattice.txt")"
n_smoke=$(wc -l < "$DUMP")

log "replay"
python3 -I "$REPO/bench/agent_check/replay.py" run --log "$LOG" --traces "$TRACES" --server "http://127.0.0.1:$PORT" \
    --label specdump --out "$OUT/replay.jsonl" --erase-slot >> "$OUT/replay.out" 2>&1
rc=$?
log "replay ended rc=$rc: $(grep -c ' gen ' "$OUT/replay.out") requests, $(grep -c ERROR "$OUT/replay.out") errors"
cleanup
trap - EXIT

# the dump split into the smoke request and the replay
head -n "$n_smoke" "$DUMP" > "$OUT/smoke-$STAMP.jsonl"
tail -n +$((n_smoke + 1)) "$DUMP" > "$OUT/replay-$STAMP.jsonl"
log "$n_smoke smoke steps, $(wc -l < "$OUT/replay-$STAMP.jsonl") replay steps"
rm "$DUMP"
for mode in markov factorized; do
    python3 -I "$REPO/bench/tree_replay/tree_replay.py" "$OUT"/replay-*.jsonl --mode "$mode" --out "$OUT/tables-$mode" \
        > "$OUT/tables-$mode.console" 2>&1
    log "tables-$mode: rc $? ($(head -n 3 "$OUT/tables-$mode/summary.md" | tail -n 1 | cut -c 1-160))"
done
date -Is > "$OUT/DONE"
log "done"
