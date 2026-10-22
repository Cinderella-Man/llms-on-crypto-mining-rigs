#!/usr/bin/env bash
# Times the EAGLE-3 draft step on the rig with a random-weight head of the right shape (make_random_head.py), so the
# cost of a drafter is known before anyone trains one. For every head in HEADS and every draft length k in DRAFT_N,
# llama-server runs with --spec-type draft-eagle3 --spec-draft-n-max k and synthetic acceptance of every draft
# (--spec-synth-len k+1): each round then drafts k tokens and verifies k+1, exactly like a perfect trained head, and
# the round time does not depend on how many drafts are accepted. The text it generates is meaningless.
#
#   ssh rig "MODEL=/mnt/models/x.gguf HEADS='/mnt/models/eagle3/flint-eagle3-rand-v32k-Q8_0.gguf' bash -s" \
#       < bench/eagle3_flint/draft_cost.sh
#
# Per run it records the round time (ms per verify step), the time llama.cpp spends inside the drafter's draft()
# call (from the speculative statistics printed at log level 4), and, when verify_cost.sh has run first, the
# draft-side cost D(k) = round(k) - v(k+1) and the acceptance length each round needs to break even or to reach 1.4x.
#
# The head runs on one card (DRAFT_DEV, default the last card in GPUS, which also holds Flint's output head and its
# last layers). A head converted without an LM head borrows the target's output matrix and must run on the card that
# holds it. Host-RAM safety as in verify_cost.sh.
# Env: BIN, MODEL, HEADS (space-separated GGUF paths), TAG (default draft_cost), GPUS (default all nine), DRAFT_DEV,
# DRAFT_N (default "1 2 3 4"), CTX (default 32768), N_PREDICT (default 256), REPS (default 2), PORT (default 8089),
# VERIFY_TSV (default /tmp/eagle3_flint/verify_cost/results.tsv), SERVER_ARGS.
set -u

BIN=${BIN:-/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61/bin}
MODEL=${MODEL:?set MODEL}
HEADS=${HEADS:?set HEADS to one or more random-head GGUFs}
TAG=${TAG:-draft_cost}
GPUS=${GPUS-0,1,2,3,4,5,6,7,8}
if [ -n "$GPUS" ]; then n_gpu=$(echo "$GPUS" | tr ',' '\n' | wc -l); DRAFT_DEV=${DRAFT_DEV:-CUDA$(( n_gpu - 1 ))}
else DRAFT_DEV=${DRAFT_DEV:-none}; fi
DRAFT_N=${DRAFT_N:-"1 2 3 4"}
CTX=${CTX:-32768}
N_PREDICT=${N_PREDICT:-256}
REPS=${REPS:-2}
PORT=${PORT:-8089}
VERIFY_TSV=${VERIFY_TSV:-/tmp/eagle3_flint/verify_cost/results.tsv}
SERVER_ARGS=${SERVER_ARGS:-}
OUT=/tmp/eagle3_flint/$TAG
mkdir -p "$OUT"
: > "$OUT/results.tsv"
echo "model $MODEL, GPUs '$GPUS', draft device $DRAFT_DEV, context $CTX, $(date -Is)" | tee "$OUT/summary.txt"

srv=
stop_server() {
    [ -n "$srv" ] || return 0
    kill "$srv" 2> /dev/null; wait "$srv" 2> /dev/null; srv=
}
trap stop_server EXIT

start_server() {
    local label=$1; shift
    local log="$OUT/server_$label.log" t0
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPUS "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
        -np 1 --no-webui -c "$CTX" -b 512 -ub 128 -bs --cache-ram 0 --ctx-checkpoints 0 $SERVER_ARGS "$@" \
        > "$log" 2>&1 < /dev/null &
    srv=$!
    (   # watchdog: with 4 GB of host RAM a bad configuration can take the whole machine into the OOM killer
        while kill -0 "$srv" 2> /dev/null; do
            avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
            if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
                echo "watchdog: MemAvailable ${avail} MiB, killing llama-server" | tee -a "$log" >&2
                kill -9 "$srv"; break
            fi
            sleep 0.5
        done
    ) &
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
        if ! kill -0 "$srv" 2> /dev/null; then
            echo "server '$label' exited during load:"; tail -15 "$log"; srv=; return 1
        fi
        sleep 2
    done
    echo "-- $label: loaded in $(( $(date +%s) - t0 )) s" | tee -a "$OUT/summary.txt"
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' | sed 's/ MiB//g; s/^/   MiB per GPU: /'
        echo
    fi
}

# measure <label> <head> <k> <server log>: warm-up, REPS timed requests, appends one row to results.tsv
measure() {
    python3 - "$PORT" "$N_PREDICT" "$REPS" "$OUT" "$@" <<'EOF' | tee -a "$OUT/summary.txt"
import json, re, sys, time, urllib.request
port, n_predict, reps, out, label, head, k, log = sys.argv[1:9]
n_predict, reps = int(n_predict), int(reps)
PROMPT = ("<|user|>Explain, step by step and with examples, how speculative decoding with a small draft head "
          "speeds up text generation on GPUs whose memory bandwidth is the bottleneck, and when it does not "
          "help.<|assistant|>")

def request(n):
    body = json.dumps({"prompt": PROMPT, "n_predict": n, "temperature": 0, "ignore_eos": True,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))["timings"]

def draft_stats():
    # "statistics draft-eagle3: #calls(b,g,a) = B G A, ... dur(b,g,a) = Tb, Tg, Ta ms" (cumulative)
    calls = dur = None
    for line in open(log, errors="replace"):
        m = re.search(r"statistics\s+draft-eagle3: #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+).*dur\(b,g,a\) = "
                      r"([\d.]+), ([\d.]+), ([\d.]+) ms", line)
        if m:
            calls, dur = int(m.group(2)), float(m.group(5))
    return calls, dur

request(16)  # warm-up
time.sleep(1)  # the statistics line is written by the server's log thread
c0, d0 = draft_stats()
n = ms = drafted = accepted = 0
for _ in range(reps):
    t = request(n_predict)
    n += t["predicted_n"] - 1
    ms += t["predicted_ms"]
    drafted += t.get("draft_n", 0)
    accepted += t.get("draft_n_accepted", 0)
steps = max(1, n - accepted)
time.sleep(1)
c1, d1 = draft_stats()
t_draft = (d1 - d0) / (c1 - c0) if c1 and c0 is not None and c1 > c0 else float("nan")
row = [head, k, 1 + drafted / steps, ms / steps, 1000 * n / ms, t_draft]
with open(f"{out}/results.tsv", "a") as f:
    f.write("\t".join(str(x) for x in row) + "\n")
print(f"{label:34s} round {ms / steps:6.1f} ms | verified per round {1 + drafted / steps:.2f} | "
      f"draft() {t_draft:5.2f} ms per round | {1000 * n / ms:5.2f} tok/s at full acceptance")
EOF
}

for head in $HEADS; do
    name=$(basename "$head" .gguf)
    for k in $DRAFT_N; do
        start_server "${name}_k$k" -md "$head" --spec-type draft-eagle3 --spec-draft-n-max "$k" \
            --spec-draft-n-min 0 --spec-draft-p-min 0 -devd "$DRAFT_DEV" -ngld 99 \
            --spec-synth-len $(( k + 1 )) -lv 4 || continue
        measure "$name k=$k" "$name" "$k" "$OUT/server_${name}_k$k.log"
        stop_server
    done
done

# summary against verify_cost.sh's results (README.md, "Deciding"): D(k) = round(k) - v(k+1), and the speed-up
# S(k) = tau(k) * T1 / round(k) for three planning values of the acceptance length tau(k), with the README's verdict
python3 - "$OUT/results.tsv" "$VERIFY_TSV" <<'EOF' | tee -a "$OUT/summary.txt"
import os, sys
# tau(k): tokens per round (accepted drafts + 1) of published chain-drafting EAGLE-3 heads on MoE targets.
# mid = mean of the per-k medians over the task rows of RedHatAI/Qwen3-30B-A3B-speculator.eagle3 and
# RedHatAI/gpt-oss-20b-speculator.eagle3; optimistic = the Qwen3-30B-A3B median; conservative = 0.85 x mid
TAU_MID = {1: 1.69, 2: 2.14, 3: 2.43, 4: 2.61, 5: 2.73}
TAU_OPT = {1: 1.74, 2: 2.25, 3: 2.60, 4: 2.82, 5: 2.97}
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
v = {}
if os.path.exists(sys.argv[2]):
    for r in (l.rstrip("\n").split("\t") for l in open(sys.argv[2]) if l.strip()):
        if r[0] == "synth":
            v[float(r[2])] = float(r[3])
t1 = v.get(1.0)
def v_at(n):  # linear interpolation over the measured server widths, extrapolated from the last two points
    pts = sorted(v.items())
    if len(pts) < 2:
        return None
    for (a, ta), (b, tb) in zip(pts, pts[1:]):
        if a <= n <= b:
            return ta + (tb - ta) * (n - a) / (b - a)
    (a, ta), (b, tb) = pts[-2:] if n > pts[-1][0] else pts[:2]
    return ta + (tb - ta) * (n - a) / (b - a)
if not t1:
    print(f"\n(no plain-generation time in {sys.argv[2]}; run verify_cost.sh first for D(k) and the speed-up table)")
    sys.exit()
print(f"\nT1 (plain generation, server) = {t1:.1f} ms per token = {1000 / t1:.2f} tok/s")
print("head | k | round ms | draft() ms | v(k+1) ms | D(k) ms | break-even tau | S conservative / mid / optimistic")
best = {}
for head, k, verified, round_ms, _, t_draft in rows:
    k, verified, round_ms = int(k), float(verified), float(round_ms)
    vk = v_at(verified)
    d = f"{round_ms - vk:6.1f}" if vk else "   n/a"
    mid = TAU_MID.get(k, TAU_MID[5]) * t1 / round_ms
    opt = TAU_OPT.get(k, TAU_OPT[5]) * t1 / round_ms
    cons = 0.85 * mid
    print(f"{head} | {k} | {round_ms:6.1f} | {float(t_draft):5.2f} | {vk if vk else float('nan'):6.1f} | {d} | "
          f"{round_ms / t1:.2f} | "
          f"{cons:.2f}x / {mid:.2f}x / {opt:.2f}x")
    if head not in best or mid > best[head][1]:
        best[head] = (k, mid, cons)
for head, (k, mid, cons) in best.items():
    verdict = "GO" if mid >= 1.4 and cons >= 1.15 else "NO-GO" if mid < 1.2 else "MARGINAL"
    print(f"{head}: best k = {k}, S mid {mid:.2f}x, conservative {cons:.2f}x -> {verdict}")
EOF
echo "== done $(date +%T); results in $OUT"
