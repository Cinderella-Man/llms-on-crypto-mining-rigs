#!/usr/bin/env bash
# Measures how long one decode step takes when the model checks n tokens at once (n = 1 is plain generation;
# n = k + 1 is a speculative-decoding step that verifies k drafted tokens). This is the "verify cost" v(n) that decides
# whether a drafter such as EAGLE-3 can pay off. Three methods, each a "part" (PARTS, default all three):
#
#   bench   llama-bench pp1..pp8 and tg64 in one model load: the time to process n tokens in a single step.
#           Fast and smooth, but each step also copies one row of logits to the host and samples nothing, so it is
#           a proxy for the real verify path.
#   synth   llama-server with backend sampling (-bs), one load per width n in WIDTHS: n-gram drafting
#           (ngram-simple, n-gram size 1, n-1 drafted tokens) plus synthetic acceptance of every draft
#           (--spec-synth-len n). Each step then verifies up to n tokens exactly the way a trained drafter's steps
#           would; an n-gram draft is shorter when the matching token is recent, so the script reports the average
#           number of tokens actually verified per step. The generated text is meaningless; only the timing counts.
#           n = 1 runs without drafting.
#   ngram   llama-server with real n-gram drafting (NGRAM_ARGS, default ngram-simple: match the last 4 tokens, draft
#           up to 4) on an edit-style prompt whose answer repeats the input: real acceptance with a drafter that
#           costs nothing.
#
#   ssh rig "MODEL=/mnt/models/x.gguf bash -s" < bench/eagle3_flint/verify_cost.sh
#   ssh rig "MODEL=... PARTS=synth WIDTHS='1 3 5' bash -s" < bench/eagle3_flint/verify_cost.sh
#
# Results go to /tmp/eagle3_flint/<TAG>/ (default TAG=verify_cost): results.tsv (one row per measurement, read by
# draft_cost.sh), summary.txt, server_*.log and bench.jsonl.
#
# Host-RAM safety for a 4 GB host: -ub 128, --cache-ram 0 and --ctx-checkpoints 0 are always passed, and a watchdog
# kills llama-server if available memory drops below MIN_AVAIL_MB (default 400).
# Env: BIN, MODEL, TAG, GPUS (CUDA_VISIBLE_DEVICES, default all nine), CTX (default 32768), WIDTHS (default
# "1 2 3 4 5 6"), BENCH_WIDTHS (default 1,2,3,4,5,6,7,8), BENCH_REPS (default 5), BENCH_DEPTH (default 0),
# N_PREDICT (tokens per request, default 256), REPS (requests per width, default 2), PORT (default 8089),
# NGRAM_ARGS, SERVER_ARGS (extra llama-server flags).
set -u

BIN=${BIN:-/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61/bin}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:-verify_cost}
GPUS=${GPUS-0,1,2,3,4,5,6,7,8}
PARTS=${PARTS:-"bench synth ngram"}
CTX=${CTX:-32768}
WIDTHS=${WIDTHS:-"1 2 3 4 5 6"}
BENCH_WIDTHS=${BENCH_WIDTHS:-1,2,3,4,5,6,7,8}
BENCH_REPS=${BENCH_REPS:-5}
BENCH_DEPTH=${BENCH_DEPTH:-0}
N_PREDICT=${N_PREDICT:-256}
REPS=${REPS:-2}
PORT=${PORT:-8089}
NGRAM_ARGS=${NGRAM_ARGS:-"--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 4"}
SERVER_ARGS=${SERVER_ARGS:-}
OUT=/tmp/eagle3_flint/$TAG
mkdir -p "$OUT"
: > "$OUT/results.tsv"
echo "model $MODEL, GPUs '$GPUS', context $CTX, $(date -Is)" | tee "$OUT/summary.txt"

srv=
stop_server() {
    [ -n "$srv" ] || return 0
    kill "$srv" 2> /dev/null; wait "$srv" 2> /dev/null; srv=
}
trap stop_server EXIT

# start_server <label> <extra llama-server args...>: starts the server, waits until it answers, prints load time
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
    echo "-- $label: loaded in $(( $(date +%s) - t0 )) s ($*)" | tee -a "$OUT/summary.txt"
}

# measure <label> <method> <width> <prompt: long|edit> <ignore_eos: 1|0>: one warm-up request, then REPS timed
# requests; appends one row to results.tsv: method, width, verified tokens per step, ms per step, tok/s, acceptance
measure() {
    python3 - "$PORT" "$N_PREDICT" "$REPS" "$OUT" "$@" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys, urllib.request
port, n_predict, reps, out, label, method, width, kind, ignore_eos = sys.argv[1:10]
n_predict, reps = int(n_predict), int(reps)

LONG = ("<|user|>Turn these notes into a well-structured technical README with headings, a parts list, a setup "
        "section, a troubleshooting section and a FAQ. Keep every number.\n\n"
        "Notes: The machine is a mining rig with nine NVIDIA P106-090 cards, each with 6 GB of GDDR5 memory on a "
        "192-bit bus, about 173 GB/s measured, five streaming multiprocessors, 640 CUDA cores and no display "
        "outputs. Every card sits on a PCIe 1.0 x1 link of about 190 MB/s and there is no peer-to-peer transfer "
        "between cards, so a model must be split by layers and only small activations cross the links. The host "
        "is a two-core Celeron 3865U without AVX and with 4 GB of RAM, which is why llama-server runs with a "
        "micro-batch of 128, no prompt cache in host memory and no context checkpoints. The power limit is 75 W "
        "per card and the core clock settles near 1700-1760 MHz while generating. CUDA 12.9 and driver 580 are "
        "the last versions that support these Pascal cards. CUDA graphs need a patch on Pascal and add 14 % for "
        "a mixture-of-experts model whose tokens need about 1,900 kernel launches. Simple 4-bit formats (Q4_0, "
        "Q4_1) and Q8_0 decode about 1.5 times faster than K-quants on these cards because five SMs cannot "
        "unpack the more complex blocks fast enough to keep memory busy. Models load from a USB 3 stick that "
        "reads at about 410 MB/s; the internal SSD is only 64 GB. Backend sampling picks the next token on the "
        "GPU and avoids copying 600 KB of scores per token over the x1 link. Two conversations can share one "
        "copy of a model with little loss per conversation. Troubleshooting: if nvidia-smi reports a driver and "
        "library mismatch after an update, reboot; if the server is killed, check available host memory; if a "
        "stick is missing after boot, mount it by hand.<|assistant|>")

EDIT = ("<|user|>Add type hints and a one-line docstring to every function in this module. Change nothing else "
        "and return the whole module.\n\n```python\n"
        "import json\nimport os\nimport time\n\n\n"
        "def load_config(path):\n    with open(path) as f:\n        return json.load(f)\n\n\n"
        "def save_config(path, config):\n    tmp = path + '.tmp'\n    with open(tmp, 'w') as f:\n"
        "        json.dump(config, f, indent=2)\n    os.replace(tmp, path)\n\n\n"
        "def merge(base, override):\n    result = dict(base)\n    for key, value in override.items():\n"
        "        if isinstance(value, dict) and isinstance(result.get(key), dict):\n"
        "            result[key] = merge(result[key], value)\n        else:\n            result[key] = value\n"
        "    return result\n\n\n"
        "def retry(fn, attempts, delay):\n    last = None\n    for i in range(attempts):\n        try:\n"
        "            return fn()\n        except Exception as e:\n            last = e\n"
        "            time.sleep(delay * (2 ** i))\n    raise last\n\n\n"
        "def chunks(items, size):\n    for i in range(0, len(items), size):\n        yield items[i:i + size]\n\n\n"
        "def flatten(nested):\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n"
        "            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out\n"
        "```<|assistant|><think></think>")

def request(n):
    body = json.dumps({"prompt": LONG if kind == "long" else EDIT, "n_predict": n, "temperature": 0,
                       "ignore_eos": ignore_eos == "1", "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))["timings"]

request(16)  # warm-up: CUDA graph capture, first-request allocations
n = ms = drafted = accepted = 0
for _ in range(reps):
    t = request(n_predict)
    n += t["predicted_n"] - 1          # the first token comes from the prompt pass, not from a decode step
    ms += t["predicted_ms"]
    drafted += t.get("draft_n", 0)
    accepted += t.get("draft_n_accepted", 0)
steps = max(1, n - accepted)           # every step emits its accepted drafts plus one token of its own
row = dict(method=method, width=int(width), verified=1 + drafted / steps, ms_step=ms / steps,
           tok_s=1000 * n / ms, accept=(accepted / drafted if drafted else 0.0), steps=steps)
with open(f"{out}/results.tsv", "a") as f:
    f.write("\t".join(str(row[k]) for k in ("method", "width", "verified", "ms_step", "tok_s", "accept", "steps")) + "\n")
print(f"{label:22s} steps {steps:4d} | verified per step {row['verified']:.2f} | {row['ms_step']:6.1f} ms per step | "
      f"{row['tok_s']:5.2f} tok/s | drafts accepted {accepted}/{drafted}")
EOF
}

gpu_memory() {
    command -v nvidia-smi > /dev/null || return 0
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' | sed 's/ MiB//g; s/^/   MiB per GPU: /'
    echo
}

for part in $PARTS; do
    case $part in
    bench)
        echo "== llama-bench: pp$BENCH_WIDTHS and tg64, depth $BENCH_DEPTH ($(date +%T))" | tee -a "$OUT/summary.txt"
        CUDA_VISIBLE_DEVICES=$GPUS "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -ub 128 -d "$BENCH_DEPTH" \
            -p "$BENCH_WIDTHS" -n 64 -r "$BENCH_REPS" -o jsonl > "$OUT/bench.jsonl" 2> "$OUT/bench.log" < /dev/null
        python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(f"{out}/bench.jsonl") if l.strip()]
pp = {r["n_prompt"]: r["avg_ns"] / 1e6 for r in rows if r["n_gen"] == 0}
tg = [r["avg_ns"] / 1e6 / r["n_gen"] for r in rows if r["n_prompt"] == 0 and r["n_gen"] > 0]
with open(f"{out}/results.tsv", "a") as f:
    if tg:
        f.write(f"bench_tg\t1\t1\t{tg[0]}\t{1000 / tg[0]}\t0\t0\n")
        print(f"tg64: {tg[0]:.1f} ms per token ({1000 / tg[0]:.2f} tok/s)")
    for n in sorted(pp):
        f.write(f"bench_pp\t{n}\t{n}\t{pp[n]}\t0\t0\t0\n")
        print(f"pp{n}: {pp[n]:7.1f} ms per step, {pp[n] / pp[min(pp)]:.2f}x pp{min(pp)}")
EOF
        ;;
    synth)
        for w in $WIDTHS; do
            if [ "$w" -eq 1 ]; then
                start_server plain || continue
                gpu_memory
                measure plain synth 1 long 1
            else
                start_server "synth_w$w" --spec-type ngram-simple --spec-ngram-simple-size-n 1 \
                    --spec-ngram-simple-size-m $(( w - 1 )) --spec-synth-len "$w" || continue
                measure "synth width $w" synth "$w" long 1
            fi
            stop_server
        done
        ;;
    ngram)
        start_server ngram $NGRAM_ARGS && measure "n-gram, edit prompt" ngram 0 edit 0
        stop_server
        ;;
    esac
done

# summary: ms per step against verified tokens per step, relative to plain generation
python3 - "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import sys
rows = [l.rstrip("\n").split("\t") for l in open(f"{sys.argv[1]}/results.tsv") if l.strip()]
syn = [(float(r[2]), float(r[3])) for r in rows if r[0] == "synth"]
plain = [ms for v, ms in syn if v == 1.0]
if syn:
    print("\nserver, backend sampling: verified tokens per step -> ms per step (ratio to plain generation)")
    for v, ms in sorted(syn):
        print(f"  {v:5.2f} -> {ms:6.1f} ms" + (f"  ({ms / plain[0]:.2f}x)" if plain else ""))
pp = {int(r[1]): float(r[3]) for r in rows if r[0] == "bench_pp"}
if pp:
    print("llama-bench: tokens per step -> ms (ratio to pp1)")
    for n in sorted(pp):
        print(f"  {n:5d} -> {pp[n]:6.1f} ms  ({pp[n] / pp[min(pp)]:.2f}x)")
EOF
echo "== done $(date +%T); results in $OUT"
