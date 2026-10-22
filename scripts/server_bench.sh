#!/usr/bin/env bash
# Benchmarks llama-server end to end: model load time, GPU memory per card once loaded, then a fixed set of chat
# prompts at temperature 0 with per-request speed and speculative-decoding acceptance. The server listens on
# 127.0.0.1 only and is stopped at the end. Results go to /tmp/server_bench/<TAG>/.
#
#   MODEL=/mnt/models/x.gguf TAG=base SERVER_ARGS="-c 32768" bash scripts/server_bench.sh
#   MODEL=... TAG=mtp SERVER_ARGS="-c 32768 --spec-type draft-mtp -md /mnt/models/mtp.gguf" bash scripts/server_bench.sh
#   PROMPTS=none: only load the model and record memory (for context-size fit tests)
#
# Host-RAM safety: --cache-ram 0 and --ctx-checkpoints 0 are always passed (llama-server defaults to an 8 GiB prompt
# cache and 32 context checkpoints per slot in host RAM); later SERVER_ARGS override them.
# Env: GPUS (CUDA_VISIBLE_DEVICES, default all), MAX_TOKENS (default 256), PORT (default 8089), LLAMA_DIR or BIN.
set -u

BIN=${BIN:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin}
MODEL=${MODEL:?set MODEL}
TAG=${TAG:?set TAG}
SERVER_ARGS=${SERVER_ARGS:-"-c 32768"}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
MAX_TOKENS=${MAX_TOKENS:-256}
PORT=${PORT:-8089}
PROMPTS=${PROMPTS:-default}
OUT=/tmp/server_bench/$TAG
mkdir -p "$OUT"

t0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$GPUS "$BIN/llama-server" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
    -np 1 --no-webui --cache-ram 0 --ctx-checkpoints 0 $SERVER_ARGS > "$OUT/server.log" 2>&1 &
srv=$!

# Watchdog: with only 4 GB of host RAM, a configuration that pins too much host memory (e.g. a very large context)
# can push the whole machine into the out-of-memory killer. Kill the server first if available RAM runs low.
(
    while kill -0 $srv 2> /dev/null; do
        avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
            echo "watchdog: MemAvailable ${avail} MiB, killing llama-server" | tee -a "$OUT/server.log" >&2
            grep -E "MemAvailable|Mlocked|Unevictable|Shmem:" /proc/meminfo >> "$OUT/server.log"
            kill -9 $srv; break
        fi
        sleep 0.5
    done
) &

until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 $srv 2> /dev/null; then echo "server exited during load:"; tail -20 "$OUT/server.log"; exit 1; fi
    sleep 2
done
load=$(echo "$(date +%s.%N) - $t0" | bc)
printf "load time: %.0f s\n" "$load" | tee "$OUT/summary.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$OUT/memory.csv" \
    | awk -F', ' '{gsub(/ MiB/,"",$2); s+=$2; printf "  GPU%s %s MiB\n", $1, $2} END {printf "  total %d MiB\n", s}' | tee -a "$OUT/summary.txt"

if [ "$PROMPTS" != none ]; then
    python3 - "$PORT" "$MAX_TOKENS" "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, sys, urllib.request
port, max_tokens, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompts = {
    "code": "Write a Python function that merges overlapping intervals, with type hints and three unit tests.",
    "explain": "Explain to a physicist how the attention mechanism in a transformer works, in about 300 words.",
    "math": "A tank fills in 6 hours through pipe A and empties in 9 hours through pipe B. Both are open. How long until it is full? Show your reasoning.",
    "rewrite": "Rewrite this paragraph in plain English, keeping every fact: 'The P106-090 is a cut-down GP106 made for "
               "cryptocurrency mining. It has no display outputs and a narrow PCIe interface. The reference design has "
               "3 GB of memory, but this board carries 6 GB. Each card has five streaming multiprocessors and reads its "
               "memory at about 173 GB/s.'",
}
tot_n = tot_ms = 0
for name, text in prompts.items():
    body = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r.get("timings", {})
    json.dump(r, open(f"{out}/{name}.json", "w"))
    acc = f" | draft accepted {t['draft_n_accepted']}/{t['draft_n']}" if t.get("draft_n") else ""
    print(f"{name:8s} prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second', 0):6.1f} t/s | "
          f"generated {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):5.2f} t/s{acc}")
    tot_n += t.get("predicted_n", 0); tot_ms += t.get("predicted_ms", 0)
if tot_ms: print(f"overall generation: {tot_n} tokens at {1000 * tot_n / tot_ms:.2f} t/s")
EOF
fi

kill $srv; wait $srv 2> /dev/null
grep -E "model buffer size|KV buffer size|compute buffer size|graph nodes|graph splits|n_ctx|CUDA graphs" "$OUT/server.log" \
    | sed 's/^[0-9.: ]*[IWD] //' | sort | uniq -c | sort -rn | head -40 > "$OUT/buffers.txt"
echo "== done; buffer sizes in $OUT/buffers.txt"
