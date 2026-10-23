#!/usr/bin/env bash
# Benchmarks llama-server end to end: model load time, GPU memory per card once loaded, then a fixed set of chat
# prompts at temperature 0 with per-request speed and speculative-decoding acceptance. The server listens on
# 127.0.0.1 only and is stopped at the end. Results go to /tmp/server_bench/<TAG>/.
#
#   MODEL=/mnt/models/x.gguf TAG=base SERVER_ARGS="-c 32768" bash scripts/server_bench.sh
#   MODEL=... TAG=mtp SERVER_ARGS="-c 32768 --spec-type draft-mtp -md /mnt/models/mtp.gguf" bash scripts/server_bench.sh
#   PROMPTS=none: only load the model and record memory (for context-size fit tests)
#   PROMPTS=chat: instead of the four one-shot prompts, one conversation of five user messages that build on each other,
#     sent the way a chat client sends them: every request carries the whole history, each answer with its reasoning
#     as `reasoning_content` (KEEP_REASONING=0 resends the visible answers only), and the server reuses what it has
#     cached. Sampling: the model's GGUF defaults (general.sampling.*) or TEMP/TOP_P/TOP_K/MIN_P, fixed seeds; min-p is
#     0 unless the GGUF or MIN_P sets it (llama-server's own default is 0.05; Flint's makers use 0). Per turn it reports
#     the new prompt tokens and the time to
#     the first token, the generated tokens (reasoning + answer) and their speed; transcript in chat.json. Models with
#     sliding-window attention can reuse the cache only through context checkpoints: pass --ctx-checkpoints N (a few)
#     in SERVER_ARGS, or every turn re-processes the whole history. Use MAX_TOKENS of 2048 or more for reasoning models.
#
# Host-RAM safety: --cache-ram 0 and --ctx-checkpoints 0 are always passed (llama-server defaults to an 8 GiB prompt
# cache and 32 context checkpoints per slot in host RAM); later SERVER_ARGS override them.
# Env: GPUS (CUDA_VISIBLE_DEVICES, default all), MAX_TOKENS (default 256), PORT (default 8089), LLAMA_DIR or BIN,
# WARMUP=1 (one untimed short request before the four prompts).
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

if [ "$PROMPTS" = chat ]; then
    python3 -u - "$PORT" "$MAX_TOKENS" "$OUT" "$MODEL" <<'EOF' | tee -a "$OUT/summary.txt"
import json, os, struct, sys, time, urllib.request
port, max_tokens, out, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

def gguf_sampling(path):
    # general.sampling.* from the GGUF header (pure Python, header only)
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    st = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return st()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [val(et) for _ in range(n)] if et in (8, 9) else [rd(SZ[et]) for _ in range(n)]
        return rd(SZ[t])
    if f.read(4) != b"GGUF": return {}
    rd("I"); rd("Q"); nkv = rd("Q"); out = {}
    for _ in range(nkv):
        k = st(); v = val(rd("I"))
        if k.startswith("general.sampling."): out[k.split(".")[-1]] = v
    return out

g = gguf_sampling(model)
sampling = {"temperature": float(os.environ.get("TEMP", g.get("temp", 0.8))),
            "top_p": float(os.environ.get("TOP_P", g.get("top_p", 0.95))),
            "top_k": int(os.environ.get("TOP_K", g.get("top_k", 40))),
            "min_p": float(os.environ.get("MIN_P", g.get("min_p", 0.0)))}
keep_reasoning = os.environ.get("KEEP_REASONING", "1") != "0"
print("sampling: " + ", ".join(f"{k} {v:g}" for k, v in sampling.items()) + f"; up to {max_tokens} tokens per reply; "
      + ("reasoning kept in the history" if keep_reasoning else "history without reasoning"))

turns = [
    "I have an old crypto-mining rig: nine NVIDIA P106 cards with 6 GB each, all on PCIe x1 risers, a 2-core Celeron "
    "and 4 GB of RAM. I'd like to run a local LLM on it for chatting and some coding help. What should I check or set "
    "up first?",
    "Write a bash script that prints each GPU's memory use, temperature and power draw every 5 seconds, and prints a "
    "warning line when a card goes above 80 °C.",
    "Why do the x1 risers matter so much when loading a model, but so little when generating text once it is loaded?",
    "Turn the checklist from your first answer into a short table with three columns: item, why it matters, how to "
    "check it.",
    "Thanks! Summarize the most important points of our conversation in three bullet points.",
]

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))

messages, log = [], []
gen_n = gen_ms = new_n = new_ms = 0
t_chat = time.time()
for i, text in enumerate(turns, 1):
    messages.append({"role": "user", "content": text})
    t0 = time.time()
    r = post("/v1/chat/completions", {"messages": messages, "max_tokens": max_tokens, "seed": 1000 + i,
                                      "cache_prompt": True, **sampling})
    wall = time.time() - t0
    msg, t = r["choices"][0]["message"], r.get("timings", {})
    answer = msg.get("content") or ""
    n_answer = len(post("/tokenize", {"content": answer})["tokens"]) if answer else 0
    reasoning = msg.get("reasoning_content") or ""
    messages.append({"role": "assistant", "content": answer,
                     **({"reasoning_content": reasoning} if keep_reasoning and reasoning else {})})
    log.append({"turn": i, "wall_s": wall, "response": r})
    gen_n += t.get("predicted_n", 0); gen_ms += t.get("predicted_ms", 0)
    new_n += t.get("prompt_n", 0); new_ms += t.get("prompt_ms", 0)
    print(f"turn {i}: context {t.get('cache_n', 0) + t.get('prompt_n', 0):5d} tok, {t.get('prompt_n', 0):4d} new at "
          f"{t.get('prompt_per_second', 0):5.1f} t/s, first token after {t.get('prompt_ms', 0) / 1000:4.1f} s | generated "
          f"{t.get('predicted_n', 0):4d} tok ({n_answer} answer) at {t.get('predicted_per_second', 0):5.2f} t/s, "
          f"{r['choices'][0].get('finish_reason')} | {wall:5.1f} s")
json.dump({"sampling": sampling, "keep_reasoning": keep_reasoning, "messages": messages, "turns": log}, open(f"{out}/chat.json", "w"), indent=1)
print(f"conversation: {time.time() - t_chat:.0f} s; generated {gen_n} tokens at {1000 * gen_n / gen_ms:.2f} t/s; "
      f"{new_n} new prompt tokens at {1000 * new_n / new_ms:.1f} t/s, {new_ms / 1000:.1f} s waiting for first tokens")
EOF
elif [ "$PROMPTS" != none ]; then
    python3 - "$PORT" "$MAX_TOKENS" "$OUT" <<'EOF' | tee -a "$OUT/summary.txt"
import json, os, sys, urllib.request
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
if os.environ.get("WARMUP", "0") != "0":  # one untimed request first: CUDA graph capture, first-request allocations
    body = json.dumps({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32, "temperature": 0,
                       "cache_prompt": False}).encode()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                  {"Content-Type": "application/json"}), timeout=1800).read()
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
grep -E "model buffer size|KV buffer size|compute buffer size|graph nodes|graph splits|n_ctx|CUDA graphs|checkpoint|re-processing|backend sampling" "$OUT/server.log" \
    | sed 's/^[0-9.: ]*[IWD] //' | sort | uniq -c | sort -rn | head -40 > "$OUT/buffers.txt"
echo "== done; buffer sizes in $OUT/buffers.txt"
