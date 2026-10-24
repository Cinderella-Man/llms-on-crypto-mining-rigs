#!/usr/bin/env bash
# dryrun.sh: runs run_matrix.sh against stub programs (llama-bench, llama-server, nvidia-smi, the memory-offset kit) on
# any Linux machine with Python 3 and curl, without GPUs and without root, to check its control flow:
#   1. untuned then tuned, two models: tables for both blocks and the tuning gain, offsets set on every card and back to
#      0, units measured for the cards that lack them, no process left behind;
#   2. an output that changes under the offset: the tuned block stops (exit 3) and the offsets go back to 0;
#   3. a failing llama-bench for one engine: the others still run, exit 1, the table lists the failure;
#   4. TERM while a server runs: the trap stops the server and resets the offsets.
# Usage: bench/release_matrix/test/dryrun.sh   (PASS/FAIL per check; exit 0 when all pass)
set -u
here=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$here/.." && pwd)
T=$(mktemp -d)
if [ "${KEEP:-0}" = 1 ]; then trap 'echo "outputs kept in $T"' EXIT; else trap 'rm -rf "$T"' EXIT; fi
fails=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); fi; }

mkdir -p "$T/stub" "$T/kit" "$T/models"
for e in stock v1 v2; do mkdir -p "$T/builds/$e/bin"; done
echo gguf > "$T/models/a.gguf"; echo gguf > "$T/models/b.gguf"

# llama-bench: two JSON lines, speeds by engine (from its path), +4 % when the stub offset is set
cat > "$T/stub/llama-bench" <<'EOF'
#!/usr/bin/env bash
eng=$(basename "$(dirname "$(dirname "$0")")")
[ "${STUB_BENCH_FAIL:-}" = "$eng" ] && { echo "stub: failing on purpose" >&2; exit 1; }
case $eng in stock) tg=16.5 pp=185;; v1) tg=32.6 pp=187;; *) tg=39.9 pp=233;; esac
[ -e "$STUB_KIT/offset_set" ] && tg=$(awk -v t="$tg" 'BEGIN { print t * 1.04 }')
printf '{"n_prompt": 512, "n_gen": 0, "avg_ts": %s}\n{"n_prompt": 0, "n_gen": 128, "avg_ts": %s}\n' "$pp" "$tg"
EOF
# llama-server: /health and /v1/chat/completions with timings; temperature 0 gives a fixed reply, which changes under
# the offset when STUB_DIFF=1
cat > "$T/stub/llama-server.py" <<'EOF'
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
eng = os.path.basename(os.path.dirname(os.path.dirname(sys.argv[1])))  # argv[1]: the engine's llama-server path
pipe = "--pipe-depth" in sys.argv
kit = os.environ["STUB_KIT"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.send(200, {"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tuned = os.path.exists(f"{kit}/offset_set")
        text = "fixed greedy reply"
        if req.get("temperature", 1.0) != 0.0:
            text = f"sampled reply {req.get('seed')}"
        elif tuned and os.environ.get("STUB_DIFF") == "1":
            text = "a different reply"
        tps = {"stock": 14.2, "v1": 28.0}.get(eng, 47.0 if pipe else 39.0) * (1.03 if tuned else 1.0)
        tps += 0.1 * (req.get("seed") or 0)
        self.send(200, {"choices": [{"message": {"content": text, "reasoning_content": "thinking"}}],
                        "timings": {"predicted_n": req.get("max_tokens"), "predicted_per_second": tps,
                                    "prompt_n": 20, "prompt_ms": 80.0}})
time.sleep(float(os.environ.get("STUB_LOAD_S", "0.5")))
HTTPServer(("127.0.0.1", port), H).serve_forever()
EOF
for e in stock v1 v2; do
    cp "$T/stub/llama-bench" "$T/builds/$e/bin/llama-bench"
    printf '#!/usr/bin/env bash\nexec python3 %s "$0" "$@"\n' "$T/stub/llama-server.py" > "$T/builds/$e/bin/llama-server"
done
# nvidia-smi: no compute processes; a CSV line per interval for -lms/-l (memory clock 4304 under the stub offset)
cat > "$T/stub/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case " $* " in *query-compute-apps*) exit 0;; esac
case " $* " in *" -lms "*|*" -l "*)
    while :; do
        mc=4004; [ -e "$STUB_KIT/offset_set" ] && mc=4304
        echo "2026/10/25 10:00:00.000, 0, P2, 1835, $mc, 35.00, 45, 0x0000000000000000"; sleep 0.2
    done;;
esac
EOF
# the kit: nvml_memoc.py set/reset/clocks write a log and an offset file; memoc_steps.sh units writes a units file
cat > "$T/kit/nvml_memoc.py" <<'EOF'
import os, sys
kit = os.path.dirname(os.path.abspath(__file__))
cmd, gpu = sys.argv[1], sys.argv[2]
with open(f"{kit}/calls.log", "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
if cmd == "set":
    if not os.path.exists(f"{kit}/memoc_units_gpu{gpu}.json"):
        print("no units file"); sys.exit(1)
    open(f"{kit}/offset_set", "w").close()
elif cmd == "reset":
    if os.path.exists(f"{kit}/offset_set"):
        os.remove(f"{kit}/offset_set")
elif cmd == "clocks":
    print(f"gpu {gpu}: P2 sm 1835 mem 4304")
EOF
cat > "$T/kit/memoc_steps.sh" <<'EOF'
#!/usr/bin/env bash
[ "$1" = units ] && echo '{"factor": 1}' > "$(dirname "$0")/memoc_units_gpu$GPU.json" && echo "units gpu $GPU"
EOF
# programs the script calls that must not touch the real machine: pgrep never sees a running llama-server, dmesg and
# journalctl are empty, id says root
cat > "$T/stub/pgrep" <<'EOF'
#!/usr/bin/env bash
[ "$1" = -x ] && [ "$2" = llama-server ] && exit 1
exec /usr/bin/pgrep "$@"
EOF
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/stub/dmesg"; cp "$T/stub/dmesg" "$T/stub/journalctl"
printf '#!/usr/bin/env bash\n[ "$1" = -u ] && { echo 0; exit 0; }\nexec /usr/bin/id "$@"\n' > "$T/stub/id"
chmod +x "$T"/stub/* "$T"/builds/*/bin/* "$T/kit/memoc_steps.sh"

run() {  # run NAME [ENV...]: run_matrix.sh with the stubs; output in $T/NAME.out, status in $T/NAME.rc
    local name=$1; shift
    rm -f "$T/kit/offset_set" "$T/kit/calls.log" "$T/kit/memoc_units_gpu2.json"
    echo '{"factor": 1}' > "$T/kit/memoc_units_gpu0.json"; echo '{"factor": 1}' > "$T/kit/memoc_units_gpu1.json"
    env PATH="$T/stub:$PATH" STUB_KIT="$T/kit" GPUS=0,1,2 PORT=18089 MAX_TOKENS=16 N_GREEDY=8 \
        BIN_stock="$T/builds/stock/bin" BIN_v1="$T/builds/v1/bin" BIN_v2="$T/builds/v2/bin" KIT="$T/kit" \
        OUT="$T/out-$name" "$@" bash "$M/run_matrix.sh" > "$T/$name.out" 2>&1
    echo $? > "$T/$name.rc"
}
left() { pgrep -f "$T/stub/llama-server.py" > /dev/null; }

run clean MODELS="A=$T/models/a.gguf B=$T/models/b.gguf" MEM_OFFSET=600
check "clean: exit 0" '[ "$(cat $T/clean.rc)" = 0 ]'
check "clean: both blocks and the gain table" 'grep -q "^\*\*Stock clocks\*\*" $T/out-clean/tables.md && grep -q "^\*\*Tuned: memory +600 MT/s on every card\*\*" $T/out-clean/tables.md && grep -q "What the tuning adds" $T/out-clean/tables.md'
check "clean: untuned v2 generation 39.9, tuned +4.0 %" 'grep -q "^| v2 engine | 39.9 |" $T/out-clean/tables.md && grep -q "^| v2 engine | +4.0 %" $T/out-clean/tables.md'
check "clean: pipelined row has no generation number" 'grep -q "^| v2 engine, pipelined speculation (\`--pipe-depth 3\`) | — | 47" $T/out-clean/tables.md'
check "clean: greedy identical in all 8 served configurations" 'grep -q "identical to stock clocks: 8 of 8" $T/out-clean/tables.md'
check "clean: units measured for gpu 2 only" '[ -f $T/kit/memoc_units_gpu2.json ] && [ "$(grep -c "no units file" $T/out-clean/offsets.log)" = 0 ]'
check "clean: offsets set on 3 cards, reset afterwards" '[ "$(grep -c "^set" $T/kit/calls.log)" = 3 ] && [ ! -e $T/kit/offset_set ] && grep -q "^reset 2" $T/kit/calls.log'
check "clean: tuned memory clock logged" 'grep -q "4304 MHz tuned" $T/out-clean/tables.md'
check "clean: no server left" '! left'

run diff MODELS="A=$T/models/a.gguf" MEM_OFFSET=600 STUB_DIFF=1
check "diff: exit 3" '[ "$(cat $T/diff.rc)" = 3 ]'
check "diff: the change is reported, offsets back to 0" 'grep -q "DIFFERS from untuned" $T/diff.out && [ ! -e $T/kit/offset_set ]'
check "diff: no server left" '! left'

run benchfail MODELS="A=$T/models/a.gguf" STUB_BENCH_FAIL=v1
check "benchfail: exit 1, untuned only" '[ "$(cat $T/benchfail.rc)" = 1 ] && ! grep -q Tuned $T/out-benchfail/tables.md'
check "benchfail: stock and v2 measured, failure listed" 'grep -q "^| Stock llama.cpp | 16.5 |" $T/out-benchfail/tables.md && grep -q "^| v2 engine | 39.9 |" $T/out-benchfail/tables.md && grep -q "Failed runs: untuned v1 A (bench)" $T/out-benchfail/tables.md'

( run term MODELS="A=$T/models/a.gguf" MEM_OFFSET=600 BLOCKS=tuned STUB_LOAD_S=30 ) &
for i in $(seq 1 60); do left && break; sleep 0.5; done
sleep 1; pkill -TERM -f "$M/run_matrix.sh"; wait
check "term: interrupted, offsets back to 0" 'grep -q interrupted $T/term.out && [ ! -e $T/kit/offset_set ] && grep -q "^reset" $T/kit/calls.log'
sleep 1
check "term: no server left" '! left'

echo "$fails failed"
[ "$fails" = 0 ]
