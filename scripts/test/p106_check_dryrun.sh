#!/usr/bin/env bash
# p106_check_dryrun.sh: runs scripts/p106_check.sh against stub programs (nvidia-smi, modinfo, llama-server) on any
# Linux machine with Python 3 and curl, without GPUs, to check its verdicts:
#   1. a ready rig, every served configuration: no FAIL;
#   2. the wrong driver branch; 3. a missing card; 4. a tool request answered without a tool call (WARN);
#   5. a server that does not start with -ub 512 (FAIL for that configuration only).
# Usage: scripts/test/p106_check_dryrun.sh   (PASS/FAIL per check; exit 0 when all pass)
set -u
here=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$here/.." && pwd)/p106_check.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); fi; }
mkdir -p "$T/stub" "$T/bin" "$T/models"
head -c 1048576 /dev/zero > "$T/models/flint.gguf"

cat > "$T/stub/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case " $* " in
    *driver_version*) for i in $(seq 1 "${STUB_CARDS:-9}"); do echo "${STUB_DRV:-580.178.04}"; done ;;
    *query-compute-apps*) ;;
    *memory.total*) for i in $(seq 0 $(( ${STUB_CARDS:-9} - 1 ))); do echo "$i, NVIDIA P106-090, 6077, 3"; done ;;
esac
EOF
printf '#!/usr/bin/env bash\necho NVIDIA\n' > "$T/stub/modinfo"
cat > "$T/stub/pgrep" <<'EOF'
#!/usr/bin/env bash
[ "$1" = -x ] && [ "$2" = llama-server ] && exit 1
exec /usr/bin/pgrep "$@"
EOF
cat > "$T/bin/server.py" <<'EOF'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
if "--list-devices" in args:
    print("Available devices:")
    for i in range(int(os.environ.get("STUB_CARDS", "9"))):
        print(f"  CUDA{i}: NVIDIA P106-090 (6077 MiB, 6000 MiB free)")
    sys.exit(0)
if os.environ.get("STUB_SERVER_FAIL") == "ub512" and "512" in args[args.index("-ub") + 1:args.index("-ub") + 2]:
    print("error: failed to allocate the compute buffers: out of memory"); sys.exit(1)
port = int(args[args.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body):
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.send({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msg = {"content": "Hash tables: caches, sets, counting."}
        if "tools" in req and os.environ.get("STUB_NOTOOL") != "1":
            msg = {"content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"}}]}
        if "TANGERINE" in req["messages"][0]["content"]:
            msg = {"content": "The code word is TANGERINE."}
        n = len(req["messages"][0]["content"]) // 4
        self.send({"choices": [{"message": msg}], "timings": {"prompt_n": n, "prompt_per_second": 700.0,
                                                              "predicted_n": 40, "predicted_per_second": 41.5}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
EOF
printf '#!/usr/bin/env bash\nexec python3 %s "$@"\n' "$T/bin/server.py" > "$T/bin/llama-server"
chmod +x "$T"/stub/* "$T/bin/llama-server"

run() {  # run NAME [ENV...]
    local name=$1; shift
    env PATH="$T/stub:$PATH" BIN="$T/bin" MODEL="$T/models/flint.gguf" PORT=18091 "$@" bash "$S" > "$T/$name.out" 2>&1
    echo $? > "$T/$name.rc"
}
run ready SERVE=1
check "ready: exit 0, no FAIL" '[ "$(cat $T/ready.rc)" = 0 ] && ! grep -q "^FAIL" $T/ready.out'
check "ready: driver, cards, engine devices pass" 'grep -q "^PASS  driver .*580.178.04" $T/ready.out && grep -q "^PASS  cards .*9 × P106-090" $T/ready.out && grep -q "^PASS  engine devices .*9 CUDA" $T/ready.out'
check "ready: four served configurations pass" '[ "$(grep -c "^PASS  serve" $T/ready.out)" = 4 ]'
check "ready: the tool call is reported" 'grep -q "tool calls: get_weather" $T/ready.out'
check "ready: the long prompt is about 16k tokens" 'grep -qE "serve long.*prompt 1[0-9]{4} tokens" $T/ready.out'
run driver STUB_DRV=550.163.01
check "driver: FAIL for the 550 branch" '[ "$(cat $T/driver.rc)" = 1 ] && grep -q "^FAIL  driver .*550" $T/driver.out'
run cards STUB_CARDS=8
check "cards: FAIL for 8 of 9, engine devices FAIL" 'grep -q "^FAIL  cards .*8 cards visible" $T/cards.out && grep -q "^FAIL  engine devices" $T/cards.out'
run notool SERVE=1 CONFIGS=tools9 STUB_NOTOOL=1
check "notool: WARN, no FAIL" '[ "$(cat $T/notool.rc)" = 0 ] && grep -q "^WARN  serve tools.*without calling the tool" $T/notool.out'
run ub512 SERVE=1 CONFIGS="chat9 long9" STUB_SERVER_FAIL=ub512
check "ub512: long fails, chat passes" 'grep -q "^FAIL  serve long" $T/ub512.out && grep -q "^PASS  serve chat" $T/ub512.out && [ "$(cat $T/ub512.rc)" = 1 ]'
check "no server left" '! pgrep -f "$T/bin/server.py" > /dev/null'
echo "$fails failed"
[ "$fails" = 0 ]
