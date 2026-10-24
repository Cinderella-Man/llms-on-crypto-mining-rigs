#!/usr/bin/env bash
# dryrun.sh: runs run_q119.sh against a stub llama-server (a small Python HTTP server answering /health and
# /v1/chat/completions) on any Linux machine, without root, a GPU or a model: checks that every mode starts a server,
# sends its requests, writes one CSV row per request and stops the server again (~2 min, mostly the 10 s pauses).
# Usage: bench/server_mmap/test/dryrun.sh
set -uo pipefail
kit=$(cd "$(dirname "$0")/.." && pwd)
d=$(mktemp -d)
mkdir -p "$d/bin" "$d/stub"
cat > "$d/bin/llama-server" <<'PY'
#!/usr/bin/env python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
print("stub llama-server args:", " ".join(sys.argv[1:]), flush=True)
time.sleep(1)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers(); self.wfile.write(b"{}")
    def do_POST(self):
        n = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["max_tokens"]
        body = json.dumps({"usage": {"completion_tokens": n}, "timings": {"predicted_per_second": 31.5}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
printf '#!/usr/bin/env bash\necho 0\n' > "$d/stub/id"
chmod +x "$d/bin/llama-server" "$d/stub/id"
PATH="$d/stub:$PATH" B="$d/bin" M=/nonexistent.gguf OUT="$d/out" PORT=18089 N=50 MLOCK_MIN_MB=1 \
    bash "$kit/run_q119.sh" > "$d/console.log" 2>&1
rc=$?
fail=0
check() { if "${@:2}"; then echo "  ok: $1"; else echo "  FAILED: $1"; fail=1; fi; }
check "exit 0 (got $rc)" [ "$rc" = 0 ]
check "6 rows, 50 tokens each" [ "$(awk -F, 'NR > 1 && $3 == 50' "$d/out/q119.csv" | wc -l)" = 6 ]
check "the mlock server got --mlock" grep -q -- '--mlock' "$d/out/server_mlock.log"
check "the nommap server got --no-mmap" grep -q -- '--no-mmap' "$d/out/server_nommap.log"
check "memory sampled in every mode" [ "$(cat "$d"/out/mem_*.csv | grep -c '^[0-9]')" -gt 3 ]
check "no server left" bash -c '! pgrep -f "[s]tub llama-server|'"$d"'/bin/[l]lama-server" > /dev/null'
[ $fail = 0 ] || tail -30 "$d/console.log"
cat "$d/out/q119.csv"
rm -rf "$d"
exit $fail
