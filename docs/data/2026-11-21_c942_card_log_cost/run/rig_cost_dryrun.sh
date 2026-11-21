#!/usr/bin/env bash
# rig_cost_dryrun.sh: rig_cost.sh on a machine without NVIDIA cards, in about 30 s. card_log.py runs on
# scripts/card_log_standin.py's stand-in for NVML (its --shim python3), and llama-server is a stub that answers /health
# and /completion with fixed timings, so no model, card or NVIDIA library is touched. Short settings (IDLE_S=1,
# REPS=1, ROUNDS=1, REQS=2). Checks that every arm ran and logged as asked, that the tables come out, and that the
# script leaves no logger or server behind. Prints one line per check and the counts; exit status 1 on a failure.
#   bash docs/data/2026-11-21_c942_card_log_cost/run/rig_cost_dryrun.sh      # KEEP=1 keeps the scratch folder
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
T=$(mktemp -d /tmp/rig_cost_dryrun.XXXXXX)
if [ "${KEEP:-0}" = 1 ]; then trap 'echo "kept $T"' EXIT; else trap 'rm -rf "$T"' EXIT; fi
pass=0 fail=0
check() { local d=$1; shift; if "$@"; then pass=$((pass + 1)); echo "ok    $d"; else fail=$((fail + 1)); echo "FAIL  $d"; fi; }
mkdir -p "$T/stub" "$T/bin" "$T/run"
python3 -I "$REPO/scripts/card_log_standin.py" --shim "$T/stub"
cp "$HERE/rig_cost.sh" "$REPO/scripts/card_log.py" "$T/run/"
cat > "$T/server.py" << 'STUB'
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, body):
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.3)
        self.reply({"content": "stub", "timings": {"prompt_n": 52, "prompt_per_second": 101.5,
                                                   "predicted_n": req["n_predict"], "predicted_per_second": 45.25}})
time.sleep(1)  # a load
HTTPServer(("127.0.0.1", port), H).serve_forever()
STUB
printf '#!/usr/bin/env bash\necho "stub llama-server $*"\nexec python3 %s "$@"\n' "$T/server.py" > "$T/bin/llama-server"
chmod +x "$T/bin/llama-server"
cat > "$T/settings.env" << 'PRESET'
PRESET_MODEL=stub.gguf
PRESET_ENV="STUB_ENV=1"
PRESET_ARGS="-ngl 99 -c 4096"
PRESET
O=$T/out
touch "$T/flint.gguf"
env PATH="$T/stub:$PATH" OUT="$O" PORT=18944 BIN="$T/bin" MODEL="$T/flint.gguf" PRESET="$T/settings.env" IDLE_S=1 REPS=1 ROUNDS=1 \
  REQS=2 N_PREDICT=16 bash "$T/run/rig_cost.sh" > "$T/console.txt" 2>&1
rc=$?
check "rig_cost.sh ends with status 0 and DONE ($rc)" test "$rc" = 0 -a -e "$O/DONE"
for a in today fast slow4; do
  check "idle $a: a log with rows and a stop line after 1 s" bash -c \
    "[ \$(grep -c '^[0-9]' '$O/idle_${a}_1.csv') -ge 2 ] && grep -q 'card_log stop (after 1 s)' '$O/idle_${a}_1.csv'"
done
check "idle today: no fast line; fast: every field; slow4: one sample in 4" bash -c \
  "! grep -q 'card_log fast' '$O/idle_today_1.csv' && grep -q 'reads every field on every sample' '$O/idle_fast_1.csv' && grep -q 'one sample in 4' '$O/idle_slow4_1.csv'"
check "idle today_profile and slow4_profile: a profile line each" bash -c \
  "grep -q 'card_log profile: ' '$O/idle_today_profile_p.csv' && grep -q 'card_log profile: ' '$O/idle_slow4_profile_p.csv' && grep -q 'one sample in 4 and at a P-state change, the other fields on every sample, profiled' '$O/idle_slow4_profile_p.csv'"
check "busy: the stub server took MODEL and the preset's settings" bash -c \
  "grep -q 'stub llama-server -m $T/flint.gguf -ngl 99 -c 4096 --host 127.0.0.1 --port 18944' '$O/server.log'"
check "busy off: no log; today, fast, slow4 and today_profile: a log ended by SIGTERM" bash -c \
  "[ ! -e '$O/busy_1_off.csv' ] && for a in 1_today 1_fast 1_slow4 p_today_profile; do grep -q 'card_log stop (signal)' \"$O/busy_\$a.csv\" || exit 1; done"
check "busy.tsv: 2 requests for each of the 4 arms and today_profile, with speeds (10 rows)" bash -c \
  "[ \$(tail -n +2 '$O/busy.tsv' | grep -c \$'\t45.25') = 10 ]"
check "summary.md: a row for each of the 5 idle and 5 busy arms, the logger's cost and the speeds by value" bash -c \
  "[ \$(grep -c '^| idle ' '$O/summary.md') = 5 ] && [ \$(grep -c '^| busy ' '$O/summary.md') = 5 ] && grep -q '^| busy | 1 | off | off | .* | 45.25, 45.25 | 101.50, 101.50 |' '$O/summary.md' && grep -q '^| busy | 1 | fast | [0-9.]* ms CPU a sample' '$O/summary.md'"
check "host_cpu.tsv: one row an arm (10)" test "$(tail -n +2 "$O/host_cpu.tsv" | wc -l)" = 10
check "no logger and no server left" bash -c "! pgrep -f 'card_log_standi[n].py.*$T' > /dev/null && ! pgrep -f '$T/server[.]py' > /dev/null"
env -u PRESET PATH="$T/stub:$PATH" OUT="$T/out2" PORT=18944 SKIP_IDLE=1 bash "$T/run/rig_cost.sh" > "$T/console2.txt" 2>&1
rc2=$?
check "the busy phase without PRESET stops at once with a reason (status $rc2)" bash -c \
  "[ $rc2 != 0 ] && grep -q 'PRESET (a release settings.env) is needed' '$T/out2/run.log'"
[ "$fail" = 0 ] || tail -n 20 "$T/console.txt" "$O/run.log" 2> /dev/null
echo "rig_cost_dryrun: $pass passed, $fail failed"
[ "$fail" = 0 ]
