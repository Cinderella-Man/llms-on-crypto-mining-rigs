#!/usr/bin/env bash
# selftest.sh OUT: exercise the runner, proxy, replay and report end to end against mock_server.py (no model, light).
# Needs the sandbox paths in the environment (see README "Setup"): AC_HOME (env/setup.sh), AC_NODE_DIR, AC_PI_ROOT;
# or AC_SANDBOX=docker (the docker/Dockerfile image, AC_IMAGE, default agent-check:0.87.1; AC_CPU_SHARES sets the
# containers' --cpu-shares). Uses ports AC_TEST_PORT
# and AC_TEST_PORT+1 (default 18080/18081) on 127.0.0.1.
# Checks, and exits non-zero if any fails:
# - 12 tasks, 3 in flight: the mock solves every task except those it is told to fail (CRC32 0 mod 4), so exactly
#   those fail; its dropped streams (CRC32 1 mod 5, first try) become re-run infrastructure tries, never scores;
#   its tool-call markup in text (CRC32 2 mod 4) is counted as malformed;
# - a second run of the same command scores nothing new (resume);
# - a turn cap of 3 stops an attempt and flags it a runaway;
# - every request has a proxy row with timings, and replay re-sends a trace (streamed and not) and summarizes it.
set -euo pipefail
OUT=${1:?usage: selftest.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
P=${AC_TEST_PORT:-18080}
Q=$((P + 1))
if [ "${AC_SANDBOX:-bwrap}" = docker ]; then
  SB=(--sandbox docker --image "${AC_IMAGE:-agent-check:0.87.1}" --docker-prefix acself
      ${AC_CPU_SHARES:+--docker-cpu-shares "$AC_CPU_SHARES"})
else
  SB=(--sandbox "${AC_SANDBOX:-bwrap}" --node-dir "$AC_NODE_DIR" --pi-root "$AC_PI_ROOT" --ac-node "$AC_HOME/node"
      --ac-py "$AC_HOME/py")
fi
rm -rf "$OUT"; mkdir -p "$OUT"
python3 "$HERE/mock_server.py" --port $P --fail-every 4 --drop-every 5 --malformed-every 4 > "$OUT/mock.log" 2>&1 &
MOCK=$!
python3 -I "$HERE/proxy.py" --listen 127.0.0.1:$Q --upstream 127.0.0.1:$P --log "$OUT/proxy.jsonl" > "$OUT/proxy.log" 2>&1 &
PROXY=$!
trap 'kill $MOCK $PROXY 2>/dev/null' EXIT
sleep 1
run() { python3 "$HERE/run.py" --base-url http://127.0.0.1:$Q/v1 --out "$OUT/res" --work-root "$OUT/work" "${SB[@]}" "$@"; }
run --arm mock --tasks 12 --parallel 3 > "$OUT/run1.txt" 2>&1
run --arm mock --tasks 12 --parallel 3 > "$OUT/run2.txt" 2>&1
run --arm mock-cap --tasks javascript-twelve-days --turn-cap 3 > "$OUT/run3.txt" 2>&1
T1=$(sed -n 1p "$HERE/tasks/polyglot/order.txt")
T3=$(sed -n 3p "$HERE/tasks/polyglot/order.txt")
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1,mock/$T3/a1/t1" \
  --server http://127.0.0.1:$P --label replay-stream --out "$OUT/replay.jsonl" > "$OUT/replay1.txt"
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1,mock/$T3/a1/t1" \
  --server http://127.0.0.1:$P --label replay-nostream --stream off --out "$OUT/replay.jsonl" > "$OUT/replay2.txt"
python3 -I "$HERE/replay.py" summary "$OUT/replay.jsonl" --log "$OUT/proxy.jsonl" --boot 200 > "$OUT/replay_summary.md"
python3 -I "$HERE/report.py" table "$OUT/res" --log "$OUT/proxy.jsonl" > "$OUT/table.md"
python3 -I "$HERE/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" --replay "$OUT/replay.jsonl" \
  --replay-label replay-stream > "$OUT/gonogo.md" || true
python3 -I - "$HERE" "$OUT" <<'EOF'
import csv, json, sys, zlib
from pathlib import Path
here, out = Path(sys.argv[1]), Path(sys.argv[2])
order = (here / "tasks/polyglot/order.txt").read_text().split()[:12]
crc = {t: zlib.crc32(t.encode()) for t in order}
rows = {r["task_id"]: r for r in csv.DictReader((out / "res/attempts.csv").open()) if r["file"] == "mock"}
fails = []
def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)
check(sorted(rows) == sorted(order), f"12 attempts scored ({len(rows)})")
check(all(int(rows[t]["pass"]) == (crc[t] % 4 != 0) for t in rows), "pass exactly where the mock solved the task")
infra = sorted(p.parent.name + "/" + p.name for p in (out / "res/mock").glob("*/a1-t*-infra"))
want = sorted(f"{t}/a1-t1-infra" for t in order if crc[t] % 5 == 1)
check(infra == want, f"dropped streams re-run, not scored: {infra}")
check(all((int(rows[t]["undecodable"]) > 0) == (crc[t] % 4 == 2) for t in rows), "markup in text counted as malformed")
check("12 scored" in (out / "run2.txt").read_text() and "start" not in (out / "run2.txt").read_text(),
      "second run resumes without re-running")
cap = json.loads(next((out / "res/mock-cap").glob("*/a1/result.json")).read_text())
check(cap["runaway"] == "turns" and cap["turns"] == 4, f"turn cap stops the attempt (runaway {cap['runaway']!r}, turns {cap['turns']})")
log = [json.loads(l) for l in (out / "proxy.jsonl").open()]
check(all(r.get("prompt_ms") is not None for r in log if not r.get("upstream_error") and r.get("status") == 200),
      f"proxy rows carry timings ({len(log)} rows)")
check(all(int(rows[t]["requests"]) == int(rows[t]["turns"]) for t in rows), "one proxy row per turn of every attempt")
rep = [json.loads(l) for l in (out / "replay.jsonl").open()]
check(len(rep) == 20 and not any(r.get("error") for r in rep), f"replay re-sent 2 traces × 5 requests twice ({len(rep)})")
check(all(r["max_tokens"] == r["rec"]["predicted_n"] for r in rep), "replay caps max_tokens at the recorded length")
check("GO" in (out / "gonogo.md").read_text(), "gonogo report written")
sys.exit(1 if fails else 0)
EOF
echo "selftest passed; reports in $OUT (table.md, gonogo.md, replay_summary.md)"
