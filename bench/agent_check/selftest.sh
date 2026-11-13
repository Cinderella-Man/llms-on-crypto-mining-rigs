#!/usr/bin/env bash
# selftest.sh OUT: exercise the runner, proxy, replay and report end to end against mock_server.py (no model, light).
# Needs the sandbox paths in the environment (see README "Setup"): AC_HOME (env/setup.sh), AC_NODE_DIR, AC_PI_ROOT;
# or AC_SANDBOX=docker (the docker/Dockerfile image, AC_IMAGE, default agent-check:0.87.1; AC_CPU_SHARES sets the
# containers' --cpu-shares). Uses ports AC_TEST_PORT
# to AC_TEST_PORT+2 (default 18080 to 18082) on 127.0.0.1.
# Checks, and exits non-zero if any fails:
# - 12 tasks, 3 in flight: the mock solves every task except those it is told to fail (CRC32 0 mod 4), so exactly
#   those fail; its dropped streams (CRC32 1 mod 5, first try) become re-run infrastructure tries, never scores;
#   its tool-call markup in text (CRC32 2 mod 4) is counted as malformed;
# - a second run of the same command scores nothing new (resume);
# - a turn cap of 3 stops an attempt and flags it a runaway;
# - every request has a proxy row with timings, and replay re-sends a trace (streamed and not) and summarizes it;
# - replay's --requests, --repeat (a seed per repeat), --reply-limit and --count-parts do what they say;
# - the token mark, against a second mock whose agent loops on the second task's first try and on every try of the
#   third: --gen-budget 9000 --mark-tries 3 stops each looping try at 3,000 generated tokens, queues a fresh try
#   behind the others, passes the second task on try 2 and scores the third's last try as it stands; a second run
#   resumes; a plain --gen-mark stops without a fresh try; token_mark.py prices the mark on a run without it (the
#   attempts it stops, the pass it loses below the passing task's tokens, the stop at the token run.py stops at) and
#   sums up the arms by the try that passed;
# - the graph reallocations from server logs (C804): gonogo's server-log lines gain the count (a resumed load's count
#   starts again, a log without the count says so), compare marks the arm that reallocates where the other does not,
#   and compare's other lines are those it writes without the logs.
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
T2=$(sed -n 2p "$HERE/tasks/polyglot/order.txt")
T3=$(sed -n 3p "$HERE/tasks/polyglot/order.txt")
python3 "$HERE/mock_server.py" --port $((P + 2)) --decode-ms 4 --loop-tasks "$T2:1,$T3" > "$OUT/mock_loop.log" 2>&1 &
LOOP=$!
trap 'kill $MOCK $PROXY $LOOP 2>/dev/null' EXIT
sleep 1
mark() { python3 "$HERE/run.py" --base-url http://127.0.0.1:$((P + 2))/v1 --out "$OUT/mark" --work-root "$OUT/work" \
  "${SB[@]}" "$@"; }
mark --arm mock-mark --tasks "$T1,$T2,$T3" --gen-budget 9000 --mark-tries 3 > "$OUT/run4.txt" 2>&1
mark --arm mock-mark --tasks "$T1,$T2,$T3" --gen-budget 9000 --mark-tries 3 > "$OUT/run5.txt" 2>&1
mark --arm mock-plain --tasks "$T3" --gen-mark 3000 > "$OUT/run6.txt" 2>&1
mark --arm mock-loop --tasks "$T1,$T3" --turn-cap 8 > "$OUT/run7.txt" 2>&1
printf 'run\tmodel\tfamily\tmachine\tdir\nloop\tmock\tmocks\trig\tmark/mock-loop\n' > "$OUT/runs.tsv"
python3 -I "$HERE/token_mark.py" price --runs "$OUT/runs.tsv" --marks 50,3000,100000 --out "$OUT/price" \
  > "$OUT/price.txt"
python3 -I "$HERE/token_mark.py" arms "$OUT/mark" --arms mock-mark,mock-plain,mock-loop > "$OUT/arms.md"
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1,mock/$T3/a1/t1" \
  --server http://127.0.0.1:$P --label replay-stream --out "$OUT/replay.jsonl" > "$OUT/replay1.txt"
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1,mock/$T3/a1/t1" \
  --server http://127.0.0.1:$P --label replay-nostream --stream off --out "$OUT/replay.jsonl" > "$OUT/replay2.txt"
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1,mock/$T3/a1/t1" \
  --server http://127.0.0.1:$P --label replay-repeat --requests 1 --repeat 3 --reply-limit 7 --count-parts \
  --out "$OUT/replay_repeat.jsonl" > "$OUT/replay3.txt"
python3 -I "$HERE/replay.py" run --log "$OUT/proxy.jsonl" --traces "mock/$T1/a1/t1" \
  --server http://127.0.0.1:$P --label replay-request --requests 1 --reply-limit request \
  --out "$OUT/replay_repeat.jsonl" > "$OUT/replay4.txt"
python3 -I "$HERE/replay.py" summary "$OUT/replay.jsonl" --log "$OUT/proxy.jsonl" --boot 200 > "$OUT/replay_summary.md"
python3 -I "$HERE/report.py" table "$OUT/res" --log "$OUT/proxy.jsonl" > "$OUT/table.md"
python3 -I "$HERE/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" --replay "$OUT/replay.jsonl" \
  --replay-label replay-stream > "$OUT/gonogo.md" || true
pe() { echo "slot print_timing: id  0 | task $1 | prompt eval time =     100.00 ms /   $2 tokens"; }
gr() { echo "slot print_timing: id  0 | task $1 |   graph reallocs =   $2    /   $3        (same size) / (changed)"; }
mkdir -p "$OUT/rig"
{ echo "=== mock: load 1"; pe 1 600; gr 1 2 1; pe 2 20; gr 2 2 1
  echo "=== mock: load 2, resumed"; echo "srv  llama_server: initializing ..."; pe 1 900; gr 1 3 1
} > "$OUT/rig/server-mock.log"
{ echo "=== mock-cap: load 1"; pe 1 600; gr 1 0 0; pe 2 30; gr 2 0 0; } > "$OUT/rig/server-mock-cap.log"
{ echo "=== nocount: load 1"; pe 1 600; } > "$OUT/rig/server-nocount.log"
python3 -I "$HERE/report.py" gonogo "$OUT/res" --log "$OUT/proxy.jsonl" --server-log "$OUT"/rig/server-*.log \
  > "$OUT/gonogo_realloc.md" || true
python3 -I "$HERE/report.py" compare "$OUT/res" --log "$OUT/proxy.jsonl" --arms mock,mock-cap \
  --turns-csv "$OUT/turns_realloc.csv" > "$OUT/compare_plain.md"
python3 -I "$HERE/report.py" compare "$OUT/res" --log "$OUT/proxy.jsonl" --arms mock,mock-cap \
  --turns-csv "$OUT/turns_realloc.csv" --server-log "$OUT"/rig/server-*.log > "$OUT/compare_realloc.md"
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
rr = [json.loads(l) for l in (out / "replay_repeat.jsonl").open()]
r3 = [r for r in rr if r["label"] == "replay-repeat"]
seed0 = {(r["trace"], r["idx"]): r["sampling"]["seed"] for r in rep if r["label"] == "replay-stream"}
check(len(r3) == 6 and not any(r.get("error") for r in r3) and all(r["idx"] == 1 for r in r3)
      and sorted(r["rep"] for r in r3) == [0, 0, 1, 1, 2, 2],
      f"replay --requests 1 --repeat 3 sends one request of each trace three times ({len(r3)})")
check(len({(r["trace"], r["sampling"]["seed"]) for r in r3}) == 6
      and all(r["sampling"]["seed"] == seed0[(r["trace"], 1)] for r in r3 if r["rep"] == 0),
      "a seed per repeat, the first one the seed of a replay without --repeat")
check(all(r["max_tokens"] == 7 and 0 < r["predicted_n"] <= 7 for r in r3), "replay --reply-limit 7 caps the reply at 7 tokens")
check(all(isinstance(r.get(k), int) for r in r3 for k in ("reasoning_n", "content_n", "tool_args_n", "reasoning_chars")),
      "replay --count-parts keeps the parts' tokens and characters")
r4 = [r for r in rr if r["label"] == "replay-request"]
body1 = {l["id"]: l["body"] for l in map(json.loads, __import__("gzip").open(out / "proxy.jsonl.bodies.gz", "rt"))}
check(len(r4) == 1 and r4[0]["max_tokens"] == body1[r4[0]["rec_id"]].get("max_tokens"),
      f"replay --reply-limit request keeps the request's own max_tokens ({r4[0]['max_tokens'] if r4 else None})")
check("GO" in (out / "gonogo.md").read_text(), "gonogo report written")
# graph reallocations (C804)
gl = {l.split(":")[0]: l for l in (out / "gonogo_realloc.md").read_text().splitlines() if l.startswith("- server log")}
want = {"mock": "graph reallocations 2 of 3 requests (5 same size, 2 changed)",
        "mock-cap": "graph reallocations 0 of 2 requests (0 same size, 0 changed)",
        "nocount": "graph reallocations not counted (no `graph reallocs` line)"}
check(len(gl) == 3 and all(gl[f"- server log server-{k}.log"].endswith(", " + v) for k, v in want.items()),
      f"gonogo: each server log's graph reallocations, a resumed load counted from its start ({list(gl.values())})")
plain = (out / "compare_plain.md").read_text().splitlines()
withl = (out / "compare_realloc.md").read_text().splitlines()
added = [l for l in withl if l.startswith("- mock") and "graph reallocations" in l]
check(added == [f"- mock: {want['mock']}: **reallocates** where mock-cap does not", f"- mock-cap: {want['mock-cap']}"],
      f"compare: the arm that reallocates is marked where the other does not ({added})")
i = next((k for k, l in enumerate(withl) if l.startswith("Graph reallocations of the target")), None)
# sorted: compare orders tasks that started in the same second (the mock's) by Python's per-process set order
check(i is not None and sorted(withl[:i - 1] + withl[i + 4:]) == sorted(plain),
      "compare: every other line as without the server logs")
# the token mark
t1, t2, t3 = (here / "tasks/polyglot/order.txt").read_text().split()[:3]
res = lambda arm, t, d="a1": json.loads((out / "mark" / arm / t / d / "result.json").read_text())
mm = {t: res("mock-mark", t) for t in (t1, t2, t3)}
folders = {t: sorted(p.name for p in (out / "mark/mock-mark" / t).glob("a1-t*")) for t in (t1, t2, t3)}
check(mm[t1]["success"] == 1 and mm[t1]["mark_try"] == 1 and folders[t1] == [],
      f"a task under the mark passes on try 1 ({mm[t1]['mark_try']}, {folders[t1]})")
check(mm[t2]["success"] == 1 and mm[t2]["mark_try"] == 2 and folders[t2] == ["a1-t1-mark"],
      f"a try stopped at the mark starts again and passes on try 2 ({mm[t2]['mark_try']}, {folders[t2]})")
check(mm[t3]["success"] == 0 and mm[t3]["mark_try"] == 3 and mm[t3]["runaway"] == "tokens"
      and folders[t3] == ["a1-t1-mark", "a1-t2-mark"],
      f"the last try stopped at the mark is scored as it stands ({mm[t3]['mark_try']}, {mm[t3]['runaway']!r}, {folders[t3]})")
stops = [json.loads((out / "mark/mock-mark" / t / f / "result.json").read_text()) for t in (t2, t3) for f in folders[t]]
gens = [s["generated"] for s in stops] + [mm[t3]["generated"]]
check(all(s["status"] == "mark_stopped" and s["runaway"] == "tokens" for s in stops)
      and all(r.get("gen_mark") == 3000 for r in stops + list(mm.values())) and all(3000 <= g < 4000 for g in gens),
      f"--gen-budget 9000 --mark-tries 3 stops each try at 3,000 tokens ({gens})")
starts = [l.split()[1].split("/")[1:4:2] for l in (out / "run4.txt").read_text().splitlines() if l.endswith(" start")]
want = [[t1, "t1"], [t2, "t1"], [t3, "t1"], [t2, "t2"], [t3, "t2"], [t3, "t3"]]
check(starts == want, f"a fresh try waits behind every try already queued ({starts})")
check(all(r["generated_all"] == sum(e["generated"] for e in r["earlier_tries"]) + r["generated"]
          and r["wall_all_s"] >= r["wall_s"] and len(r["earlier_tries"]) == r["mark_try"] - 1 for r in mm.values()),
      "the row sums the tokens and time of all tries")
check("passed on try 1: 1, 2: 1, 3: 0" in (out / "run4.txt").read_text(), "the run's log says which try passed")
r5 = (out / "run5.txt").read_text()
check("3 scored" in r5 and not any(l.endswith(" start") for l in r5.splitlines()),
      "a second run with the mark resumes without re-running")
plain = res("mock-plain", t3)
check(plain["runaway"] == "tokens" and plain["mark_try"] == 1 and plain["status"] == "scored"
      and not list((out / "mark/mock-plain" / t3).glob("a1-t*")),
      "--gen-mark alone stops the try and scores it, without a fresh try")
loop = {t: res("mock-loop", t) for t in (t1, t3)}
check(loop[t1]["success"] == 1 and loop[t3]["runaway"] == "turns", "without a mark the looping task runs to the turn cap")
prs = list(csv.DictReader((out / "price/price.tsv").open(), delimiter="\t"))
pr = {int(r["mark"]): r for r in prs if r["level"] == "model"}
check(sorted({(r["level"], r["group"]) for r in prs}) == [("family", "mocks"), ("model", "mock")],
      "price: one group per model and per family of the run list")
check(int(pr[3000]["over_mark"]) == 1 and int(pr[3000]["passes_lost"]) == 0 and int(pr[3000]["failures_cut"]) == 1
      and int(pr[3000]["capped"]) == 1 and int(pr[3000]["tokens_saved"]) > 0 and float(pr[3000]["rig_minutes"]) > 0,
      "price: a mark of 3,000 stops the looping attempt, loses no pass and saves tokens")
check(int(pr[50]["over_mark"]) == 2 and int(pr[50]["passes_lost"]) == 1 and int(pr[100000]["over_mark"]) == 0,
      "price: a mark below the passing task's tokens loses its pass; one above every attempt stops nothing")
at = {r["task_id"]: r for r in csv.DictReader((out / "price/attempts.tsv").open(), delimiter="\t")}
gap = plain["generated"] - int(at[t3]["stop_tokens_3000"])
check(0 <= gap < 600 and 0 < float(at[t3]["stop_s_3000"]) < float(at[t3]["wall_s"]),
      f"price stops at the token run.py stops at (run.py {plain['generated']}, price {at[t3]['stop_tokens_3000']}), "
      f"before the attempt's end ({at[t3]['stop_s_3000']} of {at[t3]['wall_s']} s)")
arms = (out / "arms.md").read_text()
check("| mock-mark | 3 | 2 | 1:1 2:1 | 3 |" in arms, "arms: passes by the try that passed, tries stopped at the mark")
sys.exit(1 if fails else 0)
EOF
echo "selftest passed; reports in $OUT (table.md, gonogo.md, replay_summary.md)"
