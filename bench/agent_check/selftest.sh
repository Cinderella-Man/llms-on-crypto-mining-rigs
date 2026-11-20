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
#   and compare's other lines are those it writes without the logs;
# - the work counts (C852): from a fixed per-request table with every cache path (exact, a reply read again whole and
#   in part, an older checkpoint, a row without timings), `report.py work` counts the prompt tokens read, the reply
#   tokens re-read with their turns and prompt seconds, and the longest first-token wait;
# - the class tables (C855): agent_tables.py adds a task's work counts after its old columns, the same as report.py
#   work's; class_arms.py classes a run that reads fewer tokens a gain beyond the spread while its minutes are
#   negligible, gives no share against a base of 0, and on a table without the work columns gives the time measures;
# - the energy columns (FN101): selftest_energy.sh on its own fixed inputs.
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
cat > "$OUT/work_turns.csv" <<'CSV'
arm,task_id,attempt,turn,ttft_s,t_end_s,prompt_n,cache_n,prompt_ms,predicted_n,predicted_ms,finish_reason,n_tool_calls
w,t,1,1,4.1,5,1000,0,4000,100,1000,tool_calls,1
w,t,1,2,0.4,5,50,1099,300,400,4000,tool_calls,1
w,t,1,3,2.1,5,450,1145,2000,200,2000,tool_calls,1
w,t,1,4,0.7,5,130,1695,600,50,500,tool_calls,1
w,t,1,5,9.5,10,900,1000,3600,20,200,tool_calls,1
w,t,1,6,1.0,700,,,,,,,0
w,u,1,2,0.2,1,20,509,100,6,60,stop,0
w,u,1,1,2.2,3,500,0,2000,10,100,tool_calls,1
CSV
python3 -I "$HERE/report.py" work "$OUT/work_turns.csv" --tsv "$OUT/work.tsv" > "$OUT/work.md"
# the class tables (C855): run w is the work table above, run e caches every turn exactly
mkdir -p "$OUT/class/res"
{ cat "$OUT/work_turns.csv"; cat <<'CSV'
e,t,1,1,4.0,5,1000,0,4000,100,1000,tool_calls,1
e,t,1,2,0.5,5,50,1100,250,100,1000,stop,0
e,u,1,1,2.0,3,500,0,2000,10,100,stop,0
CSV
} > "$OUT/class/res/turns.csv"
cat > "$OUT/class/res/attempts.csv" <<'CSV'
task_id,file,attempt,wall_s,pass
t,w,1,60,1
u,w,1,30,1
t,e,1,50,1
u,e,1,30,0
CSV
python3 -I "$HERE/agent_tables.py" "$OUT/class" e w > /dev/null
cat > "$OUT/class/runs.tsv" <<'TSV'
arm	equal_work_min_per_task	gen_tok_s	prompt_tok_s	ttft_median_s	ttft_p90_s	ttft_turn1_median_s	prompt_read_per_task	reread_per_task	reread_s_per_task	ttft_max_s
a1	1.80	47	200	1.7	6	3.2	2900	0	0	12
a2	1.82	48	210	1.8	6.5	3.3	3000	0	0	13
b1	1.81	47	200	1.7	6	3.2	4000	1200	5	25
b2	1.83	48	210	1.8	6.5	3.3	5000	2000	12	94
TSV
python3 -I "$HERE/class_arms.py" "$OUT/class/runs.tsv" a=a1,a2 b=b1,b2 > "$OUT/class/class_ab.tsv"
python3 -I "$HERE/class_arms.py" "$OUT/class/runs.tsv" b=b1,b2 a=a1,a2 > "$OUT/class/class_ba.tsv"
cut -f1-7 "$OUT/class/runs.tsv" > "$OUT/class/runs_time.tsv"
python3 -I "$HERE/class_arms.py" "$OUT/class/runs_time.tsv" a=a1,a2 b=b1,b2 > "$OUT/class/class_time.tsv"
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
# the work counts (C852): task t's turns 3 and 4 read the previous reply again (400 tokens, then the last 100 of 200),
# turn 5 restarts from an older checkpoint (the 50-token reply again), turn 6 has no timings; task u's rows are exact
wk = list(csv.DictReader((out / "work.tsv").open(), delimiter="\t"))
want = dict(arm="w", tasks="2", turns="8", prompt_read="3050", generated="786", reread_turns="3",
            older_checkpoint_turns="1", reread_n="550", reread_s="6.200", prompt_s="12.600", ttft_max_s="9.5000",
            ttft_max_task="t", ttft_max_turn="5", ttft_max_reread="1")
check(wk == [want], f"work: prompt tokens read, reply tokens re-read and the longest first-token wait of a turns table ({wk})")
wl = (out / "work.md").read_text().splitlines()
check("| w | 2 | 8 | 3050 (1525) | 786 (393) | 550 (3, 1 from an older checkpoint; 6.2 of 12.6) | 9.50 (t, 5, a re-read) |" in wl
      and [l.split(" | ")[2:4] for l in wl if l.startswith("| w | t | ")]
      == [["3", "reply re-read"], ["4", "reply re-read"], ["5", "older checkpoint"]],
      "work: the per-arm row, and a line per turn that read a reply again with its cache path")
# the class tables (C855): agent_tables.py's work columns, class_arms.py's work measures
tsv = lambda name: list(csv.DictReader((out / "class" / name).open(), delimiter="\t"))
ag = {r["arm"]: r for r in tsv("agent.tsv")}
cols = (out / "class/agent.tsv").read_text().splitlines()[0].split("\t")
wcols = ["prompt_read_per_task", "reread_per_task", "reread_s_per_task", "reread_turns", "ttft_max_s"]
check(cols[13] == "equal_work_vs_ref" and cols[14:] == wcols
      and [ag["w"][c] for c in wcols] == ["1525", "275", "3.10", "3", "9.50"]
      and [ag["e"][c] for c in wcols] == ["775", "0", "0.00", "0", "4.00"],
      f"agent_tables: the work columns after the old ones, a task's counts as report.py work's "
      f"({[ag[a][c] for a in 'we' for c in wcols]})")
cl = {r["measure"]: r for r in tsv("class_ab.tsv")}
want = {"minutes a task at equal work": "negligible", "prompt tokens read a task": "gain",
        "reply tokens re-read a task": "gain", "prompt s of the turns that re-read a reply, a task": "gain",
        "longest first-token wait s": "negligible"}
check(len(cl) == 10 and all(cl[m]["class"] == c for m, c in want.items()),
      f"class_arms: fewer tokens read is a gain beyond the spread, equal minutes negligible "
      f"({[(m, cl[m]['class']) for m in want if m in cl]})")
ba = {r["measure"]: r for r in tsv("class_ba.tsv")}
check(ba["reply tokens re-read a task"]["class"] == "loss" and ba["reply tokens re-read a task"]["diff_pct"] == "—",
      "class_arms: against a base that re-read nothing, a loss with no share of the base")
check([r["measure"] for r in tsv("class_time.tsv")] == [
      "minutes a task at equal work", "generation tok/s", "prompt tok/s", "first token median s", "first token p90 s",
      "first token on a task's first turn, median s"],
      "class_arms: a table without the work columns gives the time measures alone")
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
bash "$HERE/selftest_energy.sh" "$OUT/energy" > "$OUT/energy.txt" || { cat "$OUT/energy.txt"; exit 1; }
echo "selftest passed; reports in $OUT (table.md, gonogo.md, replay_summary.md, energy.txt)"
