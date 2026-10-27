#!/usr/bin/env bash
# The Elixir path's CPU check (2026-10-26, no model, no GPU), run before the Elixir pilot. Needs Docker and the local
# images python:3.12-slim and elixir:1.18.4-otp-27 at the digests below. WORK is deleted and created again.
#   BENCH=<this repository's bench/ folder> WORK=<scratch folder> Q51=<release/flint-v2/reproduce/data/q51/prompts.jsonl> \
#   bash cputest.sh
# What it does: the Elixir sandbox image, workspaces for every eligible library module, the workspaces gate, stage-1
# tasks, and stage-2 tasks from a fixture "session" per module. The fixture texts (Elixir templates and change note)
# and the fixture test files are test inputs written for this check only: they stay in WORK and never enter a
# dataset.
set -euo pipefail
B=${BENCH:?the bench/ folder of this repository}
A=$B/agentic_data
W=${WORK:?scratch folder}
Q51=${Q51:?the Q51 prompts.jsonl}
rm -rf "$W"; mkdir -p "$W"
cd "$W"
PY="nice -n 19 python3 -I"
docker build -q --label pascal-quants=1 -f "$A/sandbox/Dockerfile.elixir" -t agx-sandbox:ex118 "$A/sandbox"
# fixture texts: the reviewed Python texts plus placeholder-only Elixir texts (test input, not data)
$PY - "$A/texts.json" "$W/texts.json" << 'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
fx = {"write_tests": "FIXTURE write {test_file} for {module_file}, run {test_command}",
      "fix_bug": "FIXTURE fix {module_file}: {failing_tests} in {test_file}",
      "implement": "FIXTURE implement {function} in {module_file}, tests {test_file}",
      "change_note": "FIXTURE {source_file} renamed {modules}"}
for k, v in fx.items():
    t["templates"].setdefault(k, []).append({"id": f"ex-{k}-fixture", "kind": k, "text": v, "lang": "elixir"})
t["tools"]["ex:run_tests"] = {"id": "ex:run_tests", "text": "FIXTURE run mix test"}
t["params"]["ex:run_tests.target"] = {"id": "ex:run_tests.target", "text": "FIXTURE test file"}
t["fixture"] = True
json.dump(t, open(sys.argv[2], "w"))
EOF
mkdir -p trees scans
c=$(docker create --label pascal-quants=1 elixir:1.18.4-otp-27@sha256:a7b2c4772e05616cf0a3323758ba5abc3288af452a33d1263705840a52154e85)
docker cp "$c:/usr/local/src/elixir" trees/elixir-1.18.4 > /dev/null
docker cp "$c:/usr/share/common-licenses/Apache-2.0" trees/Apache-2.0.txt > /dev/null
docker rm "$c" > /dev/null
docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none -v "$A:$A:ro" -v "$W:$W" \
  python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2 \
  python3 -I "$A/license_scan.py" "$W/trees/elixir-1.18.4" --license Apache-2.0 --out "$W/scans/elixir-1.18.4-lib" | tail -1
echo "== workspaces (all)"
$PY "$A/workspace.py" elixir "$W/trees/elixir-1.18.4" --modules all --scan scans/elixir-1.18.4-lib.tsv \
  --notice "$A/notices/elixir-1.18.4-NOTICE" --licence-file trees/Apache-2.0.txt --texts texts.json --out ws \
  > ws.log 2>&1 || { tail -20 ws.log; exit 1; }
grep -E "kept|doctests pass" ws.log | tail -60
grep -c "^skip" ws.log || true
grep "^skip" ws.log | sed -E 's/^skip [^:]+: //' | sort | uniq -c | sort -rn | head
echo "== gate workspaces"
$PY "$A/gate.py" workspaces ws/*.manifest.json --sources "$B/staircase_drafter/sources.json" --codebases "$A/codebases.json" \
  --texts texts.json --scan scans/*.tsv --q51 "$Q51" \
  --agent-tasks "$B/agent_check/tasks" --out ws.gate.json | tail -8
echo "== stage-1 tasks"
$PY "$A/tasks.py" write-tests --workspaces ws --texts texts.json --out tasks-s1.jsonl
head -c 900 tasks-s1.jsonl; echo
echo "== fixture sessions for three modules, then stage-2 tasks"
mkdir -p s1/sessions
$PY - tasks-s1.jsonl s1/sessions << 'EOF'
import json, sys, hashlib
rows = [json.loads(x) for x in open(sys.argv[1])]
want = ["option_parser", "version", "uri"]
for t in rows:
    m = t["workspace_id"].split(":")[1]
    if m not in want:
        continue
    mod = "Stdmods." + "".join(w.capitalize() for w in m.split("_"))
    body = "".join(f'  test "fixture {i}" do\n    assert {i} == {i}\n  end\n\n' for i in range(5))
    src = f"defmodule {mod}FixtureTest do\n  use ExUnit.Case, async: true\n\n{body}end\n"
    s = {"session_id": t["task_id"], "task": t, "verify": {"passed": True},
         "final_files": {t["placeholders"]["test_file"]: {"sha256": hashlib.sha256(src.encode()).hexdigest(),
                                                          "content": src}},
         "writer": {"id": "fixture"}}
    json.dump(s, open(f"{sys.argv[2]}/{t['task_id']}.json", "w"))
EOF
$PY "$A/tasks.py" from-sessions --sessions s1/sessions --workspaces ws --texts texts.json --out s2 --per-module 3 \
  --implement 1 --max-candidates 30
$PY - s2/tasks.jsonl << 'EOF'
import json, sys
for x in open(sys.argv[1]):
    t = json.loads(x)
    print(t["task_id"], t["type"], t["change"]["kind"], t["change"]["function"], "| fails:",
          t["validation"]["failing"][:3], "| docs", t["verify"].get("docs_sha256", "")[:12])
    print("   ", t["change"]["diff"].strip().splitlines()[-2:])
EOF
echo "== done"
