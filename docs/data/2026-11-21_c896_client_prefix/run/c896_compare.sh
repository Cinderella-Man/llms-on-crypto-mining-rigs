#!/usr/bin/env bash
# c896_compare.sh ROOT: a single run of client_arm.sh before and after C896, on the same mock and settings. ROOT holds
# old/ (public v5 before C896) and new/ (with it). Each runs alone: arm `same`, the first 6 bugfix tasks, a mock that
# fails the tasks whose CRC32 is 0 mod 4 and drops the first try's stream of those 1 mod 5. Prints per task each
# version's status, success and infrastructure tries, the files each left in its OUT, the container names each used
# (sampled every 0.5 s) and what is left. Ports 18320 and 18321.
set -uo pipefail
ROOT=$(cd "${1:?usage: c896_compare.sh ROOT}" && pwd -P)
R=$ROOT/runs/compare
P=18320
rm -rf "$R"; mkdir -p "$R"
for v in old new; do
  python3 "$ROOT/$v/bench/agent_check/mock_server.py" --port $P --fail-every 4 --drop-every 5 > "$R/mock-$v.log" 2>&1 &
  M=$!
  (while true; do docker ps --format '{{.Names}}' | grep '^acrel-'; sleep 0.5; done) > "$R/names-$v.txt" 2>&1 &
  W=$!
  sleep 1
  t0=$(date +%s)
  OUT=$R/$v ARM=same UPSTREAM_PORT=$P PROXY_PORT=$((P + 1)) TASK_SET=bugfix TASKS=6 TIME_CAP_MIN=5 CPU_SHARES=256 \
    nice -n 19 bash "$ROOT/$v/bench/agent_check/client_arm.sh" > "$R/$v.log" 2>&1
  echo "$v: exit $?, $(($(date +%s) - t0)) s, rc file $(cat "$R/$v/same.rc")"
  kill $M $W 2> /dev/null; wait $M $W 2> /dev/null
done
python3 -I - "$R" << 'PY'
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1])
def rows(v):
    out = {}
    for t in sorted(d for d in (R / v / "res" / "same").iterdir() if d.is_dir()):
        a = json.loads((t / "a1" / "result.json").read_text())
        infra = len([p for p in t.iterdir() if p.name.endswith("-infra")])
        out[t.name] = (a["status"], a.get("success"), a.get("try_n"), infra, a.get("turns"))
    return out
o, n = rows("old"), rows("new")
print("task | old: status, success, try, infrastructure tries, turns | new: the same | equal")
for k in sorted(set(o) | set(n)):
    print(f"{k} | {o.get(k)} | {n.get(k)} | {o.get(k)[:4] == n.get(k)[:4] if k in o and k in n else False}")
print("same status, success, try and infrastructure tries on every task:",
      all(k in n and o[k][:4] == n[k][:4] for k in o) and set(o) == set(n))
for v in ("old", "new"):
    print(f"{v} OUT: {sorted(p.name for p in (R / v).iterdir())}")
    names = sorted({re.sub(r"-[0-9]+-s([0-9]+)", r"-<pid>-s\1", x) for x in (R / f"names-{v}.txt").read_text().split()})
    print(f"{v} containers seen: {names}")
PY
echo "left: $(docker ps -a --format '{{.Names}}' | grep '^acrel-' || echo none)"
