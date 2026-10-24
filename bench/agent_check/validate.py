"""Validate every task the way Harbor validates its tasks, with no model:
- oracle: the workspace with the reference solution and the hidden tests passes;
- empty change: the untouched workspace with the hidden tests fails;
- visible: the reference solution also passes the visible tests, and they are a strict subset of the hidden ones.

    python3 validate.py --sandbox bwrap --node-dir … --pi-root … --ac-node … --ac-py … [--tasks id,id] [--jobs 2]
    python3 validate.py --sandbox docker [--image agent-check:0.87.1]

Writes validation.json next to the tasks (tasks/polyglot/validation.json) and exits non-zero if any task fails a check.
The tests run in the sandbox without network.
"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import TASKS, Sandbox, load_tasks, make_workspace, run_check


def validate(task, sb, root):
    out = {"id": task["id"]}
    d = Path(root) / task["id"]
    nm = sb.node_modules()
    ref = task["dir"] / "reference"
    for name, kw, want in (("oracle", dict(solution_from=ref, hidden=True), True),
                           ("empty", dict(hidden=True), False),
                           ("visible", dict(solution_from=ref), True)):
        w = make_workspace(task, d / name, nm, **kw)
        r = run_check(task, w, sb, name=f"{sb.prefix}-val-{os.getpid()}-{task['id']}-{name}")
        out[name] = dict(ok=r["passed"] == want, passed=r["passed"], n_pass=r["n_pass"], n_fail=r["n_fail"],
                         n_total=r["n_total"], seconds=r["seconds"])
        if r["passed"] != want:
            out[name]["output"] = r["output"][-3000:]
    o, v = out["oracle"], out["visible"]
    out["subset_ok"] = (v["n_total"] or 0) < (o["n_total"] or 0)
    out["ok"] = o["ok"] and out["empty"]["ok"] and v["ok"] and out["subset_ok"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    Sandbox.add_args(ap)
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=TASKS / "validation.json")
    a = ap.parse_args()
    sb = Sandbox.from_args(a)
    tasks = load_tasks(a.tasks.split(",") if a.tasks else None)
    with tempfile.TemporaryDirectory(prefix="ac-validate-") as root:
        with ThreadPoolExecutor(a.jobs) as ex:
            results = list(ex.map(lambda t: validate(t, sb, root), tasks))
    bad = [r for r in results if not r["ok"]]
    for r in results:
        o, e, v = r["oracle"], r["empty"], r["visible"]
        print(f"{r['id']:40s} {'ok ' if r['ok'] else 'BAD'} oracle {o['n_pass']}/{o['n_total']} "
              f"empty {e['n_pass']}/{e['n_total']} visible {v['n_pass']}/{v['n_total']} "
              f"({o['seconds']:.1f} s)")
    summary = dict(sandbox=a.sandbox, n=len(results), ok=len(results) - len(bad), tasks=results)
    if not a.tasks:
        a.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{len(results) - len(bad)} of {len(results)} tasks valid")
    for r in bad:
        for k in ("oracle", "empty", "visible"):
            if "output" in r[k]:
                print(f"--- {r['id']} {k}:\n{r[k]['output']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
