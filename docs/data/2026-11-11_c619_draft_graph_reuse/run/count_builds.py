#!/usr/bin/env python3
"""C619 (Q1104): the draft context's graph builds per step in LLAMA_STEP_TRACE files (C405's trace).

  python3 -I count_builds.py LOAD=step_trace.jsonl [LOAD=step_trace.jsonl ...] > builds.md

Each trace line after the header is one step of the server. The draft context's micro-batches mark "d.ub.apply" and
then "d.ub.build" (its graph built and allocated) or "d.ub.reuse" (its graph reused). Per load, over the steps that
drafted (n.drf > 0): the steps, the draft micro-batches, builds and reuses, builds per step, and the median per step of
the host time from each draft micro-batch's "d.ub.apply" to its "d.ub.build" or "d.ub.reuse", summed over the step.
Standard library only.
"""
import json
import statistics
import sys


def load(path):
    steps = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            d = json.loads(line)
            if d.get("n", {}).get("drf", 0) <= 0:
                continue
            built = reused = 0
            host = 0.0
            t_apply = None
            for name, t in d.get("t", []):
                if name == "d.ub.apply":
                    t_apply = t
                elif name in ("d.ub.build", "d.ub.reuse"):
                    if name == "d.ub.build":
                        built += 1
                    else:
                        reused += 1
                    if t_apply is not None:
                        host += t - t_apply
                    t_apply = None
            steps.append((built, reused, host))
    return steps


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    print("| load | steps that drafted | draft micro-batches | built | reused | built per step | apply → build or reuse, ms a step (median) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for arg in argv:
        name, path = arg.split("=", 1)
        steps = load(path)
        n = len(steps)
        b = sum(s[0] for s in steps)
        r = sum(s[1] for s in steps)
        med = statistics.median(s[2] for s in steps) if steps else float("nan")
        print(f"| {name} | {n} | {b + r} | {b} | {r} | {b / n if n else float('nan'):.3f} | {med:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
