#!/usr/bin/env python3
"""C740 (Q1309): the two server stages that hold a speculative step's seq_rm calls, from LLAMA_STEP_TRACE files.

  python3 -I rm_stages.py LOAD=step_trace.jsonl [LOAD=step_trace.jsonl ...] > rm_stages.md

Each trace line after the header is one step of llama-server (C405's trace). Two intervals per step, in ms:
  s.drf.end -> s.ckpt    after the draft: the draft context's llama_memory_seq_rm (the drafted tokens out of its
                         memory) and the checkpoint bookkeeping; steps that drafted (n.drf > 0)
  s.acc.spec -> s.rm     after the acceptance: the slot's seq_rm on the target's and the draft's memories (the
                         rejected tokens out), the prompt's tokens updated; steps that have both marks
Per load: the steps, the median and the mean of each interval, and the median of the two together. Standard library
only.
"""
import json
import statistics
import sys


def stage(marks, a, b):
    ta = tb = None
    for name, t in marks:
        if name == a and ta is None:
            ta = t
        elif name == b and ta is not None and tb is None:
            tb = t
    return None if ta is None or tb is None else tb - ta


def load(path):
    drf, acc, both = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            d = json.loads(line)
            marks = d.get("t", [])
            x = stage(marks, "s.drf.end", "s.ckpt") if d.get("n", {}).get("drf", 0) > 0 else None
            y = stage(marks, "s.acc.spec", "s.rm")
            if x is not None:
                drf.append(x)
            if y is not None:
                acc.append(y)
            if x is not None and y is not None:
                both.append(x + y)
    return drf, acc, both


def fmt(v):
    if not v:
        return "—", "—", "0"
    return f"{statistics.median(v):.4f}", f"{statistics.mean(v):.4f}", str(len(v))


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    print("| load | steps (draft) | `s.drf.end → s.ckpt` median | mean | steps (acceptance) | `s.acc.spec → s.rm` median | mean | both, median (steps) |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for arg in argv:
        name, path = arg.split("=", 1)
        drf, acc, both = load(path)
        dm, da, dn = fmt(drf)
        am, aa, an = fmt(acc)
        bm, _, bn = fmt(both)
        print(f"| {name} | {dn} | {dm} | {da} | {an} | {am} | {aa} | {bm} ({bn}) |")
    print()
    print("ms; the steady clock of the trace")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
