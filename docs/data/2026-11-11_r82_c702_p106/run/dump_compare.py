#!/usr/bin/env python3
"""dump_compare.py: R82 (Q1297), the verify steps of a switch's load against two control loads of the same line.

  python3 -I dump_compare.py --control <dump of control load 1> --control2 <dump of control load 2> \
      --arm <dump of the switch's load> --out <tsv>

Each dump is the file the server writes with LLAMA_SPEC_DUMP: one JSON line per verify step (the drafted tokens, their
candidates and probabilities, the main model's targets and probabilities, the tokens kept). The fields that name the
process (run, step, task) are dropped; every other field must be equal. Steps are paired in file order. Where the two
control loads agree on a step, the switch's load must give that same step; where they differ, the step is the
controls' own spread and is not counted. Writes one line per step and prints the totals. Standard library only.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--control", required=True)
ap.add_argument("--control2", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DROP = ("run", "step", "task")


def load(path):
    rows = []
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            for k in DROP:
                d.pop(k, None)
            rows.append(d)
    return rows


c1, c2, s = load(a.control), load(a.control2), load(a.arm)
n = min(len(c1), len(c2), len(s))
agree = same = differ = spread = 0
first_diff = None
with open(a.out, "w") as f:
    f.write("i\tpos\tcontrols_agree\tarm_same_as_control\tdiff_fields\n")
    for i in range(n):
        ca = c1[i] == c2[i]
        sa = s[i] == c1[i]
        diff = sorted(k for k in set(c1[i]) | set(s[i]) if c1[i].get(k) != s[i].get(k))
        if ca:
            agree += 1
            if sa:
                same += 1
            else:
                differ += 1
                if first_diff is None:
                    first_diff = (i, c1[i].get("pos"), diff)
        else:
            spread += 1
        f.write(f"{i}\t{c1[i].get('pos')}\t{int(ca)}\t{int(sa)}\t{','.join(diff)}\n")
print(f"steps: control {len(c1)}, control2 {len(c2)}, arm {len(s)}; paired {n}")
print(f"controls agree on {agree} steps (differ on {spread}); the arm the same on {same} of those, different on {differ}")
print(f"first difference where the controls agree: {first_diff}")
