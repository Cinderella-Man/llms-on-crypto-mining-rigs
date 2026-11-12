#!/usr/bin/env python3
"""r84a_compare.py: R84 (a) (Q1304), C703's kept catch-up against the draft's own catch-up, verify step by verify step.

  python3 -I r84a_compare.py --draft <dump-draft.jsonl> --overlap <dump-overlap.jsonl> --out <tsv>

The dumps are the files llama-server writes with LLAMA_SPEC_DUMP: one JSON line per verify step (the drafted tokens,
the draft's candidates and probabilities at each drafted position, the main model's targets and probabilities, the
tokens kept). The control's two loads (draft1, draft2) append to one file; each server start writes its own `run`
(its start time in microseconds), so the lines are split by `run`, the earlier start first. The fields that name the
process (run, step, task) are dropped and steps are paired in file order, as R82's dump_compare.py does.

Two comparisons per step:
  draft fields  the drafted tokens and the draft's probabilities: `draft`, `cand`, `p`, `pick` (the row's pass test)
  all fields    every field the line has (the targets, their probabilities, the tokens kept, the position, ...)
Where draft1 and draft2 agree on a step, overlap1 must give that same step; where they differ, the step is the
controls' own spread and is not counted. Writes one line per step and prints the totals. Standard library only.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--draft", required=True, help="dump of the control arm (draft1 and draft2 appended)")
ap.add_argument("--overlap", required=True, help="dump of the switch's load (overlap1)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
DROP = ("run", "step", "task")
DRAFT = ("draft", "cand", "p", "pick")


def load(path):
    runs = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            r = d.get("run")
            for k in DROP:
                d.pop(k, None)
            runs.setdefault(r, []).append(d)
    return [runs[r] for r in sorted(runs)]


def sub(d, keys):
    return {k: d.get(k) for k in keys}


def max_dp(x, y):
    """the largest difference between two steps' draft probabilities, where their shapes match"""
    m = 0.0
    for rx, ry in zip(x.get("p") or [], y.get("p") or []):
        for px, py in zip(rx, ry):
            m = max(m, abs(px - py))
    return m


dr, ov = load(a.draft), load(a.overlap)
if len(dr) != 2 or len(ov) != 1:
    raise SystemExit(f"expected 2 runs in {a.draft} and 1 in {a.overlap}, found {len(dr)} and {len(ov)}")
d1, d2, o1 = dr[0], dr[1], ov[0]
n = min(len(d1), len(d2), len(o1))
t = dict(agree_all=0, same_all=0, agree_dr=0, same_dr=0, spread_dr=0, drafted=0, drafted_same=0, with_p=0)
first = None
mx_o = mx_c = 0.0
with open(a.out, "w") as f:
    f.write("i\tpos\tn_prompt\tn_draft\tn_acc\tcontrols_agree_draft\toverlap_same_draft\tcontrols_agree_all"
            "\toverlap_same_all\tdiff_fields_overlap\tdiff_fields_controls\n")
    for i in range(n):
        x, y, z = d1[i], d2[i], o1[i]
        if x.get("pos") != z.get("pos") or x.get("n_prompt") != z.get("n_prompt"):
            print(f"step {i}: position or prompt differ (draft1 {x.get('pos')}/{x.get('n_prompt')}, "
                  f"overlap1 {z.get('pos')}/{z.get('n_prompt')})")
        ca_dr, sa_dr = sub(x, DRAFT) == sub(y, DRAFT), sub(z, DRAFT) == sub(x, DRAFT)
        ca_all, sa_all = x == y, z == x
        nd = len(x.get("draft") or [])
        t["with_p"] += int(bool(x.get("p")))
        if ca_dr:
            t["agree_dr"] += 1
            t["drafted"] += nd
            if sa_dr:
                t["same_dr"] += 1
                t["drafted_same"] += nd
            elif first is None:
                first = (i, x.get("pos"), sorted(k for k in DRAFT if x.get(k) != z.get(k)))
        else:
            t["spread_dr"] += 1
        if ca_all:
            t["agree_all"] += 1
            t["same_all"] += int(sa_all)
        mx_o, mx_c = max(mx_o, max_dp(x, z)), max(mx_c, max_dp(x, y))
        diff_o = sorted(k for k in set(x) | set(z) if x.get(k) != z.get(k))
        diff_c = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
        f.write(f"{i}\t{x.get('pos')}\t{x.get('n_prompt')}\t{nd}\t{x.get('n_acc')}\t{int(ca_dr)}\t{int(sa_dr)}"
                f"\t{int(ca_all)}\t{int(sa_all)}\t{','.join(diff_o)}\t{','.join(diff_c)}\n")
print(f"steps: draft1 {len(d1)}, overlap1 {len(o1)}, draft2 {len(d2)}; paired {n}; with the draft's probabilities {t['with_p']}")
print(f"drafted tokens and probabilities: draft1 and draft2 agree on {t['agree_dr']} steps ({t['drafted']} drafted "
      f"tokens), differ on {t['spread_dr']}; overlap1 the same on {t['same_dr']} of those ({t['drafted_same']} tokens), "
      f"different on {t['agree_dr'] - t['same_dr']}")
print(f"every field: draft1 and draft2 agree on {t['agree_all']} steps; overlap1 the same on {t['same_all']} of those")
print(f"largest draft probability difference: overlap1 - draft1 {mx_o:.5f}, draft2 - draft1 {mx_c:.5f}")
print(f"first difference where the controls agree: {first}")
ok = n > 0 and t["agree_dr"] > 0 and t["same_dr"] == t["agree_dr"] and len(d1) == len(o1) == len(d2)
print("PASS" if ok else "FAIL")
