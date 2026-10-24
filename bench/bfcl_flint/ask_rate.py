"""Did the model ask, or act? BFCL v4 multi-turn tasks have turns whose ground truth is empty: in miss_param the user
leaves out a value the calls need (the next turn gives it), and the right move is to ask rather than to call.

For every conversation (task, file, attempt) in multi_turn.py's raw output, and every turn with an empty ground
truth, a turn counts as "asked" when none of its steps holds a decodable tool call, and as "acted" otherwise. A
question mark in the model's text in that turn (outside the reasoning) is logged as a second indicator. BFCL's
verdict for the whole conversation comes from multi_turn.evaluate.

With --pair REF,OTHER it also runs the one-sided exact McNemar test that OTHER passes fewer tasks than REF, on
the tasks that have an attempt-1 verdict from both (b = only REF passes, c = only OTHER passes; p = P(X >= b),
X ~ Binomial(b + c, 1/2)).

usage: python ask_rate.py OUT_DIR FILE=RAW.jsonl [FILE=RAW.jsonl ...] [--ids IDS.txt] [--category multi_turn_miss_param]
                          [--attempt N] [--pair REF,OTHER]
Writes OUT_DIR/ask_turns.csv (one row per empty-ground-truth turn) and OUT_DIR/ask_rate.md.
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import bfcl_common as bc
import multi_turn as mt

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("raw", nargs="+")
ap.add_argument("--ids", help="only these task IDs (one per line)")
ap.add_argument("--category", default="multi_turn_miss_param")
ap.add_argument("--attempt", type=int, help="only this attempt (default: all)")
ap.add_argument("--pair", help="REF,OTHER for the one-sided exact McNemar test")
args = ap.parse_args()

ids = mt.read_ids(args.ids) if args.ids else None
entries = {e["id"]: e for e in mt.load_entries(ids, mt.read_ids(mt.EXCLUDED))
           if e["id"].startswith(args.category + "_")}

turn_rows, verdicts = [], defaultdict(dict)  # verdicts[file][(task, attempt)] = pass
for spec in args.raw:
    file, path = spec.split("=", 1)
    for rec in map(__import__("json").loads, open(path)):
        e = entries.get(rec["id"])
        if e is None or (args.attempt and rec["attempt"] != args.attempt):
            continue
        ok, err, _ = mt.evaluate(e, rec)
        verdicts[file][(rec["id"], rec["attempt"])] = int(ok)
        for ti, gt in enumerate(e["ground_truth"]):
            if gt or ti >= len(rec["turns"]):
                continue
            tools = mt.tools_at(e, ti)
            acted, question = False, False
            for st in rec["turns"][ti]:
                if st.get("error"):
                    continue
                p = bc.split_output(st["content"], tools)
                try:
                    acted |= bool(bc.decode_ast(p))
                except ValueError:
                    pass
                question |= "?" in p["content"]
            turn_rows.append({"task_id": rec["id"], "file": file, "attempt": rec["attempt"], "turn": ti,
                              "asked": int(not acted), "question_mark": int(question), "pass": int(ok),
                              "error_type": err or ""})

os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "ask_turns.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(turn_rows[0]) if turn_rows else ["task_id"])
    w.writeheader()
    w.writerows(turn_rows)

lines = [f"Category {args.category}, turns whose ground truth is empty\n",
         "| File | conversations | turns | asked (no call) | question mark | pass if asked | pass if acted |",
         "|---|---|---|---|---|---|---|"]
for file in sorted({r["file"] for r in turn_rows}):
    rows = [r for r in turn_rows if r["file"] == file]
    asked = [r for r in rows if r["asked"]]
    acted = [r for r in rows if not r["asked"]]

    def rate(xs, k):
        return f"{100 * sum(x[k] for x in xs) / len(xs):.1f} % ({sum(x[k] for x in xs)}/{len(xs)})" if xs else "—"

    lines.append(f"| {file} | {len({(r['task_id'], r['attempt']) for r in rows})} | {len(rows)} | "
                 f"{rate(rows, 'asked')} | {rate(rows, 'question_mark')} | {rate(asked, 'pass')} | "
                 f"{rate(acted, 'pass')} |")
if args.pair:
    ref, other = args.pair.split(",")
    common = sorted(t for (t, a) in verdicts[ref] if a == 1 and (t, 1) in verdicts[other])
    b = sum(verdicts[ref][(t, 1)] and not verdicts[other][(t, 1)] for t in common)
    c = sum(verdicts[other][(t, 1)] and not verdicts[ref][(t, 1)] for t in common)
    n = b + c
    p = sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0
    lines += ["", f"Attempt-1 pairs: {len(common)} tasks; {ref} passes {sum(verdicts[ref][(t, 1)] for t in common)}, "
              f"{other} passes {sum(verdicts[other][(t, 1)] for t in common)}; only {ref}: {b}, only {other}: {c}; "
              f"one-sided exact McNemar p = {p:.4f}"]
text = "\n".join(lines) + "\n"
open(os.path.join(args.out_dir, "ask_rate.md"), "w").write(text)
print(text)
