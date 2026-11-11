#!/usr/bin/env python3
"""C744 (Q1314): the MTP drafter's passes per arm, from llama-server's step trace (LLAMA_STEP_TRACE, C405) and its
verify-step dump (LLAMA_SPEC_DUMP).

  python3 -I passes.py [--tsv OUT.tsv] ARM=DIR ... > passes.md

DIR holds steps.jsonl (the dump's verify steps, one line per slot and step) and, unless the arm ran without the trace,
step_trace.jsonl. Per arm:
  gen steps    trace lines with a draft (n.drf > 0) and an acceptance (s.acc)
  passes       drafting passes (mtp.dp marks) in those lines
  drafted      the drafted tokens the server kept (n.drf)
  spare        per line max(0, passes - drafted), summed, and the lines with any: the passes whose token the server cut
               (C405's count; it holds for one slot, where each pass drafts one token)
  cap          the passes LLAMA_MTP_DRAFT_CAP left out (n.cap)
  verify steps lines of the dump
  cut rows     per dump line max(0, candidate rows - draft length), summed, and the lines with any: the drafter writes one
               row of candidates per token it drafted, so these are the tokens drafted and cut, per slot
Standard library only.
"""
import argparse
import json
import sys


def trace_counts(path):
    out = {"gen": 0, "passes": 0, "drafted": 0, "accepted": 0, "spare": 0, "spare_lines": 0, "cap": 0}
    try:
        f = open(path)
    except OSError:
        return None
    with f:
        for raw in f:
            o = json.loads(raw)
            if "step" not in o:
                continue
            n = o.get("n", {})
            out["cap"] += n.get("cap", 0)
            names = [m[0] for m in o.get("t", [])]
            if n.get("drf", 0) <= 0 or "s.acc" not in names:
                continue
            n_dp = names.count("mtp.dp")
            spare = max(0, n_dp - n["drf"])
            out["gen"] += 1
            out["passes"] += n_dp
            out["drafted"] += n["drf"]
            out["accepted"] += n.get("acc", 0)
            out["spare"] += spare
            out["spare_lines"] += 1 if spare else 0
    return out


def dump_counts(path):
    out = {"steps": 0, "rows": 0, "cut_rows": 0, "cut_lines": 0}
    with open(path) as f:
        for raw in f:
            o = json.loads(raw)
            rows = len(o.get("cand", []))
            cut = max(0, rows - len(o.get("draft", [])))
            out["steps"] += 1
            out["rows"] += rows
            out["cut_rows"] += cut
            out["cut_lines"] += 1 if cut else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv")
    ap.add_argument("arms", nargs="+", metavar="ARM=DIR")
    a = ap.parse_args()

    cols = ["arm", "gen", "passes", "drafted", "accepted", "spare", "spare_lines", "cap", "steps", "rows", "cut_rows",
            "cut_lines"]
    rows = []
    for spec in a.arms:
        arm, d = spec.split("=", 1)
        t = trace_counts(f"{d}/step_trace.jsonl")
        r = {"arm": arm, **dump_counts(f"{d}/steps.jsonl")}
        if t is None:
            r.update({k: 0 for k in ("gen", "passes", "drafted", "accepted", "spare", "spare_lines", "cap")})
            r["trace"] = False
        else:
            r.update(t)
            r["trace"] = True
        rows.append(r)

    print("| arm | generation steps | drafting passes | drafted (kept) | accepted | spare passes (steps) "
          "| left out by the cap | verify steps (dump) | candidate rows | cut rows (steps) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["trace"]:
            tr = (f"{r['gen']} | {r['passes']} | {r['drafted']} | {r['accepted']} | {r['spare']} ({r['spare_lines']}) "
                  f"| {r['cap']}")
        else:
            tr = "no trace | | | | |"
        print(f"| {r['arm']} | {tr} | {r['steps']} | {r['rows']} | {r['cut_rows']} ({r['cut_lines']}) |")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
