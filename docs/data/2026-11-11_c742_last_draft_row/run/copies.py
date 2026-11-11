#!/usr/bin/env python3
"""C742 (Q1311): the MTP drafter's hidden-row copies to the host per draft, from llama-server's step trace
(LLAMA_STEP_TRACE, C405) and its verify-step dump (LLAMA_SPEC_DUMP). Made from C744's passes.py.

  python3 -I copies.py [--n-max 3] [--tsv OUT.tsv] ARM=DIR ... > copies.md

DIR holds steps.jsonl (the dump's verify steps, one line per slot and step) and, unless the arm ran without the trace,
step_trace.jsonl. A draft is the marks from mtp.drf to mtp.drf.end, its passes the mtp.dp marks between them. The
branch's drafter adds to the line of a draft's end the draft context's micro-batches that copied their hidden rows to the
host (n.h.host) and those that left them out (n.h.skip); the base's trace has neither. Per arm:
  gen          trace lines with a draft and an acceptance (s.acc), as C744 counts generation steps
  drafts       drafts, and passes their drafting passes
  full         drafts whose passes reach n_max: their last pass is the one LLAMA_MTP_SKIP_LAST_ROW marks
  h_host       hidden rows copied to the host, h_skip those left out (sums of n.h.host, n.h.skip)
  counted      lines with a draft and either count (0 for the base)
  bad          counted lines where the copies are not one per pass, the left out included (h.host + h.skip = passes), or
               where h.skip is neither 0 (the switch off) nor the line's full drafts (the switch on); "-" without counts
  per_draft    h_host / drafts
  two          lines whose drafted tokens (n.drf) exceed n_max: a draft of both slots
  steps        lines of the dump
Standard library only.
"""
import argparse
import json
import sys


def trace_counts(path, n_max):
    out = {"gen": 0, "drafts": 0, "passes": 0, "full": 0, "h_host": 0, "h_skip": 0, "counted": 0, "bad": 0, "two": 0}
    try:
        f = open(path)
    except OSError:
        return None
    seg = None  # the passes of the open draft (a draft may end on a later line than it began)
    with f:
        for raw in f:
            o = json.loads(raw)
            if "step" not in o:
                continue
            n = o.get("n", {})
            names = [m[0] for m in o.get("t", [])]
            segs = []
            for name in names:
                if name == "mtp.drf":
                    seg = 0
                elif name == "mtp.dp" and seg is not None:
                    seg += 1
                elif name == "mtp.drf.end" and seg is not None:
                    segs.append(seg)
                    seg = None
            if n.get("drf", 0) > 0 and "s.acc" in names:
                out["gen"] += 1
            if n.get("drf", 0) > n_max:
                out["two"] += 1
            if not segs:
                continue
            passes = sum(segs)
            full = sum(1 for s in segs if s == n_max)
            h, s = n.get("h.host", 0), n.get("h.skip", 0)
            out["drafts"] += len(segs)
            out["passes"] += passes
            out["full"] += full
            out["h_host"] += h
            out["h_skip"] += s
            if "h.host" in n or "h.skip" in n:
                out["counted"] += 1
                if h + s != passes or s not in (0, full):
                    out["bad"] += 1
    if out["counted"] == 0:
        out["bad"] = "-"
    return out


def dump_steps(path):
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=3)
    ap.add_argument("--tsv")
    ap.add_argument("arms", nargs="+", metavar="ARM=DIR")
    a = ap.parse_args()

    keys = ["gen", "drafts", "passes", "full", "h_host", "h_skip", "counted", "bad", "two"]
    cols = ["arm"] + keys + ["per_draft", "steps"]
    rows = []
    for spec in a.arms:
        arm, d = spec.split("=", 1)
        t = trace_counts(f"{d}/step_trace.jsonl", a.n_max)
        r = {"arm": arm, "steps": dump_steps(f"{d}/steps.jsonl")}
        if t is None:
            r.update({k: "" for k in keys})
            r["per_draft"] = ""
            r["trace"] = False
        else:
            r.update(t)
            r["per_draft"] = f"{t['h_host'] / t['drafts']:.2f}" if t["drafts"] and t["counted"] else ""
            r["trace"] = True
        rows.append(r)

    print(f"| arm | generation steps | drafts | drafting passes | drafts at n_max ({a.n_max}) | hidden rows copied to the host "
          "| left out | copies a draft | lines off the rule | drafts of both slots | verify steps (dump) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["trace"] and r["counted"]:
            tr = (f"{r['gen']} | {r['drafts']} | {r['passes']} | {r['full']} | {r['h_host']} | {r['h_skip']} "
                  f"| {r['per_draft']} | {r['bad']} | {r['two']}")
        elif r["trace"]:
            tr = f"{r['gen']} | {r['drafts']} | {r['passes']} | {r['full']} | not counted | | | | {r['two']}"
        else:
            tr = "no trace | | | | | | | |"
        print(f"| {r['arm']} | {tr} | {r['steps']} |")

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
