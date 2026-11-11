#!/usr/bin/env python3
"""stop_pairs.py: the end-of-turn stops against the same arm without them (registry Q1262, Q1291), from
run_pv_replies.sh's work folders.

    python3 -I stop_pairs.py WORK_A WORK_B --out TABLES

Reads every replay-<arm>-<set>.jsonl (bench/agent_check/replay.py's rows, one per reply, a reply being one prompt with
one seed index) and every server-<arm>-<set>.log in the given folders. Only counts are read, no time. Writes:

  pairs.tsv     per card set and pair of arms: the replies both loads have, how many of them have the same text (the
                reply's MD5) and the same length, and per load the grammar rejections, the replies with one, and the
                drafted, accepted and generated tokens
  switches.tsv  per load: whether its server log holds each start line below
"""
import argparse
import json
import sys
from pathlib import Path

PAIRS = [("pv1", "pv1s"), ("pv1b", "pv1bs"), ("pv1", "pv1b"), ("pv1s", "pv1bs"), ("pv0", "pv1")]
LINES = [  # the server's start lines (tools/server/server-context.cpp), each printed once at load
    ("block_verification_on", "speculative verification: block verification on"),
    ("rejection_stops", "token-by-token rejection stops after an accepted end of generation"),
    ("block_cut", "block verification ends each block at its first drafted end of generation"),
]


def rows(path):
    """{(trace, request, seed index): row}"""
    out = {}
    for l in path.open():
        if l.strip():
            r = json.loads(l)
            out[(r["trace"], r["idx"], r["rep"])] = r
    return out


def counts(rs):
    return dict(grammar_rejected=sum(r.get("grammar_rejected_n") or 0 for r in rs),
                replies_with_rejection=sum((r.get("grammar_rejected_n") or 0) > 0 for r in rs),
                drafted=sum(r.get("draft_n") or 0 for r in rs),
                accepted=sum(r.get("draft_n_accepted") or 0 for r in rs),
                generated=sum(r.get("predicted_n") or 0 for r in rs))


def write_tsv(path, recs, cols):
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in recs:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work", nargs="+", help="run_pv_replies.sh's WORK folders, one per card set")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    loads, logs = {}, {}
    for w in a.work:
        for p in sorted(Path(w).glob("replay-*.jsonl")):
            arm, s = p.stem[len("replay-"):].rsplit("-", 1)
            loads[(s, arm)] = rows(p)
        for p in sorted(Path(w).glob("server-*.log")):
            logs[p.stem[len("server-"):]] = p.read_text(errors="replace")
    if not loads:
        sys.exit("no replay-*.jsonl in " + " ".join(a.work))

    prec = []
    for s in sorted({s for s, _ in loads}):
        for x, y in PAIRS:
            if (s, x) not in loads or (s, y) not in loads:
                continue
            X, Y = loads[(s, x)], loads[(s, y)]
            keys = sorted(set(X) & set(Y))
            rec = dict(set=s, first=x, second=y, replies=len(keys),
                       same_text=sum(X[k].get("reply_md5") == Y[k].get("reply_md5") for k in keys),
                       same_length=sum(X[k].get("predicted_n") == Y[k].get("predicted_n") for k in keys))
            for tag, R in (("first", X), ("second", Y)):
                for c, v in counts([R[k] for k in keys]).items():
                    rec[f"{tag}_{c}"] = v
            prec.append(rec)
    cols = ["set", "first", "second", "replies", "same_text", "same_length"] + \
        [f"{t}_{c}" for t in ("first", "second") for c in
         ("grammar_rejected", "replies_with_rejection", "drafted", "accepted", "generated")]
    write_tsv(a.out / "pairs.tsv", prec, cols)

    srec = [dict(load=label, **{k: int(s in text) for k, s in LINES}) for label, text in sorted(logs.items())]
    write_tsv(a.out / "switches.tsv", srec, ["load"] + [k for k, _ in LINES])
    print(f"{len(prec)} pairs, {len(srec)} server logs -> {a.out}")


if __name__ == "__main__":
    main()
