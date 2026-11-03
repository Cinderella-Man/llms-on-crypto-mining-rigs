#!/usr/bin/env python3
"""Q550: one table of every token-equality check from checks.sh's output folders.

Reads each <dir>/<check>.log: check_identity.sh's table (one row per run: generated tokens, guesses, right, wrong,
dropped tokens and same/DIFF for tokens, logits, state and probe) and llama-pipe-recurrent-check's PIPE-RECURRENT-CHECK
line. Writes a TSV with the folder's label and the check's name in front of each row, and prints the totals per label.

Usage: python3 checks_table.py <out.tsv> <label>=<dir> [<label>=<dir> ...]
"""
import re
import sys
from collections import Counter
from pathlib import Path

COLS = ["label", "check", "run", "n_gen", "guesses", "right", "wrong", "dropped", "tokens", "logits", "state", "probe",
        "verdict"]


def rows_of(label, path):
    out = []
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        f = line.split()
        # check_identity.sh: run n_gen guess hits misses wasted tok/s tokens logits state probe winner race verdict
        if len(f) == 14 and f[1].isdigit() and f[-1] in ("same", "DIFF"):
            out.append([label, path.stem, f[0], f[1], f[2], f[3], f[4], f[5], f[7], f[8], f[9], f[10], f[13]])
    m = re.search(r"PIPE-RECURRENT-CHECK (.*)", text)
    if m:
        kv = dict(x.split("=", 1) for x in m[1].split() if "=" in x)
        verdict = "same" if all(kv.get(k) == "same" for k in ("tokens", "logits", "state", "probe")) else "DIFF"
        out.append([label, path.stem, "pipe-recurrent-check", kv.get("n_gen", ""), kv.get("guesses", ""), "", "",
                    "", kv.get("tokens", ""), kv.get("logits", ""), kv.get("state", ""), kv.get("probe", ""),
                    verdict])
    return out


def main():
    rows = []
    for arg in sys.argv[2:]:
        label, d = arg.split("=", 1)
        for log in sorted(p for p in Path(d).glob("*.log") if p.stem != "summary"):
            rows += rows_of(label, log)
    with open(sys.argv[1], "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    for label in dict.fromkeys(r[0] for r in rows):
        rs = [r for r in rows if r[0] == label and r[2] not in ("plain",)]
        piped = [r for r in rs if r[2] != "pipe-recurrent-check"]
        c = Counter(r[-1] for r in rs)
        n = lambda i: sum(int(r[i]) for r in piped if r[i].isdigit())  # noqa: E731
        print(f"{label}: {len(piped)} pipelined runs + {len(rs) - len(piped)} recurrent checks, {n(3)} tokens, "
              f"{n(4)} guesses, {n(6)} wrong, {n(7)} dropped; verdicts {dict(c)}")


if __name__ == "__main__":
    main()
