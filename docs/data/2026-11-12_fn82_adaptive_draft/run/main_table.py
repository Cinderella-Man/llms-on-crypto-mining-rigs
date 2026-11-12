#!/usr/bin/env python3
"""FN82's main tables (main.md) from adapt_replay.py's arms.tsv(.gz) and compare.tsv.

    python3 main_table.py REPLAY_DIR > main.md
"""
import csv
import gzip
import sys
from pathlib import Path

LOADS = ["d6-1", "d7-1", "d3-1"]
TABLES = ["col8", "col10", "col12", "col14", "col16", "v4est"]
KEY = [  # (arm, what it is)
    ("adapt-n{N}-run-wdef-x1-post", "the switch as built: `LLAMA_SPEC_ADAPT=1`, S run, widths 1,3,5,9,17"),
    ("adapt-n{N}-run-wall-x1-post", "the same with every width (`LLAMA_SPEC_ADAPT_WIDTHS=1,...,n_max+1`)"),
    ("adapt-n{N}-run-wdef-x1-online", "as built, decided while drafting (est.: not built)"),
    ("adapt-n{N}-run-wall-x1-online", "every width, decided while drafting (est.: not built)"),
    ("pmin0.7-n{N}", "the drafter's own stop, `--spec-draft-p-min 0.7`"),
    ("oracle-n{N}-post", "the draft cut to what is accepted (ceiling), all passes drafted"),
    ("oracle-n{N}-online", "the same, drafting stopped at the first rejection (ceiling)"),
]


def rows(path):
    p = Path(path)
    if not p.exists() and Path(str(p) + ".gz").exists():
        p = Path(str(p) + ".gz")
    f = gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
    with f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    d = Path(sys.argv[1])
    A = {(r["load"], r["table"], r["arm"]): r for r in rows(d / "arms.tsv")}
    C = {(r["load"], r["table"], r["arm"], r["against"]): r for r in rows(d / "compare.tsv")}
    out = []
    for load in LOADS:
        N = 3 if load.startswith("d3") else 6
        fixed = [f"fixed{L}" for L in range(2, 8) if (load, TABLES[0], f"fixed{L}") in A]
        out.append(f"\n### {load}: est. tok/s by cost table (n_max {N} for the rules)\n")
        out.append("| Arm | " + " | ".join(TABLES) + " |")
        out.append("|---|" + "---:|" * len(TABLES))
        best = {}
        for t in TABLES:
            best[t] = max(fixed, key=lambda a: float(A[(load, t, a)]["tok_s_est"]))
        for a in fixed:
            cells = []
            for t in TABLES:
                v = f"{float(A[(load, t, a)]['tok_s_est']):.3f}"
                cells.append(f"**{v}**" if a == best[t] else v)
            out.append(f"| {a} | " + " | ".join(cells) + " |")
        for pat, _ in KEY:
            a = pat.format(N=N)
            if (load, TABLES[0], a) not in A:
                continue
            cells = []
            for t in TABLES:
                r = A[(load, t, a)]
                c = C.get((load, t, a, best[t]))
                s = f"{float(r['tok_s_est']):.3f}"
                if c and c["lo_requests"]:
                    s += (f" ({float(c['diff']):+.3f}: {float(c['lo_requests']):+.3f} to "
                          f"{float(c['hi_requests']):+.3f})")
                cells.append(s)
            out.append(f"| {a} | " + " | ".join(cells) + " |")
        out.append("\nBold: the best fixed length. In brackets: the difference from it, with its 95 % interval over "
                   "requests (paired on the same steps).\n")
        out.append(f"\n### {load}: what each arm verifies and drafts, at `v4est`\n")
        out.append("| Arm | What | tokens a verify step | verify columns a step | drafting passes a step | "
                   "est. ms a step | drafted tokens kept: count of steps |")
        out.append("|---|---|---:|---:|---:|---:|---|")
        for a in fixed:
            r = A[(load, "v4est", a)]
            out.append(f"| {a} | | {float(r['tokens_a_verify_step']):.4f} | {float(r['columns_a_step']):.4f} | "
                       f"{float(r['passes_a_step']):.4f} | {float(r['ms_a_step_est']):.2f} | {r['kept']} |")
        for pat, what in KEY:
            a = pat.format(N=N)
            r = A.get((load, "v4est", a))
            if r:
                out.append(f"| {a} | {what} | {float(r['tokens_a_verify_step']):.4f} | "
                           f"{float(r['columns_a_step']):.4f} | {float(r['passes_a_step']):.4f} | "
                           f"{float(r['ms_a_step_est']):.2f} | {r['kept']} |")
    print("\n".join(out).lstrip("\n"))


if __name__ == "__main__":
    main()
