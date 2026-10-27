#!/usr/bin/env python3
# Per-depth speed-up tables from llama-pipespec traces: for each job and prompt (chat, code, edit, rewrite) the
# trace-based rate and speed-up over the plain run of the same prompt, with summary.tsv's tok_s and speed-up in brackets.
# It made the corrected Q87/Q160 tables (docs/data/2026-10-24_rig_c7recheck/corr.txt) from the traces in
# docs/data/2026-10-24_rig_q87/ and docs/data/2026-10-24_q88_persistent/.
# Usage: trace_speedups.py <rig_run.sh output folder with <job>-<prompt>.trace and summary.tsv> [...]
import sys, glob, os, csv
def rate(f):
    t = [int(l.split()[0]) for l in open(f).readlines()[1:]]
    return 1e6 * (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 else None
for d in sys.argv[1:]:
    summ = {r["name"]: r for r in csv.DictReader(open(os.path.join(d, "summary.tsv")), delimiter="\t")}
    print(f"\n== {d}")
    print(f"{'job':18s} " + " ".join(f"{p:>22s}" for p in ("chat", "code", "edit", "rewrite")))
    jobs = sorted({os.path.basename(f)[:-6].rsplit("-", 1)[0] for f in glob.glob(os.path.join(d, "*.trace"))},
                  key=lambda j: (j != "plain", j != "d0", j))
    for j in jobs:
        cells = []
        for p in ("chat", "code", "edit", "rewrite"):
            f = os.path.join(d, f"{j}-{p}.trace")
            pf = os.path.join(d, f"plain-{p}.trace")
            if not os.path.exists(f) or not os.path.exists(pf):
                cells.append(f"{'-':>22s}"); continue
            tr, tp = rate(f), rate(pf)
            s, sp = float(summ[f"{j}-{p}"]["tok_s"]), float(summ[f"plain-{p}"]["tok_s"])
            cells.append(f"{tr:5.1f} {tr/tp:4.2f}x ({s:5.1f} {s/sp:4.2f}x)")
        print(f"{j:18s} " + " ".join(cells))
