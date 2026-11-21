#!/usr/bin/env python3
# C932 (2026-11-21): checks.tsv from check_at.sh's verdicts.tsv: one row per check, each commit named, the trial run
# before the sweep left out (the base again, the same results).
# Usage: python3 -I checks_table.py <verdicts.tsv> <checks.tsv>
import csv
import sys

src, dst = sys.argv[1:3]
role = {
    "1551f5fe1": "v3.0 release tag (miner-llama-v3.0)",
    "eb82916d3": "v4.0 release tag (miner-llama-v4.0)",
    "1f1110c67": "v5 line: upstream b12410 merged",
    "602691153": "v5 line: upstream b12483 merged, #32434 carried",
    "6644c2bc6": "v5 line: the five contributions merged",
    "3dfad876f": "pascal-p106-v5 tip (the base)",
    "f377294d6": "pascal-p106-v3-hybrid tip, as Q550 checked it on 2026-11-03",
    "8c25952e4": "the hybrid branch as merged into the v3 line (second parent of e3b30303d)",
    "e3b30303d": "the merge of the hybrid branch into the v3 line, 2026-11-05 (first parent d29c67616 has the fused conv window)",
    "80d83fa1a": "the fix (pascal-p106-v5-pipe-logits)",
}
order = {"points": 1, "fuse-off": 2, "other-hybrids": 3, "head": 4}
rows = []
with open(src) as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for t in r:
        time, stage, commit, subj, arch, verdict, detail = t[:7]
        if stage == "trial":
            continue
        prc = verdict.startswith("prc")
        check = "llama-pipe-recurrent-check" if prc else "check_identity.sh"
        if prc:
            res = "same" if verdict.endswith("rc=0") else "DIFF"
            det = detail.replace("PIPE-RECURRENT-CHECK ", "")
        else:
            res = "same" if verdict == "SAME" else verdict
            det = detail
        rows.append((order.get(stage, 9), stage, commit, role.get(commit, subj), arch, check, res, det))
with open(dst, "w") as f:
    f.write("stage\tcommit\twhat it is\tmodel (and variables)\tcheck\tresult\tdetail\n")
    for row in sorted(rows, key=lambda x: x[0]):
        f.write("\t".join(row[1:]) + "\n")
print(f"{len(rows)} rows")
