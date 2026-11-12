#!/usr/bin/env python3
"""task_acceptance.py RES ARM...: the MTP draft's acceptance per task of an agent check, from the server's metrics.

bench/agent_check saves the server's /metrics before and after every attempt (RES/<arm>/<task>/a1/metrics_before.txt
and metrics_after.txt). The difference of llamacpp:spec_decode_num_accepted_tokens_total and
llamacpp:spec_decode_num_draft_tokens_total over an attempt is that task's accepted and drafted tokens; their ratio is
its acceptance. Prints one row per task and arm, then each arm's total and its total without the tasks in SKIP_TASKS
(comma-separated). Standard library only.

    SKIP_TASKS=bugfix-javascript-complex-numbers python3 -I task_acceptance.py <client folder>/res gains v4c2
"""
import glob
import os
import sys

ACC, DRAFT = "llamacpp:spec_decode_num_accepted_tokens_total", "llamacpp:spec_decode_num_draft_tokens_total"


def read(path):
    d = {}
    for line in open(path):
        if line.startswith("llamacpp:"):
            k, v = line.rsplit(" ", 1)
            d[k] = float(v)
    return d


if len(sys.argv) < 3:
    sys.exit(__doc__)
res, arms = sys.argv[1], sys.argv[2:]
skip = set(filter(None, os.environ.get("SKIP_TASKS", "").split(",")))
print("arm\ttask_id\taccepted\tdrafted\tacceptance")
for arm in arms:
    tot = [0, 0]
    kept = [0, 0]
    for att in sorted(glob.glob(os.path.join(res, arm, "*", "a1"))):
        task = os.path.basename(os.path.dirname(att))
        b, a = read(os.path.join(att, "metrics_before.txt")), read(os.path.join(att, "metrics_after.txt"))
        acc, dr = int(a[ACC] - b.get(ACC, 0)), int(a[DRAFT] - b.get(DRAFT, 0))
        print(f"{arm}\t{task}\t{acc}\t{dr}\t{acc / dr:.3f}" if dr else f"{arm}\t{task}\t{acc}\t{dr}\t")
        tot[0] += acc
        tot[1] += dr
        if task not in skip:
            kept[0] += acc
            kept[1] += dr
    print(f"{arm}\tall\t{tot[0]}\t{tot[1]}\t{tot[0] / tot[1]:.3f}")
    if skip:
        print(f"{arm}\twithout {','.join(sorted(skip))}\t{kept[0]}\t{kept[1]}\t{kept[0] / kept[1]:.3f}")
