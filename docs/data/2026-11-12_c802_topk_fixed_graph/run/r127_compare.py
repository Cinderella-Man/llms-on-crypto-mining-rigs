#!/usr/bin/env python3
"""r127_compare.py: R127's compare.tsv, FN84's comparison of loads request by request (prompt time, the target graph's
reallocations, the reply and the context checkpoints), for any number of loads.

    python3 -I r127_compare.py requests_reallocs.tsv name=requests.tsv[:N] [name=requests.tsv[:N] ...] > compare.tsv

`requests_reallocs.tsv` is FN84's `run/fn84_requests.py` output for the same names (the prompt times are its server
log's, as in FN84's compare.tsv). Each `requests.tsv` is
`taper_replay.py`'s request file of that load; N keeps its first N rows (R60's load a: the first 6 are the task FN84
replays). The first name is the reference: `reply_as_<ref>` and `checkpoints_as_<ref>` say, for each later load,
whether its reply md5 and its checkpoints md5 are the reference's; `-` where the reference has no such request
(the fresh prompts). Rows are matched by request number. Standard library only.
"""
import csv, sys

rr_path, specs = sys.argv[1], sys.argv[2:]
names, reqs = [], {}
for spec in specs:
    name, path = spec.split("=", 1)
    limit = None
    if ":" in path:
        path, n = path.rsplit(":", 1)
        limit = int(n)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    names.append(name)
    reqs[name] = rows[:limit] if limit is not None else rows

realloc = {}
with open(rr_path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        realloc[(row["log"], int(row["req"]))] = row

ref, others = names[0], names[1:]
n = max(len(reqs[k]) for k in names)
cols = ["req", "kind_turn", "prompt_tokens"]
cols += [f"{k}_prompt_ms" for k in names]
cols += [f"{k}_reallocs_{w}" for k in names for w in ("same", "changed")]
cols += [f"{k}_target_reallocs" for k in names] + [f"{k}_draft_reallocs" for k in names]
cols += [f"{k}_gen_md5" for k in names] + [f"{k}_checkpoints_md5" for k in names]
cols += [f"{k}_reply_as_{ref}" for k in others] + [f"{k}_checkpoints_as_{ref}" for k in others]
print("\t".join(cols))


def get(name, i, field, src="req"):
    if src == "req":
        rows = reqs[name]
        return rows[i][field] if i < len(rows) else "-"
    row = realloc.get((name, i))
    return row[field] if row else "-"


def same(a, b):
    return "-" if "-" in (a, b) else ("yes" if a == b else "no")


for i in range(n):
    base = next(reqs[k][i] for k in reversed(names) if i < len(reqs[k]))
    out = [str(i), f"{base['kind']} {base['turn']}", base["prompt_n"]]
    out += [get(k, i, "prompt_ms", "rr") for k in names]
    out += [get(k, i, f"reallocs_{w}", "rr") for k in names for w in ("same", "changed")]
    out += [get(k, i, "target_reallocs", "rr") for k in names] + [get(k, i, "draft_reallocs", "rr") for k in names]
    out += [get(k, i, "gen_md5") for k in names] + [get(k, i, "checkpoints_md5", "rr") for k in names]
    out += [same(get(ref, i, "gen_md5"), get(k, i, "gen_md5")) for k in others]
    out += [same(get(ref, i, "checkpoints_md5", "rr"), get(k, i, "checkpoints_md5", "rr")) for k in others]
    print("\t".join(out))
