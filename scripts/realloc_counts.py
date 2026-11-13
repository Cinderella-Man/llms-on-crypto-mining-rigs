#!/usr/bin/env python3
"""realloc_counts.py: the target graph's reallocations per request, from llama-server logs, and the loads whose
switch adds them (Q1408).

    python3 -I realloc_counts.py name=server.log [name=server.log ...] > reallocs.tsv
    python3 -I realloc_counts.py --requests name=server.log ... > requests_reallocs.tsv
    python3 -I realloc_counts.py --line name=server.log ...
    python3 -I realloc_counts.py --marks 'ladder' base=a.log graphs=b.log kvonly=c.log base2=d.log > realloc_marks.tsv
    python3 -I realloc_counts.py --marks 'dp=prof,prof-2 tracedp=trace' prof=... dp=... > realloc_marks.tsv

Every request's print_timing block logs its prompt (`prompt eval time = X ms / N tokens`), the graphs reused and the
target context's running count of graph reallocations (`graph reallocs = S / D`: S with the graph's size unchanged,
D changed; both count since the load). The line is logged at the server's default verbosity on Miner Llama's v4 builds.
The difference to the previous request's count is that request's reallocations. A server start in the log
(llama-server's `llama_server: initializing`, or rig_server.sh's `=== ` line, as when an agent-check arm is resumed
into the same log) or a count lower than the one before starts the count from zero again. A reallocation makes the
scheduler wait for every card before it moves the buffers, so a layer-split prompt call loses the overlap of its
micro-batches there. Only the target's context is counted: the MTP draft's own graph has its own count (C803, Q1407).

Outputs (one at a time):
- default: per log and bucket of prompt tokens (R90's buckets: under 512, 512-1,023, 1,024-1,999, 2,000 and more, and
  all), the requests, their prompt seconds, the reallocations of each kind and the requests with at least one: the
  table of docs/data/2026-11-12_r60_taper_v4/run/realloc_counts.py, unchanged;
- --requests: one row per request (its order in the log, the server's task id, prompt tokens and ms, the
  reallocations of each kind);
- --line: one line per log, as a driver's summary prints it;
- --marks SPEC: per log, the requests with a reallocation against its references' and a mark. SPEC holds
  `load=ref[,ref...]` pairs and the word `ladder`: each load against the loads of the rung before it, where a rung is
  a name and its repeats (the name plus `2` or `-2`, e.g. base2 or a-2) and the rungs follow the order of the logs; a
  pair overrides the ladder for its load. Marks: `reallocates` (the load has requests with a reallocation and no
  reference has any: the switch adds them), `reallocates more` (both have, and a larger share of the load's requests),
  `as its line` (both have, the load's share no larger), `none` (no request with one), `-` (no reference: the count
  alone).
Standard library only.
"""
import re
import sys

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))
PE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GR = re.compile(r"graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
GU = re.compile(r"graphs reused =\s*([0-9]+)")
TASK = re.compile(r"\| task ([0-9]+) \|")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
START = re.compile(r"llama_server: initializing|^=== ")


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


def requests(path):
    """The log's requests: dicts of prompt ms and tokens, graphs reused, reallocations of each kind, task id."""
    reqs, cur, prev = [], None, (0, 0)
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        if START.search(line):
            prev, cur = (0, 0), None
            continue
        m = PE.search(line)
        if m:
            t = TASK.search(line)
            cur = dict(ms=float(m.group(1)), n=int(m.group(2)), reused=0, same=0, diff=0,
                       task=t.group(1) if t else "-")
            continue
        if cur is None:
            continue
        m = GU.search(line)
        if m:
            cur["reused"] = int(m.group(1))
            continue
        m = GR.search(line)
        if m:
            s, d = int(m.group(1)), int(m.group(2))
            if s < prev[0] or d < prev[1]:
                prev = (0, 0)
            cur["same"], cur["diff"] = s - prev[0], d - prev[1]
            prev = (s, d)
            reqs.append(cur)
            cur = None
    return reqs


def total(reqs):
    return (len(reqs), sum(1 for r in reqs if r["same"] + r["diff"] > 0), sum(r["same"] for r in reqs),
            sum(r["diff"] for r in reqs))


def stem(name, names):
    """A repeat's rung: base2 and base-2 belong to base when base is a load too."""
    for suffix in ("-2", "2"):
        if name.endswith(suffix) and name[:-len(suffix)] in names:
            return name[:-len(suffix)]
    return name


def references(names, spec):
    words = spec.split()
    refs = {}
    if "ladder" in words:
        rungs = []
        for n in names:
            s = stem(n, names)
            if s not in rungs:
                rungs.append(s)
        for n in names:
            i = rungs.index(stem(n, names))
            if i > 0:
                refs[n] = [m for m in names if stem(m, names) == rungs[i - 1]]
    for w in words:
        if "=" in w:
            load, rs = w.split("=", 1)
            refs[load] = [r for r in rs.split(",") if r]
    return refs


def mark(load, refs):
    n, w = load[0], load[1]
    if not refs:
        return "-"
    if w == 0:
        return "none"
    if all(r[1] == 0 for r in refs):
        return "reallocates"
    if all(w / max(n, 1) > r[1] / max(r[0], 1) for r in refs):
        return "reallocates more"
    return "as its line"


def main(argv):
    mode, spec = "buckets", ""
    while argv and argv[0].startswith("--"):
        a = argv.pop(0)
        if a == "--requests":
            mode = "requests"
        elif a == "--line":
            mode = "line"
        elif a == "--marks":
            mode, spec = "marks", argv.pop(0)
        else:
            sys.exit(f"unknown option {a}")
    logs = [a.split("=", 1) for a in argv]
    data = {name: requests(path) for name, path in logs}
    names = [name for name, _ in logs]
    if mode == "buckets":
        print("log\tprompt_tokens\trequests\tprompt_s\treallocs_same\treallocs_changed\trequests_with_realloc\t"
              "graphs_reused")
        for name in names:
            reqs = data[name]
            for lo, hi in BUCKETS:
                b = [r for r in reqs if lo <= r["n"] < hi]
                if not b:
                    continue
                print(f"{name}\t{bname(lo, hi)}\t{len(b)}\t{sum(r['ms'] for r in b) / 1000:.1f}\t"
                      f"{sum(r['same'] for r in b)}\t{sum(r['diff'] for r in b)}\t"
                      f"{sum(1 for r in b if r['same'] + r['diff'] > 0)}\t{sum(r['reused'] for r in b)}")
    elif mode == "requests":
        print("log\treq\ttask\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed")
        for name in names:
            for i, r in enumerate(data[name]):
                print(f"{name}\t{i}\t{r['task']}\t{r['n']}\t{r['ms']:.1f}\t{r['same']}\t{r['diff']}")
    elif mode == "line":
        for name in names:
            n, w, s, d = total(data[name])
            print(f"{name}: graph reallocations: {w} of {n} requests ({s} at the same size, {d} changed)")
    else:
        refs = references(names, spec)
        print("load\tagainst\trequests\trequests_with_realloc\treallocs_same\treallocs_changed\t"
              "against_requests_with_realloc\tmark")
        for name in names:
            t = total(data[name])
            rs = [r for r in refs.get(name, []) if r in data]
            rt = [total(data[r]) for r in rs]
            against = ",".join(rs) or "-"
            ref_w = ",".join(f"{x[1]} of {x[0]}" for x in rt) or "-"
            print(f"{name}\t{against}\t{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{ref_w}\t{mark(t, rt)}")


if __name__ == "__main__":
    main(sys.argv[1:])
