#!/usr/bin/env python3
"""fn84_requests.py: per request of a llama-server log (-lv 5), the target graph's reallocations, the graph sizes at
which they happened, and the request's context checkpoints, so that two loads can be compared request by request.

    python3 -I fn84_requests.py name=server.log[:N] [name=server.log[:N] ...] > requests_reallocs.tsv

N keeps the log's first N requests (R60's load a: the first 6 are the task FN84 replays, pass 0). A request ends at
its print_timing block's `graph reallocs = S / D` line (the target context's counts since the load: S with the graph's
node count unchanged, D changed); the difference to the previous request's is the request's own. Every
`graph reallocation: nodes = N (changed = c)` line logged since the previous request is listed: a graph of more than
1,000 nodes is the target's (Swift Q2_0's prompt or verify graph), a smaller one the MTP draft's. The checkpoints are
the request's `created/restored context checkpoint` lines without their times, as an md5. Standard library only.
"""
import hashlib, re, sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PE = re.compile(r"prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens")
GR = re.compile(r"graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
RA = re.compile(r"graph reallocation: nodes = ([0-9]+) \(changed = ([01])\)")
CK = re.compile(r"((?:created|restored) context checkpoint.*?)(?:, [0-9.]+ ms)?\)\s*$")

print("log\treq\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed\ttarget_reallocs\tdraft_reallocs\t"
      "checkpoints\tcheckpoints_md5")
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    limit = None
    if ":" in path:
        path, n = path.rsplit(":", 1)
        limit = int(n)
    req, prev = 0, (0, 0)
    target, draft, ckpts, pe = [], 0, [], None
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = RA.search(line)
        if m:
            nodes, ch = int(m.group(1)), m.group(2) == "1"
            if nodes > 1000:
                target.append(f"{nodes}{'c' if ch else ''}")
            else:
                draft += 1
            continue
        m = CK.search(line)
        if m:
            ckpts.append(m.group(1))
            continue
        m = PE.search(line)
        if m:
            pe = (float(m.group(1)), int(m.group(2)))
            continue
        m = GR.search(line)
        if m and pe:
            s, d = int(m.group(1)), int(m.group(2))
            md5 = hashlib.md5("\n".join(ckpts).encode()).hexdigest()[:10]
            print(f"{name}\t{req}\t{pe[1]}\t{pe[0]:.1f}\t{s - prev[0]}\t{d - prev[1]}\t{','.join(target) or '-'}\t"
                  f"{draft}\t{len(ckpts)}\t{md5}")
            prev, target, draft, ckpts, pe = (s, d), [], 0, [], None
            req += 1
            if limit is not None and req >= limit:
                break
