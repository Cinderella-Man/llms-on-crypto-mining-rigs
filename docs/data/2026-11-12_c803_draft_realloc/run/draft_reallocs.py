#!/usr/bin/env python3
"""draft_reallocs.py: where in a request the MTP draft's graph reallocations fall, from llama-server logs (-lv 5).

    python3 -I draft_reallocs.py [--draft-max-nodes 1000] name=server.log [name=server.log ...] \
        > requests.tsv 2> summary.tsv

Every reallocation of a graph whose backends stayed the same logs a line `graph reallocation: nodes = N (changed = C)`
(llama_context::sched_graph_realloc), for the target and the draft context alike; the draft's graph has a few hundred
nodes (213 for Swift Qwen3.8-Flash-Next's MTP draft on nine cards), the target's thousands, so a line with at most
--draft-max-nodes nodes is the draft's. A request runs from its `new prompt` line to its `graph reallocs` timing line;
its prompt phase ends at its first sampled token (`n_gen = 1`), and everything after is generation (draft steps,
verification and the draft's catch-up over the accepted tokens).

stdout, one row per request: log, task, prompt tokens (the timing line's), context tokens before and after, generated
tokens, the draft's reallocations in the prompt phase and in generation, the draft's of each kind (same size, changed
size), the target's in each phase, and the seconds from the request's first draft reallocation to its first token.
stderr, one row per log and bucket of prompt tokens: requests, the draft's reallocations (prompt, generation), the
requests with at least one, the distribution of reallocations a request (0 to 5+), and the nodes and changed flags seen.
Standard library only.
"""
import re
import sys
from collections import Counter

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
NEW = re.compile(r"\| task (\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, task\.n_tokens = (\d+)")
CACHED = re.compile(r"\| task (\d+) \| cached n_tokens = (\d+)")
GEN1 = re.compile(r"\| task (\d+) \| .*n_gen = 1, ")
PE = re.compile(r"\| task (\d+) \| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")
EV = re.compile(r"\| task (\d+) \|\s+eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")
END = re.compile(r"\| task (\d+) \|\s+graph reallocs =")
RA = re.compile(r"graph reallocation: nodes = (\d+) \(changed = (\d+)\), splits = (\d+), n_same = (\d+), n_diff = (\d+)")
BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))


def ts(line):
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(g) for g in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


def parse(path, max_nodes):
    reqs, cur, seen = [], None, Counter()
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        m = NEW.search(line)
        if m:
            cur = dict(task=int(m.group(1)), n_tokens=int(m.group(2)), cached=0, prompt=0, gen=0, phase="prompt",
                       d_prompt=0, d_gen=0, d_same=0, d_diff=0, t_prompt=0, t_gen=0, t_first=None, t_gen1=None)
            continue
        if cur is None:
            continue
        m = CACHED.search(line)
        if m and int(m.group(1)) == cur["task"]:
            cur["cached"] = int(m.group(2))
            continue
        m = RA.search(line)
        if m:
            nodes, changed = int(m.group(1)), int(m.group(2))
            if nodes <= max_nodes:
                seen[(nodes, changed)] += 1
                cur["d_" + cur["phase"]] += 1
                cur["d_diff" if changed else "d_same"] += 1
                if cur["t_first"] is None:
                    cur["t_first"] = ts(line)
            else:
                cur["t_" + cur["phase"]] += 1
            continue
        m = GEN1.search(line)
        if m and int(m.group(1)) == cur["task"] and cur["phase"] == "prompt":
            cur["phase"] = "gen"
            cur["t_gen1"] = ts(line)
            continue
        m = PE.search(line)
        if m and int(m.group(1)) == cur["task"]:
            cur["prompt"] = int(m.group(3))
            continue
        m = EV.search(line)
        if m and int(m.group(1)) == cur["task"]:
            cur["gen"] = int(m.group(3))
            continue
        m = END.search(line)
        if m and int(m.group(1)) == cur["task"]:
            reqs.append(cur)
            cur = None
    return reqs, seen


def main(argv):
    max_nodes = 1000
    if argv and argv[0] == "--draft-max-nodes":
        max_nodes, argv = int(argv[1]), argv[2:]
    print("log\ttask\tprompt_tokens\tctx_before\tctx_after\tgen_tokens\tdraft_prompt\tdraft_gen\tdraft_same\t"
          "draft_changed\ttarget_prompt\ttarget_gen\tfirst_draft_realloc_to_first_token_s")
    summ = []
    for spec in argv:
        name, path = spec.split("=", 1)
        reqs, seen = parse(path, max_nodes)
        for r in reqs:
            lead = "" if r["t_first"] is None or r["t_gen1"] is None or r["t_first"] > r["t_gen1"] else \
                f"{r['t_gen1'] - r['t_first']:.3f}"
            print(f"{name}\t{r['task']}\t{r['prompt']}\t{r['cached']}\t{r['n_tokens']}\t{r['gen']}\t{r['d_prompt']}\t"
                  f"{r['d_gen']}\t{r['d_same']}\t{r['d_diff']}\t{r['t_prompt']}\t{r['t_gen']}\t{lead}")
        for lo, hi in BUCKETS:
            b = [r for r in reqs if lo <= r["prompt"] < hi]
            if not b:
                continue
            dist = Counter(min(r["d_prompt"] + r["d_gen"], 5) for r in b)
            summ.append([name, bname(lo, hi), len(b), sum(r["d_prompt"] for r in b), sum(r["d_gen"] for r in b),
                         sum(1 for r in b if r["d_prompt"] + r["d_gen"] > 0),
                         " ".join(f"{k if k < 5 else '5+'}:{dist[k]}" for k in sorted(dist)),
                         sum(r["prompt"] for r in b),
                         " ".join(f"{n}/{c}:{v}" for (n, c), v in sorted(seen.items())) if (lo, hi) == (0, 10 ** 9) else ""])
    sys.stderr.write("log\tprompt_tokens\trequests\tdraft_prompt\tdraft_gen\trequests_with_draft_realloc\t"
                     "reallocs_per_request\tprompt_tokens_sum\tdraft_nodes/changed:count\n")
    for row in summ:
        sys.stderr.write("\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
