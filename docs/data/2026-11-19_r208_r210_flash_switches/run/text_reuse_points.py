#!/usr/bin/env python3
"""Where text reuse kept the cache: every `same text` line of a run's llama-server log, placed in the agent's turns.

With LLAMA_SERVER_TEXT_REUSE=1 the server logs, for each request whose token ids differ from the cached ones while
both spell the same text:

    prompt ids differ from the cache at N, same text: K more cached tokens kept (A -> B prompt tokens)

N is the first token where the ids differ, K the cached tokens kept past it, A and B the prompt's length before and
after the cached ids replace the prompt's own. This script puts each line on its request (the i-th `selected slot by`
line of the log is the arm's i-th request in the proxy's rows), says how far into the previous reply N lies (N less
the previous prompt's end, the previous request's cached plus read tokens), whether that is before the reply's tool
call or inside it (the previous request's `sampling = S tokens ... G while the grammar applied` line: the grammar's
part starts at S - G), and whether the client sent the previous reply back as the server gave it (C884's check).

Usage, one argument per run, standard library only:

    python3 -I text_reuse_points.py OUT_TSV LABEL:ARM:PROXY_JSONL:BODIES_GZ:SERVER_LOG ...

Writes OUT_TSV (one row a `same text` line) and prints one summary line a run.
"""
import gzip
import json
import re
import sys

SEL = re.compile(r"selected slot by (LRU|LCP similarity)")
REUSE = re.compile(r"prompt ids differ from the cache at (\d+), same text: (\d+) more cached tokens kept "
                   r"\((\d+) -> (\d+) prompt tokens\)")
SAMP = re.compile(r"sampling = +(\d+) tokens, .*?(\d+) while the grammar applied")


def blocks(path):
    out = []
    for line in open(path, errors="replace"):
        if SEL.search(line):
            out.append(dict(reuse=[], sampled=None, grammar=None))
            continue
        if not out:
            continue
        m = REUSE.search(line)
        if m:
            out[-1]["reuse"].append(tuple(int(x) for x in m.groups()))
        m = SAMP.search(line)
        if m:
            out[-1]["sampled"], out[-1]["grammar"] = int(m.group(1)), int(m.group(2))
    return out


def client_check(prev_body, cur_body, rep):
    pm, m = prev_body["messages"], cur_body["messages"]
    if m[:len(pm)] != pm or len(m) <= len(pm) or m[len(pm)].get("role") != "assistant":
        return "history changed"
    a = m[len(pm)]
    diff = []
    if (a.get("reasoning_content") or "") != (rep.get("reasoning_content") or ""):
        diff.append("reasoning")
    if (a.get("content") or "") != (rep.get("content") or ""):
        diff.append("content")
    tr = [(t.get("name"), t.get("arguments")) for t in rep.get("tool_calls") or []]
    ta = [((t.get("function", t)).get("name"), (t.get("function", t)).get("arguments")) for t in a.get("tool_calls") or []]

    def parsed(x):
        try:
            return json.loads(x) if isinstance(x, str) else x
        except ValueError:
            return x
    if [n for n, _ in tr] != [n for n, _ in ta] or [parsed(x) for _, x in tr] != [parsed(x) for _, x in ta]:
        diff.append("tool calls")
    if diff:
        return "differs: " + ", ".join(diff)
    return "same" if tr == ta else "same, arguments re-serialized"


def run(spec):
    label, arm, proxy, bodies, log = spec.split(":")
    rows = [r for r in (json.loads(x) for x in open(proxy)) if r["tag"].split("/")[0] == arm]
    want = {r["id"] for r in rows}
    body = {}
    for x in gzip.open(bodies, "rt"):
        b = json.loads(x)
        if b["id"] in want:
            body[b["id"]] = b
    bl = blocks(log)
    if len(bl) != len(rows):
        sys.exit(f"{label}: {len(bl)} requests in the server log, {len(rows)} proxy rows of arm {arm}")
    out = []
    for i, (r, b) in enumerate(zip(rows, bl)):
        for n, k, a, bb in b["reuse"]:
            t = dict(run=label, task=r["tag"].split("/")[1], turn=r["tag"].rsplit("/", 1)[1], differ_at=n, kept=k,
                     prompt_before=a, prompt_after=bb, cache_n=r.get("cache_n"), prompt_n=r.get("prompt_n"),
                     prompt_ms=r.get("prompt_ms"), prev_prompt_end="", prev_reply_tokens="", into_reply="",
                     grammar_from="", where="", client="")
            p = rows[i - 1] if i > 0 else None
            if p is not None and p["tag"].rsplit("/", 1)[0] == r["tag"].rsplit("/", 1)[0] and \
                    None not in (p.get("cache_n"), p.get("prompt_n")):
                end = p["cache_n"] + p["prompt_n"]
                t["prev_prompt_end"], t["prev_reply_tokens"] = end, p.get("predicted_n")
                t["into_reply"] = n - end
                pb = bl[i - 1]
                if n < end:
                    t["where"] = "before the previous reply"
                elif pb["sampled"] is not None:
                    g0 = pb["sampled"] - pb["grammar"]
                    t["grammar_from"] = g0
                    t["where"] = "before the tool call" if pb["grammar"] == 0 or n - end < g0 else "in the tool call"
                rep = body[p["id"]]["reply"] or {}
                t["client"] = client_check(body[p["id"]]["body"], body[r["id"]]["body"], rep)
            out.append(t)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rows = []
    for spec in sys.argv[2:]:
        got = run(spec)
        rows.extend(got)
        print(f"{spec.split(':')[0]}: {len(got)} same-text lines, {sum(t['kept'] for t in got)} cached tokens kept")
    keys = ["run", "task", "turn", "differ_at", "kept", "prompt_before", "prompt_after", "cache_n", "prompt_n",
            "prompt_ms", "prev_prompt_end", "prev_reply_tokens", "into_reply", "grammar_from", "where", "client"]
    with open(sys.argv[1], "w") as f:
        f.write("\t".join(keys) + "\n")
        for t in rows:
            f.write("\t".join("" if t[k] is None else str(t[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
