#!/usr/bin/env python3
"""Reply re-reads of agent-check runs: where each one happens and what the client sent back (C884, Q1598).

A reply re-read is a turn whose cached prefix stops short of the previous reply's end, so the server reads that reply
again as prompt (report.py's cache paths "reply re-read" and "older checkpoint", with its SLACK of 16 tokens). For
every such turn this script checks, from the proxy's recorded bodies, whether the client sent the previous reply back
as the server gave it (reasoning, content, tool-call names and arguments), and, when the run's server log is given,
where inside that reply the new prompt stopped matching the cached tokens and whether that point lies before the tool
call (free text) or inside it (the part generated under the tool-call grammar).

Usage, one argument per run, standard library only:

    python3 -I reply_rereads.py OUT_PREFIX LABEL:ARM:PROXY_JSONL:BODIES_GZ[:SERVER_LOG] ...

LABEL names the run in the tables, ARM is the arm's tag prefix in the proxy's rows (`v4r/<task>/a1/t1/<turn>`),
PROXY_JSONL the proxy's request rows, BODIES_GZ its `proxy.jsonl.bodies.gz` (the request body and the parsed reply of
each request), SERVER_LOG the run's llama-server log (optional). Writes OUT_PREFIX_runs.tsv (one row a run) and
OUT_PREFIX_turns.tsv (one row a re-read turn).

From the server log: the i-th `selected slot by ...` line is the arm's i-th request. Its `f_keep` (rounded to a
thousandth) times the cached tokens of the previous request's `stop processing: n_tokens` line is the common prefix,
within 0.0005 x those tokens; less the previous prompt's end (those tokens less the previous `eval time` tokens) it is
the point inside the previous reply. The previous request's `sampling = S tokens ... G while the grammar applied` line
puts the start of the grammar's part of that reply at token S - G.
"""
import gzip
import json
import re
import sys

SLACK = 16  # bench/agent_check/report.py's

SEL = re.compile(r"selected slot by (LRU|LCP similarity)(?:, f_sim_best = [0-9.]+ \(> [0-9.]+ thold\), f_keep = ([0-9.]+))?")
REL = re.compile(r"stop processing: n_tokens = (\d+)")
EVAL = re.compile(r"\|\s+eval time = +[0-9.]+ ms / +(\d+) tokens")
SAMP = re.compile(r"sampling = +(\d+) tokens, .*?(\d+) while the grammar applied")


def server_blocks(path):
    """Per request in log order: f_keep (None for LRU), released n_tokens, eval tokens, sampled, grammar tokens."""
    out = []
    for line in open(path, errors="replace"):
        m = SEL.search(line)
        if m:
            out.append(dict(f_keep=float(m.group(2)) if m.group(2) else None, n_tokens=None, gen=None,
                            sampled=None, grammar=None))
            continue
        if not out:
            continue
        b = out[-1]
        m = EVAL.search(line)
        if m:
            b["gen"] = int(m.group(1))
        m = SAMP.search(line)
        if m:
            b["sampled"], b["grammar"] = int(m.group(1)), int(m.group(2))
        m = REL.search(line)
        if m:
            b["n_tokens"] = int(m.group(1))
    return out


def tool_calls_of_reply(rep):
    return [(t.get("name"), t.get("arguments")) for t in rep.get("tool_calls") or []]


def tool_calls_of_msg(msg):
    out = []
    for t in msg.get("tool_calls") or []:
        f = t.get("function", t)
        out.append((f.get("name"), f.get("arguments")))
    return out


def parsed(args):
    try:
        return json.loads(args) if isinstance(args, str) else args
    except ValueError:
        return args


def client_check(prev_body, cur_body, rep):
    """What the client sent back of the previous reply: 'same', 'same, arguments re-serialized' or what differs."""
    pm, m = prev_body["messages"], cur_body["messages"]
    if m[:len(pm)] != pm or len(m) <= len(pm) or m[len(pm)].get("role") != "assistant":
        return "history changed"
    a = m[len(pm)]
    diff = []
    if (a.get("reasoning_content") or "") != (rep.get("reasoning_content") or ""):
        diff.append("reasoning")
    if (a.get("content") or "") != (rep.get("content") or ""):
        diff.append("content")
    tr, ta = tool_calls_of_reply(rep), tool_calls_of_msg(a)
    if [n for n, _ in tr] != [n for n, _ in ta] or [parsed(x) for _, x in tr] != [parsed(x) for _, x in ta]:
        diff.append("tool calls")
    if diff:
        return "differs: " + ", ".join(diff)
    return "same" if tr == ta else "same, arguments re-serialized"


def reply_text(rep):
    return (rep.get("reasoning_content") or "") + (rep.get("content") or "") + "".join(
        x or "" for _, x in tool_calls_of_reply(rep))


def run(spec):
    parts = spec.split(":")
    label, arm, proxy, bodies = parts[:4]
    log = parts[4] if len(parts) > 4 else None
    rows = [r for r in (json.loads(x) for x in open(proxy)) if r["tag"].split("/")[0] == arm]
    want = {r["id"] for r in rows}
    body = {}
    for x in gzip.open(bodies, "rt"):
        b = json.loads(x)
        if b["id"] in want:
            body[b["id"]] = b
    blocks = server_blocks(log) if log else None
    if blocks is not None and len(blocks) != len(rows):
        sys.exit(f"{label}: {len(blocks)} requests in the server log, {len(rows)} proxy rows of arm {arm}")
    s = dict(run=label, arm=arm, requests=len(rows), continuation_turns=0, reply_tokens_before=0, reread_turns=0,
             reread_tokens=0, reread_prompt_s=0.0, client_same=0, client_reserialized=0, client_differs=0,
             replies_untrimmed_reasoning=0, before_tool_call=0, in_tool_call=0, at_boundary=0,
             turns_after_non_ascii_reply=0, rereads_after_non_ascii_reply=0)
    turns = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if prev["tag"].rsplit("/", 1)[0] != cur["tag"].rsplit("/", 1)[0]:
            continue  # a task's first turn
        if None in (prev.get("cache_n"), prev.get("prompt_n"), cur.get("cache_n")):
            continue
        gen = prev.get("predicted_n") or 0
        s["continuation_turns"] += 1
        s["reply_tokens_before"] += gen
        rep = body[prev["id"]]["reply"] or {}
        non_ascii = any(ord(c) > 127 for c in reply_text(rep))
        s["turns_after_non_ascii_reply"] += non_ascii
        prev_prompt = prev["cache_n"] + prev["prompt_n"]
        if cur["cache_n"] >= prev_prompt + gen - SLACK:
            continue
        s["rereads_after_non_ascii_reply"] += non_ascii
        chk = client_check(body[prev["id"]]["body"], body[cur["id"]]["body"], rep)
        rc = rep.get("reasoning_content") or ""
        s["reread_turns"] += 1
        s["reread_tokens"] += max(0, min(gen, cur.get("prompt_n") or 0, prev_prompt + gen - cur["cache_n"]))
        s["reread_prompt_s"] += (cur.get("prompt_ms") or 0) / 1000
        s["client_same" if chk == "same" else "client_reserialized" if chk.startswith("same,") else "client_differs"] += 1
        s["replies_untrimmed_reasoning"] += rc != rc.strip()
        t = dict(run=label, task=cur["tag"].split("/")[1], turn=cur["tag"].rsplit("/", 1)[1], reply_tokens=gen,
                 cached=cur["cache_n"], client=chk, reasoning_trimmed="yes" if rc == rc.strip() else "no",
                 non_ascii_chars=sum(ord(c) > 127 for c in reply_text(rep)), n_tool_calls=len(tool_calls_of_reply(rep)),
                 lcp_into_reply="", lcp_err="", grammar_from="", where="")
        if blocks is not None:
            pb, cb = blocks[i - 1], blocks[i]
            if cb["f_keep"] is not None and pb["n_tokens"] and pb["gen"] is not None:
                n = pb["n_tokens"]
                off = cb["f_keep"] * n - (n - pb["gen"])
                err = 0.0005 * n
                t["lcp_into_reply"], t["lcp_err"] = round(off), round(err, 1)
                if pb["sampled"] is not None:
                    g0 = pb["sampled"] - pb["grammar"]
                    t["grammar_from"] = g0
                    if pb["grammar"] == 0 or off + err < g0:
                        t["where"] = "before the tool call"
                    elif off - err >= g0:
                        t["where"] = "in the tool call"
                    else:
                        t["where"] = "at the tool call's start"
                    s[{"before the tool call": "before_tool_call", "in the tool call": "in_tool_call"}.get(
                        t["where"], "at_boundary")] += 1
        turns.append(t)
    if blocks is None:
        for k in ("before_tool_call", "in_tool_call", "at_boundary"):
            s[k] = ""
    s["reread_prompt_s"] = round(s["reread_prompt_s"], 1)
    s["rereads_per_1000_reply_tokens"] = round(1000 * s["reread_turns"] / max(1, s["reply_tokens_before"]), 3)
    return s, turns


def write(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = list(rows[0])
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs, turns = [], []
    for spec in sys.argv[2:]:
        s, t = run(spec)
        runs.append(s)
        turns.extend(t)
    write(sys.argv[1] + "_runs.tsv", runs)
    write(sys.argv[1] + "_turns.tsv", turns)
    for s in runs:
        print(f"{s['run']}: {s['reread_turns']} re-read turns of {s['continuation_turns']} "
              f"({s['rereads_per_1000_reply_tokens']} per 1,000 reply tokens), client sent the reply back unchanged "
              f"on {s['client_same'] + s['client_reserialized']}")


if __name__ == "__main__":
    main()
