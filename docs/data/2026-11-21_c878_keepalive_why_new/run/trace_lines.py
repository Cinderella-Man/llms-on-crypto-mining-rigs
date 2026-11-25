#!/usr/bin/env python3
# C926: the keep-alive trace's lines (GGML_CUDA_KEEPALIVE_TRACE, C878) of one llama-server load at -lv 5, each placed in
# the request and the step it came from, with where its nearest earlier key was first made on that card.
# A step is read from the slot's lines as short_passes.py reads them: "max possible draft" starts a generation step, the
# draft model's evaluations follow until the "decode: n_batch" line, and from there the main model's pass runs (one
# token plus the drafts of the "generate_draft" line); a request ends with its "print_timing" block. Lines before a
# request's first generation step belong to its prompt. Requests are numbered from 0 in the order they ended; with the
# load's requests CSV (deep_lane's requests-<load>.csv, same order) each one is named by its kind and depth.
# Usage: python3 -I trace_lines.py <server log (.gz or plain)> [requests csv] > trace_lines.tsv
# Columns: line (the log's line number), req, request (kind@depth#pass), task, step (prompt, draft or pass),
# pass_tokens and cells_before (the step's main pass), last_of_reply (the request's last pass), card, key, card_seq (the
# card's new key number), nodes, recorded, near_key, near_seq, near_from (where the nearest key was made: request, step,
# tokens and cells), n_diff, field (the first differing node's field), first (the line's text from its first node on).
# A line naming a missing field ("the trace misses a field") is counted on stderr and listed with step "WARNING".
import csv, gzip, re, sys

TRACE = re.compile(r"ggml_cuda_keepalive_trace: CUDA(\d+): key 0x([0-9a-f]+) new at EAGER=(\d+) \(the card's new key "
                   r"(\d+), (-?\d+) nodes, (\d+) recorded\)(.*)$")
NEAR = re.compile(r"^: nearest earlier key 0x([0-9a-f]+) \(new key (\d+), (-?\d+) nodes, (\d+) recorded\), (\d+) nodes "
                  r"differ(.*)$")


def opener(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


def field_of(first):
    for name in ("the top-k's path", "the attention's pick by the cache's length", "op and type", "no counterpart",
                 "differs (no field found)", "differs (the earlier key's fields", "other fields"):
        if name in first:
            return name
    m = re.search(r"\): ((?:src\[\d+\]\.)?(?:ne|nb)\[\d\]|op_params(?:\[\d+\]| 64-bit word \d+)|src\[\d+\] type) ", first)
    return m.group(1) if m else "?"


def main():
    log = sys.argv[1]
    names = []
    if len(sys.argv) > 2:
        for r in csv.DictReader(open(sys.argv[2])):
            names.append(f"{r['kind']}@{r['depth']}#{r['pass']}")
    rows, warnings = [], 0
    req, task, step, tokens, cells, drafts = 0, None, "prompt", None, None, 0
    n_tokens = None
    for i, line in enumerate(opener(log), 1):
        m = re.search(r"\| task (\d+) \|", line)
        if m:
            task = int(m.group(1))
        if "get_n_draft_: " in line and "max possible draft:" in line:
            step, drafts = "draft", 0
            continue
        m = re.search(r"generate_draft: .*#tokens=(\d+), #draft=(\d+)", line)
        if m and step == "draft":
            n_tokens, drafts = int(m.group(1)), int(m.group(2))
            continue
        m = re.search(r"new n_tokens = (\d+)", line)
        if m:
            n_tokens = int(m.group(1))
            continue
        if " decode: n_batch" in line and step == "draft":
            step, tokens, cells = "pass", 1 + drafts, n_tokens
            continue
        if "print_timing" in line and "prompt eval time" in line:
            req, step, tokens, cells = req + 1, "prompt", None, None
            continue
        if "the trace misses a field" in line:
            warnings += 1
            rows.append({"line": i, "req": req, "task": task, "step": "WARNING", "first": line.strip()})
            continue
        m = TRACE.search(line)
        if not m:
            continue
        card, key, level, seq, nodes, rec, rest = m.groups()
        r = {"line": i, "req": req, "task": task, "step": step, "card": int(card), "key": key, "card_seq": int(seq),
             "nodes": int(nodes), "recorded": int(rec)}
        if step == "pass":
            r["pass_tokens"], r["cells_before"] = tokens, cells
        n = NEAR.match(rest)
        if n:
            r["near_key"], r["near_seq"], r["n_diff"] = n.group(1), int(n.group(2)), int(n.group(5))
            tail = n.group(6)
            r["first"] = tail[len("; first "):] if tail.startswith("; first ") else tail.lstrip("; ")
            r["field"] = field_of(r["first"].split("; then ")[0])
        else:
            r["first"] = rest.lstrip(": ")
            r["field"] = "-"
        rows.append(r)

    # where each key was first made, per card
    made = {}
    for r in rows:
        if "key" in r:
            made.setdefault((r["card"], r["key"]), r)
    # each request's last main pass: the line of its last "decode: n_batch" after a draft step
    last_pass = {}
    step, req, task, drafts = "prompt", 0, None, 0
    for i, line in enumerate(opener(log), 1):
        m = re.search(r"\| task (\d+) \|", line)
        if m:
            task = int(m.group(1))
        if "max possible draft:" in line:
            step = "draft"
        elif " decode: n_batch" in line and step == "draft":
            step = "pass"
            last_pass[req] = i
        elif "print_timing" in line and "prompt eval time" in line:
            req, step = req + 1, "prompt"

    def name(q):
        return names[q] if q < len(names) else f"req{q}"

    def origin(o):
        s = f"{name(o['req'])} {o['step']}"
        if o["step"] == "pass":
            s += f" {o['pass_tokens']}t@{o['cells_before']}"
        return s

    cols = ["line", "req", "request", "task", "step", "pass_tokens", "cells_before", "last_of_reply", "card", "key",
            "card_seq", "nodes", "recorded", "near_key", "near_seq", "near_from", "n_diff", "field", "first"]
    print("\t".join(cols))
    for r in rows:
        r["request"] = name(r["req"])
        if r.get("step") == "pass":
            # the pass's decode line is the request's last one: the trace line comes after it, before the request ends
            lp = last_pass.get(r["req"])
            r["last_of_reply"] = "yes" if lp is not None and r["line"] > lp else "no"
        if "near_key" in r:
            o = made.get((r["card"], r["near_key"]))
            r["near_from"] = origin(o) if o is not None else "before the trace"
        print("\t".join(str(r.get(c, "")) for c in cols))
    print(f"trace lines: {sum(1 for r in rows if 'key' in r)}; warnings that the trace misses a field: {warnings}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
