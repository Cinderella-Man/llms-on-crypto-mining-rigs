#!/usr/bin/env python3
"""C966 (Q1696, LLMRig point 13): two conversations in one KV cache, so that removals leave holes between the two
slots' cells. Sent to llama-server's /completion endpoint with `-np 2 -kvu` (two slots, one stream of cells), each
request pinned to its slot (`id_slot`), one request at a time. Greedy (temperature 0, top-k 1), 128 tokens, the token
ids returned; nothing timed is written. The requests are C952's (docs/data/2026-11-21_c952_kv_seq_rm_gpu/run/session.py),
sent to slot 0 and slot 1 in turn, plus a short prompt on slot 0 that leaves a gap below slot 1's cells and four
requests that use it:

   1 s0-fresh   about DEPTH0 tokens of docs/switches.md from its first line, question A, cache_prompt off
   2 s1-fresh   about DEPTH1 tokens of the same file from line START1, question A, cache_prompt off
   3 s0-turn    fresh's prompt, its reply and question B: the cache reused as it is
   4 s1-turn
   5 s0-edit    a changed history inside fresh's reply: a checkpoint restored, the cells after it removed
   6 s1-edit
   7 s0-cancel  edit's prompt and question D, streamed, ignore_eos, 400 tokens; the client hangs up after CANCEL_AT
                chunks. Only those chunks are compared: how many tokens the server makes before it sees the hang-up
                varies, so the same slot's next request (after) follows at once and removes them before the other
                slot writes, and no cell of the other slot depends on that number
   8 s0-after   cancel's prompt, a withdrawal note and question E
   9 s1-cancel
  10 s1-after
  11 s0-short   about SHORT tokens of the file from its first line, question A, cache_prompt off: slot 0's cells are
                freed and only a few refilled, so a gap opens below slot 1's cells and the head sits at its bottom
  12 s1-back    slot 1's document with question F: a removal of slot 1's tail, all of it above the head. The fork's
                rule puts the head on slot 1's first removed cell; upstream's (LLAMA_KV_HEAD_TAIL=0) lowers the head
                only to a freed cell below it, so the head stays in slot 0's gap and slot 1's re-read fills the gap
  13 s0-sturn   short's prompt, its reply and question B: slot 0 appends
  14 s0-sedit   a changed history inside short's reply: slot 0 removes its tail, the head goes into slot 0's region
  15 s1-bturn   back's prompt, its reply and question B: slot 1 appends where the head is, which can be below its
                earlier cells
  16 s1-bedit   a changed history inside back's reply: a removal of slot 1's tail across both parts, where the first
                removed position's cell and the lowest freed cell can differ

Usage: python3 -I session2.py <server URL> <output folder> <round tag> <text file> [DEPTH0 (6000)] [DEPTH1 (5000)]
       [START1 (200)] [SHORT (1000)] [CANCEL_AT (24)] [N (16): send only the first N requests]
Writes <out>/<round>-<request>.json (the server's reply; for a cancel the chunks read) and appends one line per request
to <out>/session.tsv: round, request, prompt tokens read (timings.prompt_n), cached tokens used (timings.cache_n),
generated tokens, the md5 of the token ids, the md5 of the text, drafted tokens, accepted drafts, the stop reason and
the reply's first 60 characters (JSON-escaped).
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT, ROUND, TEXT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4]
ARG = [int(x) for x in sys.argv[5:]]
DEPTH0, DEPTH1, START1, SHORT, CANCEL_AT, N = (ARG + [6000, 5000, 200, 1000, 24, 16][len(ARG):])[:6]

QA = "\n\nQuestion: which switches above change how prompt checkpoints are taken? Answer in a few lines.\nAnswer:"
QB = "\n\nQuestion: which of those are on in a release preset?\nAnswer:"
QC = "\n\nQuestion: name three switches above that change the KV cache, one line each.\nAnswer:"
QD = "\n\nQuestion: write a shell command that prints every switch name in the table above.\nAnswer:"
QE = "\n\n(The previous answer was withdrawn.)\n\nQuestion: name one switch above that changes speculative decoding.\nAnswer:"
QF = "\n\nQuestion: summarise the first table above in two sentences.\nAnswer:"
MARK = " [...]\n(the rest of this answer was cut)"
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1}
SENT = 0


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def n_tokens(text):
    return len(post("/tokenize", {"content": text, "add_special": True}, 120)["tokens"])


def document(start, depth):
    with open(TEXT, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)[start:]
    lo, hi = 1, len(lines)  # the most whole lines whose prompt fits in depth tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens("".join(lines[:mid]) + QA) <= depth:
            lo = mid
        else:
            hi = mid - 1
    return "".join(lines[:lo])


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def record(name, reply, toks, text):
    t = reply.get("timings") or {}
    row = [ROUND, name, str(t.get("prompt_n", "")), str(t.get("cache_n", "")), str(len(toks)),
           md5(",".join(str(x) for x in toks)) if toks else "", md5(text) if toks else "",
           str(t.get("draft_n", "")), str(t.get("draft_n_accepted", "")), str(reply.get("stop_type", "")),
           json.dumps(text[:60] if toks else str(reply.get("error", ""))[:60])]
    with open(os.path.join(OUT, "session.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row[:10]), flush=True)


def more():
    global SENT
    SENT += 1
    return SENT <= N


def complete(slot, name, prompt, cache=True):
    if not more():
        return ""
    name = f"s{slot}-{name}"
    body = dict(GREEDY, prompt=prompt, n_predict=128, cache_prompt=cache, return_tokens=True, id_slot=slot)
    try:
        reply = post("/completion", body)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{ROUND}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    text = str(reply.get("content", ""))
    record(name, reply, toks, text)
    return text


def cancel(slot, name, prompt):
    if not more():
        return
    name = f"s{slot}-{name}"
    body = dict(GREEDY, prompt=prompt, n_predict=400, cache_prompt=True, return_tokens=True, stream=True,
                ignore_eos=True, id_slot=slot)
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, toks, text, err = [], [], "", ""
    try:
        r = urllib.request.urlopen(req, timeout=900)
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            c = json.loads(line[6:])
            chunks.append(c)
            toks += c.get("tokens") or []
            text += str(c.get("content", ""))
            if len(chunks) >= CANCEL_AT or c.get("stop"):
                break
        r.close()  # the hang-up: the server stops the task at its next send
    except Exception as e:
        err = str(e)
    with open(os.path.join(OUT, f"{ROUND}-{name}.json"), "w") as f:
        json.dump({"chunks": chunks, "error": err}, f)
    record(name, {"stop_type": f"cancelled after {len(chunks)} chunks", "error": err}, toks, text)


os.makedirs(OUT, exist_ok=True)
doc0, doc1, docs = document(0, DEPTH0), document(START1, DEPTH1), document(0, SHORT)
with open(os.path.join(OUT, f"{ROUND}-document.md5"), "w") as f:
    for k, d in (("doc0", doc0), ("doc1", doc1), ("short", docs)):
        f.write(f"{md5(d)}  {k} {len(d)} characters\n")

p0, p1 = doc0 + QA, doc1 + QA
r0 = complete(0, "fresh", p0, cache=False)
r1 = complete(1, "fresh", p1, cache=False)
complete(0, "turn", p0 + r0 + QB)
complete(1, "turn", p1 + r1 + QB)
e0 = p0 + r0[: len(r0) // 3] + MARK + QC
e1 = p1 + r1[: len(r1) // 3] + MARK + QC
c0 = complete(0, "edit", e0)
c1 = complete(1, "edit", e1)
cancel(0, "cancel", e0 + c0 + QD)
complete(0, "after", e0 + c0 + QD + QE)
cancel(1, "cancel", e1 + c1 + QD)
complete(1, "after", e1 + c1 + QD + QE)
ps = docs + QA
rs = complete(0, "short", ps, cache=False)
pb = doc1 + QF
rb = complete(1, "back", pb)
complete(0, "sturn", ps + rs + QB)
complete(0, "sedit", ps + rs[: len(rs) // 3] + MARK + QC)
complete(1, "bturn", pb + rb + QB)
complete(1, "bedit", pb + rb[: len(rb) // 3] + MARK + QC)
