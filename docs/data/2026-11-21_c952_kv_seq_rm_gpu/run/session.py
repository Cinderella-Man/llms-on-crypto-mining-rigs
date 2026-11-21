#!/usr/bin/env python3
"""C952 (Q1687 point 6): requests that make llama-server remove cells from the KV cache, against its /completion
endpoint on one slot (-np 1). Every request but the first reuses the slot's cache (cache_prompt on), so the server
keeps the shared prefix and removes the rest, through a context checkpoint on these hybrid models (Gated DeltaNet
layers cannot be cut back, so the server restores the newest checkpoint below the first differing token, removes every
KV cell after it and reads the rest of the prompt again). Greedy (temperature 0, top-k 1), 128 tokens, the token ids
returned; nothing timed is written.

  fresh    about DEPTH tokens of this repository's docs/switches.md (whole lines from its start, cut with the server's
           /tokenize) and question A, cache_prompt off: the slot's cache cleared, the prompt read whole
  turn     fresh's prompt, its reply and question B: the cache reused as it is
  edit     a changed history: fresh's prompt, the first third of its reply, a marker and question C, so the prompt leaves
           the cache inside the reply: a checkpoint restored, the cells after it removed
  cancel   edit's prompt and question D, streamed, ignore_eos, 400 tokens; the client hangs up after CANCEL_AT chunks
           (a cancel mid-reply: the server stops at its next send). Its full length depends on when the server sees the
           hang-up, so only the first CANCEL_AT chunks are compared
  after    cancel's prompt, a withdrawal note and question E: the cancelled reply removed from the cache
  back     fresh's document with question F only: the slot goes back to the document's end (a checkpoint there, if one
           of the newest is old enough, or the whole prompt read again)

Usage: python3 -I session.py <server URL> <output folder> <round tag> <text file> [DEPTH (6000)] [CANCEL_AT (24)]
Writes <out>/<round>-<request>.json (the server's reply; for cancel the chunks read) and appends one line per request
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
DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
CANCEL_AT = int(sys.argv[6]) if len(sys.argv) > 6 else 24

QA = "\n\nQuestion: which switches above change how prompt checkpoints are taken? Answer in a few lines.\nAnswer:"
QB = "\n\nQuestion: which of those are on in a release preset?\nAnswer:"
QC = "\n\nQuestion: name three switches above that change the KV cache, one line each.\nAnswer:"
QD = "\n\nQuestion: write a shell command that prints every switch name in the table above.\nAnswer:"
QE = "\n\n(The previous answer was withdrawn.)\n\nQuestion: name one switch above that changes speculative decoding.\nAnswer:"
QF = "\n\nQuestion: summarise the first table above in two sentences.\nAnswer:"
MARK = " [...]\n(the rest of this answer was cut)"
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1}


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def n_tokens(text):
    return len(post("/tokenize", {"content": text, "add_special": True}, 120)["tokens"])


def document():
    with open(TEXT, encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)
    lo, hi = 1, len(lines)  # the most whole lines whose prompt fits in DEPTH tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens("".join(lines[:mid]) + QA) <= DEPTH:
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


def complete(name, prompt, cache=True):
    body = dict(GREEDY, prompt=prompt, n_predict=128, cache_prompt=cache, return_tokens=True)
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


def cancel(name, prompt):
    body = dict(GREEDY, prompt=prompt, n_predict=400, cache_prompt=True, return_tokens=True, stream=True,
                ignore_eos=True)
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
doc = document()
with open(os.path.join(OUT, f"{ROUND}-document.md5"), "w") as f:
    f.write(md5(doc) + f"  {len(doc)} characters\n")
p_fresh = doc + QA
r_a = complete("fresh", p_fresh, cache=False)
complete("turn", p_fresh + r_a + QB)
p_edit = p_fresh + r_a[: len(r_a) // 3] + MARK + QC
r_c = complete("edit", p_edit)
p_cancel = p_edit + r_c + QD
cancel("cancel", p_cancel)
complete("after", p_cancel + QE)
complete("back", doc + QF)
