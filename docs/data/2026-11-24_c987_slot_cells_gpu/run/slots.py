#!/usr/bin/env python3
"""C987 (Q1720, LLMRig point 17): one fixed greedy request on one slot after a given history on the other slot.

Derived from C966's session2.py (docs/data/2026-11-23_c966_kv_head_tail_gpu/run/session2.py): the same documents
(parts of docs/switches.md), the same questions, the same greedy settings and /completion calls, each request pinned to
its slot (`id_slot`), one request at a time. Nothing timed is written.

History, sent to slot 0 first:
  none    nothing
  long    s0-fresh: about DEPTH0 tokens of the file from its first line and question A, cache_prompt off;
          s0-turn: that prompt, its reply and question B, the cache reused as it is. Slot 0 then holds one
          conversation of about DEPTH0 + 290 cells, and the cache's head sits just after it
  short   s0-fresh as above; then s0-short: about SHORT tokens of the file and question A, cache_prompt off (slot 0's
          cells are freed and only the first few refilled); then s0-sturn: that prompt, its reply and question B.
          Slot 0 then holds about SHORT + 290 cells
Then the fixed request on slot SLOT (1 with two slots; 0 with one): about DEPTH1 tokens of the file from line START1
and question A, cache_prompt off, greedy, NPRED tokens, the token ids returned:
  fix-1   the request
  fix-2   the same again: the slot's cells are removed and the prompt read again (the fork's head rule puts it back
          into the same cells): a repeat inside the load
  fix-p   the same with n_probs = 2: the server then samples on the host (n_probs turns the card's sampling chain off
          for that request) and returns each step's two most likely tokens with their log-probabilities

Usage: python3 -I slots.py <server URL> <output folder> <tag> <text file> <history> <slot> [DEPTH0 (6000)]
       [DEPTH1 (5000)] [START1 (200)] [SHORT (1000)] [NPRED (256)]
Writes <out>/<tag>-<request>.json (the server's reply) and appends one line per request to <out>/session.tsv: tag,
request, slot, prompt tokens read (timings.prompt_n), cached tokens used (timings.cache_n), generated tokens, the md5 of
the token ids, the md5 of the text, the stop reason and the reply's first 60 characters (JSON-escaped). For fix-p also
<out>/<tag>-fix-p.probs.tsv: per step the chosen token id and log-probability, the runner-up's id and log-probability
(ids and numbers only, no text).
"""
import hashlib
import json
import os
import sys
import urllib.request

URL, OUT, TAG, TEXT, HISTORY, SLOT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
ARG = [int(x) for x in sys.argv[7:]]
DEPTH0, DEPTH1, START1, SHORT, NPRED = (ARG + [6000, 5000, 200, 1000, 256][len(ARG):])[:5]
assert HISTORY in ("none", "long", "short"), HISTORY

# session2.py's questions and greedy settings
QA = "\n\nQuestion: which switches above change how prompt checkpoints are taken? Answer in a few lines.\nAnswer:"
QB = "\n\nQuestion: which of those are on in a release preset?\nAnswer:"
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1}


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


def complete(slot, name, prompt, cache=True, n_probs=0):
    body = dict(GREEDY, prompt=prompt, n_predict=NPRED if name.startswith("fix") else 128, cache_prompt=cache,
                return_tokens=True, id_slot=slot)
    if n_probs:
        body["n_probs"] = n_probs
    try:
        reply = post("/completion", body)
    except Exception as e:  # recorded, so a failed request shows in the table
        reply = {"error": str(e)}
    with open(os.path.join(OUT, f"{TAG}-{name}.json"), "w") as f:
        json.dump(reply, f)
    toks = reply.get("tokens") or []
    text = str(reply.get("content", ""))
    t = reply.get("timings") or {}
    row = [TAG, name, str(slot), str(t.get("prompt_n", "")), str(t.get("cache_n", "")), str(len(toks)),
           md5(",".join(str(x) for x in toks)) if toks else "", md5(text) if toks else "",
           str(reply.get("stop_type", "")), json.dumps(text[:60] if toks else str(reply.get("error", ""))[:60])]
    with open(os.path.join(OUT, "session.tsv"), "a") as f:
        f.write("\t".join(row) + "\n")
    print(" ".join(row[:9]), flush=True)
    if n_probs:
        with open(os.path.join(OUT, f"{TAG}-{name}.probs.tsv"), "w") as f:
            f.write("step\tid\tlogprob\tid2\tlogprob2\n")
            for i, p in enumerate(reply.get("completion_probabilities") or []):
                top = p.get("top_logprobs") or p.get("probs") or []
                alt = [x for x in top if x.get("id") != p.get("id")]
                id2, lp2 = (alt[0].get("id", ""), repr(alt[0].get("logprob", ""))) if alt else ("", "")
                f.write(f"{i}\t{p.get('id', '')}\t{p.get('logprob', '')!r}\t{id2}\t{lp2}\n")
    return text


os.makedirs(OUT, exist_ok=True)
doc0, doc1, docs = document(0, DEPTH0), document(START1, DEPTH1), document(0, SHORT)
with open(os.path.join(OUT, f"{TAG}-document.md5"), "w") as f:
    for k, d in (("doc0", doc0), ("doc1", doc1), ("short", docs)):
        f.write(f"{md5(d)}  {k} {len(d)} characters\n")

if HISTORY in ("long", "short"):
    p0 = doc0 + QA
    r0 = complete(0, "s0-fresh", p0, cache=False)
    if HISTORY == "long":
        complete(0, "s0-turn", p0 + r0 + QB)
    else:
        ps = docs + QA
        rs = complete(0, "s0-short", ps, cache=False)
        complete(0, "s0-sturn", ps + rs + QB)

pf = doc1 + QA
complete(SLOT, "fix-1", pf, cache=False)
complete(SLOT, "fix-2", pf, cache=False)
complete(SLOT, "fix-p", pf, cache=False, n_probs=2)
