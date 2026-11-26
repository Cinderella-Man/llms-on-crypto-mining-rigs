#!/usr/bin/env python3
"""R244 (Q1924): a one-request proxy log for bench/agent_check/replay.py whose request is a recorded one with
"cache_prompt": false added to its body, so llama-server reads the whole prompt from position 0 on a cleared cache
(its prompt-cache reuse runs only when the request's cache_prompt is true): an empty-cache read later in a load,
with the server's command line unchanged. The slot-erase endpoint replay.py's --erase-slot calls
(/slots/0?action=erase) answers 501 on a server started without --slot-save-path, a flag the preset does not have.

    python3 -I no_cache_log.py LOG TRACE OUT_LOG

LOG is a proxy.jsonl with its .bodies.gz beside it, TRACE a trace tag (e.g. v4r/bugfix-python-affine-cipher/a1/t1)
whose first request is copied; OUT_LOG (and OUT_LOG.bodies.gz) get it under the trace "nocache/<task>/a1/t1", with
the recorded row's counters (so replay.py's recorded reply length applies). Standard library only.
"""
import gzip
import json
import sys

log, trace, out = sys.argv[1:4]
row = next(r for r in (json.loads(x) for x in open(log) if x.strip()) if r.get("tag") == f"{trace}/1")
body = next(b for b in (json.loads(x) for x in gzip.open(log + ".bodies.gz", "rt")) if b["id"] == row["id"])
assert isinstance(body["body"], dict) and "cache_prompt" not in body["body"], "body already sets cache_prompt"
task = trace.split("/")[1]
row = dict(row, tag=f"nocache/{task}/a1/t1/1", id=row["id"] + "-nocache")
body = dict(body, id=row["id"], body=dict(body["body"], cache_prompt=False))
with open(out, "w") as w:
    w.write(json.dumps(row) + "\n")
with gzip.open(out + ".bodies.gz", "wt") as w:
    w.write(json.dumps(body) + "\n")
print(f"{row['tag'][:-2]}: {row['prompt_n']} recorded prompt tokens, reply {row['predicted_n']} tokens, cache_prompt false")
