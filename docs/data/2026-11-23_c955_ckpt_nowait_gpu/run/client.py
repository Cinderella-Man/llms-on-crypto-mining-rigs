# C955: the client of run.sh. Sends R183's four recall prompts (docs/data/2026-11-20_r183_ornith_recall/prompts/, the
# messages as Ornith's server got them) to a llama-server, then two restore pairs, and writes one JSON line per
# request: the reply's reasoning and content, each reply token with its top-3 log-probabilities, and the server's
# counters (prompt_n, cache_n, the restored checkpoint, the checkpoint counters). C907's client.py
# (docs/data/2026-11-21_c907_ckpt_stall/run/) with longer replies and a second, deeper restore pair.
#   kinds-8000, kinds-16000, kinds-32000, kinds-47000: as R183 sent them (greedy, seed 1546, cache_prompt false), but
#     MAX_TOKENS reply tokens;
#   rst-a: the 16,000-token prompt behind a system message no other prompt has, cache_prompt true (a whole read);
#   rst-b: rst-a with the user message at 62 % of the list changed, cache_prompt true: the server restores its newest
#     checkpoint before the change (the one taken between two calls near 8,200 tokens) and reads the rest;
#   rst-c: the 47,000-token prompt behind another new system message, cache_prompt true (a whole read);
#   rst-d: rst-c with the user message at 93 % of the list changed, cache_prompt true: the server restores the
#     checkpoint taken between two calls near 41,000 tokens and reads the rest.
# Usage: python3 -I client.py <base url> <prompts dir> <out.jsonl> [max tokens, default 64]
import gzip
import json
import os
import sys
import urllib.request

base, prompts, out = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TIMINGS = ("prompt_n", "cache_n", "cache_src", "cache_ckpt_n", "ckpt_created_n", "ckpt_incall_n", "ckpt_nowait_n",
           "ckpt_dev_n")


def load(target):
    with gzip.open(os.path.join(prompts, f"kinds-{target}.json.gz")) as f:
        return json.load(f)


def edit(msgs, frac):
    """the messages with the first user message at or after frac of the list changed"""
    i = int(len(msgs) * frac)
    while msgs[i]["role"] != "user":
        i += 1
    edited = [dict(m) for m in msgs]
    edited[i]["content"] = edited[i]["content"] + " Really."
    return edited, i


def ask(name, msgs, cache_prompt, f, note=None):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "seed": 1546, "cache_prompt": cache_prompt, "logprobs": True, "top_logprobs": 3}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3000) as r:
            d = json.load(r)
        ch = d["choices"][0]
        msg = ch.get("message", {})
        lp = (ch.get("logprobs") or {}).get("content") or []
        t = d.get("timings", {})
        row = {"req": name, "n_messages": len(msgs), "note": note, "finish": ch.get("finish_reason"),
               "reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
               "tokens": [[x.get("token"), x.get("logprob"), [[y.get("token"), y.get("logprob")] for y in x.get("top_logprobs", [])]]
                          for x in lp],
               "timings": {k: t.get(k) for k in TIMINGS}}
    except Exception as e:  # a failed request is a result too
        row = {"req": name, "n_messages": len(msgs), "note": note, "error": repr(e)}
    f.write(json.dumps(row) + "\n")
    f.flush()
    print(name, row.get("timings", row.get("error")), flush=True)
    return row


with open(out, "w") as f:
    for target in (8000, 16000, 32000, 47000):
        ask(f"kinds-{target}", load(target), False, f)
    msgs = [{"role": "system", "content": "Session 955: a restore check."}] + load(16000)
    ask("rst-a", msgs, True, f)
    edited, i = edit(msgs, 0.62)
    ask("rst-b", edited, True, f, note=f"message {i} of {len(msgs)} changed")
    msgs = [{"role": "system", "content": "Session 955: a deep restore check."}] + load(47000)
    ask("rst-c", msgs, True, f)
    edited, i = edit(msgs, 0.93)
    ask("rst-d", edited, True, f, note=f"message {i} of {len(msgs)} changed")
