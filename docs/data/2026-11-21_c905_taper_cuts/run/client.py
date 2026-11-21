# C905: the server check's client. Sends one request set to a llama-server on 127.0.0.1 and writes one JSON line per
# request: the token ids of the reply, the top-3 probabilities of each, prompt_n, cache_n and the checkpoint the
# server restored. Every prompt is a list of token ids, so each call has a known size:
#   fresh prompts of 30, 302, 516, 520, 577, 583, 610, 775, 896, 1,086, 1,400, 1,534, 1,985 and 3,313 tokens (the
#   call sizes of C222's replay where the in-call cuts fall, and around them), each with a first token no other prompt
#   has, so the server reuses nothing and reads the whole prompt in one call;
#   after 577: the same prompt, its reply and 600 new tokens (a continuation: the server reads the reply and the new
#   tokens on top of its cache);
#   after 1,534: the same prompt with its token at 1,100 changed (the server restores the checkpoint it took inside
#   the call, 516 tokens before its end, and reads the rest).
# With a taper given, every request sends it as "ubatch_taper" (the server's own setting otherwise).
# Usage: python3 -I client.py <port> <out.jsonl> [<ubatch_taper>]
import json
import sys
import urllib.request

port, out = sys.argv[1], sys.argv[2]
taper = sys.argv[3] if len(sys.argv) > 3 else None
base = f"http://127.0.0.1:{port}"

with urllib.request.urlopen(base + "/v1/models", timeout=60) as r:
    n_vocab = json.load(r)["data"][0]["meta"]["n_vocab"]


def toks(n, seed, first):
    # a Park-Miller sequence below the vocabulary size, its first token set (unique per prompt)
    x, res = seed, []
    for _ in range(n):
        x = (x * 16807) % 2147483647
        res.append(1 + x % (n_vocab - 1))
    res[0] = first
    return res


def complete(name, prompt, f):
    body = {"prompt": prompt, "n_predict": 16, "temperature": 0.0, "top_k": 1, "n_probs": 3,
            "return_tokens": True, "cache_prompt": True}
    if taper is not None:
        body["ubatch_taper"] = taper
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        t = d.get("timings", {})
        row = {"req": name, "n_prompt": len(prompt), "tokens": d.get("tokens"),
               "probs": d.get("completion_probabilities"), "prompt_n": t.get("prompt_n"),
               "cache_n": t.get("cache_n"), "cache_ckpt_n": t.get("cache_ckpt_n"),
               "ckpt_incall_n": t.get("ckpt_incall_n")}
    except Exception as e:  # a failed request is a result too
        row = {"req": name, "n_prompt": len(prompt), "error": repr(e)}
    f.write(json.dumps(row) + "\n")
    f.flush()
    return row


with open(out, "w") as f:
    first = 1
    for n in (30, 302, 516, 520, 577, 583, 610, 775, 896, 1086, 1400, 1534, 1985, 3313):
        p = toks(n, 1000 + n, first)
        first += 1
        row = complete(f"fresh-{n}", p, f)
        if n == 577:
            complete("cont-577", p + (row.get("tokens") or []) + toks(600, 9577, 1 + (first + 50) % (n_vocab - 1)), f)
        if n == 1534:
            q = list(p)
            q[1100] = 1 + (q[1100] % (n_vocab - 1))
            complete("edit-1534", q, f)
