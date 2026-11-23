# C956: the sized prompts' client. Sends one request set per pass to a llama-server and writes one JSON line per
# request: the reply's token ids, the top-3 log-probabilities of each, and the server's counters (prompt_n, cache_n,
# the restored checkpoint, the in-call checkpoints, drafted and accepted tokens; no times). Every prompt is a list of
# token ids, so each prompt call has a known size:
#   fresh prompts of 30, 302, 516, 520, 569, 577, 583, 599, 610, 775, 803, 815, 896, 936, 1,086, 1,130, 1,300, 1,359,
#   1,400, 1,534, 1,985 and 3,313 tokens: C905's server check's sizes (C222's call sizes where the in-call cuts fall,
#   and around them) and the 14 call sizes whose plan cuts=1 changes under short=1400:64. Each is a stretch of text
#   tokens (the given files, tokenized by the server, taken from a different offset per size) whose first token no
#   other prompt starts with, so the server reuses nothing and reads the whole prompt in one call;
#   after 577: the same prompt, its reply and 600 new tokens (a continuation read on top of the cache);
#   after 1,534: the same prompt with its token at 1,100 changed (the server restores the checkpoint it took inside the
#   call, 516 tokens before its end, and reads the rest).
# A pass is "label:spec": with spec "-" the requests leave the taper to the server's setting, otherwise each request
# sends "ubatch_taper": spec. Greedy (temperature 0, top-k 1).
# Usage: python3 -I client.py <url> <out.jsonl> "<label:spec> ..." <max_tokens> <text file> [<text file> ...]
import json
import sys
import urllib.request

url, out, passes, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3].split(), int(sys.argv[4])
texts = sys.argv[5:]
SIZES = (30, 302, 516, 520, 569, 577, 583, 599, 610, 775, 803, 815, 896, 936, 1086, 1130, 1300, 1359, 1400, 1534,
         1985, 3313)


def post(path, body, timeout=1800):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


pool = []
for t in texts:
    with open(t, encoding="utf-8", errors="replace") as f:
        pool += post("/tokenize", {"content": f.read(), "add_special": False})["tokens"]
firsts = list(dict.fromkeys(pool))  # distinct tokens in order of appearance: one prompt start each


def prompt(n, k):
    start = (k * 997) % len(pool)
    res = [pool[(start + i) % len(pool)] for i in range(n)]
    res[0] = firsts[k]
    return res


def top3(probs):
    # per reply token: [id, [[id, logprob] x 3]], from either of the server's two formats
    res = []
    for p in probs or []:
        tops = p.get("top_logprobs") or p.get("top_probs") or []
        res.append([p.get("id"), [[x.get("id"), x.get("logprob", x.get("prob"))] for x in tops]])
    return res


def complete(f, label, spec, name, toks):
    body = {"prompt": toks, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1, "seed": 1, "n_probs": 3,
            "return_tokens": True, "cache_prompt": True}
    if spec != "-":
        body["ubatch_taper"] = spec
    try:
        d = post("/completion", body)
        t = d.get("timings", {})
        row = {"pass": label, "spec": spec, "req": name, "n_prompt": len(toks), "tokens": d.get("tokens"),
               "probs": top3(d.get("completion_probabilities")),
               "counts": {k: v for k, v in sorted(t.items()) if not k.endswith(("_ms", "_per_second"))}}
    except Exception as e:  # a failed request is a result too
        row = {"pass": label, "spec": spec, "req": name, "n_prompt": len(toks), "error": repr(e)}
    f.write(json.dumps(row) + "\n")
    f.flush()
    return row


with open(out, "w") as f:
    for ps in passes:
        label, _, spec = ps.partition(":")
        for k, n in enumerate(SIZES):
            p = prompt(n, k)
            row = complete(f, label, spec, f"fresh-{n}", p)
            if n == 577:
                cont = p + (row.get("tokens") or []) + prompt(600, len(SIZES) + 1)
                complete(f, label, spec, "cont-577", cont)
            if n == 1534:
                q = list(p)
                q[1100] = firsts[len(SIZES) + 2] if firsts[len(SIZES) + 2] != q[1100] else firsts[len(SIZES) + 3]
                complete(f, label, spec, "edit-1534", q)
        print(f"pass {label} ({spec}): {sum(1 for _ in open(out))} lines", flush=True)
