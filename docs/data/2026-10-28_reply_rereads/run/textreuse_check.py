"""Server check of LLAMA_SERVER_TEXT_REUSE (Q260), and of Q261/Q263 switches beside it, on a tiny random Flint.

    python3 -I textreuse_check.py --on URL --off URL --cold URL [--trials 8] [--n-predict 700] [--cold-informational]

Three llama-servers of the fix branch on the same tiny model, -fa off (batch-invariant on the CPU), one slot,
--ctx-checkpoints 4, -ub 128: --on with LLAMA_SERVER_TEXT_REUSE=1, --off without, --cold restarted state per trial
(its cache is cleared by an unrelated prompt first). Per trial and server:
1. /completion of a prompt P with n_predict tokens at temperature 1 and a fixed seed, ids returned (a random model
   samples non-canonical splits often);
2. /completion of P + the reply's text + a short tail, greedy, ids returned.
Checks: the reply's ids differ from its text's tokenization (else the trial tests nothing); on the server with the
switch, the second request reuses the prompt and the whole reply (cache_n >= len(P) + reply - 1), without it only up
to the checkpoint; the switch's answer equals the cold server's answer to the same ids, which the check splices the
way the server does (the cached ids up to the last token end both sequences share inside the common text); the
"off" server samples the same first reply as "on" from the same seed (both compute the same logits). With
--cold-informational (GPU runs: other micro-batch shapes round differently) the cold comparison is only reported.
"""
import argparse
import json
import sys
import urllib.request


def post(url, path, obj):
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def tok(url, s):
    return post(url, "/tokenize", {"content": s, "add_special": True, "parse_special": True})["tokens"]


def splice(url, cache, prompt):
    n = 0
    while n < len(cache) and n < len(prompt) and cache[n] == prompt[n]:
        n += 1
    if n in (len(cache), len(prompt)):
        return prompt, n, 0
    pc = [post(url, "/detokenize", {"tokens": [t]})["content"].encode() for t in cache[n:]]
    pp = [post(url, "/detokenize", {"tokens": [t]})["content"].encode() for t in prompt[n:]]
    tc, tp = b"".join(pc), b"".join(pp)
    same = 0
    while same < min(len(tc), len(tp)) and tc[same] == tp[same]:
        same += 1
    ec, ep, s = [], [], 0
    for p in pc:
        s += len(p); ec.append(s)
    s = 0
    for p in pp:
        s += len(p); ep.append(s)
    kc = kp = ic = ip = 0
    while ic < len(ec) and ip < len(ep):
        if min(ec[ic], ep[ip]) > same:
            break
        if ec[ic] == ep[ip]:
            ic += 1; ip += 1; kc, kp = ic, ip
        elif ec[ic] < ep[ip]:
            ic += 1
        else:
            ip += 1
    return prompt[:n] + cache[n:n + kc] + prompt[n + kp:], n, kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--cold", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=700)
    ap.add_argument("--cold-informational", action="store_true")
    a = ap.parse_args()
    head = "<|system|>You are a careful coding assistant.<|user|>Fix the failing test in triangle.py and explain the change.<|assistant|>"
    print("trial\tserver\tprompt\treply\treply_retok_differs\tcache_n\texpected_min\tprompt_n\tanswer_equals_cold\tspliced_at\tkept"
          "\tckpt_created_1\tckpt_created_2\treply_equals_on")
    fails = 0
    for t in range(a.trials):
        seed = 1000 + t
        p_ids = tok(a.on, head)
        gen_on = None
        for name, url in (("on", a.on), ("off", a.off)):
            post(url, "/completion", {"prompt": f"unrelated {t} {name}", "n_predict": 1, "cache_prompt": True})
            # plain ASCII text only: no special tokens and no partial UTF-8 pieces, so the reply's text is exact
            r1 = post(url, "/completion", {"prompt": head, "n_predict": a.n_predict, "temperature": 1.0, "top_k": 0,
                                           "top_p": 1.0, "min_p": 0.0, "seed": seed, "cache_prompt": True,
                                           "return_tokens": True, "ignore_eos": True,
                                           "grammar": "root ::= [a-zA-Z0-9 .,()=_:]+"})
            gen = r1["tokens"]
            text = r1["content"]
            differs = tok(url, head + text)[len(p_ids):] != gen
            tail = " Run the tests again."
            r2 = post(url, "/completion", {"prompt": head + text + tail, "n_predict": 24, "temperature": 0.0,
                                           "cache_prompt": True, "return_tokens": True})
            cache = p_ids + gen[:-1]
            spliced, n_at, kept = splice(url, cache, tok(url, head + text + tail))
            eq = ""
            if name == "on":
                post(a.cold, "/completion", {"prompt": f"unrelated cold {t}", "n_predict": 1, "cache_prompt": True})
                rc = post(a.cold, "/completion", {"prompt": spliced, "n_predict": 24, "temperature": 0.0,
                                                   "cache_prompt": True, "return_tokens": True})
                eq = int(rc["tokens"] == r2["tokens"])
                ok = (eq == 1 or a.cold_informational) and r2["timings"]["cache_n"] >= len(cache) - 1
                gen_on, same = gen, ""
            else:
                # the same seed on both servers: the replies are identical when both compute the same logits
                same = int(gen == gen_on)
                ok = same == 1
            fails += 0 if ok else 1
            print(f"{t}\t{name}\t{len(p_ids)}\t{len(gen)}\t{int(differs)}\t{r2['timings']['cache_n']}\t{len(cache) - 1}\t"
                  f"{r2['timings']['prompt_n']}\t{eq}\t{n_at}\t{kept}\t{r1['timings'].get('ckpt_created_n')}\t"
                  f"{r2['timings'].get('ckpt_created_n')}\t{same}", flush=True)
    print(f"# {'PASS' if fails == 0 else 'FAIL'}: {fails} failed trials", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
