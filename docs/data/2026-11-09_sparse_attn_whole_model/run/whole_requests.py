#!/usr/bin/env python3
# whole_requests.py PORT SET ARM DEPTHS OUTDIR [PROBE [PROBE_TOKENS]]: Q1171's requests, a changed copy of
# ../../2026-11-04_qsa_sparse_attn/run/long_requests.py. Against a llama-server on this machine's 127.0.0.1: per depth
# a fresh prompt of Python standard-library source (PSF) from this machine, built as long_requests.py builds it, raw
# text through /completion (no template), then 128 greedy tokens (temperature 0, fixed seed, end of text ignored).
# Greedy decoding writes the same text from the same prompt unless the arithmetic moved a token. The changes: no time
# is printed or kept; each answer's text, token ids and counts are saved; and with PROBE=1, after the last depth, each
# prompt runs once more for PROBE_TOKENS (16) greedy tokens with the top 10 log-probabilities of each (n_probs takes
# the server off its greedy sampler's short path, so the probe is a request of its own and the 128-token text never
# depends on it).
# Prints one CSV line per 128-token request:
# set,arm,depth,prompt_tokens,prompt_md5,gen_tokens,draft_tokens,draft_accepted,text_md5
# Writes OUTDIR/<set>-<arm>-<depth>.json and, for the probe, OUTDIR/probe-<set>-<arm>-<depth>.json.
import glob, hashlib, json, sys, urllib.request

port, name, arm, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5]
depths = [int(x) for x in sys.argv[4].split(",")]
probe = len(sys.argv) > 6 and sys.argv[6] == "1"
probe_tokens = int(sys.argv[7]) if len(sys.argv) > 7 else 16
files = sorted(glob.glob("/usr/lib/python3*/*.py"))


def text(chars, skip):
    out, n = [], 0
    for f in files[skip:] + files[:skip]:
        t = open(f, errors="replace").read()
        out.append("# file " + f.split("/")[-1] + "\n" + t)
        n += len(t)
        if n >= chars:
            break
    return "\n".join(out)[:chars]


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def n_tokens(s):
    return len(post("/tokenize", {"content": s})["tokens"])


def counts(timings):
    # the counters only: tokens of the prompt, generated, drafted and accepted
    return {k: v for k, v in timings.items() if k.endswith("_n") or k.endswith("_accepted")}


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


prompts = {}
for i, depth in enumerate(depths):
    # about 3 characters per token of Python source; trim to the depth by the server's own count
    s = text(4*depth, 7*i + 3)
    lo, hi = 0, len(s)
    while hi - lo > 256:
        mid = (lo + hi)//2
        if n_tokens(s[:mid]) < depth:
            lo = mid
        else:
            hi = mid
    prompt = prompts[depth] = s[:lo]
    r = post("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0, "seed": 1, "ignore_eos": True,
                             "cache_prompt": False, "return_tokens": True})
    c = counts(r["timings"])
    with open(f"{outdir}/{name}-{arm}-{depth}.json", "w") as f:
        json.dump({"set": name, "arm": arm, "depth": depth, "prompt_md5": md5(prompt), "counts": c,
                   "content": r["content"], "tokens": r.get("tokens")}, f)
    print(f"{name},{arm},{depth},{c['prompt_n']},{md5(prompt)},{c['predicted_n']},{c.get('draft_n', '')},"
          f"{c.get('draft_n_accepted', '')},{md5(r['content'])}", flush=True)

if probe:
    for depth in depths:
        r = post("/completion", {"prompt": prompts[depth], "n_predict": probe_tokens, "n_probs": 10, "temperature": 0, "seed": 1,
                                 "ignore_eos": True, "cache_prompt": False, "return_tokens": True})
        with open(f"{outdir}/probe-{name}-{arm}-{depth}.json", "w") as f:
            json.dump({"set": name, "arm": arm, "depth": depth, "prompt_md5": md5(prompts[depth]),
                       "counts": counts(r["timings"]), "content": r["content"], "tokens": r.get("tokens"),
                       "probs": r.get("completion_probabilities")}, f)
