#!/usr/bin/env python3
# long_requests.py PORT LABEL [DEPTHS]: decode after long prompts, for Q230 (the sparse decode attention reads ~2,051
# cells per QSA layer however long the context; the dense kernels read all of them). Against a llama-server on this
# machine (the rig), so no prompt crosses the network: per depth (default 8192,32768 tokens) a fresh prompt of Python
# standard-library source (PSF) from this machine, raw text through /completion (no template), then 128 greedy tokens
# (temperature 0, fixed seed, end of text ignored). Greedy decoding makes both arms generate the same text from the
# same prompt, so they route the same experts and the decode times compare. Prints one CSV line per request:
# label,depth,prompt_tokens,prompt_ms,prompt_tok_s,gen_tokens,gen_ms,ms_per_token,gen_tok_s,text_md5
import glob, hashlib, json, sys, urllib.request

port, label = sys.argv[1], sys.argv[2]
depths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "8192,32768").split(",")]
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


print("label,depth,prompt_tokens,prompt_ms,prompt_tok_s,gen_tokens,gen_ms,ms_per_token,gen_tok_s,text_md5", flush=True)
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
    prompt = s[:lo]
    r = post("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0, "seed": 1, "ignore_eos": True,
                             "cache_prompt": False})
    t = r["timings"]
    md5 = hashlib.md5(r["content"].encode()).hexdigest()
    print(f"{label},{depth},{t['prompt_n']},{t['prompt_ms']:.0f},{t['prompt_per_second']:.1f},{t['predicted_n']},"
          f"{t['predicted_ms']:.0f},{t['predicted_per_token_ms']:.2f},{t['predicted_per_second']:.2f},{md5}", flush=True)
