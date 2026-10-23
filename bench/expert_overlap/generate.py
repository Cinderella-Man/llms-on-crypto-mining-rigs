"""Generate answers for the expert-overlap prompts with llama-server (started with --jinja) and keep the exact token
ids, so the routing pass sees the same sequence the model produced. Sampling is the server's default, which llama.cpp
takes from the GGUF's general.sampling.* keys (the makers' recommended settings); each prompt has a fixed seed.

usage: python generate.py URL PROMPTS.jsonl OUT.jsonl SEQS.txt [CONCURRENCY] [N_PREDICT]
  OUT.jsonl: one row per prompt (id, category, prompt and generated token ids, text, stop reason)
  SEQS.txt:  "<seq_id> <n_prompt> <tokens...>" lines for llama-route-dump
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_one(base, row, seed, n_predict):
    body = {"messages": row["messages"]}
    if "tools" in row:
        body["tools"] = row["tools"]
    prompt = post(f"{base}/apply-template", body)["prompt"]
    ids = post(f"{base}/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    res = post(f"{base}/completion", {"prompt": ids, "n_predict": n_predict, "seed": seed, "return_tokens": True,
                                      "cache_prompt": False, "stream": False})
    return {"id": row["id"], "category": row["category"], "seed": seed, "prompt_tokens": ids,
            "gen_tokens": res["tokens"], "text": res["content"], "stop_type": res.get("stop_type")}


def main():
    base, prompts_path, out_path, seqs_path = sys.argv[1:5]
    conc = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    n_predict = int(sys.argv[6]) if len(sys.argv) > 6 else 3072
    rows = [json.loads(l) for l in open(prompts_path)]
    with urllib.request.urlopen(f"{base}/props", timeout=60) as r:
        props = json.loads(r.read())
    params = props.get("default_generation_settings", {}).get("params", {})
    print("sampling:", {k: params.get(k) for k in ("temperature", "top_k", "top_p", "min_p", "samplers")},
          file=sys.stderr)
    with ThreadPoolExecutor(conc) as ex:
        results = list(ex.map(lambda ir: run_one(base, ir[1], 1000 + ir[0], n_predict), enumerate(rows)))
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(seqs_path, "w") as f:
        for i, r in enumerate(results):
            toks = r["prompt_tokens"] + r["gen_tokens"]
            f.write(f"{i} {len(r['prompt_tokens'])} " + " ".join(map(str, toks)) + "\n")
    for r in results:
        print(f"{r['id']:10s} {r['category']:6s} prompt {len(r['prompt_tokens']):5d} gen {len(r['gen_tokens']):5d} "
              f"stop {r['stop_type']}", file=sys.stderr)


if __name__ == "__main__":
    main()
