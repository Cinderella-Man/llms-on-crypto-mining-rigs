"""Generate Flint's answers for the training prompts with llama-server (started with --jinja) and keep the exact token
ids, at the makers' sampling (temperature 1.0, top-k 20, top-p 0.95, min-p 0), reasoning kept.

Several client processes, one per server copy, can share one prompt file: each prompt is claimed by creating
CLAIMS/<id> exclusively, so copies started later simply pick up the prompts nobody has claimed yet. Each process
appends its finished rows to its own OUT file. Stop issuing new requests by creating the file STOP_FILE (running
requests finish). While the server is unreachable (a restart) a request is retried every 15 s for up to 30 minutes;
after that its claim is released and the thread stops, so a dead server never uses up the prompt pool.

usage: python generate.py URL PROMPTS.jsonl OUT.jsonl CLAIMS_DIR [--conc N] [--n-predict N] [--stop-file PATH]
       python generate.py URL ... --api vllm --tokenizer FLINT_HF_DIR      # vLLM's OpenAI-compatible server
OUT.jsonl rows: id, category, source, seed, prompt_tokens, gen_tokens, stop_type. The seed is 20000 + the prompt's id
number, so prompt files appended later (prompts.py --extend) get their own seeds.

With --api vllm the chat template and tokenizer of the Hugging Face directory build the prompt ids (transformers
needed; vllm/check_prompts.py checks they equal llama-server's), /v1/completions samples, and the rows follow
llama-server's conventions: stop_type "eos" with the end token kept as the last id, or "limit" at N_PREDICT tokens.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
END_IDS = [150001, 150003, 150005, 150018]   # Flint's generation_config eos_token_id


def post(url, body, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_one(base, row, seed, n_predict, max_prompt):
    body = {"messages": row["messages"]}
    if "tools" in row:
        body["tools"] = row["tools"]
    prompt = post(f"{base}/apply-template", body)["prompt"]
    ids = post(f"{base}/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
    if len(ids) > max_prompt:
        raise ValueError(f"prompt of {len(ids)} tokens")
    res = post(f"{base}/completion", {"prompt": ids, "n_predict": n_predict, "seed": seed, "return_tokens": True,
                                      "cache_prompt": False, "stream": False, **SAMPLING})
    return {"id": row["id"], "category": row["category"], "source": row.get("source"), "seed": seed,
            "prompt_tokens": ids, "gen_tokens": res["tokens"], "stop_type": res.get("stop_type")}


def prompt_ids_hf(tok, row):
    """llama-server's /apply-template + /tokenize (parse_special, no BOS) with the Hugging Face template."""
    messages = []
    for m in row["messages"]:
        if m.get("tool_calls"):   # llama-server parses string arguments; the template iterates arguments.items()
            m = dict(m, tool_calls=[dict(tc, function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"])))
                                    if isinstance(tc.get("function", {}).get("arguments"), str) else tc
                                    for tc in m["tool_calls"]])
        messages.append(m)
    text = tok.apply_chat_template(messages, tools=row.get("tools"), add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def run_one_vllm(base, row, seed, n_predict, max_prompt, tok, model):
    ids = prompt_ids_hf(tok, row)
    if len(ids) > max_prompt:
        raise ValueError(f"prompt of {len(ids)} tokens")
    res = post(f"{base}/v1/completions", {"model": model, "prompt": ids, "max_tokens": n_predict, "seed": seed,
                                          "stop_token_ids": END_IDS, "return_token_ids": True,
                                          "skip_special_tokens": False, **SAMPLING})
    choice = res["choices"][0]
    gen = list(choice["token_ids"])
    if choice["finish_reason"] == "length":
        stop_type = "limit"
    elif choice["finish_reason"] == "stop":
        stop_type = "eos"
        if not gen or gen[-1] not in END_IDS:   # keep the end token as llama-server's rows do
            if not isinstance(choice.get("stop_reason"), int):
                raise ValueError(f"stop without an end token: {choice.get('stop_reason')!r}")
            gen.append(choice["stop_reason"])
    else:
        raise ValueError(f"finish_reason {choice['finish_reason']!r}")
    return {"id": row["id"], "category": row["category"], "source": row.get("source"), "seed": seed,
            "prompt_tokens": ids, "gen_tokens": gen, "stop_type": stop_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("claims")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--n-predict", type=int, default=2048)
    ap.add_argument("--max-prompt", type=int, default=6144, help="skip prompts longer than this many tokens")
    ap.add_argument("--stop-file")
    ap.add_argument("--api", choices=["llama", "vllm"], default="llama")
    ap.add_argument("--tokenizer", help="--api vllm: Flint's Hugging Face directory (chat template and tokenizer)")
    ap.add_argument("--model", default="flint", help="--api vllm: the served model name")
    args = ap.parse_args()
    if args.api == "vllm":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

        def generate_one(row, seed):
            return run_one_vllm(args.url, row, seed, args.n_predict, args.max_prompt, tok, args.model)
    else:
        def generate_one(row, seed):
            return run_one(args.url, row, seed, args.n_predict, args.max_prompt)
    os.makedirs(args.claims, exist_ok=True)
    rows = [json.loads(l) for l in open(args.prompts)]
    seeds = {r["id"]: 20000 + int(r["id"].split("_")[-1]) for r in rows}   # the id number: unique across files
    lock = threading.Lock()
    state = {"next": 0, "done": 0, "tokens": 0, "failed": 0}
    out = open(args.out, "a")
    t0 = time.time()

    def claim():
        with lock:
            while state["next"] < len(rows):
                row = rows[state["next"]]
                state["next"] += 1
                try:
                    os.close(os.open(os.path.join(args.claims, row["id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    return row
                except FileExistsError:
                    continue
            return None

    def worker():
        while not (args.stop_file and os.path.exists(args.stop_file)):
            row = claim()
            if row is None:
                return
            r = None
            for _ in range(120):   # server unreachable (a restart): retry every 15 s for 30 min, keeping the claim
                try:
                    r = generate_one(row, seeds[row["id"]])
                    break
                except urllib.error.HTTPError as e:
                    r = e
                    break
                except (urllib.error.URLError, ConnectionError, http.client.HTTPException) as e:
                    r = e
                    time.sleep(15)
                except Exception as e:  # noqa: BLE001 - a failed request is logged and skipped
                    r = e
                    break
            if isinstance(r, Exception):
                with lock:
                    state["failed"] += 1
                print(f"failed {row['id']}: {r}", file=sys.stderr)
                if isinstance(r, (urllib.error.URLError, ConnectionError, http.client.HTTPException)) \
                        and not isinstance(r, urllib.error.HTTPError):
                    os.remove(os.path.join(args.claims, row["id"]))   # never answered: back to the pool
                    return
                continue
            with lock:
                out.write(json.dumps(r) + "\n")
                out.flush()
                state["done"] += 1
                state["tokens"] += len(r["gen_tokens"])
                if state["done"] % 50 == 0:
                    print(f"{state['done']} done, {state['tokens']} generated tokens, {state['failed']} failed, "
                          f"{(time.time() - t0) / 60:.1f} min", file=sys.stderr, flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {state['done']} done, {state['tokens']} generated tokens, {state['failed']} failed",
          file=sys.stderr)


if __name__ == "__main__":
    main()
