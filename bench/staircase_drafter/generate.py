"""Generate Flint's answers for the training prompts with llama-server (started with --jinja) and keep the exact token
ids, at the makers' sampling (temperature 1.0, top-k 20, top-p 0.95, min-p 0), reasoning kept.

Several client processes, one per server copy, can share one prompt file: each prompt is claimed by creating
CLAIMS/<id> exclusively, so copies started later simply pick up the prompts nobody has claimed yet. Each process
appends its finished rows to its own OUT file. Stop issuing new requests by creating the file STOP_FILE (running
requests finish).

usage: python generate.py URL PROMPTS.jsonl OUT.jsonl CLAIMS_DIR [--conc N] [--n-predict N] [--stop-file PATH]
OUT.jsonl rows: id, category, source, seed, prompt_tokens, gen_tokens, stop_type. The seed is 20000 + the prompt's id
number, so prompt files appended later (prompts.py --extend) get their own seeds.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}


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
    args = ap.parse_args()
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
            try:
                r = run_one(args.url, row, seeds[row["id"]], args.n_predict, args.max_prompt)
            except Exception as e:  # noqa: BLE001 - a failed request is logged and skipped
                with lock:
                    state["failed"] += 1
                print(f"failed {row['id']}: {e}", file=sys.stderr)
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
