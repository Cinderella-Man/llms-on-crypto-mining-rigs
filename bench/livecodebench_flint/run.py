"""Sample K solutions per LiveCodeBench problem from llama-server's /completion endpoint with the makers' settings
and a 32,768-token budget (reasoning plus answer), and store the raw text. Attempt n samples with seed n +
--seed-offset, the same seed for every problem and file. Resumable: (problem, attempt) pairs already in OUT are
skipped. Attempts run in order, so a stopped run still holds whole attempts.

usage: python run.py URL TASKS.jsonl OUT.jsonl [--attempts 1-5] [--conc 8] [--n-predict 32768] [--shuffle SEED]
                     [--part I/N]
Writes OUT and OUT.manifest.json. --shuffle and --part select a slice of the problems in a seeded random order, as in
../bfcl_flint/run.py. Refuses to start if a server slot cannot hold the longest prompt plus the budget,
so that every file gets the same budget.
"""
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from lcb_common import complete, count_tokens, manifest, parse_attempts, server_props, sha256_of

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("tasks")
ap.add_argument("out")
ap.add_argument("--attempts", default="1-5")
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--conc", type=int, default=8, help="requests in flight; match llama-server's -np")
ap.add_argument("--n-predict", type=int, default=32768)
ap.add_argument("--cache-prompt", action="store_true")
ap.add_argument("--shuffle", type=int)
ap.add_argument("--part", default="1/1")
args = ap.parse_args()

tasks = [json.loads(line) for line in open(args.tasks)]
if args.shuffle is not None:
    random.Random(args.shuffle).shuffle(tasks)
i, n = map(int, args.part.split("/"))
tasks = tasks[(i - 1) * len(tasks) // n:i * len(tasks) // n]
attempts = parse_attempts(args.attempts)
done = set()
try:
    done = {(r["id"], r["attempt"]) for r in map(json.loads, open(args.out))}
except FileNotFoundError:
    pass
todo = [(t, a) for a in attempts for t in tasks if (t["id"], a) not in done]

slot_ctx = server_props(args.url)["n_ctx_slot"]
n_prompt = {t["id"]: count_tokens(args.url, t["prompt"]) for t in tasks}
need = max(n_prompt.values()) + args.n_predict + 8
if slot_ctx < need:
    raise SystemExit(f"slot n_ctx {slot_ctx} < {need} (longest prompt + budget): raise llama-server's -c")
json.dump(manifest(args.url, {"lane": "livecodebench", "tasks_sha256": sha256_of(
    {"id": t["id"], "prompt": t["prompt"]} for t in tasks), "n_tasks": len(tasks), "attempts": attempts,
    "seed_offset": args.seed_offset, "n_predict": args.n_predict, "cache_prompt": args.cache_prompt,
    "shuffle": args.shuffle, "part": args.part}),
    open(args.out + ".manifest.json", "w"), indent=1)
print(f"slot n_ctx={slot_ctx}, longest prompt {max(n_prompt.values())}, todo={len(todo)}/{len(tasks) * len(attempts)}",
      flush=True)

lock = threading.Lock()
out = open(args.out, "a")


def one(item):
    t, a = item
    rec = {"id": t["id"], "category": t["category"], "attempt": a, "seed": a + args.seed_offset,
           "n_prompt": n_prompt[t["id"]], "n_predict": args.n_predict}
    rec.update(complete(args.url, t["prompt"], args.n_predict, rec["seed"], args.cache_prompt))
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
    return rec


n = 0
with ThreadPoolExecutor(args.conc) as ex:
    for rec in ex.map(one, todo):
        n += 1
        if n % 20 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} {rec['id']} attempt={rec['attempt']} tokens={rec['tokens_predicted']} "
                  f"stop={rec['stop_type']}", flush=True)
print("DONE", args.out, flush=True)
