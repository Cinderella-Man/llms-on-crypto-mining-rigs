"""Send every pre-rendered prompt to llama-server's native /completion endpoint with the makers' sampling, K times,
and store the raw text. Attempt n samples with seed n + --seed-offset, the same seed for every task and file, so
two files' attempts share their random numbers. Resumable: (task, attempt) pairs already in OUT are skipped.
Attempts run in order (all tasks of attempt 1 first), so a stopped run still holds whole attempts.

usage: python run.py URL TASKS.jsonl OUT.jsonl [--attempts 1-5] [--conc 8] [--n-predict 4096] [--cache-prompt]
Writes OUT and OUT.manifest.json (server, build, sampling, task-list hash).
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from bfcl_common import complete, count_tokens, manifest, parse_attempts, server_props, sha256_of

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("tasks")
ap.add_argument("out")
ap.add_argument("--attempts", default="1-5")
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--conc", type=int, default=8, help="requests in flight; match llama-server's -np")
ap.add_argument("--n-predict", type=int, default=4096)
ap.add_argument("--cache-prompt", action="store_true", help="let the server reuse a slot's prompt cache")
args = ap.parse_args()

tasks = [json.loads(line) for line in open(args.tasks)]
attempts = parse_attempts(args.attempts)
done = set()
try:
    done = {(r["id"], r["attempt"]) for r in map(json.loads, open(args.out))}
except FileNotFoundError:
    pass
todo = [(t, a) for a in attempts for t in tasks if (t["id"], a) not in done]

slot_ctx = server_props(args.url)["n_ctx_slot"]
json.dump(manifest(args.url, {"lane": "bfcl_single_turn", "tasks_sha256": sha256_of(
    {"id": t["id"], "prompt": t["prompt"]} for t in tasks), "n_tasks": len(tasks), "attempts": attempts,
    "seed_offset": args.seed_offset, "n_predict": args.n_predict, "cache_prompt": args.cache_prompt}),
    open(args.out + ".manifest.json", "w"), indent=1)
print(f"slot n_ctx={slot_ctx}, todo={len(todo)}/{len(tasks) * len(attempts)}", flush=True)

lock = threading.Lock()
out = open(args.out, "a")


def one(item):
    t, a = item
    n_prompt = count_tokens(args.url, t["prompt"])
    n_predict = min(args.n_predict, slot_ctx - n_prompt - 8)
    rec = {"id": t["id"], "category": t["category"], "attempt": a, "seed": a + args.seed_offset,
           "n_prompt": n_prompt, "n_predict": n_predict}
    if n_predict < 256:
        rec.update(error="prompt too long", content="")
    else:
        rec.update(complete(args.url, t["prompt"], n_predict, rec["seed"], args.cache_prompt))
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
    return rec


n = 0
with ThreadPoolExecutor(args.conc) as ex:
    for rec in ex.map(one, todo):
        n += 1
        if n % 100 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} {rec['id']} attempt={rec['attempt']} stop={rec.get('stop_type')}", flush=True)
print("DONE", args.out, flush=True)
