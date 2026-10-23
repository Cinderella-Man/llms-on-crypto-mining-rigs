"""Send every pre-rendered prompt to llama-server's native /completion endpoint (greedy)
and store the raw text. Resumable: tasks already in OUT are skipped.

usage: python run.py URL TASKS.jsonl OUT.jsonl [CONCURRENCY] [N_PREDICT]
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests

url, tasks_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
n_predict_max = int(sys.argv[5]) if len(sys.argv) > 5 else 4096

tasks = [json.loads(l) for l in open(tasks_path)]
done = set()
try:
    done = {json.loads(l)["id"] for l in open(out_path)}
except FileNotFoundError:
    pass
todo = [t for t in tasks if t["id"] not in done]

props = requests.get(f"{url}/props", timeout=30).json()
slot_ctx = props["default_generation_settings"]["n_ctx"]
print(f"slot n_ctx={slot_ctx}, todo={len(todo)}/{len(tasks)}", flush=True)

lock = threading.Lock()
out = open(out_path, "a")


def one(t):
    toks = requests.post(f"{url}/tokenize", json={"content": t["prompt"]}, timeout=120).json()["tokens"]
    n_prompt = len(toks)
    n_predict = min(n_predict_max, slot_ctx - n_prompt - 8)
    rec = {"id": t["id"], "category": t["category"], "n_prompt": n_prompt, "n_predict": n_predict}
    if n_predict < 256:
        rec.update(error="prompt too long", content="")
    else:
        r = requests.post(f"{url}/completion", json={
            "prompt": t["prompt"],
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": False,
            "seed": 0,
            "stream": False,
        }, timeout=3600)
        r.raise_for_status()
        j = r.json()
        rec.update(
            content=j["content"],
            tokens_predicted=j.get("tokens_predicted"),
            tokens_evaluated=j.get("tokens_evaluated"),
            stop_type=j.get("stop_type"),
            stopping_word=j.get("stopping_word"),
            truncated=j.get("truncated"),
        )
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
    return rec


n = 0
with ThreadPoolExecutor(conc) as ex:
    for rec in ex.map(one, todo):
        n += 1
        if n % 25 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} {rec['id']} stop={rec.get('stop_type')}", flush=True)
print("DONE", out_path, flush=True)
