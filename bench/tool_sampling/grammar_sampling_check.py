#!/usr/bin/env python3
"""Tool requests through llama-server's chat endpoint, with backend sampling on and off (Q66, Q67).

Sends BFCL tasks (the tasks.jsonl of bench/bfcl_flint/prepare.py: messages and tools per task) to
/v1/chat/completions with the tools and tool_choice "auto", so that llama-server builds Flint's lazy tool-call grammar,
at the makers' sampling (bench/flint_client.py). Each task runs once per variant, temperature and seed. A variant is a
name and the request fields it sets:
  device:backend_sampling=true      the backend sampler draws, the host checks that token against the grammar (Q67)
  host:backend_sampling=false       the host draws from the full row and checks it (llama-server's host path)
  device-none:backend_sampling=true;tool_choice=none    the same prompt without a grammar
One JSON line per request: task, category, variant, temperature, seed, finish reason, reasoning, content, tool calls
and the server's token counts, including the sampling counters of a build with Q67 (sampled_on_device, grammar_n,
grammar_rejected_n). Speeds are left out unless --keep-speed is given.

usage:
  grammar_sampling_check.py run --url URL --tasks tasks.jsonl --out runs.jsonl [--per-category N] [--temps 0,1]
      [--seeds 1,2] [--variants device:backend_sampling=true host:backend_sampling=false] [--max-tokens 2048]
  grammar_sampling_check.py compare runs.jsonl [--a device] [--b host]
      per temperature: how many (task, seed) pairs give the same reasoning, content and calls in both variants
  grammar_sampling_check.py rejects runs.jsonl [--variant device]
      Q66: the share of tokens sampled while the grammar applied, and how often it rejected the sampled token
A run resumes: requests already answered in the output file are skipped (failed ones are sent again).
"""
import argparse
import collections
import json
import os
import random
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flint_client import SAMPLING  # noqa: E402

KEEP_TIMINGS = ("prompt_n", "cache_n", "predicted_n", "sampled_on_device", "grammar_n", "grammar_rejected_n")


def parse_value(v):
    if v in ("true", "false"):
        return v == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_variant(spec):
    name, _, fields = spec.partition(":")
    out = {}
    for kv in filter(None, fields.split(";")):
        k, _, v = kv.partition("=")
        out[k] = parse_value(v)
    return name, out


def load_tasks(path, per_category, seed):
    tasks = [json.loads(line) for line in open(path)]
    if per_category:
        by_cat = collections.defaultdict(list)
        for t in tasks:
            by_cat[t["category"]].append(t)
        tasks = []
        for cat, ts in by_cat.items():
            rng = random.Random(f"{seed}-{cat}")
            tasks += [ts[i] for i in sorted(rng.sample(range(len(ts)), min(per_category, len(ts))))]
    return tasks


def key(row):
    return (row["task"], row["variant"], row["temperature"], row["seed"])


def cmd_run(args):
    tasks = load_tasks(args.tasks, args.per_category, args.sample_seed)
    variants = [parse_variant(v) for v in args.variants]
    done = set()
    if os.path.exists(args.out):
        done = {key(r) for r in map(json.loads, open(args.out)) if "error" not in r}
    temps = [float(t) for t in args.temps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    with open(args.out, "a") as f:
        for temp in temps:
            for seed in seeds if temp > 0 else seeds[:1]:
                for t in tasks:
                    for name, fields in variants:
                        row = {"task": t["id"], "category": t["category"], "variant": name, "temperature": temp,
                               "seed": seed}
                        if key(row) in done:
                            continue
                        body = dict(SAMPLING)
                        body.update({"messages": t["messages"], "tools": t["tools"], "tool_choice": "auto",
                                     "temperature": temp, "seed": seed, "max_tokens": args.max_tokens,
                                     "cache_prompt": False, "stream": False})
                        body.update(fields)
                        r = requests.post(args.url.rstrip("/") + "/v1/chat/completions", json=body, timeout=3600)
                        if r.status_code != 200:
                            row["error"] = r.text[:500]
                        else:
                            res = r.json()
                            msg = res["choices"][0]["message"]
                            timings = res.get("timings", {})
                            row.update({
                                "finish_reason": res["choices"][0].get("finish_reason"),
                                "reasoning": msg.get("reasoning_content") or "",
                                "content": msg.get("content") or "",
                                "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                               for c in msg.get("tool_calls") or []],
                                "timings": timings if args.keep_speed else
                                           {k: timings[k] for k in KEEP_TIMINGS if k in timings},
                            })
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        print(f"{row['task']:28} {name:12} T {temp} seed {seed}: "
                              f"{row.get('timings', {}).get('predicted_n', '-')} tokens"
                              f"{' ERROR' if 'error' in row else ''}", flush=True)


def signature(row):
    return (row.get("reasoning"), row.get("content"), json.dumps(row.get("tool_calls"), sort_keys=True))


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def cmd_compare(args):
    rows = [json.loads(line) for line in open(args.runs)]
    by = collections.defaultdict(dict)
    for r in rows:
        if "error" not in r:
            by[(r["task"], r["temperature"], r["seed"])][r["variant"]] = r
    stats = collections.defaultdict(lambda: [0, 0])
    diffs = []
    for (task, temp, seed), vs in sorted(by.items()):
        if args.a not in vs or args.b not in vs:
            continue
        same = signature(vs[args.a]) == signature(vs[args.b])
        stats[temp][0] += 1
        stats[temp][1] += same
        if not same:
            ta = vs[args.a]["reasoning"] + "\x00" + vs[args.a]["content"]
            tb = vs[args.b]["reasoning"] + "\x00" + vs[args.b]["content"]
            diffs.append((task, temp, seed, first_diff(ta, tb), len(ta), len(tb)))
    print(f"{args.a} against {args.b}")
    for temp, (n, same) in sorted(stats.items()):
        print(f"  T {temp}: {same} of {n} (task, seed) pairs identical (reasoning, content, calls)")
    for d in diffs[:args.show]:
        print(f"  differs: {d[0]} T {d[1]} seed {d[2]}: first difference at character {d[3]} of {d[4]} / {d[5]}")
    return 0 if all(n == same for n, same in stats.values()) else 1


def cmd_rejects(args):
    rows = [json.loads(line) for line in open(args.runs)]
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r.get("variant") != args.variant or "error" in r or "grammar_n" not in r.get("timings", {}):
            continue
        t = r["timings"]
        for k in ("predicted_n", "sampled_on_device", "grammar_n", "grammar_rejected_n"):
            agg[r["category"]][k] += t.get(k, 0)
            agg["all"][k] += t.get(k, 0)
        agg[r["category"]]["requests"] += 1
        agg["all"]["requests"] += 1
        agg[r["category"]]["with_call"] += bool(r["tool_calls"])
        agg["all"]["with_call"] += bool(r["tool_calls"])
    print(f"variant {args.variant}: tokens sampled while the grammar applied, and its rejections of the sampled token")
    print(f"{'category':20} {'requests':>8} {'calls':>6} {'tokens':>8} {'device':>8} {'grammar':>8} {'share':>7} "
          f"{'rejected':>8} {'of grammar':>10}")
    for cat in sorted(agg, key=lambda c: (c == "all", c)):
        a = agg[cat]
        share = a["grammar_n"] / a["predicted_n"] if a["predicted_n"] else 0.0
        rej = a["grammar_rejected_n"] / a["grammar_n"] if a["grammar_n"] else 0.0
        print(f"{cat:20} {a['requests']:8d} {a['with_call']:6d} {a['predicted_n']:8d} {a['sampled_on_device']:8d} "
              f"{a['grammar_n']:8d} {share:7.1%} {a['grammar_rejected_n']:8d} {rej:10.2%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--url", default="http://127.0.0.1:8089")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--temps", default="0,1")
    p.add_argument("--seeds", default="1")
    p.add_argument("--variants", nargs="+", default=["device:backend_sampling=true", "host:backend_sampling=false"])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--keep-speed", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("runs")
    p.add_argument("--a", default="device")
    p.add_argument("--b", default="host")
    p.add_argument("--show", type=int, default=20)
    p = sub.add_parser("rejects")
    p.add_argument("runs")
    p.add_argument("--variant", default="device")
    args = ap.parse_args()
    sys.exit({"run": cmd_run, "compare": cmd_compare, "rejects": cmd_rejects}[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
