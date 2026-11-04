#!/usr/bin/env python3
"""taper_replay.py: the agent check's prompts, replayed against a llama-server on this machine (the rig), with the
micro-batch taper off and on per request ("ubatch_taper"), so that both arms run on one model load.

The request plan is the agent check's own (Q218: agent/turns.csv of docs/data/2026-11-03_retype_agent, arm hcq8):
12 tasks, 62 turns; turn t of a task sends prompt_n new tokens on top of the task's history. Each task starts from the
same 1,549-token prefix (the agent's system prompt and tool schemas, which the check had in the slot): an unmeasured
request evaluates it first (n_predict 0), except for a turn 1 with nothing cached (the run's cold first request),
which goes out cold. Every measured request asks for GEN tokens (greedy) instead of the agent's reply, so the context
is shallower by the replies' length (prompt time showed no depth effect below ~6.8k tokens; PAD=1 adds the missing
tokens with unmeasured requests). The text is Python standard-library source from this machine, tokenized by the
server, the same tokens in every pass.

Passes run in the given order (default off, on, on, off, so the n-gram table's page cache warms both arms alike).
Fresh prompts (--fresh K): K texts of --fresh-n tokens with nothing cached, each run with the taper off and on (the
order alternating), a different first token per run so the slot cannot reuse the other run.

Output in --out: requests.tsv (one row per request), summary.txt (per arm: prompt time and tok/s over the replayed
turns, by the agent check's size bins, on/off per turn paired by task and turn; the fresh prompts), and the summary
on stdout. Standard library only.
"""
import argparse, csv, glob, hashlib, json, statistics, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--turns", required=True, help="agent/turns.csv of the agent check")
ap.add_argument("--arm", default="hcq8", help="the arm of turns.csv to replay")
ap.add_argument("--passes", default="off on on off", help="space-separated: off, on or a taper spec (e.g. short=2048:128)")
ap.add_argument("--gen", type=int, default=8)
ap.add_argument("--pad", type=int, default=0)
ap.add_argument("--fresh", type=int, default=4)
ap.add_argument("--fresh-n", type=int, default=2048)
ap.add_argument("--prefix", type=int, default=1549)
ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

URL = f"http://127.0.0.1:{a.port}"


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# the plan: tasks in file order, turns in order
turns = [r for r in csv.DictReader(open(a.turns)) if r["arm"] == a.arm]
want = set(a.tasks.split(",")) if a.tasks else None
tasks = []
for r in turns:
    if want and r["task_id"] not in want:
        continue
    if not tasks or tasks[-1][0] != r["task_id"]:
        tasks.append((r["task_id"], []))
    tasks[-1][1].append(r)

need = a.prefix + sum(int(r["prompt_n"]) for _, ts in tasks for r in ts) + a.fresh * (a.fresh_n + 8) + 64
if a.pad:
    need += sum(max(0, int(r["predicted_n"]) - a.gen) for _, ts in tasks for r in ts)

# the token pool: standard-library source, tokenized by the server
pool = []
for f in sorted(glob.glob("/usr/lib/python3*/*.py")):
    if len(pool) >= need:
        break
    text = "# file " + f.split("/")[-1] + "\n" + open(f, errors="replace").read()
    pool += post("/tokenize", {"content": text, "add_special": False})["tokens"]
if len(pool) < need:
    sys.exit(f"only {len(pool)} tokens of text, {need} needed")
prefix = pool[:a.prefix]
off_task = a.prefix
off_fresh = need - a.fresh * (a.fresh_n + 8) - 64

rows = []
out_tsv = open(f"{a.out}/requests.tsv", "w")
cols = ["label", "pass", "arm", "kind", "task", "turn", "n_new", "prompt_n", "cache_n", "prompt_ms", "predicted_n",
        "predicted_ms", "wall_s", "gen_md5"]
out_tsv.write("\t".join(cols) + "\n")


def spec(arm):
    return arm  # "off", "on" or a spec, passed to the server as is


def run(pass_i, arm, kind, task, turn, prompt, n_new, n_predict):
    body = dict(prompt=prompt, n_predict=n_predict, temperature=0, top_k=1, seed=1, cache_prompt=True,
                return_tokens=True, ubatch_taper=spec(arm))
    t0 = time.time()
    res = post("/completion", body)
    wall = time.time() - t0
    t = res.get("timings", {})
    gen = res.get("tokens", [])
    row = dict(label=a.label, **{"pass": pass_i}, arm=arm, kind=kind, task=task, turn=turn, n_new=n_new,
               prompt_n=t.get("prompt_n"), cache_n=t.get("cache_n"), prompt_ms=round(t.get("prompt_ms", 0), 1),
               predicted_n=t.get("predicted_n"), predicted_ms=round(t.get("predicted_ms", 0), 1),
               wall_s=round(wall, 2), gen_md5=hashlib.md5(json.dumps(gen).encode()).hexdigest()[:10])
    out_tsv.write("\t".join(str(row[c]) for c in cols) + "\n")
    out_tsv.flush()
    rows.append(row)
    return gen


for pi, arm in enumerate(a.passes.split()):
    t_pass = time.time()
    off = off_task
    for task, ts in tasks:
        history = []
        for r in ts:
            n = int(r["prompt_n"])
            if int(r["turn"]) == 1:
                if int(r["cache_n"]) == 0:
                    history = list(prefix)
                    n = max(1, n - a.prefix)  # cold: the prefix goes out with the turn
                else:
                    run(pi, arm, "reset", task, 0, list(prefix), 0, 0)
                    history = list(prefix)
            new = pool[off:off + n]
            off += n
            prompt = history + new
            gen = run(pi, arm, "turn", task, int(r["turn"]), prompt, n, a.gen)
            history = prompt + gen
            if a.pad:
                p = max(0, int(r["predicted_n"]) - a.gen)
                if p:
                    history += pool[off:off + p]
                    off += p
                    run(pi, arm, "pad", task, int(r["turn"]), history, p, 0)
    print(f"pass {pi} ({arm}): {time.time() - t_pass:.0f} s", flush=True)

# fresh prompts: nothing cached, the order alternating, a run-specific first token
for k in range(a.fresh):
    text = pool[off_fresh + k * (a.fresh_n + 8): off_fresh + k * (a.fresh_n + 8) + a.fresh_n]
    order = ["off", "on"] if k % 2 == 0 else ["on", "off"]
    m0 = pool[(7 + 31 * k) % len(prefix)]
    m1 = next(x for x in prefix if x != m0)
    for j, arm in enumerate(order):
        run(len(a.passes.split()), arm, "fresh", f"fresh{k}", j, [(m0, m1)[j]] + text, len(text) + 1, a.gen)
out_tsv.close()

# summary
BINS = [(1, 64), (65, 255), (256, 511), (512, 1023), (1024, 1999), (2000, 10 ** 9)]


def med(v):
    return statistics.median(v) if v else float("nan")


lines = []
arms = sorted({r["arm"] for r in rows if r["kind"] == "turn"}, key=lambda x: (x != "off", x))
per = {}  # (arm, task, turn) -> [prompt_ms]
for r in rows:
    if r["kind"] == "turn":
        per.setdefault((r["arm"], r["task"], r["turn"]), []).append(r)
for arm in arms:
    rs = [r for r in rows if r["kind"] == "turn" and r["arm"] == arm]
    n_tok = sum(r["prompt_n"] or 0 for r in rs)
    ms = sum(r["prompt_ms"] for r in rs)
    passes = len({r["pass"] for r in rs})
    lines.append(f"{arm}: {passes} passes, {len(rs)} turns, {n_tok} prompt tokens in {ms / 1000:.1f} s "
                 f"({n_tok / max(ms, 1) * 1000:.1f} tok/s); per pass {ms / 1000 / max(passes, 1):.1f} s")
if "off" in arms:
    for arm in arms:
        if arm == "off":
            continue
        ratios, saved, by_bin = [], 0.0, {}
        for (ar, task, turn), rs in per.items():
            if ar != arm or ("off", task, turn) not in per:
                continue
            on_ms = statistics.mean(r["prompt_ms"] for r in rs)
            off_ms = statistics.mean(r["prompt_ms"] for r in per[("off", task, turn)])
            n = rs[0]["prompt_n"] or 0
            ratios.append(off_ms / max(on_ms, 1e-3))
            saved += off_ms - on_ms
            for lo, hi in BINS:
                if lo <= n <= hi:
                    by_bin.setdefault((lo, hi), []).append((off_ms, on_ms))
        lines.append(f"{arm} against off, paired by task and turn: {len(ratios)} turns, speed-up median "
                     f"{med(ratios):.2f}x, {saved / 1000:.1f} s saved per pass")
        for (lo, hi), v in sorted(by_bin.items()):
            lines.append(f"  {lo}-{hi if hi < 10 ** 9 else 'up'} new tokens: {len(v)} turns, off median "
                         f"{med([x for x, _ in v]) / 1000:.2f} s, {arm} median {med([y for _, y in v]) / 1000:.2f} s, "
                         f"sum {sum(x for x, _ in v) / 1000:.1f} -> {sum(y for _, y in v) / 1000:.1f} s")
same = sum(1 for (ar, task, turn), rs in per.items() if ar != "off" and ("off", task, turn) in per
           and {r["gen_md5"] for r in rs} == {r["gen_md5"] for r in per[("off", task, turn)]})
lines.append(f"greedy output identical to off's on {same} of {sum(1 for k in per if k[0] != 'off')} turns of the other arms")
fr = [r for r in rows if r["kind"] == "fresh"]
for arm in ("off", "on"):
    v = [r["prompt_ms"] for r in fr if r["arm"] == arm]
    if v:
        lines.append(f"fresh {a.fresh_n + 1}-token prompts, {arm}: median {med(v) / 1000:.2f} s "
                     f"({', '.join(f'{x / 1000:.2f}' for x in v)})")
open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
