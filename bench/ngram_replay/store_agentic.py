#!/usr/bin/env python3
"""The answer store (registry row Q161) on agent sessions: how often each guess source guesses and is right, one token
ahead, and the speed of pipelined decoding with and without the store on a measured card setup.

Input rows:
- session rows of bench/agentic_data (export.py's sessions_seq.jsonl, heldout.py's --test-rows): prompt_tokens,
  gen_tokens and gen_ranges, the session's generated spans in absolute positions. Each span becomes one request, as
  llama-server sees an agent loop: the prompt is the session up to the span (system prompt, task, earlier turns with
  their reasoning, tool results), the output is the span;
- answer rows (Q51's gen.jsonl: prompt_tokens, gen_tokens, category) as they are.

The guess sources follow Miner Llama's (common/pipe-guess.cpp and common/pipe-store.cpp):
  copy      the copy lookup: the token after the longest earlier match, 4 to 12 tokens, of the text's end
  request2  the request's own continuation counts (the prompt's last 32,768 tokens and the output so far): the most
            frequent token after the longest 2..8-token context
  store     the store's most frequent token after the longest context, up to 32 tokens, that it holds, if that context
            has at least MIN_LEN tokens and the token holds at least MIN_SHARE of its occurrences
  request   the request's continuation counts of the longest 1..8-token context
`cont` (the engine's default guesser) is copy, else request. With --pipe-store it is copy, request2, store, request;
the engine compiles in MIN_LEN 2 and MIN_SHARE 0.3. --grid tries other thresholds.

Runs:
  --accuracy  one token ahead on the true prefix: per workload (class and segment: reasoning, tool call, the rest),
              the share of positions guessed and guessed right, on fresh and on copied text (copies of >= 8 tokens),
              and per source the share of positions it guessed and how often it was right
  --sim       pipeline_sim.py's model of pipelined decoding on --setup (plain ms per token, layers per card; the
              output head on the last card) at --depths, without a guess, with `cont` and with each store

usage: python store_agentic.py ROWS.jsonl [...] --lib libdatastore.so --store NAME=FILE [...] [--grid LENS:SHARES]
       [--accuracy] [--sim] [--sim-grid] [--setup NAME=MS:L1,L2,...] [--depths 4] [--jobs N] [--out RESULTS.json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location("datastore_sim", Path(__file__).with_name("datastore_sim.py"))
dsim = importlib.util.module_from_spec(_spec)
sys.modules["datastore_sim"] = dsim
_spec.loader.exec_module(dsim)
psim, replay = dsim.psim, dsim.replay

MAX_COUNT = 32768    # the engine counts continuations over the prompt's last 32,768 tokens (common_pipe_guesser)
ENGINE = (2, 0.3)    # the engine's store thresholds: context of 2 tokens or more, top token >= 30 % of occurrences


def session_rows(r):
    """One request per generated span of a session row."""
    full = r["prompt_tokens"] + r["gen_tokens"]
    out = []
    for k, (s, e) in enumerate(r["gen_ranges"]):
        gen = full[s:e]
        out.append({"id": f"{r['id']}/{k}", "category": "agentic", "prompt_tokens": full[:s], "gen_tokens": gen,
                    "segs": replay.segments_of(gen)})
    return out


def load(paths):
    rows = []
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if "gen_ranges" in r:
                rows += session_rows(r)
            else:
                r["segs"] = replay.segments_of(r["gen_tokens"])
                rows.append(r)
    return rows


class RequestCounts(dsim.RequestCounts):
    """The request's continuation counts over the prompt's last MAX_COUNT tokens and the output, as the engine keeps
    them; min_len as in datastore_sim."""

    def next(self, full, n_v, overlay):
        if self.n_counted == 0:     # the first guess of the request: n_v is the prompt plus the first sampled token
            self.n_counted = max(0, n_v - 1 - MAX_COUNT)
        return super().next(full, n_v, overlay)


def policy(min_len, min_share):
    name = f"L >= {min_len}, share >= {min_share}"
    dsim.POLICIES[name] = (min_len, 1, min_share)
    return name


def chains(stores, grid):
    """(name, stage names, factory): `cont`, then per store the engine's order at each threshold pair."""
    out = [("cont", ("copy", "request"), lambda: dsim.StagedChain([dsim.copy4(), RequestCounts()]))]
    for sname, store in stores.items():
        for ml, sh in grid:
            p = policy(ml, sh)
            out.append((f"cont + store [{sname}, {p}]", ("copy", "request2", "store", "request"),
                        lambda st=store, p=p: dsim.StagedChain([dsim.copy4(), RequestCounts(2),
                                                                dsim.DatastoreGuesser(st, p), RequestCounts()])))
    return out


def keys_of(row, j):
    c = row["category"]
    return c, "all", c + "/" + row["segs"][j]


# ---------------------------------------------------------------------------------------------------------------------

_SHARED = None


def acc_rows(idx):
    """Every source once per position (one token ahead, nothing in flight), then each chain from those guesses: the
    chains' stages are the same lookups in a different order or with other store thresholds."""
    rows, stores, grid = _SHARED
    names = {(s, ml, sh): (f"cont + store [{s}, {policy(ml, sh)}]", f"copy, store [{s}, {policy(ml, sh)}], request")
             for s in stores for ml, sh in grid}
    acc = defaultdict(Counter)
    for ri in idx:
        row = rows[ri]
        prompt, gen = row["prompt_tokens"], row["gen_tokens"]
        full = prompt + gen
        index = psim.SeqIndex(full)
        flags = dsim.in_copy_of(row)
        copy, req = dsim.copy4(), RequestCounts()
        copy.begin(full, index)
        req.begin(full, index)
        for j in range(1, len(gen)):
            n_v = len(prompt) + j
            g_copy = copy.next(full, n_v, [])
            g_any = req.next(full, n_v, [])
            g_req2 = None
            if g_any is not None:
                for ln in range(min(req.max_ctx, n_v), 1, -1):
                    if tuple(full[n_v - ln:n_v]) in req.req:
                        g_req2 = g_any
                        break
            found = {s: st.longest(full[max(0, n_v - dsim.MAX_LEN):n_v]) for s, st in stores.items()}
            picks = [("cont", ((g_copy, "copy"), (g_any, "request")))]
            for s, (L, o, t, c) in found.items():
                for ml, sh in grid:
                    g_st = t if L >= ml and L > 0 and c >= sh * o else None
                    eng, first = names[(s, ml, sh)]
                    picks.append((eng, ((g_copy, "copy"), (g_req2, "request2"), (g_st, "store"), (g_any, "request"))))
                    if (ml, sh) == ENGINE:
                        picks.append((first, ((g_copy, "copy"), (g_st, "store"), (g_any, "request"))))
            where = "copy" if flags[j] else "fresh"
            for cname, stages in picks:
                guess, stage = next(((g, st) for g, st in stages if g is not None), (None, "none"))
                for key in keys_of(row, j):
                    c = acc[(cname, key)]
                    c[where] += 1
                    c[f"{where}_guessed"] += guess is not None
                    c[f"{where}_right"] += guess == gen[j]
                    c[f"stage_{stage}"] += 1
                    c[f"stage_{stage}_right"] += guess == gen[j]
        print(f"accuracy {row['id']}", file=sys.stderr)
    return {f"{k[0]}|{k[1]}": dict(v) for k, v in acc.items()}


def fork_map(fn, n_rows, jobs):
    tasks = [sl for sl in (list(range(n_rows))[k::jobs] for k in range(jobs)) if sl]
    if jobs == 1:
        return [fn(t) for t in tasks]
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(fn, tasks)


def accuracy(rows, stores, grid, jobs):
    global _SHARED
    _SHARED = (rows, stores, grid)
    tot = defaultdict(Counter)
    for part in fork_map(acc_rows, len(rows), jobs):
        for k, d in part.items():
            tot[k] += Counter(d)
    out = {}
    for k, c in tot.items():
        cname, key = k.split("|", 1)
        n = c["fresh"] + c["copy"]
        out.setdefault(cname, {})[key] = {
            "n": n, "right": (c["fresh_right"] + c["copy_right"]) / n,
            "guessed": (c["fresh_guessed"] + c["copy_guessed"]) / n,
            "fresh_share": c["fresh"] / n, "fresh_guessed": c["fresh_guessed"] / max(1, c["fresh"]),
            "fresh_right": c["fresh_right"] / max(1, c["fresh"]), "copy_right": c["copy_right"] / max(1, c["copy"]),
            "stages": {s[6:]: [c[s] / n, c[s + "_right"] / max(1, c[s])] for s in c
                       if s.startswith("stage_") and not s.endswith("_right")}}
    return out


def print_accuracy(res):
    keys = sorted({k for by in res.values() for k in by}, key=lambda k: (k != "all", k))
    print("\n| Guesser | Workload | Positions | Right | Guessed | Fresh text: share / guessed / right | Copied text right |"
          " copy: share / right | request2 | store | request |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cname, by in res.items():
        for k in keys:
            r = by.get(k)
            if not r:
                continue
            st = [r["stages"].get(s) for s in ("copy", "request2", "store", "request")]
            sts = " | ".join(f"{100 * s[0]:.1f} / {100 * s[1]:.1f}" if s else "–" for s in st)
            print(f"| {cname} | {k} | {r['n']} | {100 * r['right']:.1f} % | {100 * r['guessed']:.1f} % | "
                  f"{100 * r['fresh_share']:.1f} / {100 * r['fresh_guessed']:.1f} / {100 * r['fresh_right']:.1f} % | "
                  f"{100 * r['copy_right']:.1f} % | {sts} |")


# ---------------------------------------------------------------------------------------------------------------------

def simulate(rows, chs, setups, depths, jobs):
    configs = []
    for si, (sname, total, layers) in enumerate(setups):
        for d in depths:
            for cname, _, make in chs:
                configs.append((f"{sname} | {cname} | depth {d}", si, lambda e, make=make: make(), d,
                                psim.HOP_MS, psim.GAP_MS, 1.0))
    plain = {}
    for cfg, si, make, depth, hop, gap, scale in configs:
        st = psim.stage_times(setups[si][1], setups[si][2], hop, gap)
        plain[cfg] = gap + sum(st) + (len(st) - 1) * hop
    table = psim.ContinuationTable(rows[:1], members=[])   # unused by these guessers
    psim._SHARED = (rows, setups, configs, table, None)
    runs = psim.Runs()
    for part in fork_map(psim.run_rows, len(rows), jobs):
        for c, d in part["t"].items():
            for k, x in d.items():
                runs.t[c][k] += x
        for c, d in part["n"].items():
            for k, x in d.items():
                runs.n[c][k] += x
        for c, d in part["stats"].items():
            runs.stats[c] += Counter(d)
    out = {}
    for cfg, *_ in configs:
        s, n_all = runs.stats[cfg], runs.n[cfg]["all"]
        keys = sorted(runs.n[cfg], key=lambda k: (k != "all", k))
        out[cfg] = {"plain_ms": plain[cfg],
                    "speedup": {k: round(runs.speedup(cfg, k, plain[cfg]), 4) for k in keys},
                    "tok_s": {k: round(runs.speedup(cfg, k, plain[cfg]) * 1000 / plain[cfg], 2) for k in keys},
                    "hit_share": s["hit"] / n_all, "miss_share": s["miss"] / n_all,
                    "wasted_ms_per_token": s["wasted_ms"] / n_all}
    return out


def print_sim(res):
    keys = sorted({k for r in res.values() for k in r["speedup"]}, key=lambda k: (k != "all", k))
    print("\n| Config | " + " | ".join(keys) + " | tok/s, all | Right guesses | Wrong guesses | Cancelled card-ms per token |")
    print("|---|" + "---|" * (len(keys) + 4))
    for cfg, r in res.items():
        print(f"| {cfg} | " + " | ".join(f"{r['speedup'].get(k, 0):.3f}" for k in keys) +
              f" | {r['tok_s']['all']:.1f} | {100 * r['hit_share']:.1f} % | {100 * r['miss_share']:.1f} % |"
              f" {r['wasted_ms_per_token']:.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--store", action="append", default=[], help="NAME=FILE")
    ap.add_argument("--grid", default="2:0.3", help="store thresholds MIN_LENS:MIN_SHARES, e.g. 1,2,3,4:0,0.2,0.3,0.5")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--sim-grid", default=None, help="thresholds for --sim (default: the engine's, 2:0.3)")
    ap.add_argument("--setup", action="append", default=[], help="NAME=MS:L1,L2,... (plain ms per token, layers per "
                    "card; default the release split of Flint Q8_0 on nine cards)")
    ap.add_argument("--depths", default="4")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    dsim.load_lib(args.lib)
    rows = load(args.rows)
    print(f"{len(rows)} requests, {sum(len(r['gen_tokens']) for r in rows)} generated tokens, "
          f"{sum(len(r['prompt_tokens']) for r in rows)} prompt tokens", file=sys.stderr)
    stores = {}
    for spec in args.store:
        name, path = spec.split("=", 1)
        stores[name] = dsim.Store(path)

    def grid(spec):
        lens, shares = spec.split(":")
        return [(int(a), float(b)) for a in lens.split(",") for b in shares.split(",")]
    res = {"requests": len(rows), "generated": sum(len(r["gen_tokens"]) for r in rows), "stores": {}}
    for n, s in stores.items():
        res["stores"][n] = {"tokens": s.n}
    if args.accuracy:
        res["accuracy"] = accuracy(rows, stores, grid(args.grid), args.jobs)
        print_accuracy(res["accuracy"])
    if args.sim:
        setups = []
        for spec in args.setup or ["9 cards, -ts 6,6,5,5,6,5,5,6,6 (layers 6/6/5/5/6/5/5/6/5 + head)="
                                   "34.1:6,6,5,5,6,5,5,6,5"]:
            name, rest = spec.rsplit("=", 1)
            ms, layers = rest.split(":")
            setups.append((name, float(ms), [int(x) for x in layers.split(",")]))
        sim_grid = grid(args.sim_grid) if args.sim_grid else [ENGINE]
        res["sim"] = simulate(rows, chains(stores, sim_grid), setups, [int(x) for x in args.depths.split(",")],
                              args.jobs)
        print_sim(res["sim"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
