#!/usr/bin/env python3
"""A continuation datastore behind the copy lookup (registry row Q161), replayed on Q51's held-out transcripts.

The datastore (datastore.cpp, loaded with ctypes) is a suffix array over the generated tokens of many Flint answers
(datastore_corpus.py). Its guess is the most frequent next token after the longest context, up to MAX_LEN tokens, that
occurs in it; policies back off to a context that occurs often enough, or guess only when that token's share is high.
The chains put it between the copy lookup and the request's own continuation counts, the two stages of the executor's
`cont` guesser (pipeline_sim.py's "copy >= 4, else request counts (no table)").

Runs:
  --selftest   a small store against a brute-force scan of its tokens (longest match, occurrences, top next token)
  --accuracy   one token ahead on the true prefix of every transcript: per class and segment the share of positions
               guessed and guessed right, fresh and copied text, and which stage of the chain made the guess; plus the
               datastore's work per query (suffix-array probes, entries scanned), for the lookup cost
  --sim        pipeline_sim.py's model of pipelined speculation on nine cards (both of its nine-card splits) at DEPTHS,
               today's `cont` against the chains with each store

usage: python datastore_sim.py GEN.jsonl --lib libdatastore.so --store NAME=FILE.ds [--store ...] [--selftest TOKENS.i32]
       [--accuracy] [--sim] [--policies all|LIST] [--depths 1,2,3,4] [--jobs N] [--out RESULTS.json]
"""
import argparse
import ctypes
import json
import random
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location("pipeline_sim", Path(__file__).with_name("pipeline_sim.py"))
psim = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_sim"] = psim   # so that the workers can unpickle pipeline_sim.run_rows
_spec.loader.exec_module(psim)
replay = psim.replay

MAX_LEN = 32
LIB = None


class Store:
    """A datastore file, loaded once (before the workers fork, so its pages are shared)."""

    def __init__(self, path):
        self.h = LIB.ds_load(path.encode())
        if not self.h:
            raise SystemExit(f"cannot load {path}")
        self.n = LIB.ds_size(self.h)
        self.occ = (ctypes.c_int32 * MAX_LEN)()
        self.tok = (ctypes.c_int32 * MAX_LEN)()
        self.cnt = (ctypes.c_int32 * MAX_LEN)()

    def query(self, ctx):
        """(longest length, occurrences, top next tokens, their counts) for the context, per length 1..longest."""
        ctx = ctx[-MAX_LEN:]
        arr = (ctypes.c_int32 * len(ctx))(*ctx)
        n = LIB.ds_query(self.h, arr, len(ctx), MAX_LEN, self.occ, self.tok, self.cnt)
        return n, self.occ[:n], self.tok[:n], self.cnt[:n]

    def longest(self, ctx):
        """(longest length, its occurrences, top next token, its count), counting next tokens in that range only."""
        ctx = ctx[-MAX_LEN:]
        arr = (ctypes.c_int32 * len(ctx))(*ctx)
        o, t, c = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        n = LIB.ds_longest(self.h, arr, len(ctx), MAX_LEN, ctypes.byref(o), ctypes.byref(t), ctypes.byref(c))
        return n, o.value, t.value, c.value

    def stats(self):
        out = (ctypes.c_int64 * 5)()
        LIB.ds_stats(self.h, out)
        return list(out)


def load_lib(path):
    global LIB
    LIB = ctypes.CDLL(path)
    LIB.ds_load.restype = ctypes.c_void_p
    LIB.ds_load.argtypes = [ctypes.c_char_p]
    LIB.ds_build.restype = ctypes.c_void_p
    LIB.ds_build.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int64]
    LIB.ds_size.restype = ctypes.c_int64
    LIB.ds_size.argtypes = [ctypes.c_void_p]
    LIB.ds_query.restype = ctypes.c_int
    LIB.ds_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                             ctypes.POINTER(ctypes.c_int32)]
    LIB.ds_longest.restype = ctypes.c_int
    LIB.ds_longest.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_int32)]
    LIB.ds_stats.restype = None
    LIB.ds_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]


# policy name -> (min context length, min occurrences of the context used, min share of its top next token)
POLICIES = {
    "longest": (1, 1, 0.0),
    "longest, L >= 2": (2, 1, 0.0),
    "longest, L >= 3": (3, 1, 0.0),
    "occurs >= 2": (1, 2, 0.0),
    "occurs >= 4": (1, 4, 0.0),
    "share >= 0.3": (1, 1, 0.3),
    "share >= 0.5": (1, 1, 0.5),
    "L >= 2, share >= 0.3": (2, 1, 0.3),
}


class DatastoreGuesser:
    """The top next token of the longest context (oldest token first: verified tokens, then the guesses in flight) that
    occurs at least min_occ times in the store, if that context has at least min_len tokens and the token holds at least
    min_share of its occurrences; else no guess."""

    def __init__(self, store, policy):
        self.store = store
        self.policy = policy
        self.min_len, self.min_occ, self.min_share = POLICIES[policy]

    def begin(self, full, index):
        pass

    def next(self, full, n_v, overlay):
        ctx = full[max(0, n_v - MAX_LEN):n_v] + overlay[-MAX_LEN:]
        if self.min_occ == 1:   # the longest length: ds_longest counts next tokens in its range only
            L, o, t, c = self.store.longest(ctx)
            if L < self.min_len or L == 0 or c < self.min_share * o:
                return None
            return t
        n, occ, tok, cnt = self.store.query(ctx)
        L = n
        while L >= self.min_len and L > 0 and occ[L - 1] < self.min_occ:
            L -= 1
        if L < self.min_len or L == 0:
            return None
        if cnt[L - 1] < self.min_share * occ[L - 1]:
            return None
        return tok[L - 1]


class StagedChain:
    """The first stage's guess, else the next one's; remembers which stage made the last guess."""

    def __init__(self, stages):
        self.stages = stages
        self.last = -1

    def begin(self, full, index):
        for s in self.stages:
            s.begin(full, index)

    def next(self, full, n_v, overlay):
        for i, s in enumerate(self.stages):
            g = s.next(full, n_v, overlay)
            if g is not None:
                self.last = i
                return g
        self.last = -1
        return None


def copy4():
    return psim.CopyGuesser(4)


def request_counts():
    return psim.TableGuesser(psim.EMPTY_VIEW, request=True)


class RequestCounts(psim.TableGuesser):
    """The request's own continuation counts, guessing only from a context of at least min_len tokens."""

    def __init__(self, min_len=1):
        super().__init__(psim.EMPTY_VIEW, request=True)
        self.min_len = min_len

    def next(self, full, n_v, overlay):
        g = super().next(full, n_v, overlay)
        if g is None or self.min_len <= 1:
            return g
        seq = full[max(0, n_v - self.max_ctx):n_v] + overlay[-self.max_ctx:]
        for ln in range(min(self.max_ctx, len(seq)), 0, -1):
            if tuple(seq[len(seq) - ln:]) in self.req:
                return g if ln >= self.min_len else None
        return None


class OracleSelect:
    """An upper bound for any rule that chooses between two guessers: the true token if either guesses it, else the
    first one's guess, else the second's (accuracy study only: it reads the transcript's next token)."""

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.last = 0

    def begin(self, full, index):
        self.first.begin(full, index)
        self.second.begin(full, index)

    def next(self, full, n_v, overlay):
        a = self.first.next(full, n_v, overlay)
        b = self.second.next(full, n_v, overlay)
        truth = full[n_v + len(overlay)]
        if truth in (a, b):
            return truth
        return a if a is not None else b


def chains(stores, policies, extras=False, kinds=("ds",)):
    """(name, stage names, factory) for the accuracy study and the simulation. kinds: "ds" puts the datastore right
    behind the copy lookup, "req2ds" behind the request counts of contexts of two tokens or more."""
    out = [("cont (copy >= 4, else request counts)", ("copy", "request"),
            lambda: StagedChain([copy4(), request_counts()]))]
    for sname, store in stores.items():
        for p in policies:
            if "ds" in kinds:
                out.append((f"copy >= 4, else datastore [{sname}, {p}], else request counts",
                            ("copy", "store", "request"),
                            lambda st=store, p=p: StagedChain([copy4(), DatastoreGuesser(st, p), request_counts()])))
            if "req2ds" in kinds:
                out.append((f"copy >= 4, else request counts (L >= 2), else datastore [{sname}, {p}], else request "
                            f"counts", ("copy", "request2", "store", "request"),
                            lambda st=store, p=p: StagedChain([copy4(), RequestCounts(2), DatastoreGuesser(st, p),
                                                               request_counts()])))
        if not extras:
            continue
        best = "L >= 2, share >= 0.3"
        out += [
            (f"copy >= 4, else request counts (L >= 3), else datastore [{sname}, {best}], else request counts",
             ("copy", "request3", "store", "request"),
             lambda st=store: StagedChain([copy4(), RequestCounts(3), DatastoreGuesser(st, best), request_counts()])),
            (f"copy >= 4, else request counts (L >= 2), else datastore [{sname}, {best}], else request counts",
             ("copy", "request2", "store", "request"),
             lambda st=store: StagedChain([copy4(), RequestCounts(2), DatastoreGuesser(st, best), request_counts()])),
            (f"copy >= 4, else datastore [{sname}, {best}], else request counts (L >= 2), else datastore "
             f"[{sname}, longest], else request counts", ("copy", "store", "request2", "store-any", "request"),
             lambda st=store: StagedChain([copy4(), DatastoreGuesser(st, best), RequestCounts(2),
                                           DatastoreGuesser(st, "longest"), request_counts()])),
            (f"copy >= 4, else oracle choice of datastore [{sname}, longest] and request counts (bound)",
             ("copy", "oracle"),
             lambda st=store: StagedChain([copy4(), OracleSelect(DatastoreGuesser(st, "longest"), request_counts())])),
        ]
    return out


# ---------------------------------------------------------------------------------------------------------------------

def selftest(tokens_path, n_tok=40000, n_q=300, seed=5):
    """Builds a store from the first answers of a token stream and checks every query against a brute-force scan."""
    raw = open(tokens_path, "rb").read(4 * n_tok)
    toks = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    while toks and toks[-1] != -1:
        toks.pop()
    arr = (ctypes.c_int32 * len(toks))(*toks)
    h = LIB.ds_build(arr, len(toks))
    store = Store.__new__(Store)
    store.h, store.n = h, len(toks)
    store.occ, store.tok, store.cnt = ((ctypes.c_int32 * MAX_LEN)() for _ in range(3))
    rng = random.Random(seed)
    bad = 0
    for qi in range(n_q):
        # half the contexts from the stream itself (long matches), half with a changed token (short ones)
        e = rng.randrange(MAX_LEN, len(toks))
        ctx = toks[e - MAX_LEN:e]
        if -1 in ctx:
            ctx = ctx[len(ctx) - ctx[::-1].index(-1):]
        if not ctx:
            continue
        if qi % 2:
            ctx = list(ctx)
            ctx[rng.randrange(len(ctx))] = rng.randrange(1000)
        n, occ, tok, cnt = store.query(ctx)
        # brute force: the end positions of the context's last L tokens, narrowed one token at a time; the next token
        # after each (not a separator, not past the end)
        exp = []
        ends = [i for i in range(len(toks) - 1) if toks[i] == ctx[-1]]
        for L in range(1, min(len(ctx), MAX_LEN) + 1):
            if L > 1:
                ends = [i for i in ends if i - L + 1 >= 0 and toks[i - L + 1] == ctx[-L]]
            c = Counter(toks[i + 1] for i in ends if toks[i + 1] >= 0)
            if not c:
                break
            top = max(c.items(), key=lambda kv: (kv[1], -kv[0]))
            exp.append((sum(c.values()), top[0], top[1]))
        got = list(zip(occ, tok, cnt))
        lg = store.longest(ctx)
        if (lg[0], tuple(lg[1:])) != (len(exp), exp[-1] if exp else (0, 0, 0)) and not (lg[0] == 0 and not exp):
            bad += 1
            print(f"selftest: ds_longest {lg} against {len(exp)}, {exp[-1:]}", file=sys.stderr)
        if got != exp:
            bad += 1
            if bad <= 3:
                print(f"selftest mismatch: ctx {ctx[-6:]}…: got {got[:4]}, expected {exp[:4]}", file=sys.stderr)
    print(f"selftest: {n_q} queries on a store of {len(toks)} tokens, {bad} mismatches")
    return bad == 0


# ---------------------------------------------------------------------------------------------------------------------

ACC_KEYS = ("chat", "code", "edit", "agent", "all", "edit/answer", "agent/tool")


def in_copy_of(row):
    prompt, gen = row["prompt_tokens"], row["gen_tokens"]
    flags = [False] * len(gen)
    for start, ln, _ in replay.copy_factors(prompt, gen, 8):
        for j in range(start, start + ln):
            flags[j] = True
    return flags


_SHARED = None


def acc_rows(idx):
    rows, chs = _SHARED
    acc = defaultdict(Counter)
    for ri in idx:
        row = rows[ri]
        prompt, gen = row["prompt_tokens"], row["gen_tokens"]
        full = prompt + gen
        index = psim.SeqIndex(full)
        flags = in_copy_of(row)
        for cname, stage_names, make in chs:
            g = make()
            g.begin(full, index)
            for j in range(1, len(gen)):
                guess = g.next(full, len(prompt) + j, [])
                where = "copy" if flags[j] else "fresh"
                stage = stage_names[g.last] if guess is not None else "none"
                for key in (row["category"], "all", row["category"] + "/" + row["segs"][j]):
                    c = acc[(cname, key)]
                    c[where] += 1
                    c[f"{where}_guessed"] += guess is not None
                    c[f"{where}_right"] += guess == gen[j]
                    c[f"stage_{stage}"] += 1
                    c[f"stage_{stage}_right"] += guess == gen[j]
        print(f"accuracy {row['id']}", file=sys.stderr)
    return {f"{k[0]}|{k[1]}": dict(v) for k, v in acc.items()}


def accuracy(rows, stores, policies, jobs, extras=False, kinds=("ds",)):
    global _SHARED
    chs = chains(stores, policies, extras, kinds)
    _SHARED = (rows, chs)
    before = {n: s.stats() for n, s in stores.items()}
    tasks = [sl for sl in (list(range(len(rows)))[k::jobs] for k in range(jobs)) if sl]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            parts = pool.map(acc_rows, tasks)
    else:
        parts = [acc_rows(t) for t in tasks]
    tot = defaultdict(Counter)
    for part in parts:
        for k, d in part.items():
            tot[k] += Counter(d)
    out = {}
    for k, c in tot.items():
        cname, key = k.split("|", 1)
        n = c["fresh"] + c["copy"]
        r = {"n": n, "right": (c["fresh_right"] + c["copy_right"]) / n,
             "guessed": (c["fresh_guessed"] + c["copy_guessed"]) / n,
             "fresh_n": c["fresh"], "fresh_guessed": c["fresh_guessed"] / max(1, c["fresh"]),
             "fresh_right": c["fresh_right"] / max(1, c["fresh"]), "copy_right": c["copy_right"] / max(1, c["copy"]),
             "stages": {s[6:]: [c[s] / n, c[s + "_right"] / max(1, c[s])] for s in c
                        if s.startswith("stage_") and not s.endswith("_right")}}
        out.setdefault(cname, {})[key] = r
    work = {}
    if jobs == 1:   # the counters live in the process that queried
        for n, s in stores.items():
            a, b = before[n], s.stats()
            q = b[0] - a[0]
            work[n] = {"queries": q, "probes_per_query": (b[1] - a[1]) / max(1, q),
                       "scanned_per_query": (b[2] - a[2]) / max(1, q), "cache_hits_per_query": (b[3] - a[3]) / max(1, q),
                       "cache_entries": b[4], "tokens": s.n, "bytes": 8 * s.n}
    return out, work


def print_accuracy(res):
    print("\n| Guesser | Right: chat / code / edit / agent / all | Fresh right: chat / code / edit / agent / all |"
          " File rewrites right | Tool calls right | Guessed (all) | Datastore guesses: share / right |")
    print("|---|---|---|---|---|---|---|")
    for cname, by in res.items():
        right = " / ".join(f"{100 * by[k]['right']:.1f}" for k in ("chat", "code", "edit", "agent", "all"))
        fresh = " / ".join(f"{100 * by[k]['fresh_right']:.1f}" for k in ("chat", "code", "edit", "agent", "all"))
        st = by["all"]["stages"].get("store")
        sts = f"{100 * st[0]:.1f} % / {100 * st[1]:.1f} %" if st else "–"
        rw = by.get("edit/answer", {}).get("right", 0)
        tc = by.get("agent/tool", {}).get("right", 0)
        print(f"| {cname} | {right} | {fresh} | {100 * rw:.1f} | {100 * tc:.1f} | {100 * by['all']['guessed']:.1f} % |"
              f" {sts} |")


# ---------------------------------------------------------------------------------------------------------------------

def simulate(rows, stores, policies, depths, jobs, kinds=("ds",)):
    setups = [s for s in psim.SETUPS if s[0].startswith("9 cards")]
    configs = []
    for si, (sname, total, layers) in enumerate(setups):
        for d in depths:
            for cname, _, make in chains(stores, policies, kinds=kinds):
                configs.append((f"{sname} | {cname} | depth {d}", si, lambda e, make=make: make(), d,
                                psim.HOP_MS, psim.GAP_MS, 1.0))
    plain = {}
    for cfg, si, make, depth, hop, gap, scale in configs:
        st = psim.stage_times(setups[si][1], setups[si][2], hop, gap)
        plain[cfg] = gap + sum(st) + (len(st) - 1) * hop
    table = psim.ContinuationTable(rows[:1], members=[])   # unused by these guessers
    psim._SHARED = (rows, setups, configs, table, None)
    tasks = [sl for sl in (list(range(len(rows)))[k::jobs] for k in range(jobs)) if sl]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            parts = pool.map(psim.run_rows, tasks)
    else:
        parts = [psim.run_rows(t) for t in tasks]
    runs = psim.Runs()
    for part in parts:
        for c, d in part["t"].items():
            for k, x in d.items():
                runs.t[c][k] += x
        for c, d in part["n"].items():
            for k, x in d.items():
                runs.n[c][k] += x
        for c, d in part["stats"].items():
            runs.stats[c] += Counter(d)
    keys = psim.KEYS + ("agent/tool",)
    out = {}
    for cfg, *_ in configs:
        s = runs.stats[cfg]
        n_all = runs.n[cfg]["all"]
        out[cfg] = {"speedup": {k: round(runs.speedup(cfg, k, plain[cfg]), 4) for k in keys},
                    "tok_s": {k: round(runs.speedup(cfg, k, plain[cfg]) * 1000 / plain[cfg], 2) for k in keys},
                    "hit_share": s["hit"] / n_all, "miss_share": s["miss"] / n_all,
                    "wasted_ms_per_token": s["wasted_ms"] / n_all}
    return out


def print_sim(res):
    names = ("chat", "code", "edit", "agent", "all", "edit/answer", "agent/tool")
    print("\n| Config | Chat | Code | Edits | Agent | All | File rewrites | Tool calls | Right guesses | Wrong guesses |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cfg, r in res.items():
        print(f"| {cfg} | " + " | ".join(f"{r['speedup'][k]:.2f}" for k in names) +
              f" | {100 * r['hit_share']:.1f} % | {100 * r['miss_share']:.1f} % |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--store", action="append", default=[], help="NAME=FILE.ds")
    ap.add_argument("--selftest", help="a token stream (int32, answers separated by -1) for the brute-force check")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--policies", default="all", help="policy names separated by semicolons, or all")
    ap.add_argument("--extras", action="store_true", help="accuracy: also other chain orders and the oracle-choice bound")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--kinds", default="ds", help="comma-separated chain kinds for --accuracy/--sim: ds, req2ds")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    load_lib(args.lib)
    res = {}
    if args.selftest:
        res["selftest_ok"] = selftest(args.selftest)
    rows = replay.load(args.gen)
    stores = {}
    for spec in args.store:
        name, path = spec.split("=", 1)
        stores[name] = Store(path)
    policies = list(POLICIES) if args.policies == "all" else args.policies.split(";")
    if args.accuracy:
        res["accuracy"], res["work"] = accuracy(rows, stores, policies, args.jobs, args.extras, args.kinds.split(","))
        print_accuracy(res["accuracy"])
        for n, w in res["work"].items():
            print(f"work {n}: {w}")
    if args.sim:
        res["sim"] = simulate(rows, stores, policies, [int(x) for x in args.depths.split(",")], args.jobs,
                              args.kinds.split(","))
        print_sim(res["sim"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
