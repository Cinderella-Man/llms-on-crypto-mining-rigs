#!/usr/bin/env python3
"""Simulate pipelined speculation (registry row Q87) on transcripts Flint generated itself.

On a layer split, one card works at a time: token t passes card 1, then card 2, and so on, and the last card samples
token t+1. Pipelined speculation feeds a guess for token t+1 into card 1 as soon as card 1 has finished token t, the next
guess one stage later, and so on. Every guess is an ordinary one-token step. When the last card samples the real token,
a matching guess is already on its way; a wrong one is cancelled together with everything behind it, and the sampled
token enters card 1. Sample-and-match keeps the output identical to plain decoding
(docs/research/notes/breakthrough_candidates.md, candidate 1).

The model, per transcript:
- Card j runs one stage per position, s_j ms: its layers times a per-layer time, plus the output head and sampling on the
  last card. Positions pass the cards in order; a position starts on card j when it has left card j-1 plus a hop (copy
  and launch) and card j has finished the previous position. Plain decoding takes T = gap + sum(s_j) + (S-1) hop per
  token, where gap is the host's time from a sample to the next launch on card 1.
- The host launches each stage once its input has arrived and its card is free, one hop after the later of the two, so
  with every card busy a token leaves the pipeline once per slowest stage plus a hop. At most `depth` guessed positions
  are in flight beyond the newest verified token; a guess is launched on card 1 one hop after a depth slot opens. The
  sampled token after a miss or after no guess enters card 1 `gap` after its sample.
- A miss cancels every position behind it. Stages already running finish (CUDA work cannot be withdrawn); stages not yet
  started are dropped.
- Guessers see the verified tokens plus the guesses in flight: the copy lookup of `ngram-copy` (longest match between
  min and max length, newest on ties; a fixed key length gives the lookup of `ngram-simple` and `ngram-map-k`), the
  1-4-gram counts of `ngram-cache`, the copy lookup with `ngram-cache` behind it, and an oracle that guesses every
  token right with probability p.

usage: python pipeline_sim.py GEN.jsonl [--out RESULTS.json] [--quick]
"""
import argparse
import bisect
import importlib.util
import json
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("replay.py"))
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

N_LAYERS = 49
HEAD_MS = 2.5    # output head (327 MB of Q8_0 at ~150 GB/s) and GPU sampling, on the last card
HOP_MS = 0.3     # one card to the next: a few KB staged through the host, event query and launch
GAP_MS = 1.5     # sample read back, inputs built and launched on card 1 (host)

# (name, total ms per token, layers per card); the head is on the last card
SETUPS = [
    ("6 cards, 9/8/8/8/8/8 layers", 26.8, [9, 8, 8, 8, 8, 8]),
    ("9 cards, 6/6/6/6/5/5/5/5/5 layers", 31.3, [6, 6, 6, 6, 5, 5, 5, 5, 5]),
    ("9 cards, head + 2 layers on the last", 31.3, [6, 6, 6, 6, 6, 6, 6, 5, 2]),
]


def stage_times(total_ms, layers, hop=HOP_MS, gap=GAP_MS, head=HEAD_MS):
    """Per-card stage times whose plain-decoding sum with hops and gap is total_ms."""
    per_layer = (total_ms - gap - (len(layers) - 1) * hop - head) / sum(layers)
    st = [n * per_layer for n in layers]
    st[-1] += head
    return st


# ---------------------------------------------------------------------------------------------------------------------
# Guessers: next(full, n_v, overlay) -> token or None. full is prompt + output; full[:n_v] is verified; overlay holds
# the guesses in flight after it.
# ---------------------------------------------------------------------------------------------------------------------

class SeqIndex:
    """End positions of every n-gram (lengths lo..hi) of one transcript, for "newest end <= limit" queries."""

    def __init__(self, full, lo=2, hi=12):
        self.ends = {}
        for ln in range(lo, hi + 1):
            d = defaultdict(list)
            for e in range(ln - 1, len(full)):
                d[tuple(full[e - ln + 1:e + 1])].append(e)
            self.ends[ln] = d

    def newest(self, ln, key, limit):
        lst = self.ends[ln].get(key)
        if not lst:
            return -1
        k = bisect.bisect_right(lst, limit) - 1
        return lst[k] if k >= 0 else -1


class CopyGuesser:
    """The ngram-copy lookup on verified tokens plus guesses: the longest match of the suffix (min_len..max_len tokens,
    newest on ties, overlap allowed); the guess is the token after it."""

    def __init__(self, min_len=6, max_len=12):
        self.lo, self.hi = min_len, max_len

    def begin(self, full, index):
        self.index = index

    def _newest(self, full, n_v, overlay, ln):
        m = n_v + len(overlay)
        tail = overlay

        def at(i):
            return full[i] if i < n_v else tail[i - n_v]
        key = tuple(at(i) for i in range(m - ln, m))
        for e in range(m - 2, max(ln - 1, n_v - 1) - 1, -1):     # ends among the newest tokens, checked directly
            if all(at(e - k) == key[ln - 1 - k] for k in range(ln)):
                return e
        return self.index.newest(ln, key, n_v - 2)

    def next(self, full, n_v, overlay):
        m = n_v + len(overlay)
        if m <= self.lo:
            return None
        e = self._newest(full, n_v, overlay, self.lo)
        if e < 0:
            return None
        lo, hi = self.lo, min(self.hi, m - 1)
        e_hi = self._newest(full, n_v, overlay, hi) if hi > lo else -1
        if e_hi >= 0:
            e = e_hi
        else:   # largest length with a match; lengths are monotone (a match of n is one of n - 1)
            hi -= 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                em = self._newest(full, n_v, overlay, mid)
                if em >= 0:
                    lo, e = mid, em
                else:
                    hi = mid - 1
        src = e + 1
        return full[src] if src < n_v else overlay[src - n_v]


class CacheGuesser:
    """ngram-cache without files: the counts of the token after each 1-4-gram of the verified tokens; the guess is the
    top continuation of the longest n-gram with enough samples and share (ties to the one seen second)."""

    def begin(self, full, index):
        self.ctx, self.n_counted = {}, 0

    def next(self, full, n_v, overlay):
        for ng in range(1, 5):
            for i in range(max(self.n_counted, ng), n_v):
                part = self.ctx.setdefault(tuple(full[i - ng:i]), {})
                part[full[i]] = part.get(full[i], 0) + 1
        self.n_counted = max(self.n_counted, n_v)
        seq = full[max(0, n_v - 4):n_v] + overlay[-4:]
        for i in range(3, -1, -1):
            part = self.ctx.get(tuple(seq[len(seq) - i - 1:]))
            if part is None:
                continue
            max_c, sum_c, max_t = 0, 0, None
            for t, c in reversed(part.items()):
                if c > max_c:
                    max_c, max_t = c, t
                sum_c += c
            if sum_c < replay.CACHE_MIN_SAMPLES[i] or 100 * max_c < replay.CACHE_MIN_PERCENT[i] * sum_c:
                continue
            return max_t
        return None


class ChainGuesser:
    """The first guesser's guess, else the second's."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def begin(self, full, index):
        self.first.begin(full, index)
        self.second.begin(full, index)

    def next(self, full, n_v, overlay):
        g = self.first.next(full, n_v, overlay)
        return g if g is not None else self.second.next(full, n_v, overlay)


class OracleGuesser:
    """Guesses every token, right with probability p (fixed per position, the same draws for every p)."""

    def __init__(self, p, seed=0):
        self.p, self.seed = p, seed

    def begin(self, full, index):
        rng = random.Random(self.seed ^ zlib.crc32(bytes(str(len(full)) + str(full[:16]), "ascii")))
        self.u = [rng.random() for _ in full]

    def next(self, full, n_v, overlay):
        i = n_v + len(overlay)
        return full[i] if self.u[i] < self.p else -1


# ---------------------------------------------------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------------------------------------------------

def simulate(full, n_prompt, guesser, stages, hop, gap, depth, index=None):
    """Times (ms) between consecutive samples, one per generated token after the first, and counters."""
    S = len(stages)
    n = len(full) - n_prompt
    g0 = n_prompt
    card_free = [0.0] * S
    card_log = [[] for _ in range(S)]   # (start, end, position) per card, in position order
    sched = {}                          # position -> [token, starts, ends]
    stats = Counter()
    guesser.begin(full, index)

    def schedule(pos, tok, ready):
        st, en = [0.0] * S, [0.0] * S
        for j in range(S):
            if j:
                ready = en[j - 1] + hop
            free = card_free[j] + hop       # the host launches a stage only once the card is done: one more hop
            s0 = ready if ready > free else free
            e0 = s0 + stages[j]
            card_free[j] = e0
            log = card_log[j]
            log.append((s0, e0, pos))
            if len(log) > 64:
                del log[:-32]
            st[j], en[j] = s0, e0
        sched[pos] = [tok, st, en]

    def cancel_after(pos, t):
        for p in [p for p in sched if p > pos]:
            del sched[p]
            stats["cancelled"] += 1
        for j in range(S):
            log = card_log[j]
            while log and log[-1][2] > pos and log[-1][0] >= t:
                log.pop()
            k = len(log) - 1
            while k >= 0 and log[k][2] > pos:
                stats["wasted_ms"] += log[k][1] - log[k][0]
                k -= 1
            card_free[j] = log[-1][1] if log else 0.0

    times = []
    schedule(0, full[g0], gap)
    last, v = 0, 0

    def extend(now, last):
        while last - v < depth and last < n - 2:
            g = guesser.next(full, g0 + v + 1, [sched[p][0] for p in range(v + 1, last + 1)])
            if g is None:
                break
            last += 1
            schedule(last, g, max(card_free[0], now) + hop)
            stats["guessed"] += 1
        return last

    last = extend(0.0, last)
    prev = 0.0
    for q in range(n - 1):
        t_s = sched[q][2][S - 1]
        times.append(t_s - prev)
        prev = t_s
        if q + 1 == n - 1:
            break
        x = full[g0 + q + 1]
        nxt = sched.get(q + 1)
        if nxt is not None and nxt[0] == x:
            stats["hit"] += 1
        else:
            if nxt is not None:
                stats["miss"] += 1
                cancel_after(q, t_s)
            else:
                stats["none"] += 1
            schedule(q + 1, x, t_s + gap)
            last = q + 1
        del sched[q]
        v = q + 1
        last = extend(t_s, last)
    return times, stats


# ---------------------------------------------------------------------------------------------------------------------
# Runs and report
# ---------------------------------------------------------------------------------------------------------------------

KEYS = ("chat", "code", "edit", "agent", "all", "edit/answer")
KEY_NAMES = {"chat": "Chat", "code": "Code", "edit": "Edits", "agent": "Agent", "all": "All",
             "edit/answer": "File rewrites"}


def guessers():
    return [
        ("copy, match >= 2", lambda: CopyGuesser(2)),
        ("copy, match >= 3", lambda: CopyGuesser(3)),
        ("copy, match >= 4", lambda: CopyGuesser(4)),
        ("copy, match >= 6", lambda: CopyGuesser(6)),
        ("copy, match >= 8", lambda: CopyGuesser(8)),
        ("copy, match = 12", lambda: CopyGuesser(12)),
        ("fixed key 4 (ngram-simple/map-k lookup)", lambda: CopyGuesser(4, 4)),
        ("ngram-cache counts", lambda: CacheGuesser()),
        ("copy >= 4, else ngram-cache", lambda: ChainGuesser(CopyGuesser(4), CacheGuesser())),
    ]


ORACLE_P = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


class Runs:
    """Accumulates per-config time sums per workload key."""

    def __init__(self):
        self.t = defaultdict(lambda: defaultdict(float))
        self.n = defaultdict(lambda: defaultdict(int))
        self.stats = defaultdict(Counter)

    def add(self, cfg, row, times, stats):
        for j, dt in enumerate(times, 1):
            seg = row["segs"][j]
            for key in (row["category"], "all", row["category"] + "/" + seg):
                self.t[cfg][key] += dt
                self.n[cfg][key] += 1
        self.stats[cfg] += stats

    def speedup(self, cfg, key, plain):
        return self.n[cfg][key] * plain / self.t[cfg][key] if self.t[cfg][key] else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen")
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="first setup and main guessers only")
    args = ap.parse_args()
    rows = replay.load(args.gen)

    configs = []   # (cfg name, setup index, guesser factory, depth, hop, gap, stage scale)
    setups = SETUPS[:1] if args.quick else SETUPS
    for si, (sname, total, layers) in enumerate(setups):
        S = len(layers)
        for gname, make in guessers():
            if args.quick and gname not in ("copy, match >= 4", "copy >= 4, else ngram-cache"):
                continue
            configs.append((f"{sname} | {gname} | depth {S - 1}", si, make, S - 1, HOP_MS, GAP_MS, 1.0))
        for p in ORACLE_P:
            configs.append((f"{sname} | oracle p {p} | depth {S - 1}", si, lambda p=p: OracleGuesser(p), S - 1,
                            HOP_MS, GAP_MS, 1.0))
        for p in (0.5, 0.7):
            configs.append((f"{sname} | copy >= 4, else oracle p {p} | depth {S - 1}", si,
                            lambda p=p: ChainGuesser(CopyGuesser(4), OracleGuesser(p)), S - 1, HOP_MS, GAP_MS, 1.0))
        for d in range(1, S - 1):
            configs.append((f"{sname} | copy >= 4, else ngram-cache | depth {d}", si,
                            lambda: ChainGuesser(CopyGuesser(4), CacheGuesser()), d, HOP_MS, GAP_MS, 1.0))
            configs.append((f"{sname} | oracle p 0.7 | depth {d}", si, lambda: OracleGuesser(0.7), d, HOP_MS,
                            GAP_MS, 1.0))
        if si == 0:
            for hop in (0.1, 0.6, 1.0):
                for gname, make in (("copy >= 4, else ngram-cache", lambda: ChainGuesser(CopyGuesser(4),
                                                                                         CacheGuesser())),
                                    ("oracle p 0.7", lambda: OracleGuesser(0.7))):
                    configs.append((f"{sname} | {gname} | depth {S - 1} | hop {hop}", si, make, S - 1, hop, GAP_MS,
                                    1.0))
            for gname, make in (("copy >= 4, else ngram-cache", lambda: ChainGuesser(CopyGuesser(4), CacheGuesser())),
                                ("oracle p 0.7", lambda: OracleGuesser(0.7))):
                configs.append((f"{sname} | {gname} | depth {S - 1} | stages x1.1", si, make, S - 1, HOP_MS, GAP_MS,
                                1.1))

    runs = Runs()
    plain = {}
    for cfg, si, make, depth, hop, gap, scale in configs:
        total, layers = setups[si][1], setups[si][2]
        st = stage_times(total, layers, hop, gap)
        plain[cfg] = gap + sum(st) + (len(st) - 1) * hop
    for ri, row in enumerate(rows):
        full = row["prompt_tokens"] + row["gen_tokens"]
        index = SeqIndex(full)
        for cfg, si, make, depth, hop, gap, scale in configs:
            total, layers = setups[si][1], setups[si][2]
            st = [x * scale for x in stage_times(total, layers, hop, gap)]
            times, stats = simulate(full, len(row["prompt_tokens"]), make(), st, hop, gap, depth, index)
            runs.add(cfg, row, times, stats)
        print(f"{ri + 1}/{len(rows)} {row['id']}", file=sys.stderr)

    res = {"setups": [], "configs": {}}
    for sname, total, layers in setups:
        st = stage_times(total, layers)
        res["setups"].append({"name": sname, "total_ms": total, "layers": layers,
                              "stage_ms": [round(x, 3) for x in st], "hop_ms": HOP_MS, "gap_ms": GAP_MS,
                              "plain_tok_s": round(1000 / total, 2),
                              "cap_tok_s": round(1000 / (max(st) + HOP_MS), 1)})
    for cfg, *_ in configs:
        s = runs.stats[cfg]
        n_all = runs.n[cfg]["all"]
        res["configs"][cfg] = {
            "speedup": {k: round(runs.speedup(cfg, k, plain[cfg]), 4) for k in KEYS},
            "tok_s": {k: round(runs.speedup(cfg, k, plain[cfg]) * 1000 / plain[cfg], 2) for k in KEYS},
            "hit_share": s["hit"] / n_all, "miss_share": s["miss"] / n_all,
            "wasted_ms_per_token": s["wasted_ms"] / n_all, "cancelled": s["cancelled"], "guessed": s["guessed"],
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    print("| Setup | Stage ms per card | Plain tokens/s | One token per slowest stage + hop |")
    print("|---|---|---|---|")
    for s in res["setups"]:
        print(f"| {s['name']} | {' / '.join(f'{x:.2f}' for x in s['stage_ms'])} | {s['plain_tok_s']} |"
              f" {s['cap_tok_s']} |")
    print("\n| Config | " + " | ".join(KEY_NAMES[k] for k in KEYS) + " | Right guesses | Wrong guesses |"
          " Wasted card-ms per token |")
    print("|---|" + "---|" * (len(KEYS) + 3))
    for cfg, r in res["configs"].items():
        print(f"| {cfg} | " + " | ".join(f"{r['speedup'][k]:.2f}" for k in KEYS) +
              f" | {100 * r['hit_share']:.1f} % | {100 * r['miss_share']:.1f} % | {r['wasted_ms_per_token']:.2f} |")


if __name__ == "__main__":
    main()
