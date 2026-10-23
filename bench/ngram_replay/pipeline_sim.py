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
  1-4-gram counts of `ngram-cache`, a continuation table of the other transcripts' outputs (the most frequent next token
  after the longest 1-8-token context it has seen; never the transcript's own text), combinations of these, and an
  oracle that guesses every token right with probability p (Q87, Q96).
- Gumbel coupling (Q96, with --topk): the sampler draws each token as argmax(log p + G) over the tokens Flint's sampler
  chain keeps, with noise G keyed by position and token, which is an exact sample; the guesser picks argmax(log q + G)
  with the same G over its own counts q. The hit probability per position, along the transcript's prefix, is
  computed exactly from Flint's top-20 log-probabilities (logit_dump/), and the pipeline replays it with fixed draws.

usage: python pipeline_sim.py GEN.jsonl [--out RESULTS.json] [--only REGEX] [--jobs N] [--accuracy] [--topk TOP.bin]
"""
import argparse
import bisect
import importlib.util
import json
import math
import random
import re
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

    def dist(self, full, n_v, overlay):
        g = self.next(full, n_v, overlay)
        return None if g is None else {g: 1}


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
            self.last_part = part
            return max_t
        return None

    def dist(self, full, n_v, overlay):
        return dict(self.last_part) if self.next(full, n_v, overlay) is not None else None


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

    def dist(self, full, n_v, overlay):
        d = self.first.dist(full, n_v, overlay)
        return d if d is not None else self.second.dist(full, n_v, overlay)


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


class ContinuationTable:
    """Counts of the token after every 1..max_ctx-token context in a set of outputs (generated tokens only), queried
    with one transcript held out: a transcript's guesses never come from its own text."""

    def __init__(self, rows, members=None, max_ctx=8):
        self.rows, self.max_ctx = rows, max_ctx
        self.members = set(range(len(rows))) if members is None else set(members)
        self.counts = {}
        for i in sorted(self.members):
            for key, d in self._own(i).items():
                g = self.counts.setdefault(key, {})
                for t, c in d.items():
                    g[t] = g.get(t, 0) + c
        self.tokens = sum(len(rows[i]["gen_tokens"]) for i in self.members)

    def _own(self, i):
        gen, own = self.rows[i]["gen_tokens"], {}
        for j in range(1, len(gen)):
            for ln in range(1, min(self.max_ctx, j) + 1):
                d = own.setdefault(tuple(gen[j - ln:j]), {})
                d[gen[j]] = d.get(gen[j], 0) + 1
        return own

    def held_out(self, i):
        """{context: (top token, its count, all counts)} without transcript i, built lazily."""
        own = self._own(i) if i in self.members else {}
        return TableView(self.counts, own)


class TableView:
    def __init__(self, counts, minus):
        self.counts, self.minus, self.cache = counts, minus, {}

    def top(self, key):
        r = self.cache.get(key, 0)
        if r != 0:
            return r
        g, m = self.counts.get(key), self.minus.get(key)
        best = None
        if g:
            bt, bc, tot = None, 0, 0
            for t, c in g.items():
                if m:
                    c -= m.get(t, 0)
                if c > 0:
                    tot += c
                if c > bc:
                    bt, bc = t, c
            if bt is not None:
                best = (bt, bc, tot)
        self.cache[key] = best
        return best


class TableGuesser:
    """The most frequent next token of the longest context (max_ctx down to 1 token) the table has seen; with
    request=True the current request's own counts are added to the table's. With min_share > 0 it guesses only when
    that token holds at least that share of the context's counts."""

    def __init__(self, view, max_ctx=8, request=False, min_share=0.0):
        self.view, self.max_ctx, self.request, self.min_share = view, max_ctx, request, min_share

    def begin(self, full, index):
        self.req, self.n_counted = {}, 0

    def next(self, full, n_v, overlay):
        if self.request:
            for j in range(max(self.n_counted, 1), n_v):
                for ln in range(1, min(self.max_ctx, j) + 1):
                    d = self.req.setdefault(tuple(full[j - ln:j]), {})
                    d[full[j]] = d.get(full[j], 0) + 1
            self.n_counted = max(self.n_counted, n_v)
        seq = full[max(0, n_v - self.max_ctx):n_v] + overlay[-self.max_ctx:]
        for ln in range(min(self.max_ctx, len(seq)), 0, -1):
            key = tuple(seq[len(seq) - ln:])
            top = self.view.top(key)
            r = self.req.get(key) if self.request else None
            if r:
                best, bc = (top[0], top[1] + r.get(top[0], 0)) if top else (None, 0)
                for t, c in r.items():
                    c += self.view.counts.get(key, {}).get(t, 0) - self.view.minus.get(key, {}).get(t, 0)
                    if c > bc:
                        best, bc = t, c
                total = (top[2] if top else 0) + sum(r.values())
                return best if bc >= self.min_share * total else None
            if top:
                return top[0] if top[1] >= self.min_share * top[2] else None
        return None

    def dist(self, full, n_v, overlay, keep=64):
        """The counts behind next(): the longest context's next-token counts (table without the held-out transcript,
        plus the request's own if request=True), the `keep` largest."""
        g = self.next(full, n_v, overlay)
        if g is None:
            return None
        seq = full[max(0, n_v - self.max_ctx):n_v] + overlay[-self.max_ctx:]
        for ln in range(min(self.max_ctx, len(seq)), 0, -1):
            key = tuple(seq[len(seq) - ln:])
            r = self.req.get(key) if self.request else None
            if not (self.view.top(key) or r):
                continue
            d = {}
            minus = self.view.minus.get(key, {})
            for t, c in self.view.counts.get(key, {}).items():
                c -= minus.get(t, 0)
                if c > 0:
                    d[t] = c
            for t, c in (r or {}).items():
                d[t] = d.get(t, 0) + c
            if len(d) > keep:
                d = dict(sorted(d.items(), key=lambda kv: -kv[1])[:keep])
            return d
        return None


class Env:
    """What a guesser factory gets for one transcript: the table without it, and its per-position hit probabilities."""

    def __init__(self, view, probs=None):
        self.view, self.probs = view, probs


class ProbGuesser:
    """Guesses token i right with probability a[i] (a full-sequence list; None: no guess), with fixed draws per position
    so that runs with different a share their randomness."""

    def __init__(self, a, seed=1):
        self.a, self.seed = a, seed

    def begin(self, full, index):
        rng = random.Random(self.seed ^ zlib.crc32(bytes(str(len(full)) + str(full[:16]), "ascii")))
        self.u = [rng.random() for _ in full]

    def next(self, full, n_v, overlay):
        i = n_v + len(overlay)
        a = self.a[i]
        if a is None:
            return None
        return full[i] if self.u[i] < a else -1


def load_topk(path):
    """{seq_id: [(tokens, log-probabilities) per generated token]} from llama-logit-dump's output."""
    import struct
    out = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            seq_id, n_prompt, n_gen, k = struct.unpack("<4i", hdr)
            recs = []
            raw = f.read(n_gen * k * 8)
            vals = struct.unpack(f"<{n_gen * k * 2}i", raw)
            fl = struct.unpack(f"<{n_gen * k * 2}f", raw)
            for g in range(n_gen):
                base = g * k * 2
                recs.append(([vals[base + 2 * i] for i in range(k)], [fl[base + 2 * i + 1] for i in range(k)]))
            out[seq_id] = recs
    return out


def sampler_probs(toks, lps, top_k=20, top_p=0.95, min_p=0.05):
    """llama.cpp's chain at temperature 1 on the top log-probabilities: top-k, top-p (keeps the token that crosses p),
    min-p (relative to the most likely token), renormalized."""
    pairs = sorted(zip(lps, toks), reverse=True)[:top_k]
    m = pairs[0][0]
    w = [math.exp(lp - m) for lp, _ in pairs]
    z = sum(w)
    p = [x / z for x in w]
    keep, cum = 0, 0.0
    for x in p:
        keep += 1
        cum += x
        if cum >= top_p:
            break
    kept = [(pairs[i][1], p[i]) for i in range(keep) if p[i] >= min_p * p[0]]
    z = sum(x for _, x in kept)
    return {t: x / z for t, x in kept}


def coupled_match(p, q):
    """P(argmax(log p + G) == argmax(log q + G)) with the same Gumbel noise G for both (q need not be normalized).
    With G = -log E, E ~ Exp(1) per token, token i wins both races iff E_j > E_i max(p_j/p_i, q_j/q_i) for every j,
    which has probability 1 / sum_j max(p_j/p_i, q_j/q_i) (the j = i term is 1)."""
    q_out = sum(c for t, c in q.items() if t not in p)
    total = 0.0
    for i, pi in p.items():
        qi = q.get(i, 0)
        if qi <= 0:
            continue
        s = q_out / qi
        for j, pj in p.items():
            s += max(pj / pi, q.get(j, 0) / qi)
        total += 1.0 / s
    return total


COUPLED = [
    ("copy >= 4, else ngram-cache", lambda e: ChainGuesser(CopyGuesser(4), CacheGuesser())),
    ("copy >= 4, else table", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view))),
    ("copy >= 4, else table + request counts", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True))),
    ("table + request counts", lambda e: TableGuesser(e.view, request=True)),
]


def hit_probs(row, topk, view, index):
    """Per coupled guesser, two full-sequence lists of per-position hit probabilities on the true prefix: the top guess
    under ordinary sampling, p(top guess), and the Gumbel-coupled guess, coupled_match(p, q)."""
    prompt, gen = row["prompt_tokens"], row["gen_tokens"]
    full = prompt + gen
    out = {}
    for name, make in COUPLED:
        g = make(Env(view))
        g.begin(full, index)
        top, coup = [None] * len(full), [None] * len(full)
        for j in range(1, len(gen)):
            q = g.dist(full, len(prompt) + j, [])
            if q is None:
                continue
            p = sampler_probs(*topk[j])
            best = max(q.items(), key=lambda kv: kv[1])[0]
            top[len(prompt) + j] = p.get(best, 0.0)
            coup[len(prompt) + j] = coupled_match(p, q)
        out[name] = (top, coup)
    return out


EMPTY_VIEW = TableView({}, {})   # an empty table, for the request's own counts alone


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
    """(name, factory(table view)) for the n-gram guessers."""
    return [
        ("copy, match >= 2", lambda e: CopyGuesser(2)),
        ("copy, match >= 3", lambda e: CopyGuesser(3)),
        ("copy, match >= 4", lambda e: CopyGuesser(4)),
        ("copy, match >= 6", lambda e: CopyGuesser(6)),
        ("copy, match >= 8", lambda e: CopyGuesser(8)),
        ("copy, match = 12", lambda e: CopyGuesser(12)),
        ("fixed key 4 (ngram-simple/map-k lookup)", lambda e: CopyGuesser(4, 4)),
        ("ngram-cache counts", lambda e: CacheGuesser()),
        ("copy >= 4, else ngram-cache", lambda e: ChainGuesser(CopyGuesser(4), CacheGuesser())),
        ("table (held out)", lambda e: TableGuesser(e.view)),
        ("table + request counts", lambda e: TableGuesser(e.view, request=True)),
        ("copy >= 4, else table", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view))),
        ("copy >= 4, else table + request counts", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True))),
        ("copy >= 4, else ngram-cache, else table", lambda e: ChainGuesser(ChainGuesser(CopyGuesser(4), CacheGuesser()),
                                                                           TableGuesser(e.view))),
        ("copy >= 4, else request counts (no table)", lambda e: ChainGuesser(CopyGuesser(4),
                                                                             TableGuesser(EMPTY_VIEW, request=True))),
        ("copy >= 4, else table + request counts, share >= 0.3",
         lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True, min_share=0.3))),
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


def build_configs(setups, coupled=False):
    """(name, setup index, factory(env), depth, hop, gap, stage scale) for every run."""
    copy_cache = lambda e: ChainGuesser(CopyGuesser(4), CacheGuesser())     # noqa: E731
    configs = []
    for si, (sname, total, layers) in enumerate(setups):
        S = len(layers)
        for gname, make in guessers():
            configs.append((f"{sname} | {gname} | depth {S - 1}", si, make, S - 1, HOP_MS, GAP_MS, 1.0))
        for p in ORACLE_P:
            configs.append((f"{sname} | oracle p {p} | depth {S - 1}", si, lambda e, p=p: OracleGuesser(p), S - 1,
                            HOP_MS, GAP_MS, 1.0))
        for p in (0.5, 0.7):
            configs.append((f"{sname} | copy >= 4, else oracle p {p} | depth {S - 1}", si,
                            lambda e, p=p: ChainGuesser(CopyGuesser(4), OracleGuesser(p)), S - 1, HOP_MS, GAP_MS, 1.0))
        for d in range(1, S - 1):
            configs.append((f"{sname} | copy >= 4, else ngram-cache | depth {d}", si, copy_cache, d, HOP_MS, GAP_MS,
                            1.0))
            configs.append((f"{sname} | oracle p 0.7 | depth {d}", si, lambda e: OracleGuesser(0.7), d, HOP_MS,
                            GAP_MS, 1.0))
        if coupled:
            for gname, _ in COUPLED:
                for mode, k in (("sampled top guess, expected", 0), ("Gumbel-coupled", 1)):
                    configs.append((f"{sname} | {gname} | {mode} | depth {S - 1}", si,
                                    lambda e, gname=gname, k=k: ProbGuesser(e.probs[gname][k]), S - 1, HOP_MS, GAP_MS,
                                    1.0))
        if si == 0:
            for gname, make in (("copy >= 4, else ngram-cache", copy_cache),
                                ("copy >= 4, else table + request counts",
                                 lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True))),
                                ("oracle p 0.7", lambda e: OracleGuesser(0.7))):
                for hop in (0.1, 0.6, 1.0):
                    configs.append((f"{sname} | {gname} | depth {S - 1} | hop {hop}", si, make, S - 1, hop, GAP_MS,
                                    1.0))
                configs.append((f"{sname} | {gname} | depth {S - 1} | stages x1.1", si, make, S - 1, HOP_MS, GAP_MS,
                                1.1))
    return configs


_SHARED = None   # (rows, setups, configs, table): set before the workers fork, so nothing is pickled


def run_rows(idx):
    """Worker: simulate every config on a slice of the transcripts."""
    rows, setups, configs, table, topk = _SHARED
    runs, coup = Runs(), defaultdict(Counter)
    for ri in idx:
        row = rows[ri]
        full = row["prompt_tokens"] + row["gen_tokens"]
        index = SeqIndex(full)
        view = table.held_out(ri)
        env = Env(view, hit_probs(row, topk[ri], view, index) if topk else None)
        if env.probs:
            prompt, gen = row["prompt_tokens"], row["gen_tokens"]
            in_copy = [False] * len(gen)
            for start, ln, _ in replay.copy_factors(prompt, gen, 8):
                for j in range(start, start + ln):
                    in_copy[j] = True
            for name, (top, cp) in env.probs.items():
                for j in range(1, len(gen)):
                    i = len(prompt) + j
                    where = "copy" if in_copy[j] else "fresh"
                    for key in (row["category"], "all"):
                        c = coup[f"{name}|{key}"]
                        c[where] += 1
                        c[where + "_top"] += top[i] or 0.0
                        c[where + "_coupled"] += cp[i] or 0.0
        for cfg, si, make, depth, hop, gap, scale in configs:
            total, layers = setups[si][1], setups[si][2]
            st = [x * scale for x in stage_times(total, layers, hop, gap)]
            times, stats = simulate(full, len(row["prompt_tokens"]), make(env), st, hop, gap, depth, index)
            runs.add(cfg, row, times, stats)
        print(f"{row['id']}", file=sys.stderr)
    return {"t": {c: dict(v) for c, v in runs.t.items()}, "n": {c: dict(v) for c, v in runs.n.items()},
            "stats": {c: dict(v) for c, v in runs.stats.items()}, "coupling": {k: dict(v) for k, v in coup.items()}}


ACC_GUESSERS = [
    ("copy >= 4", lambda e: CopyGuesser(4)),
    ("ngram-cache counts", lambda e: CacheGuesser()),
    ("table (held out)", lambda e: TableGuesser(e.view)),
    ("table + request counts", lambda e: TableGuesser(e.view, request=True)),
    ("copy >= 4, else ngram-cache", lambda e: ChainGuesser(CopyGuesser(4), CacheGuesser())),
    ("copy >= 4, else table", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view))),
    ("copy >= 4, else table + request counts", lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True))),
    ("request counts (no table)", lambda e: TableGuesser(EMPTY_VIEW, request=True)),
    ("copy >= 4, else table + request counts, share >= 0.3",
     lambda e: ChainGuesser(CopyGuesser(4), TableGuesser(e.view, request=True, min_share=0.3))),
]


def accuracy(rows, tables):
    """One token ahead on the true prefix: share of positions guessed, and guessed right, on fresh and copied text."""
    acc = defaultdict(Counter)
    for ri, row in enumerate(rows):
        prompt, gen = row["prompt_tokens"], row["gen_tokens"]
        full = prompt + gen
        index = SeqIndex(full)
        in_copy = [False] * len(gen)
        for start, ln, _ in replay.copy_factors(prompt, gen, 8):
            for j in range(start, start + ln):
                in_copy[j] = True
        for tname, table in tables:
            view = table.held_out(ri)
            for gname, make in ACC_GUESSERS:
                if tname != "all" and "table" not in gname:
                    continue
                g = make(Env(view))
                g.begin(full, index)
                name = gname if tname == "all" else f"{gname} [{tname}]"
                for j in range(1, len(gen)):
                    guess = g.next(full, len(prompt) + j, [])
                    where = "copy" if in_copy[j] else "fresh"
                    for key in (row["category"], "all"):
                        c = acc[(name, key)]
                        c[where] += 1
                        c[where + "_guessed"] += guess is not None
                        c[where + "_right"] += guess == gen[j]
        print(f"accuracy {row['id']}", file=sys.stderr)
    out = {}
    for (name, key), c in acc.items():
        n = c["fresh"] + c["copy"]
        out.setdefault(name, {})[key] = {
            "right": (c["fresh_right"] + c["copy_right"]) / n,
            "fresh_guessed": c["fresh_guessed"] / max(1, c["fresh"]), "fresh_right": c["fresh_right"] / max(1, c["fresh"]),
            "copy_right": c["copy_right"] / max(1, c["copy"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen")
    ap.add_argument("--out")
    ap.add_argument("--only", help="run only the configs whose name matches this regex")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (each takes a share of the transcripts)")
    ap.add_argument("--accuracy", action="store_true", help="also the one-token accuracy of the guessers")
    ap.add_argument("--topk", help="llama-logit-dump output along the same transcripts: adds Gumbel-coupled guessing")
    args = ap.parse_args()
    rows = replay.load(args.gen)
    setups = SETUPS
    table = ContinuationTable(rows)
    topk = None
    if args.topk:
        dump = load_topk(args.topk)
        topk = [dump[i] for i in range(len(rows))]
        for row, recs in zip(rows, topk):
            assert len(recs) == len(row["gen_tokens"]), row["id"]
    configs = build_configs(setups, coupled=topk is not None)
    if args.only:
        configs = [c for c in configs if re.search(args.only, c[0])]

    plain = {}
    for cfg, si, make, depth, hop, gap, scale in configs:
        st = stage_times(setups[si][1], setups[si][2], hop, gap)
        plain[cfg] = gap + sum(st) + (len(st) - 1) * hop
    runs, coup_sum = Runs(), {}
    global _SHARED
    _SHARED = (rows, setups, configs, table, topk)
    tasks = [sl for sl in (list(range(len(rows)))[k::args.jobs] for k in range(args.jobs)) if sl]
    if args.jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
            parts = pool.map(run_rows, tasks)
    else:
        parts = [run_rows(t) for t in tasks]
    for part in parts:
        for c, d in part["t"].items():
            for k, x in d.items():
                runs.t[c][k] += x
        for c, d in part["n"].items():
            for k, x in d.items():
                runs.n[c][k] += x
        for c, d in part["stats"].items():
            runs.stats[c] += Counter(d)
        for k, d in part.get("coupling", {}).items():
            coup_sum[k] = coup_sum.get(k, Counter()) + Counter(d)

    res = {"setups": [], "configs": {}, "table_tokens": table.tokens}
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
    if coup_sum:
        res["coupling"] = {}
        for k, c in coup_sum.items():
            name, key = k.split("|")
            n = c["fresh"] + c["copy"]
            res["coupling"].setdefault(name, {})[key] = {
                "top": (c["fresh_top"] + c["copy_top"]) / n, "coupled": (c["fresh_coupled"] + c["copy_coupled"]) / n,
                "fresh_top": c["fresh_top"] / max(1, c["fresh"]), "fresh_coupled": c["fresh_coupled"] / max(1, c["fresh"]),
                "copy_top": c["copy_top"] / max(1, c["copy"]), "copy_coupled": c["copy_coupled"] / max(1, c["copy"])}
    if args.accuracy:
        rng = random.Random(1)
        order = list(range(len(rows)))
        rng.shuffle(order)
        tables = [("all", table)] + [(f"{f} of the table", ContinuationTable(rows, order[:round(f * len(rows))]))
                                     for f in (0.25, 0.5)]
        res["accuracy"] = accuracy(rows, tables)
        res["table_sizes"] = {name: t.tokens for name, t in tables}
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
    if "coupling" in res:
        print("\n| Guesser | Expected right, top guess: all / fresh / copied | Gumbel-coupled: all / fresh / copied |"
              " Fresh, coupled: chat / code / edit / agent |")
        print("|---|---|---|---|")
        for name, by in res["coupling"].items():
            a = by["all"]
            per = " / ".join(f"{100 * by[k]['fresh_coupled']:.1f}" for k in ("chat", "code", "edit", "agent"))
            print(f"| {name} | {100 * a['top']:.1f} / {100 * a['fresh_top']:.1f} / {100 * a['copy_top']:.1f} % |"
                  f" {100 * a['coupled']:.1f} / {100 * a['fresh_coupled']:.1f} / {100 * a['copy_coupled']:.1f} % | {per} |")
    if args.accuracy:
        print("\n| Guesser | Right (all) | Fresh text: guessed | Fresh text: right | Copied text: right |"
              " Fresh right: chat / code / edit / agent |")
        print("|---|---|---|---|---|---|")
        for name, by in res["accuracy"].items():
            a = by["all"]
            per = " / ".join(f"{100 * by[k]['fresh_right']:.1f}" for k in ("chat", "code", "edit", "agent"))
            print(f"| {name} | {100 * a['right']:.1f} % | {100 * a['fresh_guessed']:.1f} % |"
                  f" {100 * a['fresh_right']:.1f} % | {100 * a['copy_right']:.1f} % | {per} |")


if __name__ == "__main__":
    main()
