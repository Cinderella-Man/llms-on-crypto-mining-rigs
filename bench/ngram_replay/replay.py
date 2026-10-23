#!/usr/bin/env python3
"""Replay llama.cpp's n-gram drafters, and the proposed copy-gated `ngram-copy` drafter (Q53), on transcripts that
Flint generated itself, with exact acceptance (registry row Q64).

Each transcript holds the prompt and the generated token ids of one request (bench/expert_overlap/generate.py writes
them). The replay steps through the generated tokens the way llama-server does: at each step it asks the drafter for a
draft from the history and the last sampled token, keeps the longest prefix of the draft that equals what the model
generated next, and advances by the accepted tokens plus one. The transcripts are samples from the model at its own
settings, and sample-and-match verification keeps a draft token exactly when the target's sample at that position equals
it, so the acceptance counted here is what the server would count on the same output.

The drafters are line-by-line ports of the code in the llama.cpp fork (branch pascal-p106 on upstream d77256999):
common/ngram-map.cpp (ngram-simple, ngram-map-k, ngram-map-k4v), common/ngram-mod.cpp with its driver in
common/speculative.cpp (ngram-mod), common/ngram-cache.cpp (ngram-cache), plus the server's draft room and the rule that
a drafter hears back only after a non-empty draft (tools/server/server-context.cpp). fork_check.cpp replays the same
transcripts through the fork's own C++ and must print the same steps (see the README).

usage: python replay.py GEN.jsonl [--out RESULTS.json] [--vn 1=26.8,2=33.0,...] [--only REGEX] [--dump-steps FILE]
  --vn          a measured verify-cost table v(n) (any unit; divided by v(1)), added as the curve "measured"
  --only        replay only the drafter settings whose name matches REGEX
  --dump-steps  write every step as "<setting> TAB <transcript> TAB <position> TAB <drafted> TAB <accepted>"
  --no-copy     skip the copy-structure pass
"""
import argparse
import bisect
import functools
import json
import math
import re
import sys
from collections import Counter, defaultdict

# Flint's special tokens: the reasoning block and the tool-call block
THINK_OPEN, THINK_CLOSE, TOOL_OPEN, TOOL_CLOSE = 150006, 150007, 150010, 150011
CLASSES = ("chat", "code", "edit", "agent")
SEGMENTS = ("think", "answer", "tool")

# What Q51's generation used: llama-server -c 65536 -np 4 (16,384 tokens per slot), n_predict 3,072
N_CTX, N_PREDICT = 16384, 3072


def segments_of(gen):
    """Label each generated token: reasoning ("think"), tool call ("tool") or the rest ("answer")."""
    out, seg = [], "answer"
    for t in gen:
        if t == THINK_OPEN:
            seg = "think"
        elif t == TOOL_OPEN:
            seg = "tool"
        out.append(seg)
        if t in (THINK_CLOSE, TOOL_CLOSE):
            seg = "answer"
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Drafters. Interface: begin(prompt) at the start of a request; draft(hist, last, room) -> draft tokens, where hist is
# every token already in the KV cache (prompt plus output, append-only) and last is the sampled token not yet in it;
# accept(n_accepted) after the verify of a non-empty draft.
# ---------------------------------------------------------------------------------------------------------------------

class NgramSimple:
    """ngram-simple: the newest earlier occurrence of the last size_n tokens (the sampled one included) that starts
    after position 0 and does not end at the newest history token; drafts min(size_m, tokens after the match), nothing
    if fewer than size_n follow or the history has at most size_n + size_m + 1 tokens."""

    def __init__(self, n=12, m=48):
        self.n, self.m = n, m

    def begin(self, prompt):
        self.starts = defaultdict(list)  # n-gram -> start positions in the history, ascending
        self.n_indexed = 0

    def draft(self, hist, last, room):
        n, m, cur = self.n, self.m, len(hist)
        if cur <= n + m + 1:
            return []
        for j in range(self.n_indexed, cur - n + 1):
            self.starts[tuple(hist[j:j + n])].append(j)
        self.n_indexed = cur - n + 1
        starts = self.starts.get(tuple(hist[cur - n + 1:]) + (last,))
        if not starts:
            return []
        k = bisect.bisect_right(starts, cur - n - 1) - 1
        if k < 0 or starts[k] == 0:
            return []
        j = starts[k]
        copy_max = min(m, cur - (j + n))
        if copy_max < n:
            return []
        return hist[j + n:j + n + copy_max]

    def accept(self, n_accepted):
        pass


KEY_MAP_MASK = 262144 - 1   # COMMON_NGRAM_HASH_MAP_SIZE; the 32-bit LCG hash modulo 2^18
LCG_FACTOR = 2654435761
MAX_VALUE_COUNT = 16380


class NgramMap:
    """ngram-map-k (key_only) and ngram-map-k4v: a 2^18-slot hash of key n-grams (first writer wins, collisions
    included) over the history; ngram-map-k drafts the size_m tokens after the found key, cut to the accepted count of
    that key's last draft; ngram-map-k4v keeps up to 4 continuations per key and drafts only if the most frequent one
    has at least twice the count of the others together."""

    def __init__(self, n=12, m=48, key_only=True, min_hits=1):
        self.n, self.m, self.key_only, self.min_hits = n, m, key_only, min_hits

    def begin(self, prompt):
        self.key_map = [0] * (KEY_MAP_MASK + 1)
        self.key_map_last_idx = 0
        self.size_last_begin = len(prompt)
        self.keys = []      # [key_idx, stat_idx, key_num, values: [[value_idx, value_num, n_accepted] x 4]]
        self.key_off = {}   # key n-gram -> offset in keys (keys are unique by content, as the linear search finds them)
        self.last = None    # (key offset, value slot) of the last draft

    def _hash(self, seq, start):
        h = 0
        for i in range(start, start + self.n):
            h = (h * LCG_FACTOR + seq[i]) & KEY_MAP_MASK
        return h

    def draft(self, hist, last, room):
        self.last = None
        n, m, cur = self.n, self.m, len(hist)
        if cur < 2 * n + m:
            return []
        key = hist[cur - n + 1:] + [last]
        hi = cur - n - m - 1            # newest key start whose m-gram lies inside the history
        lo = self.key_map_last_idx
        slb = self.size_last_begin

        match = 0
        idx = self.key_map[self._hash(key, 0)]
        if idx != 0 and idx < hi and hist[idx:idx + n] == key:
            match = idx
        if match == 0 and slb > n + m + 1:
            for j in range(slb - n - m - 1, lo, -1):
                if hist[j:j + n] == key:
                    match = j
                    break
        if match == 0:
            for j in range(hi, max(slb, lo), -1):
                if hist[j:j + n] == key:
                    match = j
                    break

        if slb > n + m + 1:
            for j in range(slb - n - m - 1, lo, -1):
                h = self._hash(hist, j)
                if self.key_map[h] == 0:
                    self.key_map[h] = j
        for j in range(hi, max(slb, lo), -1):
            h = self._hash(hist, j)
            if self.key_map[h] == 0:
                self.key_map[h] = j
        self.key_map_last_idx = max(hi, lo)

        if match == 0:
            return []
        tkey = tuple(key)
        off = self.key_off.get(tkey)
        if off is None:
            off = len(self.keys)
            self.keys.append([match, 0, 0, [[0, 0, m] for _ in range(4)]])
            self.key_off[tkey] = off
        ck = self.keys[off]
        ck[2] = min(ck[2] + 1, MAX_VALUE_COUNT)
        values = ck[3]

        if self.key_only:
            self.last = (off, 0)
            return hist[match + n:match + n + min(m, values[0][2])]

        if ck[2] < self.min_hits:
            return []
        for i in range(ck[1], match + 1):
            if hist[i:i + n] != key:
                continue
            vb = i + n
            slot = -1
            for v in range(4):
                val = values[v]
                if val[0] == 0:
                    val[0], val[1], val[2] = vb, 0, m
                    slot = v
                    break
                if hist[vb:vb + m] == hist[val[0]:val[0] + m]:
                    slot = v
                    break
            if slot >= 0:
                values[slot][1] = min(values[slot][1] + 1, MAX_VALUE_COUNT)
        ck[1] = match

        max_occur, slot_max = 0, 0
        for v in range(4):
            if values[v][1] > max_occur:
                max_occur, slot_max = values[v][1], v
        sum_occur = sum(values[v][1] for v in range(4) if v != slot_max)
        if sum_occur > 0 and max_occur < 2 * sum_occur:
            return []
        self.last = (off, slot_max)
        # the fork drafts the tokens after the found key, whichever continuation won the count
        return hist[match + n:match + n + min(m, values[slot_max][2])]

    def accept(self, n_accepted):
        if self.last is not None:
            off, slot = self.last
            self.keys[off][3][slot][2] = n_accepted


MOD_MASK = 4 * 1024 * 1024 - 1   # 4 Mi slots; the 64-bit LCG hash modulo 2^22
MOD_A = 6364136223846793005 & MOD_MASK


class NgramMod:
    """ngram-mod (--spec-default): a 4 Mi-slot hash of n_match-token contexts -> next token, no key check, last writer
    wins. Drafts by chaining lookups through its own output until an empty slot; no draft if that happens before n_min
    tokens. New history enters the table only once it has grown by more than 8 tokens past the last update. Five verifies
    in a row with under 25 % of the draft accepted clear the table. The fork shares the table across requests; here every
    request starts empty, as on a freshly started server."""

    def __init__(self, n_match=24, n_min=48, n_max=64):
        self.n, self.n_min, self.n_max = n_match, n_min, n_max
        self.p_out = pow(MOD_A, n_match - 1, MOD_MASK + 1)

    def _hash(self, seq, start):
        h = 0
        for i in range(start, start + self.n):
            h = (h * MOD_A + seq[i]) & MOD_MASK
        return h

    def _add(self, seq, a, b):
        if a >= b:
            return
        n, h = self.n, self._hash(seq, a)
        for i in range(a, b):
            if i > a:
                h = ((h - seq[i - 1] * self.p_out) * MOD_A + seq[i + n - 1]) & MOD_MASK
            self.entries[h] = seq[i + n]

    def begin(self, prompt):
        self.entries = {}   # slot -> token; an absent slot is EMPTY
        self.i_last, self.n_draft_last, self.n_low = 0, 0, 0
        if len(prompt) < self.n:
            return
        self._add(prompt, 0, len(prompt) - self.n)
        self.i_last = len(prompt) - self.n
        if len(self.entries) / (MOD_MASK + 1) > 0.25:
            self.entries = {}

    def draft(self, hist, last, room):
        self.n_draft_last = 0
        n, cur = self.n, len(hist)
        if cur < n:
            return []
        if self.i_last + 32 < cur:
            self._add(hist, self.i_last, cur - n)
            self.i_last = cur - n
        seq = hist[cur - n + 1:] + [last]
        h = self._hash(seq, 0)
        for i in range(self.n_max):
            tok = self.entries.get(h)
            if tok is None:
                if i < self.n_min:
                    return []
                break
            seq.append(tok)
            h = ((h - seq[i] * self.p_out) * MOD_A + tok) & MOD_MASK
        out = seq[n:]
        self.n_draft_last = len(out)
        return out

    def accept(self, n_accepted):
        if self.n_draft_last > 0:
            if n_accepted / self.n_draft_last < 0.25:
                self.n_low += 1
                if self.n_low >= 5:
                    self.entries = {}
                    self.n_low = 0
                    self.i_last = 0
            else:
                self.n_low = 0


CACHE_MIN_SAMPLES = (2, 2, 1, 1)   # draft_min_sample_size_lax, for 1- to 4-grams
CACHE_MIN_PERCENT = (66, 50, 50, 50)


class NgramCache:
    """ngram-cache without cache files: counts of the token after every 1- to 4-gram of the request; drafts up to 8
    tokens, each from the longest n-gram whose top continuation has enough samples and share. The only order-dependent
    case, two continuations with equal counts, goes to the one seen second, as libstdc++'s unordered_map iterates
    a two-element map."""

    def __init__(self, n_draft=8):
        self.n_draft = n_draft

    def begin(self, prompt):
        self.ctx = {}
        self.cache_size = 0

    def draft(self, hist, last, room):
        size = len(hist) + 1   # the history plus the sampled token
        if self.cache_size < size:
            for ng in range(1, 5):
                for i in range(max(self.cache_size, ng), size):
                    tok = hist[i] if i < size - 1 else last
                    part = self.ctx.setdefault(tuple(hist[i - ng:i]), {})
                    part[tok] = part.get(tok, 0) + 1
            self.cache_size = size
        seq = hist[-3:] + [last]
        out = []
        while len(out) < self.n_draft:
            tok = None
            for i in range(3, -1, -1):
                part = self.ctx.get(tuple(seq[len(seq) - i - 1:]))
                if part is None:
                    continue
                max_c, sum_c, max_t = 0, 0, None
                for t, c in reversed(part.items()):
                    if c > max_c:
                        max_c, max_t = c, t
                    sum_c += c
                if sum_c < CACHE_MIN_SAMPLES[i] or 100 * max_c < CACHE_MIN_PERCENT[i] * sum_c:
                    continue
                tok = max_t
                break
            if tok is None:
                break
            out.append(tok)
            seq.append(tok)
        return out

    def accept(self, n_accepted):
        pass


class NgramCopy:
    """Q53's proposed ngram-copy (docs/research/notes/speculation_plumbing.md section 5): the longest match of the
    history's suffix (n_strong to n_longest tokens, newest on ties, may overlap the suffix, continues periodically past
    the end); k_short drafted tokens per match; k_long after n_enter fully accepted steps in a row, while the match has
    k_long tokens of continuation or overlaps its own suffix; long is held through n_hold missed steps.
    Variants: tiers = (k_short, k_long, k_longer, ...) climbs one tier after every n_enter fully accepted steps and falls
    back to k_short like the long state; enter_len > 0 goes long at once on a match of at least enter_len tokens."""

    def __init__(self, n_strong=6, n_longest=12, k_short=2, k_long=7, n_enter=2, n_hold=2, enter_len=0, tiers=None):
        self.ns, self.nl = n_strong, n_longest
        self.tiers = list(tiers) if tiers else [k_short, k_long]
        self.n_enter, self.n_hold, self.enter_len = n_enter, n_hold, enter_len

    def begin(self, prompt):
        self.ends = defaultdict(list)   # n_strong-gram -> end positions in the history, ascending
        self.n_indexed = 0
        self.level, self.n_full, self.hold, self.n_last = 0, 0, 0, 0

    def lookup(self, hist, last):
        """(match length, match end) of the longest qualifying match, or None."""
        ns, nl, t = self.ns, self.nl, len(hist)
        for e in range(max(self.n_indexed, ns - 1), t):
            self.ends[tuple(hist[e - ns + 1:e + 1])].append(e)
        self.n_indexed = max(self.n_indexed, t)
        cands = self.ends.get(tuple(hist[t - ns + 1:]) + (last,))
        if not cands:
            return None
        best_len, best_end = 0, -1
        for e in reversed(cands):
            ln = ns
            while ln < nl and e - ln >= 0 and hist[e - ln] == hist[t - ln]:
                ln += 1
            if ln > best_len:
                best_len, best_end = ln, e
                if ln == nl:
                    break
        return best_len, best_end

    @staticmethod
    def continuation(hist, last, end, k):
        t, out = len(hist), []
        for i in range(k):
            src = end + 1 + i
            out.append(hist[src] if src < t else last if src == t else out[src - t - 1])
        return out

    def _miss(self):
        if self.hold == 0:
            self.level, self.n_full = 0, 0
        else:
            self.hold -= 1

    def draft(self, hist, last, room):
        self.n_last = 0
        found = self.lookup(hist, last)
        if found is None:
            if self.level:
                self._miss()
            return []
        ln, end = found
        n_avail = len(hist) - end

        def fits(lvl):   # enough continuation, or a periodic one
            return n_avail >= self.tiers[lvl] or n_avail <= ln
        if not self.level and self.enter_len and ln >= self.enter_len and fits(1):
            self.level, self.hold, self.n_full = 1, self.n_hold, 0
        lvl = self.level if fits(self.level) else 0
        while lvl and self.tiers[lvl] > room:
            lvl -= 1
        k = self.tiers[lvl] if self.tiers[lvl] <= room else 0
        self.n_last = k
        return self.continuation(hist, last, end, k) if k else []

    def accept(self, n_accepted):
        if n_accepted == self.n_last:
            self.n_full += 1
            if self.level:
                self.hold = self.n_hold
            if self.n_full >= self.n_enter and self.level < len(self.tiers) - 1:
                self.level, self.n_full, self.hold = self.level + 1, 0, self.n_hold
        else:
            self.n_full = 0
            if self.level:
                self._miss()


class OracleCopy:
    """A bound, not a drafter: it sees the output and drafts exactly the rest of the copy (from the greedy parse,
    >= 8 tokens) that the next token begins or continues, at most k tokens. Every drafted token is accepted and nothing
    is drafted in fresh text, so it is the most that copying from the context can give at a given block cap. With skip,
    it drafts only from the skip-th token of a copy on, as a drafter that needs a key of that length must."""

    def __init__(self, k, skip=0):
        self.k, self.skip = k, skip

    def set_transcript(self, prompt, gen):
        self.gen, self.rest = gen, [0] * (len(gen) + 1)
        for start, ln, _ in copy_factors(prompt, gen, 8):
            for j in range(start + self.skip, start + ln):
                self.rest[j] = start + ln - j

    def begin(self, prompt):
        self.base = len(prompt)

    def draft(self, hist, last, room):
        j = len(hist) - self.base + 1   # gen index of the token after `last`
        if j >= len(self.gen):
            return []
        return self.gen[j:j + min(self.k, self.rest[j])]

    def accept(self, n_accepted):
        pass


def drafter_settings():
    """(name, factory) for every drafter setting replayed. Defaults are the fork's; the rest are the documented
    examples and shorter keys and drafts."""
    s = [
        ("ngram-simple 12/48 (default)", lambda: NgramSimple(12, 48)),
        ("ngram-simple 6/16", lambda: NgramSimple(6, 16)),
        ("ngram-simple 4/8", lambda: NgramSimple(4, 8)),
        ("ngram-simple 3/4", lambda: NgramSimple(3, 4)),
        ("ngram-map-k 12/48 (default)", lambda: NgramMap(12, 48, True)),
        ("ngram-map-k 6/16", lambda: NgramMap(6, 16, True)),
        ("ngram-map-k 4/8", lambda: NgramMap(4, 8, True)),
        ("ngram-map-k4v 12/48 (default)", lambda: NgramMap(12, 48, False, 1)),
        ("ngram-map-k4v 8/8 hits 2 (docs)", lambda: NgramMap(8, 8, False, 2)),
        ("ngram-map-k4v 8/24", lambda: NgramMap(8, 24, False, 1)),
        ("ngram-mod 24/48/64 (default)", lambda: NgramMod(24, 48, 64)),
        ("ngram-mod 24/16/32", lambda: NgramMod(24, 16, 32)),
        ("ngram-mod 16/8/16", lambda: NgramMod(16, 8, 16)),
        ("ngram-mod 12/2/8", lambda: NgramMod(12, 2, 8)),
        ("ngram-cache (8)", lambda: NgramCache(8)),
        ("ngram-copy Q53 6/12 k2/7 e2 h2", lambda: NgramCopy()),
        ("ngram-copy k_short 1", lambda: NgramCopy(k_short=1)),
        ("ngram-copy k_short 3", lambda: NgramCopy(k_short=3)),
        ("ngram-copy k_long 5", lambda: NgramCopy(k_long=5)),
        ("ngram-copy k_long 11", lambda: NgramCopy(k_long=11)),
        ("ngram-copy k_long 15", lambda: NgramCopy(k_long=15)),
        ("ngram-copy k_long 31", lambda: NgramCopy(k_long=31)),
        ("ngram-copy no long (k 2)", lambda: NgramCopy(k_long=2)),
        ("ngram-copy n_strong 4", lambda: NgramCopy(n_strong=4)),
        ("ngram-copy n_strong 8", lambda: NgramCopy(n_strong=8)),
        ("ngram-copy n_longest 24", lambda: NgramCopy(n_longest=24)),
        ("ngram-copy enter 1", lambda: NgramCopy(n_enter=1)),
        ("ngram-copy enter 3", lambda: NgramCopy(n_enter=3)),
        ("ngram-copy hold 0", lambda: NgramCopy(n_hold=0)),
        ("ngram-copy hold 4", lambda: NgramCopy(n_hold=4)),
        ("ngram-copy long at once", lambda: NgramCopy(enter_len=6)),
        ("ngram-copy long only, from 12", lambda: NgramCopy(k_short=0, enter_len=12)),
        ("ngram-copy tiers 2/7/15", lambda: NgramCopy(tiers=(2, 7, 15))),
        ("ngram-copy tiers 2/7/15/31", lambda: NgramCopy(tiers=(2, 7, 15, 31))),
        ("ngram-copy tiers 2/7/15/63", lambda: NgramCopy(tiers=(2, 7, 15, 63))),
        ("ngram-copy tiers 2/7/31", lambda: NgramCopy(tiers=(2, 7, 31))),
        ("ngram-copy tiers 2/7/15/31, n_strong 4", lambda: NgramCopy(n_strong=4, tiers=(2, 7, 15, 31))),
        ("oracle copy <= 7 (bound)", lambda: OracleCopy(7)),
        ("oracle copy <= 15 (bound)", lambda: OracleCopy(15)),
        ("oracle copy <= 63 (bound)", lambda: OracleCopy(63)),
        ("oracle copy <= 7, after 6 (bound)", lambda: OracleCopy(7, 6)),
        ("oracle copy <= 31, after 6 (bound)", lambda: OracleCopy(31, 6)),
    ]
    return s


# ---------------------------------------------------------------------------------------------------------------------
# The server loop
# ---------------------------------------------------------------------------------------------------------------------

def replay(drafter, prompt, gen, n_ctx=N_CTX, n_predict=N_PREDICT):
    """Steps of one request as (position of the sampled token in gen, drafted, accepted). gen[0] comes from the prompt
    pass; every step verifies gen[i] plus the draft and yields accepted + 1 tokens (fewer at the end of the output)."""
    hist = list(prompt)
    drafter.begin(hist)
    steps, i, n_gen = [], 0, len(gen)
    while i < n_gen - 1:
        room = min(n_ctx - len(hist) - 2, n_predict - (i + 1) - 1)   # slot.get_n_draft_max()
        d = drafter.draft(hist, gen[i], room) if room > 0 else []
        if len(d) > room:
            d = d[:room]
        a = 0
        if d:
            lim = min(len(d), n_gen - 1 - i)
            while a < lim and d[a] == gen[i + 1 + a]:
                a += 1
            drafter.accept(a)
        steps.append((i, len(d), a))
        adv = min(a + 1, n_gen - 1 - i)
        hist.extend(gen[i:i + adv])
        i += adv
    return steps


# ---------------------------------------------------------------------------------------------------------------------
# Copy structure of the output, independent of any drafter
# ---------------------------------------------------------------------------------------------------------------------

def copy_factors(prompt, gen, min_len=8):
    """Greedy longest-previous-match parse of gen[1:] against the prompt and the earlier output (sources may overlap
    the copy). Returns (start in gen, length, 'prompt' | 'output') for every copy of at least min_len tokens."""
    s, base = prompt + gen, len(prompt)
    total = len(s)
    starts = defaultdict(list)
    n_indexed, p, out = 0, base + 1, []
    while p < total:
        for q in range(n_indexed, min(p, total - min_len + 1)):
            starts[tuple(s[q:q + min_len])].append(q)
        n_indexed = max(n_indexed, min(p, total - min_len + 1))
        best, src = 0, -1
        if p + min_len <= total:
            for q in reversed(starts.get(tuple(s[p:p + min_len]), ())):
                ln = min_len
                while p + ln < total and s[q + ln] == s[p + ln]:
                    ln += 1
                if ln > best:
                    best, src = ln, q
        if best >= min_len:
            out.append((p - base, best, "prompt" if src < base else "output"))
            p += best
        else:
            p += 1
    return out


def copy_runs(factors):
    """Lengths of maximal stretches of back-to-back copies."""
    runs, end = [], None
    for start, ln, _ in factors:
        if end == start:
            runs[-1] += ln
        else:
            runs.append(ln)
        end = start + ln
    return runs


def match_profile(prompt, gen, in_copy, n_strong=6, n_longest=12):
    """For every gen position j >= 1: ("copy" | "fresh", is there an ngram-copy match of >= n_strong tokens before it,
    does the match's continuation equal gen[j])."""
    cp = NgramCopy(n_strong=n_strong, n_longest=n_longest)
    hist = list(prompt)
    cp.begin(hist)
    out = []
    for j in range(1, len(gen)):
        found = cp.lookup(hist, gen[j - 1])
        right = bool(found) and cp.continuation(hist, gen[j - 1], found[1], 1)[0] == gen[j]
        out.append(("copy" if in_copy[j] else "fresh", bool(found), right))
        hist.append(gen[j - 1])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Cost model: v(n) = time of a decode step that verifies n tokens, in T1 (one plain token on the fork's kernels)
# ---------------------------------------------------------------------------------------------------------------------

STOCK = 1.28        # n >= 2 runs llama.cpp's stock multi-token kernels: the kernels-off token, 34.25 / 26.76 ms
PP_FLOOR = 0.197    # no verify is cheaper per token than prompt processing: ~190 tokens/s pp512 against 37.4 tokens/s
Q51_DISTINCT = {1: 6.0, 2: 11.05, 4: 20.13, 8: 35.58, 16: 59.74}   # Q51: distinct experts per layer for n tokens


def random_distinct(n):
    return 256 * (1 - (1 - 6 / 256) ** n)


def q51_distinct(n):
    if n >= 16:
        return random_distinct(n) * Q51_DISTINCT[16] / random_distinct(16)
    lo = max(k for k in Q51_DISTINCT if k <= n)
    hi = min(k for k in Q51_DISTINCT if k >= n)
    if lo == hi:
        return Q51_DISTINCT[lo]
    f = (math.log2(n) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return Q51_DISTINCT[lo] + f * (Q51_DISTINCT[hi] - Q51_DISTINCT[lo])


def model_curves():
    """The curves of speculation_plumbing.md section 6.1, with the prompt-processing floor; "Q51" is the pessimistic
    byte model with Flint's measured expert overlap in place of random routing."""
    def floor(f):
        return lambda n: 1.0 if n == 1 else max(f(n), PP_FLOOR * n)
    return {
        "optimistic": floor(lambda n: STOCK + 0.0747 * (n - 1)),
        "mid": floor(lambda n: STOCK + 0.209 * (n - 1)),
        "Q51": floor(lambda n: STOCK * (0.68 + 0.32 * q51_distinct(n) / 6)),
        "pessimistic": floor(lambda n: STOCK * (0.68 + 0.32 * random_distinct(n) / 6)),
    }


def measured_curve(spec):
    """v(n) from "1=26.8,2=33.0,..." (any unit), divided by v(1); linear between points, beyond the last one along the
    last segment's slope, never below the prompt-processing floor."""
    pts = sorted((int(k), float(v)) for k, v in (x.split("=") for x in spec.split(",")))
    v1 = dict(pts)[1]
    pts = [(n, v / v1) for n, v in pts]

    def v(n):
        if n == 1:
            return 1.0
        for (n0, v0), (n1, v1_) in zip(pts, pts[1:]):
            if n0 <= n <= n1:
                return v0 + (v1_ - v0) * (n - n0) / (n1 - n0)
        (n0, v0), (n1, v1_) = pts[-2], pts[-1]
        return max(v1_ + (v1_ - v0) / (n1 - n0) * (n - n1), PP_FLOOR * n)
    return v


# Host costs of a change of verify width (speculation_plumbing.md 1.6), in T1: graph rebuild R, eager step U, capture C
GRAPH_R, GRAPH_U, GRAPH_C = 0.15, 0.18, 0.35
GRAPH_MODES = ("q52", "today")   # q52: one graph per width (Q52 layers 1 + 2), no switching cost


def graph_costs(widths, mode):
    if mode == "q52":
        return [0.0] * len(widths)
    out, prev, switched = [], 1, False
    for w in widths:
        if w != prev:
            out.append(GRAPH_R + GRAPH_U)
            switched = True
        else:
            out.append(GRAPH_C if switched else 0.0)
            switched = False
        prev = w
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------------------------------------------------

def width_bucket(w):
    return "1" if w == 1 else "2-3" if w <= 3 else "4-8" if w <= 8 else "9-16" if w <= 16 else "17+"


WIDTH_BUCKETS = ("1", "2-3", "4-8", "9-16", "17+")


def new_acc():
    return {"tokens": 0, "steps": 0, "draft_steps": 0, "drafted": 0, "accepted": 0, "full": 0,
            "widths": Counter(), "graph": Counter()}


def add_transcript(accs, steps, n_gen, category, segs):
    """Add one request's steps to the totals of its class, of all classes, of its step's segment, and of both."""
    widths = [k + 1 for _, k, _ in steps]
    gcost = [(g, graph_costs(widths, g)) for g in GRAPH_MODES]
    for s, (i, k, a) in enumerate(steps):
        w = k + 1
        seg = segs[i]
        for key in (category, "all", "seg_" + seg, category + "/" + seg):
            acc = accs[key]
            acc["tokens"] += min(a + 1, n_gen - 1 - i)
            acc["steps"] += 1
            acc["widths"][w] += 1
            if k:
                acc["draft_steps"] += 1
                acc["drafted"] += k
                acc["accepted"] += a
                acc["full"] += a == k
            for g, gc in gcost:
                acc["graph"][g] += gc[s]


def summarize(acc, curves):
    t, st = acc["tokens"], acc["steps"]
    cost = {(c, g): sum(n * v(w) for w, n in acc["widths"].items()) + acc["graph"][g]
            for c, v in curves.items() for g in GRAPH_MODES}
    out = {
        "tokens": t, "steps": st, "draft_steps": acc["draft_steps"], "drafted": acc["drafted"],
        "accepted": acc["accepted"],
        "draft_step_share": acc["draft_steps"] / st if st else 0,
        "acceptance": acc["accepted"] / acc["drafted"] if acc["drafted"] else 0,
        "full_share": acc["full"] / acc["draft_steps"] if acc["draft_steps"] else 0,
        "tokens_per_step": t / st if st else 0,
        "accepted_per_draft_step": acc["accepted"] / acc["draft_steps"] if acc["draft_steps"] else 0,
        "from_drafts": acc["accepted"] / t if t else 0,
        "widths": {b: sum(c for w, c in acc["widths"].items() if width_bucket(w) == b) / st if st else 0
                   for b in WIDTH_BUCKETS},
        "width_hist": dict(sorted(acc["widths"].items())),
        "speedup": {f"{c}/{g}": (t / cost[(c, g)] if cost[(c, g)] else 0)
                    for c in curves for g in GRAPH_MODES},
    }
    return out


def load(path):
    rows = [json.loads(line) for line in open(path)]
    for r in rows:
        r["segs"] = segments_of(r["gen_tokens"])
    return rows


def copy_report(rows):
    """Copy structure per class, for all classes, and per class and segment."""
    tok, runs, prof = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    for r in rows:
        gen, segs, cls = r["gen_tokens"], r["segs"], r["category"]
        fac = copy_factors(r["prompt_tokens"], gen, 8)
        in_copy = [False] * len(gen)
        for start, ln, src in fac:
            for j in range(start, start + ln):
                in_copy[j] = True
                for key in (cls, "all", cls + "/" + segs[j]):
                    tok[key]["copy8_" + src] += 1
                    tok[key]["copy8_" + segs[j]] += 1
                    if ln >= 32:
                        tok[key]["copy32"] += 1
        for j in range(1, len(gen)):
            for key in (cls, "all", cls + "/" + segs[j]):
                tok[key]["gen"] += 1
                tok[key]["seg_" + segs[j]] += 1
        for key in (cls, "all"):
            runs[key] += copy_runs(fac)
        for seg in SEGMENTS:   # runs that start in this segment
            runs[cls + "/" + seg] += copy_runs([f for f in fac if segs[f[0]] == seg])
        for j, (where, found, right) in enumerate(match_profile(r["prompt_tokens"], gen, in_copy), 1):
            for key in (cls, "all", cls + "/" + segs[j]):
                prof[key][where] += 1
                prof[key][where + "_match"] += found
                prof[key][where + "_right"] += right
    rep = {}
    keys = list(CLASSES) + ["all"] + [f"{c}/{s}" for c in CLASSES for s in SEGMENTS]
    for key in keys:
        t, p, rs = tok[key], prof[key], sorted(runs[key])
        if not t["gen"]:
            continue
        copy8 = t["copy8_prompt"] + t["copy8_output"]
        half, acc, med = sum(rs) / 2, 0, 0
        for x in rs:
            acc += x
            if acc >= half:
                med = x
                break
        rep[key] = {
            "transcripts": sum(1 for r in rows if key.split("/")[0] in ("all", r["category"])),
            "gen_tokens": t["gen"],
            "segments": {s: t["seg_" + s] for s in SEGMENTS},
            "copy8_share": copy8 / t["gen"], "copy8_from_prompt": t["copy8_prompt"] / t["gen"],
            "copy8_from_output": t["copy8_output"] / t["gen"], "copy32_share": t["copy32"] / t["gen"],
            "copy8_share_by_segment": {s: (t["copy8_" + s] / t["seg_" + s] if t["seg_" + s] else 0)
                                       for s in SEGMENTS},
            "runs": len(rs), "run_mean": sum(rs) / len(rs) if rs else 0, "run_token_median": med,
            "run_max": rs[-1] if rs else 0,
            "fresh_match": p["fresh_match"] / p["fresh"] if p["fresh"] else 0,
            "fresh_right_given_match": p["fresh_right"] / p["fresh_match"] if p["fresh_match"] else 0,
            "copy_match": p["copy_match"] / p["copy"] if p["copy"] else 0,
            "copy_right_given_match": p["copy_right"] / p["copy_match"] if p["copy_match"] else 0,
        }
    return rep


def pct(x):
    return f"{100 * x:.1f}"


def print_report(res, curves):
    p = print
    p("## Transcripts and copy structure\n")
    p("| Class | Transcripts | Generated | Think / answer / tool | In copies ≥ 8 (prompt / own output) | In copies ≥ 32 |"
      " Copy runs | Mean run | Token-weighted median run | Longest |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for cls, c in res["copy"].items():
        sg = c["segments"]
        p(f"| {cls} | {c['transcripts']} | {c['gen_tokens']:,} | {sg['think']:,} / {sg['answer']:,} / {sg['tool']:,} |"
          f" {pct(c['copy8_share'])} % ({pct(c['copy8_from_prompt'])} / {pct(c['copy8_from_output'])}) |"
          f" {pct(c['copy32_share'])} % | {c['runs']} | {c['run_mean']:.1f} | {c['run_token_median']} | {c['run_max']} |")
    p("\n| Class | Copy share in think / answer / tool | ≥ 6-token match before a fresh token | its next token right |"
      " ≥ 6-token match inside a copy | its next token right |")
    p("|---|---|---|---|---|---|")
    for cls, c in res["copy"].items():
        b = c["copy8_share_by_segment"]
        p(f"| {cls} | {pct(b['think'])} / {pct(b['answer'])} / {pct(b['tool'])} % | {pct(c['fresh_match'])} % |"
          f" {pct(c['fresh_right_given_match'])} % | {pct(c['copy_match'])} % | {pct(c['copy_right_given_match'])} % |")

    cn = list(curves)
    for cls in CLASSES + ("all", "edit/answer"):
        p(f"\n## {cls}\n")
        p("| Drafter | Steps with a draft | Accepted / drafted | Fully accepted | Tokens per step |"
          " Widths 1 / 2-3 / 4-8 / 9-16 / 17+ (% of steps) | " + " | ".join(f"{c}" for c in cn) + " | mid, today |")
        p("|---|---|---|---|---|---|" + "---|" * len(cn) + "---|")
        for name, by in res["drafters"].items():
            s = by[cls]
            w = " / ".join(f"{100 * s['widths'][b]:.0f}" for b in WIDTH_BUCKETS)
            sp = " | ".join(f"{s['speedup'][c + '/q52']:.2f}" for c in cn)
            p(f"| {name} | {pct(s['draft_step_share'])} % | {pct(s['acceptance'])} % | {pct(s['full_share'])} % |"
              f" {s['tokens_per_step']:.2f} | {w} | {sp} | {s['speedup']['mid/today']:.2f} |")

    p("\n## By segment (all classes)\n")
    p("| Drafter | Segment | Tokens | Steps with a draft | Accepted / drafted | Tokens per step | mid |")
    p("|---|---|---|---|---|---|---|")
    for name, by in res["drafters"].items():
        for seg in SEGMENTS:
            s = by["seg_" + seg]
            p(f"| {name} | {seg} | {s['tokens']:,} | {pct(s['draft_step_share'])} % | {pct(s['acceptance'])} % |"
              f" {s['tokens_per_step']:.2f} | {s['speedup']['mid/q52']:.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen")
    ap.add_argument("--out")
    ap.add_argument("--vn")
    ap.add_argument("--only")
    ap.add_argument("--dump-steps")
    ap.add_argument("--no-copy", action="store_true", help="skip the copy-structure pass")
    args = ap.parse_args()

    rows = load(args.gen)
    curves = model_curves()
    if args.vn:
        curves = {"measured": measured_curve(args.vn), **curves}
    curves = {c: functools.lru_cache(maxsize=None)(v) for c, v in curves.items()}
    res = {"curves": {c: {n: round(v(n), 3) for n in (2, 3, 4, 8, 16, 32, 49, 65)} for c, v in curves.items()},
           "copy": {} if args.no_copy else copy_report(rows), "drafters": {}}
    dump = open(args.dump_steps, "w") if args.dump_steps else None
    for name, make in drafter_settings():
        if args.only and not re.search(args.only, name):
            continue
        accs = defaultdict(new_acc)
        for r in rows:
            gen = r["gen_tokens"]
            d = make()
            if hasattr(d, "set_transcript"):
                d.set_transcript(r["prompt_tokens"], gen)
            steps = replay(d, r["prompt_tokens"], gen)
            if dump:
                for i, k, a in steps:
                    dump.write(f"{name}\t{r['id']}\t{i}\t{k}\t{a}\n")
            add_transcript(accs, steps, len(gen), r["category"], r["segs"])
        res["drafters"][name] = {k: summarize(v, curves) for k, v in accs.items()}
        a = res["drafters"][name]["all"]
        print(f"{name}: {a['tokens_per_step']:.3f} tokens/step, mid {a['speedup']['mid/q52']:.3f}", file=sys.stderr)
    if dump:
        dump.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print_report(res, curves)


if __name__ == "__main__":
    main()
