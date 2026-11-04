"""Self-test of the match code against brute force (no model, no data):  python3 selftest.py

- build_sam / match_profile: for random sequences over a small alphabet (many repeats), the longest earlier match of
  every position and the end of its newest occurrence equal the brute-force answer, with and without the response's
  own text as a source, with and without the prompt tail, and with a small n-gram cap.
- covered_flags / runs: against a direct definition.
- analyze.simulate / oracle / chains: the guesser's steps against a plain re-implementation that recomputes the longest
  match from scratch at every step.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze  # noqa: E402
from common import build_sam, covered_flags, match_profile, runs  # noqa: E402


def brute_profile(ctx, g, tail, own, maxn):
    tail = min(tail, len(ctx))
    base = len(ctx) - tail
    hist = ctx[:base] + ctx[base:] + g          # = ctx + g, coordinates as in match_profile
    stream_start = base
    res = []
    for e in range(len(g)):
        t = tail + e                              # position in the stream
        cur = base + t                            # position in hist of the token just generated
        best_l, best_q = 0, -1
        limit = t + 1                             # the matched suffix cannot start before the stream
        for q in range(cur):                      # earlier ends only
            in_stream = q >= stream_start
            if in_stream and not own:
                continue
            l = 0
            cap = min(limit, maxn, q - stream_start + 1) if in_stream else limit
            while l < cap and q - l >= 0 and hist[q - l] == hist[cur - l]:
                l += 1
            if l > best_l or (l == best_l and l > 0 and q > best_q):
                best_l, best_q = l, q
        res.append((best_l, best_q))
    return res


def check_profiles(rng):
    n = 0
    for _ in range(300):
        sigma = rng.choice([2, 3, 5])
        ctx = [rng.randrange(sigma) for _ in range(rng.randrange(1, 70))]
        g = [rng.randrange(sigma) for _ in range(rng.randrange(1, 60))]
        if rng.random() < 0.5 and len(ctx) > 10:
            s = rng.randrange(len(ctx) - 5)
            g[:5] = ctx[s:s + 5]                  # plant a copy
        for tail, own, maxn in ((0, False, 48), (4, True, 48), (7, True, 5), (3, False, 48)):
            pl, pq = match_profile(ctx, g, tail=tail, own=own, maxn=maxn)
            want = brute_profile(ctx, g, tail, own, maxn)
            got = list(zip(pl, pq))
            if want != got:
                i = next(k for k in range(len(g)) if want[k] != got[k])
                raise SystemExit("profile mismatch: tail %d own %s maxn %d pos %d: got %s want %s\nctx %s\ng %s" % (
                    tail, own, maxn, i, got[i], want[i], ctx, g))
            n += 1
    return n


def check_flags(rng):
    for _ in range(500):
        pl = []                                   # match lengths grow by at most one per position
        for _ in range(rng.randrange(1, 40)):
            pl.append(min(rng.randrange(0, 9), (pl[-1] + 1) if pl else 1))
        for ml in (1, 3, 5):
            want = [False] * len(pl)
            for e, l in enumerate(pl):
                L = min(l, e + 1)
                if L >= ml:
                    for p in range(e - L + 1, e + 1):
                        want[p] = True
            assert covered_flags(pl, ml) == want, (pl, ml)
    assert runs([True, True, False, True, False, False, True]) == [(0, 2), (3, 1), (6, 1)]
    return 1500


def naive_sim(ctx, g, pl, pq, n_draft, m_min):
    hist, lc = ctx + g, len(ctx)
    i, steps, tokens, props, acc, cols = 1, 0, 0, 0, 0, 0
    while i < len(g):
        l, q = pl[i - 1], pq[i - 1]
        draft = []
        if l >= m_min and q >= 0:
            end = min(q + 1 + analyze.draft_len(n_draft, l), lc + i)
            draft = hist[q + 1:end]
        a = 0
        while a < len(draft) and i + a < len(g) and draft[a] == g[i + a]:
            a += 1
        steps += 1
        cols += 1 + len(draft)
        tokens += min(a + 1, len(g) - i)
        if draft:
            props += 1
            acc += a
        i += a + 1
    return [steps, tokens, props, acc, cols]


def naive_occ(ctx, g, n_draft, m):
    hist, lc = ctx + g, len(ctx)
    i, steps, tokens, props, acc, cols = 1, 0, 0, 0, 0, 0
    while i < len(g):
        cur = lc + i
        best = 0
        for q in range(m - 1, cur - 1):          # earlier occurrences of the last m tokens, with a next token
            if cur >= m and hist[q - m + 1:q + 1] == hist[cur - m:cur]:
                a = 0
                while a < n_draft and i + a < len(g) and q + 1 + a < cur and hist[q + 1 + a] == hist[cur + a]:
                    a += 1
                best = max(best, a)
        steps += 1
        cols += 1 + best
        tokens += min(best + 1, len(g) - i)
        props += best > 0
        acc += best
        i += best + 1
    return [steps, tokens, props, acc, cols]


def check_sim(rng):
    n = 0
    for _ in range(200):
        ctx = [rng.randrange(3) for _ in range(rng.randrange(20, 80))]
        s = rng.randrange(len(ctx) - 12)
        g = [rng.randrange(3) for _ in range(5)] + ctx[s:s + 12] + [rng.randrange(3) for _ in range(rng.randrange(3, 30))]
        pl, pq = match_profile(ctx, g, tail=4, own=True)
        for nd in (2, 8, "adapt"):
            for m in (1, 3, 6):
                got = analyze.simulate(ctx, g, pl, pq, nd, m, ["r"] * len(g))
                want = naive_sim(ctx, g, pl, pq, nd, m)
                assert [got[0], got[1], got[2], got[3], got[5]] == want, (nd, m, got, want)
                assert got[1] == len(g) - 1 or got[0] == 0, (got, len(g))
                n += 1
        for nd in (2, 8):
            for m in (1, 2, 3):
                got = analyze.occ_oracle(ctx, g, nd, m)
                want = naive_occ(ctx, g, nd, m)
                assert [got[0], got[1], got[2], got[3], got[5]] == want, (nd, m, got, want)
        orc = analyze.oracle(g, covered_flags(pl, 4), 8, 2)
        assert orc[1] == len(g) - 1, orc
        ch = analyze.chains(ctx, g, pl, pq, covered_flags(pl, 4))
        assert all(v[0] >= v[1] >= v[3] >= v[4] for v in ch.values())
    return n


if __name__ == "__main__":
    rng = random.Random(11)
    print("match profiles equal brute force in %d cases" % check_profiles(rng))
    print("covered_flags/runs equal their definition in %d cases" % check_flags(rng))
    print("guesser steps equal the plain re-implementation in %d cases" % check_sim(rng))
    sam = build_sam([1, 2, 1, 2, 1])
    print("ok (suffix automaton with %d states)" % len(sam[0]))
