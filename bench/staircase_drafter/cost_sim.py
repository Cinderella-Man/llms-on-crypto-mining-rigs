"""What one guess of the staircase head may cost on agent sessions: pipelined speculation (the event model of
bench/ngram_replay/pipeline_sim.py) with llama-server's guess order and the head's time charged, swept over the head's
time per guess and its acceptance, against the release line's guesser (`cont`).

The guessers, from common/pipe-guess.cpp:
  cont   the copy lookup (match >= 4 of up to 12 tokens, newest 256 occurrences), else the request's most frequent
         continuation of the longest 1-8-token context (the release line's --pipe-guess cont)
  new    the copy lookup, else the request's continuations of 2-token or longer contexts, else the head
         (--pipe-guess staircase from fork branch pascal-p106-v3-cheapguess on)
  old    the copy lookup, else the head (--pipe-guess staircase without a store before that branch)
  copy   the copy lookup alone;  none  no guessing
The lookups run on the true prefix of each position (one pass per session, history grown across its requests; the
continuation counts cover the committed tokens up to two before the guess, and the server's 32,768-token cap on them
is left out); while a wrong guess is in flight they guess nothing (a lookup on a wrong token rarely matches, and the
position is cancelled anyway).

The head (staircase, on card 1 by default): after each submission whose next position the lookups cannot guess, the
server launches the head for that position. It starts when card 1 has finished the submitted token (plus --tap-ms
when it runs on another card), holds its card for --head-ms (the head and Flint's stage share the card's memory
bandwidth, so their times add), and its guess is right with probability A (a fixed draw per position). The server
waits for the guess before it does anything else, so the host is busy until it is ready. A launch is never withdrawn:
a head running for a position that a miss cancels still holds its card.

Pipeline (pipeline_sim.simulate): per-card stage times from the plain time per token and the layers per card (output
head and sampling on the last card), a hop per card boundary, the host's gap from a sample to the next launch, at
most --depth guesses beyond the newest sampled token, a miss cancels everything behind it (running stages finish).
Generation tokens per second = generated tokens over the time from each request's first token to its last sample.

usage: python cost_sim.py SESSIONS_SEQ.jsonl --ids test.txt [--total-ms 34.1] [--layers 6,6,5,5,6,5,5,6,5]
                          [--depth 4] [--head-ms 0,1,2,3,5.3] [--accept 0.15,0.2,0.3] [--jobs 8] [--out OUT.json]
Needs no numpy.
"""
import argparse
import json
import math
import random
import zlib
from collections import Counter
from multiprocessing import Pool

MAX_CTX, MAX_COPY, MAX_CAND = 8, 12, 256
HEAD_MS = 2.5   # Flint's output head and sampling on the last card (pipeline_sim.HEAD_MS)
WRONG = -1


class ServerGuesser:
    """common_pipe_guesser's copy lookup and continuation counts, on one growing history."""

    def __init__(self, min_match=4):
        self.hist, self.pos_of, self.counts, self.n_counted, self.min_match = [], {}, {}, 0, min_match

    def push(self, t):
        self.pos_of.setdefault(t, []).append(len(self.hist))
        self.hist.append(t)

    def count_to(self, n):
        h, counts = self.hist, self.counts
        for j in range(max(self.n_counted, 1), n):
            nxt = h[j]
            for ln in range(1, min(MAX_CTX, j) + 1):
                d = counts.setdefault(tuple(h[j - ln:j]), {})
                d[nxt] = d.get(nxt, 0) + 1
        self.n_counted = max(self.n_counted, n)

    def copy_guess(self):
        h = self.hist
        n = len(h)
        if n < 2:
            return None
        best_len, best_end, seen = 0, -1, 0
        for i in reversed(self.pos_of.get(h[-1], ())):
            if i >= n - 1:
                continue
            seen += 1
            if seen > MAX_CAND:
                break
            ln = 1
            while ln < MAX_COPY and i - ln >= 0 and h[i - ln] == h[n - 1 - ln]:
                ln += 1
            if ln > best_len:
                best_len, best_end = ln, i
                if ln == MAX_COPY:
                    break
        return h[best_end + 1] if best_len >= self.min_match else None

    def cont_guess(self, min_len):
        h = self.hist
        n = len(h)
        for ln in range(min(MAX_CTX, n), min_len - 1, -1):
            d = self.counts.get(tuple(h[n - ln:]))
            if d is None:
                continue
            best, bc = None, 0
            for t, c in d.items():
                if c > bc:
                    best, bc = t, c
            return best
        return None


def session_spans(r):
    """(generated tokens, (copy, cont2, cont1) guesses per position) for every request of a session."""
    seq = r["prompt_tokens"] + r["gen_tokens"]
    ranges = r.get("gen_ranges") or [(len(r["prompt_tokens"]), len(seq))]
    g = ServerGuesser()
    out, k = [], 0
    for s, e in ranges:
        while k < s:
            g.push(seq[k])
            k += 1
        gen, guesses = seq[s:e], []
        for p in range(len(gen)):
            if p == 0:
                guesses.append((None, None, None))
            else:
                g.count_to(len(g.hist) - 2)
                c1 = g.cont_guess(1)
                c2 = g.cont_guess(2)
                guesses.append((g.copy_guess(), c2, c1))
            g.push(gen[p])
            k += 1
        if len(gen) >= 2:
            out.append((gen, guesses))
    return out


def stage_times(total_ms, layers, hop, gap, head=HEAD_MS):
    per_layer = (total_ms - gap - (len(layers) - 1) * hop - head) / sum(layers)
    st = [n * per_layer for n in layers]
    st[-1] += head
    return st


def simulate(gen, guesses, cfg, seed=1):
    """(ms from the first token's launch to the last sample, Counter) for one request."""
    stages, hop, gap, depth = cfg["stages"], cfg["hop"], cfg["gap"], cfg["depth"]
    order, head_ms, acc, head_card, tap = cfg["order"], cfg["head_ms"], cfg["accept"], cfg["head_card"], cfg["tap_ms"]
    S, n = len(stages), len(gen)
    rng = random.Random(seed ^ zlib.crc32(bytes(str(n) + str(gen[:16]), "ascii")))
    u = [rng.random() for _ in range(n)]
    card_free = [0.0] * S
    card_log = [[] for _ in range(S)]   # (start, end, position, kept) per card; head runs are kept on a cancel
    spare_free = [0.0]
    sched, launch = {}, {}
    host = [0.0]
    st = Counter()
    uses_head = order in ("new", "old")

    def lookup(p):
        """the lookups' guess for position p, or None; never while a wrong guess is in flight"""
        for q in range(v[0] + 1, p):
            if sched[q][0] != gen[q]:
                return None
        cp, c2, c1 = guesses[p]
        if order == "none":
            return None
        if cp is not None:
            return cp, "copy"
        if order == "cont":
            # the longest context wins, so c1 is c2 whenever a 2-token or longer context has continuations
            return (c1, "cont2" if c2 is not None else "cont1") if c1 is not None else None
        if order == "new" and c2 is not None:
            return c2, "cont2"
        return None

    def occupy(j, s0, e0, pos, kept):
        card_log[j].append((s0, e0, pos, kept))
        if len(card_log[j]) > 96:
            del card_log[j][:-48]
        card_free[j] = max(card_free[j], e0)

    def launch_head(p):
        t_in = sched[p - 1][2][0] + (tap if head_card != 0 else 0.0)
        if head_card < 0:
            s0 = max(t_in, spare_free[0])
            spare_free[0] = s0 + head_ms
        else:
            s0 = max(t_in, card_free[head_card])
            occupy(head_card, s0, s0 + head_ms, p, True)
        launch[p] = s0 + head_ms
        st["head_launches"] += 1
        st["head_ms_total"] += head_ms

    def schedule(pos, tok, src, ready):
        s_, e_ = [0.0] * S, [0.0] * S
        for j in range(S):
            if j:
                ready = e_[j - 1] + hop
            s0 = max(ready, card_free[j] + hop)
            e0 = s0 + stages[j]
            occupy(j, s0, e0, pos, False)
            s_[j], e_[j] = s0, e0
        sched[pos] = [tok, s_, e_, src]
        # the server starts the head on the next position right after a submission the lookups cannot follow
        if uses_head and pos + 1 < n and lookup(pos + 1) is None:
            launch_head(pos + 1)

    def cancel_after(pos, t):
        for p in [p for p in sched if p > pos]:
            del sched[p]
            st["cancelled"] += 1
        for p in [p for p in launch if p > pos + 1]:
            del launch[p]
        for j in range(S):
            log = [x for x in card_log[j] if x[3] or x[2] <= pos or x[0] < t]
            for x in card_log[j]:
                if not (x[3] or x[2] <= pos) and x[0] < t:
                    st["wasted_ms"] += x[1] - x[0]
            card_log[j] = log
            card_free[j] = max((x[1] for x in log), default=0.0)

    def extend(now):
        last = max(sched)
        while last - v[0] < depth and last < n - 1:
            p = last + 1
            r = lookup(p)
            if r is not None:
                schedule(p, r[0], r[1], max(card_free[0], now) + hop)
            elif uses_head:
                if p not in launch:
                    launch_head(p)
                ready = launch.pop(p)
                host[0] = max(host[0], ready)
                tok = gen[p] if u[p] < acc else WRONG
                schedule(p, tok, "head", max(ready, now) + hop)
            else:
                break
            st["guessed"] += 1
            last = p

    v = [0]
    schedule(0, gen[0], "first", gap)
    extend(0.0)
    t_end = 0.0
    for q in range(n - 1):
        t_s = sched[q][2][S - 1]
        t_end = t_s
        now = max(t_s, host[0])
        x = gen[q + 1]
        nxt = sched.get(q + 1)
        del sched[q]
        v[0] = q + 1
        if nxt is not None and nxt[0] == x:
            st["hit_" + nxt[3]] += 1
        else:
            if nxt is not None:
                st["miss_" + nxt[3]] += 1
                cancel_after(q, t_s)
            else:
                st["none"] += 1
            schedule(q + 1, x, "sampled", now + gap)
        extend(now)
    t_end = sched[n - 1][2][S - 1]
    return t_end, st


def run_one(args):
    spans, cfg = args
    t, n, st = 0.0, 0, Counter()
    for gen, guesses in spans:
        dt, s = simulate(gen, guesses, cfg)
        t += dt
        n += len(gen)
        st.update(s)
    return cfg, t, n, st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("--ids")
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--total-ms", type=float, default=34.1, help="plain decoding's time per token (no guessing)")
    ap.add_argument("--layers", default="6,6,5,5,6,5,5,6,5")
    ap.add_argument("--hop", type=float, default=0.3)
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--head-ms", default="0,0.5,1,1.5,2,2.5,3,4,5.3")
    ap.add_argument("--accept", default="0.1,0.15,0.2,0.25,0.3,0.4")
    ap.add_argument("--orders", default="new,old")
    ap.add_argument("--head-card", type=int, default=0, help="the head's card (0 = card 1, -1 = a spare card)")
    ap.add_argument("--tap-ms", type=float, default=0.3, help="extra input latency of a head off card 1")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    keep = {x.strip() for x in open(args.ids) if x.strip()} if args.ids else None
    rows = []
    for line in open(args.sessions):
        r = json.loads(line)
        if keep is None or str(r.get("id")) in keep:
            rows.append(r)
            if args.max_sessions and len(rows) >= args.max_sessions:
                break
    with Pool(args.jobs) as pool:
        spans = [s for ss in pool.map(session_spans, rows) for s in ss]
    n_tok = sum(len(g) for g, _ in spans)
    layers = [int(x) for x in args.layers.split(",")]
    stages = stage_times(args.total_ms, layers, args.hop, args.gap)
    base = dict(stages=stages, hop=args.hop, gap=args.gap, depth=args.depth, head_card=args.head_card,
                tap_ms=args.tap_ms, head_ms=0.0, accept=0.0)
    cfgs = [dict(base, order=o) for o in ("none", "copy", "cont")]
    for o in args.orders.split(","):
        for h in [float(x) for x in args.head_ms.split(",")]:
            for a in [float(x) for x in args.accept.split(",")]:
                cfgs.append(dict(base, order=o, head_ms=h, accept=a))
    # spread the spans over the jobs per configuration
    with Pool(args.jobs) as pool:
        res = pool.map(run_one, [(spans, c) for c in cfgs], chunksize=1)
    print(f"{len(rows)} sessions, {len(spans)} requests, {n_tok} generated tokens; stages "
          + ", ".join(f"{s:.2f}" for s in stages) + f" ms; depth {args.depth}")
    out = []
    ref = {}
    for cfg, t, n, st in res:
        tps = 1000.0 * n / t
        row = {k: cfg[k] for k in ("order", "head_ms", "accept", "depth", "head_card")}
        row.update(tok_s=tps, stats=dict(st))
        out.append(row)
        if cfg["order"] in ("none", "copy", "cont"):
            ref[cfg["order"]] = tps
            hits = {k: v for k, v in st.items() if k.startswith(("hit_", "miss_"))}
            print(f"{cfg['order']:5s}: {tps:6.1f} tok/s  {hits}")
    for o in args.orders.split(","):
        hs = sorted({r["head_ms"] for r in out if r["order"] == o})
        accs = sorted({r["accept"] for r in out if r["order"] == o})
        print(f"\n{o}: tok/s by head time per guess (rows, ms) and acceptance (columns); cont = {ref['cont']:.1f}")
        print("| ms | " + " | ".join(f"{a:.2f}" for a in accs) + " |")
        print("|---|" + "---|" * len(accs))
        for h in hs:
            cells = []
            for a in accs:
                r = next(r for r in out if r["order"] == o and r["head_ms"] == h and r["accept"] == a)
                cells.append(f"{r['tok_s']:.1f}")
            print(f"| {h:g} | " + " | ".join(cells) + " |")
        for a in accs:
            row = sorted((r for r in out if r["order"] == o and r["accept"] == a), key=lambda r: r["head_ms"])
            ok = [r["head_ms"] for r in row if r["tok_s"] >= ref["cont"]]
            print(f"acceptance {a:.2f}: at least the release line's speed up to {max(ok) if ok else 'none'} ms per guess")
    if args.out:
        json.dump({"sessions": len(rows), "tokens": n_tok, "stages": stages, "rows": out}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
