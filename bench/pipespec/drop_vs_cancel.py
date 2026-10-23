#!/usr/bin/env python3
"""Event model of pipelined speculation on a layer split (one sequence).

Stages are cards with fixed per-token times s[j] (hop included); each runs its queue in FIFO order. A token enters
stage 0's queue when the host submits it and stage j's queue when it leaves stage j-1. The host keeps the newest
committed token plus up to D guesses in flight; when the oldest token leaves the last stage the host samples
(host_ms) and compares with the next guess (hit with probability p, i.i.d.).
Policies for the tokens behind a wrong guess:
  drop   - they stay queued and run on every card (llama_pipe_* in the fork: CUDA work cannot be withdrawn)
  cancel - a card skips them if it would start them after the miss is known (ideal per-stage withdrawal)
Prints the table in docs/research/notes/pipelined_speculation.md ("Dropping instead of withdrawing"): the stage times
of bench/ngram_replay's card setups with 0.3 ms per hop and 1.5 ms of host time per sample.
Usage: drop_vs_cancel.py [host ms per sample]
"""
import heapq, random, sys


def run(s, D, p, n_tokens=20000, policy="drop", host_ms=0.0, submit_ms=0.0, seed=1):
    S = len(s)
    rng = random.Random(seed)
    queues = [[] for _ in range(S)]   # FIFO of token ids
    busy = [False] * S
    ev = []                           # (time, seq, kind, data)
    seq = [0]

    def push(t, kind, data):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, data))

    tokens = {}                       # id -> dict(valid, inval_t)
    nid = [0]
    win = []                          # ids: [committed in flight, guesses...]
    host_t = [0.0]

    def submit(t):
        nid[0] += 1
        i = nid[0]
        tokens[i] = {"inval": None}
        win.append(i)
        queues[0].append(i)
        push(t, "kick", 0)
        return i

    def kick(j, t):
        while not busy[j] and queues[j]:
            i = queues[j].pop(0)
            inv = tokens[i]["inval"]
            if policy == "cancel" and inv is not None and inv <= t:
                continue  # skipped
            busy[j] = True
            push(t + s[j], "done", (j, i))

    produced = 0
    t = 0.0
    submit(0.0)
    while len(win) < D + 1:
        submit(0.0)
    while produced < n_tokens:
        t, _, kind, data = heapq.heappop(ev)
        if kind == "kick":
            kick(data, t)
            continue
        j, i = data
        busy[j] = False
        kick(j, t)
        if j + 1 < S:
            queues[j + 1].append(i)
            kick(j + 1, t)
            continue
        # token i left the last stage
        if not win or win[0] != i:
            continue  # a dropped token finishing
        win.pop(0)
        produced += 1
        th = t + host_ms
        hit = win and rng.random() < p
        if not hit:
            for w in win:
                tokens[w]["inval"] = th
            win.clear()
            submit(th)
        k = 0
        while len(win) < D + 1:
            k += 1
            submit(th + k * submit_ms)
    return n_tokens / t


def plain(s, host_ms=0.0):
    return 1.0 / (sum(s) + host_ms)


if __name__ == "__main__":
    host = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    hop = 0.3
    setups = {
        "6 cards (9/8/8/8/8/8)": [3.91 + hop] + [3.48 + hop] * 4 + [5.98],
        "9 cards (6/6/6/6/5/5/5/5/5)": [3.05 + hop] * 4 + [2.54 + hop] * 4 + [5.04],
        "9 cards (6 x 7/5/2)": [3.05 + hop] * 7 + [2.54 + hop] + [3.52],
    }
    print("setup | S_eff | D | every guess right | p = 0.7 dropped (cancelled) | every guess wrong")
    for name, s in setups.items():
        base = plain(s, host)
        for D in range(1, len(s)):
            right = run(s, D, 1.0, 3000, "drop", host) / base
            p7 = run(s, D, 0.7, 3000, "drop", host) / base
            p7c = run(s, D, 0.7, 3000, "cancel", host) / base
            wrong = run(s, D, 0.0, 3000, "drop", host) / base
            print(f"{name} | {sum(s) / max(s):.1f} | {D} | {right:.2f}x | {p7:.2f}x ({p7c:.2f}x) | {wrong:.2f}x")
