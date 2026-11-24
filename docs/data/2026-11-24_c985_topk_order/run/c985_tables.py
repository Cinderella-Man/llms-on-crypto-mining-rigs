#!/usr/bin/env python3
"""C985: the tables of this folder from run.sh's output.

Usage: python3 -I c985_tables.py <this folder> <W, run.sh's work dir>
Reads W/units/ (test-top-k-sort, test-backend-ops, test-backend-sampler) and W/session/<arm>/session_r14.tsv, writes:
  topk_sort.tsv     test-top-k-sort, every case per switch, mode and device
  ops.tsv           test-backend-ops -o TOP_K per switch: cases, passed, failed, its summary line
  sampler.tsv       test-backend-sampler, each test's exit code per switch
  rounds.tsv        the session: every round of every load, R14's list, and whether the requests before R14 (greedy) gave
                    the same replies as in every other round
  lists.tsv         the session per switch and R14 variant: rounds, loads, the lists and how many rounds gave each
  draws.tsv         R14's rounds with n_probs and post_sampling_probs (variant "probs": the pipe off, the same draws):
                    per round, the tokens, the positions where the order of the candidates decides the token, and how
                    many tokens are the pick of the candidates in descending order of probability (the order the switch
                    gives), with the draws the fork keys by seed and position (as C962's c962_tables.py)
  probs_bits.tsv    the same rounds compared in pairs per switch, along their shared tokens: positions whose dist-step
                    probabilities differ in any bit
The draws: reply token t of a prompt of n tokens is drawn at position n - 1 + t: splitmix64 of (seed, position) seeds a
std::mt19937, whose first uniform_real_distribution<double> draw, as a float, is u. The dist step picks the first
candidate whose running sum of probabilities passes u, in the order the candidates come to it.
"""
import collections
import csv
import glob
import hashlib
import json
import os
import re
import struct
import sys
from itertools import combinations

OUT, W = sys.argv[1], sys.argv[2]
SEED = 7  # C950's R14
# C962's names for C950's two R14 replies (the first 12 hex digits of the md5 of the token ids)
KNOWN = {"fe36bf356f32": "C950 first (greedy)", "de544dad17e0": "C950 second"}


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def topk():
    rows = []
    for sw in ("off", "on"):
        for mode in ("plain", "rewrite"):
            p = os.path.join(W, "units", f"topk-{sw}-{mode}.txt")
            if not os.path.exists(p):
                continue
            hdr = None
            for line in open(p, errors="replace"):
                f = line.rstrip("\n").split("\t")
                if f[0] == "case":
                    hdr = f
                elif hdr and len(f) == len(hdr) and not line.startswith("#"):
                    rows.append([sw, mode] + f)
    write("topk_sort.tsv", ["switch", "mode", "case", "device", "cols", "rows", "k", "calls", "orders",
                            "reference_order_calls", "set_errors", "u", "picks"], rows)


def ops():
    rows = []
    for sw in ("off", "on"):
        p = os.path.join(W, "units", f"ops-{sw}.txt")
        if not os.path.exists(p):
            continue
        txt = open(p, errors="replace").read()
        cases = re.findall(r"^\s*TOP_K\(.*?\):.*?(OK|FAIL|not supported)", txt, re.M)
        c = collections.Counter(cases)
        summ = re.findall(r"^\s*(\d+/\d+ tests passed)", txt, re.M)
        rows.append([sw, len(cases), c.get("OK", 0), c.get("FAIL", 0), c.get("not supported", 0),
                     summ[-1] if summ else "-"])
    write("ops.tsv", ["switch", "top_k_cases", "ok", "fail", "not_supported", "summary"], rows)


def sampler():
    res = {}
    for sw in ("off", "on"):
        p = os.path.join(W, "units", f"sampler-{sw}.tsv")
        if not os.path.exists(p):
            continue
        for line in open(p):
            t, rc = line.rstrip("\n").split("\t")
            res.setdefault(t, {})[sw] = rc
    write("sampler.tsv", ["test", "exit_off", "exit_on"],
          [[t, v.get("off", "-"), v.get("on", "-")] for t, v in res.items()])


def session():
    rounds = []
    for d in sorted(glob.glob(os.path.join(W, "session", "*", "session_r14.tsv"))):
        arm = os.path.basename(os.path.dirname(d))
        sw = arm.split("-")[0]
        by_tag = collections.OrderedDict()
        for line in open(d):
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            by_tag.setdefault(f[0], []).append(f)
        for tag, reqs in by_tag.items():
            variant = reqs[0][1]
            r14 = [r for r in reqs if r[2] == "R14"]
            r3 = [r for r in reqs if r[2] == "R3"]
            greedy = [r for r in reqs if r[2] not in ("R14", "R3")]
            # the md5s of the greedy requests' token ids in order, hashed together: one id for the session's greedy turns
            greedy_id = hashlib.md5("|".join(r[2] + ":" + r[6] for r in greedy).encode()).hexdigest()[:12]
            m = r14[0][6] if r14 else "-"
            rounds.append([arm, sw, tag, variant, len(greedy), greedy_id, r3[0][6] if r3 else "-",
                           r14[0][3] if r14 else "-", m, KNOWN.get(m, "")])
    # the greedy turns' replies: the same in every round of both switches?
    common = collections.Counter(r[5] for r in rounds).most_common(1)
    common_id = common[0][0] if common else ""
    rows = [r[:6] + ["yes" if r[5] == common_id else "NO"] + r[6:] for r in rounds]
    write("rounds.tsv", ["arm", "switch", "round", "r14_variant", "greedy_requests", "greedy_id",
                         "greedy_as_every_round", "r3_md5", "r14_tokens", "r14_md5", "r14_known_as"], rows)
    lists = collections.OrderedDict()
    for r in rounds:
        for k, m in (((r[1], "R14 " + r[3]), r[8]), ((r[1], "R3"), r[6])):
            e = lists.setdefault(k, {"rounds": 0, "loads": set(), "lists": collections.Counter()})
            e["rounds"] += 1
            e["loads"].add(r[0])
            e["lists"][m + (f" ({KNOWN[m]})" if m in KNOWN else "")] += 1
    out = []
    for (sw, v), e in sorted(lists.items(), key=lambda x: (x[0][1] == "R3", x[0])):
        out.append([sw, v, e["rounds"], len(e["loads"]), len(e["lists"]),
                    "; ".join(f"{k}: {n}" for k, n in e["lists"].most_common())])
    write("lists.tsv", ["switch", "request", "rounds", "loads", "distinct_lists", "lists"], out)


# --- the draws, from C962's c962_tables.py (docs/data/2026-11-23_c962_sampled_reply_gpu/run/) ---
class MT19937:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.i = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for k in range(1, 624):
            self.mt[k] = (1812433253 * (self.mt[k - 1] ^ (self.mt[k - 1] >> 30)) + k) & 0xFFFFFFFF

    def next(self):
        if self.i >= 624:
            for k in range(624):
                y = (self.mt[k] & 0x80000000) | (self.mt[(k + 1) % 624] & 0x7FFFFFFF)
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self.i = 0
        y = self.mt[self.i]
        self.i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def uniform(self):  # libstdc++ generate_canonical<double, 53> on a 32-bit engine, then (float)
        a = float(self.next())
        b = float(self.next())
        r = (a + b * 4294967296.0) / 18446744073709551616.0
        if r >= 1.0:
            r = 1.0 - 2.0 ** -53
        return struct.unpack("f", struct.pack("f", r))[0]


def pos_seed(seed, pos):
    m = 0xFFFFFFFFFFFFFFFF
    x = ((seed << 32) ^ (pos & 0xFFFFFFFF)) & m
    x = (x + 0x9E3779B97F4A7C15) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    x ^= x >> 31
    return (x ^ (x >> 32)) & 0xFFFFFFFF


def pos_uniform(seed, pos):
    return MT19937(pos_seed(seed, pos)).uniform()


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def pick(probs, u):
    """index the dist step picks over probabilities in this order: the first whose running sum (float) passes u"""
    cum, above = 0.0, 0
    for p in probs:
        cum = f32(cum + p)
        above += (cum - u) > 0
    above = min(max(above, 1), len(probs))
    return len(probs) - above


def pickable(cands, u):
    """the tokens some order of the candidates makes the dist step pick"""
    out = set()
    for i, (tok, p) in enumerate(cands):
        others = [q for j, (_, q) in enumerate(cands) if j != i]
        for r in range(len(others) + 1):
            if any(sum(x) <= u < sum(x) + p for x in combinations(others, r)):
                out.add(tok)
                break
    return out


def probs_rounds():
    rounds = []  # (arm, switch, tag, reply)
    for d in sorted(glob.glob(os.path.join(W, "session", "*", "session_r14.tsv"))):
        arm = os.path.basename(os.path.dirname(d))
        for line in open(d):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[1] == "probs" and f[2] == "R14":
                p = os.path.join(os.path.dirname(d), f"{f[0]}-R14.json")
                if os.path.exists(p):
                    rounds.append((arm, arm.split("-")[0], f[0], json.load(open(p))))
    rows = []
    for arm, sw, tag, r in rounds:
        probs = r.get("completion_probabilities") or []
        tm = r.get("timings") or {}
        n = tm.get("prompt_n", 0) + tm.get("cache_n", 0)
        dec = eq = eq_dec = 0
        for t, e in enumerate(probs):
            cands = [(c["id"], c["prob"]) for c in e["top_probs"]]
            u = pos_uniform(SEED, n - 1 + t)
            srt = sorted(cands, key=lambda c: (-c[1], c[0]))
            sp = srt[pick([q for _, q in srt], u)][0]
            deciding = len(pickable(cands, u)) > 1
            dec += deciding
            eq += e["id"] == sp
            eq_dec += deciding and e["id"] == sp
        toks = r.get("tokens") or []
        rows.append([arm, sw, tag, n, len(probs), dec, eq, eq_dec,
                     hashlib.md5(json.dumps(toks, separators=(",", ":")).encode()).hexdigest()[:12]])
    write("draws.tsv", ["arm", "switch", "round", "context_tokens", "tokens", "positions_where_the_order_decides",
                        "tokens_eq_descending_pick", "at_deciding_positions_eq_descending", "md5"], rows)
    out = []
    for i in range(len(rounds)):
        for j in range(i + 1, len(rounds)):
            a, b = rounds[i], rounds[j]
            if a[1] != b[1]:
                continue
            pa = a[3].get("completion_probabilities") or []
            pb = b[3].get("completion_probabilities") or []
            shared = diff = 0
            big = 0.0
            for x, y in zip(pa, pb):
                if x["id"] != y["id"]:
                    break
                shared += 1
                cx = sorted((c["id"], c["prob"]) for c in x["top_probs"])
                cy = sorted((c["id"], c["prob"]) for c in y["top_probs"])
                if cx != cy:
                    diff += 1
                    if [c[0] for c in cx] == [c[0] for c in cy]:
                        big = max(big, max(abs(p - q) for (_, p), (_, q) in zip(cx, cy)))
            out.append([a[1], f"{a[0]} {a[2]}", f"{b[0]} {b[2]}", shared, diff, f"{big:.3g}"])
    write("probs_bits.tsv", ["switch", "round_1", "round_2", "positions_compared", "positions_with_other_probabilities",
                             "largest_difference"], out)


topk()
ops()
sampler()
session()
probs_rounds()
