#!/usr/bin/env python3
"""C962 (Q1692): the tables of this folder from run.sh's output (WORK/out and WORK/probe), on any computer.

Usage: python3 -I c962_tables.py <data folder> <run output folder> [--probe <probe folder>] [--probe-inputs <folder>]
  --probe-inputs: write one probe input per position where two sampled lists of the same request part, from the
                  n_probs replies (the candidates and probabilities there), and stop
Writes to the data folder: replies.tsv, lists.tsv, draws.tsv, partings.tsv, logits.tsv, arms.tsv, and probe.tsv with
--probe.

The draws: with --pipe-depth the server keys every sampled token's draw by the position of the token evaluated
(llama_sampler_set_pos: splitmix64 of (seed, position) seeds a std::mt19937, whose first uniform_real_distribution<double>
draw, as a float, is u); reply token t of a prompt of n tokens is drawn at position n - 1 + t. Without the pipe the draws
come one after another from std::mt19937(seed), two 32-bit outputs a draw. The dist step picks the first candidate whose
running sum of probabilities passes u, in the order the candidates come to it.
"""
import glob
import hashlib
import json
import math
import os
import struct
import sys
from itertools import combinations

DATA, OUT = sys.argv[1], sys.argv[2]
PROBE = sys.argv[sys.argv.index("--probe") + 1] if "--probe" in sys.argv else None
PROBE_INPUTS = sys.argv[sys.argv.index("--probe-inputs") + 1] if "--probe-inputs" in sys.argv else None
SEED, TEMP = 7, 0.7
# C950's R14 lists (docs/data/2026-11-21_c950_flint_slot_plan_gpu/tokens.tsv)
C950 = {"fe36bf356f32": "C950 sp0 r1, sp0 r2, base r2", "de544dad17e0": "C950 sp1 r1, sp1 r2, base r1"}


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


def seq_uniforms(seed, n):
    r = MT19937(seed)
    return [r.uniform() for _ in range(n)]


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
    """the tokens some order of the candidates makes the dist step pick: t is pickable when a set S of the others has
    sum(S) <= u < sum(S) + p_t"""
    out = set()
    for i, (tok, p) in enumerate(cands):
        others = [q for j, (_, q) in enumerate(cands) if j != i]
        for r in range(len(others) + 1):
            if any(sum(s) <= u < sum(s) + p for s in combinations(others, r)):
                out.add(tok)
                break
    return out


def md5(toks):
    return hashlib.md5(json.dumps(toks, separators=(",", ":")).encode()).hexdigest()[:12]


def load(path):
    with open(path) as f:
        return json.load(f)


arms = sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d)))
order = ["pipe-a", "pipe-b", "pipe-c", "nopipe-a", "nopipe-b", "p106k", "spipe-a", "spipe-b", "snopipe-a", "spipe-c", "spipe-d"]
arms = [a for a in order if a in arms] + [a for a in arms if a not in order]
pipe = {a: "nopipe" not in a for a in arms}
kern = {a: "p106" if a == "p106k" else "stock" for a in arms}

# every reply: (arm, request, kind, reply)
replies = []
for a in arms:
    o = os.path.join(OUT, a)
    if os.path.exists(os.path.join(o, "s1-R14.json")):
        replies.append((a, "session R14", "session-R14", load(os.path.join(o, "s1-R14.json"))))
    # STEP=session: R14 in the session's context, one variant a round (session_r14.tsv names it)
    if os.path.exists(os.path.join(o, "session_r14.tsv")):
        for ln in open(os.path.join(o, "session_r14.tsv")):
            tag, variant, req = ln.split("\t")[:3]
            if req == "R14":
                replies.append((a, f"session {tag} R14 {variant}", f"session-R14-{variant}",
                                load(os.path.join(o, f"{tag}-R14.json"))))
    for p in sorted(glob.glob(os.path.join(o, "*-*.json"))):
        name = os.path.basename(p)[:-5]
        if name.startswith("s1-") or name[:1] == "v" and name[1:2].isdigit():
            continue
        kind = name.rsplit("-", 1)[0]
        replies.append((a, name, kind, load(p)))

# lists per kind, named A, B, C ... by first appearance; the first token where each parts from A
lists = {}
for a, name, kind, r in replies:
    lists.setdefault(kind, [])
    t = r.get("tokens") or []
    if t not in lists[kind]:
        lists[kind].append(t)


def list_name(kind, toks):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[lists[kind].index(toks)]


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return min(len(x), len(y)) if len(x) != len(y) else -1


with open(os.path.join(DATA, "replies.tsv"), "w") as f:
    f.write("arm\tpipe\trequest\tkind\ttokens\tprompt_tokens\tmd5\tlist\tparts_from_A_at_token\tnote\n")
    for a, name, kind, r in replies:
        t = r.get("tokens") or []
        tm = r.get("timings") or {}
        ln = list_name(kind, t)
        d = first_diff(t, lists[kind][0])
        f.write(f"{a}\t{'yes' if pipe[a] else 'no'}\t{name}\t{kind}\t{len(t)}\t{tm.get('prompt_n', '')}\t{md5(t)}\t{ln}\t"
                f"{'' if ln == 'A' else d}\t{C950.get(md5(t), '')}\n")

with open(os.path.join(DATA, "lists.tsv"), "w") as f:
    f.write("kind\tlist\tmd5\treplies\tarms\tparts_from_A_at_token\n")
    for kind, ls in lists.items():
        for i, t in enumerate(ls):
            who = [(a, name) for a, name, k, r in replies if k == kind and (r.get("tokens") or []) == t]
            f.write(f"{kind}\t{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}\t{md5(t)}\t{len(who)}\t"
                    f"{', '.join(sorted(set(a for a, _ in who)))}\t{'' if i == 0 else first_diff(t, ls[0])}\n")

# the draws of the n_probs replies (post-sampling probabilities: the dist step's own, sorted by the server); the
# context: fresh (R14's prompt read whole) or session (R14 on the cache R1 to R13 built)
PROBS = ("k8probs", "session-R14-probs")
draws = []  # (arm, name, t, pos, u, cands [(tok, p)], sampled, sorted pick, id-order pick, pickable, context)
for a, name, kind, r in replies:
    if kind not in PROBS:
        continue
    probs = r.get("completion_probabilities") or []
    tm = r.get("timings") or {}
    n = tm.get("prompt_n") + tm.get("cache_n")
    us = seq_uniforms(SEED, len(probs)) if not pipe[a] else None
    for t, e in enumerate(probs):
        cands = [(c["id"], c["prob"]) for c in e["top_probs"]]
        pos = n - 1 + t
        u = pos_uniform(SEED, pos) if pipe[a] else us[t]
        srt = sorted(cands, key=lambda c: -c[1])
        byid = sorted(cands, key=lambda c: c[0])
        draws.append((a, name, t, pos, u, cands, e["id"], srt[pick([p for _, p in srt], u)][0],
                      byid[pick([p for _, p in byid], u)][0], pickable(cands, u),
                      "session" if kind.startswith("session") else "fresh"))

with open(os.path.join(DATA, "draws.tsv"), "w") as f:
    f.write("arm\trequest\tcontext\tdraws\ttokens\tpositions_where_the_order_decides\tsampled_not_pickable_by_any_order\t"
            "sampled_eq_descending_probability_pick\tat_order_deciding_positions_sampled_eq_descending\n")
    for a in arms:
        for name in sorted(set(d[1] for d in draws if d[0] == a)):
            ds = [d for d in draws if d[0] == a and d[1] == name]
            dec = [d for d in ds if len(d[9]) > 1]
            f.write(f"{a}\t{name}\t{ds[0][10]}\t{'position-keyed' if pipe[a] else 'sequential'}\t{len(ds)}\t{len(dec)}\t"
                    f"{sum(d[6] not in d[9] for d in ds)}\t{sum(d[6] == d[7] for d in ds)}\t"
                    f"{sum(d[6] == d[7] for d in dec)}\n")

# where two k8probs lists of the same draws part: the candidates there in each, the same or not
partings = []
for draws_kind, ctx in (("position-keyed", "fresh"), ("position-keyed", "session"), ("sequential", "fresh"),
                        ("sequential", "session")):
    runs = {}
    for d in draws:
        if (draws_kind == "position-keyed") == pipe[d[0]] and d[10] == ctx:
            runs.setdefault((d[0], d[1]), []).append(d)
    keys = sorted(runs)
    seen = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = runs[keys[i]], runs[keys[j]]
            k = first_diff([d[6] for d in x], [d[6] for d in y])
            if k < 0 or k >= min(len(x), len(y)):
                continue
            sig = (draws_kind, ctx, k, x[k][6], y[k][6])
            if sig in seen:
                continue
            seen.add(sig)
            partings.append((f"{draws_kind}, {ctx}", keys[i], keys[j], k, x[k], y[k]))

with open(os.path.join(DATA, "partings.tsv"), "w") as f:
    f.write("draws\tkernels\trun_1\trun_2\ttoken\tposition\tu\tcandidates_the_same_bits\tcandidates (id:probability, by probability)\t"
            "sampled_1\tsampled_2\tdescending_probability_pick\tpickable_by_some_order (run 1's candidates)\n")
    for dk, k1, k2, t, d1, d2 in partings:
        same = sorted(d1[5]) == sorted(d2[5])
        cs = " ".join(f"{tok}:{p:.6f}" for tok, p in sorted(d1[5], key=lambda c: -c[1]))
        f.write(f"{dk}\t{kern[k1[0]]}/{kern[k2[0]]}\t{'/'.join(k1)}\t{'/'.join(k2)}\t{t}\t{d1[3]}\t{d1[4]:.9g}\t{'yes' if same else 'no'}\t{cs}\t"
                f"{d1[6]}\t{d2[6]}\t{d1[7]}\t{' '.join(str(x) for x in sorted(d1[9]))}\n")

# the logits between runs: at each token where two replies of a kind share every token before, are the candidates and
# probabilities the same bits?
with open(os.path.join(DATA, "logits.tsv"), "w") as f:
    f.write("kind\tkernels\treplies\tpairs\tpositions_compared\tpositions_with_other_probabilities\tlargest_difference\t"
            "first_such (pair, token)\n")
    for kind in ("k8probs", "session-R14-probs", "greedyprobs", "session-R14-greedy"):
      for kn in ("stock", "p106"):
        rs = [(a, name, r) for a, name, k, r in replies if k == kind and kern[a] == kn]
        if not rs:
            continue
        n_cmp = n_diff = n_pairs = 0
        big = 0.0
        first = ""
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                # the draws must be the same kind for the tokens before to be the same input
                if pipe[rs[i][0]] != pipe[rs[j][0]] and kind in PROBS:
                    continue
                n_pairs += 1
                p1 = rs[i][2].get("completion_probabilities") or []
                p2 = rs[j][2].get("completion_probabilities") or []
                for t in range(min(len(p1), len(p2))):
                    n_cmp += 1
                    c1 = [(c["id"], c.get("prob", c.get("logprob"))) for c in p1[t].get("top_probs", p1[t].get("top_logprobs"))]
                    c2 = [(c["id"], c.get("prob", c.get("logprob"))) for c in p2[t].get("top_probs", p2[t].get("top_logprobs"))]
                    if c1 != c2:
                        n_diff += 1
                        big = max([big] + [abs(x[1] - y[1]) for x, y in zip(c1, c2) if x[0] == y[0]])
                        if not first:
                            first = f"{rs[i][0]}/{rs[i][1]} {rs[j][0]}/{rs[j][1]} token {t}"
                    if p1[t]["id"] != p2[t]["id"]:
                        break
        f.write(f"{kind}\t{kn}\t{len(rs)}\t{n_pairs}\t{n_cmp}\t{n_diff}\t{big:.3g}\t{first}\n")

# the draws checked where the order is known: with top-p 0.95 the top-p step sorts the candidates (descending) before
# the dist step, so each k8p95 token follows from the draw and the probabilities alone. The probabilities: the greedy
# reply's raw top 8 (temperature 1) of the same load, along the greedy list until the k8p95 reply leaves it; top-p keeps
# the fewest of them whose softmax over the 8 reaches 0.95, the temperature 0.7 softmax of those is the dist step's
def topp_predict(top8, u, top_p=0.95, temp=TEMP):
    srt = sorted(top8, key=lambda c: -c[1])
    m = srt[0][1]
    e = [math.exp(lp - m) for _, lp in srt]
    z = sum(e)
    keep, cum = 0, 0.0
    for x in e:
        keep += 1
        cum += x / z
        if cum >= top_p:
            break
    kept = srt[:keep]
    lt = [lp / temp for _, lp in kept]
    mm = max(lt)
    e = [math.exp(x - mm) for x in lt]
    z = sum(e)
    probs = [x / z for x in e]
    return kept[pick(probs, u)][0], [(tok, p) for (tok, _), p in zip(kept, probs)]


with open(os.path.join(DATA, "drawcheck.tsv"), "w") as f:
    f.write("arm\trequest\tdraws\ttokens_checked\tpositions_with_two_or_more_kept\tmispredicted\t"
            "leaves_greedy_at_token\tu_there\tkept_there (id:probability)\ttoken_predicted\ttoken_sampled\n")
    for a in arms:
        gp = os.path.join(OUT, a, "greedyprobs-1.json")
        if not os.path.exists(gp):
            continue
        gr = load(gp)
        tm = gr.get("timings") or {}
        n = tm.get("prompt_n") + tm.get("cache_n")
        us = seq_uniforms(SEED, 256)
        for p in sorted(glob.glob(os.path.join(OUT, a, "k8p95-*.json"))):
            toks = load(p).get("tokens") or []
            checked = multi = bad = 0
            leave = ("", "", "", "", "")
            for t, e in enumerate(gr["completion_probabilities"]):
                u = pos_uniform(SEED, n - 1 + t) if pipe[a] else us[t]
                pred, kept = topp_predict([(c["id"], c["logprob"]) for c in e["top_logprobs"]], u)
                checked += 1
                multi += len(kept) > 1
                bad += pred != toks[t]
                if toks[t] != e["id"]:
                    leave = (t, f"{u:.4f}", " ".join(f"{tok}:{q:.4f}" for tok, q in kept), pred, toks[t])
                    break
            f.write(f"{a}\t{os.path.basename(p)[:-5]}\t{'position-keyed' if pipe[a] else 'sequential'}\t{checked}\t{multi}\t"
                    f"{bad}\t" + "\t".join(str(x) for x in leave) + "\n")

with open(os.path.join(DATA, "arms.tsv"), "w") as f:
    f.write("arm\tpipe_depth\tkernels\tloaded\tpipe_line\ttop_k_rows_line\tn_vocab\tpipelined_lines (guesses/right/wrong, per request in order)\tMiB_per_card\n")
    for a in arms:
        o = os.path.join(OUT, a)
        cmd = open(os.path.join(o, "command.txt")).read()
        log = open(os.path.join(o, "server.log"), errors="replace").read().splitlines()
        pl = next((ln.split("pipelined speculation: ", 1)[1] for ln in log if "pipelined speculation: up to" in ln), "")
        nv = next((ln.split("=")[-1].strip() for ln in log if "print_info: n_vocab" in ln), "")
        tk = sum("one top-k over the output rows" in ln for ln in log)
        gl = []
        for ln in log:
            if "pipelined =" in ln:
                s = ln.split("pipelined =", 1)[1].split(";")[0].split(",")
                gl.append("/".join(x.strip().split()[0] for x in s[:3]))
        mem = open(os.path.join(o, "mem.csv")).read().strip().replace("\n", "; ") if os.path.exists(os.path.join(o, "mem.csv")) else ""
        f.write(f"{a}\t{cmd.split('--pipe-depth ')[1].split()[0]}\t{'P106 kernels forced' if 'MMVQ_PASCAL=2' in cmd else 'stock'}\t"
                f"{open(os.path.join(o, 'load.txt')).read().strip()}\t{pl}\t{tk}\t{nv}\t{' '.join(gl)}\t{mem}\n")

if PROBE_INPUTS:
    os.makedirs(PROBE_INPUTS, exist_ok=True)
    nv = None
    for a in arms:
        for ln in open(os.path.join(OUT, a, "server.log"), errors="replace"):
            if "print_info: n_vocab" in ln:
                nv = int(ln.split("=")[-1])
                break
        if nv:
            break
    # token 24 of R14 (position 2312) in both contexts, from the first stock pipelined load that has it, and every
    # position where two lists of the same draws and context part
    picks = []
    for ctx in ("fresh", "session"):
        d = next((d for d in draws if d[10] == ctx and d[2] == 23 and pipe[d[0]] and d[0] != "p106k"), None)
        if d:
            picks.append((f"{ctx}-pos{d[3]}", d))
    for dk, k1, k2, t, d1, d2 in partings:
        if dk.startswith("position-keyed"):
            picks.append((f"{d1[10]}-{d1[0]}-pos{d1[3]}", d1))  # e.g. fresh-p106k-pos2399
    for label, d1 in picks:
        fn = os.path.join(PROBE_INPUTS, f"{label}.in")
        with open(fn, "w") as f:
            f.write(f"{nv} {len(d1[5])} {SEED} {d1[3]} {TEMP}\n")
            for tok, p in sorted(d1[5], key=lambda c: -c[1]):
                f.write(f"{tok} {p!r}\n")
        print("wrote", fn)

if PROBE:
    with open(os.path.join(DATA, "probe.tsv"), "w") as f:
        f.write("input\tmode\tcard\tdevice\tu\tcalls\tdistinct_orders\torders (calls: order -> token picked)\tpicked (token: calls)\t"
                "descending_probability_pick\tascending_id_pick\n")
        for p in sorted(glob.glob(os.path.join(PROBE, "*.gpu*.txt"))):
            b = os.path.basename(p)[:-4]
            inp, card = b.rsplit(".gpu", 1)
            inp, mode = inp.rsplit(".", 1)
            lines = open(p).read().splitlines()
            dev = next((ln.split(": ", 1)[1].split(",")[0] for ln in lines if ln.startswith("device ")), "")
            u = next((ln.split("draw u = ")[1] for ln in lines if "draw u = " in ln), "")
            calls = next((ln for ln in lines if "distinct orders" in ln), "")
            ords = [ln.split("\t") for ln in lines if ln.startswith("order\t")]
            pk = [ln.split("\t") for ln in lines if ln.startswith("picked\t")]
            dp = next((ln.split("picks ")[1] for ln in lines if ln.startswith("descending-probability order")), "")
            ip = next((ln.split("picks ")[1] for ln in lines if ln.startswith("ascending-id order")), "")
            f.write(f"{inp}\t{mode}\t{card}\t{dev}\t{u.split(';')[0]}\t{calls.split(' calls')[0]}\t{calls.split(': ')[-1].split()[0] if calls else ''}\t"
                    f"{'; '.join(f'{o[1]}: {o[2]} -> {o[4]}' for o in ords[:12])}{' ...' if len(ords) > 12 else ''}\t"
                    f"{' '.join(f'{x[1]}: {x[2]}' for x in pk)}\t{dp}\t{ip}\n")
print("tables written to", DATA)
