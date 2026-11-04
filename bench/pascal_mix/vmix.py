#!/usr/bin/env python3
"""Q901 (C339): a verify-shaped Pascal mix of Qwen3.8-Flash-Next. pmix.py chose each dense tensor's type by what one
decoded token reads; with a draft, the rig spends its time in k-token verify passes, where the P106's kernels rank the
types differently. This picks each dense tensor's type to minimise the expected time of a pass,
    E[T] = sum_k P(k) * T(type, shape, k),
under two budgets against the start (pmix-q2-fast): bytes on the cards (no more than the start's, by default) and a
predicted KLD change (no more than zero, by default).

  vmix.py times [--perf TSV ...] [--ggml-est TSV ...] [--bench CSV ...] [--dispatch min|on|off] [--basis bench|ggml]
        one table of µs per product: type, K (row width), M (rows), n (tokens), us, source. --bench reads
        bench/mmvq_pascal's mmvq_kcol CSV (the activation quantizer and the product, every launch shape; the fastest
        variant counts, never `prod`, whose records are quantized beforehand); --perf reads test-backend-ops perf
        tables (docs/data/2026-11-04_kcurve_p106's perf_dense.tsv, perf_cov.tsv: the product inside ggml, the k-column
        kernels off and on; --dispatch picks which runs: min = the faster, as the -kshape build's dispatch table
        intends; on = the k-column kernels always, -ladder3's line); --ggml-est reads docs/data/2026-11-04_kshape's
        ggml_est.tsv (those products at Q823's launch shapes, estimated). A later file of a kind overrides an earlier
        one. The two bases differ (ggml / bench at one token is ~0.74, at 2-8 ~1.03-1.20, per product 0.4-2.0), so
        types are compared on one: with --basis bench (the default; C323 times the missing types in the same bench)
        a bench time wins, and a perf or ggml-est time of a product the bench lacks is divided by the median ratio
        ggml / bench at its token count over the products timed both ways; --basis ggml the reverse.
  vmix.py pk --metrics BEFORE AFTER                                        P(k) of the verify passes from llama.cpp's
        /metrics counters of an agent-check arm (llama-server, a fixed --spec-draft-n-max): k = draft tokens / drafts
        + 1 for the draft passes, k = 1 for the tokens decoded outside them
  vmix.py select --start START.json --published PUB.json --times TSV --qerr DIR --out DIR
        [--pk 4=0.989,1=0.011] [--gain G | --calib TSV] [--byte-slack B] [--kld-slack D] [--classes ...]
        [--types IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0,Q4_0,Q5_0] [--loose] [--objective time|kld] [--ms-slack MS]
        [--name NAME]
        START is the start's header dump (bench/file_choice/dump.py), PUB the published file it was made from (the
        build's base, so take.txt and types.txt are relative to it, as pmix.py writes them for run/build.sh).
        Quality: a tensor's noise is qerr.py's importance-weighted squared error against the Q8_0 reference
        (measured where the tensor was sampled in that type, else its fitted level a(t) times the type's factor
        c(type); a tensor not sampled takes its class's median a(t) and power per weight, a class not sampled the
        median over all classes), and the predicted KLD change of a choice is G(class) times the change in noise
        (Q8_0, the reference's own type, has none). G comes from --calib (a TSV of class, gain: what
        `calib` prints; a class without its own row takes the `all` row) or one --gain for every class (`calib` with
        pmix-q2-spine6 as the arm gives it: 0.0233 of agent KLD for spine6's change in noise). The shared expert's
        gate and up of a layer take one type (one fused launch), and a layer whose pair has the routed experts' type
        keeps it. A type with no time at a tensor's shape (see Times) is not offered there; --loose allows times
        scaled by bytes from another shape of the type.
        --objective time (the default) gives the least E[T] within the bytes and the KLD budget (--kld-slack);
        --objective kld the least predicted KLD within the bytes and an E[T] budget (--ms-slack ms of E[T] over the
        start's). Each is solved with a multiplier per budget over a grid, then single-tensor moves that keep both.
        Writes, like pmix.py: plan.tsv, types.txt, take.txt, retype.txt, summary.tsv (E[T] per k and in all, bytes,
        predicted KLD, against the start) and NAME.json (the planned header dump).
  vmix.py calib --start START.json --qerr DIR --kld TSV --arm NAME=DUMP.json ... [--text agent] [--base pmix-q2-fast]
        a gain per class: each arm (the start with one class changed, e.g. C326's single-class spine files) divides
        its KLD change on the text by its change in noise; `all` is the pooled gain of every arm
Standard library only.
"""
import argparse, collections, csv, json, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmix import BLOCK, bits, nbytes, tclass  # noqa: E402

KINDS = {"bench": 0, "ggml-est": 1, "perf": 2}
FAMILY = {"Q4_0": "legacy", "Q5_0": "legacy", "Q8_0": "legacy", "IQ4_NL": "legacy", "Q2_0": "legacy",
          "Q4_K": "k", "Q5_K": "k", "Q6_K": "k", "Q3_K": "k", "IQ4_XS": "k"}
FAMILY_REF = {"legacy": ("Q5_0", "Q4_0", "Q8_0"), "k": ("IQ4_XS", "Q4_K", "Q6_K", "Q5_K")}
MiB = 2 ** 20


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, header, rows):
    f = sys.stdout if path == "-" else open(path, "w")
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
    if f is not sys.stdout:
        f.close()


# ---- times -------------------------------------------------------------------------------------------------------

def cmd_times(a):
    raw = {kind: {} for kind in KINDS}  # kind -> (type, K, M, n) -> (us, source); a later file overrides

    for path in a.bench:
        best = {}  # the fastest launch shape of each product in this file
        for r in csv.DictReader(open(path)):
            if not r.get("type") or r.get("check", "PASS") != "PASS" or r["variant"] == "prod":
                continue  # `prod` is the product with its records quantized beforehand: not a launch the model makes
            key = (r["type"].upper(), int(r["K"]), int(r["N"]), int(r["tokens"]))
            us = float(r["us_median"])
            if key not in best or us < best[key][0]:
                best[key] = (us, f"bench {os.path.basename(path)} {r['variant']}")
        raw["bench"].update(best)
    for path in a.ggml_est:
        for r in read_tsv(path):
            if not r["us_on_new_est"] or not r["us_off"]:
                continue
            on, off = float(r["us_on_new_est"]), float(r["us_off"])
            raw["ggml-est"][(r["type"].upper(), int(r["K"]), int(r["rows"]), int(r["k"]))] = (
                {"min": min(on, off), "on": on, "off": off}[a.dispatch], f"ggml-est {os.path.basename(path)}")
    for path in a.perf:
        for r in read_tsv(path):
            if r.get("kind") not in ("kcol", "bf16", "mtok", None):
                continue
            on, off = float(r["us_on"]), float(r["us_off"])
            raw["perf"][(r["type"].upper(), int(r["k"]), int(r["m"]), int(r["n"]))] = (
                {"min": min(on, off), "on": on, "off": off}[a.dispatch], f"perf {os.path.basename(path)}")
    # the two bases: the bench times a kernel alone (with its quantizer), test-backend-ops the product inside ggml.
    # Per token count, the median ratio ggml / bench over the products timed both ways converts one into the other
    ratio = collections.defaultdict(list)
    for k, (us, _s) in raw["bench"].items():
        if k in raw["perf"]:
            ratio[k[3]].append(raw["perf"][k][0] / us)
    r_n = {n: statistics.median(v) for n, v in ratio.items()}
    for n in sorted(r_n):
        print(f"# ggml / bench at {n} tokens: median {r_n[n]:.3f} over {len(ratio[n])} products "
              f"({min(ratio[n]):.2f}-{max(ratio[n]):.2f})", file=sys.stderr)
    order = ("bench", "perf", "ggml-est") if a.basis == "bench" else ("perf", "ggml-est", "bench")
    t = {}
    for kind in order:  # the first kind that has a product gives its time
        own = (kind == "bench") == (a.basis == "bench")
        for k, (us, src) in raw[kind].items():
            if k in t:
                continue
            if own:
                t[k] = (us, src)
            elif k[3] in r_n:
                f = 1 / r_n[k[3]] if a.basis == "bench" else r_n[k[3]]
                t[k] = (us * f, f"{src} x {f:.3f} ({a.basis} basis)")
    write_tsv("-", ("type", "K", "M", "n", "us", "source"), [(*k, f"{v[0]:.2f}", v[1]) for k, v in sorted(t.items())])


class Times:
    """µs of one product of a type at a shape (K inputs, M rows) and token count n. In order: measured; scaled by M
    from the same type, K and n, at most 4x more or fewer rows (the row width sets a kernel's speed per row, the
    rows the grid); the type's
    one-token time at that shape times a family member's growth from 1 to n tokens at the same K (else at any K);
    with loose, scaled by bytes from the same type's nearest shape at n. A type with none of these has no time at
    that shape, and select leaves it out there (C323 times Q8_0, Q5_0, Q4_0 and Q6_K at Flash-Next's shapes)."""

    def __init__(self, path, loose=False):
        self.m, self.loose = {}, loose
        for r in read_tsv(path):
            self.m[(r["type"], int(r["K"]), int(r["M"]), int(r["n"]))] = (float(r["us"]), r["source"])

    def get(self, typ, K, M, n):
        if (typ, K, M, n) in self.m:
            return self.m[(typ, K, M, n)][0], "measured"
        same_k = [(abs(math.log(m2 / M)), us * M / m2, m2) for (t2, k2, m2, n2), (us, _s) in self.m.items()
                  if t2 == typ and k2 == K and n2 == n and max(m2, M) <= 4 * min(m2, M)]
        if same_k:
            _d, us, m2 = min(same_k)
            return us, f"scaled from {K}x{m2}"
        if n != 1:
            one, src = self.get(typ, K, M, 1)
            if one is not None:
                for ref in FAMILY_REF[FAMILY.get(typ, "k")]:
                    g = self.growth(ref, n, K)
                    if g:
                        return one * g, f"{src}, n = {n} by {ref}'s growth"
        if self.loose:
            b = nbytes(typ, [K, M])
            near = [(abs(math.log(nbytes(typ, [k2, m2]) / b)), us / nbytes(typ, [k2, m2]), (k2, m2))
                    for (t2, k2, m2, n2), (us, _s) in self.m.items() if t2 == typ and n2 == n]
            if near:
                _d, per, shape = min(near)
                return per * b, f"scaled by bytes from {shape[0]}x{shape[1]}"
        return None, "none"

    def growth(self, typ, n, K):
        for same_k in (True, False):
            r = [us / self.m[(t2, k2, m2, 1)][0] for (t2, k2, m2, n2), (us, _s) in self.m.items()
                 if t2 == typ and n2 == n and (t2, k2, m2, 1) in self.m and (k2 == K or not same_k)]
            if r:
                return statistics.median(r)
        return None


# ---- P(k) ----------------------------------------------------------------------------------------------------------

def metrics(path):
    out = {}
    for line in open(path):
        m = re.match(r"^llamacpp:(\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m and not m.group(2):
            out[m.group(1)] = float(m.group(3))
    return out


def cmd_pk(a):
    b, e = metrics(a.metrics[0]), metrics(a.metrics[1])
    d = {k: e.get(k, 0) - b.get(k, 0) for k in e}
    drafts, dtok = d["spec_decode_num_drafts_total"], d["spec_decode_num_draft_tokens_total"]
    acc, pred = d["spec_decode_num_accepted_tokens_total"], d["tokens_predicted_total"]
    k = round(dtok / drafts) + 1
    plain = max(pred - acc - drafts, 0)
    tot = drafts + plain
    print(f"{k}={drafts / tot:.4f},1={plain / tot:.4f}")
    print(f"# {drafts:.0f} draft passes of {dtok / drafts:.3f} drafted tokens, acceptance {acc / dtok:.4f}, "
          f"{plain:.0f} tokens outside them; {pred:.0f} predicted", file=sys.stderr)


def parse_pk(s):
    pk = {int(k): float(v) for k, v in (x.split("=") for x in s.split(","))}
    tot = sum(pk.values())
    return {k: v / tot for k, v in pk.items()}


# ---- quality --------------------------------------------------------------------------------------------------------

class Quality:
    """Importance-weighted noise of a tensor in a type, from qerr.py's tables."""

    def __init__(self, qdir):
        self.meas = {}  # (tensor, type) -> noise
        self.power, self.cls, self.nw = {}, {}, {}
        for r in read_tsv(os.path.join(qdir, "qerr.tsv")):
            n = r["tensor"]
            self.meas[(n, r["type"])] = float(r["noise"])
            self.power[n], self.cls[n], self.nw[n] = float(r["power"]), r["class"], int(r["weights"])
        self.c = {r["type"]: math.log2(float(r["factor_vs_iq4_xs"])) for r in read_tsv(os.path.join(qdir, "types.tsv"))}
        lv = collections.defaultdict(list)
        for (n, t), v in self.meas.items():
            if t in self.c and v > 0:
                lv[n].append(math.log2(v / self.power[n]) - self.c[t])
        self.a = {n: sum(v) / len(v) for n, v in lv.items()}
        by = collections.defaultdict(list)
        for n in self.a:
            by[self.cls[n]].append((self.a[n], self.power[n] / self.nw[n]))
        self.class_a = {c: statistics.median(x[0] for x in v) for c, v in by.items()}
        self.class_ppw = {c: statistics.median(x[1] for x in v) for c, v in by.items()}
        pooled = [x for v in by.values() for x in v]  # a class not sampled at all
        self.class_a = collections.defaultdict(lambda: statistics.median(x[0] for x in pooled), self.class_a)
        self.class_ppw = collections.defaultdict(lambda: statistics.median(x[1] for x in pooled), self.class_ppw)

    def noise(self, name, cls, typ, nweights):
        if typ in ("Q8_0", "BF16", "F16", "F32"):
            return 0.0
        if (name, typ) in self.meas:
            return self.meas[(name, typ)]
        if typ not in self.c:
            raise SystemExit(f"qerr has no factor for type {typ}")
        if name in self.a:
            return self.power[name] * 2 ** (self.a[name] + self.c[typ])
        return self.class_ppw[cls] * nweights * 2 ** (self.class_a[cls] + self.c[typ])


def qclass(name):
    if name == "output.weight":
        return "output"
    m = re.search(r"\.(attn_(?:q|k|v)|attn_output|attn_qkv|attn_gate|ssm_out|ple_key|ffn_(?:gate|up|down)_shexp)\.", name)
    if not m:
        return None
    return {"attn_q": "fa_qkv", "attn_k": "fa_qkv", "attn_v": "fa_qkv", "attn_output": "fa_out", "attn_qkv": "gdn_qkv",
            "attn_gate": "gdn_gate", "ssm_out": "gdn_out", "ple_key": "ple"}.get(m.group(1), "shexp")


def load_dump(path):
    d = json.load(open(path))
    return d, {n: (t, dims, size) for n, t, dims, size, _sh in d["tensors"]}


def cmd_calib(a):
    q = Quality(a.qerr)
    _sd, start = load_dump(a.start)
    k = {(r["name"], r["text"]): float(r["kld"]) for r in read_tsv(a.kld)}
    ref = k[(a.base, a.text)]
    rows, dn_all, dk_all = [], 0.0, 0.0
    for item in a.arm:
        name, path = item.split("=", 1)
        _d, arm = load_dump(path)
        per = collections.Counter()
        for n, (t0, dims, _s) in start.items():
            t1 = arm[n][0]
            c = qclass(n)
            if t1 == t0 or c is None:
                continue
            nw = math.prod(dims)
            per[c] += q.noise(n, c, t1, nw) - q.noise(n, c, t0, nw)
        dn = sum(per.values())
        if (name, a.text) not in k or dn == 0:
            continue
        dk = k[(name, a.text)] - ref
        dn_all += dn
        dk_all += dk
        rows.append((name, "+".join(sorted(per)), f"{dk:.5f}", f"{dn:.6g}", f"{dk / dn:.6g}"))
    rows.append(("all", "all", f"{dk_all:.5f}", f"{dn_all:.6g}", f"{dk_all / dn_all:.6g}" if dn_all else ""))
    write_tsv("-", ("arm", "class", "dkld", "dnoise", "gain"), rows)


# ---- select ---------------------------------------------------------------------------------------------------------

def cmd_select(a):
    pk = parse_pk(a.pk)
    times = Times(a.times, a.loose)
    q = Quality(a.qerr)
    sd, start = load_dump(a.start)
    _pd, pub = load_dump(a.published)
    gain = collections.defaultdict(lambda: a.gain)
    if a.calib:
        for r in read_tsv(a.calib):
            if r["gain"]:
                gain[r["class"]] = float(r["gain"])
        if "all" in gain:
            gain.default_factory = lambda: gain["all"]
    cand = a.types.split(",")
    classes = set(a.classes.split(","))
    # decision units: one tensor, or a layer's shared gate + up (one type)
    units, fixed = [], []
    names = [n for n, _t, _d, _s, _sh in sd["tensors"]]
    pairs = {}
    for n in names:
        m = re.match(r"(blk\.\d+)\.ffn_(gate|up)_shexp\.weight$", n)
        if m:
            pairs.setdefault(m[1], []).append(n)
    done = set()
    for n in names:
        t0, dims, size = start[n]
        c = qclass(n)
        if c is None or c not in classes or len(dims) < 2 or n in done:
            continue
        m = re.match(r"(blk\.\d+)\.ffn_(gate|up)_shexp\.weight$", n)
        members = pairs[m[1]] if m else [n]
        done.update(members)
        if m and all(start[x][0] == start.get(f"{m[1]}.ffn_gate_exps.weight", ("",))[0] for x in members):
            fixed.extend(members)  # fused with the routed experts: kept
            continue
        opts = []
        for typ in dict.fromkeys([t0] + cand):
            k_, _b = BLOCK.get(typ, (0, 0))
            if not k_ or dims[0] % k_:
                continue
            tot_us, src, by_k, ok = 0.0, set(), {}, True
            for x in members:
                for kk, p in pk.items():
                    us, s = times.get(typ, start[x][1][0], start[x][1][1], kk)
                    if us is None:
                        ok = False
                        break
                    by_k[kk] = by_k.get(kk, 0.0) + us
                    tot_us += p * us
                    src.add(s)
            if not ok:
                continue
            by = sum((nbytes(typ, start[x][1]) + 31) // 32 * 32 if typ != start[x][0] else start[x][2] for x in members)
            noise = sum(q.noise(x, c, typ, math.prod(start[x][1])) for x in members)
            opts.append(dict(type=typ, us=tot_us, by_k=by_k, bytes=by, dkld=gain[c] * noise, src=sorted(src)))
        if not any(o["type"] == t0 for o in opts):
            fixed.extend(members)  # no time for its own type: kept
            continue
        base = [o for o in opts if o["type"] == t0][0]
        for o in opts:
            o["dbytes"], o["ddkld"] = o["bytes"] - base["bytes"], o["dkld"] - base["dkld"]
        units.append(dict(members=members, cls=c, opts=opts, base=base))

    # the objective and the second budget: least E[T] within the KLD budget, or least KLD within the E[T] budget
    for u in units:
        for o in u["opts"]:
            o["dus"] = o["us"] - u["base"]["us"]
    obj, other, cap = ("dus", "ddkld", a.kld_slack) if a.objective == "time" else ("ddkld", "dus", a.ms_slack * 1000)

    def solve(lb, lx):
        pick = [min(u["opts"], key=lambda o: (o[obj] + lb * o["dbytes"] + lx * o[other], o["type"] != u["base"]["type"]))
                for u in units]
        return pick, *(sum(o[f] for o in pick) for f in (obj, "dbytes", other))

    best = None
    scale = 1.0 if a.objective == "time" else 1e-6  # µs of E[T] or units of KLD per byte, per unit of the other
    grid_b = [0.0] + [scale * 10 ** (e / 4) for e in range(-40, 1)]
    grid_x = [0.0] + ([10 ** (e / 4) for e in range(-4, 40)] if a.objective == "time" else
                      [10 ** (e / 4) for e in range(-48, 0)])
    for lb in grid_b:
        for lx in grid_x:
            pick, v, db, dx = solve(lb, lx)
            if db <= a.byte_slack and dx <= cap and (best is None or v < best[1]):
                best = (pick, v, db, dx, lb, lx)
    if best is None:
        raise SystemExit("no feasible choice (the start itself should be): check the inputs")
    pick = list(best[0])
    # greedy repair: single-unit moves that keep both budgets and lower the objective
    improved = True
    while improved:
        improved = False
        db, dx = sum(o["dbytes"] for o in pick), sum(o[other] for o in pick)
        for i, u in enumerate(units):
            for o in u["opts"]:
                if o[obj] < pick[i][obj] - 1e-12 and db - pick[i]["dbytes"] + o["dbytes"] <= a.byte_slack and \
                        dx - pick[i][other] + o[other] <= cap:
                    db, dx = db - pick[i]["dbytes"] + o["dbytes"], dx - pick[i][other] + o[other]
                    pick[i] = o
                    improved = True
    write_plan(a, sd, start, pub, units, pick, fixed, pk, best)


def write_plan(a, sd, start, pub, units, pick, fixed, pk, best):
    os.makedirs(a.out, exist_ok=True)
    new = {n: t for n, (t, _d, _s) in start.items()}
    info = {}
    for u, o in zip(units, pick):
        for x in u["members"]:
            new[x] = o["type"]
            info[x] = (u, o)
    rows, tensors = [], []
    for n, t, dims, size, shard in sd["tensors"]:
        pt = pub[n][0]
        c = tclass(n, pt, dims)
        nt = new[n]
        nsize = pub[n][2] if nt == pt else (size if nt == t else (nbytes(nt, dims) + 31) // 32 * 32)
        how = "keep" if nt == pt else ("retype" if c == "router" else "quantize")
        u, o = info.get(n, (None, None))
        layer = re.match(r"blk\.(\d+)\.", n)
        rows.append((n, qclass(n) or c, layer[1] if layer else "", "x".join(map(str, dims)), pt, t, nt, how, size, nsize,
                     f"{u['base']['us']:.2f}" if u else "", f"{o['us']:.2f}" if o else "",
                     f"{o['ddkld']:.6f}" if o else "", ";".join(o["src"]) if o else ""))
        tensors.append([n, nt, dims, nsize, shard])
    write_tsv(os.path.join(a.out, "plan.tsv"),
              ("tensor", "class", "layer", "shape", "published_type", "start_type", "new_type", "how", "start_bytes",
               "new_bytes", "unit_us_start", "unit_us_new", "unit_dkld", "time_sources"), rows)
    with open(os.path.join(a.out, "types.txt"), "w") as f:
        for n, _t, dims, _s, _sh in sd["tensors"]:
            if len(dims) >= 2 and tclass(n, pub[n][0], dims) != "router":
                f.write(f"^{re.escape(n)}$={new[n].lower()}\n")
    with open(os.path.join(a.out, "take.txt"), "w") as f:
        f.writelines(f"{r[0]}\n" for r in rows if r[7] == "quantize")
    with open(os.path.join(a.out, "retype.txt"), "w") as f:
        f.write(r"ffn_gate_inp\.weight$=BF16@F32" + "\n")
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(dict(sd, name=a.name, tensors=tensors), f, separators=(",", ":"))
        f.write("\n")
    ks = sorted(pk)
    srow = []
    for label, sel in (("start", [u["base"] for u in units]), (a.name, pick)):
        by_k = {k: sum(o["by_k"][k] for o in sel) / 1000 for k in ks}
        srow.append((label, *(f"{by_k[k]:.3f}" for k in ks), f"{sum(o['us'] for o in sel) / 1000:.3f}",
                     f"{sum(o['dbytes'] for o in sel) / MiB:.1f}", f"{sum(o['ddkld'] for o in sel):.5f}"))
    changes = collections.Counter((o0["type"], o["type"]) for u, o, o0 in zip(units, pick, (u["base"] for u in units))
                                  if o["type"] != o0["type"])
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write("file\t" + "\t".join(f"ms_k{k}" for k in ks) + "\tms_expected\tdelta_mib\tpred_dkld\n")
        for r in srow:
            f.write("\t".join(r) + "\n")
        f.write(f"# P(k) {','.join(f'{k}={pk[k]:.4f}' for k in ks)}; dense units {len(units)}, kept {len(fixed)} "
                f"tensors; objective {a.objective}, multipliers {best[4]:.3g} a byte, {best[5]:.3g} a unit of the "
                f"other budget\n")
        for (t0, t1), n in sorted(changes.items()):
            f.write(f"# {t0} -> {t1}: {n} units\n")
    print(open(os.path.join(a.out, "summary.tsv")).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("times")
    p.add_argument("--perf", action="append", default=[])
    p.add_argument("--ggml-est", action="append", default=[])
    p.add_argument("--bench", action="append", default=[])
    p.add_argument("--dispatch", default="min", choices=("min", "on", "off"))
    p.add_argument("--basis", default="bench", choices=("bench", "ggml"))
    p = sub.add_parser("pk")
    p.add_argument("--metrics", nargs=2, required=True)
    p = sub.add_parser("select")
    p.add_argument("--start", required=True)
    p.add_argument("--published", required=True)
    p.add_argument("--times", required=True)
    p.add_argument("--qerr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pk", default="4=0.989,1=0.011")
    p.add_argument("--gain", type=float)
    p.add_argument("--calib")
    p.add_argument("--byte-slack", type=float, default=0.0)
    p.add_argument("--kld-slack", type=float, default=0.0)
    p.add_argument("--objective", default="time", choices=("time", "kld"))
    p.add_argument("--ms-slack", type=float, default=0.0)
    p.add_argument("--classes", default="fa_qkv,fa_out,gdn_qkv,gdn_gate,gdn_out,shexp,ple")
    p.add_argument("--types", default="IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0,Q4_0,Q5_0")
    p.add_argument("--loose", action="store_true")
    p.add_argument("--name", default="vmix")
    p = sub.add_parser("calib")
    p.add_argument("--start", required=True)
    p.add_argument("--qerr", required=True)
    p.add_argument("--kld", required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--text", default="agent")
    p.add_argument("--base", default="pmix-q2-fast")
    a = ap.parse_args()
    if a.cmd == "select" and a.gain is None and not a.calib:
        ap.error("select needs --gain or --calib (vmix.py calib prints the table)")
    {"times": cmd_times, "pk": cmd_pk, "select": cmd_select, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
