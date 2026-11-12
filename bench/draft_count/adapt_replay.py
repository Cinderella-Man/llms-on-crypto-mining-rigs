#!/usr/bin/env python3
"""The draft length chosen per step (LLAMA_SPEC_ADAPT, Q687), replayed offline on dumped MTP drafts (FN82, Q1402).

    python3 adapt_replay.py DUMP.jsonl [...] --out DIR [--n-max 3,4,5,6] [--fixed 3,4,5,6] [--cols 8,10,12,14,16]
                            [--table NAME=FILE ...] [--step3 128.92] [--pass-ms 3.645] [--scales 0.5,0.75,1,1.5,2]
                            [--pmin 0.3,0.5,0.7,0.8,0.9,0.95] [--boot 2000] [--seed 1] [--check-cut CUT.tsv]
                            [--also ARM,...]
    python3 adapt_replay.py --selftest

Each DUMP is one load's LLAMA_SPEC_DUMP (run_draft_count.sh with DUMP=1): one line per verify step with the drafted
tokens, the accepted count n_acc and the drafter's candidates per position ("cand", "p", "pick"). The drafter's
probability of the drafted token at position i is p[i][pick[i]]: the same value, from the same candidate list, that
llama-server's LLAMA_SPEC_ADAPT reads (spec_push_p and spec_info_add_marginal in common/speculative.cpp).

Every arm is scored on the same steps. Verifying a prefix of a draft accepts the same positions as verifying all of it
(each position's test depends only on the tokens before it), so a step cut to n verified drafted tokens keeps
1 + min(n_acc, n) tokens. The arms:

- fixed L: the draft cut to L tokens (L drafting passes, a verify pass of L + 1);
- adapt: the switch's rule (common_spec_adapt_choose, common_spec_adapt_state; ported below line by line) on a draft
  of n_max tokens, with S "run" (the default) or "step", widths "def" (the default 1,3,5,9,17) or "all" (every width
  1..n_max + 1) and a threshold scale. "post" is the switch as built: the drafter drafts all n_max tokens and the rule
  then cuts the draft, so a step always pays n_max drafting passes. "online" (est.: not built) is the same rule taken
  while drafting, which stops at the first length that does not pay and so drafts only up to it;
- pmin x: the drafter's own stop (--spec-draft-p-min x): drafting stops at the first position whose most probable
  candidate is below x; that position's pass was run, its token is not drafted;
- oracle: the draft cut to exactly what is accepted (a ceiling, not a rule), "post" with n_max passes, "online" with
  the passes up to the first rejection.

A step's time (est.) is its drafting passes x pass_ms + V(n + 1), V(k) the step without drafting at a verify width of
k. --cols c gives V(k) = step3 - 3 x pass_ms + (k - 4) x c (FN80's model: the 3-token step plus one column and one
drafting pass a token); --table NAME=FILE reads V(k) from a TSV with columns k and ms. The rule's own table
(LLAMA_SPEC_ADAPT_COST) is V(1..n_max + 1) and its draft_ms (LLAMA_SPEC_ADAPT_DRAFT_MS) is n_max x pass_ms, so its
modelled cost C(d) = draft_ms + T(d + 1) is the step's est. time.

Output in DIR: arms.tsv (every arm, cost table and load: tokens a verify step, verify columns a step, drafting passes a
step, est. ms a step and tok/s, how often each length was kept), compare.tsv (the main arms against every fixed length:
the est. tok/s difference with a 95 % bootstrap interval over requests, paired on the same steps), calibration.tsv (the
drafter's probability against acceptance by position), check.tsv and summary.md. Standard library only.
"""
import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_WIDTHS = [1, 3, 5, 9, 17]  # common_spec_adapt_params::widths


# ---------------------------------------------------------------- the switch's rule (common/speculative.cpp)


class Rule:
    """common_spec_adapt_params with common_spec_adapt_cost, _lengths, _choose and common_spec_adapt_state."""

    def __init__(self, cost_ms, widths, n_max, draft_ms, s_mode="run", scale=1.0, s0=1.0, ema=1.0 / 32, s_fixed=1.0):
        self.t = list(cost_ms)
        self.widths = list(widths)
        self.n_max = n_max
        self.draft_ms = draft_ms
        self.s_mode = s_mode
        self.scale = scale
        self.s0 = s0
        self.ema = ema
        self.s_fixed = s_fixed
        self.lengths = self.make_lengths(n_max)
        self.tok = 0.0
        self.cost = 0.0

    def T(self, k):
        t, n = self.t, len(self.t)
        assert n and k >= 1
        if k <= n:
            return t[k - 1]
        slope = t[n - 1] - t[n - 2] if n >= 2 else t[0]
        return t[n - 1] + (k - n) * slope

    def make_lengths(self, n_max):
        out = {0}
        for w in self.widths:
            out.add(max(0, min(w - 1, max(0, n_max))))
        return sorted(out)

    def choose(self, p, s_run):
        """Returns (n_keep, the length whose test failed or None)."""
        t1 = self.T(1)

        def s_of(e_tau, cost):
            if self.s_mode == "step":
                return e_tau * t1 / cost
            if self.s_mode == "run":
                return s_run
            return self.s_fixed

        n_keep, e_tau, cost = 0, 1.0, self.draft_ms + t1
        s = s_of(e_tau, cost)
        path = 1.0
        for ln in self.lengths:
            if ln <= n_keep:
                continue
            if ln > len(p):
                break
            gain, pp = 0.0, path
            for i in range(n_keep, ln):
                pp *= max(0.0, min(1.0, p[i]))
                gain += pp
            dt = self.T(ln + 1) - self.T(n_keep + 1)
            if not (gain > self.scale * s * dt / t1):
                return n_keep, ln
            n_keep = ln
            e_tau += gain
            cost += dt
            s = s_of(e_tau, cost)
            path = pp
        return n_keep, None

    # common_spec_adapt_state
    def init(self):
        self.tok = self.s0
        self.cost = self.T(1)

    def update(self, n_keep, n_acc, draft_ms=None):
        if self.cost <= 0.0:
            self.init()
        a = self.ema
        dm = self.draft_ms if draft_ms is None else draft_ms
        self.tok = (1.0 - a) * self.tok + a * (1.0 + n_acc)
        self.cost = (1.0 - a) * self.cost + a * (dm + self.T(n_keep + 1))

    def s(self):
        if self.cost <= 0.0:
            return self.s0
        return self.T(1) * self.tok / self.cost


# ---------------------------------------------------------------- loading


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def read_dump(path):
    """One load's verify steps, in order: the request, the drafted count, n_acc, p of each drafted token, top-1 p."""
    steps = []
    with open_text(path) as f:
        for line in f:
            s = json.loads(line)
            d = s.get("draft") or []
            if not d:
                continue
            n = len(d)
            p, top = [], []
            for i in range(n):
                k = s["pick"][i]
                p.append(float(s["p"][i][k]) if k is not None and k >= 0 else 0.0)
                top.append(float(s["p"][i][0]) if s["p"][i] else 0.0)
            steps.append({"req": (s.get("run"), s.get("slot"), s.get("task")), "n": n, "acc": int(s["n_acc"]),
                          "p": p, "top": top})
    return steps


def label_of(path):
    name = Path(path).name
    for suf in (".gz", ".jsonl"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name[5:] if name.startswith("dump-") else name


# ---------------------------------------------------------------- cost tables


def uniform_table(step3, pass_ms, col, kmax):
    base = step3 - 3 * pass_ms
    return [base + (k - 4) * col for k in range(1, kmax + 1)]


def read_table(path):
    rows = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        ik, im = head.index("k"), head.index("ms")
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            x = line.rstrip("\n").split("\t")
            rows[int(x[ik])] = float(x[im])
    ks = sorted(rows)
    if ks != list(range(1, len(ks) + 1)):
        raise SystemExit(f"{path}: k must run 1, 2, 3, ...")
    return [rows[k] for k in ks]


def table_T(t, k):
    n = len(t)
    if k <= n:
        return t[k - 1]
    slope = t[n - 1] - t[n - 2] if n >= 2 else t[0]
    return t[n - 1] + (k - n) * slope


# ---------------------------------------------------------------- arms


def run_fixed(steps, L):
    out = []
    for s in steps:
        n = min(L, s["n"])
        out.append((n, n, min(s["acc"], n)))  # verified drafted, drafting passes, accepted
    return out


def run_adapt(steps, t, n_max, pass_ms, s_mode, widths, scale, online):
    rule = Rule(t[: n_max + 1], widths, n_max, n_max * pass_ms, s_mode=s_mode, scale=scale)
    rule.init()
    out = []
    for s in steps:
        p = s["p"][:n_max]
        nd = len(p)
        n_keep, failed = rule.choose(p, rule.s())
        acc = min(s["acc"], n_keep)
        if online:
            q = failed if failed is not None else n_keep
            q = max(1, min(q, nd))
            rule.update(n_keep, acc, draft_ms=q * pass_ms)
        else:
            q = nd
            rule.update(n_keep, acc)
        out.append((n_keep, q, acc))
    return out


def run_pmin(steps, n_max, x):
    out = []
    for s in steps:
        nd = min(n_max, s["n"])
        n, q = nd, nd
        for i in range(nd):
            if s["top"][i] < x:
                n, q = i, i + 1
                break
        out.append((n, q, min(s["acc"], n)))
    return out


def run_oracle(steps, n_max, online):
    out = []
    for s in steps:
        nd = min(n_max, s["n"])
        a = min(s["acc"], nd)
        q = min(a + 1, nd) if online else nd
        out.append((a, max(q, 1), a))
    return out


def score(steps, picks, t, pass_ms):
    """Per request: steps, tokens, verify columns, drafting passes, est. ms; and the histogram of kept lengths."""
    per = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    hist = defaultdict(int)
    for s, (n, q, a) in zip(steps, picks):
        r = per[s["req"]]
        r[0] += 1
        r[1] += 1 + a
        r[2] += n + 1
        r[3] += q
        r[4] += q * pass_ms + table_T(t, n + 1)
        hist[n] += 1
    return per, hist


def totals(per, keys=None):
    keys = per.keys() if keys is None else keys
    z = [0, 0, 0, 0, 0.0]
    for k in keys:
        r = per[k]
        for i in range(5):
            z[i] += r[i]
    return z


def boot_matrix(reqs, n_boot, seed):
    """n_boot resamples of the requests (with replacement), each as (request, times drawn) pairs."""
    rng = random.Random(seed)
    R = len(reqs)
    mat = []
    for _ in range(n_boot):
        c = defaultdict(int)
        for _ in range(R):
            c[reqs[rng.randrange(R)]] += 1
        mat.append(list(c.items()))
    return mat


def boot_rates(per, mat):
    """The est. tok/s of every resample (the same resamples for every arm, so differences are paired)."""
    out = []
    for pick in mat:
        t = m = 0.0
        for k, c in pick:
            r = per[k]
            t += c * r[1]
            m += c * r[4]
        out.append(1000.0 * t / m)
    return out


def boot_diff(rates_a, rates_b):
    diffs = sorted(x - y for x, y in zip(rates_a, rates_b))
    n = len(diffs)
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


# ---------------------------------------------------------------- calibration


def calibration(label, steps, n_max):
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001]
    rows = []
    for i in range(n_max):
        b = defaultdict(lambda: [0, 0, 0.0])
        for s in steps:
            if s["n"] <= i or s["acc"] < i:
                continue  # position i + 1 not drafted or not reached
            x = s["p"][i]
            j = max(0, min(len(edges) - 2, next(e for e in range(len(edges) - 1) if x < edges[e + 1])))
            c = b[j]
            c[0] += 1
            c[1] += 1 if s["acc"] > i else 0
            c[2] += x
        for j in range(len(edges) - 1):
            c = b.get(j)
            if not c:
                continue
            rows.append({"load": label, "position": i + 1, "p_from": edges[j], "p_to": min(1.0, edges[j + 1]),
                         "reached": c[0], "accepted": c[1], "accept_rate": c[1] / c[0], "mean_p": c[2] / c[0]})
        tot = [sum(b[j][k] for j in b) for k in range(3)]
        if tot[0]:
            rows.append({"load": label, "position": i + 1, "p_from": "all", "p_to": "",
                         "reached": tot[0], "accepted": tot[1], "accept_rate": tot[1] / tot[0],
                         "mean_p": tot[2] / tot[0]})
    return rows


# ---------------------------------------------------------------- output


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.4f}"
    return str(x)


def write_tsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(c, "")) for c in cols) + "\n")


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- main


def analyse(a):
    loads = [(label_of(p), read_dump(p)) for p in a.dumps]
    tables = [(f"col{c:g}", uniform_table(a.step3, a.pass_ms, c, 9)) for c in a.cols]
    for spec in a.table:
        name, _, path = spec.partition("=")
        tables.append((name, read_table(path)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    arms_rows, cmp_rows, cal_rows, check_rows = [], [], [], []
    main_keys = {}
    booted = set()
    for label, steps in loads:
        dmax = max(s["n"] for s in steps)
        cal_rows += calibration(label, steps, dmax)
        reqs = sorted({s["req"] for s in steps}, key=str)
        mat = boot_matrix(reqs, a.boot, a.seed)
        for tname, t in tables:
            arms = []  # (name, family, n_max, picks)
            for L in a.fixed:
                if L <= dmax:
                    arms.append((f"fixed{L}", "fixed", L, run_fixed(steps, L)))
            for N in a.n_max:
                if N > dmax:
                    continue
                for mode in ("run", "step"):
                    for wname, widths in (("def", DEFAULT_WIDTHS), ("all", list(range(1, N + 2)))):
                        for x in a.scales:
                            for acct in ("post", "online"):
                                picks = run_adapt(steps, t, N, a.pass_ms, mode, widths, x, acct == "online")
                                arms.append((f"adapt-n{N}-{mode}-w{wname}-x{x:g}-{acct}", f"adapt-{acct}", N, picks))
                for x in a.pmin:
                    arms.append((f"pmin{x:g}-n{N}", "pmin", N, run_pmin(steps, N, x)))
                arms.append((f"oracle-n{N}-post", "oracle-post", N, run_oracle(steps, N, False)))
                arms.append((f"oracle-n{N}-online", "oracle-online", N, run_oracle(steps, N, True)))
            scored = {}
            for name, fam, N, picks in arms:
                per, hist = score(steps, picks, t, a.pass_ms)
                z = totals(per)
                row = {"load": label, "table": tname, "arm": name, "family": fam, "n_max": N, "steps": z[0],
                       "tokens_a_verify_step": z[1] / z[0], "columns_a_step": z[2] / z[0],
                       "passes_a_step": z[3] / z[0], "ms_a_step_est": z[4] / z[0],
                       "tok_s_est": 1000.0 * z[1] / z[4],
                       "kept": " ".join(f"{k}:{hist[k]}" for k in sorted(hist))}
                arms_rows.append(row)
                scored[name] = (row, per, fam, N)
            # the arms compared with a bootstrap: the fixed lengths, the switch as built (S run, default widths,
            # scale 1, post) and with every width, and each family's best by est. tok/s at each n_max
            fixed = [n for n in scored if scored[n][2] == "fixed"]
            best_fixed = max(fixed, key=lambda n: scored[n][0]["tok_s_est"])
            chosen = []
            for N in a.n_max:
                for nm in [f"adapt-n{N}-run-wdef-x1-post", f"adapt-n{N}-run-wall-x1-post",
                           f"adapt-n{N}-run-wall-x1-online"] + [x.replace("{N}", str(N)) for x in a.also]:
                    if nm in scored:
                        chosen.append(nm)
                for fam in ("adapt-post", "adapt-online", "pmin", "oracle-post", "oracle-online"):
                    c = [n for n in scored if scored[n][2] == fam and scored[n][3] == N]
                    if c:
                        chosen.append(max(c, key=lambda n: scored[n][0]["tok_s_est"]))
            seen = set()
            stem = label.rsplit("-", 1)[0]
            do_boot = (stem, tname) not in booted
            booted.add((stem, tname))
            rates = {}

            def rate(nm):
                if nm not in rates:
                    rates[nm] = boot_rates(scored[nm][1], mat)
                return rates[nm]

            for nm in chosen + fixed:
                if nm in seen:
                    continue
                seen.add(nm)
                row, per, fam, N = scored[nm]
                for ref in fixed:
                    if ref == nm:
                        continue
                    rref = scored[ref][0]
                    # the second load of a length repeats the first's steps (check.tsv): no second bootstrap
                    lo, hi = boot_diff(rate(nm), rate(ref)) if do_boot else (float("nan"), float("nan"))
                    cmp_rows.append({"load": label, "table": tname, "arm": nm, "against": ref,
                                     "best_fixed": "yes" if ref == best_fixed else "",
                                     "tok_s_est": row["tok_s_est"], "against_tok_s_est": rref["tok_s_est"],
                                     "diff": row["tok_s_est"] - rref["tok_s_est"],
                                     "ratio": row["tok_s_est"] / rref["tok_s_est"],
                                     "lo_requests": lo, "hi_requests": hi})
            main_keys[(label, tname)] = (best_fixed, chosen)

    # checks: the fixed cuts against FN80's cut.tsv (tokens a verify step), the two loads of a length the same
    if a.check_cut:
        ref = {}
        with open(a.check_cut) as f:
            head = f.readline().rstrip("\n").split("\t")
            for line in f:
                x = dict(zip(head, line.rstrip("\n").split("\t")))
                ref[(x["label"], int(x["cut_to"]))] = float(x["tokens_a_verify_step"])
        for r in arms_rows:
            if r["family"] != "fixed" or r["table"] != tables[0][0]:
                continue
            want = ref.get((r["load"], r["n_max"]))
            if want is None:
                continue
            ok = abs(want - r["tokens_a_verify_step"]) < 5e-5
            check_rows.append({"check": "fixed cut = FN80's cut.tsv", "load": r["load"], "L": r["n_max"],
                               "here": r["tokens_a_verify_step"], "cut_tsv": want, "ok": "OK" if ok else "FAIL"})
    by = defaultdict(dict)
    for r in arms_rows:
        by[(r["table"], r["arm"])][r["load"]] = r
    pairs = defaultdict(list)
    for label, _ in loads:
        stem = label.rsplit("-", 1)[0]
        pairs[stem].append(label)
    for stem, labs in pairs.items():
        if len(labs) < 2:
            continue
        same = all(len({(round(v["tokens_a_verify_step"], 9), round(v["ms_a_step_est"], 9))
                        for l, v in d.items() if l in labs}) == 1
                   for d in by.values() if all(l in d for l in labs))
        check_rows.append({"check": "loads of one length give the same arms", "load": " ".join(labs), "L": "",
                           "here": "", "cut_tsv": "", "ok": "OK" if same else "DIFFER"})

    write_tsv(out / "arms.tsv", arms_rows)
    write_tsv(out / "compare.tsv", cmp_rows)
    write_tsv(out / "calibration.tsv", cal_rows)
    write_tsv(out / "check.tsv", check_rows)

    # summary
    lines = [f"# adapt_replay: {', '.join(l for l, _ in loads)}\n",
             f"step3 {a.step3} ms, drafting pass {a.pass_ms} ms; tables: {', '.join(n for n, _ in tables)} (est.)\n"]
    for (label, tname), (best_fixed, chosen) in main_keys.items():
        rows = [r for r in arms_rows if r["load"] == label and r["table"] == tname
                and (r["family"] == "fixed" or r["arm"] in chosen)]
        cm = {r["arm"]: r for r in cmp_rows if r["load"] == label and r["table"] == tname and r["best_fixed"]}
        rows2 = []
        for r in rows:
            c = cm.get(r["arm"])
            r = dict(r)
            if c and not math.isnan(c["lo_requests"]):
                r["vs_best_fixed"] = f"{c['diff']:+.3f} ({c['lo_requests']:+.3f} to {c['hi_requests']:+.3f})"
            elif c:
                r["vs_best_fixed"] = f"{c['diff']:+.3f}"
            else:
                r["vs_best_fixed"] = f"best fixed ({best_fixed})" if r["arm"] == best_fixed else ""
            rows2.append(r)
        lines += [f"\n## {label}, {tname}\n",
                  md(rows2, ["arm", "tokens_a_verify_step", "columns_a_step", "passes_a_step", "ms_a_step_est",
                             "tok_s_est", "vs_best_fixed", "kept"])]
    if check_rows:
        lines += ["\n## Checks\n", md(check_rows, ["check", "load", "L", "here", "cut_tsv", "ok"])]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    bad = [r for r in check_rows if r["ok"] != "OK"]
    print(f"{len(arms_rows)} arm rows, {len(cmp_rows)} comparisons, {len(check_rows)} checks, {len(bad)} not OK",
          file=sys.stderr)
    return 1 if bad else 0


def selftest():
    # T = 100, 110, 120, 130 (a column of 10 ms), draft_ms 0, S fixed at 2: a length pays when its added path
    # probabilities exceed 2 x 10 / 100 = 0.2 a position
    r = Rule([100, 110, 120, 130], [1, 2, 3, 4], 3, 0.0, s_mode="fixed", s_fixed=2.0)
    assert r.choose([0.9, 0.5, 0.3], 0) == (2, 3), r.choose([0.9, 0.5, 0.3], 0)  # P = 0.9, 0.45, 0.135
    assert r.choose([0.1, 0.9, 0.9], 0) == (0, 1)
    assert r.choose([0.9, 0.9, 0.9], 0) == (3, None)
    # default widths with n_max 3: lengths 0, 2, 3; the first test adds two positions at once
    r = Rule([100, 110, 120, 130], DEFAULT_WIDTHS, 3, 0.0, s_mode="fixed", s_fixed=2.0)
    assert r.lengths == [0, 2, 3], r.lengths
    assert r.choose([0.3, 0.3], 0) == (0, 2)  # 0.3 + 0.09 = 0.39 < 0.4
    assert r.choose([0.5, 0.5, 0.1], 0) == (2, 3)  # 0.75 > 0.4, then 0.025 < 0.2
    # past the table: its last slope
    assert r.T(6) == 150
    # S run: tok and cost averages
    r = Rule([100, 110, 120, 130], [1, 2, 3, 4], 3, 10.0, s_mode="run", ema=0.5)
    r.init()
    assert abs(r.s() - 1.0) < 1e-12
    r.update(3, 3)  # tok 0.5 + 2 = 2.5, cost 50 + 0.5 x 140 = 120
    assert abs(r.s() - 100 * 2.5 / 120) < 1e-12
    # fixed and oracle cuts
    st = [{"req": 1, "n": 6, "acc": 4, "p": [0.9] * 6, "top": [0.9, 0.9, 0.4, 0.9, 0.9, 0.9]}]
    assert run_fixed(st, 3) == [(3, 3, 3)] and run_fixed(st, 6) == [(6, 6, 4)]
    assert run_pmin(st, 6, 0.5) == [(2, 3, 2)]
    assert run_oracle(st, 6, False) == [(4, 6, 4)] and run_oracle(st, 6, True) == [(4, 5, 4)]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--n-max", default="3,4,5,6")
    ap.add_argument("--fixed", default="3,4,5,6")
    ap.add_argument("--cols", default="8,10,12,14,16")
    ap.add_argument("--table", action="append", default=[], help="NAME=FILE, a TSV with columns k and ms (est.)")
    ap.add_argument("--step3", type=float, default=128.92)
    ap.add_argument("--pass-ms", type=float, default=3.645)
    ap.add_argument("--scales", default="0.5,0.75,1,1.5,2")
    ap.add_argument("--pmin", default="0.3,0.5,0.7,0.8,0.9,0.95")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check-cut", help="draft_table.py's cut.tsv, for the fixed cuts' check")
    ap.add_argument("--also", default="", help="more arms to compare with a bootstrap, comma separated, {N} = n_max "
                    "(e.g. adapt-n{N}-run-wdef-x1-online,pmin0.7-n{N})")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.dumps or not a.out:
        ap.error("DUMP ... and --out are required")
    a.n_max = [int(x) for x in a.n_max.split(",") if x]
    a.fixed = [int(x) for x in a.fixed.split(",") if x]
    a.cols = [float(x) for x in a.cols.split(",") if x]
    a.scales = [float(x) for x in a.scales.split(",") if x]
    a.pmin = [float(x) for x in a.pmin.split(",") if x]
    a.also = [x for x in a.also.split(",") if x]
    return analyse(a)


if __name__ == "__main__":
    sys.exit(main())
