#!/usr/bin/env python3
"""C815 (Q1424): LLAMA_SPEC_ADAPT_ONLINE=1 and =2 as built, priced on FN80's dumps with FN82's replay (est.).

    python3 price_built.py DUMP.jsonl [...] --table v4est=v4_est.tsv [--cols 8,10,12,14,16] [--n-max 6] --out DIR

FN82 (bench/draft_count/adapt_replay.py) priced "decided while drafting" with the rule's running speed-up (S run)
charged with the passes actually drafted. The build keeps the switch's state as built (LLAMA_SPEC_ADAPT_DRAFT_MS for
every step), so that its verified tokens are exactly the switch's as built; only the drafting passes change. This
script replays, on the same steps, with the switch's defaults (S run, widths 1,3,5,9,17, scale 1) and its cost table
V(1..n_max + 1) with draft_ms = n_max x pass_ms:
- post:   the switch as built (n_max drafting passes);
- fn82:   FN82's "online" arm (the state charged with the passes drafted);
- built1: LLAMA_SPEC_ADAPT_ONLINE=1 (as post, drafting stops at the first allowed length whose test fails, or at the
          last allowed length);
- built2: LLAMA_SPEC_ADAPT_ONLINE=2 (also as soon as the next allowed length cannot pay whatever the probabilities of
          the tokens still to draft): common_spec_adapt_continue ported line by line;
- fixed3: the draft cut to 3 tokens;
- post-wall, fn82-wall, built1-wall, built2-wall: the same with every width (LLAMA_SPEC_ADAPT_WIDTHS=1,...,n_max + 1).
Each step's est. time is its drafting passes x pass_ms + V(verified + 1), as in FN82. Checks: built1 and built2 keep
and accept exactly post's tokens on every step; built2's passes <= built1's <= n_max; built1's passes are FN82's
online rule's stopping point on post's choices. Output: DIR/price.tsv (per load, table and arm), DIR/compare.tsv
(differences with 95 % bootstrap intervals over requests, paired), DIR/checks.txt. Standard library only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "bench" / "draft_count"))
import adapt_replay as ar  # noqa: E402


def walk(rule, p, s_run):
    """common_spec_adapt_choose's walk with C815's extra: (n_keep, n_fail, len_next, gain, pp, thr)."""
    t1 = rule.T(1)

    def s_of(e_tau, cost):
        if rule.s_mode == "step":
            return e_tau * t1 / cost
        if rule.s_mode == "run":
            return s_run
        return rule.s_fixed

    n_keep, e_tau, cost = 0, 1.0, rule.draft_ms + t1
    s = s_of(e_tau, cost)
    path = 1.0
    for ln in rule.lengths:
        if ln <= n_keep:
            continue
        if ln > len(p):
            gain, pp = 0.0, path
            for i in range(n_keep, len(p)):
                pp *= max(0.0, min(1.0, p[i]))
                gain += pp
            dt = rule.T(ln + 1) - rule.T(n_keep + 1)
            return n_keep, 0, ln, gain, pp, rule.scale * s * dt / t1
        gain, pp = 0.0, path
        for i in range(n_keep, ln):
            pp *= max(0.0, min(1.0, p[i]))
            gain += pp
        dt = rule.T(ln + 1) - rule.T(n_keep + 1)
        if not (gain > rule.scale * s * dt / t1):
            return n_keep, ln, 0, 0.0, 1.0, 0.0
        n_keep = ln
        e_tau += gain
        cost += dt
        s = s_of(e_tau, cost)
        path = pp
    return n_keep, 0, 0, 0.0, 1.0, 0.0


def cont(rule, p, s_run, online):
    """common_spec_adapt_continue."""
    _, n_fail, len_next, gain, pp, thr = walk(rule, p, s_run)
    if n_fail > 0 or len_next == 0:
        return False
    if online >= 2:
        g = gain
        for _ in range(len(p), len_next):
            g += pp
        if not (g > thr):
            return False
    return True


def run(steps, t, n_max, pass_ms, widths):
    """post, built1, built2 (one state trajectory) and fn82's online arm."""
    rule = ar.Rule(t[: n_max + 1], widths, n_max, n_max * pass_ms)
    rule.init()
    post, b1, b2, bad = [], [], [], 0
    for s in steps:
        p = s["p"][:n_max]
        nd = len(p)
        s_run = rule.s()
        n_keep, _ = rule.choose(p, s_run)
        acc = min(s["acc"], n_keep)
        qs = []
        # the drafter's n_last: the last allowed length within the draft as built (a dump's shorter draft was cut at
        # the slot's cap)
        last = max(ln for ln in rule.lengths if ln <= nd)
        for online in (1, 2):
            q = nd
            for j in range(1, nd + 1):
                if j >= last or not cont(rule, p[:j], s_run, online):
                    q = j
                    break
            # what the shorter draft keeps: the same tokens (the drafter's own check)
            if rule.choose(p[:q], s_run)[0] != n_keep:
                bad += 1
            qs.append(max(1, q))
        rule.update(n_keep, acc)
        post.append((n_keep, nd, acc))
        b1.append((n_keep, qs[0], acc))
        b2.append((n_keep, qs[1], acc))
    fn82 = ar.run_adapt(steps, t, n_max, pass_ms, "run", widths, 1.0, True)
    return post, b1, b2, fn82, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--cols", default="8,10,12,14,16")
    ap.add_argument("--table", action="append", default=[])
    ap.add_argument("--step3", type=float, default=128.92)
    ap.add_argument("--pass-ms", type=float, default=3.645)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tables = [(f"col{float(c):g}", ar.uniform_table(a.step3, a.pass_ms, float(c), 9)) for c in a.cols.split(",") if c]
    for spec in a.table:
        name, _, path = spec.partition("=")
        tables.append((name, ar.read_table(path)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    price, comp, checks = [], [], []
    for path in a.dumps:
        label = ar.label_of(path)
        steps = ar.read_dump(path)
        if max(s["n"] for s in steps) < a.n_max:
            continue
        reqs = sorted({s["req"] for s in steps}, key=str)
        mat = ar.boot_matrix(reqs, a.boot, a.seed)
        for tname, t in tables:
            arms = {"fixed3": ar.run_fixed(steps, 3)}
            # the switch's default widths, and every width (LLAMA_SPEC_ADAPT_WIDTHS=1,...,n_max + 1): arms "-wall"
            for wsuf, widths in (("", ar.DEFAULT_WIDTHS), ("-wall", list(range(1, a.n_max + 2)))):
                post, b1, b2, fn82, bad = run(steps, t, a.n_max, a.pass_ms, widths)
                arms.update({"post" + wsuf: post, "fn82" + wsuf: fn82, "built1" + wsuf: b1, "built2" + wsuf: b2})
                same = all(x[0] == y[0] and x[2] == y[2] for x, y in zip(post, b1)) and \
                    all(x[0] == y[0] and x[2] == y[2] for x, y in zip(post, b2))
                order = all(z[1] <= y[1] <= x[1] for x, y, z in zip(post, b1, b2))
                # FN82's online stop on post's choices: the first allowed length past n_keep, else the last allowed
                rule = ar.Rule(t[: a.n_max + 1], widths, a.n_max, a.n_max * a.pass_ms)
                fn82stop = 0
                for (n_keep, nd, _), (_, q, _) in zip(post, b1):
                    nxt = [ln for ln in rule.lengths if n_keep < ln <= nd]
                    fn82stop += (max(1, nxt[0] if nxt else n_keep) == q)
                checks.append(f"{label}\t{tname}\twidths {'every' if wsuf else 'default'}\tsteps {len(steps)}: built1 "
                              f"and built2 keep and accept post's tokens: {'yes' if same else 'NO'}; passes built2 <= "
                              f"built1 <= post: {'yes' if order else 'NO'}; shorter drafts the rule cuts to other tokens "
                              f"than post's: {bad} of {2 * len(steps)}; built1 at FN82's online stop: {fn82stop} of "
                              f"{len(steps)}")
            scored = {}
            for nm, picks in arms.items():
                per, hist = ar.score(steps, picks, t, a.pass_ms)
                z = ar.totals(per)
                scored[nm] = per
                price.append({"load": label, "table": tname, "arm": nm, "steps": z[0],
                              "tokens_a_verify_step": z[1] / z[0], "columns_a_step": z[2] / z[0],
                              "passes_a_step": z[3] / z[0], "ms_a_step_est": z[4] / z[0],
                              "tok_s_est": 1000.0 * z[1] / z[4]})
            rates = {nm: ar.boot_rates(scored[nm], mat) for nm in scored}
            tok = {r["arm"]: r["tok_s_est"] for r in price if r["load"] == label and r["table"] == tname}
            pairs = []
            for w in ("", "-wall"):
                pairs += [("built1" + w, "fixed3"), ("built2" + w, "fixed3"), ("post" + w, "fixed3"),
                          ("fn82" + w, "fixed3"), ("built1" + w, "post" + w), ("built2" + w, "post" + w),
                          ("built2" + w, "built1" + w), ("built1" + w, "fn82" + w)]
            pairs += [("built2-wall", "built2"), ("built2-wall", "post")]
            for nm, ref in pairs:
                lo, hi = ar.boot_diff(rates[nm], rates[ref])
                comp.append({"load": label, "table": tname, "arm": nm, "against": ref,
                             "diff_tok_s_est": tok[nm] - tok[ref], "lo95": lo, "hi95": hi})
    cols = ["load", "table", "arm", "steps", "tokens_a_verify_step", "columns_a_step", "passes_a_step", "ms_a_step_est",
            "tok_s_est"]
    with open(out / "price.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in price:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    cols = ["load", "table", "arm", "against", "diff_tok_s_est", "lo95", "hi95"]
    with open(out / "compare.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in comp:
            f.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    (out / "checks.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
