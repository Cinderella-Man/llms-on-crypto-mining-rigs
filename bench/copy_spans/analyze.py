"""Step 2 of the copy-span analysis: how much of the model's output is copied from its context, how long the copied
runs are, where they start, and what a prompt-lookup guesser would gain.

    WORK=/path/to/work python3 analyze.py            # reads $WORK/prep/tokens.jsonl.gz, writes $WORK/tables/

Four ways of looking for the source of a generated token (each is a "variant"):
  ctx      the prompt only (the request's context, verbatim);
  own      the prompt and the response's own earlier tokens (what a prompt-lookup guesser sees at decode time: the
           last 16 prompt tokens count as part of the stream and are not a source for themselves);
  esc      the prompt plus the JSON-escaped text of every tool result and user message (what a model has to copy
           when it writes the text inside a JSON string: a newline is the two characters \\n);
  own+esc  both.
A token is "copied at minimum length L" when it lies inside a stretch of at least L consecutive generated tokens
that occurs verbatim earlier in the variant's source. Copy spans are the maximal such stretches.

The guesser simulation follows bench/ngram_replay: at each step the history is the prompt plus the response so far;
the longest suffix of it that occurred earlier (newest occurrence on ties) gives the draft, the n tokens that followed;
the step keeps the draft's longest prefix equal to what the model generated and adds one token.
"""
import bisect
import csv
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_sam, covered_flags, match_profile, quantile, runs  # noqa: E402

MIN_LENS = (4, 8, 16)
VARIANTS = ("ctx", "own", "esc", "own+esc")
TAIL = 16
DRAFT_N = (2, 8, 16, 32, "adapt")
MATCH_MIN = (2, 4, 6, 8, 12, 16)
ORACLE_SKIP = (0, 2, 4, 6, 8, 12, 16)      # tokens of a span the oracle waits for before it drafts the rest


def match_values(v):
    return ORACLE_SKIP if v == "oracle" else MATCH_MIN
BUCKETS = (4, 8, 16, 32, 64, 128, 10 ** 9)
# cost of a k-column verify pass in units of a one-token step: 1 + r (k - 1); r from Q583 (Flash-Next on the rig,
# 136 ms plus 50-59 ms per extra token) and the estimate after the multi-token kernels (8-12 %). `draft` is the
# DFlash2 draft pass added to a DFlash2 step (15-25 ms).
COSTS = {"today": dict(r=0.40, draft=0.15), "kernels": dict(r=0.10, draft=0.22)}
DFLASH_TOKENS, DFLASH_COLS = 2.56, 3        # Q561: two drafted tokens, 2.56 tokens per step (overall)


def draft_len(n, l):
    return min(32, 2 * l) if n == "adapt" else n


def simulate(hist, g, pl, pq, n_draft, m_min, regs):
    """One response through the guesser. Returns [steps, tokens, proposals, accepted, proposed, columns, by_region]."""
    lc, n_tok = len(hist), len(g)
    steps = tokens = props = acc = proposed = cols = 0
    by_region = Counter()
    i = 1
    while i < n_tok:
        l, q = pl[i - 1], pq[i - 1]
        d = 0
        if l >= m_min and q >= 0:
            start = q + 1
            d = min(draft_len(n_draft, l), lc + i - start)
        a = 0
        if d > 0:
            while a < d and i + a < n_tok:
                pos = start + a
                if (hist[pos] if pos < lc else g[pos - lc]) != g[i + a]:
                    break
                a += 1
            for k in range(a):
                by_region[regs[i + k]] += 1
        steps += 1
        cols += 1 + d
        tokens += min(a + 1, n_tok - i)
        if d > 0:
            props += 1
            proposed += d
            acc += a
        i += a + 1
    return [steps, tokens, props, acc, proposed, cols, by_region]


def oracle(g, flags, n_draft, skip):
    """A bound, not a guesser (Q64's "oracle copy"): inside a copy span, after `skip` of its tokens were generated,
    it drafts exactly the rest of the span (at most n tokens), always accepted; elsewhere it does not draft."""
    n_tok = len(g)
    end_of = [0] * n_tok            # end of the span containing position p (exclusive), 0 outside
    start_of = [0] * n_tok
    for s, ln in runs(flags):
        for p in range(s, s + ln):
            end_of[p], start_of[p] = s + ln, s
    cap = 32 if n_draft == "adapt" else n_draft
    steps = tokens = props = acc = proposed = cols = 0
    i = 1
    while i < n_tok:
        d = 0
        if end_of[i] and i - start_of[i] >= skip:
            d = min(cap, end_of[i] - i)
        steps += 1
        cols += 1 + d
        tokens += min(d + 1, n_tok - i)
        if d > 0:
            props += 1
            proposed += d
            acc += d
        i += d + 1
    return [steps, tokens, props, acc, proposed, cols, Counter()]


def argument_values(regs, mk, flags):
    """One tuple per tool-call argument value (a <parameter=..> value or a JSON string inside one): (region, tokens,
    copied tokens, offset of the first copied token or -1, whether a copy span runs into the value from before it)."""
    n_tok, out = len(regs), []
    for idx, kind in mk:
        if not (kind == "param" or kind.startswith("jstr")) or idx >= n_tok or not regs[idx].startswith("arg:") \
                or regs[idx].endswith(".scaf"):
            continue
        reg, e = regs[idx], idx
        while e < n_tok and regs[e] == reg:
            e += 1
        first = next((k for k in range(idx, e) if flags[k]), -1)
        out.append((reg, e - idx, sum(flags[idx:e]), first - idx if first >= 0 else -1, bool(idx and flags[idx] and flags[idx - 1])))
    return out


def occ_oracle(c, g, n_draft, m, max_cand=400):
    """A bound on any copy guesser whose key is the last m tokens: at each step it picks, among the (newest 400)
    earlier occurrences of those m tokens in the prompt and the response so far, the one whose continuation is right
    for longest, and drafts exactly that many tokens (at most n); with no right first token it takes a plain step."""
    hist = c + g
    lc, n_tok = len(c), len(g)
    cap = 32 if n_draft == "adapt" else n_draft
    idx = defaultdict(list)             # m-gram -> end positions, ascending
    for q in range(m - 1, lc - 1):
        idx[tuple(hist[q - m + 1:q + 1])].append(q)
    nxt = max(m - 1, lc - 1)
    steps = tokens = props = acc = cols = 0
    i = 1
    while i < n_tok:
        cur = lc + i
        while nxt <= cur - 2:
            idx[tuple(hist[nxt - m + 1:nxt + 1])].append(nxt)
            nxt += 1
        best = 0
        if cur >= m:
            cands = idx.get(tuple(hist[cur - m:cur]))
            if cands:
                for q in reversed(cands[-max_cand:]):
                    a = 0
                    while a < cap and i + a < n_tok and q + 1 + a < cur and hist[q + 1 + a] == hist[cur + a]:
                        a += 1
                    if a > best:
                        best = a
                        if best == cap:
                            break
        steps += 1
        cols += 1 + best
        tokens += min(best + 1, n_tok - i)
        if best:
            props += 1
            acc += best
        i += best + 1
    return [steps, tokens, props, acc, acc, cols, Counter()]


CHAIN_BUCKETS = ((2, 3), (4, 5), (6, 7), (8, 11), (12, 15), (16, 23), (24, 47), (48, 10 ** 9))


def chains(hist, g, pl, pq, flags):
    """Per position i >= 1 with a match of l >= 2 tokens before it: is the next token the match's continuation, and
    how many tokens in a row does the continuation get right (up to 32). Split by the length of the match and by
    whether position i lies inside a copy span (flags). Returns {(bucket, inside): [positions, first_ok, run_sum,
    run>=8, run>=16]}."""
    lc, n_tok = len(hist), len(g)
    out = defaultdict(lambda: [0, 0, 0, 0, 0])
    for i in range(1, n_tok):
        l, q = pl[i - 1], pq[i - 1]
        if l < 2 or q < 0 or q + 1 >= lc + i:
            continue
        b = next(k for k, (lo, hi) in enumerate(CHAIN_BUCKETS) if lo <= l <= hi)
        a = 0
        while a < 32 and i + a < n_tok and q + 1 + a < lc + i:
            pos = q + 1 + a
            if (hist[pos] if pos < lc else g[pos - lc]) != g[i + a]:
                break
            a += 1
        o = out[(b, flags[i])]
        o[0] += 1
        o[1] += a >= 1
        o[2] += a
        o[3] += a >= 8
        o[4] += a >= 16
    return dict(out)


def analyze_request(rec):
    c, g, aux = rec["ctx"], rec["out"], rec["aux"]
    names = rec["rnames"]
    regs = [names[i] for i in rec["reg"]]
    mk = rec["markers"]
    mk_idx = [m[0] for m in mk]
    n_tok = len(g)
    profiles, hists = {}, {}
    profiles["ctx"] = match_profile(c, g, tail=0, own=False)
    profiles["esc"] = match_profile(c + aux, g, tail=0, own=False)
    profiles["own"] = match_profile(c, g, tail=TAIL, own=True)
    hists["own"] = c
    c2 = c[:max(0, len(c) - TAIL)] + aux + c[max(0, len(c) - TAIL):]
    profiles["own+esc"] = match_profile(c2, g, tail=TAIL, own=True)
    hists["own+esc"] = c2

    out = {"meta": {k: rec[k] for k in ("run", "arm", "task", "attempt", "try_", "turn", "session", "id", "finish")},
           "n": n_tok, "regions": Counter(regs), "cov": {}, "spans": {}, "sim": {}, "first_copy": {}}
    for v in VARIANTS:
        pl = profiles[v][0]
        out["cov"][v], out["spans"][v] = {}, {}
        for ml in MIN_LENS:
            flags = covered_flags(pl, ml)
            out["cov"][v][ml] = Counter(r for r, f in zip(regs, flags) if f)
            sp = []
            for s, ln in runs(flags):
                j = bisect.bisect_right(mk_idx, s) - 1
                kind, dist = (mk[j][1], s - mk[j][0]) if j >= 0 else ("none", -1)
                sp.append((ln, regs[s], kind, dist))
            out["spans"][v][ml] = sp
    for v in ("own", "own+esc"):
        pl, pq = profiles[v]
        out["sim"][v] = {}
        for n in DRAFT_N:
            for m in MATCH_MIN:
                out["sim"][v][(n, m)] = simulate(hists[v], g, pl, pq, n, m, regs)
    flags8 = covered_flags(profiles["own"][0], 8)
    out["sim"]["occ-oracle"] = {(n, m): occ_oracle(c, g, n, m) for n in DRAFT_N for m in MATCH_MIN}
    out["sim"]["oracle"] = {(n, m): oracle(g, flags8, n, m) for n in DRAFT_N for m in ORACLE_SKIP}
    out["values"] = {v: argument_values(regs, mk, covered_flags(profiles[v][0], 8)) for v in ("ctx", "own")}
    out["chains"] = chains(hists["own"], g, profiles["own"][0], profiles["own"][1], flags8)
    return out


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(sessions, fn, n_boot=1000, seed=7):
    """95 % interval of fn(sum of per-session vectors) over resamples of the sessions."""
    keys = list(sessions)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        tot = None
        for _ in keys:
            vec = sessions[keys[rng.randrange(len(keys))]]
            tot = list(vec) if tot is None else [a + b for a, b in zip(tot, vec)]
        try:
            vals.append(fn(tot))
        except ZeroDivisionError:
            pass
    vals.sort()
    return quantile(vals, 0.025), quantile(vals, 0.975)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def region_group(r):
    if r.startswith("arg:"):
        tool, _, rest = r[4:].partition(".")
        return "arg:" + tool + "." + rest
    return r


def main():
    work = os.environ["WORK"]
    recs = load(os.path.join(work, "prep", "tokens.jsonl.gz"))
    tab = os.path.join(work, "tables")
    os.makedirs(tab, exist_ok=True)
    with Pool(int(os.environ.get("JOBS", "6"))) as pool:
        res = pool.map(analyze_request, recs, chunksize=4)
    print("analysed %d responses, %d generated tokens" % (len(res), sum(r["n"] for r in res)), file=sys.stderr)

    scopes = {"all": lambda m: True}
    for arm in sorted({(r["meta"]["run"], r["meta"]["arm"]) for r in res}):
        scopes["%s/%s" % arm] = (lambda a: lambda m: (m["run"], m["arm"]) == a)(arm)

    # ---- per request -------------------------------------------------------------------------------------------
    rows = []
    for r in res:
        m = r["meta"]
        row = [m["run"], m["arm"], m["task"], m["attempt"], m["try_"], m["turn"], m["id"], m["finish"], r["n"]]
        for v in VARIANTS:
            for ml in MIN_LENS:
                row.append(sum(r["cov"][v][ml].values()))
        row += [r["sim"]["own"][(16, 4)][1], r["sim"]["own"][(16, 4)][0]]
        rows.append(row)
    write_csv(os.path.join(tab, "per_request.csv"),
              ["run", "arm", "task", "attempt", "try", "turn", "id", "finish", "tokens"]
              + ["covered_%s_L%d" % (v, ml) for v in VARIANTS for ml in MIN_LENS] + ["sim_tokens_n16_m4", "sim_steps_n16_m4"],
              rows)

    # ---- share of tokens in copy spans -------------------------------------------------------------------------
    out = []
    for sc, pick in scopes.items():
        sel = [r for r in res if pick(r["meta"])]
        for v in VARIANTS:
            for ml in MIN_LENS:
                per = defaultdict(lambda: [0, 0])
                for r in sel:
                    s = per[r["meta"]["session"]]
                    s[0] += sum(r["cov"][v][ml].values())
                    s[1] += r["n"]
                cov = sum(x[0] for x in per.values())
                tot = sum(x[1] for x in per.values())
                lo, hi = boot(per, lambda t: t[0] / t[1]) if len(per) > 1 else (float("nan"),) * 2
                out.append([sc, v, ml, len(sel), len(per), tot, cov, "%.4f" % (cov / tot), "%.4f" % lo, "%.4f" % hi])
    write_csv(os.path.join(tab, "share_overall.csv"),
              ["scope", "variant", "min_len", "responses", "sessions", "tokens", "copied_tokens", "share", "ci95_lo",
               "ci95_hi"], out)

    # per response distribution (all scope): share quantiles
    out = []
    for v in VARIANTS:
        for ml in MIN_LENS:
            shares = sorted(sum(r["cov"][v][ml].values()) / r["n"] for r in res if r["n"])
            out.append([v, ml] + ["%.3f" % quantile(shares, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                       + ["%.3f" % (sum(1 for s in shares if s >= 0.15) / len(shares)),
                          "%.3f" % (sum(1 for s in shares if s >= 0.5) / len(shares))])
    write_csv(os.path.join(tab, "share_per_response.csv"),
              ["variant", "min_len", "p10", "p25", "median", "p75", "p90", "responses_with_share_ge_0.15",
               "responses_with_share_ge_0.5"], out)

    # ---- by region ---------------------------------------------------------------------------------------------
    tot_reg = Counter()
    for r in res:
        tot_reg.update(r["regions"])
    out = []
    for v in VARIANTS:
        for ml in MIN_LENS:
            cov_reg = Counter()
            for r in res:
                cov_reg.update(r["cov"][v][ml])
            cov_all = sum(cov_reg.values())
            for reg in sorted(tot_reg, key=lambda x: -tot_reg[x]):
                out.append([v, ml, reg, tot_reg[reg], "%.4f" % (tot_reg[reg] / sum(tot_reg.values())), cov_reg[reg],
                            "%.4f" % (cov_reg[reg] / tot_reg[reg]), "%.4f" % (cov_reg[reg] / cov_all if cov_all else 0)])
            for name, pred in (("ALL tool-call arguments", lambda x: x.startswith("arg:")),
                               ("ALL file-edit arguments (edit.*)", lambda x: x.startswith("arg:edit."))):
                t_g = sum(tot_reg[x] for x in tot_reg if pred(x))
                c_g = sum(cov_reg[x] for x in tot_reg if pred(x))
                out.append([v, ml, name, t_g, "%.4f" % (t_g / sum(tot_reg.values())), c_g, "%.4f" % (c_g / t_g),
                            "%.4f" % (c_g / cov_all if cov_all else 0)])
    write_csv(os.path.join(tab, "share_by_region.csv"),
              ["variant", "min_len", "region", "tokens", "share_of_output", "copied_tokens", "share_copied",
               "share_of_all_copied"], out)

    out = []
    for v in VARIANTS:
        for ml in MIN_LENS:
            cov_reg = Counter()
            for r in res:
                cov_reg.update(r["cov"][v][ml])
            t_all, t_struct = sum(tot_reg.values()), tot_reg["struct"]
            c_all, c_struct = sum(cov_reg.values()), cov_reg["struct"]
            out.append([v, ml, t_all - t_struct, c_all - c_struct, "%.4f" % ((c_all - c_struct) / (t_all - t_struct))])
    write_csv(os.path.join(tab, "share_without_scaffolding.csv"),
              ["variant", "min_len", "tokens_without_call_scaffolding", "copied_tokens", "share"], out)

    # ---- span lengths and starts ---------------------------------------------------------------------------------
    out_len, out_start = [], []
    for v in VARIANTS:
        for ml in MIN_LENS:
            spans = [s for r in res for s in r["spans"][v][ml]]
            lens = sorted(s[0] for s in spans)
            n_sp, n_tok = len(lens), sum(lens)
            for lo, hi in zip(BUCKETS, BUCKETS[1:]):
                if hi <= ml:
                    continue
                b = [x for x in lens if lo <= x < hi]
                out_len.append([v, ml, "%d-%s" % (lo, "" if hi > 10 ** 8 else hi - 1) if hi < 10 ** 8 else "%d+" % lo,
                                len(b), "%.4f" % (len(b) / n_sp), sum(b), "%.4f" % (sum(b) / n_tok)])
            tw = sum(x * x for x in lens) / n_tok
            out_len.append([v, ml, "summary", n_sp, "mean %.1f" % (n_tok / n_sp), "median %d" % quantile(lens, 0.5),
                            "p90 %d, token-weighted mean %.1f, max %d" % (quantile(lens, 0.9), tw, lens[-1])])
            kinds = Counter()
            for ln, reg, kind, dist in spans:
                cls = ("none" if kind == "none" else "0" if dist == 0 else "1-2" if dist <= 2 else "3-5" if dist <= 5
                       else "6-10" if dist <= 10 else "11+")
                kinds[(reg, kind, cls, 0)] += 1
                kinds[(reg, kind, cls, 1)] += ln
            for (rg, k, cls, which), val in sorted(kinds.items()):
                if which == 0:
                    out_start.append([v, ml, rg, k, cls, val, kinds[(rg, k, cls, 1)]])
    write_csv(os.path.join(tab, "span_lengths.csv"),
              ["variant", "min_len", "span_length", "spans", "share_of_spans", "tokens", "share_of_copied_tokens"], out_len)
    write_csv(os.path.join(tab, "span_starts.csv"),
              ["variant", "min_len", "start_region", "preceding_marker", "tokens_after_marker", "spans", "tokens_in_spans"], out_start)

    # ---- the guesser -------------------------------------------------------------------------------------------
    gout, rout = [], []
    for sc, pick in scopes.items():
        sel = [r for r in res if pick(r["meta"])]
        for v in ("own", "own+esc", "oracle", "occ-oracle"):
            for n in DRAFT_N:
                for m in match_values(v):
                    per = defaultdict(lambda: [0] * 6)
                    for r in sel:
                        s = per[r["meta"]["session"]]
                        for k, x in enumerate(r["sim"][v][(n, m)][:6]):
                            s[k] += x
                    t = [sum(x[k] for x in per.values()) for k in range(6)]
                    steps, tokens, props, acc, proposed, cols = t
                    row = [sc, v, n, m, steps, tokens, "%.3f" % (tokens / steps), "%.3f" % (props / steps),
                           "%.2f" % (acc / props if props else 0), "%.3f" % (acc / steps),
                           "%.2f" % (proposed / props if props else 0), "%.2f" % (cols / steps),
                           "%.3f" % (acc / proposed if proposed else 0), "%.3f" % (acc / tokens)]
                    for cn, cm in COSTS.items():
                        sp = lambda t_, r_=cm["r"]: (t_[1] / t_[0]) / (1 + r_ * (t_[5] / t_[0] - 1))
                        row.append("%.3f" % sp(t))
                        if sc == "all" and len(per) > 1 and (n in (8, 16, 32) and m in (4, 8)):
                            lo, hi = boot(per, sp)
                            row += ["%.3f" % lo, "%.3f" % hi]
                        else:
                            row += ["", ""]
                    gout.append(row)
            if sc == "all":
                for n in (8, 16, 32):
                    reg = Counter()
                    for r in sel:
                        reg.update(r["sim"][v][(n, 4)][6])
                    tot_a = sum(reg.values())
                    for k, val in reg.most_common():
                        rout.append([v, n, 4, k, val, "%.4f" % (val / tot_a)])
    write_csv(os.path.join(tab, "guesser.csv"),
              ["scope", "variant", "draft_tokens", "min_match", "steps", "tokens", "tokens_per_step",
               "steps_with_draft", "accepted_per_draft_step", "accepted_per_step", "drafted_per_draft_step",
               "columns_per_step", "draft_acceptance", "share_of_tokens_from_drafts",
               "speedup_today", "ci95_lo", "ci95_hi", "speedup_kernels", "ci95_lo", "ci95_hi"], gout)
    write_csv(os.path.join(tab, "guesser_by_region.csv"),
              ["variant", "draft_tokens", "min_match", "region", "accepted_tokens", "share_of_accepted"], rout)

    # ---- with DFlash2 ----------------------------------------------------------------------------------------------
    taus = [float(x) for x in os.environ.get("TAU_OUT", "2.56").split(",")]
    cout = []
    for v in ("own", "own+esc", "oracle", "occ-oracle"):
        for n in DRAFT_N:
            for m in match_values(v):
                per = defaultdict(lambda: [0] * 6)
                for r in res:
                    s = per[r["meta"]["session"]]
                    for k, x in enumerate(r["sim"][v][(n, m)][:6]):
                        s[k] += x
                steps, tokens, props, acc, proposed, cols = [sum(x[k] for x in per.values()) for k in range(6)]
                if not props:
                    continue
                # steps that are not copy drafts become DFlash2 steps at tau tokens
                n_all = tokens
                copy_tokens = acc + props           # tokens from draft steps (accepted plus the model's own)
                rest = n_all - copy_tokens
                plain_steps = steps - props
                for tau in taus:
                    d_steps = (n_all - copy_tokens) / tau
                    st = props + d_steps
                    row = [v, n, m, tau, "%.3f" % (n_all / st), "%.2f" % ((cols - plain_steps + d_steps * DFLASH_COLS) / st)]
                    for cn, cm in COSTS.items():
                        cost_step = (props * (1 + cm["r"] * ((cols - plain_steps) / props - 1))
                                     + d_steps * (1 + cm["r"] * (DFLASH_COLS - 1) + cm["draft"])) / st
                        base = (DFLASH_TOKENS / (1 + cm["r"] * (DFLASH_COLS - 1) + cm["draft"]))   # DFlash2 alone
                        row += ["%.3f" % ((n_all / st) / cost_step), "%.3f" % ((n_all / st) / cost_step / base)]
                    cout.append(row)
    write_csv(os.path.join(tab, "combined.csv"),
              ["variant", "draft_tokens", "min_match", "dflash2_tokens_per_step_outside_copies", "tokens_per_pass",
               "columns_per_pass", "speedup_over_plain_today", "gain_over_dflash2_today", "speedup_over_plain_kernels",
               "gain_over_dflash2_kernels"], cout)
    json.dump({"tau_out": taus, "dflash_tokens": DFLASH_TOKENS, "dflash_columns": DFLASH_COLS, "costs": COSTS},
              open(os.path.join(tab, "model.json"), "w"), indent=1)

    vrows = []
    for v in ("ctx", "own"):
        grp = defaultdict(list)
        for r in res:
            for x in r["values"][v]:
                grp[x[0]].append(x)
                grp["all argument values"].append(x)
        for reg, xs in sorted(grp.items(), key=lambda kv: -len(kv[1])):
            n_v = len(xs)
            vrows.append([v, reg, n_v, "%.1f" % (sum(x[1] for x in xs) / n_v), "%.3f" % (sum(x[2] for x in xs) / sum(x[1] for x in xs)),
                          "%.3f" % (sum(1 for x in xs if x[2]) / n_v), "%.3f" % (sum(1 for x in xs if x[2] >= 0.9 * x[1]) / n_v),
                          "%.3f" % (sum(1 for x in xs if x[4]) / n_v),
                          "%.3f" % (sum(1 for x in xs if x[3] == 0 and not x[4]) / n_v),
                          "%.3f" % (sum(1 for x in xs if 1 <= x[3] <= 2) / n_v),
                          "%.3f" % (sum(1 for x in xs if x[3] >= 3) / n_v), "%.3f" % (sum(1 for x in xs if x[3] < 0) / n_v)])
    write_csv(os.path.join(tab, "argument_values.csv"),
              ["variant", "argument", "values", "mean_tokens", "share_of_tokens_copied", "values_with_a_copy_span",
               "values_90pct_copied", "copy_runs_into_value", "copy_starts_at_first_token", "copy_starts_at_token_2_3",
               "copy_starts_later", "no_copy"], vrows)

    # ---- how far a match's continuation holds ----------------------------------------------------------------------
    tot = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in res:
        for (b, inside), x in r["chains"].items():
            for key in ((b, inside), (b, "any")):
                for k in range(5):
                    tot[key][k] += x[k]
    rows = []
    for (b, inside), x in sorted(tot.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        lo, hi = CHAIN_BUCKETS[b]
        label = "%d-%d" % (lo, hi) if hi < 10 ** 8 else "%d+" % lo
        where = {True: "inside a copy span", False: "outside", "any": "anywhere"}[inside]
        rows.append([label, where, x[0], "%.3f" % (x[1] / x[0]), "%.2f" % (x[2] / x[0]), "%.3f" % (x[3] / x[0]),
                     "%.3f" % (x[4] / x[0])])
    write_csv(os.path.join(tab, "continuation_by_match_length.csv"),
              ["match_length", "position", "positions", "next_token_right", "mean_tokens_right_in_a_row_cap32",
               "share_right_for_8", "share_right_for_16"], rows)

    # ---- by task: share per task (all arms) -------------------------------------------------------------------
    per_task = defaultdict(lambda: [0, 0, 0])
    for r in res:
        t = per_task[r["meta"]["task"]]
        t[0] += r["n"]
        t[1] += sum(r["cov"]["own"][8].values())
        t[2] += sum(r["cov"]["ctx"][8].values())
    write_csv(os.path.join(tab, "share_by_task.csv"),
              ["task", "tokens", "copied_own_L8", "share_own_L8", "copied_ctx_L8", "share_ctx_L8"],
              [[k, t[0], t[1], "%.4f" % (t[1] / t[0]), t[2], "%.4f" % (t[2] / t[0])] for k, t in sorted(per_task.items())])

    per_turn = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in res:
        t = per_turn[min(r["meta"]["turn"], 8)]
        t[0] += 1
        t[1] += r["n"]
        t[2] += sum(r["cov"]["own"][8].values())
        t[3] += sum(r["cov"]["ctx"][8].values())
        t[4] += r["regions"]["reasoning"]
    write_csv(os.path.join(tab, "share_by_turn.csv"),
              ["turn", "responses", "tokens", "share_own_L8", "share_ctx_L8", "reasoning_share_of_tokens"],
              [[("%d+" % k if k == 8 else k), t[0], t[1], "%.4f" % (t[2] / t[1]), "%.4f" % (t[3] / t[1]), "%.4f" % (t[4] / t[1])]
               for k, t in sorted(per_turn.items())])
    print("tables in " + tab, file=sys.stderr)


if __name__ == "__main__":
    main()
