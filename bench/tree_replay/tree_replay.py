#!/usr/bin/env python3
"""Offline tree-value replay of speculative verify steps dumped by llama-server (LLAMA_SPEC_DUMP, Q688).

    python3 tree_replay.py DUMP.jsonl[.gz] [...] --out DIR [--budgets 3,4,6,8,12,16] [--r 0.08,0.10,0.12,0.40]
                           [--draft-cost 0.22] [--mode markov|factorized] [--copy-min 4]
                           [--ctx-bins 4096,8192,16384,32768]
    python3 tree_replay.py --check DUMP.jsonl[.gz] [...]
    python3 tree_replay.py --selftest

Each dump line is one verify step of the server's speculative loop: the position of the last sampled token, the drafted
tokens, the target's tokens (the accepted drafts, then its own), the drafter's candidates per position (for DFlash2 its
selector lattice: conditional probabilities from each candidate of the previous position), the target's most probable
tokens at the sampled positions, and the longest earlier match of the context's suffix with the tokens that followed it.

The truth for a step is the target's own continuation after it: the tokens of that request's later steps, rebuilt by
position (a line may also carry it as "truth"). The server verifies greedy drafts by sample-and-match: it samples each
position from the target and keeps a drafted token while the sample equals it. A token tree verified the same way
accepts the longest root path that matches the target's continuation, so every node in it is worth 1 if its whole path
matches the truth and 0 otherwise, and the accepted length of a tree is 1 plus its accepted nodes. A step whose truth
is not known to the lattice's depth (the end of a request) is left out.

Trees are built best-first by path probability, as DDTree does: the B nodes with the highest path probability, which
always form a tree. "markov" multiplies the lattice's conditional rows along the path (DFlash2's own model);
"factorized" multiplies per-position marginals, as DDTree does with DFlash's marginals, and is the only choice for
drafters that record marginals given their drafted prefix. The chain is the drafter's own walk (its greedy path),
truncated to the budget.

Output in DIR: summary.md (all tables) and chain.csv, budgets.csv, ranks.csv, calibration.csv, by_ctx.csv,
by_content.csv, check.txt. Standard library only.
"""
import argparse
import gzip
import heapq
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_BUDGETS = [3, 4, 6, 8, 12, 16]
DEFAULT_R = [0.08, 0.10, 0.12, 0.40]


# ---------------------------------------------------------------- loading


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_steps(paths):
    """All dump lines, each tagged with its file, in file order."""
    steps = []
    for fi, p in enumerate(paths):
        with open_text(p) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{p}:{ln}: not JSON ({e})")
                s["_file"] = fi
                steps.append(s)
    return steps


def attach_truth(steps, depth_cap=64):
    """Rebuild each request's token sequence by position and give every step its continuation (step["_y"]).

    Returns a list of problems found while rebuilding (two steps that disagree about a position)."""
    problems = []
    groups = defaultdict(list)
    for s in steps:
        groups[(s["_file"], s.get("run"), s.get("task"), s.get("slot"))].append(s)
    for key, g in groups.items():
        g.sort(key=lambda s: s.get("step", 0))
        tok = {}
        for s in g:
            seq = [s["last"]] + list(s["sampled"])
            for j, t in enumerate(seq):
                p = s["pos"] + j
                if p in tok and tok[p] != t:
                    problems.append(f"task {key[2]}: position {p} is {tok[p]} in one step and {t} in step {s.get('step')}")
                tok[p] = t
        for s in g:
            if "truth" in s:
                s["_y"] = list(s["truth"])
                continue
            y = []
            p = s["pos"] + 1
            while p in tok and len(y) < depth_cap:
                y.append(tok[p])
                p += 1
            s["_y"] = y
        s_prev = None
        for s in g:
            if s_prev is not None:
                s["_gap"] = s["pos"] != s_prev["pos"] + len(s_prev["sampled"])
            s_prev = s
    return problems


# ---------------------------------------------------------------- one step's lattice


class Lattice:
    """One step's candidates: ids[i] per position, conditional rows (markov) or per-position probabilities."""

    def __init__(self, s, mode):
        self.ids = [list(c) for c in s.get("cand", [])]
        self.D = len(self.ids)
        self.lattice = bool(s.get("lattice", False))
        self.pick = list(s.get("pick", [-1] * self.D))
        p = s.get("p", [])
        self.K = [len(c) for c in self.ids]
        if self.lattice:
            self.rows = p  # p[0]: K0 values, p[i]: K[i-1] * K[i]
            # forward marginals over the kept candidates
            m = [list(p[0])] if self.D else []
            for i in range(1, self.D):
                Ki, Kp = self.K[i], self.K[i - 1]
                m.append([sum(m[i - 1][a] * p[i][a * Ki + b] for a in range(Kp)) for b in range(Ki)])
            self.marg = m
        else:
            self.rows = None
            self.marg = [list(x) for x in p]
        self.mode = "markov" if (mode == "markov" and self.lattice) else "factorized"

    def cond(self, depth, parent_b, b):
        """P(candidate b at depth | the path's candidate parent_b at depth - 1)."""
        if self.mode == "markov":
            if depth == 0:
                return self.rows[0][b]
            return self.rows[depth][parent_b * self.K[depth] + b]
        return self.marg[depth][b]

    def chain(self):
        """The drafter's walk as candidate indices, up to where it stopped."""
        out = []
        for i in range(self.D):
            if i >= len(self.pick) or self.pick[i] is None or self.pick[i] < 0:
                break
            out.append(self.pick[i])
        return out


def best_first(lat, budget):
    """The budget nodes with the highest path probability, in the order a best-first expansion adds them.

    Each node is (depth, candidate index, parent node index or -1, path probability, on the drafter's walk)."""
    walk = lat.chain()
    nodes = []
    heap = []
    tie = 0
    if lat.D == 0:
        return nodes
    for b in range(lat.K[0]):
        pr = lat.cond(0, -1, b)
        heapq.heappush(heap, (-pr, tie, 0, b, -1, len(walk) > 0 and walk[0] == b))
        tie += 1
    while heap and len(nodes) < budget:
        neg, _, d, b, parent, on = heapq.heappop(heap)
        pr = -neg
        idx = len(nodes)
        nodes.append((d, b, parent, pr, on))
        if d + 1 < lat.D:
            for c in range(lat.K[d + 1]):
                q = pr * lat.cond(d + 1, b, c)
                if q <= 0.0:
                    continue
                child_on = on and len(walk) > d + 1 and walk[d + 1] == c
                heapq.heappush(heap, (-q, tie, d + 1, c, idx, child_on))
                tie += 1
    return nodes


# ---------------------------------------------------------------- per-step scoring


def score_step(s, lat, budgets, tgt):
    """Per step: chain acceptance by position, the best-first tree's node values, the n-gram chain."""
    y = s["_y"]
    out = {"pos": s["pos"], "ng_len": s.get("ng_len", 0), "n_acc": s["n_acc"], "n_draft": len(s["draft"])}
    # the drafter's walk
    walk = lat.chain()
    acc = 0
    for i, b in enumerate(walk):
        if i < len(y) and lat.ids[i][b] == y[i]:
            acc += 1
        else:
            break
    out["chain_acc"] = acc
    out["walk_len"] = len(walk)
    # the walk's path probabilities, for a chain cut where they fall below a threshold (Q687)
    wp, pr, prev_b = [], 1.0, -1
    for i, b in enumerate(walk):
        pr *= lat.cond(i, prev_b, b)
        wp.append(pr)
        prev_b = b
    out["walk_p"] = wp
    # the tree
    B = max(budgets)
    nodes = best_first(lat, B)
    accepted = []
    expected = []
    n_acc_live = s["n_acc"]
    tgt_ids = s.get("tgt_ids") or []
    tgt_p = s.get("tgt_p") or []
    for (d, b, parent, pr, on) in nodes:
        par_ok = parent < 0 or accepted[parent]
        a = 1 if (par_ok and d < len(y) and lat.ids[d][b] == y[d]) else 0
        accepted.append(a)
        # the target's distribution at depth d is known when the path to the parent is the accepted chain prefix
        if tgt and d < len(tgt_ids) and (parent < 0 or (nodes[parent][4] and nodes[parent][0] < n_acc_live)):
            tok = lat.ids[d][b]
            e = 0.0
            for t, q in zip(tgt_ids[d], tgt_p[d]):
                if t == tok:
                    e = q
                    break
            expected.append(e)
        else:
            expected.append(float(a))
    out["nodes"] = [(n[0], n[3], n[4], accepted[i], expected[i]) for i, n in enumerate(nodes)]
    # the copy drafter's chain: the tokens that followed the context's longest earlier match
    ng = s.get("ng") or []
    ng_acc = 0
    for i, t in enumerate(ng):
        if i < len(y) and t == y[i]:
            ng_acc += 1
        else:
            break
    out["ng_acc"] = ng_acc
    out["ng_n"] = len(ng)
    out["y_len"] = len(y)
    return out


# ---------------------------------------------------------------- the analysis


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def tau_tree(rec, B):
    return 1 + sum(n[3] for n in rec["nodes"][:B])


def tau_chain(rec, B):
    return 1 + min(rec["chain_acc"], B)


def tau_ng(rec, B):
    return 1 + min(rec["ng_acc"], B)


# the columns a draft of budget B verifies beyond the anchor: a chain or an n-gram run can be shorter than B
def cols_tree(rec, B):
    return min(B, len(rec["nodes"]))


def cols_chain(rec, B):
    return min(B, rec["walk_len"])


def cols_ng(rec, B):
    return min(B, rec["ng_n"])


def analyse(steps, budgets, mode, rs, draft_cost, copy_min, ctx_bins, tgt=True):
    """Every table, as lists of dict rows, plus the consistency report."""
    report = {"n_lines": len(steps)}
    recs = []
    n_short = 0
    n_nocand = 0
    types = defaultdict(int)
    for s in steps:
        types[s.get("type") or "none"] += 1
        if not s.get("cand"):
            n_nocand += 1
            continue
        lat = Lattice(s, mode)
        if len(s["_y"]) < lat.D:
            n_short += 1
            continue
        recs.append(score_step(s, lat, budgets, tgt))
    report.update(n_scored=len(recs), n_short=n_short, n_nocand=n_nocand, types=dict(types),
                  n_lattice=sum(1 for s in steps if s.get("lattice")))
    if not recs:
        return report, {}
    D = max(r["walk_len"] for r in recs)
    tables = {}

    # chain acceptance by position
    rows = []
    prev = None
    for j in range(1, D + 1):
        have = [r for r in recs if r["walk_len"] >= j]
        a = mean(1 if r["chain_acc"] >= j else 0 for r in have)
        live = [r for r in recs if r["n_draft"] >= j]
        a_live = mean(1 if r["n_acc"] >= j else 0 for r in live) if live else float("nan")
        rows.append({"position": j, "steps": len(have), "cumulative": a, "conditional": a / prev if prev else a,
                     "verified_steps": len(live), "verified_cumulative": a_live})
        prev = a
    tables["chain"] = rows

    # budgets
    rows = []
    for B in budgets:
        tt = [tau_tree(r, B) for r in recs]
        tc = [tau_chain(r, B) for r in recs]
        n_off = [sum(1 for n in r["nodes"][:B] if not n[2]) for r in recs]
        acc_off = [sum(n[3] for n in r["nodes"][:B] if not n[2]) for r in recs]
        row = {"budget": B, "tau_chain": mean(tc), "tau_tree": mean(tt), "tree_minus_chain": mean(tt) - mean(tc),
               "se_tree_minus_chain": se([a - b for a, b in zip(tt, tc)]),
               "off_chain_nodes": mean(n_off),
               "value_per_off_chain_node": (sum(acc_off) / sum(n_off)) if sum(n_off) else float("nan")}
        cc = mean(cols_chain(r, B) for r in recs)
        ct = mean(cols_tree(r, B) for r in recs)
        for r_ in rs:
            row[f"S_chain_r{r_:g}"] = mean(tc) / (1.0 + draft_cost + cc * r_)
            row[f"S_tree_r{r_:g}"] = mean(tt) / (1.0 + draft_cost + ct * r_)
        rows.append(row)
    for i, row in enumerate(rows):
        if i == 0:
            row["dtau_per_node"] = float("nan")
        else:
            dB = row["budget"] - rows[i - 1]["budget"]
            row["dtau_per_node"] = (row["tau_tree"] - rows[i - 1]["tau_tree"]) / dB
    tables["budgets"] = rows

    # node values by best-first rank
    rows = []
    Bmax = max(budgets)
    for k in range(Bmax):
        have = [r["nodes"][k] for r in recs if len(r["nodes"]) > k]
        off = [n for n in have if not n[2]]
        rows.append({"rank": k + 1, "steps": len(have), "value": mean(n[3] for n in have),
                     "value_se": se(n[3] for n in have), "value_expected": mean(n[4] for n in have),
                     "predicted": mean(n[1] for n in have), "off_chain_share": len(off) / len(have) if have else float("nan"),
                     "off_chain_value": mean(n[3] for n in off) if off else float("nan"),
                     "mean_depth": mean(n[0] + 1 for n in have)})
    tables["ranks"] = rows

    # calibration of the path probability
    edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    bins = defaultdict(list)
    for r in recs:
        for n in r["nodes"]:
            for lo, hi in zip(edges, edges[1:]):
                if lo <= n[1] < hi:
                    bins[(lo, hi, n[2])].append(n)
                    break
    rows = []
    for (lo, hi, on) in sorted(bins):
        ns = bins[(lo, hi, on)]
        rows.append({"p_low": lo, "p_high": min(hi, 1.0), "on_walk": int(on), "nodes": len(ns),
                     "predicted": mean(n[1] for n in ns), "value": mean(n[3] for n in ns)})
    tables["calibration"] = rows

    # by context length and by content
    def split_rows(key_fn, name):
        groups = defaultdict(list)
        for r in recs:
            groups[key_fn(r)].append(r)
        out = []
        for key in sorted(groups):
            g = groups[key]
            row = {name: key, "steps": len(g), "chain_pos1": mean(1 if r["chain_acc"] >= 1 else 0 for r in g)}
            for B in (3, 8, 16):
                if B > Bmax:
                    continue
                row[f"tau_chain_{B}"] = mean(tau_chain(r, B) for r in g)
                row[f"tau_tree_{B}"] = mean(tau_tree(r, B) for r in g)
                row[f"tau_ngram_{B}"] = mean(tau_ng(r, B) for r in g)
            for k in (4, 8):
                vals = [r["nodes"][k - 1][3] for r in g if len(r["nodes"]) >= k]
                row[f"value_rank{k}"] = mean(vals)
            out.append(row)
        return out

    def ctx_key(r):
        lo = 0
        for e in ctx_bins:
            if r["pos"] < e:
                return f"{lo}-{e - 1}"
            lo = e
        return f">={lo}"

    def ctx_sort_key(r):
        return r["pos"]

    by_ctx = split_rows(ctx_key, "context")
    # keep the bins in numeric order
    order = {}
    for r in sorted(recs, key=ctx_sort_key):
        order.setdefault(ctx_key(r), len(order))
    by_ctx.sort(key=lambda row: order[row["context"]])
    tables["by_ctx"] = by_ctx
    tables["by_content"] = split_rows(lambda r: "copy" if r["ng_len"] >= copy_min else "fresh", "content")

    # copy-gated: the n-gram chain where the context's suffix matches for copy_min tokens or more, else the tree
    rows = []
    for B in budgets:
        tok = [tau_ng(r, B) if r["ng_len"] >= copy_min else tau_tree(r, B) for r in recs]
        row = {"budget": B, "tau": mean(tok)}
        for r_ in rs:
            num = sum(tok)
            den = sum((1.0 + cols_ng(r, B) * r_) if r["ng_len"] >= copy_min
                      else (1.0 + draft_cost + cols_tree(r, B) * r_) for r in recs)
            row[f"S_r{r_:g}"] = num / den
        rows.append(row)
    tables["copy_gated"] = rows

    # the whole budget curve, for the best budget of each kind at each r
    rows = []
    for B in range(1, Bmax + 1):
        row = {"budget": B}
        for kind, tau, cols, dc in (("chain", tau_chain, cols_chain, draft_cost), ("tree", tau_tree, cols_tree, draft_cost)):
            t = mean(tau(r, B) for r in recs)
            c = mean(cols(r, B) for r in recs)
            row[f"tau_{kind}"] = t
            for r_ in rs:
                row[f"S_{kind}_r{r_:g}"] = t / (1.0 + dc + c * r_)
        rows.append(row)
    tables["curve"] = rows

    # the chain cut where the walk's path probability falls below t (the draft length chosen per step, Q687)
    rows = []
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        L = [sum(1 for q in r["walk_p"] if q >= t) for r in recs]
        tau = [1 + min(r["chain_acc"], n) for r, n in zip(recs, L)]
        row = {"threshold": t, "tau": mean(tau), "columns": mean(L), "no_draft_share": mean(1 if n == 0 else 0 for n in L)}
        for r_ in rs:
            # a step that drafts nothing pays no draft pass only if the pass is skipped, which needs the confidence
            # before drafting; here every step pays it
            row[f"S_r{r_:g}"] = mean(tau) / (1.0 + draft_cost + mean(L) * r_)
        rows.append(row)
    tables["adaptive"] = rows
    report["copy_share"] = mean(1 if r["ng_len"] >= copy_min else 0 for r in recs)
    report["live_tau"] = mean(r["n_acc"] + 1 for r in recs)
    report["live_draft"] = mean(r["n_draft"] for r in recs)
    return report, tables


def best_s(tables, rs, kind):
    out = {}
    for r_ in rs:
        best = max(tables["curve"], key=lambda row: row[f"S_{kind}_r{r_:g}"])
        out[r_] = (best["budget"], best[f"S_{kind}_r{r_:g}"])
    return out


# ---------------------------------------------------------------- checks


def check(steps):
    """Consistency of the dump with itself; returns (ok, lines of text)."""
    lines = []
    bad = 0
    problems = attach_truth(steps)
    for p in problems[:20]:
        lines.append("conflict: " + p)
    bad += len(problems)
    n_gap = sum(1 for s in steps if s.get("_gap"))
    n_eog = 0
    for s in steps:
        d, smp, n = s["draft"], s["sampled"], s["n_acc"]
        if len(smp) != n + 1 or n > len(d):
            lines.append(f"step {s.get('step')}: sampled has {len(smp)} tokens for n_acc {n} and {len(d)} drafted")
            bad += 1
            continue
        if d[:n] != smp[:n]:
            lines.append(f"step {s.get('step')}: the accepted drafts are not the target's tokens")
            bad += 1
        if n < len(d) and d[n] == smp[n]:
            n_eog += 1  # sample-and-match stops after an end-of-generation token even when it matches
        y = s["_y"]
        if s.get("verify", "match") == "match":
            m = 0
            while m < len(d) and m < len(y) and d[m] == y[m]:
                m += 1
            if m != n and not (n < len(d) and d[n] == smp[n]):
                lines.append(f"step {s.get('step')}: n_acc {n}, the drafts match the rebuilt continuation for {m}")
                bad += 1
        cand, pick = s.get("cand"), s.get("pick")
        if cand:
            for i, t in enumerate(d):
                if i >= len(cand) or i >= len(pick) or pick[i] < 0 or cand[i][pick[i]] != t:
                    lines.append(f"step {s.get('step')}: drafted token {i} is not the walk's candidate")
                    bad += 1
                    break
            if s.get("lattice"):
                K = [len(c) for c in cand]
                p = s["p"]
                if len(p[0]) != K[0] or any(len(p[i]) != K[i - 1] * K[i] for i in range(1, len(K))):
                    lines.append(f"step {s.get('step')}: lattice rows do not match the candidate counts")
                    bad += 1
        if s.get("tgt_ps") is not None and len(s["tgt_ps"]) not in (0, len(smp)):
            lines.append(f"step {s.get('step')}: {len(s['tgt_ps'])} target rows for {len(smp)} sampled tokens")
            bad += 1
    lines.insert(0, f"{len(steps)} steps, {bad} problems, {n_gap} gaps between consecutive steps of a request, "
                    f"{n_eog} stops at an end-of-generation token")
    return bad == 0, lines


# ---------------------------------------------------------------- output


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"
    return str(x)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for c in r:
            if c not in cols:
                cols.append(c)
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r.get(c, "")) for c in cols) + "\n")


def md_table(rows, cols=None, names=None):
    if not rows:
        return "(no rows)\n"
    cols = cols or list(rows[0].keys())
    names = names or cols
    out = "| " + " | ".join(names) + " |\n|" + "|".join("---:" for _ in cols) + "|\n"
    for r in rows:
        out += "| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |\n"
    return out


def summary_md(report, tables, args, rs):
    s = []
    s.append("# Tree-value replay\n")
    s.append(f"Lines {report['n_lines']} ({', '.join(f'{k} {v}' for k, v in report['types'].items())}); scored "
             f"{report['n_scored']}; left out: {report['n_short']} without the continuation to the lattice's depth, "
             f"{report['n_nocand']} without candidates. Mode {args.mode} ({report['n_lattice']} steps carry a DFlash2 "
             f"lattice; the others are factorized); draft pass {args.draft_cost:g} of a one-token step; verified live: "
             f"{fmt(report.get('live_draft', float('nan')))} drafted, {fmt(report.get('live_tau', float('nan')))} "
             f"tokens per step. Copy share (context match of {args.copy_min}+ tokens): "
             f"{fmt(report.get('copy_share', float('nan')))}.\n")
    if not tables:
        return "\n".join(s)
    s.append("## Chain acceptance by position (the drafter's walk against the continuation)\n")
    s.append(md_table(tables["chain"]))
    s.append("\n## Budgets: accepted tokens per step (bonus included) and speed-up over one-token decoding\n")
    s.append("S = tau / (1 + draft pass + budget x r), r the cost of one more verified token in one-token steps.\n")
    s.append(md_table(tables["budgets"]))
    bc, bt = best_s(tables, rs, "chain"), best_s(tables, rs, "tree")
    s.append("\n| r | best chain (budget, S) | best tree (budget, S) | tree / chain |\n|---:|---|---|---:|")
    for r_ in rs:
        s.append(f"| {r_:g} | {bc[r_][0]}, {bc[r_][1]:.3f} | {bt[r_][0]}, {bt[r_][1]:.3f} | {bt[r_][1] / bc[r_][1]:.3f} |")
    small = [r for r in tables["budgets"] if r["budget"] <= 8 and not math.isnan(r["value_per_off_chain_node"])]
    if small:
        top = max(small, key=lambda r: r["value_per_off_chain_node"])
        s.append(f"\nQ614's gate (a node off the drafter's walk must add at least 0.15 tokens at a budget of 8 or "
                 f"fewer): the best is {top['value_per_off_chain_node']:.3f} per off-walk node at budget "
                 f"{top['budget']} ({top['off_chain_nodes']:.2f} such nodes per step), "
                 f"{'met' if top['value_per_off_chain_node'] >= 0.15 else 'not met'}.\n")
    s.append("\n## Node values by best-first rank (the B-th node's value is the tree's gain from budget B - 1 to B)\n")
    s.append(md_table(tables["ranks"]))
    s.append("\n## Calibration of the path probability (all nodes of the largest budget)\n")
    s.append(md_table(tables["calibration"]))
    s.append("\n## By context length (position of the step)\n")
    s.append(md_table(tables["by_ctx"]))
    s.append(f"\n## By content (copy: the context's suffix occurs earlier for {args.copy_min}+ tokens)\n")
    s.append(md_table(tables["by_content"]))
    s.append("\n## The chain cut where its path probability falls below a threshold (draft length per step, Q687)\n")
    s.append(md_table(tables["adaptive"]))
    s.append("\n## Copy-gated: the n-gram chain on copy steps (no draft pass), the tree elsewhere\n")
    s.append(md_table(tables["copy_gated"]))
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------- self-test


def synth_markov_step(rng, step, task, K=4, D=5, n_draft=3, leak=0.1, vocab=1000):
    """A random lattice and a continuation drawn from it: each truth token follows the lattice's conditional row
    over the kept candidates, and with the row's missing mass (leak) a token outside them, after which the truth is
    random. A node's acceptance probability is then exactly its path probability."""
    ids = [rng.sample(range(1, vocab), K) for _ in range(D)]

    def row():
        w = [rng.random() ** 2 for _ in range(K)]
        z = sum(w) / (1.0 - leak)
        return [x / z for x in w]

    p = [row()]
    for _ in range(1, D):
        p.append([x for _ in range(K) for x in row()])
    # the drafter's walk: argmax of each conditional row
    pick = [max(range(K), key=lambda b: p[0][b])]
    for i in range(1, D):
        a = pick[-1]
        pick.append(max(range(K), key=lambda b: p[i][a * K + b]))
    # the truth
    y, a = [], None
    for i in range(D):
        r_ = p[0] if i == 0 else p[i][a * K:(a + 1) * K]
        u = rng.random()
        b = None
        for c in range(K):
            u -= r_[c]
            if u < 0:
                b = c
                break
        if b is None:  # outside the candidates: the rest is unrelated
            y.append(vocab + 1)
            y += [vocab + 2 + rng.randrange(5) for _ in range(D - i - 1)]
            break
        y.append(ids[i][b])
        a = b
    draft = [ids[i][pick[i]] for i in range(n_draft)]
    n = 0
    while n < n_draft and draft[n] == y[n]:
        n += 1
    sampled = draft[:n] + [y[n]]
    return {"v": 1, "step": step, "task": task, "slot": 0, "pos": 100, "n_prompt": 50, "verify": "match",
            "type": "synthetic", "last": 7, "draft": draft, "sampled": sampled, "n_acc": n, "lattice": True,
            "k": K, "cand": ids, "p": p, "pick": pick, "truth": y, "ng_len": 0, "ng": []}


def selftest():
    ok = True

    def expect(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok &= bool(cond)

    # 1. a hand-made lattice with a known order and known values
    #    p0 = [0.6, 0.4]; from candidate 0: [0.7, 0.3], from candidate 1: [0.9, 0.1]
    #    paths: (0) .6, (0,0) .42, (1) .4, (1,0) .36, (0,1) .18, (1,1) .04; the truth is path (1,0)
    s = {"v": 1, "step": 0, "task": 1, "slot": 0, "pos": 10, "n_prompt": 5, "verify": "match", "type": "synthetic",
         "last": 5, "draft": [11, 21], "sampled": [12], "n_acc": 0, "lattice": True, "k": 2,
         "cand": [[11, 12], [21, 22]], "p": [[0.6, 0.4], [0.7, 0.3, 0.9, 0.1]], "pick": [0, 0],
         "ng_len": 2, "ng": [12, 21, 99], "_file": 0}
    s2 = {"v": 1, "step": 1, "task": 1, "slot": 0, "pos": 11, "n_prompt": 5, "verify": "match", "type": "synthetic",
          "last": 12, "draft": [21], "sampled": [21, 30], "n_acc": 1, "lattice": True, "k": 2,
          "cand": [[21, 22], [30, 31]], "p": [[0.8, 0.2], [0.5, 0.5, 0.5, 0.5]], "pick": [0, 0],
          "ng_len": 0, "ng": [], "_file": 0}
    steps = [s, s2]
    good, lines = check(steps)
    expect(good, "hand-made dump is consistent: " + lines[0])
    expect(s["_y"][:2] == [12, 21], "the continuation is rebuilt from the next step (12, 21)")
    lat = Lattice(s, "markov")
    order = [(n[0], n[1]) for n in best_first(lat, 6)]
    expect(order == [(0, 0), (1, 0), (0, 1), (1, 0), (1, 1), (1, 1)], f"best-first order by path probability {order}")
    probs = [round(n[3], 6) for n in best_first(lat, 6)]
    expect(probs == [0.6, 0.42, 0.4, 0.36, 0.18, 0.04], f"path probabilities {probs}")
    rec = score_step(s, lat, [1, 2, 3, 4, 6], tgt=False)
    taus = [tau_tree(rec, B) for B in (1, 2, 3, 4, 6)]
    expect(taus == [1, 1, 2, 3, 3], f"tree tau at budgets 1, 2, 3, 4, 6: {taus}")
    expect(tau_chain(rec, 2) == 1, "the walk (11, 21) accepts nothing")
    expect(rec["ng_acc"] == 2, "the n-gram chain (12, 21, 99) accepts 2")
    fact = Lattice(s, "factorized")  # marginals: (0.6, 0.4), (0.78, 0.22)
    pf = [round(n[3], 6) for n in best_first(fact, 6)]
    expect(pf == [0.6, 0.468, 0.4, 0.312, 0.132, 0.088], f"factorized path probabilities {pf}")

    # 2. a broken dump is caught
    bad = json.loads(json.dumps(steps[0]))
    bad["n_acc"] = 1
    bad["sampled"] = [11, 12]
    good2, _ = check([bad, json.loads(json.dumps(s2))])
    expect(not good2, "an n_acc that the drafts and the target's tokens contradict is caught")

    # 3. calibrated random lattices: each rank's value equals its mean path probability within sampling error,
    #    and the tree's tau equals 1 plus the summed path probabilities
    rng = random.Random(1234)
    steps = [synth_markov_step(rng, i, 100 + i) for i in range(6000)]
    for st in steps:
        st["_file"] = 0
    good3, lines3 = check(steps)
    expect(good3, "synthetic dump is consistent: " + lines3[0])
    budgets = [1, 2, 3, 4, 6, 8, 12, 16]
    rep, tab = analyse(steps, budgets, "markov", [0.1], 0.2, 4, [4096], tgt=False)
    worst = 0.0
    for row in tab["ranks"]:
        z = abs(row["value"] - row["predicted"]) / max(row["value_se"], 1e-9)
        worst = max(worst, z)
    expect(worst < 4.5, f"node values match path probabilities over 16 ranks (largest |z| {worst:.2f})")
    for B in (3, 8, 16):
        row = [r for r in tab["budgets"] if r["budget"] == B][0]
        pred = 1 + mean(sum(n[3] for n in best_first(Lattice(st, "markov"), B)) for st in steps)
        expect(abs(row["tau_tree"] - pred) < 0.05, f"tau at budget {B}: {row['tau_tree']:.3f} against {pred:.3f}")
    ch = tab["chain"]
    expect(all(ch[i]["cumulative"] >= ch[i + 1]["cumulative"] for i in range(len(ch) - 1)),
           "chain acceptance falls with the position")
    expect(all(r["tau_tree"] >= r["tau_chain"] - 1e-9 for r in tab["budgets"] if r["budget"] <= 5),
           "within the walk's depth the best-first tree accepts at least the chain")
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    ap.add_argument("--r", default=",".join(map(str, DEFAULT_R)))
    ap.add_argument("--draft-cost", type=float, default=0.22, help="the draft pass, in one-token steps (default 0.22)")
    ap.add_argument("--mode", choices=["markov", "factorized"], default="markov")
    ap.add_argument("--copy-min", type=int, default=4)
    ap.add_argument("--ctx-bins", default="4096,8192,16384,32768")
    ap.add_argument("--no-target", action="store_true", help="leave out the expected values from the target's rows")
    ap.add_argument("--check", action="store_true", help="only check the dumps' consistency")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dumps:
        ap.error("no dump files")
    steps = load_steps(args.dumps)
    good, lines = check(steps)
    if args.check:
        print("\n".join(lines))
        return 0 if good else 1
    budgets = sorted(int(x) for x in args.budgets.split(","))
    rs = [float(x) for x in args.r.split(",")]
    ctx_bins = [int(x) for x in args.ctx_bins.split(",")]
    report, tables = analyse(steps, budgets, args.mode, rs, args.draft_cost, args.copy_min, ctx_bins,
                             tgt=not args.no_target)
    text = summary_md(report, tables, args, rs)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "check.txt").write_text("\n".join(lines) + "\n")
        (out / "summary.md").write_text(text)
        for name, rows in tables.items():
            write_csv(out / f"{name}.csv", rows)
    print(text)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
