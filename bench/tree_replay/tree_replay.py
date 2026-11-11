#!/usr/bin/env python3
"""Offline tree-value replay of speculative verify steps dumped by llama-server (LLAMA_SPEC_DUMP, Q688).

    python3 tree_replay.py DUMP.jsonl[.gz] [...] --out DIR [--budgets 3,4,6,8,12,16] [--r 0.08,0.10,0.12,0.40]
                           [--draft-cost 0.22] [--mode markov|factorized] [--copy-min 4]
                           [--ctx-bins 4096,8192,16384,32768] [--lattice auto|fb|prod]
                           [--tool-open ID --tool-close ID --think-open ID --think-close ID]
                           [--shapes auto|none|WxMxL,...] [--pass-cost 0.025]
    python3 tree_replay.py --check DUMP.jsonl[.gz] [...]
    python3 tree_replay.py --selftest

Each dump line is one verify step of the server's speculative loop: the position of the last sampled token, the drafted
tokens, the target's tokens (the accepted drafts, then its own), the drafter's candidates per position (for DFlash2 its
selector lattice: conditional probabilities from each candidate of the previous position), the target's most probable
tokens at the sampled positions, and the longest earlier match of the context's suffix with the tokens that followed it.
With LLAMA_SPEC_DUMP_FULL_BLOCK a line also carries "fb": the lattice of a second block the drafter decoded for the
record (DFlash2's whole trained block, 7 positions after the anchor for Inco AI's drafter) next to the draft's own, which
stays what the line drafts; the tables walk the second block by default (--lattice).

Acceptance by position (block*.csv): along the drafter's greedy walk, P_j (the first j tokens all accepted) and the
conditional acceptance at each position j = 1..7 (or as deep as the block goes), overall and by context length, by
segment of the anchor token (tool call, reasoning, text: from the marker token ids), by the server's tool-call grammar
flag and by content (copy), then tau(k) and the speed-up of drafting k tokens against the research note's two tails.

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

With verification by rejection (the dump's "verify": "reject", --spec-draft-sampling probabilistic), the continuation
is the same rebuilt sequence: where the walk's token is accepted it is the walk's, and at its first rejection it is the
target's draw from the residual. The chain's value is then exactly what the server accepted, and a node beside the walk
at that position is worth 1 when the residual's draw is its token: the value of a tree verified as the chain is, with the
residual's draw matched against the position's other nodes. Nodes below an off-walk node are factorized estimates.

Trees for a sequential drafter (shapes.csv, siblings.csv; MTP, a draft model): the walk cut to L tokens with w - 1
sibling leaves at each of its first m positions, which the drafter's candidates give without another drafting pass and
which are valued exactly; full trees of width w over the first m positions (estimates); each against the chain with as
many verified tokens, on the same steps.

Output in DIR: summary.md (all tables) and chain.csv, budgets.csv, ranks.csv, calibration.csv, by_ctx.csv,
by_content.csv, shapes.csv, siblings.csv, check.txt. Standard library only.
"""
import argparse
import gzip
import heapq
import json
import math
import random
import struct
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


def segment_states(tok, markers):
    """The state after each position's token of one request, as {position: segment}: "tool" between the tool-call
    markers, "think" between the reasoning markers, "text" elsewhere ("other" when no reasoning markers are given).
    The dump holds the generated tokens, not the prompt, so a request's first tokens are classified by the first
    marker seen: a closing marker first means the request began inside it (a chat template that opens the reasoning
    in the prompt), an opening marker or none means outside."""
    if not markers or not any(v is not None for v in markers.values()):
        return {}
    t_open, t_close = markers.get("tool_open"), markers.get("tool_close")
    k_open, k_close = markers.get("think_open"), markers.get("think_close")
    order = sorted(tok)
    first_tool = next((("open" if tok[p] == t_open else "close") for p in order if tok[p] in (t_open, t_close) and tok[p] is not None), None)
    first_think = next((("open" if tok[p] == k_open else "close") for p in order if tok[p] in (k_open, k_close) and tok[p] is not None), None)
    in_tool = first_tool == "close"
    think = first_think == "close"
    have_think = k_open is not None or k_close is not None
    out = {}
    for p in order:
        t = tok[p]
        if t_open is not None and t == t_open:
            in_tool = True
        elif t_close is not None and t == t_close:
            in_tool = False
        if k_open is not None and t == k_open:
            think = True
        elif k_close is not None and t == k_close:
            think = False
        out[p] = "tool" if in_tool else ("think" if think else ("text" if have_think else "other"))
    return out


def attach_truth(steps, depth_cap=64, markers=None):
    """Rebuild each request's token sequence by position and give every step its continuation (step["_y"]) and, when
    markers are given ({"tool_open": id, ...}), the segment its anchor token is in (step["_seg"]).

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
        seg = segment_states(tok, markers)
        for s in g:
            s["_key"] = key
            if seg:
                s["_seg"] = seg.get(s["pos"])
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


def block_view(s, source="auto"):
    """The step as seen through one lattice: the second block's ("fb", LLAMA_SPEC_DUMP_FULL_BLOCK) or the draft's own
    ("prod"). "auto" takes the second block where the step has one. Returns None when `source` asks for a lattice the
    step does not have. The result is a step (the same fields) whose cand, p, pick and lattice are the chosen block's."""
    fb = s.get("fb")
    if source in ("auto", "fb") and fb and fb.get("cand"):
        v = dict(s)
        v.update(cand=fb["cand"], p=fb["p"], pick=fb["pick"], lattice=True, kept=fb.get("kept", []), _from="fb")
        return v
    if source == "fb" or not s.get("cand"):
        return None
    v = dict(s)
    v["_from"] = "prod"
    return v


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
    # the target's rows are conditional on the tokens it sampled: a node's row is known where the path to its parent
    # is on the walk and the walk's tokens so far are the target's own (the lattice may come from a block that is not
    # the one the draft was made from, so the walk need not equal the draft)
    n_acc_live = 0
    for i, b in enumerate(walk):
        if i < len(s["sampled"]) and lat.ids[i][b] == s["sampled"][i]:
            n_acc_live += 1
        else:
            break
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


def analyse(steps, budgets, mode, rs, draft_cost, copy_min, ctx_bins, tgt=True, source="auto"):
    """Every table, as lists of dict rows, plus the consistency report. `source` picks the lattice (block_view)."""
    report = {"n_lines": len(steps)}
    recs = []
    n_short = 0
    n_nocand = 0
    types = defaultdict(int)
    for s in steps:
        types[s.get("type") or "none"] += 1
        v = block_view(s, source)
        if v is None:
            n_nocand += 1
            continue
        lat = Lattice(v, mode)
        if len(s["_y"]) < lat.D:
            n_short += 1
            continue
        recs.append(score_step(v, lat, budgets, tgt))
    report.update(n_scored=len(recs), n_short=n_short, n_nocand=n_nocand, types=dict(types),
                  n_lattice=sum(1 for s in steps if s.get("lattice") or s.get("fb")),
                  n_fb=sum(1 for s in steps if s.get("fb")), source=source)
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


# ---------------------------------------------------------------- trees for a sequential drafter (MTP)


def siblings_at(lat, j, w):
    """The w - 1 most probable candidates at position j other than the walk's own, as candidate indices."""
    walk = lat.chain()
    own = walk[j] if j < len(walk) else -1
    order = sorted(range(lat.K[j]), key=lambda b: -lat.marg[j][b])
    return [b for b in order if b != own][:w - 1]


def walk_tokens(lat):
    return [lat.ids[i][b] for i, b in enumerate(lat.chain())]


def tau_siblings(lat, y, w, m, L):
    """Accepted tokens + 1 of the walk cut to L with w - 1 sibling leaves at each of its first m positions. Exact for a
    drafter whose candidates are conditional on its own walk (MTP, a draft model): a sibling's row is the one the walk
    already produced, and the tree is verified as the chain is (the walk first; on its first rejection, the token the
    target draws there is matched against that position's siblings, whose next token the same pass has computed)."""
    wt = walk_tokens(lat)
    a = 0
    while a < L and a < len(wt) and a < len(y) and wt[a] == y[a]:
        a += 1
    bonus = 0
    if a < L and a < m and a < len(y):
        bonus = 1 if y[a] in [lat.ids[a][b] for b in siblings_at(lat, a, w)] else 0
    return 1 + a + bonus, bonus


def tau_full(lat, y, w, m, L):
    """Accepted tokens + 1 of every path over the w most probable candidates at each of the first m positions, the walk
    going on to L below its own node at depth m. An estimate for a sequential drafter: below an off-walk node the
    candidates are the walk's (factorized), where a drafter would have drafted from that node."""
    wt = walk_tokens(lat)
    d = 0
    while d < m and d < len(y) and d < lat.D:
        top = [lat.ids[d][b] for b in sorted(range(lat.K[d]), key=lambda b: -lat.marg[d][b])[:w]]
        if wt[d] not in top:
            top = top[:w - 1] + [wt[d]]
        if y[d] not in top:
            break
        d += 1
    if d == m and y[:m] == wt[:m]:
        while d < L and d < len(wt) and d < len(y) and wt[d] == y[d]:
            d += 1
    return 1 + d


def tau_best_first(lat, y, B):
    nodes = best_first(lat, B)
    acc = []
    for (d, b, parent, pr, on) in nodes:
        acc.append(1 if ((parent < 0 or acc[parent]) and d < len(y) and lat.ids[d][b] == y[d]) else 0)
    return 1 + sum(acc)


def parse_shapes(spec, D):
    """'auto' or 'wxmxL,...': sibling and full trees of width w over the first m positions on a walk of L."""
    if spec == "none":
        return []
    if spec == "auto":
        return [(w, m, L) for L in range(2, D + 1) for w in (2, 3) for m in range(1, min(3, L) + 1)]
    out = []
    for x in spec.split(","):
        w, m, L = (int(v) for v in x.lower().split("x"))
        if not (w >= 2 and 1 <= m <= L):
            raise SystemExit(f"--shapes: {x} is not width >= 2, 1 <= m <= L")
        out.append((w, m, L))
    return out


def analyse_shapes(steps, shapes, rs, pass_cost, source="prod"):
    """Trees a sequential drafter can verify, each against the chain with as many verified tokens, on the same steps.
    A step counts for a shape when its walk reaches max(L, nodes) positions and the continuation is known that far."""
    recs = []
    for s in steps:
        v = block_view(s, source)
        if v is None:
            continue
        lat = Lattice(v, "factorized")
        if lat.D == 0 or len(s["_y"]) < lat.D or len(lat.chain()) < lat.D:
            continue
        recs.append((lat, s["_y"], s.get("_key")))
    if not recs:
        return {}
    D = max(lat.D for lat, _, _ in recs)
    rows = []
    for (w, m, L) in shapes:
        for kind in ("siblings", "full"):
            nodes = L + (w - 1) * m if kind == "siblings" else sum(w ** i for i in range(1, m + 1)) + (L - m)
            # the chain of as many verified tokens needs a walk that long; without one the tree is scored alone
            chain_known = nodes <= D
            need = max(L, nodes) if chain_known else L
            have = [(lat, y) for lat, y, _ in recs if lat.D >= need]
            row = {"shape": f"{kind} {w}x{m} on {L}", "kind": kind, "width": w, "positions": m, "walk": L,
                   "nodes": nodes, "exact": "yes" if kind == "siblings" else "est.", "steps": len(have)}
            if not have:
                rows.append(row)
                continue
            if kind == "siblings":
                tt, hits = zip(*(tau_siblings(lat, y, w, m, L) for lat, y in have))
            else:
                tt = [tau_full(lat, y, w, m, L) for lat, y in have]
                hits = None
            nan = float("nan")
            tc = [1 + min(_acc(lat, y), nodes) for lat, y in have] if chain_known else None
            tb = [tau_best_first(lat, y, nodes) for lat, y in have]
            row.update({"tau_tree": mean(tt), "tau_chain_same_nodes": mean(tc) if tc else nan,
                   "tree_minus_chain": mean(tt) - mean(tc) if tc else nan,
                   "se": se([a - b for a, b in zip(tt, tc)]) if tc else nan,
                   "tau_best_first_same_nodes": mean(tb),
                   "tau_walk_L": mean(1 + min(_acc(lat, y), L) for lat, y in have),
                   "sibling_hits": mean(hits) if hits else float("nan"),
                   "passes_tree": L, "passes_chain": nodes})
            for r_ in rs:
                row[f"S_tree_r{r_:g}"] = row["tau_tree"] / (1.0 + L * pass_cost + nodes * r_)
                row[f"S_chain_r{r_:g}"] = row["tau_chain_same_nodes"] / (1.0 + nodes * pass_cost + nodes * r_)
            rows.append(row)
    # the siblings' value by position: the walk's first rejection at j, and whether the target's token there is among
    # the k most probable other candidates
    sib = []
    K = max(max(lat.K) for lat, _, _ in recs)
    for j in range(D):
        rej = [(lat, y) for lat, y, _ in recs if lat.D > j and _acc(lat, y) == j and j < len(y)]
        row = {"position": j + 1, "rejections": len(rej)}
        for k in range(1, K):
            row[f"in_top{k}_others"] = mean(1 if y[j] in [lat.ids[j][b] for b in siblings_at(lat, j, k + 1)] else 0
                                            for lat, y in rej) if rej else float("nan")
        sib.append(row)
    return {"shapes": rows, "siblings": sib}


def _acc(lat, y):
    """The walk's accepted length against the continuation."""
    wt = walk_tokens(lat)
    a = 0
    while a < len(wt) and a < len(y) and wt[a] == y[a]:
        a += 1
    return a


# ---------------------------------------------------------------- acceptance by position along the drafter's walk

# tails to read the measurement against (Q804's research note, section 3.2): P_j, the probability that the first j
# drafted tokens are all accepted
REFERENCE_TAILS = {
    "note's est. (rig-anchored)": [0.86, 0.70, 0.54, 0.39, 0.28, 0.20, 0.145],
    "INT4 port's collapse": [0.76, 0.45, 0.31, 0.14, 0.10, 0.02, 0.0],
}


def leading_matches(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def block_records(steps, source="auto", mode="markov"):
    """One record per step for the per-position tables: the drafter's greedy walk through the chosen lattice (its
    tokens, its path probabilities), the continuation known after the step, and how far the walk matches it."""
    recs = []
    for s in steps:
        v = block_view(s, source)
        if v is None:
            continue
        lat = Lattice(v, mode)
        walk = lat.chain()
        toks = [lat.ids[i][b] for i, b in enumerate(walk)]
        wp, pr, prev_b = [], 1.0, -1
        for i, b in enumerate(walk):
            pr *= lat.cond(i, prev_b, b)
            wp.append(pr)
            prev_b = b
        y = s["_y"]
        draft = list(s.get("draft", []))
        recs.append({"pos": s["pos"], "toks": toks, "n": len(toks), "wp": wp, "y_len": len(y),
                     "acc": leading_matches(toks, y), "key": s.get("_key"), "seg": s.get("_seg"), "gram": s.get("gram"),
                     "ng_len": s.get("ng_len", 0), "from": v["_from"], "draft": draft,
                     "draft_acc": leading_matches(draft, y), "equal": leading_matches(draft, toks)})
    return recs


def ratio_se(pairs):
    """The standard error of sum(a) / sum(n) over clusters of (a, n), clustered (a request's steps are not
    independent): sqrt(T / (T - 1) * sum_t (a_t - r n_t)^2) / sum(n)."""
    A = sum(a for a, _ in pairs)
    N = sum(n for _, n in pairs)
    T = len(pairs)
    if N == 0 or T < 2:
        return float("nan")
    r = A / N
    return math.sqrt(T / (T - 1) * sum((a - r * n) ** 2 for a, n in pairs)) / N


def position_rows(recs, W):
    """Acceptance by position j = 1..W along the walk. A step counts at j when its walk reaches j and the target's
    continuation is known to j (a request's last steps are shorter: `censored`). cumulative = P_j, the probability
    that the first j tokens are all accepted; conditional = P_j / P_(j-1) on the same steps; strict = P_j on the steps
    whose continuation is known to W (the same steps at every j); predicted = the selector's own path probability of
    the walk, averaged over the same steps."""
    full = [r for r in recs if r["n"] >= W and r["y_len"] >= W]
    rows = []
    for j in range(1, W + 1):
        have = [r for r in recs if r["n"] >= j and r["y_len"] >= j]
        cens = sum(1 for r in recs if r["n"] >= j and r["y_len"] < j)
        reach = [1 if r["acc"] >= j else 0 for r in have]
        prev = sum(1 for r in have if r["acc"] >= j - 1)
        by_key = defaultdict(lambda: [0, 0])
        for r, a in zip(have, reach):
            by_key[r["key"]][0] += a
            by_key[r["key"]][1] += 1
        rows.append({"position": j, "steps": len(have), "censored": cens,
                     "cumulative": mean(reach), "cumulative_se": ratio_se(list(by_key.values())),
                     "conditional": (sum(reach) / prev) if prev else float("nan"),
                     "strict_steps": len(full), "strict_cumulative": mean(1 if r["acc"] >= j else 0 for r in full),
                     "predicted": mean(r["wp"][j - 1] for r in have)})
    return rows


def speed_rows(P, rs, draft_cost):
    """tau(k) = 1 + P_1 + ... + P_k and S(k) = tau(k) / (1 + draft pass + k r) for k = 1..len(P): k drafted tokens
    verified in one pass of k + 1 columns, the pass costing 1 + k r one-token steps."""
    rows, tau = [], 1.0
    for k, p in enumerate(P, 1):
        tau += p
        row = {"k": k, "P_k": p, "tau": tau}
        for r_ in rs:
            row[f"S_r{r_:g}"] = tau / (1.0 + draft_cost + k * r_)
        rows.append(row)
    return rows


def best_k(P, rs, draft_cost):
    rows = speed_rows(P, rs, draft_cost)
    return {r_: max(rows, key=lambda row: row[f"S_r{r_:g}"]) for r_ in rs}


def group_position_rows(recs, key_fn, name, W, ks=(3, 6)):
    """Per group of steps, P_j and the conditional acceptance by position, tau of the whole block and of the first k."""
    groups = defaultdict(list)
    for r in recs:
        k = key_fn(r)
        if k is not None:
            groups[k].append(r)
    out = []
    for key in sorted(groups, key=lambda k: (isinstance(k, str), k)):
        g = groups[key]
        pr = position_rows(g, W)
        row = {name: key, "steps": len(g)}
        for p in pr:
            row[f"n{p['position']}"] = p["steps"]
        for p in pr:
            row[f"P{p['position']}"] = p["cumulative"]
        for p in pr:
            row[f"a{p['position']}"] = p["conditional"]
        Ps = [p["cumulative"] for p in pr]
        for k in ks:
            if k <= W and all(not math.isnan(x) for x in Ps[:k]):
                row[f"tau_k{k}"] = 1 + sum(Ps[:k])
        if all(not math.isnan(x) for x in Ps):
            row["tau_all"] = 1 + sum(Ps)
        out.append(row)
    return out


def width_rows(recs, W):
    """Where steps carry both the draft's own block and a second one: for the draft's first j tokens, how often they are
    the second block's first j walk tokens (`same`), and the draft's and the second block's own P_j on the steps where
    both exist and the continuation is known to j. A difference is the block width's effect on the drafter (DFlash2's
    attention across the block is not causal)."""
    pair = [r for r in recs if r["from"] == "fb" and r["draft"]]
    rows = []
    for j in range(1, W + 1):
        have = [r for r in pair if len(r["draft"]) >= j and r["n"] >= j and r["y_len"] >= j]
        if not have:
            break
        rows.append({"position": j, "steps": len(have), "same": mean(1 if r["equal"] >= j else 0 for r in have),
                     "draft_P": mean(1 if r["draft_acc"] >= j else 0 for r in have),
                     "full_block_P": mean(1 if r["acc"] >= j else 0 for r in have),
                     "draft_minus_full": mean((1 if r["draft_acc"] >= j else 0) - (1 if r["acc"] >= j else 0) for r in have)})
    return rows


def analyse_blocks(steps, rs, draft_cost, copy_min, ctx_bins, source="auto", mode="markov"):
    """The per-position tables of the drafter's walk (the second block's where the dump has one): overall, by context
    length, by segment (the markers), by the server's grammar flag, by content (copy), and the block width's effect."""
    recs = block_records(steps, source, mode)
    report = {"n_block_steps": len(recs), "n_second_block": sum(1 for r in recs if r["from"] == "fb"),
              "block_positions": max((r["n"] for r in recs), default=0)}
    tables = {}
    W = report["block_positions"]
    if not recs or W == 0:
        return report, tables
    tables["block"] = position_rows(recs, W)
    P = [row["cumulative"] for row in tables["block"]]
    if all(not math.isnan(x) for x in P):
        tables["block_speed"] = speed_rows(P, rs, draft_cost)
    # the same arithmetic on the reference tails, to read the measured tail against
    ref = []
    for name, tail in REFERENCE_TAILS.items():
        best = best_k(tail, rs, draft_cost)
        row = {"tail": name}
        for r_ in rs:
            row[f"best_k_r{r_:g}"] = best[r_]["k"]
            row[f"S_r{r_:g}"] = best[r_][f"S_r{r_:g}"]
        ref.append(row)
    if all(not math.isnan(x) for x in P):
        best = best_k(P, rs, draft_cost)
        row = {"tail": "measured"}
        for r_ in rs:
            row[f"best_k_r{r_:g}"] = best[r_]["k"]
            row[f"S_r{r_:g}"] = best[r_][f"S_r{r_:g}"]
        ref.insert(0, row)
    tables["block_vs_reference"] = ref

    def ctx_key(r):
        lo = 0
        for e in ctx_bins:
            if r["pos"] < e:
                return f"{lo}-{e - 1}"
            lo = e
        return f">={lo}"

    by_ctx = group_position_rows(recs, ctx_key, "context", W)
    order = {}
    for r in sorted(recs, key=lambda r: r["pos"]):
        order.setdefault(ctx_key(r), len(order))
    by_ctx.sort(key=lambda row: order[row["context"]])
    tables["block_by_ctx"] = by_ctx
    if any(r["seg"] is not None for r in recs):
        tables["block_by_segment"] = group_position_rows(recs, lambda r: r["seg"], "segment", W)
    if any(r["gram"] is not None for r in recs):
        tables["block_by_gram"] = group_position_rows(
            recs, lambda r: None if r["gram"] is None else ("tool-call grammar active" if r["gram"] else "no grammar"),
            "grammar", W)
    tables["block_by_content"] = group_position_rows(
        recs, lambda r: "copy" if r["ng_len"] >= copy_min else "fresh", "content", W)
    wr = width_rows(recs, W)
    if wr:
        tables["block_width"] = wr
    return report, tables


def best_s(tables, rs, kind):
    out = {}
    for r_ in rs:
        best = max(tables["curve"], key=lambda row: row[f"S_{kind}_r{r_:g}"])
        out[r_] = (best["budget"], best[f"S_{kind}_r{r_:g}"])
    return out


# ---------------------------------------------------------------- checks


def check(steps, markers=None):
    """Consistency of the dump with itself; returns (ok, lines of text)."""
    lines = []
    bad = 0
    problems = attach_truth(steps, markers=markers)
    for p in problems[:20]:
        lines.append("conflict: " + p)
    bad += len(problems)
    n_gap = sum(1 for s in steps if s.get("_gap"))
    n_eog = 0
    n_fb = 0
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
        fb = s.get("fb")
        if fb is not None:
            fc, fp, fk, fw = fb.get("cand") or [], fb.get("p") or [], fb.get("pick") or [], fb.get("w", 0)
            K = [len(c) for c in fc]
            if not fc or fw != len(fc) + 1 or len(fp) != len(fc) or len(fk) != len(fc):
                lines.append(f"step {s.get('step')}: the second block has {len(fc)} positions, {len(fp)} rows, "
                             f"{len(fk)} picks for a width of {fw}")
                bad += 1
            elif len(fp[0]) != K[0] or any(len(fp[i]) != K[i - 1] * K[i] for i in range(1, len(K))):
                lines.append(f"step {s.get('step')}: the second block's lattice rows do not match its candidate counts")
                bad += 1
            elif any(k < 0 or k >= K[i] for i, k in enumerate(fk)):
                lines.append(f"step {s.get('step')}: the second block's walk leaves its candidates")
                bad += 1
            n_fb += 1
        if s.get("tgt_ps") is not None and len(s["tgt_ps"]) not in (0, len(smp)):
            lines.append(f"step {s.get('step')}: {len(s['tgt_ps'])} target rows for {len(smp)} sampled tokens")
            bad += 1
    lines.insert(0, f"{len(steps)} steps, {bad} problems, {n_gap} gaps between consecutive steps of a request, "
                    f"{n_eog} stops at an end-of-generation token, {n_fb} with a second block")
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
             f"lattice; the others are factorized); lattice: {report.get('source', 'auto')} ({report.get('n_fb', 0)} steps "
             f"carry a second block, which `auto` and `fb` use in place of the draft's own); draft pass "
             f"{args.draft_cost:g} of a one-token step; verified live: "
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


def shapes_summary_md(tables, args, rs):
    s = ["## Trees for a sequential drafter, against the chain of as many verified tokens\n"]
    s.append("A *siblings* tree is the walk cut to L tokens with w - 1 leaves beside each of its first m tokens: the "
             "drafter's next most probable candidates there, from the passes that drafted the walk, so it needs no "
             "drafting pass more than the walk's L. Its value is exact: the walk is verified as a chain is, and on its "
             "first rejection the target's token there is matched against that position's leaves (a leaf that matches "
             "is accepted, and the pass has already computed the token after it). A *full* tree verifies every path "
             "over the w most probable candidates at each of the first m positions and goes on along the walk below "
             "its own node; it is an estimate: below a candidate off the walk it takes the walk's candidates "
             "(factorized), where the drafter would have drafted from that candidate. *nodes* is the verified tokens "
             "beyond the anchor, the chain's length for the same verify width; *tau* is accepted tokens + 1 per verify "
             "step on the steps whose walk reaches max(L, nodes); *se* is the standard error of the paired difference; "
             "*best-first* is the factorized best-first tree of as many nodes (an estimate); S = tau / (1 + passes x "
             f"{args.pass_cost:g} + nodes x r) in one-token steps.\n")
    cols = ["shape", "exact", "nodes", "steps", "tau_tree", "tau_chain_same_nodes", "tree_minus_chain", "se",
            "tau_best_first_same_nodes", "tau_walk_L", "sibling_hits"]
    s.append(md_table(tables["shapes"], cols))
    s.append("\n### Speed-ups of the same shapes (est.: the pass cost and r are assumptions)\n")
    s.append(md_table(tables["shapes"], ["shape", "nodes", "passes_tree", "passes_chain"]
                      + [c for r_ in rs for c in (f"S_tree_r{r_:g}", f"S_chain_r{r_:g}")]))
    s.append("\n### The siblings' value by position: the walk's first rejection there, and the share of those where "
             "the target's token is among the k most probable other candidates\n")
    s.append(md_table(tables["siblings"]))
    return "\n".join(s) + "\n"


def block_summary_md(report, tables, args, rs):
    s = ["## Acceptance by position along the drafter's walk\n"]
    s.append(f"{report['n_block_steps']} steps, {report['n_second_block']} of them through the second block of "
             f"`LLAMA_SPEC_DUMP_FULL_BLOCK` (the rest through the draft's own lattice), the walk covering up to "
             f"{report['block_positions']} positions after the anchor. A step counts at position j when its walk reaches j "
             f"and the target's continuation is known to j; `censored` steps ended before j (the end of a request). "
             f"*cumulative* is P_j, the probability that the first j drafted tokens are all accepted (standard error "
             f"clustered by request); *conditional* is P_j / P_(j-1) on the same steps; *strict* is P_j on the steps "
             f"whose continuation is known to the whole block; *predicted* is the selector's own path probability of "
             f"the walk. tau(k) = 1 + P_1 + ... + P_k; S(k) = tau(k) / (1 + draft pass {args.draft_cost:g} + k r).\n")
    if not tables:
        return "\n".join(s) + "\n"
    s.append(md_table(tables["block"]))
    if "block_speed" in tables:
        s.append("\n### tau and speed-up by drafted length k (one pass of k + 1 columns)\n")
        s.append(md_table(tables["block_speed"]))
    if "block_vs_reference" in tables:
        s.append("\n### The best drafted length and its speed-up, measured and on the research note's two tails\n")
        s.append(md_table(tables["block_vs_reference"]))
    s.append(f"\n### By context length (position of the step)\n")
    s.append(md_table(tables["block_by_ctx"]))
    if "block_by_segment" in tables:
        s.append("\n### By segment of the anchor token (tool call, reasoning, text: from the token markers)\n")
        s.append(md_table(tables["block_by_segment"]))
    if "block_by_gram" in tables:
        s.append("\n### By the server's tool-call grammar at the draft (the dump's `gram` flag)\n")
        s.append(md_table(tables["block_by_gram"]))
    s.append(f"\n### By content (copy: the context's suffix occurs earlier for {args.copy_min}+ tokens)\n")
    s.append(md_table(tables["block_by_content"]))
    if "block_width" in tables:
        s.append("\n### The block width's effect: the draft's own block against the second block, same steps\n")
        s.append("`same` is the share of steps whose first j drafted tokens equal the second block's walk; "
                 "`draft_P` and `full_block_P` are the two walks' P_j against the same continuation.\n")
        s.append(md_table(tables["block_width"]))
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


def synth_block_step(rng, step, task, a, pos, cut=None, n_draft=2, K=3, vocab=1000, same=1.0):
    """A step with a second block of len(a) positions whose walk (candidate 0 at every position) is accepted at position
    j with conditional probability a[j - 1], so the chain's P_j is the product a[0] ... a[j - 1] exactly. The rows give
    candidate 0 the probability a[j - 1] after any predecessor, so the selector's path probability of the walk is that
    product too. `cut` leaves only that many positions of the continuation (the end of a request); the draft's own block
    has n_draft positions and starts with the walk's tokens with probability `same`."""
    W = len(a)
    ids = [rng.sample(range(1, vocab), K) for _ in range(W)]

    def row(x):
        return [x, (1.0 - x) * 0.6, (1.0 - x) * 0.4]

    p = [row(a[0])] + [[v for _ in range(K) for v in row(a[i])] for i in range(1, W)]
    toks = [ids[i][0] for i in range(W)]
    y = []
    for i in range(W):
        if rng.random() < a[i]:
            y.append(toks[i])
        else:
            y.append(vocab + 1)
            y += [vocab + 2 + rng.randrange(5) for _ in range(W - i - 1)]
            break
    draft = toks[:n_draft] if rng.random() < same else [vocab + 500 + i for i in range(n_draft)]
    n = 0
    while n < n_draft and draft[n] == y[n]:
        n += 1
    sampled = draft[:n] + [y[n]]
    if cut is not None:
        y = y[:max(cut, n + 1)]
    # the draft's own block: n_draft positions, its walk the draft
    d_ids = [[draft[i], vocab + 600 + i, vocab + 700 + i] for i in range(n_draft)]
    d_p = [[0.5, 0.3, 0.2]] + [[v for _ in range(K) for v in (0.5, 0.3, 0.2)] for _ in range(1, n_draft)]
    return {"v": 1, "step": step, "task": task, "slot": 0, "pos": pos, "n_prompt": 50, "verify": "match",
            "type": "synthetic", "last": 7, "draft": draft, "sampled": sampled, "n_acc": n, "lattice": True,
            "k": K, "cand": d_ids, "p": d_p, "pick": [0] * n_draft, "truth": y, "ng_len": 0, "ng": [],
            "fb": {"w": W + 1, "cand": ids, "p": p, "pick": [0] * W, "kept": [1.0] * W}}


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

    # 4. by-position acceptance on a hand-made request with a known answer. Request tokens by position: 10 anchor 5;
    #    step 0 (pos 10) drafts 11, 21 and the target says 11, 99; step 1 (pos 12, anchor 99) drafts 31, 41 and the
    #    target says 31, 41, 51; step 2 (pos 15, anchor 51) drafts 61 and the target says 61, 71, the request's end.
    #    The second block's walks: 11 21 31 41 / 31 41 51 61 / 61 71 81 91.
    def blk(step, pos, last, draft, sampled, walk):
        W = len(walk)
        ids = [[t, 7000 + 10 * i, 7001 + 10 * i] for i, t in enumerate(walk)]
        p = [[0.5, 0.3, 0.2]] + [[v for _ in range(3) for v in (0.5, 0.3, 0.2)] for _ in range(1, W)]
        n = len(sampled) - 1
        return {"v": 1, "step": step, "task": 5, "slot": 0, "pos": pos, "n_prompt": 9, "verify": "match",
                "type": "draft-dflash", "last": last, "draft": draft, "sampled": sampled, "n_acc": n, "lattice": True,
                "k": 3, "cand": [[t, 7000, 7001] for t in draft],
                "p": [[0.5, 0.3, 0.2]] + [[v for _ in range(3) for v in (0.5, 0.3, 0.2)] for _ in range(1, len(draft))],
                "pick": [0] * len(draft), "ng_len": 0, "ng": [], "gram": 1 if step == 1 else 0, "_file": 0,
                "fb": {"w": W + 1, "cand": ids, "p": p, "pick": [0] * W, "kept": [1.0] * W}}

    hand = [blk(0, 10, 5, [11, 21], [11, 99], [11, 21, 31, 41]),
            blk(1, 12, 99, [31, 41], [31, 41, 51], [31, 41, 51, 61]),
            blk(2, 15, 51, [61], [61, 71], [61, 71, 81, 91])]
    good4, lines4 = check(hand)
    expect(good4, "hand-made second-block dump is consistent: " + lines4[0])
    expect(hand[0]["_y"] == [11, 99, 31, 41, 51, 61, 71] and hand[2]["_y"] == [61, 71],
           "the continuation is rebuilt by position across steps (11 99 31 41 51 61 71)")
    rep4, tab4 = analyse_blocks(hand, [0.1], 0.2, 4, [4096])
    pos = tab4["block"]
    expect([r["steps"] for r in pos] == [3, 3, 2, 2] and [r["censored"] for r in pos] == [0, 0, 1, 1],
           f"steps counted per position, the request's end censored: {[(r['steps'], r['censored']) for r in pos]}")
    expect([round(r["cumulative"], 4) for r in pos] == [1.0, 0.6667, 0.5, 0.5],
           f"P_1..P_4 {[round(r['cumulative'], 4) for r in pos]} (1, 2/3, 1/2, 1/2)")
    expect([round(r["conditional"], 4) for r in pos] == [1.0, 0.6667, 1.0, 1.0],
           f"conditional {[round(r['conditional'], 4) for r in pos]} (1, 2/3, 1, 1)")
    expect([round(r["strict_cumulative"], 4) for r in pos] == [1.0, 0.5, 0.5, 0.5] and pos[0]["strict_steps"] == 2,
           "P_j on the steps whose continuation is known to the whole block (1, 1/2, 1/2, 1/2)")
    sp = tab4["block_speed"]
    expect([round(r["tau"], 4) for r in sp] == [2.0, 2.6667, 3.1667, 3.6667]
           and abs(sp[3]["S_r0.1"] - 3.6667 / (1 + 0.2 + 4 * 0.1)) < 1e-3, "tau(k) = 1 + the sum of P_j, S(k) = tau / (1 + d + k r)")
    wd = tab4["block_width"]
    expect(wd[0]["steps"] == 3 and wd[0]["same"] == 1.0 and wd[1]["steps"] == 2
           and wd[1]["draft_P"] == 0.5 and wd[1]["full_block_P"] == 0.5,
           f"the width table: same {wd[0]['same']}, P_2 of the draft {wd[1]['draft_P']} and of the block {wd[1]['full_block_P']}")
    gram = {r["grammar"]: r for r in tab4["block_by_gram"]}
    expect(gram["tool-call grammar active"]["steps"] == 1 and gram["no grammar"]["steps"] == 2
           and gram["tool-call grammar active"]["P4"] == 1.0, "the server's grammar flag splits the steps (1 and 2)")
    r_fb, t_fb = analyse(hand, [1, 2, 3, 4], "markov", [0.1], 0.2, 4, [4096], tgt=False)
    r_pr, t_pr = analyse(hand, [1, 2], "markov", [0.1], 0.2, 4, [4096], tgt=False, source="prod")
    expect(r_fb["n_fb"] == 3 and len(t_fb["chain"]) == 4 and len(t_pr["chain"]) == 2,
           "the tree tables take the second block's lattice by default (4 positions) and the draft's own on request (2)")

    # 5. segments from the token markers: think closes at 900, a tool call between 901 and 902
    tk = {10: 7, 11: 8, 12: 900, 13: 5, 14: 901, 15: 6, 16: 6, 17: 902, 18: 9}
    seg = segment_states(tk, {"tool_open": 901, "tool_close": 902, "think_open": 899, "think_close": 900})
    expect([seg[p] for p in sorted(seg)] == ["think", "think", "text", "text", "tool", "tool", "tool", "text", "text"],
           "segments by position: reasoning until its close, a tool call between its markers")
    seg2 = segment_states(tk, {"tool_open": 901, "tool_close": 902})
    expect([seg2[p] for p in sorted(seg2)] == ["other", "other", "other", "other", "tool", "tool", "tool", "other", "other"],
           "tool markers alone")
    expect(segment_states(tk, None) == {}, "no markers, no segments")
    st_mk = [dict(blk(0, 10, 7, [8, 900], [8, 900, 5], [8, 900, 5, 901]), _file=0),
             dict(blk(1, 13, 5, [901, 6], [901, 6, 6], [901, 6, 6, 902]), _file=0)]
    check(st_mk, markers={"tool_open": 901, "tool_close": 902, "think_open": 899, "think_close": 900})
    expect([s_["_seg"] for s_ in st_mk] == ["think", "text"], "a step takes the segment of its anchor token")

    # 6. calibrated random second blocks: P_j is the product of the conditionals, in each group, with a request's end
    #    cutting a quarter of the continuations, and the selector's own path probability is that product
    rng = random.Random(77)
    a_tool = [0.95, 0.95, 0.90, 0.90, 0.85, 0.85, 0.80]
    a_text = [0.70, 0.60, 0.50, 0.50, 0.40, 0.40, 0.30]
    steps6 = []
    for i in range(8000):
        tool = i % 2 == 0
        cut = rng.randint(3, 6) if rng.random() < 0.25 else None
        st = synth_block_step(rng, i, 200 + i, a_tool if tool else a_text, 1000 if tool else 20000, cut=cut,
                              same=0.8 if tool else 1.0)
        st["_file"] = 0
        st["gram"] = 1 if tool else 0
        st["ng_len"] = 6 if tool else 0
        steps6.append(st)
    good6, lines6 = check(steps6)
    expect(good6, "synthetic second-block dump is consistent: " + lines6[0])
    for st in steps6:
        st["_seg"] = "tool" if st["gram"] else "text"
    rep6, tab6 = analyse_blocks(steps6, [0.08, 0.10], 0.22, 4, [4096, 16384])

    def prod(a, j):
        r_ = 1.0
        for x in a[:j]:
            r_ *= x
        return r_

    def zmax(rows, a):
        # the largest |z| of P_j, and of the conditional (its standard error from the steps still alive at j - 1)
        worst, worst_cond = 0.0, 0.0
        for j, row in enumerate(rows, 1):
            n = row["steps"]
            pj = prod(a, j)
            worst = max(worst, abs(row["cumulative"] - pj) / math.sqrt(max(pj * (1 - pj), 1e-9) / n))
            alive = max(n * prod(a, j - 1), 1.0)
            worst_cond = max(worst_cond, abs(row["conditional"] - a[j - 1]) / math.sqrt(a[j - 1] * (1 - a[j - 1]) / alive))
        return worst, worst_cond
    groups = {"tool": a_tool, "text": a_text}
    for seg_row in tab6["block_by_segment"]:
        a = groups[seg_row["segment"]]
        rows = [{"steps": seg_row[f"n{j}"], "cumulative": seg_row[f"P{j}"], "conditional": seg_row[f"a{j}"]} for j in range(1, 8)]
        z, zc = zmax(rows, a)
        expect(z < 4.5 and zc < 4.5, f"{seg_row['segment']}: P_1..P_7 and the conditionals within sampling error "
                                     f"(largest |z| {z:.2f} and {zc:.2f})")
    by_ctx = {r["context"]: r for r in tab6["block_by_ctx"]}
    expect(set(by_ctx) == {"0-4095", ">=16384"}, f"two context bins {sorted(by_ctx)}")
    lo = by_ctx["0-4095"]
    rows_lo = [{"steps": lo[f"n{j}"], "cumulative": lo[f"P{j}"], "conditional": lo[f"a{j}"]} for j in range(1, 8)]
    z, zc = zmax(rows_lo, a_tool)
    expect(z < 4.5 and zc < 4.5, f"the short-context bin carries the tool tail (largest |z| {z:.2f} and {zc:.2f})")
    pr = {r["position"]: r for r in tab6["block"]}
    expect(all(abs(pr[j]["predicted"] - 0.5 * (prod(a_tool, j) + prod(a_text, j))) < 0.02 for j in range(1, 8)),
           "the selector's path probability equals the true P_j (mean of the two groups)")
    expect(all(pr[j]["censored"] > 0 for j in range(4, 8)) and all(pr[j]["censored"] == 0 for j in range(1, 4)),
           "a request's end censors positions 4 to 7 only")
    expect(all(abs(pr[j]["strict_cumulative"] - 0.5 * (prod(a_tool, j) + prod(a_text, j))) < 0.03 for j in range(1, 8)),
           "P_j on the steps with a full continuation agrees with P_j on all steps with the position known")
    wd6 = tab6["block_width"]
    expect(abs(wd6[0]["same"] - 0.9) < 0.02, f"the draft equals the second block's walk on 0.9 of the steps ({wd6[0]['same']:.3f})")
    sp6 = {r["k"]: r for r in tab6["block_speed"]}
    tau_true = 1 + sum(0.5 * (prod(a_tool, j) + prod(a_text, j)) for j in range(1, 7))
    expect(abs(sp6[6]["tau"] - tau_true) < 0.06, f"tau at 6 drafted tokens {sp6[6]['tau']:.3f} against {tau_true:.3f}")
    ref = {r["tail"]: r for r in tab6["block_vs_reference"]}
    expect({"measured", "note's est. (rig-anchored)", "INT4 port's collapse"} <= set(ref)
           and ref["note's est. (rig-anchored)"]["best_k_r0.08"] >= 5 and ref["INT4 port's collapse"]["best_k_r0.08"] <= 4,
           "the reference tails: the note's est. drafts 5 or more at r = 0.08, the collapsing tail 4 or fewer")

    # 7. the marker helper reads a vocabulary from a GGUF header: a tiny file with scalar, numeric-array and string-array
    #    values around the tokenizer's
    import os
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gguf_markers

    def gstr(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def kv(key, t, payload):
        return gstr(key) + struct.pack("<I", t) + payload

    vocab = ["a", "<think>", "</think>", "<tool_call>", "</tool_call>", "b"]
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 5)
    blob += kv("general.architecture", 8, gstr("qwen4exp"))
    blob += kv("x.layers", 4, struct.pack("<I", 7))
    blob += kv("x.floats", 9, struct.pack("<I", 6) + struct.pack("<Q", 3) + struct.pack("<3f", 1, 2, 3))
    blob += kv("tokenizer.ggml.tokens", 9, struct.pack("<I", 8) + struct.pack("<Q", len(vocab)) + b"".join(gstr(t) for t in vocab))
    blob += kv("tokenizer.ggml.token_type", 9, struct.pack("<I", 5) + struct.pack("<Q", len(vocab)) + struct.pack(f"<{len(vocab)}i", *range(6)))
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tf:
        tf.write(blob)
    try:
        toks = gguf_markers.vocabulary(tf.name)
    finally:
        os.unlink(tf.name)
    expect(toks == vocab and [toks.index(t) for t in ("<tool_call>", "</tool_call>", "<think>", "</think>")] == [3, 4, 1, 2],
           "gguf_markers reads the vocabulary from a GGUF header past scalar and numeric-array values")

    # 8. trees for a sequential drafter (MTP), verified by rejection: siblings and full trees on two hand-made steps,
    #    then random steps whose target accepts each walk token with 0.6 and, on a rejection, takes the best candidate
    #    other than the walk's with 0.5, the second with 0.2 and a token outside the candidates with 0.3
    def mtp(task, pick, truth, n_acc):
        cand = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        p = [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.7, 0.2, 0.1]]
        return {"v": 1, "step": 0, "task": task, "slot": 0, "pos": 100, "n_prompt": 50, "verify": "reject",
                "type": "draft-mtp", "last": 5, "draft": [cand[i][b] for i, b in enumerate(pick)],
                "sampled": truth[:n_acc + 1], "n_acc": n_acc, "lattice": False, "k": 3, "cand": cand, "p": p,
                "pick": pick, "truth": truth, "_file": 0}

    m1 = mtp(1, [0, 0, 0], [11, 22, 40, 41], 1)
    m2 = mtp(2, [2, 0, 0], [11, 21, 31, 41], 0)
    good8, lines8 = check([m1, m2])
    expect(good8, "hand-made MTP steps are consistent: " + lines8[0])
    l1, l2 = Lattice(m1, "factorized"), Lattice(m2, "factorized")
    expect(tau_siblings(l1, m1["_y"], 2, 2, 3) == (3, 1), "a sibling at position 2 takes the target's token: tau 3")
    expect(tau_siblings(l1, m1["_y"], 2, 1, 3) == (2, 0), "with a sibling at position 1 only, the step keeps tau 2")
    expect(tau_siblings(l2, m2["_y"], 2, 1, 3) == (2, 1),
           "a walk token drafted outside the top 2 has the best candidate as its sibling")
    expect(tau_full(l1, m1["_y"], 2, 2, 3) == 3, "the full 2x2 tree accepts the off-walk path (11, 22)")
    expect(tau_full(l2, m2["_y"], 2, 2, 3) == 3, "the full 2x2 tree keeps the walk's token at position 1 and stops at 2")
    expect(_acc(l1, m1["_y"]) == 1 and tau_best_first(l1, m1["_y"], 2) == 2, "the walk accepts 1; best-first of 2 nodes too")
    rng8 = random.Random(88)
    st8 = []
    for i in range(4000):
        pick = [rng8.choice((0, 0, 0, 1, 2)) for _ in range(3)]
        cand = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        truth, n_acc = [], 3
        for j in range(3):
            if rng8.random() < 0.6:
                truth.append(cand[j][pick[j]])
                continue
            others = [cand[j][b] for b in range(3) if b != pick[j]]
            u = rng8.random()
            truth.append(others[0] if u < 0.5 else (others[1] if u < 0.7 else 999))
            n_acc = j
            break
        truth += [600 + rng8.randrange(50) for _ in range(4 - len(truth))]
        st8.append(mtp(1000 + i, pick, truth, n_acc))
    good8b, lines8b = check(st8)
    expect(good8b, "random MTP steps are consistent: " + lines8b[0])
    tab8 = analyse_shapes(st8, parse_shapes("auto", 3), [0.1], 0.025)
    sib = {r["position"]: r for r in tab8["siblings"]}
    for j in (1, 2, 3):
        n8 = sib[j]["rejections"]
        for k, want in ((1, 0.5), (2, 0.7)):
            got = sib[j][f"in_top{k}_others"]
            z = abs(got - want) / math.sqrt(want * (1 - want) / max(n8, 1))
            expect(z < 4.0, f"position {j}: the target's token among the {k} best other candidates on "
                            f"{got:.3f} of {n8} rejections (want {want}, |z| {z:.2f})")
    row8 = [r for r in tab8["shapes"] if r["shape"] == "siblings 2x3 on 3"][0]
    want8 = 1 + (0.24 + 2 * 0.144 + 3 * 0.216) + (1 - 0.216) * 0.5
    expect(abs(row8["tau_tree"] - want8) < 0.06 and row8["nodes"] == 6,
           f"siblings 2x3 on a walk of 3: tau {row8['tau_tree']:.3f} against {want8:.3f}, 6 nodes")
    expect(all(r["steps"] == 0 or r["tau_tree"] >= r["tau_walk_L"] - 1e-9 for r in tab8["shapes"]),
           "a tree never accepts less than its own walk")
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
    ap.add_argument("--lattice", choices=["auto", "fb", "prod"], default="auto",
                    help="the lattice the tables walk: the second block of LLAMA_SPEC_DUMP_FULL_BLOCK (fb), the draft's "
                         "own (prod), or the second block where a step has one (auto, the default)")
    for name in ("tool-open", "tool-close", "think-open", "think-close"):
        ap.add_argument(f"--{name}", type=int, default=None, metavar="TOKEN_ID",
                        help="token id of the marker that opens or closes a tool call / the reasoning; with them the "
                             "tables split by segment (gguf_markers.py prints them for a model)")
    ap.add_argument("--no-target", action="store_true", help="leave out the expected values from the target's rows")
    ap.add_argument("--shapes", default="auto",
                    help="trees for a sequential drafter, against the chain of as many verified tokens: 'auto' (width 2 "
                         "and 3 over the first 1 to 3 positions, every walk length), 'none', or a list wxmxL,...")
    ap.add_argument("--pass-cost", type=float, default=0.025,
                    help="one drafting pass of a sequential drafter, in one-token steps, for the shapes' S (default "
                         "0.025)")
    ap.add_argument("--check", action="store_true", help="only check the dumps' consistency")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dumps:
        ap.error("no dump files")
    markers = {"tool_open": args.tool_open, "tool_close": args.tool_close,
               "think_open": args.think_open, "think_close": args.think_close}
    steps = load_steps(args.dumps)
    good, lines = check(steps, markers=markers)
    if args.check:
        print("\n".join(lines))
        return 0 if good else 1
    budgets = sorted(int(x) for x in args.budgets.split(","))
    rs = [float(x) for x in args.r.split(",")]
    ctx_bins = [int(x) for x in args.ctx_bins.split(",")]
    report, tables = analyse(steps, budgets, args.mode, rs, args.draft_cost, args.copy_min, ctx_bins,
                             tgt=not args.no_target, source=args.lattice)
    breport, btables = analyse_blocks(steps, rs, args.draft_cost, args.copy_min, ctx_bins, source=args.lattice,
                                      mode=args.mode)
    text = summary_md(report, tables, args, rs)
    if btables:
        text += "\n" + block_summary_md(breport, btables, args, rs)
    stables = {}
    if args.shapes != "none" and tables:
        D = max(r["position"] for r in tables["chain"])
        stables = analyse_shapes(steps, parse_shapes(args.shapes, D), rs, args.pass_cost, source=args.lattice)
        if stables:
            text += "\n" + shapes_summary_md(stables, args, rs)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "check.txt").write_text("\n".join(lines) + "\n")
        (out / "summary.md").write_text(text)
        for name, rows in list(tables.items()) + list(btables.items()) + list(stables.items()):
            write_csv(out / f"{name}.csv", rows)
    print(text)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
