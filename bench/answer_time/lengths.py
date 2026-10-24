#!/usr/bin/env python3
"""Does a quantized file make Flint reason longer, and what does that do to the time per correct answer?

Reads Q41's task runs (bench/bfcl_flint single- and multi-turn, bench/livecodebench_flint): the raw llama-server
outputs (`raw/<lane>-<file>[-a<attempt>-p<i>of<n>].jsonl`) and the scored per-attempt tables (`out/*/per_attempt.csv`)
of one or more data folders. For every sample it counts reasoning tokens (`<think>` ... the last `</think>`, with
Flint's tokenizer, see flint_tokens.py) and answer tokens (llama-server's `tokens_predicted` minus the reasoning), and
for multi-turn conversations the same per step, per user turn and per conversation, plus the new prompt tokens of
every step.

Against a reference file (UD-Q4_K_XL), per lane and paired by task (each file's mean over its attempts first):
- the ratio of means and the median per-task ratio of total, reasoning and answer tokens, with 95 % bootstrap
  intervals that resample clusters of tasks (BFCL live sources, multi-turn scenarios, LiveCodeBench contests);
- the same with runaway samples left out, quantile ratios (p50 ... p99) and the share of the length difference that
  sits in tasks both files solve, both fail, or split on;
- runaway rates (stopped by the token limit, reasoning never closed);
- the noise floor: one file's attempt a against its attempt b, analysed as if they were two files;
- with --time, a time-per-answer model (see TIME MODEL below): seconds per sample and per correct answer, and their
  ratio to the reference.

Data folders are pooled per lane when their (file, task, attempt) keys don't collide; otherwise the later folder's
lane is kept apart as `<lane>@<folder>`.

usage: python lengths.py --tokenizer tokenizer.json [--ref ud] [--alias ud-q4kxl=ud] [--bootstrap 2000]
                         [--time [--token-ms FILE=MS ...] [model options]] [--json OUT] DATA_DIR [DATA_DIR ...]
Standard library only.

TIME MODEL (all times est.; the defaults are this rig's numbers, see docs/research/notes/answer_time.md):
  a generated token at context depth d costs  token_ms[file] + serve_ms + host_ms (tool lanes) + depth_ms * d / 1000
  a request costs  request_s + new_prompt_tokens * prefill_ms / 1000  before its first token
  multi-turn: new prompt tokens are the step's prompt minus the previous step's prompt (policy "today": llama-server
  re-reads the previous step's output, from 4 tokens before the previous prompt's end) or minus the previous prompt
  and output (policy "exact": the generated turn is reused).
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint_tokens import FlintTokenizer  # noqa: E402

LANES = {"bfcl_single_turn": "e1", "bfcl_multi_turn": "e2", "livecodebench": "e3"}
NAME = {"e1": "BFCL v4 single-turn", "e2": "BFCL v4 multi-turn", "e3": "LiveCodeBench"}
TOOL_LANES = {"e1", "e2"}
# Words that open a hesitation or re-check: the marker class whose sampling probability quantization noise raises at
# high-entropy positions in quantized reasoning models (arXiv 2606.00206). Counted word-initially in the reasoning.
MARKERS = re.compile(r"\b(wait|but|alternatively|perhaps|maybe|however|hmm+|actually|reconsider|backtrack|wrong|"
                     r"double-check|recheck|re-check|verify|mistake|oops|hold on)\b", re.I)
# Decode ms per token at depth 0, six cards, llama-bench tg128 on the fork's v2 build: UD-Q4_K_XL measured (Q76);
# the others from bytes per token, 12.0 ms + GB / 151 GB/s, the line through S4-L0's measured 26.8 ms at 2.236 GB.
DEFAULT_TOKEN_MS = {"ud": 26.84, "g3": 27.40, "g3h": 26.63, "k2h": 26.16, "k1h": 25.86}


def split(text):
    """(reasoning text with its tags, closed). Reasoning runs to the last </think>, as bench/bfcl_flint counts it."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[0] + "</think>", True
    if text.lstrip().startswith("<think>"):
        return text, False
    return "", True


class Gen:
    """One generation: total, reasoning and answer tokens, flags."""
    __slots__ = ("total", "reason", "answer", "closed", "limit", "extra_close", "prompt", "markers")

    def __init__(self, r, tok):
        text = r.get("content") or ""
        rtext, self.closed = split(text)
        self.total = int(r.get("tokens_predicted") or 0)
        self.reason = min(tok.count(rtext), self.total) if rtext else 0
        self.answer = self.total - self.reason
        self.markers = len(MARKERS.findall(rtext)) if rtext else 0
        self.limit = r.get("stop_type") == "limit" or bool(r.get("truncated") and r.get("stop_type") != "eos")
        self.extra_close = text.count("</think>") > 1
        self.prompt = int(r.get("tokens_evaluated") or r.get("n_prompt") or 0)


class Sample:
    """One attempt at one task: a generation (single-turn, coding) or a conversation (multi-turn)."""

    def __init__(self, lane, file, rec, tok):
        self.lane, self.file, self.task, self.attempt = lane, file, rec["id"], int(rec["attempt"])
        self.category, self.cluster, self.passed = rec.get("category", ""), rec["id"], None
        if lane == "e2":
            self.steps = []  # (turn index, Gen)
            for ti, steps in enumerate(rec.get("turns", [])):
                for st in steps:
                    self.steps.append((ti, Gen(st, tok)))
            gens = [g for _, g in self.steps]
            self.total = sum(g.total for g in gens)
            self.reason = sum(g.reason for g in gens)
            self.answer = self.total - self.reason
            self.markers = sum(g.markers for g in gens)
            self.runaway = any(g.limit or not g.closed for g in gens) or bool(rec.get("force_quit"))
            self.limit = any(g.limit for g in gens)
            self.n_steps = len(gens)
            self.n_turns = len(rec.get("turns", []))
            self.prompt = gens[0].prompt if gens else 0
            self.max_context = max((g.prompt + g.total for g in gens), default=0)
        else:
            g = Gen(rec, tok)
            self.gen = g
            self.total, self.reason, self.answer, self.markers = g.total, g.reason, g.answer, g.markers
            self.runaway = g.limit or not g.closed
            self.limit = g.limit
            self.prompt = g.prompt
            self.extra_close = g.extra_close

    def new_prompt(self, policy):
        """Prompt tokens to process per step under a cache policy (multi-turn only)."""
        out, prev = [], None
        for _, g in self.steps:
            if prev is None:
                out.append(g.prompt)
            elif policy == "exact":
                out.append(max(g.prompt - prev.prompt - prev.total, 0))
            else:
                out.append(max(g.prompt - prev.prompt + 4, 0))
            prev = g
        return out


def parse_name(path):
    m = re.match(r"(e\d)-(.+?)(?:-a\d+(?:-p\d+of\d+)?)?\.jsonl$", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else (None, None)


def load(dirs, tok, alias, keep_unscored=False):
    lanes = {}  # lane key -> list of Sample
    for d in dirs:
        verdicts = {}
        for path in glob.glob(os.path.join(d, "out", "*", "per_attempt.csv")):
            for r in csv.DictReader(open(path)):
                lane = LANES.get(r["lane"])
                f = alias.get(r["file"], r["file"])
                verdicts[(lane, f, r["task_id"], int(r["attempt"]))] = r
        here = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(d, "raw", "*.jsonl"))):
            lane, f = parse_name(path)
            if lane is None:
                continue
            f = alias.get(f, f)
            for line in open(path):
                rec = json.loads(line)
                s = Sample(lane, f, rec, tok)
                v = verdicts.get((lane, f, s.task, s.attempt))
                if v:
                    s.passed = int(v["pass"])
                    s.category = v["category"] or s.category
                    s.cluster = v["cluster"] or s.task
                here[lane].append(s)
        if verdicts and not keep_unscored:
            for lane in here:
                n = len(here[lane])
                here[lane] = [s for s in here[lane] if s.passed is not None]
                if len(here[lane]) < n:
                    print(f"note: {d}: {n - len(here[lane])} {lane} samples without a verdict left out", file=sys.stderr)
        tag = os.path.basename(os.path.normpath(d))
        for lane, samples in here.items():
            keys = {(s.file, s.task, s.attempt) for s in samples}
            if lane in lanes and keys & {(s.file, s.task, s.attempt) for s in lanes[lane]}:
                lanes[f"{lane}@{tag}"] = samples
            else:
                lanes.setdefault(lane, []).extend(samples)
    return lanes


# ---------------------------------------------------------------- statistics

def by_task(samples, fn):
    """file -> task -> list of fn(sample), in attempt order."""
    tmp = defaultdict(lambda: defaultdict(list))
    for s in samples:
        v = fn(s)
        if v is not None:
            tmp[s.file][s.task].append((s.attempt, v))
    out = defaultdict(lambda: defaultdict(list))
    for f, tasks in tmp.items():
        for t, vals in tasks.items():
            out[f][t] = [v for _, v in sorted(vals, key=lambda x: x[0])]
    return out


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def boot_ci(clusters, stat, B, seed=1):
    """95 % percentile interval of stat(list of items) over cluster resamples; clusters = list of item lists."""
    rng = random.Random(seed)
    n = len(clusters)
    vals = []
    if stat is ratio_of_means and n:
        sums = [(sum(x[1] for x in c), sum(x[2] for x in c)) for c in clusters]
        for _ in range(B):
            a = b = 0.0
            for _ in range(n):
                x, y = sums[rng.randrange(n)]
                a += x
                b += y
            if a:
                vals.append(b / a)
        vals.sort()
        if not vals:
            return float("nan"), float("nan")
        return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    for _ in range(B):
        items = []
        for _ in range(n):
            items.extend(clusters[rng.randrange(n)])
        v = stat(items)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]


def ratio_of_means(items):
    a = sum(x[1] for x in items)
    return sum(x[2] for x in items) / a if a else float("nan")


def median_ratio(items):
    r = [x[2] / x[1] for x in items if x[1] > 0 and x[2] > 0]
    return statistics.median(r) if r else float("nan")


def paired_items(samples, ref, cand, fn, clusters, match=False):
    """[(cluster, ref per-task mean, cand per-task mean)] over tasks both files have. With match, each file's mean
    is over the same number of attempts (its first ones): a median of per-task ratios is biased when one side
    averages more attempts than the other, because single draws are more skewed than means."""
    t = by_task(samples, fn)
    a, b = t.get(ref, {}), t.get(cand, {})
    out = []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if match:
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
        out.append((clusters[k], mean(x), mean(y)))
    return out


def grouped(items):
    g = defaultdict(list)
    for it in items:
        g[it[0]].append(it)
    return list(g.values())


def fmt_ratio(x, lo, hi):
    return f"{x:.3f} ({lo:.3f}–{hi:.3f})"


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = math.floor(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


# ---------------------------------------------------------------- report

def files_of(samples, ref):
    fs = sorted({s.file for s in samples})
    return [ref] + [f for f in fs if f != ref] if ref in fs else fs


def pseudo(samples, f, a):
    """One file's attempt a as its own pseudo-file 'f#a' (noise floor)."""
    out = []
    for s in samples:
        if s.file == f and s.attempt == a:
            c = Sample.__new__(Sample)
            c.__dict__.update(s.__dict__)
            c.file = f"{f}#{a}"
            out.append(c)
    return out


def describe(lane, samples, ref, B, out):
    clusters = {s.task: s.cluster for s in samples}
    files = files_of(samples, ref)
    out.append(f"\n## {NAME.get(lane.split('@')[0], lane)} ({lane})\n")
    out.append("| File | samples | tasks | attempts | mean total | mean reasoning | mean answer | reasoning share | "
               "median total | p90 | p99 | runaway | at token limit | pass@1 |")
    out.append("|---|" + "---|" * 13)
    for f in files:
        ss = [s for s in samples if s.file == f]
        tot = [s.total for s in ss]
        rea = sum(s.reason for s in ss)
        att = sorted({s.attempt for s in ss})
        pv = [s.passed for s in ss if s.passed is not None]
        pstr = f"{100 * mean(pv):.1f} %" if pv else "—"
        out.append(f"| {f} | {len(ss)} | {len({s.task for s in ss})} | {','.join(map(str, att))} | "
                   f"{mean(tot):.0f} | {rea / len(ss):.0f} | {mean([s.answer for s in ss]):.0f} | "
                   f"{rea / max(sum(tot), 1):.3f} | {quantile(tot, .5):.0f} | {quantile(tot, .9):.0f} | "
                   f"{quantile(tot, .99):.0f} | {100 * mean([s.runaway for s in ss]):.2f} % | "
                   f"{100 * mean([s.limit for s in ss]):.2f} % | {pstr} |")
    # pairs: each candidate against ref (all attempts), plus noise floors
    pairs = [(samples, ref, f, f"{f} vs {ref}") for f in files if f != ref]
    for f in files:
        att = sorted({s.attempt for s in samples if s.file == f})
        if len(att) >= 2:
            a, b = att[0], att[1]
            ps = pseudo(samples, f, a) + pseudo(samples, f, b)
            pairs.append((ps, f"{f}#{a}", f"{f}#{b}", f"noise: {f} attempt {b} vs {a}"))
    if len([f for f in files if f != ref]) and len({s.attempt for s in samples if s.file == ref}) >= 2:
        a = min(s.attempt for s in samples if s.file == ref)
        for f in files:
            if f != ref and a in {s.attempt for s in samples if s.file == f}:
                ps = pseudo(samples, ref, a) + pseudo(samples, f, a)
                pairs.append((ps, f"{ref}#{a}", f"{f}#{a}", f"{f} vs {ref}, attempt {a} only"))
    out.append("\nPaired by task, 95 % cluster-bootstrap intervals. Ratio of means = sum of per-task means; median = "
               "median per-task ratio over equal numbers of attempts. 'No runaway' drops samples stopped by the limit or with open reasoning.\n")
    out.append("| Pair | tasks | total: ratio of means | total: median ratio | reasoning: ratio of means | "
               "reasoning: median ratio | answer: ratio of means | total, no runaway: ratio of means |")
    out.append("|---|---|---|---|---|---|---|---|")
    res = {}
    for ps, a, b, label in pairs:
        row = [label]
        n = None
        for fn, kind in ((lambda s: s.total, "rom"), (lambda s: s.total, "med"), (lambda s: s.reason, "rom"),
                         (lambda s: s.reason, "med"), (lambda s: s.answer, "rom"),
                         (lambda s: None if s.runaway else s.total, "rom")):
            items = paired_items(ps, a, b, fn, clusters, match=(kind == "med"))
            n = n or len(items)
            st = ratio_of_means if kind == "rom" else median_ratio
            x = st(items)
            lo, hi = boot_ci(grouped(items), st, B)
            row.append(fmt_ratio(x, lo, hi))
            res.setdefault(label, []).append((x, lo, hi))
        out.append(f"| {row[0]} | {n} | " + " | ".join(row[1:]) + " |")
    # hesitation-marker density
    out.append("\nHesitation markers (wait, but, maybe, perhaps, however, alternatively, actually, ...) per 1,000 "
               "reasoning tokens, and the candidate's rate over the reference's (95 % cluster-bootstrap interval):\n")
    out.append("| Pair | tasks | reference rate | candidate rate | ratio |")
    out.append("|---|---|---|---|---|")
    for ps, a, b, label in pairs:
        M = by_task(ps, lambda s: s.markers)
        Rt = by_task(ps, lambda s: s.reason)
        tasks = sorted(set(M.get(a, {})) & set(M.get(b, {})))
        items = [(clusters[t], sum(M[a][t]) / len(M[a][t]), sum(Rt[a][t]) / len(Rt[a][t]),
                  sum(M[b][t]) / len(M[b][t]), sum(Rt[b][t]) / len(Rt[b][t])) for t in tasks]
        if not items:
            continue

        def rr(it):
            ra = sum(x[1] for x in it) / max(sum(x[2] for x in it), 1)
            rb = sum(x[3] for x in it) / max(sum(x[4] for x in it), 1)
            return rb / ra if ra else float("nan")
        lo, hi = boot_ci(grouped(items), rr, max(B // 2, 200))
        ra = 1000 * sum(x[1] for x in items) / max(sum(x[2] for x in items), 1)
        rb = 1000 * sum(x[3] for x in items) / max(sum(x[4] for x in items), 1)
        out.append(f"| {label} | {len(items)} | {ra:.2f} | {rb:.2f} | {fmt_ratio(rr(items), lo, hi)} |")
    # quantiles of per-sample totals
    out.append("\nQuantiles of total tokens per sample (ratio to the reference, 95 % cluster-bootstrap interval):\n")
    qs = (.5, .75, .9, .95, .99)
    out.append("| Pair | " + " | ".join(f"p{int(q * 100)}" for q in qs) + " |")
    out.append("|---|" + "---|" * len(qs))
    for ps, a, b, label in pairs[:len(files) - 1] + [p for p in pairs if p[3].startswith("noise")]:
        per = defaultdict(lambda: ([], []))
        for s in ps:
            if s.file == a:
                per[s.task][0].append(s.total)
            elif s.file == b:
                per[s.task][1].append(s.total)
        cl = defaultdict(list)
        for t, (x, y) in per.items():
            if x and y:
                cl[clusters[t]].append((x, y))
        groups = list(cl.values())
        cells = []
        for q in qs:
            def st(items, q=q):
                xa = [v for it in items for v in it[0]]
                xb = [v for it in items for v in it[1]]
                qa = quantile(xa, q)
                return quantile(xb, q) / qa if qa else float("nan")
            allit = [it for g in groups for it in g]
            lo, hi = boot_ci(groups, st, max(B // 4, 200))
            cells.append(fmt_ratio(st(allit), lo, hi))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    # where the difference sits
    out.append("\nWhere the total-token difference sits: contribution of each class of tasks to the mean per-task "
               "difference (candidate minus reference, tokens), tasks classed by both files' per-task pass rates:\n")
    out.append("| Pair | mean difference (tokens per task) | both always pass | both never pass | split | "
               "tasks with a runaway sample | top decile of the reference's length | ratio on tasks both always pass |")
    out.append("|---|---|---|---|---|---|---|---|")
    for ps, a, b, label in pairs:
        T = by_task(ps, lambda s: s.total)
        P = by_task(ps, lambda s: s.passed)
        R = by_task(ps, lambda s: int(s.runaway))
        tasks = sorted(set(T.get(a, {})) & set(T.get(b, {})))
        if not tasks:
            continue
        d = {t: mean(T[b][t]) - mean(T[a][t]) for t in tasks}
        tot = sum(d.values())
        cls = defaultdict(float)
        for t in tasks:
            pa, pb = mean(P.get(a, {}).get(t, [])), mean(P.get(b, {}).get(t, []))
            k = "pass" if pa == 1 and pb == 1 else "fail" if pa == 0 and pb == 0 else "split"
            cls[k] += d[t]
            if max(R[a][t]) or max(R[b][t]):
                cls["run"] += d[t]
        ref_len = sorted(tasks, key=lambda t: mean(T[a][t]))
        top = ref_len[int(0.9 * len(ref_len)):]
        cls["top"] = sum(d[t] for t in top)
        both = [t for t in tasks if mean(P.get(a, {}).get(t, [0])) == 1 and mean(P.get(b, {}).get(t, [0])) == 1]
        rb = sum(mean(T[b][t]) for t in both) / max(sum(mean(T[a][t]) for t in both), 1)
        sh = lambda k: f"{cls[k] / len(tasks):+.1f}"
        out.append(f"| {label} | {tot / len(tasks):+.1f} | {sh('pass')} | {sh('fail')} | {sh('split')} | "
                   f"{sh('run')} | {sh('top')} | {rb:.3f} ({len(both)} tasks) |")
    # length of passing against failing samples, per file
    out.append("\nLength of failed against passed samples, per file (mean total tokens):\n")
    out.append("| File | passed | failed | failed / passed | runaway samples among failures |")
    out.append("|---|---|---|---|---|")
    for f in files:
        ss = [s for s in samples if s.file == f and s.passed is not None]
        ok = [s.total for s in ss if s.passed]
        bad = [s for s in ss if not s.passed]
        if ok and bad:
            out.append(f"| {f} | {mean(ok):.0f} | {mean([s.total for s in bad]):.0f} | "
                       f"{mean([s.total for s in bad]) / mean(ok):.2f} | {sum(s.runaway for s in bad)} of {len(bad)} |")
    # by category
    cats = sorted({s.category for s in samples})
    if len(cats) > 1 and len(files) > 1:
        out.append("\nBy category: ratio of means of total and reasoning tokens (95 % interval).\n")
        cands = [f for f in files if f != ref]
        out.append("| Category | tasks | " + " | ".join(f"{c} / {ref} total" for c in cands) + " | " +
                   " | ".join(f"{c} / {ref} reasoning" for c in cands) + " |")
        out.append("|---|---|" + "---|" * (2 * len(cands)))
        for c in cats:
            sub = [s for s in samples if s.category == c]
            cells_t, cells_r, n = [], [], 0
            for f in cands:
                for fn, cells in ((lambda s: s.total, cells_t), (lambda s: s.reason, cells_r)):
                    items = paired_items(sub, ref, f, fn, clusters)
                    n = max(n, len(items))
                    lo, hi = boot_ci(grouped(items), ratio_of_means, max(B // 4, 200))
                    cells.append(fmt_ratio(ratio_of_means(items), lo, hi))
            out.append(f"| {c} | {n} | " + " | ".join(cells_t + cells_r) + " |")
    return res


def describe_turns(lane, samples, ref, B, out):
    """Multi-turn only: per user turn and per conversation."""
    clusters = {s.task: s.cluster for s in samples}
    files = files_of(samples, ref)
    turns = []
    for s in samples:
        agg = defaultdict(lambda: [0, 0, 0])
        for ti, g in s.steps:
            agg[ti][0] += g.total
            agg[ti][1] += g.reason
            agg[ti][2] += 1
        for ti, (t, r, n) in agg.items():
            c = Sample.__new__(Sample)
            c.__dict__.update(s.__dict__)
            c.task, c.total, c.reason, c.n_steps = f"{s.task}#{ti}", t, r, n
            turns.append(c)
            clusters[c.task] = s.cluster
    out.append("\nMulti-turn per user turn (turns paired by task and turn index) and per conversation:\n")
    out.append("| Pair | unit | units | total: ratio of means | reasoning: ratio of means | steps: ratio of means | "
               "new prompt tokens (exact reuse): ratio |")
    out.append("|---|---|---|---|---|---|---|")
    for f in files:
        if f == ref:
            continue
        for unit, ss in (("turn", turns), ("conversation", samples)):
            cells, n = [], 0
            fns = [lambda s: s.total, lambda s: s.reason, lambda s: s.n_steps]
            if unit == "conversation":
                fns.append(lambda s: sum(s.new_prompt("exact")))
            for fn in fns:
                items = paired_items(ss, ref, f, fn, clusters)
                n = max(n, len(items))
                lo, hi = boot_ci(grouped(items), ratio_of_means, B)
                cells.append(fmt_ratio(ratio_of_means(items), lo, hi))
            if unit == "turn":
                cells.append("—")
            out.append(f"| {f} vs {ref} | {unit} | {n} | " + " | ".join(cells) + " |")
    out.append("\n| File | conversations | steps per conversation | total tokens per step | reasoning per step | "
               "first prompt | new prompt tokens per conversation, exact reuse / today | context reached (median) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for f in files:
        ss = [s for s in samples if s.file == f]
        steps = sum(s.n_steps for s in ss)
        out.append(f"| {f} | {len(ss)} | {steps / len(ss):.1f} | {sum(s.total for s in ss) / steps:.0f} | "
                   f"{sum(s.reason for s in ss) / steps:.0f} | {mean([s.prompt for s in ss]):.0f} | "
                   f"{mean([sum(s.new_prompt('exact')) for s in ss]):.0f} / "
                   f"{mean([sum(s.new_prompt('today')) for s in ss]):.0f} | "
                   f"{statistics.median([s.max_context for s in ss]):.0f} |")


# ---------------------------------------------------------------- time model

def decode_s(L, depth0, ms, depth_ms):
    """Seconds to generate L tokens starting at context depth0."""
    return (L * ms + depth_ms * (depth0 * L + L * (L - 1) / 2) / 1000) / 1000


def sample_time(s, a, policy):
    ms = a.token_ms[s.file] + a.serve_ms + (a.host_ms if s.lane.split("@")[0] in TOOL_LANES else 0)
    if s.lane.startswith("e2"):
        t = 0.0
        for (_, g), new in zip(s.steps, s.new_prompt(policy)):
            t += a.request_s + new * a.prefill_ms / 1000 + decode_s(g.total, g.prompt, ms, a.depth_ms)
        return t
    return a.request_s + s.prompt * a.prefill_ms / 1000 + decode_s(s.total, s.prompt, ms, a.depth_ms)


def describe_time(lane, samples, ref, a, B, out):
    clusters = {s.task: s.cluster for s in samples}
    files = [f for f in files_of(samples, ref) if f in a.token_ms]
    if ref not in files:
        return
    policies = ["today", "exact"] if lane.startswith("e2") else ["—"]
    out.append(f"\nTime model, {lane}: seconds per sample and per correct answer (sum of per-task mean time over sum "
               "of per-task pass@1), ratio to the reference with a 95 % cluster-bootstrap interval.\n")
    out.append("| File | token ms (+ serve, host) | cache policy | s per sample | s per correct | "
               "time per sample / ref | time per correct / ref | of which length × speed |")
    out.append("|---|---|---|---|---|---|---|---|")
    for pol in policies:
        T = by_task(samples, lambda s: sample_time(s, a, pol))
        P = by_task(samples, lambda s: s.passed)
        for f in files:
            tasks = sorted(set(T[ref]) & set(T[f]) & set(P.get(ref, {})) & set(P.get(f, {})))
            items = [(clusters[t], mean(T[ref][t]), mean(T[f][t]), mean(P[ref][t]), mean(P[f][t])) for t in tasks]
            if not items:
                continue
            per_s = lambda it: sum(x[2] for x in it) / sum(x[1] for x in it)
            per_c = lambda it: (sum(x[2] for x in it) / max(sum(x[4] for x in it), 1e-9)) / \
                               (sum(x[1] for x in it) / max(sum(x[3] for x in it), 1e-9))
            g = grouped(items)
            s_lo, s_hi = boot_ci(g, per_s, B)
            c_lo, c_hi = boot_ci(g, per_c, B)
            n = len(items)
            ext = a.serve_ms + (a.host_ms if lane.split("@")[0] in TOOL_LANES else 0)
            out.append(f"| {f} | {a.token_ms[f]:.2f} (+{ext:.1f}) | {pol} | {sum(x[2] for x in items) / n:.1f} | "
                       f"{sum(x[2] for x in items) / max(sum(x[4] for x in items), 1e-9):.1f} | "
                       f"{fmt_ratio(per_s(items), s_lo, s_hi)} | {fmt_ratio(per_c(items), c_lo, c_hi)} | "
                       f"{per_s(items):.3f} × pass ratio {sum(x[3] for x in items) / max(sum(x[4] for x in items), 1e-9):.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="data folders with raw/*.jsonl and out/*/per_attempt.csv")
    ap.add_argument("--tokenizer", required=True, help="Flint's tokenizer.json")
    ap.add_argument("--ref", default="ud")
    ap.add_argument("--alias", action="append", default=[], help="OLD=NEW file label, e.g. ud-q4kxl=ud")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--token-ms", action="append", default=[],
                    help="FILE=MS decode time per token at depth 0 (default: rig estimates for ud, g3, g3h, k2h, k1h)")
    ap.add_argument("--serve-ms", type=float, default=4.5, help="served nine-card overhead per token over llama-bench")
    ap.add_argument("--host-ms", type=float, default=3.3, help="host-side sampling per token for tool requests")
    ap.add_argument("--depth-ms", type=float, default=0.3, help="extra ms per token per 1,000 tokens of context")
    ap.add_argument("--prefill-ms", type=float, default=7.1, help="ms per new prompt token")
    ap.add_argument("--request-s", type=float, default=2.2, help="fixed seconds per request")
    ap.add_argument("--keep-unscored", action="store_true",
                    help="keep samples that have no verdict in the folder's per_attempt.csv (default: leave them out)")
    ap.add_argument("--lanes", help="comma-separated lane keys to report (default: all)")
    ap.add_argument("--json")
    a = ap.parse_args()
    alias = dict(x.split("=", 1) for x in a.alias)
    overrides = {k: float(v) for k, v in (x.split("=", 1) for x in a.token_ms)}
    a.token_ms = dict(DEFAULT_TOKEN_MS, **overrides)
    tok = FlintTokenizer(a.tokenizer)
    lanes = load(a.dirs, tok, alias, a.keep_unscored)
    out, res = [], {}
    for lane in sorted(lanes):
        if a.lanes and lane not in a.lanes.split(","):
            continue
        res[lane] = describe(lane, lanes[lane], a.ref, a.bootstrap, out)
        if lane.startswith("e2"):
            describe_turns(lane, lanes[lane], a.ref, a.bootstrap, out)
        if a.time:
            describe_time(lane, lanes[lane], a.ref, a, a.bootstrap, out)
    print("\n".join(out))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
