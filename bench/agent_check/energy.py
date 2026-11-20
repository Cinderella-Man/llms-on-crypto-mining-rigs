#!/usr/bin/env python3
"""energy.py: the cards' energy over an agent-check run, from the rig's card log and the proxy's rows (FN101, Q1518).

    python3 energy.py RUN=CARDS.csv[,CARDS.csv]=PROXY.jsonl[=ARM] [RUN=…] [--skip TASK,…] [--shift S]
                      [--tsv OUT.tsv] [--cards-tsv OUT.tsv] [--fit]

CARDS.csv is scripts/card_log.py's log (rig_server.sh writes it as cards.csv: every card's power every 0.5 s), PROXY.jsonl
the relay proxy's request rows (proxy.py). ARM picks the rows whose tag starts with that arm (default: every row with a
task tag, for a proxy log that holds one run). report.py (table, compare: --cards) and agent_tables.py import it.

How it counts:
- power: the sum of every card's g<i>_w column (NVML's power draw, in W): the cards' power, without the host's own;
- energy over a span of time: each card's power integrated with the trapezoid rule, linear between two samples; two
  samples more than GAP_S (5 s) apart (a log stopped and started again) add nothing, and that time is not covered;
- the session window: from the run's first request to its last request's end, as the proxy logged them (t_wall, the
  request's start in Unix seconds on the client's clock, and t_end, its seconds to the reply's last byte). The tools,
  the tests and the next task's start between turns lie inside it; the first task's start before its first request
  and the last task's check after its last reply, a few seconds of a session, lie outside;
- generating: every request from its first token to its end; reading the prompt: from its start to its first token (a
  request without a first token counts whole); idle between turns: the rest of the window;
- joules a task: the window's energy over the tasks (the task attempts with a request in the window); joules a
  generated token: the window's energy over the tokens generated (Σ predicted_n), so both carry the prompt reading
  and the idle time between turns, the whole cost of the work as an agent sees it;
- the watts of a part: its energy over its covered seconds, for all cards together;
- --skip leaves tasks out: their requests are not counted, and each one's span (its first request to its last
  request's end) is cut from the window;
- --shift S adds S seconds to the proxy's times before they are matched with the card log. --fit prints, per run, the
  shift between -3 and +3 s that best lines the requests up with the card power (the most watts inside requests
  against outside them): it holds the two clocks' offset, the relay's delay and the lag of NVML's power reading, and
  the split is sound when it is a small part of a turn.

Prints a Markdown table a run; --tsv writes the same as a TSV (TSV_COLUMNS), --cards-tsv one row a run and card (the
card's index in the log, never its identity). Standard library only.
"""
import argparse
import bisect
import csv
import json
import math
import re
import sys
from collections import defaultdict

GAP_S = 5.0
TAG_RE = re.compile(r"^([^/]+)/([^/]+)/a(\d+)/t\d+/\d+$")
TSV_COLUMNS = ["run", "tasks", "requests", "generated", "window_s", "covered_s", "energy_kj", "joules_per_task",
               "joules_per_token", "w_mean", "generating_s", "w_generating", "prompt_s", "w_prompt", "idle_s",
               "w_idle"]
NAN = float("nan")


class Cards:
    """The card logs' samples in time order: `t` (Unix seconds), `total` (the cards' summed W, None where a card's
    value is missing) and `per` (each card's W). Header lines may recur (a resumed run appends its own log)."""

    def __init__(self, paths):
        rows, ncards = [], 0
        for p in paths:
            head = None
            for line in open(p):
                if line.startswith("#") or not line.strip():
                    continue
                f = next(csv.reader([line]))
                if f[0] == "epoch":
                    head = {int(m.group(1)): i for i, h in enumerate(f) for m in [re.fullmatch(r"g(\d+)_w", h)] if m}
                    width = len(f)
                    ncards = max([ncards] + [c + 1 for c in head])
                    continue
                if head is None or len(f) != width:  # a line cut by a stop
                    continue
                try:
                    t = float(f[0])
                except ValueError:
                    continue
                rows.append((t, {c: float(f[i]) if f[i] else None for c, i in head.items()}))
        rows.sort(key=lambda r: r[0])
        self.ncards = ncards
        self.t = [r[0] for r in rows]
        self.per = [[r[1].get(c) for r in rows] for c in range(ncards)]
        self.total = [None if any(v is None for v in vals) else sum(vals)
                      for vals in zip(*self.per)] if rows else []
        self.missing = sum(v is None for v in self.total)
        self._int = {}

    def integral(self, which="total"):
        if which not in self._int:
            vals = self.total if which == "total" else self.per[which]
            self._int[which] = Integral(self.t, vals)
        return self._int[which]

    def covered(self):
        if "covered" not in self._int:
            self._int["covered"] = Integral(self.t, [1.0] * len(self.t))
        return self._int["covered"]


class Integral:
    """F(x): the integral of a sampled value from the first sample to x, linear between two samples; a pair of
    samples more than GAP_S apart, or with a missing value, adds nothing."""

    def __init__(self, t, v):
        self.t, self.v, self.c = t, v, [0.0]
        for k in range(len(t) - 1):
            self.c.append(self.c[-1] + self._part(k, t[k + 1]))

    def _part(self, k, x):
        t0, t1, v0, v1 = self.t[k], self.t[k + 1], self.v[k], self.v[k + 1]
        if v0 is None or v1 is None or t1 - t0 > GAP_S or t1 <= t0:
            return 0.0
        vx = v0 + (v1 - v0) * (x - t0) / (t1 - t0)
        return (x - t0) * (v0 + vx) / 2

    def __call__(self, x):
        if not self.t or x <= self.t[0]:
            return 0.0
        if x >= self.t[-1]:
            return self.c[-1]
        k = bisect.bisect_right(self.t, x) - 1
        return self.c[k] + self._part(k, x)

    def over(self, spans):
        return sum(self(b) - self(a) for a, b in spans)


def union(spans):
    out = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(spans, cut):
    """spans minus cut, both unions."""
    out = []
    for a, b in spans:
        for c, d in cut:
            if d <= a or c >= b:
                continue
            if c > a:
                out.append((a, c))
            a = max(a, d)
            if a >= b:
                break
        if a < b:
            out.append((a, b))
    return out


def intersect(spans, other):
    """The time in both, of two unions."""
    out, i, j = [], 0, 0
    while i < len(spans) and j < len(other):
        a, b = max(spans[i][0], other[j][0]), min(spans[i][1], other[j][1])
        if a < b:
            out.append((a, b))
        if spans[i][1] < other[j][1]:
            i += 1
        else:
            j += 1
    return out


def length(spans):
    return sum(b - a for a, b in spans)


def requests_from_rows(rows):
    """The proxy's rows with a task tag (dicts as proxy.py writes them): arm, task, attempt; start, first token and end
    in Unix seconds on the client's clock; tokens generated."""
    out = []
    for r in rows:
        m = TAG_RE.match(r.get("tag") or "")
        if not m or r.get("t_wall") is None:
            continue
        t0, tf = r["t_wall"], r.get("t_first_token")
        out.append(dict(arm=m.group(1), task=m.group(2), attempt=int(m.group(3)), t0=t0,
                        t1=t0 + (r.get("t_end") or tf or 0), tf=None if tf is None else t0 + tf,
                        gen=r.get("predicted_n") or 0))
    return out


def load_requests(path, arm=None):
    """requests_from_rows of a proxy log, of `arm` alone when given."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [q for q in requests_from_rows(rows) if arm is None or q["arm"] == arm]


def parts(reqs, skip=(), shift=0.0):
    """The window and its parts (unions of spans): window, generating, prompt, idle, with the counted requests."""
    keep = [q for q in reqs if q["task"] not in skip]
    if not keep:
        return None
    window = [(min(q["t0"] for q in keep) + shift, max(q["t1"] for q in keep) + shift)]
    spans = {}
    for q in reqs:
        if q["task"] in skip:
            a, b = spans.get((q["task"], q["attempt"]), (math.inf, -math.inf))
            spans[(q["task"], q["attempt"])] = (min(a, q["t0"] + shift), max(b, q["t1"] + shift))
    window = subtract(window, union(spans.values()))
    req = intersect(union([(q["t0"] + shift, q["t1"] + shift) for q in keep]), window)
    gen = intersect(union([(q["tf"] + shift, q["t1"] + shift) for q in keep if q["tf"] is not None]), window)
    return dict(keep=keep, window=window, generating=gen, prompt=subtract(req, gen), idle=subtract(window, req))


def summary(cards, reqs, skip=(), shift=0.0, which="total"):
    """A run's energy (all cards, or card `which`): the TSV_COLUMNS after `run`, as numbers; None without requests or
    card samples in the window."""
    p = parts(reqs, skip, shift)
    if p is None or not cards.t:
        return None
    e, cov = cards.integral(which), cards.covered()
    covered = cov.over(p["window"])
    if covered <= 0:
        return None
    energy = e.over(p["window"])
    tasks = len({(q["task"], q["attempt"]) for q in p["keep"]})
    gen = sum(q["gen"] for q in p["keep"])
    s = dict(tasks=tasks, requests=len(p["keep"]), generated=gen, window_s=length(p["window"]), covered_s=covered,
             energy_kj=energy / 1000, joules_per_task=energy / tasks, joules_per_token=energy / gen if gen else NAN,
             w_mean=energy / covered)
    for name in ("generating", "prompt", "idle"):
        c = cov.over(p[name])
        s[f"{name}_s"] = length(p[name])
        s[f"w_{name}"] = e.over(p[name]) / c if c > 0 else NAN
    return s


def fit(cards, reqs, skip=(), lo=-3.0, hi=3.0, step=0.05):
    """The shift (s) that gives the most watts inside requests against outside them, and that contrast (W)."""
    best = (NAN, -math.inf)
    e, cov = cards.integral(), cards.covered()
    for k in range(int(round((hi - lo) / step)) + 1):
        d = lo + k * step
        p = parts(reqs, skip, d)
        req = union(p["generating"] + p["prompt"])
        ci, co = cov.over(req), cov.over(p["idle"])
        if ci <= 0 or co <= 0:
            continue
        contrast = e.over(req) / ci - e.over(p["idle"]) / co
        if contrast > best[1]:
            best = (round(d, 3), contrast)
    return best


TSV_FORMAT = dict(tasks="{:d}", requests="{:d}", generated="{:d}", window_s="{:.1f}", covered_s="{:.1f}",
                  energy_kj="{:.2f}", joules_per_task="{:.0f}", joules_per_token="{:.2f}", w_mean="{:.1f}",
                  generating_s="{:.1f}", w_generating="{:.1f}", prompt_s="{:.1f}", w_prompt="{:.1f}",
                  idle_s="{:.1f}", w_idle="{:.1f}")


def tsv_cells(s):
    return ["nan" if isinstance(s[k], float) and math.isnan(s[k]) else TSV_FORMAT[k].format(s[k])
            for k in TSV_COLUMNS[1:]]


# report.py's per-arm columns (table, compare: --cards)
HEAD = ("card energy kJ (J a task)", "J a generated token", "W generating / reading the prompt / idle (all cards)")


def cells(s):
    """report.py's energy cells of a run's summary."""
    if s is None:
        return ["—", "—", "—"]
    w = lambda x: "—" if math.isnan(x) else f"{x:.1f}"
    return [f"{s['energy_kj']:.1f} ({s['joules_per_task']:.0f})",
            "—" if math.isnan(s["joules_per_token"]) else f"{s['joules_per_token']:.2f}",
            f"{w(s['w_generating'])} / {w(s['w_prompt'])} / {w(s['w_idle'])}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="RUN=CARDS.csv[,CARDS.csv]=PROXY.jsonl[=ARM]")
    ap.add_argument("--skip", default="", help="task ids left out, comma-separated")
    ap.add_argument("--shift", type=float, default=0.0, help="seconds added to the proxy's times")
    ap.add_argument("--tsv", help="the per-run rows as a TSV")
    ap.add_argument("--cards-tsv", help="one row a run and card")
    ap.add_argument("--fit", action="store_true", help="also the shift that best lines the requests up")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    rows, card_rows = [], []
    print("| run | tasks | turns | tokens generated | window s (covered) | card energy kJ | J a task | "
          "J a generated token | W mean | W generating (s) | W reading the prompt (s) | W idle between turns (s) |"
          + (" best shift s (W inside requests above outside) |" if a.fit else ""))
    print("|---" * (12 + a.fit) + "|")
    for spec in a.runs:
        f = spec.split("=")
        if len(f) not in (3, 4):
            sys.exit(f"not RUN=CARDS=PROXY[=ARM]: {spec}")
        run, cards_paths, proxy = f[0], f[1].split(","), f[2]
        cards = Cards(cards_paths)
        reqs = load_requests(proxy, f[3] if len(f) == 4 else None)
        s = summary(cards, reqs, skip, a.shift)
        if s is None:
            print(f"| {run} | no requests or no card samples in the window |" + " |" * (10 + a.fit))
            continue
        rows.append([run] + tsv_cells(s))
        w = lambda x: "—" if math.isnan(x) else f"{x:.1f}"
        line = (f"| {run} | {s['tasks']} | {s['requests']} | {s['generated']} | {s['window_s']:.1f} "
                f"({s['covered_s']:.1f}) | {s['energy_kj']:.1f} | {s['joules_per_task']:.0f} | "
                f"{s['joules_per_token']:.2f} | {s['w_mean']:.1f} | {w(s['w_generating'])} ({s['generating_s']:.1f}) | "
                f"{w(s['w_prompt'])} ({s['prompt_s']:.1f}) | {w(s['w_idle'])} ({s['idle_s']:.1f}) |")
        if a.fit:
            d, c = fit(cards, reqs, skip)
            line += f" {d:+.2f} ({c:.1f}) |"
        print(line)
        for c in range(cards.ncards):
            sc = summary(cards, reqs, skip, a.shift, which=c)
            if sc is not None:
                card_rows.append([run, str(c), f"{sc['energy_kj'] * 1000:.0f}", f"{sc['w_mean']:.2f}",
                                  f"{sc['w_generating']:.2f}", f"{sc['w_prompt']:.2f}", f"{sc['w_idle']:.2f}"])
        if cards.missing:
            print(f"\n{run}: {cards.missing} samples with a card's power missing (their spans add nothing)\n")
    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("\t".join(TSV_COLUMNS) + "\n")
            fh.writelines("\t".join(r) + "\n" for r in rows)
    if a.cards_tsv:
        with open(a.cards_tsv, "w") as fh:
            fh.write("run\tcard\tjoules\tw_mean\tw_generating\tw_prompt\tw_idle\n")
            fh.writelines("\t".join(r) + "\n" for r in card_rows)


if __name__ == "__main__":
    main()
