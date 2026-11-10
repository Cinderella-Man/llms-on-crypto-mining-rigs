#!/usr/bin/env python3
"""pv_replies_table.py: the tables of the reply-length check (registry Q1184) from run_pv_replies.sh's work folders.

    python3 -I pv_replies_table.py WORK_A WORK_B --out TABLES [--long 4000] [--boot 10000]

Reads every replay-<arm>-<set>.jsonl in the given folders: the rows bench/agent_check/replay.py wrote with --repeat
and --count-parts, one per reply, a load labelled <arm>-<set>. A reply is one prompt (a task's recorded request) with
one seed index (the repeat); every load got the same prompts and seeds. Only counts are read, no time. Writes:

  replies.tsv   one row per reply: its length in generated tokens (llama-server's predicted_n), the reasoning's tokens
                (the reasoning text tokenized again by the server), the rest (answer = length minus reasoning: the
                text, the tool call and the markup), how it ended, the draft counters, the reply's MD5
  loads.tsv     per load: replies, generated tokens, median and mean length, the longest, replies of --long tokens or
                more, replies stopped at the limit, reasoning against answer tokens, draft acceptance, the server's
                sampling and grammar counters
  tasks.tsv     per prompt and load: the lengths by seed index, their mean
  compare.tsv   one group of loads against another, paired by prompt and seed index: the difference of the mean
                length with a 95 % bootstrap interval over prompts (the prompts resampled, each with all its seeds),
                the geometric mean of the paired ratios the same way, and in how many pairs the first is longer
  summary.md    the same as Markdown
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load(folders):
    """{label: {(task, request, seed index): row}}, the last row without an error per key."""
    by = defaultdict(dict)
    for f in folders:
        for p in sorted(Path(f).glob("replay-*.jsonl")):
            for l in p.open():
                if not l.strip():
                    continue
                r = json.loads(l)
                if r.get("error") or r.get("status") != 200 or not r.get("predicted_n"):
                    continue
                by[r["label"]][(r["trace"].split("/")[1], r["idx"], r.get("rep", 0))] = r
    return dict(by)


def length(r):
    return r["predicted_n"]


def reasoning(r):
    return r.get("reasoning_n") or 0


def answer(r):
    return r["predicted_n"] - reasoning(r)


def at_limit(r):
    return r.get("finish_reason") == "length" or (r.get("max_tokens") and r["predicted_n"] >= r["max_tokens"])


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def pct(lo, hi, v):
    v = sorted(v)
    return v[int(lo * len(v))], v[min(len(v) - 1, int(hi * len(v)))]


def compare(by, name, xs, ys, value, what, boot, rng):
    """Loads xs against loads ys on the pairs (prompt, seed index) every one of them holds."""
    keys = sorted(set.intersection(*[set(by[l]) for l in xs + ys]))
    if not keys:
        return None
    tasks = sorted({k[0] for k in keys})
    of = defaultdict(list)
    for k in keys:
        x = mean(value(by[l][k]) for l in xs)
        y = mean(value(by[l][k]) for l in ys)
        of[k[0]].append((x, y))

    def stat(ts):
        pairs = [p for t in ts for p in of[t]]
        mx, my = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
        geo = math.exp(mean(math.log((p[0] + 1) / (p[1] + 1)) for p in pairs))
        return mx - my, geo

    diff, geo = stat(tasks)
    boots = [stat([rng.choice(tasks) for _ in tasks]) for _ in range(boot)]
    d_lo, d_hi = pct(0.025, 0.975, [b[0] for b in boots])
    g_lo, g_hi = pct(0.025, 0.975, [b[1] for b in boots])
    pairs = [p for t in tasks for p in of[t]]
    mx, my = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
    return dict(comparison=name, what=what, pairs=len(pairs), prompts=len(tasks), mean_first=fmt(mx), mean_second=fmt(my),
                diff=fmt(diff), diff_pct=fmt(100 * diff / my, 2) if my else "", diff_lo=fmt(d_lo), diff_hi=fmt(d_hi),
                median_first=fmt(median(p[0] for p in pairs)), median_second=fmt(median(p[1] for p in pairs)),
                geo_ratio=fmt(geo, 4), geo_lo=fmt(g_lo, 4), geo_hi=fmt(g_hi, 4),
                first_longer=sum(p[0] > p[1] for p in pairs), first_shorter=sum(p[0] < p[1] for p in pairs),
                equal=sum(p[0] == p[1] for p in pairs))


def write_tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--long", type=int, default=4000, help="a long reply has this many tokens or more")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    by = load(a.folders)
    if not by:
        sys.exit("no replay rows")
    a.out.mkdir(parents=True, exist_ok=True)
    labels = sorted(by, key=lambda l: (l.rsplit("-", 1)[0], l.rsplit("-", 1)[1]))
    arms = sorted({l.rsplit("-", 1)[0] for l in labels})

    # replies.tsv
    rcols = ["load", "arm", "set", "task", "request", "seed_index", "seed", "prompt_tokens", "length", "reasoning",
             "answer", "content", "tool_args", "finish_reason", "tool_calls", "at_limit", "drafted", "accepted",
             "grammar_rejected", "reply_md5"]
    rrows = []
    for l in labels:
        arm, cset = l.rsplit("-", 1)
        for k in sorted(by[l]):
            r = by[l][k]
            rrows.append(dict(load=l, arm=arm, set=cset, task=k[0], request=k[1], seed_index=k[2],
                              seed=(r.get("sampling") or {}).get("seed"),
                              prompt_tokens=(r.get("prompt_n") or 0) + (r.get("cache_n") or 0), length=length(r),
                              reasoning=reasoning(r), answer=answer(r), content=r.get("content_n"),
                              tool_args=r.get("tool_args_n"), finish_reason=r.get("finish_reason"),
                              tool_calls=r.get("n_tool_calls"), at_limit=int(bool(at_limit(r))),
                              drafted=r.get("draft_n") or 0, accepted=r.get("draft_n_accepted") or 0,
                              grammar_rejected=r.get("grammar_rejected_n") or 0,
                              reply_md5=r.get("reply_md5")))
    write_tsv(a.out / "replies.tsv", rrows, rcols)

    # loads.tsv
    lcols = ["load", "arm", "set", "replies", "generated", "median", "mean", "longest", "long_replies", "long_share",
             "at_limit", "reasoning_tokens", "answer_tokens", "reasoning_share", "reasoning_median", "reasoning_mean",
             "answer_median", "answer_mean", "replies_without_reasoning", "ended_tool_calls", "ended_stop", "drafted",
             "accepted", "share_accepted", "tokens_a_step", "sampled_on_device", "grammar_checked", "grammar_rejected",
             "replies_with_grammar_rejection"]
    lrows = []
    for l in labels:
        arm, cset = l.rsplit("-", 1)
        rows = list(by[l].values())
        n = [length(r) for r in rows]
        re_, an = [reasoning(r) for r in rows], [answer(r) for r in rows]
        dr, ac = sum(r.get("draft_n") or 0 for r in rows), sum(r.get("draft_n_accepted") or 0 for r in rows)
        lrows.append(dict(load=l, arm=arm, set=cset, replies=len(rows), generated=sum(n), median=fmt(median(n)),
                          mean=fmt(mean(n)), longest=max(n), long_replies=sum(x >= a.long for x in n),
                          long_share=fmt(sum(x >= a.long for x in n) / len(n), 4),
                          at_limit=sum(bool(at_limit(r)) for r in rows), reasoning_tokens=sum(re_),
                          answer_tokens=sum(an), reasoning_share=fmt(sum(re_) / sum(n), 4),
                          reasoning_median=fmt(median(re_)), reasoning_mean=fmt(mean(re_)),
                          answer_median=fmt(median(an)), answer_mean=fmt(mean(an)),
                          replies_without_reasoning=sum(x == 0 for x in re_),
                          ended_tool_calls=sum(r.get("finish_reason") == "tool_calls" for r in rows),
                          ended_stop=sum(r.get("finish_reason") == "stop" for r in rows), drafted=dr, accepted=ac,
                          share_accepted=fmt(ac / dr, 4) if dr else "",
                          tokens_a_step=fmt(sum(n) / (sum(n) - ac), 3) if sum(n) > ac else "",
                          sampled_on_device=sum(r.get("sampled_on_device") or 0 for r in rows),
                          grammar_checked=sum(r.get("grammar_n") or 0 for r in rows),
                          grammar_rejected=sum(r.get("grammar_rejected_n") or 0 for r in rows),
                          replies_with_grammar_rejection=sum((r.get("grammar_rejected_n") or 0) > 0 for r in rows)))
    write_tsv(a.out / "loads.tsv", lrows, lcols)

    # tasks.tsv
    tasks = sorted({k[0] for l in labels for k in by[l]})
    tcols = ["task", "prompt_tokens"] + [f"{l}_{s}" for l in labels for s in ("mean", "lengths", "reasoning")]
    trows = []
    for t in tasks:
        row = dict(task=t)
        for l in labels:
            ks = sorted(k for k in by[l] if k[0] == t)
            if not ks:
                continue
            r0 = by[l][ks[0]]
            row.setdefault("prompt_tokens", (r0.get("prompt_n") or 0) + (r0.get("cache_n") or 0))
            row[f"{l}_mean"] = fmt(mean(length(by[l][k]) for k in ks))
            row[f"{l}_lengths"] = ", ".join(str(length(by[l][k])) for k in ks)
            row[f"{l}_reasoning"] = ", ".join(str(reasoning(by[l][k])) for k in ks)
        trows.append(row)
    write_tsv(a.out / "tasks.tsv", trows, tcols)

    # compare.tsv
    rng = random.Random(1)
    of_arm = {arm: [l for l in labels if l.rsplit("-", 1)[0] == arm] for arm in arms}
    plans = []
    if "pv0" in of_arm and "pv1" in of_arm:
        plans.append(("pv1 against pv0, all loads", of_arm["pv1"], of_arm["pv0"]))
        for x in of_arm["pv1"]:
            for y in of_arm["pv0"]:
                plans.append((f"{x} against {y}", [x], [y]))
    for arm in arms:
        ls = of_arm[arm]
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                plans.append((f"{ls[j]} against {ls[i]} (two loads of {arm})", [ls[j]], [ls[i]]))
    ccols = ["comparison", "what", "pairs", "prompts", "mean_first", "mean_second", "diff", "diff_pct", "diff_lo",
             "diff_hi", "median_first", "median_second", "geo_ratio", "geo_lo", "geo_hi", "first_longer",
             "first_shorter", "equal"]
    crows = []
    for name, xs, ys in plans:
        for what, value in (("length", length), ("reasoning", reasoning), ("answer", answer)):
            c = compare(by, name, xs, ys, value, what, a.boot, rng)
            if c:
                crows.append(c)
    write_tsv(a.out / "compare.tsv", crows, ccols)

    md = [f"# Reply lengths: {', '.join(labels)}", "",
          f"A long reply has {a.long} tokens or more. Intervals: 95 % bootstrap over prompts, {a.boot} resamples.", "",
          "## Loads", "", md_table(lrows, lcols), "", "## Comparisons, paired by prompt and seed index", "",
          md_table(crows, ccols), "", "## Prompts: the lengths by seed index", "",
          md_table(trows, ["task", "prompt_tokens"] + [f"{l}_{s}" for l in labels for s in ("mean", "lengths")]), ""]
    (a.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
