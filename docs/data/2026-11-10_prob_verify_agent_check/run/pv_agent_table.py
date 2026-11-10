#!/usr/bin/env python3
"""pv_agent_table.py: the tables of run_pv_agent.sh's runs (registry Q1183). Counts only: no time is read.

    python3 -I pv_agent_table.py OUT --out TABLES [--ref pv0] [--arm pv1]

OUT is the agent check's results folder (res/<run>/<task>/a1/result.json and proxy.jsonl). A run is labelled
<arm>-<n>. Per task of a run, the attempt's row (result.json: passed, turns, tool calls and errors, caps) is joined
with the server's counters of its requests (the proxy's rows tagged <run>/<task>/a1/t<try>/<turn>: generated tokens,
drafted and accepted draft tokens). Tokens a step = generated / (generated - accepted): every verify step yields the
draft tokens it accepts and one token of the model's own.

Writes into TABLES:
  tasks.tsv    one row per run and task (first_reply: the tokens of the reply to the task's first request, whose
               prompt is the same in every run; second_reply: of the reply to the second request, which holds
               the first reply and its tool result; later_replies: the rest)
  requests.tsv one row per request: prompt tokens, generated, drafted and accepted tokens, tool calls, finish reason
  runs.tsv     one row per run: the sums over its tasks; generated tokens a turn; the median and the largest reply
               and the replies of 1,000 tokens or more
  paired.tsv   one row per task: passes, turns and generated tokens of every run side by side, the arm's mean minus
               the reference's mean, and the difference between the two runs of each arm
  compare.tsv  one row per measure: every run's value, the arm against the reference (difference of the means over
               runs), the difference between the two runs of the reference and of the arm, a 95 % bootstrap interval
               over tasks for the per-task measures, and the number of tasks where the arm is higher, lower and equal
  summary.md   the same as Markdown tables
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

TASK_COLS = ["run", "arm", "task_id", "status", "passed", "tests_passed", "tests_total", "tests_untouched", "turns",
             "requests", "generated", "drafted", "accepted", "share_accepted", "tokens_per_step", "tool_calls",
             "tool_errors", "malformed_calls", "length_stops", "runaway", "infra_tries", "context_last", "first_reply",
             "second_reply", "later_replies", "largest_reply", "drawn_on_device", "grammar_rejected"]
SUMS = ["passed", "turns", "requests", "generated", "drafted", "accepted", "tool_calls", "tool_errors",
        "malformed_calls", "length_stops", "infra_tries", "first_reply", "second_reply", "later_replies",
        "drawn_on_device", "grammar_rejected"]
REQ_COLS = ["run", "arm", "task_id", "turn", "prompt_tokens", "generated", "drafted", "accepted", "tool_calls",
            "finish_reason"]


def ratio(a, b, nd=4):
    return round(a / b, nd) if b else ""


def per_step(generated, accepted):
    return ratio(generated, generated - accepted, 3)


def load(out):
    """{run: {task: row}} from result.json and the proxy's rows, the runs in the order they started; the request rows."""
    by_tag = defaultdict(list)
    for line in open(out / "proxy.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.match(r"^(.+/a\d+/t\d+)/\d+$", r.get("tag") or "")
        if m:
            by_tag[m.group(1)].append(r)
    runs, started, reqrows = {}, {}, []
    for rdir in sorted(p for p in (out / "res").iterdir() if p.is_dir() and re.match(r"^.+-\d+$", p.name)):
        rows = {}
        for res in sorted(rdir.glob("*/a1/result.json")):
            r = json.loads(res.read_text())
            task = r["task_id"]
            reqs = by_tag.get(r.get("tag") or "", [])
            tm = [q.get("timings") or {} for q in reqs]
            gen = sum(t.get("predicted_n") or 0 for t in tm)
            drafted = sum(t.get("draft_n") or 0 for t in tm)
            acc = sum(t.get("draft_n_accepted") or 0 for t in tm)
            last = (reqs[-1].get("usage") or {}).get("prompt_tokens", "") if reqs else ""
            for i, (q, t) in enumerate(zip(reqs, tm), 1):
                reqrows.append(dict(
                    run=rdir.name, arm=rdir.name.rsplit("-", 1)[0], task_id=task, turn=i,
                    prompt_tokens=(q.get("usage") or {}).get("prompt_tokens", ""), generated=t.get("predicted_n") or 0,
                    drafted=t.get("draft_n") or 0, accepted=t.get("draft_n_accepted") or 0,
                    tool_calls=q.get("n_tool_calls", ""), finish_reason=q.get("finish_reason", "")))
            rows[task] = dict(
                run=rdir.name, arm=rdir.name.rsplit("-", 1)[0], task_id=task, status=r.get("status", ""),
                passed=r.get("success", 0), tests_passed=r.get("n_pass", ""), tests_total=r.get("n_total", ""),
                tests_untouched=r.get("tests_untouched", ""), turns=r.get("turns", 0), requests=len(reqs),
                generated=gen, drafted=drafted, accepted=acc, share_accepted=ratio(acc, drafted),
                tokens_per_step=per_step(gen, acc), tool_calls=r.get("tool_calls", 0),
                tool_errors=r.get("tool_errors", 0), malformed_calls=r.get("malformed_calls", 0),
                length_stops=r.get("length_stops", 0), runaway=r.get("runaway", ""),
                infra_tries=len(list(res.parent.parent.glob("a1-t*-infra"))), context_last=last,
                first_reply=(tm[0].get("predicted_n") or 0) if tm else 0,
                second_reply=(tm[1].get("predicted_n") or 0) if len(tm) > 1 else 0,
                later_replies=sum(t.get("predicted_n") or 0 for t in tm[2:]),
                largest_reply=max((t.get("predicted_n") or 0 for t in tm), default=0),
                drawn_on_device=sum(t.get("sampled_on_device") or 0 for t in tm),
                grammar_rejected=sum(t.get("grammar_rejected_n") or 0 for t in tm))
            started[rdir.name] = min(started.get(rdir.name, "9"), r.get("started") or "9")
        if rows:
            runs[rdir.name] = rows
    return {k: runs[k] for k in sorted(runs, key=lambda k: started[k])}, reqrows


def run_sums(rows, replies):
    s = {k: sum(int(r[k] or 0) for r in rows.values()) for k in SUMS}
    s["tasks"] = len(rows)
    s["share_accepted"] = ratio(s["accepted"], s["drafted"])
    s["tokens_per_step"] = per_step(s["generated"], s["accepted"])
    s["generated_per_turn"] = ratio(s["generated"], s["turns"], 1)
    replies = sorted(replies)
    n = len(replies)
    s["median_reply"] = (replies[n // 2] if n % 2 else (replies[n // 2 - 1] + replies[n // 2]) / 2) if n else ""
    s["largest_reply"] = replies[-1] if n else ""
    s["replies_1000_plus"] = sum(x >= 1000 for x in replies)
    s["failed_tasks"] = ",".join(t for t, r in rows.items() if not r["passed"]) or "-"
    s["runaways"] = ",".join(f"{t}({r['runaway']})" for t, r in rows.items() if r["runaway"]) or "-"
    return s


def boot(diffs, n=10000, seed=1183):
    """95 % percentile interval of the mean of per-task differences, resampling tasks."""
    if len(diffs) < 2:
        return "", ""
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return round(means[int(0.025 * n)], 2), round(means[int(0.975 * n) - 1], 2)


def write_tsv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def md_table(cols, rows):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--out", dest="tables", type=Path, required=True)
    ap.add_argument("--ref", default="pv0")
    ap.add_argument("--arm", default="pv1")
    a = ap.parse_args()
    runs, reqrows = load(a.out)
    if not runs:
        sys.exit("no run with a result.json")
    a.tables.mkdir(parents=True, exist_ok=True)
    labels = list(runs)
    write_tsv(a.tables / "tasks.tsv", TASK_COLS, [r for rows in runs.values() for r in rows.values()])
    order = {k: i for i, k in enumerate(labels)}
    reqrows.sort(key=lambda r: order[r["run"]])
    write_tsv(a.tables / "requests.tsv", REQ_COLS, reqrows)
    sums = {k: run_sums(v, [r["generated"] for r in reqrows if r["run"] == k]) for k, v in runs.items()}
    run_cols = ["run", "arm", "tasks", "passed", "failed_tasks", "turns", "requests", "generated", "tool_calls",
                "tool_errors", "malformed_calls", "length_stops", "runaways", "infra_tries", "drafted", "accepted",
                "share_accepted", "tokens_per_step", "generated_per_turn", "median_reply", "largest_reply",
                "replies_1000_plus", "first_reply", "second_reply", "later_replies", "drawn_on_device",
                "grammar_rejected"]
    run_rows = [dict(run=k, arm=k.rsplit("-", 1)[0], **s) for k, s in sums.items()]
    write_tsv(a.tables / "runs.tsv", run_cols, run_rows)

    of = {arm: [k for k in labels if k.rsplit("-", 1)[0] == arm] for arm in (a.ref, a.arm)}
    tasks = [t for t in next(iter(runs.values())) if all(t in runs[k] for k in labels)]

    def mean_of(arm, task, key):
        v = [float(runs[k][task][key] or 0) for k in of[arm]]
        return sum(v) / len(v) if v else 0.0

    def between(arm, task, key):  # the second run of an arm minus its first
        k = of[arm]
        return float(runs[k[1]][task][key] or 0) - float(runs[k[0]][task][key] or 0) if len(k) > 1 else ""

    paired_cols = ["task_id"]
    for key in ("passed", "turns", "generated", "tokens_per_step", "first_reply", "second_reply"):
        paired_cols += [f"{key}_{k}" for k in labels]
    for key in ("turns", "generated", "first_reply", "second_reply"):
        paired_cols += [f"{key}_{a.arm}_minus_{a.ref}", f"{key}_{a.ref}_2_minus_1", f"{key}_{a.arm}_2_minus_1"]
    paired = []
    for t in tasks:
        row = {"task_id": t}
        for key in ("passed", "turns", "generated", "tokens_per_step", "first_reply", "second_reply"):
            for k in labels:
                row[f"{key}_{k}"] = runs[k][t][key]
        for key in ("turns", "generated", "first_reply", "second_reply"):
            row[f"{key}_{a.arm}_minus_{a.ref}"] = round(mean_of(a.arm, t, key) - mean_of(a.ref, t, key), 1)
            row[f"{key}_{a.ref}_2_minus_1"] = between(a.ref, t, key)
            row[f"{key}_{a.arm}_2_minus_1"] = between(a.arm, t, key)
        paired.append(row)
    write_tsv(a.tables / "paired.tsv", paired_cols, paired)

    cmp_cols = ["measure"] + labels + [f"{a.ref}_mean", f"{a.arm}_mean", "difference", "difference_pct",
                                       f"{a.ref}_2_minus_1", f"{a.arm}_2_minus_1", "within_ref_spread",
                                       "ci95_low", "ci95_high", "tasks_higher", "tasks_lower", "tasks_equal",
                                       f"{a.ref}_2_vs_1_higher", f"{a.ref}_2_vs_1_lower", f"{a.ref}_2_vs_1_equal"]
    cmp_rows = []
    for key, per_task, nd in (("passed", True, 2), ("turns", True, 2), ("generated", True, 1),
                              ("tool_calls", True, 2), ("tool_errors", True, 2), ("malformed_calls", True, 2),
                              ("first_reply", True, 1), ("second_reply", True, 1), ("later_replies", True, 1),
                              ("generated_per_turn", False, 1),
                              ("median_reply", False, 1), ("replies_1000_plus", False, 1),
                              ("share_accepted", False, 4), ("tokens_per_step", False, 3)):
        row = {"measure": key}
        for k in labels:
            row[k] = sums[k][key]
        vals = {arm: [float(sums[k][key] or 0) for k in of[arm]] for arm in of}
        if not vals[a.ref] or not vals[a.arm]:
            cmp_rows.append(row)
            continue
        m = {arm: sum(v) / len(v) for arm, v in vals.items()}
        d = m[a.arm] - m[a.ref]
        row[f"{a.ref}_mean"], row[f"{a.arm}_mean"] = round(m[a.ref], nd + 1), round(m[a.arm], nd + 1)
        row["difference"] = round(d, nd + 1)
        row["difference_pct"] = round(100 * d / m[a.ref], 2) if m[a.ref] else ""
        for arm in of:
            row[f"{arm}_2_minus_1"] = round(vals[arm][1] - vals[arm][0], nd + 1) if len(vals[arm]) > 1 else ""
        if len(vals[a.ref]) > 1:
            row["within_ref_spread"] = "yes" if abs(d) <= abs(vals[a.ref][1] - vals[a.ref][0]) else "no"
        if per_task:
            diffs = [mean_of(a.arm, t, key) - mean_of(a.ref, t, key) for t in tasks]
            row["ci95_low"], row["ci95_high"] = boot([x * len(tasks) for x in diffs])
            row["tasks_higher"] = sum(x > 0 for x in diffs)
            row["tasks_lower"] = sum(x < 0 for x in diffs)
            row["tasks_equal"] = sum(x == 0 for x in diffs)
            if len(of[a.ref]) > 1:
                b = [between(a.ref, t, key) for t in tasks]
                row[f"{a.ref}_2_vs_1_higher"] = sum(x > 0 for x in b)
                row[f"{a.ref}_2_vs_1_lower"] = sum(x < 0 for x in b)
                row[f"{a.ref}_2_vs_1_equal"] = sum(x == 0 for x in b)
        cmp_rows.append(row)
    write_tsv(a.tables / "compare.tsv", cmp_cols, cmp_rows)

    with open(a.tables / "summary.md", "w") as f:
        f.write(f"Runs in the order they started: {', '.join(labels)}; {len(tasks)} tasks in every run.\n\n")
        f.write("Per run:\n\n" + md_table(run_cols, run_rows) + "\n")
        f.write(f"`{a.arm}` against `{a.ref}` (difference of the means over runs; the interval is a 95 % bootstrap "
                "over tasks of the difference in the run's sum):\n\n" + md_table(cmp_cols, cmp_rows) + "\n")
        f.write("Per task:\n\n" + md_table(paired_cols, paired) + "\n")
        for k in labels:
            f.write(f"Run `{k}`, per task:\n\n" + md_table(TASK_COLS[2:], list(runs[k].values())) + "\n")
    print(open(a.tables / "summary.md").read())


if __name__ == "__main__":
    main()
