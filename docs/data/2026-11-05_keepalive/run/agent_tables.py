#!/usr/bin/env python3
"""agent_tables.py OUT REF ARM...: C533's agent-check arms side by side (agent.tsv, keepalive_blocks.tsv in OUT).

OUT is the agent check's collected folder (release_check_s12.sh's report and collect steps): res/turns.csv,
res/attempts.csv and the rig's server logs rig/server-<arm>.log. Per arm, against REF:
  - tasks passed, turns, the session's prompt and generation tok/s (the server's own times, as report.py);
  - the time to first token per turn as the client sees it, median and p90 (report.py's percentile), and the first
    turn's median, where a cold card or a kernel's first launch would show;
  - the draft's acceptance over the session (accepted / drafted tokens, the server's print_timing lines) and tokens a
    step, 1 + NMAX x acceptance (NMAX, default 3, the draft's n-max: every step drafts NMAX tokens);
  - minutes a task at equal work: REF's prompt tokens at the arm's prompt speed, REF's generated tokens at the arm's
    generation speed, plus REF's own time outside the server (tools, tests, the relay), over REF's tasks. Minutes for
    all tasks depend on what each attempt generated; this column compares the engines on the same work.
keepalive_blocks.tsv: every ggml_cuda_keepalive_log line of each arm's server log, one row a card and a report: runs,
ended by the host, by the timeout, without a report, seconds, and the SM clock percentiles the block measured.
SKIP_TASKS (comma-separated task ids) leaves those tasks out of every arm; OUT_TAG names the two files' suffix.
Runs anywhere with Python 3 (no packages).
"""
import csv
import os
import re
import sys
from statistics import median

NMAX = int(os.environ.get("NMAX", "3"))
TAG = os.environ.get("OUT_TAG", "")


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def main():
    out, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    if ref not in arms:
        arms = [ref] + arms
    skip = set(filter(None, os.environ.get("SKIP_TASKS", "").split(",")))  # tasks left out of every arm
    turns = [t for t in csv.DictReader(open(os.path.join(out, "res", "turns.csv"))) if t["task_id"] not in skip]
    atts = [a for a in csv.DictReader(open(os.path.join(out, "res", "attempts.csv"))) if a["task_id"] not in skip]
    acc_re = re.compile(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")
    ka_re = re.compile(r"ggml_cuda_keepalive_log: (CUDA\d+): (\d+) runs \((\d+) ended by the host, (\d+) by the timeout, "
                       r"(\d+) without a report\), ([0-9.]+) s; SM clock while running: p10 (\d+), median (\d+), "
                       r"p90 (\d+) MHz, (\d+) % below 1400, (\d+) % at 1800 or more \((\d+) samples")
    st, blocks = {}, []
    for arm in arms:
        tt = [t for t in turns if t["arm"] == arm]
        aa = [a for a in atts if a["file"] == arm]
        f = lambda k: sum(float(t[k] or 0) for t in tt)
        ttft = [float(t["ttft_s"]) for t in tt if t["ttft_s"]]
        first = [float(t["ttft_s"]) for t in tt if t["ttft_s"] and t["turn"] == "1"]
        acc = gen = 0
        log = os.path.join(out, "rig", f"server-{arm}.log")
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                m = acc_re.search(line)
                if m:
                    acc, gen = acc + int(m.group(1)), gen + int(m.group(2))
                m = ka_re.search(line)
                if m:
                    blocks.append([arm] + list(m.groups()))
        wall = sum(float(a["wall_s"]) for a in aa)
        st[arm] = dict(n=len(aa), passed=sum(int(a["pass"]) for a in aa), turns=len(tt), pn=f("prompt_n"),
                       pms=f("prompt_ms"), gn=f("predicted_n"), gms=f("predicted_ms"), wall=wall, ttft=ttft,
                       first=first, acc=acc / gen if gen else float("nan"),
                       mins=median(float(a["wall_s"]) / 60 for a in aa) if aa else float("nan"))
    r = st[ref]
    r_other = r["wall"] - r["pms"] / 1000 - r["gms"] / 1000
    cols = ["arm", "passed", "turns", "prompt_tok_s", "gen_tok_s", "ttft_median_s", "ttft_p90_s", "ttft_turn1_median_s",
            "acceptance", "tokens_per_step", "minutes_all", "minutes_per_task_median", "equal_work_min_per_task",
            "equal_work_vs_ref"]
    rows = []
    for arm in arms:
        s = st[arm]
        pr = s["pn"] / s["pms"] * 1000 if s["pms"] else float("nan")
        gr = s["gn"] / s["gms"] * 1000 if s["gms"] else float("nan")
        eq = (r_other + r["pn"] / pr + r["gn"] / gr) / r["n"] / 60 if r["n"] else float("nan")
        rows.append([arm, f"{s['passed']}/{s['n']}", s["turns"], f"{pr:.1f}", f"{gr:.2f}", f"{median(s['ttft']):.2f}",
                     f"{pct(s['ttft'], 0.9):.2f}", f"{median(s['first']):.2f}" if s["first"] else "nan",
                     f"{s['acc']:.3f}", f"{1 + NMAX * s['acc']:.2f}", f"{s['wall'] / 60:.1f}", f"{s['mins']:.2f}",
                     f"{eq:.2f}", ""])
    base = float(rows[arms.index(ref)][12])
    for row in rows:
        row[13] = f"{(float(row[12]) / base - 1) * 100:+.1f} %"
    with open(os.path.join(out, f"agent{TAG}.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")
    bcols = ["arm", "card", "runs", "ended_by_host", "ended_by_timeout", "without_report", "seconds", "sm_p10_mhz",
             "sm_median_mhz", "sm_p90_mhz", "pct_below_1400", "pct_1800_up", "samples"]
    with open(os.path.join(out, f"keepalive_blocks{TAG}.tsv"), "w") as fh:
        fh.write("\t".join(bcols) + "\n")
        for b in blocks:
            fh.write("\t".join(b) + "\n")
    print(open(os.path.join(out, f"agent{TAG}.tsv")).read())
    print(f"keep-alive block lines: {len(blocks)}; ended by the timeout: {sum(int(b[4]) for b in blocks)}")


if __name__ == "__main__":
    main()
