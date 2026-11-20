#!/usr/bin/env python3
"""agent_tables.py OUT REF ARM...: the agent check's runs side by side (agent.tsv, keepalive_blocks.tsv in OUT).

OUT is the agent check's collected folder (release_check_s12.sh's report and collect steps), or a table folder made
from published rows: res/turns.csv, res/attempts.csv and, where they exist, the rig's server logs
rig/server-<arm>.log. Each ARM is one run (the rows' arm). Per run, against REF:
  - tasks passed, turns, the session's prompt and generation tok/s (the server's own times, as report.py);
  - the time to first token per turn as the client sees it, median and p90 (report.py's percentile), and the first
    turn's median, where a cold card or a kernel's first launch would show;
  - the draft's acceptance over the session (accepted / drafted tokens, the server's print_timing lines) and tokens a
    step, 1 + NMAX x acceptance (NMAX, default 3, the draft's n-max: every step drafts NMAX tokens);
  - minutes a task at equal work: REF's prompt tokens at the run's prompt speed, REF's generated tokens at the run's
    generation speed, plus REF's own time outside the server (tools, tests, the relay), over REF's tasks. Minutes for
    all tasks depend on what each attempt generated; this column compares the engines on the same work;
  - the work the run did (C855, Q1526; report.py's work counts, C852): prompt tokens read a task (the prompt tokens
    the server processed, the rest came from its cache), reply tokens re-read a task (tokens of the previous reply
    read again as prompt, on turns whose cached prefix stops short of that reply's end) and those turns' prompt
    seconds a task, the number of such turns, and the longest first-token wait. A task is one attempt with rows in
    turns.csv. These are the run's own counts, not scaled to REF's work: a change that reads fewer tokens shows here
    and not in the equal-work minutes.
  - the cards' energy (FN101, Q1518; energy.py), where the folder holds the run's card log and proxy rows: the card
    energy over the run's session window (its first request to its last request's end) in kJ, the joules a task and a
    generated token, and the cards' summed watts while generating, reading the prompt and idle between turns; and the
    joules a task at equal work, the counterpart of the minutes at equal work: REF's prompt tokens at the run's prompt
    speed and watts while reading the prompt, REF's generated tokens at the run's generation speed and watts while
    generating, and REF's own time outside the server at the run's watts while idle, over REF's tasks. The session's
    joules a task and a token depend on what each attempt generated (a run that generates more spreads its prompt
    reading and idle time over more tokens); the joules at equal work compare the engines on the same work. A run's
    card log is rig_<arm>/cards.csv, else rig/cards.csv (the release check's log of every arm); its proxy rows those of
    proxy_<arm>/proxy.jsonl whose tag starts with the arm, or every task row in it when none does (one run's log under
    the harness's own arm name), else the rows of OUT/proxy.jsonl whose tag starts with the arm. SKIP_TASKS also cuts
    a left-out task's span out of the window.
The columns up to equal_work_vs_ref are those of docs/data/2026-11-05_keepalive/run/agent_tables.py, cell for cell;
the work columns follow them, then the energy columns when any run has a card log and proxy rows (a table without
them is as before).
keepalive_blocks.tsv: every ggml_cuda_keepalive_log line of each run's server log, one row a card and a report: runs,
ended by the host, by the timeout, without a report, seconds, and the SM clock percentiles the block measured.
SKIP_TASKS (comma-separated task ids) leaves those tasks out of every run; OUT_TAG names the two files' suffix.
Runs anywhere with Python 3 (no packages); it reads report.py from its own folder.
"""
import csv
import os
import re
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import load_turns, work  # noqa: E402
import energy as card_energy  # noqa: E402

NMAX = int(os.environ.get("NMAX", "3"))
TAG = os.environ.get("OUT_TAG", "")
WORK_COLS = ["prompt_read_per_task", "reread_per_task", "reread_s_per_task", "reread_turns", "ttft_max_s"]
ENERGY_COLS = ["energy_kj", "joules_per_task", "joules_per_token", "w_generating", "w_prompt", "w_idle",
               "equal_work_j_per_task", "equal_work_j_vs_ref"]


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def work_cells(traces):
    """The work columns of a run from its traces (report.py's work): per task, prompt tokens read, reply tokens
    re-read and those turns' prompt seconds; the re-read turns; the longest first-token wait."""
    if not traces:
        return ["nan"] * len(WORK_COLS)
    w, n = work(traces), len(traces)
    return [f"{w['prompt_n'] / n:.0f}", f"{w['reread_n'] / n:.0f}", f"{w['reread_s'] / n:.2f}", w["reread_turns"],
            "nan" if w["ttft_max"] is None else f"{w['ttft_max']:.2f}"]


def energy_summary(out, arm, skip):
    """A run's energy summary (energy.py), or None when the folder lacks its card log or proxy rows."""
    cards = next((p for p in (os.path.join(out, f"rig_{arm}", "cards.csv"), os.path.join(out, "rig", "cards.csv"))
                  if os.path.exists(p)), None)
    own, shared = os.path.join(out, f"proxy_{arm}", "proxy.jsonl"), os.path.join(out, "proxy.jsonl")
    if cards is None or not (os.path.exists(own) or os.path.exists(shared)):
        return None
    if os.path.exists(own):
        reqs = card_energy.load_requests(own, arm) or card_energy.load_requests(own)
    else:
        reqs = card_energy.load_requests(shared, arm)
    return card_energy.summary(card_energy.Cards([cards]), reqs, skip)


def main():
    out, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    if ref not in arms:
        arms = [ref] + arms
    skip = set(filter(None, os.environ.get("SKIP_TASKS", "").split(",")))  # tasks left out of every arm
    turns_csv = os.path.join(out, "res", "turns.csv")
    turns = [t for t in csv.DictReader(open(turns_csv)) if t["task_id"] not in skip]
    atts = [a for a in csv.DictReader(open(os.path.join(out, "res", "attempts.csv"))) if a["task_id"] not in skip]
    traces = {k: v for k, v in load_turns([turns_csv]).items() if k[1] not in skip}  # (arm, task, attempt): requests
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
                       mins=median(float(a["wall_s"]) / 60 for a in aa) if aa else float("nan"),
                       work=work_cells([v for k, v in traces.items() if k[0] == arm]),
                       energy=energy_summary(out, arm, skip))
    r = st[ref]
    r_other = r["wall"] - r["pms"] / 1000 - r["gms"] / 1000
    cols = ["arm", "passed", "turns", "prompt_tok_s", "gen_tok_s", "ttft_median_s", "ttft_p90_s", "ttft_turn1_median_s",
            "acceptance", "tokens_per_step", "minutes_all", "minutes_per_task_median", "equal_work_min_per_task",
            "equal_work_vs_ref"] + WORK_COLS
    with_energy = any(st[arm]["energy"] for arm in arms)
    if with_energy:
        cols += ENERGY_COLS
    rows, eq_j = [], {}
    for arm in arms:
        s = st[arm]
        pr = s["pn"] / s["pms"] * 1000 if s["pms"] else float("nan")
        gr = s["gn"] / s["gms"] * 1000 if s["gms"] else float("nan")
        eq = (r_other + r["pn"] / pr + r["gn"] / gr) / r["n"] / 60 if r["n"] else float("nan")
        rows.append([arm, f"{s['passed']}/{s['n']}", s["turns"], f"{pr:.1f}", f"{gr:.2f}", f"{median(s['ttft']):.2f}",
                     f"{pct(s['ttft'], 0.9):.2f}", f"{median(s['first']):.2f}" if s["first"] else "nan",
                     f"{s['acc']:.3f}", f"{1 + NMAX * s['acc']:.2f}", f"{s['wall'] / 60:.1f}", f"{s['mins']:.2f}",
                     f"{eq:.2f}", ""] + s["work"])
        e = s["energy"]
        if with_energy and e is None:
            rows[-1] += ["nan"] * len(ENERGY_COLS)
        elif with_energy:
            eq_j[arm] = ((r_other * e["w_idle"] + r["pn"] / pr * e["w_prompt"] + r["gn"] / gr * e["w_generating"])
                         / r["n"] if r["n"] else float("nan"))
            cells = dict(zip(card_energy.TSV_COLUMNS[1:], card_energy.tsv_cells(e)))
            rows[-1] += [cells[c] for c in ENERGY_COLS[:-2]] + [f"{eq_j[arm]:.0f}", ""]
    base = float(rows[arms.index(ref)][12])
    for row in rows:
        row[13] = f"{(float(row[12]) / base - 1) * 100:+.1f} %"
        if with_energy:
            row[-1] = (f"{(eq_j[row[0]] / eq_j[ref] - 1) * 100:+.1f} %" if row[0] in eq_j and ref in eq_j
                       else "nan")
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
