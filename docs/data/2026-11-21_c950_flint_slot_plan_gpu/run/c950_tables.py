#!/usr/bin/env python3
"""c950_tables.py: C950's tables from run/run.sh's output folder.

    python3 -I c950_tables.py OUTDIR RUNDIR [ARM ...]

RUNDIR is run.sh's WORK/RUN, with one folder per arm (server.log, session.tsv, mem.csv, command.txt); the arms default
to sp0 sp1 base. Requests are matched to the server's tasks by order: run/session.py sends them one at a time, and each
gets one task with one print_timing block. Writes:
  OUTDIR/tokens.tsv   per request: its kind, the generated tokens, the md5 of the token list (first 12 hex digits, as
                      session.tsv has it) in every arm and round, and how many different lists there are
  OUTDIR/replans.tsv  per arm, round and request: the pipe's guesses (and right, wrong), the pipe slots' re-plans after no
                      change of device and after one, and the plans over the whole memory (the server's running count,
                      `pipe reallocs = S / D ..., F plans`, as a difference; empty in the base, which has no such line),
                      the trace's sched-replan lines of the pipe slots (all, after a change of device) and of the decode
                      path, and the context's own graph reallocs (same size, changed size)
  OUTDIR/causes.tsv   per arm: the trace's re-plans grouped by cause (the node that changed device, or the first
                      galloc-trigger tensor that outgrew its place), with their count
  OUTDIR/plans.tsv    per arm: the pipe_slot_plan_full lines (plans over the whole memory) by node count and outcome
  OUTDIR/arms.tsv     per arm: the build, the switch, the load, the pipeline-parallelism and switch lines, the totals,
                      and MiB per card after the session
The trace's lines carry host times (microseconds); they are not read: the machine takes no speed measurements.
A sched-replan line belongs to the decode path when a ubatch-trace line follows it before the next decode-trace line,
task start, print_timing block or the log's end, and to a pipe slot otherwise (R181's rule,
docs/data/2026-11-19_r181_flint_realloc/run/r181_replans.py). Standard library only.
"""
import os
import re
import sys
from collections import Counter, OrderedDict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
START = re.compile(r"\| task ([0-9]+) \| processing task")
GUESS = re.compile(r"\| task ([0-9]+) \|\s+pipelined =\s*([0-9]+) guesses, ([0-9]+) right, ([0-9]+) wrong")
GR = re.compile(r"\| task ([0-9]+) \|\s+graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
PR = re.compile(r"\| task ([0-9]+) \|\s+pipe reallocs =\s*([0-9]+)\s*/\s*([0-9]+).*?,\s*([0-9]+) plans")
REPLAN = re.compile(r"sched-replan: #[0-9]+ backends_changed = ([01])(?: \((node|leaf) [0-9]+ '([^']*)' (\S+) -> (\S+)\))?")
TRIG = re.compile(r"galloc-trigger: (?:node|src|leaf) [0-9]+ '([^']*)'(?: of '[^']*')? \(([^)]*)\) ne = \[[^\]]*\] "
                  r"needs [0-9]+ bytes(?: in ([^\s,]+))?")
UB = re.compile(r"ubatch-trace: ")
DEC = re.compile(r"decode-trace: ")
PLAN = re.compile(r"pipe_slot_plan_full: slot [0-9]+ re-planned( after a change of device)?; .*?: ([0-9]+) nodes, "
                  r"[0-9.]+ ms, (.*?) \(plan [0-9]+\)")
NUM = re.compile(r"[0-9]+")
BUILD = {"sp0": "branch 19048bb6d", "sp1": "branch 19048bb6d", "base": "base 3dfad876f"}
SWITCH = {"sp0": "unset", "sp1": "LLAMA_PIPE_SLOT_PLAN=1", "base": "unset (not in this build)"}


def session_rows(path):
    rows = []
    for line in open(path, errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 8:
            rows.append(dict(round=f[0], request=f[1], kind=f[2], tokens=f[4], md5=f[7]))
    return rows


def parse_log(path):
    """Per task (in order): guesses, graph and pipe counts, trace re-plans; plus the causes and the plan lines."""
    tasks = OrderedDict()
    cur = None
    pending = []  # re-plans waiting for their micro-batch: (backends_changed, cause)
    trig = []  # galloc-trigger lines before the next sched-replan
    causes = Counter()
    plans = Counter()
    info = dict(pp_line=0, switch_line=0, replans=0, replans_dev=0)

    def flush(kind):
        for dev, cause in pending:
            t = tasks.get(cur)
            if t is not None:
                t[kind] += 1
                if kind == "slot" and dev:
                    t["slot_dev"] += 1
            causes[(kind, cause)] += 1
        pending.clear()

    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        if "pipeline parallelism enabled" in line:
            info["pp_line"] += 1
        if "pipe_init: LLAMA_PIPE_SLOT_PLAN=1" in line:
            info["switch_line"] += 1
        m = TRIG.search(line)
        if m:
            trig.append(f"outgrew its place: '{NUM.sub('#', m.group(1))}' ({m.group(2)})"
                        + (f" in {m.group(3)}" if m.group(3) else ""))
            continue
        m = REPLAN.search(line)
        if m:
            dev = m.group(1) == "1"
            info["replans"] += 1
            info["replans_dev"] += dev
            if dev:
                cause = f"device change: {m.group(2)} '{NUM.sub('#', m.group(3))}' {m.group(4)} -> {m.group(5)}"
            else:
                cause = trig[0] if trig else "no trigger line"
            trig = []
            pending.append((dev, cause))
            continue
        m = PLAN.search(line)
        if m:
            plans[("after a change of device" if m.group(1) else "after no change of device", m.group(2), m.group(3))] += 1
            continue
        if UB.search(line):
            flush("decode")
            continue
        if DEC.search(line):
            flush("slot")
            continue
        m = START.search(line)
        if m:
            flush("slot")
            cur = m.group(1)
            tasks[cur] = dict(task=cur, guesses="", right="", wrong="", gr=None, pr=None, slot=0, slot_dev=0, decode=0)
            continue
        m = GUESS.search(line)
        if m:
            flush("slot")
            t = tasks.setdefault(m.group(1), dict(task=m.group(1), gr=None, pr=None, slot=0, slot_dev=0, decode=0))
            t.update(guesses=m.group(2), right=m.group(3), wrong=m.group(4))
            continue
        m = GR.search(line)
        if m and m.group(1) in tasks:
            tasks[m.group(1)]["gr"] = (int(m.group(2)), int(m.group(3)))
            continue
        m = PR.search(line)
        if m and m.group(1) in tasks:
            tasks[m.group(1)]["pr"] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            continue
    flush("slot")
    return list(tasks.values()), causes, plans, info


def main():
    outdir, rundir = sys.argv[1], sys.argv[2]
    arms = sys.argv[3:] or ["sp0", "sp1", "base"]
    os.makedirs(outdir, exist_ok=True)

    sess = {a: session_rows(os.path.join(rundir, a, "session.tsv")) for a in arms}
    rounds = list(OrderedDict.fromkeys(r["round"] for r in sess[arms[0]]))
    reqs = list(OrderedDict.fromkeys(r["request"] for r in sess[arms[0]]))
    look = {(a, r["round"], r["request"]): r for a in arms for r in sess[a]}

    tok = ["request\tkind\ttokens\t" + "\t".join(f"{a}_{rd}" for a in arms for rd in rounds) + "\tdifferent_lists"]
    for q in reqs:
        cells = [look.get((a, rd, q), {}).get("md5", "missing") for a in arms for rd in rounds]
        first = look[(arms[0], rounds[0], q)]
        tok.append(f"{q}\t{first['kind']}\t{first['tokens']}\t" + "\t".join(cells) + f"\t{len(set(cells))}")

    rep = ["arm\tround\trequest\tkind\ttask\tguesses\tright\twrong\tpipe_replans_same_devices\tpipe_replans_changed_devices\t"
           "plans_whole_memory\ttrace_slot_replans\ttrace_slot_after_device_change\ttrace_decode_replans\t"
           "graph_reallocs_same\tgraph_reallocs_changed"]
    cau = ["arm\tscheduler\tcause\treplans"]
    pl = ["arm\tafter\tnodes\toutcome\tlines"]
    arm_rows = ["arm\tbuild\tswitch\tloaded\tpipeline_parallelism_line\tswitch_line\trequests\tsched_replan_lines\t"
                "after_device_change\tpipe_slot_replans_trace\tpipe_counter_same_devices\tpipe_counter_changed_devices\t"
                "plans_whole_memory\tmib_card0\tmib_card1\tmib_card2"]
    for a in arms:
        tasks, causes, plans, info = parse_log(os.path.join(rundir, a, "server.log"))
        order = sess[a]
        if len(tasks) != len(order):
            print(f"{a}: {len(tasks)} tasks in the log, {len(order)} requests in session.tsv", file=sys.stderr)
        prev_g, prev_p = (0, 0), (0, 0, 0)
        tot_slot = 0
        for t, r in zip(tasks, order):
            g = t["gr"] or prev_g
            dg = (g[0] - prev_g[0], g[1] - prev_g[1])
            prev_g = g
            if t["pr"] is not None:
                dp = [x - y for x, y in zip(t["pr"], prev_p)]
                prev_p = t["pr"]
                pcells = "\t".join(str(x) for x in dp)
            else:
                pcells = "\t\t"
            tot_slot += t["slot"]
            rep.append(f"{a}\t{r['round']}\t{r['request']}\t{r['kind']}\t{t['task']}\t{t['guesses']}\t{t['right']}\t"
                       f"{t['wrong']}\t{pcells}\t{t['slot']}\t{t['slot_dev']}\t{t['decode']}\t{dg[0]}\t{dg[1]}")
        for (kind, cause), n in sorted(causes.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            cau.append(f"{a}\t{'pipe slot' if kind == 'slot' else 'decode path'}\t{cause}\t{n}")
        for (after, nodes, outcome), n in sorted(plans.items()):
            pl.append(f"{a}\t{after}\t{nodes}\t{outcome}\t{n}")
        try:
            mib = [ln.split(",")[1].strip() for ln in open(os.path.join(rundir, a, "mem.csv")) if ln.strip()]
        except OSError:
            mib = ["", "", ""]
        try:
            loaded = open(os.path.join(rundir, a, "load.txt")).read().strip()
        except OSError:
            loaded = ""
        last_p = prev_p if any(t["pr"] is not None for t in tasks) else ("", "", "")
        arm_rows.append(f"{a}\t{BUILD.get(a, '')}\t{SWITCH.get(a, '')}\t{loaded}\t{info['pp_line']}\t{info['switch_line']}\t"
                        f"{len(order)}\t{info['replans']}\t{info['replans_dev']}\t{tot_slot}\t{last_p[0]}\t{last_p[1]}\t"
                        f"{last_p[2]}\t" + "\t".join((mib + ["", "", ""])[:3]))

    for name, rows in (("tokens.tsv", tok), ("replans.tsv", rep), ("causes.tsv", cau), ("plans.tsv", pl),
                       ("arms.tsv", arm_rows)):
        with open(os.path.join(outdir, name), "w") as f:
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
