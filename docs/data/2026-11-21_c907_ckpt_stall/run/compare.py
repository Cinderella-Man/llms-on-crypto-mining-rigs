# C907: compares the server runs of run.sh (step server). Per run: client.py's jsonl (replies, top-3 log-probabilities,
# timings) and its server log at -lv 4: each request's prompt calls (the "prompt processing, n_tokens = N" progress
# lines, printed at each call's start once the prompt has run 3 s), its checkpoints in order (taken between two calls,
# with "no wait" where the switch filled them so, or inside a call) and the checkpoint a restore used.
# Writes <dir>/calls.tsv and <dir>/checkpoints.tsv, and prints server.tsv: one row per pair and request, then the
# verdict per pair.
# Usage: python3 -I compare.py <dir> > server.tsv
import json
import os
import re
import sys

d = sys.argv[1]

# run, reference, what has to hold
PAIRS = [
    ("head-off", "base-off", "the switch unset: replies, log-probabilities, counters, calls and checkpoints the base's"),
    ("head-on", "head-off", "LLAMA_SERVER_CKPT_NOWAIT=1: the same replies, log-probabilities, calls, checkpoints and restores;"
                            " every checkpoint between calls filled without a wait"),
]

ansi = re.compile(r"\x1b\[[0-9;]*m")
launch = re.compile(r"task (\d+) \| processing task")
prog = re.compile(r"task (\d+) \| prompt processing, n_tokens = +(\d+),")
between = re.compile(r"task (\d+) \| created context checkpoint \d+ of \d+ \(pos_min = (-?\d+), pos_max = (-?\d+), "
                     r"n_tokens = (\d+), size = ([0-9.]+) MiB, on the devices ([0-9.]+) MiB, store (-?\d+)(, no wait)?, ([0-9.]+) ms\)")
inside = re.compile(r"task (\d+) \| created context checkpoint \d+ of \d+ inside the call \(pos_min = (-?\d+), "
                    r"pos_max = (-?\d+), n_tokens = (\d+),")
restored = re.compile(r"task (\d+) \| restored context checkpoint.*?n_tokens = (\d+)")


def load_rows(name):
    path = os.path.join(d, name + ".jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(x) for x in f]


def load_log(name):
    """per task, in launch order: (calls, checkpoints, restores)"""
    path = os.path.join(d, name + ".log")
    tasks, info = [], {}
    if not os.path.exists(path):
        return []
    for line in open(path, errors="replace"):
        line = ansi.sub("", line)
        m = launch.search(line)
        if m:
            t = m.group(1)
            if t not in info:
                tasks.append(t)
                info[t] = {"calls": [], "ckpts": [], "restored": []}
            continue
        m = prog.search(line)
        if m and m.group(1) in info:
            info[m.group(1)]["calls"].append(int(m.group(2)))
            continue
        m = between.search(line)
        if m and m.group(1) in info:
            info[m.group(1)]["ckpts"].append(("between", int(m.group(4)), int(m.group(2)), int(m.group(3)),
                                              m.group(8) is not None, float(m.group(9)), float(m.group(5)), float(m.group(6))))
            continue
        m = inside.search(line)
        if m and m.group(1) in info:
            info[m.group(1)]["ckpts"].append(("inside", int(m.group(4)), int(m.group(2)), int(m.group(3)), False, None, None, None))
            continue
        m = restored.search(line)
        if m and m.group(1) in info:
            info[m.group(1)]["restored"].append(int(m.group(2)))
    return [info[t] for t in tasks]


runs = {}
for name in ("base-off", "head-off", "head-on", "head-cms"):
    rows = load_rows(name)
    if rows is None:
        continue
    runs[name] = (rows, load_log(name))

with open(os.path.join(d, "calls.tsv"), "w") as fc, open(os.path.join(d, "checkpoints.tsv"), "w") as fk:
    fc.write("run\treq\tprompt_n\tcall_starts\n")
    fk.write("run\treq\tkind\tn_tokens\tpos_min\tpos_max\tno_wait\tsave_ms\thost_mib\tdevice_mib\n")
    for name, (rows, logs) in runs.items():
        for r, lg in zip(rows, logs):
            fc.write(f"{name}\t{r['req']}\t{(r.get('timings') or {}).get('prompt_n')}\t{','.join(map(str, lg['calls']))}\n")
            for k in lg["ckpts"]:
                f = lambda x, fmt: "" if x is None else format(x, fmt)
                fk.write(f"{name}\t{r['req']}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{'yes' if k[4] else 'no'}\t{f(k[5], '.1f')}\t"
                         f"{f(k[6], '.3f')}\t{f(k[7], '.3f')}\n")

COUNTS = ("prompt_n", "cache_n", "cache_src", "cache_ckpt_n", "ckpt_created_n", "ckpt_incall_n", "ckpt_dev_n")

print("pair\treq\tprompt_n\treply\tlogprobs\tcounters\tcalls\tcheckpoints\trestored\tbetween_calls\tno_wait\tnowait_n\tverdict")
verdicts = []
for run, ref, what in PAIRS:
    if run not in runs or ref not in runs:
        verdicts.append((run, ref, "MISSING", what))
        continue
    (ra, la), (rb, lb) = runs[run], runs[ref]
    ok_all = len(ra) == len(rb) == 6 and len(la) == len(lb) == len(ra)
    for i, (a, b) in enumerate(zip(ra, rb)):
        if "error" in a or "error" in b:
            ok_all = False
            print(f"{run}/{ref}\t{a['req']}\t\terror: {a.get('error') or b.get('error')}\t\t\t\t\t\t\t\t\tFAIL")
            continue
        ta, tb = a["timings"], b["timings"]
        same_reply = a["reasoning"] == b["reasoning"] and a["content"] == b["content"] and a["finish"] == b["finish"]
        same_lp = a["tokens"] == b["tokens"]
        same_counts = all(ta.get(k) == tb.get(k) for k in COUNTS)
        ga = la[i] if i < len(la) else {"calls": [], "ckpts": [], "restored": []}
        gb = lb[i] if i < len(lb) else {"calls": [], "ckpts": [], "restored": []}
        same_calls = ga["calls"] == gb["calls"]
        same_ckpts = [k[:4] for k in ga["ckpts"]] == [k[:4] for k in gb["ckpts"]]
        same_rst = ga["restored"] == gb["restored"]
        n_between = sum(1 for k in ga["ckpts"] if k[0] == "between")
        n_nowait = sum(1 for k in ga["ckpts"] if k[4])
        nowait_n = ta.get("ckpt_nowait_n") or 0
        if run == "head-on":
            # every checkpoint between calls filled without a wait, and counted so
            ok_nw = n_nowait == n_between and nowait_n == n_between
        else:
            ok_nw = n_nowait == 0 and nowait_n == 0
        ok = same_reply and same_lp and same_counts and same_calls and same_ckpts and same_rst and ok_nw
        ok_all &= ok
        yn = lambda x: "same" if x else "DIFFERENT"
        print(f"{run}/{ref}\t{a['req']}\t{ta.get('prompt_n')}\t{yn(same_reply)}\t{yn(same_lp)}\t{yn(same_counts)}\t"
              f"{yn(same_calls)} ({len(ga['calls'])})\t{yn(same_ckpts)} ({len(ga['ckpts'])})\t{yn(same_rst)} {ga['restored']}\t"
              f"{n_between}\t{n_nowait}\t{nowait_n}\t{'OK' if ok else 'FAIL'}")
    verdicts.append((run, ref, "PASS" if ok_all else "FAIL", what))

# C945, not part of the verdict: --checkpoint-min-step 7680 against 8192, both with the switch unset
if "head-cms" in runs and "head-off" in runs:
    print()
    print("info\treq\tprompt_n\tcalls (8192 / 7680)\tcheckpoints between calls (8192 / 7680)\tinside a call (8192 / 7680)"
          "\tcheckpoint positions at 7680\treply the same")
    (ra, la), (rb, lb) = runs["head-cms"], runs["head-off"]
    for i, (a, b) in enumerate(zip(ra, rb)):
        if "error" in a or "error" in b or i >= len(la) or i >= len(lb):
            print(f"info\t{a['req']}\terror")
            continue
        nb = lambda g: sum(1 for k in g["ckpts"] if k[0] == "between")
        ni = lambda g: sum(1 for k in g["ckpts"] if k[0] == "inside")
        same = a["reasoning"] == b["reasoning"] and a["content"] == b["content"]
        print(f"info\t{a['req']}\t{a['timings'].get('prompt_n')}\t{len(lb[i]['calls'])} / {len(la[i]['calls'])}\t"
              f"{nb(lb[i])} / {nb(la[i])}\t{ni(lb[i])} / {ni(la[i])}\t{','.join(str(k[1]) for k in la[i]['ckpts'])}\t"
              f"{'yes' if same else 'no'}")

print()
for run, ref, v, what in verdicts:
    print(f"verdict\t{run} against {ref}\t{v}\t{what}")
