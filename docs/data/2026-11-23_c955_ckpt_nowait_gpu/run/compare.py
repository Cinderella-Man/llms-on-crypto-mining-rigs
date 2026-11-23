# C955: compares the arms of run.sh. Per arm: client.py's jsonl (replies, top-3 log-probabilities, counters) and its
# server log at -lv 4: each request's checkpoints in order (taken between two calls, with "no wait" where
# LLAMA_SERVER_CKPT_NOWAIT filled them so, or inside a call), the checkpoint each restore used, the checkpoints moved
# from the cards to the host, and the prompt calls the server printed (its "prompt processing, n_tokens = N" lines, one
# at each call's start once a prompt has run 3 s, so how many of a prompt's first calls show depends on the cards'
# speed: two arms' lists are compared from the later of their first printed calls on). C907's compare.py
# (docs/data/2026-11-21_c907_ckpt_stall/run/) for the GPU arms, without any time.
# Writes <out>/checkpoints.tsv, <out>/calls.tsv, <out>/moves.tsv, <out>/replies.tsv and prints server.tsv: one row per
# pair and request, then the verdict per pair.
# Usage: python3 -I compare.py <load dir with one folder per arm> <out dir> > server.tsv
import hashlib
import json
import os
import re
import sys

d, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

ARMS = ("base-off", "head-off", "head-on", "head-off-spill", "head-on-spill")
# run, reference, what has to hold
PAIRS = [
    ("head-off", "base-off", "the switch unset: replies, log-probabilities, counters, calls, checkpoints and restores the base's"),
    ("head-on", "head-off", "LLAMA_SERVER_CKPT_NOWAIT=1: the same replies, log-probabilities, counters, calls, checkpoints and"
                            " restores; every checkpoint between calls filled without a wait"),
    ("head-on-spill", "head-off-spill", "the same with 8 checkpoints of which the newest 2 stay on the cards (older ones"
                                        " moved to the host): also the same moves"),
]
REQS = 8

ansi = re.compile(r"\x1b\[[0-9;]*m")
launch = re.compile(r"task (\d+) \| processing task")
prog = re.compile(r"task (\d+) \| prompt processing, n_tokens = +(\d+),")
between = re.compile(r"task (-?\d+) \| created context checkpoint \d+ of \d+ \(pos_min = (-?\d+), pos_max = (-?\d+), "
                     r"n_tokens = (\d+), size = ([0-9.]+) MiB, on the devices ([0-9.]+) MiB, store (-?\d+)(, no wait)?,")
inside = re.compile(r"task (-?\d+) \| created context checkpoint \d+ of \d+ inside the call \(pos_min = (-?\d+), "
                    r"pos_max = (-?\d+), n_tokens = (\d+),")
restored = re.compile(r"task (-?\d+) \| restored context checkpoint \(pos_min = (-?\d+), pos_max = (-?\d+), n_tokens = (\d+), "
                      r"n_past = (-?\d+), size = ([0-9.]+) MiB, on the devices ([0-9.]+) MiB")
moved = re.compile(r"context checkpoint \(n_tokens = (\d+)\) moved to the host, ([0-9.]+) MiB")
full = re.compile(r"forcing full prompt re-processing")
bad = re.compile(r"error|abort|assert|failed|erasing context checkpoint \(n_tokens", re.I)


def load_rows(arm):
    path = os.path.join(d, arm, "client.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(x) for x in f]


def load_log(arm):
    """per task, in launch order: calls, checkpoints, restores, moves (landing while that task is the newest), full reads;
    and the log's error-like lines"""
    path = os.path.join(d, arm, "server.log")
    tasks, info, errors = [], {}, []
    if not os.path.exists(path):
        return [], ["no server log"]
    cur = None
    for line in open(path, errors="replace"):
        line = ansi.sub("", line).rstrip("\n")
        m = launch.search(line)
        if m:
            t = m.group(1)
            if t not in info:
                tasks.append(t)
                info[t] = {"calls": [], "ckpts": [], "restored": [], "moved": [], "full": 0}
            cur = t
            continue
        if cur is None:
            if bad.search(line) and "warmup" not in line:
                errors.append(line)
            continue
        g = info[cur]
        m = prog.search(line)
        if m:
            g["calls"].append(int(m.group(2)))
            continue
        m = between.search(line)
        if m:
            g["ckpts"].append({"kind": "between", "n_tokens": int(m.group(4)), "pos_min": int(m.group(2)),
                               "pos_max": int(m.group(3)), "no_wait": m.group(8) is not None,
                               "host_mib": float(m.group(5)), "dev_mib": float(m.group(6)), "store": int(m.group(7))})
            continue
        m = inside.search(line)
        if m:
            g["ckpts"].append({"kind": "inside", "n_tokens": int(m.group(4)), "pos_min": int(m.group(2)),
                               "pos_max": int(m.group(3)), "no_wait": False, "host_mib": None, "dev_mib": None, "store": None})
            continue
        m = restored.search(line)
        if m:
            g["restored"].append({"n_tokens": int(m.group(4)), "n_past": int(m.group(5)),
                                  "on_dev": float(m.group(7)) > 0})
            continue
        m = moved.search(line)
        if m:
            g["moved"].append(int(m.group(1)))
            continue
        if full.search(line):
            g["full"] += 1
            continue
        if bad.search(line):
            errors.append(line)
    return [info[t] for t in tasks], errors


def calls_agree(a, b):
    """two arms' printed call starts agree from the later of their first printed calls on"""
    if not a or not b:
        return True, 0
    lo = max(a[0], b[0])
    ta, tb = [x for x in a if x >= lo], [x for x in b if x >= lo]
    return ta == tb, len(ta)


def tok_hash(row):
    return hashlib.sha1(json.dumps([t[0] for t in row.get("tokens", [])]).encode()).hexdigest()[:12]


runs = {}
for arm in ARMS:
    rows = load_rows(arm)
    if rows is None:
        continue
    logs, errors = load_log(arm)
    runs[arm] = (rows, logs, errors)


def ckpt_of(logs, i, n_tokens):
    """the checkpoint of an earlier request of this run whose n_tokens a restore in request i used"""
    for j in range(i - 1, -1, -1):
        for k in logs[j]["ckpts"]:
            if k["n_tokens"] == n_tokens:
                return k
    return None


with open(os.path.join(out, "calls.tsv"), "w") as fc, open(os.path.join(out, "checkpoints.tsv"), "w") as fk, \
        open(os.path.join(out, "moves.tsv"), "w") as fm, open(os.path.join(out, "replies.tsv"), "w") as fr:
    fc.write("arm\treq\tprompt_n\tcall_starts_printed\n")
    fk.write("arm\treq\tkind\tn_tokens\tpos_min\tpos_max\tno_wait\tstore\thost_mib\tdevice_mib\n")
    fm.write("arm\treq\tmoved_to_host_n_tokens\n")
    fr.write("arm\treq\tnote\tfinish\treply_tokens\ttokens_sha1\tsame_tokens_as_base-off\tsame_logprobs_as_base-off\t"
             "prompt_n\tcache_n\trestored_n_tokens\trestored_from\tfull_reads\treasoning_start\n")
    ref = runs.get("base-off")
    for arm, (rows, logs, _) in runs.items():
        for i, r in enumerate(rows):
            lg = logs[i] if i < len(logs) else {"calls": [], "ckpts": [], "restored": [], "moved": [], "full": 0}
            t = r.get("timings") or {}
            fc.write(f"{arm}\t{r['req']}\t{t.get('prompt_n')}\t{','.join(map(str, lg['calls']))}\n")
            for k in lg["ckpts"]:
                f = lambda x, fmt: "" if x is None else format(x, fmt)
                fk.write(f"{arm}\t{r['req']}\t{k['kind']}\t{k['n_tokens']}\t{k['pos_min']}\t{k['pos_max']}\t"
                         f"{'yes' if k['no_wait'] else 'no'}\t{f(k['store'], 'd')}\t{f(k['host_mib'], '.3f')}\t{f(k['dev_mib'], '.3f')}\n")
            fm.write(f"{arm}\t{r['req']}\t{','.join(map(str, lg['moved']))}\n")
            rb = ref[0][i] if ref and i < len(ref[0]) else None
            same_t = rb is not None and "error" not in r and "error" not in rb and [x[0] for x in r["tokens"]] == [x[0] for x in rb["tokens"]]
            same_lp = rb is not None and "error" not in r and "error" not in rb and r["tokens"] == rb["tokens"]
            rst = lg["restored"]
            src = []
            for x in rst:
                k = ckpt_of(logs, i, x["n_tokens"])
                src.append("?" if k is None else (k["kind"] + (" no-wait" if k["no_wait"] else "") +
                                                   (" (cards)" if x["on_dev"] else " (host)")))
            fr.write(f"{arm}\t{r['req']}\t{r.get('note') or ''}\t{r.get('finish', r.get('error'))}\t{len(r.get('tokens', []))}\t"
                     f"{tok_hash(r)}\t{'yes' if same_t else 'NO'}\t{'yes' if same_lp else 'NO'}\t{t.get('prompt_n')}\t"
                     f"{t.get('cache_n')}\t{','.join(str(x['n_tokens']) for x in rst)}\t{'; '.join(src)}\t{lg['full']}\t"
                     f"{(r.get('reasoning') or '')[:60].replace(chr(10), ' ').replace(chr(9), ' ')}\n")

COUNTS = ("prompt_n", "cache_n", "cache_src", "cache_ckpt_n", "ckpt_created_n", "ckpt_incall_n", "ckpt_dev_n")

print("pair\treq\tprompt_n\treply\tlogprobs\tcounters\tcalls\tcheckpoints\trestored\tmoved\tbetween_calls\tno_wait\tnowait_n\tverdict")
verdicts = []
for run, ref_arm, what in PAIRS:
    if run not in runs or ref_arm not in runs:
        verdicts.append((run, ref_arm, "MISSING", what))
        continue
    (ra, la, ea), (rb, lb, eb) = runs[run], runs[ref_arm]
    ok_all = len(ra) == len(rb) == REQS and len(la) == len(lb) == len(ra) and not ea and not eb
    for i, (a, b) in enumerate(zip(ra, rb)):
        if "error" in a or "error" in b:
            ok_all = False
            print(f"{run}/{ref_arm}\t{a['req']}\t\terror: {a.get('error') or b.get('error')}\t\t\t\t\t\t\t\t\t\tFAIL")
            continue
        ta, tb = a["timings"], b["timings"]
        same_reply = a["reasoning"] == b["reasoning"] and a["content"] == b["content"] and a["finish"] == b["finish"]
        same_lp = a["tokens"] == b["tokens"]
        same_counts = all(ta.get(k) == tb.get(k) for k in COUNTS)
        empty = {"calls": [], "ckpts": [], "restored": [], "moved": [], "full": 0}
        ga = la[i] if i < len(la) else empty
        gb = lb[i] if i < len(lb) else empty
        same_calls, n_calls = calls_agree(ga["calls"], gb["calls"])
        key = lambda k: (k["kind"], k["n_tokens"], k["pos_min"], k["pos_max"])
        same_ckpts = [key(k) for k in ga["ckpts"]] == [key(k) for k in gb["ckpts"]]
        same_rst = ga["restored"] == gb["restored"] and ga["full"] == gb["full"]
        same_moved = ga["moved"] == gb["moved"]
        n_between = sum(1 for k in ga["ckpts"] if k["kind"] == "between")
        n_nowait = sum(1 for k in ga["ckpts"] if k["no_wait"])
        nowait_n = ta.get("ckpt_nowait_n") or 0
        if run.startswith("head-on"):
            # every checkpoint between calls filled without a wait, and counted so
            ok_nw = n_nowait == n_between and nowait_n == n_between
        else:
            ok_nw = n_nowait == 0 and nowait_n == 0
        ok = same_reply and same_lp and same_counts and same_calls and same_ckpts and same_rst and same_moved and ok_nw
        ok_all &= ok
        yn = lambda x: "same" if x else "DIFFERENT"
        print(f"{run}/{ref_arm}\t{a['req']}\t{ta.get('prompt_n')}\t{yn(same_reply)}\t{yn(same_lp)}\t{yn(same_counts)}\t"
              f"{yn(same_calls)} ({n_calls})\t{yn(same_ckpts)} ({len(ga['ckpts'])})\t"
              f"{yn(same_rst)} {[x['n_tokens'] for x in ga['restored']]}\t{yn(same_moved)} ({len(ga['moved'])})\t"
              f"{n_between}\t{n_nowait}\t{nowait_n}\t{'OK' if ok else 'FAIL'}")
    verdicts.append((run, ref_arm, "PASS" if ok_all else "FAIL", what))

print()
for arm, (_, _, errors) in runs.items():
    print(f"log\t{arm}\t{len(errors)} error-like lines" + "".join(f"\n\t\t{e[:200]}" for e in errors[:5]))
print()
for run, ref_arm, v, what in verdicts:
    print(f"verdict\t{run} against {ref_arm}\t{v}\t{what}")
