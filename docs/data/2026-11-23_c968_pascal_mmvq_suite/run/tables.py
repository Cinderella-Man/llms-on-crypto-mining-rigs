#!/usr/bin/env python3
"""C968 (Q1698): the tables of run.sh's output.

Usage: python3 -I tables.py <run.sh's WORK folder, fetched> <output folder>
Writes:
  bops.tsv      one line per test-backend-ops run: share, run, setting, cases, passed, failed, not recorded (replays of
                cases the record left out), exit status, the failing cases
  moved.tsv     per share, operator, weight type and token count (1, 2-4, 5-8, 9 or more): the cases replayed in every
                setting; how many repeat their error when replayed unset twice; and how many get another error than unset
                under p106 and under fn (another kernel computed them), how many fn changes against p106, and how many
                p106-kcol0 changes against p106 (a k-column kernel computed them under p106), and how many fn-f32
                changes against fn (GGML_CUDA_MMVF_F32_MT=2's F32 products)
  verdict.tsv   per share: cases recorded, cases replayed in every setting, per setting those cases passed and failed,
                the cases p106 and fn move, the moved cases of 9 or more tokens, the cases whose unset error does not
                repeat, and the cases only the forced settings replay (the large-activation cases, supported only where
                the fork's Pascal kernels are on) with how many of them failed in any forced setting
  cases.tsv.gz  one line per replayed case: share, case, and for each setting the verdict and the errors (normalized
                squared error against the CPU, the output's first, then any other compared node's)
"""
import collections
import csv
import gzip
import os
import re
import sys

WORK, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
B = os.path.join(WORK, "bops")
SHARES = ("mul_mat", "mul_mat_id", "fused")
RUNS = (("live-p106", "p106 (live)"), ("live-fn", "fn (live)"), ("record", "unset (live, record written)"),
        ("replay-unset", "unset (replayed)"), ("replay-unset2", "unset (replayed again)"),
        ("replay-p106", "p106 (replayed)"), ("replay-fn", "fn (replayed)"),
        ("replay-p106-kcol0", "p106-kcol0 (replayed)"), ("replay-fn-f32", "fn-f32 (replayed)"))
SETTINGS = ("unset", "unset2", "p106", "fn", "p106-kcol0", "fn-f32")
BASE = SETTINGS[:5]  # every case is counted where these five replays hold it; fn-f32 ran later and is optional


def strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def shape(case):
    """operator, weight type, token count of a case string: n, except in the fused gate/up cases (MUL_MAT_VEC_FUSION),
    whose tokens are m (n is the weight rows there); in MUL_MAT_ID cases it is the batch's tokens, of which each expert
    gets about n * n_used / n_mats"""
    op = case.split("(", 1)[0]
    m = re.search(r"type_a=(\w+)", case) or re.search(r"\btype=(\w+)", case)
    t = m.group(1) if m else "?"
    m = re.search(r"[(,]m=(\d+)" if op.startswith("MUL_MAT_VEC_FUSION") else r"[(,]n=(\d+)", case)
    n = int(m.group(1)) if m else 0
    return op, t, n


def nbucket(n):
    if n == 0:
        return "-"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    return "9+"


# ---- every run
rcs = {}
if os.path.exists(os.path.join(B, "rc.txt")):
    for line in open(os.path.join(B, "rc.txt"), encoding="utf-8"):
        f = line.split()
        if len(f) == 2:
            rcs[f[0]] = f[1].split("=")[1]
rows = []
for share in SHARES:
    for run, setting in RUNS:
        tag = f"{share}-{run}"
        log = os.path.join(B, tag + ".log")
        if not os.path.exists(log):
            continue
        txt = strip(open(log, encoding="utf-8", errors="replace").read())
        cmd = os.path.join(B, tag + ".cmd")
        if run == "record" and os.path.exists(cmd) and "GGML_CUDA_MMVQ_PASCAL=2" in open(cmd, encoding="utf-8").read():
            setting = "p106 (live, record written)"
        ok = len(re.findall(r"^  \S.*\): OK$", txt, re.M))
        fails = re.findall(r"(?:^  |  )([A-Z][A-Z0-9_]*\(.*\)): FAIL$", txt, re.M)
        norec = len(re.findall(r": NOREC ", txt))
        tp = re.findall(r"(\d+)/(\d+) tests passed", txt)
        rows.append([share, run, setting, ok + len(fails), ok, len(fails), norec, rcs.get(tag, ""),
                     "; ".join(fails) or "-"])
write("bops.tsv", ["share", "run", "setting", "cases", "passed", "failed", "not recorded", "exit", "failing cases"],
      rows)

# ---- the replays, case by case
moved = collections.OrderedDict()
verdict_rows = []
with gzip.open(os.path.join(OUT, "cases.tsv.gz"), "wt", encoding="utf-8", newline="") as gz:
    w = csv.writer(gz, delimiter="\t", lineterminator="\n")
    w.writerow(["share", "case"] + [f"{s} {x}" for s in SETTINGS for x in ("verdict", "errors")])
    for share in SHARES:
        logs = {}
        for s in SETTINGS:
            logs[s] = {}
            p = os.path.join(B, f"{share}-replay-{s}.tsv")
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3 and f[2] != "NOREC":
                    logs[s][f[0]] = (f[1], f[2], f[3] if len(f) > 3 else "")
        if not logs["unset"]:
            continue
        ids = [c for c in logs["unset"] if all(c in logs[s] for s in BASE)]
        n_p106 = n_fn = n_big = n_rep = 0
        for c in ids:
            name = logs["unset"][c][0]
            op, t, n = shape(name)
            e = {s: logs[s][c][2] for s in BASE}
            e["fn-f32"] = logs["fn-f32"][c][2] if c in logs["fn-f32"] else None
            rep = e["unset"] == e["unset2"]
            key = (share, op, t, nbucket(n))
            g = moved.setdefault(key, [0, 0, 0, 0, 0, 0, 0])
            g[0] += 1
            g[1] += rep
            g[2] += e["p106"] != e["unset"]
            g[3] += e["fn"] != e["unset"]
            g[4] += e["fn"] != e["p106"]
            g[5] += e["p106-kcol0"] != e["p106"]
            g[6] += e["fn-f32"] is not None and e["fn-f32"] != e["fn"]
            n_rep += rep
            n_p106 += e["p106"] != e["unset"]
            n_fn += e["fn"] != e["unset"]
            n_big += n > 8 and (e["p106"] != e["unset"] or e["fn"] != e["unset"])
            w.writerow([share, name] + [x for s in SETTINGS for x in (logs[s].get(c, ("", "-", "-"))[1:])])
        # cases only the forced settings run: the large-activation cases, supported only where the fork's Pascal kernels
        # are on (recorded where the record step ran forced)
        only = [c for c in logs["p106"] if c not in logs["unset"]]
        forced = ("p106", "fn", "p106-kcol0", "fn-f32")
        only_fail = sum(1 for c in only if any(c in logs[s] and logs[s][c][1] != "OK" for s in forced))
        for c in only:
            w.writerow([share, logs["p106"][c][0]] + [x for s in SETTINGS for x in (logs[s].get(c, ("", "-", "-"))[1:])])
        per = []
        for s in SETTINGS:
            held = [c for c in ids if c in logs[s]]
            ok = sum(1 for c in held if logs[s][c][1] == "OK")
            per += [ok, len(held) - ok]
        verdict_rows.append([share, len(set(logs["unset"]) | set(logs["p106"])), len(ids)] + per
                            + [n_p106, n_fn, n_big, len(ids) - n_rep, len(only), only_fail])

write("verdict.tsv", ["share", "cases recorded", "cases replayed in every setting"]
      + [f"{s} {x}" for s in SETTINGS for x in ("passed", "failed")]
      + ["moved by p106", "moved by fn", "moved cases of 9+ tokens", "unset error not repeated",
         "cases replayed only forced", "of them failed in a forced setting"], verdict_rows)
write("moved.tsv", ["share", "operator", "type", "tokens", "cases", "unset error repeats", "p106 moves", "fn moves",
                    "fn moves against p106", "kcol0 moves against p106", "fn-f32 moves against fn"],
      [list(k) + v for k, v in moved.items()])
print(f"wrote {OUT}: bops.tsv ({len(rows)} runs), verdict.tsv, moved.tsv ({len(moved)} groups), cases.tsv.gz")
