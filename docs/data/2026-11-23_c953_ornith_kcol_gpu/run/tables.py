#!/usr/bin/env python3
"""C953 (Q1687 point 7): the tables of run.sh's output.

Usage: python3 -I tables.py <run.sh's WORK folder, fetched> <output folder>
Writes:
  bops.tsv        one line per test-backend-ops run: part, run, setting, cases, passed, failed, exit status
  bops-cases.tsv  one line per replayed case: part, case, type, values in (K), rows (M), tokens (n), whether
                  GGML_CUDA_MMVQ_PASCAL_MT=2's rule takes it (the fork's mmvq-pascal-pick.h, table 2, written out below),
                  the verdict and normalized squared error against the CPU with the switch unset and with =2, whether
                  the two errors differ, and which kernel family's run (the kernels step: read-once, mc, k-column)
                  gives each the same error
  bops-kernels.tsv  the kernels step's replays: part, setting, cases, passed, failed, exit status
  bops-verdict.tsv  per part: cases; the cases the rule takes and how many of them =2 computes as the read-once kernel
                  does (the same error as the kernels step's mtk0); the cases it leaves and how many =2 computes as unset;
                  the cases whose result =2 changes; the cases off the rule
  replies.tsv     every request of every load: model, arm, round, prompt, tokens, md5 of the token ids, drafted, accepted
  compare.tsv     per model and prompt, each arm against new-unset of the same model, and the drafting model's arms against
                  plain decoding (ornith new-unset): the plain greedy replies (every round) the same token
                  ids or the first differing token; for the n_probs request the first differing token, the two
                  candidates' log-probabilities there in both arms, and the largest change of the chosen token's
                  log-probability before it
"""
import csv
import glob
import json
import os
import re
import sys

WORK, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

Q5_0, Q8_0 = "q5_0", "q8_0"
BIG = 1 << 62
# table 2 of mmvq-pascal-pick.h at 37c05b76a: type, K min, K max, rows min, rows max, token counts it takes
RULE = [
    (Q5_0, 4097, BIG, 1, BIG, {2, 3, 4}),
    (Q5_0, 2049, 4096, 1, 4096, {2, 3}),
    (Q5_0, 2049, 4096, 4097, BIG, {3}),
    (Q5_0, 1, 2048, 1, BIG, {3}),
    (Q8_0, 1, 4096, 1, BIG, {2, 3, 4}),
]


def rule_takes(t, k, m, n):
    if n < 2 or n > 4:
        return False
    for rt, k0, k1, m0, m1, ns in RULE:
        if rt == t and k0 <= k <= k1 and m0 <= m <= m1:
            return n in ns
    return False


def strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def shape(case):
    """type, K, M, n of a case string (test_mul_mat's vars, or the test file's generic op)"""
    m = re.search(r"type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+)", case)
    if m:
        return m.group(1), int(m.group(4)), int(m.group(2)), int(m.group(3))
    m = re.search(r"sources=(\w+)\[(\d+),(\d+),1,1\],f32\[\d+,(\d+),1,1\]", case)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return "?", 0, 0, 0


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ---- the card suite
b = os.path.join(WORK, "bops")
rcs = {}
if os.path.exists(os.path.join(b, "rc.txt")):
    for line in open(os.path.join(b, "rc.txt"), encoding="utf-8"):
        tag, rc = line.split()
        rcs[tag] = rc.split("=")[1]
rows = []
for part in ("suite", "ornith"):
    for run, setting in (("record", "unset (live, record written)"), ("replay-unset", "unset (replayed)"),
                         ("replay-mt2", "=2 (replayed)"), ("live-mt2", "=2 (live)")):
        tag = f"{part}-{run}"
        log = os.path.join(b, tag + ".log")
        if not os.path.exists(log):
            continue
        txt = strip(open(log, encoding="utf-8", errors="replace").read())
        ok = len(re.findall(r"^  MUL_MAT\(.*\): OK$", txt, re.M))
        fail = len(re.findall(r"^  MUL_MAT\(.*\): FAIL", txt, re.M))
        tp = re.findall(r"(\d+)/(\d+) tests passed", txt)
        cases = tp[-1][1] if tp else str(ok + fail)
        rows.append([part, run, setting, cases, tp[-1][0] if tp else ok, fail, rcs.get(tag, "")])
write("bops.tsv", ["part", "run", "GGML_CUDA_MMVQ_PASCAL_MT", "cases", "passed", "failed", "exit"], rows)

cases_rows, verdict_rows = [], []
for part in ("suite", "ornith"):
    logs = {}
    for s in ("unset", "mt2", "mtk0", "kcol0", "kcol2"):
        p = os.path.join(b, f"{part}-{s}.tsv")
        logs[s] = {}
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3:
                    # the output's error (the first compared node; the inputs' entries follow)
                    logs[s][f[0]] = (f[2], f[3].split(",")[0] if len(f) > 3 else "", f[1])  # by case id
    names = [c for c in logs["unset"] if c in logs["mt2"]]
    n_take = n_diff = n_ro = n_same = 0
    off = []  # cases where =2 is neither the read-once kernel's result (the rule takes it) nor unset's (it does not)
    for c in names:
        (vu, eu, name), (vm, em, _) = logs["unset"][c], logs["mt2"][c]
        t, k, m, n = shape(name)
        take = rule_takes(t, k, m, n)
        diff = eu != em
        refs = {"read-once": logs["mtk0"].get(c), "mc": logs["kcol0"].get(c), "k-column": logs["kcol2"].get(c)}

        def same_as(e):
            # the kernel families whose run gives this case the same error; n = 1 runs the one-token kernel in mtk0 and
            # kcol0, so only k-column (kcol2) is told apart there
            if not any(refs.values()):
                return ""
            return "/".join(f for f, r in refs.items() if r is not None and r[1] == e) or "none"
        n_take += take
        n_diff += diff
        if take:
            ok = refs["read-once"] is not None and refs["read-once"][1] == em
            n_ro += ok
        else:
            ok = not diff
            n_same += ok
        if not ok:
            off.append(f"{t} K={k} M={m} n={n}")
        cases_rows.append([part, name, t, k, m, n, "yes" if take else "no", vu, eu, vm, em, "yes" if diff else "no",
                           same_as(eu), same_as(em)])
    verdict_rows.append([part, len(names), n_take, n_ro, len(names) - n_take, n_same, n_diff, "; ".join(off) or "none"])
cases_rows.sort(key=lambda r: (r[0], r[2], r[3], r[4], r[5]))
write("bops-cases.tsv", ["part", "case", "type", "K", "M", "n", "rule takes", "unset verdict", "unset error",
                         "=2 verdict", "=2 error", "errors differ", "unset error the same as",
                         "=2 error the same as"], cases_rows)
# the kernel runs' verdicts
krows = []
for part in ("suite", "ornith"):
    for s in ("mtk0", "kcol0", "kcol2"):
        p = os.path.join(b, f"{part}-{s}.tsv")
        if os.path.exists(p):
            v = [line.rstrip("\n").split("\t")[2] for line in open(p, encoding="utf-8") if line.count("\t") >= 2]
            krows.append([part, s, len(v), v.count("OK"), len(v) - v.count("OK"), rcs.get(f"{part}-replay-{s}", "")])
if krows:
    write("bops-kernels.tsv", ["part", "setting", "cases", "passed", "failed", "exit"], krows)
write("bops-verdict.tsv", ["part", "cases replayed in both settings", "cases the rule takes",
                           "of those, =2 gives the read-once kernel's result", "cases the rule leaves",
                           "of those, =2 gives unset's result", "cases whose result =2 changes", "cases off the rule"],
      verdict_rows)

# ---- the replies
rep_rows = []
loads = {}
for d in sorted(glob.glob(os.path.join(WORK, "load", "*"))):
    model, arm = os.path.basename(d).split("-", 1)
    loads[(model, arm)] = d
    p = os.path.join(d, "replies.tsv")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            rep_rows.append([model, arm] + f[:3] + [f[3][:8]] + f[5:8])
write("replies.tsv", ["model", "arm", "round", "prompt", "tokens", "md5 of the token ids", "drafted", "accepted",
                      "stop"], rep_rows)


def reply(d, tag, prompt):
    p = os.path.join(d, f"{tag}-{prompt}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cands(r, i):
    """the two candidates at token i of an n_probs reply: 'id:logprob id:logprob'"""
    try:
        top = r["completion_probabilities"][i]["top_logprobs"]
        return " ".join(f"{c['id']}:{c['logprob']:.6f}" for c in top[:2])
    except (KeyError, IndexError, TypeError):
        return ""


def chosen_lp(r):
    try:
        return [c["logprob"] for c in r["completion_probabilities"]]
    except (KeyError, TypeError):
        return []


cmp_rows = []
# each arm against new-unset of its own model, and the drafting model's arms against plain decoding (ornith new-unset)
pairs = []
for model in sorted({k[0] for k in loads}):
    for arm in ("new-mt2", "base-unset", "new-unset"):
        pairs.append((model, "new-unset", model, arm))
for arm in ("new-unset", "new-mt2", "base-unset"):
    pairs.append(("ornith", "new-unset", "ornmtp", arm))
for rmodel, rarm, model, arm in pairs:
    ref, d = loads.get((rmodel, rarm)), loads.get((model, arm))
    if not ref or not d:
        continue
    for prompt in ("code", "prose", "prog"):
        plain = []
        tags = sorted({os.path.basename(x).split("-")[0] for x in glob.glob(os.path.join(d, "r*-*.json"))})
        for tag in tags:
            a, r = reply(ref, "r1", prompt), reply(d, tag, prompt)
            if a is None or r is None:
                plain.append(f"{tag}: missing")
                continue
            i = first_diff(a.get("tokens", []), r.get("tokens", []))
            plain.append(f"{tag}: same" if i is None else f"{tag}: differs at token {i}")
        pa, pr = reply(ref, "p", prompt), reply(d, "p", prompt)
        pdiff = pc_ref = pc_arm = ""
        maxd = ""
        if pa is not None and pr is not None:
            ta, tr = pa.get("tokens", []), pr.get("tokens", [])
            i = first_diff(ta, tr)
            pdiff = "same" if i is None else str(i)
            if i is not None:
                pc_ref, pc_arm = cands(pa, i), cands(pr, i)
            la, lr = chosen_lp(pa), chosen_lp(pr)
            upto = len(ta) if i is None else i
            ds = [abs(x - y) for x, y in zip(la[:upto], lr[:upto])]
            maxd = f"{max(ds):.3g}" if ds else ""
        cmp_rows.append([f"{rmodel} {rarm}", f"{model} {arm}", prompt, "; ".join(plain), pdiff, pc_ref, pc_arm,
                         maxd])
write("compare.tsv", ["reference (round 1)", "model and arm", "prompt", "plain greedy replies, each round",
                      "n_probs reply: first differing token", "the reference's two candidates there (id:logprob)",
                      "the arm's two candidates there", "largest change of the chosen token's logprob before it"],
      cmp_rows)
print("wrote", OUT)
