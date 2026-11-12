#!/usr/bin/env python3
"""R73 (1) (Q1272): the tables of the qsa-prompt replay on a P106.

Reads scripts/replay_checks.sh's output (OUT: results.tsv, console.txt, <run>.tsv, <run>.log), the set's run list and
case list (C705's folder: runs.tsv, set.tsv) and FN63's replay through the compute_61 PTX (its replay/ folder), and
writes in DEST:
  results.tsv      the replay's verdict per run, with the cases the filter selected that the set holds no record of
  rule_check.tsv   per run: the cases replayed and passing, the kernel the rule gives each (S sparse, d dense) and
                   whether each case's errors equal qp-stock's, the cases where rule and errors disagree
  prompt_cases.tsv the 24 prompt cases (12 shapes x F16/Q8_0) x 7 runs: C705's table (S or d), the rule as computed
                   here, the errors against qp-stock's (= or x), and the verdict
  cases.tsv        per case and run: verdict, = or x against qp-stock's errors, the rule's kernel
  fn63_side.tsv    per case: qp-stock's and qp-m2's first error on the P106 and through the PTX on the RTX 3080 (FN63)
  faults.tsv       per fault run: the cases failing, and how many of them the kernel takes in mode 2
Prints a short summary.

Which cases the sparse kernel can take at all (its shape and type limits) comes from FN63's qp-m2 through the PTX: a
case whose errors there differ from FN63's qp-stock took the kernel (mode 2 with _PROMPT=1 _PROMPT_NQ=512 sends every
call it can take). The rule (fasp_rule_takes in ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh) then decides per run:
  mode 0: dense; n_q <= _NQ (8): mode 2 takes it, mode 1 from n_kv >= _MIN_RATIO x n_kv_max x n_q;
  n_q > _NQ: only with _PROMPT=1 and n_q <= _PROMPT_NQ, then mode 2 takes it, mode 1 from n_kv >= _PROMPT_MIN_RATIO x n_kv_max.
Usage: python3 -I r73_tables.py OUT C705_DIR FN63_REPLAY DEST
"""
import os
import re
import sys

OUT, C705, FN63, DEST = sys.argv[1:5]
os.makedirs(DEST, exist_ok=True)

# C705's table of the prompt cases' kernel per run (README "The check set for a P106", rule.txt)
PROMPT_SHAPES = [  # (n_q, cells, n_kv_max, sequences)
    (9, 8192, 2051, 1), (16, 8192, 2051, 1), (64, 8192, 2051, 1), (128, 8192, 2051, 1),
    (9, 3072, 2051, 1), (128, 3072, 2051, 1), (128, 32768, 2051, 1), (64, 4096, 2051, 2),
    (32, 4096, 1027, 1), (129, 4096, 2051, 1), (256, 4096, 2051, 1), (512, 4096, 2051, 1),
]
TABLE_RUNS = ["qp-sparse1", "qp-prompt", "qp-r100", "qp-pratio", "qp-pnq512", "qp-m2", "qp-m2-off"]
TABLE = {
    (9, 8192, 2051, 1): "dSSSSSd", (16, 8192, 2051, 1): "dSSSSSd", (64, 8192, 2051, 1): "dSSSSSd",
    (128, 8192, 2051, 1): "dSSSSSd", (9, 3072, 2051, 1): "dSSdSSd", (128, 3072, 2051, 1): "dSSdSSd",
    (128, 32768, 2051, 1): "dSSSSSd", (64, 4096, 2051, 2): "dSSdSSd", (32, 4096, 1027, 1): "dSSSSSd",
    (129, 4096, 2051, 1): "ddddSSd", (256, 4096, 2051, 1): "ddddSSd", (512, 4096, 2051, 1): "ddddSSd",
}


def read_tsv(path):
    with open(path) as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def parse_case(s):
    p = {}
    for k in ("hsk", "hsv", "nh", "kv", "nb", "n_kv_max", "sinks"):
        m = re.search(r"[(,]%s=(-?\d+)" % k, s)
        p[k] = int(m.group(1)) if m else None
    m = re.search(r"nr23=\[(\d+),(\d+)\]", s)
    p["nr2"], p["nr3"] = int(m.group(1)), int(m.group(2))
    p["type_K"] = re.search(r"type_K=(\w+)", s).group(1)
    p["type_V"] = re.search(r"type_V=(\w+)", s).group(1)
    return p


def per_case(path):  # id -> (verdict, errors as printed)
    d = {}
    if os.path.exists(path):
        for r in read_tsv(path):
            d[r[0]] = (r[2], r[3] if len(r) > 3 else "")
    return d


def settings(envs):
    e = dict(kv.split("=", 1) for kv in envs.split() if "=" in kv)
    g = lambda k, dflt: e.get("GGML_CUDA_FATTN_SPARSE_PASCAL" + k, dflt)
    return dict(mode=int(g("", "0")), nq=int(g("_NQ", "8")), min_ratio=float(g("_MIN_RATIO", "1.5")),
                prompt=int(g("_PROMPT", "0")), prompt_nq=int(g("_PROMPT_NQ", "128")),
                prompt_min_ratio=float(g("_PROMPT_MIN_RATIO", "1.0")), fault=int(g("_FAULT", "0")))


def rule_takes(r, n_q, n_kv, n_kv_max):  # fasp_rule_takes without _UNION
    if r["mode"] == 0:
        return False
    if n_q <= r["nq"]:
        return r["mode"] != 1 or not (n_kv < r["min_ratio"] * n_kv_max * n_q)
    if r["prompt"] == 0 or n_q > r["prompt_nq"]:
        return False
    return r["mode"] != 1 or not (n_kv < r["prompt_min_ratio"] * n_kv_max)


runs = [r for r in read_tsv(os.path.join(C705, "runs.tsv"))]
run_names = [r[0] for r in runs]
run_set = {r[0]: settings(r[1]) for r in runs}
run_expect = {r[0]: r[4] for r in runs}
cases = {r[0]: r[1] for r in read_tsv(os.path.join(C705, "set.tsv"))[1:]}
shape = {i: parse_case(s) for i, s in cases.items()}

res = {r[0]: r for r in read_tsv(os.path.join(OUT, "results.tsv"))[1:]}
pc = {run: per_case(os.path.join(OUT, run + ".tsv")) for run in run_names}
f_stock = per_case(os.path.join(FN63, "qp-stock.tsv"))
f_m2 = per_case(os.path.join(FN63, "qp-m2.tsv"))
eligible = {i for i in f_m2 if i in f_stock and f_m2[i][1] != f_stock[i][1]}
stock = pc["qp-stock"]


def log_summary(run):
    path = os.path.join(OUT, run + ".log")
    txt = open(path, errors="replace").read() if os.path.exists(path) else ""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    m = re.search(r"(\d+) cases replayed, (\d+) selected by the filter have no record", txt)
    cc = re.search(r"replay cpu-check: (\d+) cases, (\d+) references identical, (\d+) verdicts differ", txt)
    norec = re.findall(r"no record[^\n]*", txt)
    return (m.group(1) if m else "", m.group(2) if m else "", "/".join(cc.groups()) if cc else "", norec)


# results.tsv
with open(os.path.join(DEST, "results.tsv"), "w") as f:
    f.write("run\tswitches\texpect\tcases_passed\tfail_lines\texit\tverdict\treplayed\tselected_no_record\tcpu_check(cases/identical/differ)\n")
    for run in run_names:
        r = res.get(run)
        rep, nr, cc, _ = log_summary(run)
        if r:
            f.write("\t".join([run, r[1], r[2], r[3], r[4], r[5], r[7], rep, nr, cc]) + "\n")

# cases.tsv, rule_check.tsv
lines = []
summary = []
with open(os.path.join(DEST, "rule_check.tsv"), "w") as rc:
    rc.write("run\texpect\treplayed\tOK\tFAIL\trule_S\trule_d\tS_and_differs\td_and_equal\tS_but_equal\td_but_differs\tnot_judged\n")
    for run in run_names:
        if run == "qp-stock":
            continue
        rs = run_set[run]
        cnt = dict(rep=0, ok=0, fail=0, S=0, d=0, Sx=0, de=0, Se=0, dx=0, nj=0)
        for i, (v, err) in pc[run].items():
            cnt["rep"] += 1
            cnt["ok" if v == "OK" else "fail"] += 1
            p = shape[i]
            S = i in eligible and rule_takes(rs, p["nb"], p["kv"], p["n_kv_max"])
            cnt["S" if S else "d"] += 1
            if i not in stock or stock[i][0] != "OK" or v != "OK":
                cnt["nj"] += 1
                continue
            same = err == stock[i][1]
            if S and not same:
                cnt["Sx"] += 1
            elif not S and same:
                cnt["de"] += 1
            elif S and same:
                cnt["Se"] += 1
            else:
                cnt["dx"] += 1
        rc.write("\t".join(map(str, [run, run_expect[run], cnt["rep"], cnt["ok"], cnt["fail"], cnt["S"], cnt["d"],
                                     cnt["Sx"], cnt["de"], cnt["Se"], cnt["dx"], cnt["nj"]])) + "\n")
        summary.append((run, cnt))

with open(os.path.join(DEST, "cases.tsv"), "w") as f:
    f.write("id\thsk\thsv\tkv_heads\tq_heads_per_kv\tsequences\tcells\tquery_tokens\tn_kv_max\ttype_K\ttype_V\tsparse_in_mode2_ptx\t"
            + "\t".join(run_names) + "\n")
    for i in sorted(stock, key=lambda x: (shape[x]["nb"], shape[x]["kv"], x)):
        p = shape[i]
        cells = []
        for run in run_names:
            v, err = pc[run].get(i, ("-", ""))
            if run == "qp-stock":
                cells.append(v)
                continue
            S = i in eligible and rule_takes(run_set[run], p["nb"], p["kv"], p["n_kv_max"])
            same = "=" if err == stock[i][1] else "x"
            cells.append("%s %s %s" % (v, same, "S" if S else "d"))
        f.write("\t".join(map(str, [i, p["hsk"], p["hsv"], p["nh"], p["nr2"], p["nr3"], p["kv"], p["nb"], p["n_kv_max"],
                                    p["type_K"], p["type_V"], "yes" if i in eligible else "no"] + cells)) + "\n")

# prompt_cases.tsv
pmis = 0
pn = 0
with open(os.path.join(DEST, "prompt_cases.tsv"), "w") as f:
    f.write("id\tquery_tokens\tcells\tn_kv_max\tsequences\ttype\t"
            + "\t".join(r + " (table rule errors verdict)" for r in TABLE_RUNS) + "\tqp-stock\n")
    for i in sorted(cases, key=lambda x: (PROMPT_SHAPES.index((shape[x]["nb"], shape[x]["kv"], shape[x]["n_kv_max"], shape[x]["nr3"]))
                                           if (shape[x]["nb"], shape[x]["kv"], shape[x]["n_kv_max"], shape[x]["nr3"]) in PROMPT_SHAPES else 99, x)):
        p = shape[i]
        key = (p["nb"], p["kv"], p["n_kv_max"], p["nr3"])
        if key not in TABLE or p["type_K"] not in ("f16", "q8_0") or p["type_K"] != p["type_V"] or p["hsk"] not in (128, 256):
            continue
        pn += 1
        row = [i, p["nb"], p["kv"], p["n_kv_max"], p["nr3"], p["type_K"]]
        for k, run in enumerate(TABLE_RUNS):
            t = TABLE[key][k]
            rr = "S" if (i in eligible and rule_takes(run_set[run], p["nb"], p["kv"], p["n_kv_max"])) else "d"
            v, err = pc[run].get(i, ("-", ""))
            same = "=" if (i in stock and err == stock[i][1]) else "x"
            ok = (t == rr) and ((t == "S") == (same == "x")) and v == "OK"
            pmis += 0 if ok else 1
            row.append("%s %s %s %s" % (t, rr, same, v))
        row.append(stock.get(i, ("-", ""))[0])
        f.write("\t".join(map(str, row)) + "\n")

# fn63_side.tsv
ident = dict(stock=0, m2=0, n=0)
with open(os.path.join(DEST, "fn63_side.tsv"), "w") as f:
    f.write("id\tquery_tokens\tcells\ttype_K\tsparse_in_mode2_ptx\tqp-stock_p106\tqp-stock_ptx\tqp-m2_p106\tqp-m2_ptx\tstock_identical\tm2_identical\n")
    for i in sorted(stock, key=lambda x: (shape[x]["nb"], shape[x]["kv"], x)):
        p = shape[i]
        a, b = stock[i][1].split(",")[0], f_stock.get(i, ("", ""))[1].split(",")[0]
        c, d = pc["qp-m2"].get(i, ("", ""))[1].split(",")[0], f_m2.get(i, ("", ""))[1].split(",")[0]
        si = stock[i][1] == f_stock.get(i, ("", ""))[1]
        mi = pc["qp-m2"].get(i, ("", ""))[1] == f_m2.get(i, ("", ""))[1]
        ident["n"] += 1
        ident["stock"] += si
        ident["m2"] += mi
        f.write("\t".join(map(str, [i, p["nb"], p["kv"], p["type_K"], "yes" if i in eligible else "no", a, b, c, d,
                                    "yes" if si else "no", "yes" if mi else "no"])) + "\n")

# faults.tsv
with open(os.path.join(DEST, "faults.tsv"), "w") as f:
    f.write("run\texpect\treplayed\tfailing\tfailing_that_mode2_sends_to_the_kernel\tlargest_failing_error\n")
    for run in run_names:
        if run_set[run]["fault"] == 0:
            continue
        fails = [i for i, (v, _) in pc[run].items() if v != "OK"]
        big = max((float(pc[run][i][1].split(",")[0]) for i in fails if pc[run][i][1]), default=0.0)
        f.write("\t".join(map(str, [run, run_expect[run], len(pc[run]), len(fails), sum(i in eligible for i in fails),
                                    "%.6g" % big if fails else "-"])) + "\n")

print("eligible (sparse in FN63's qp-m2):", len(eligible), "of", len(f_m2))
print("qp-stock replayed", len(stock), "OK", sum(v == "OK" for v, _ in stock.values()))
for run, c in summary:
    print("%-11s rep %3d ok %3d fail %3d  S %2d d %2d  S&x %2d d&= %2d  S&= %d d&x %d  not judged %d"
          % (run, c["rep"], c["ok"], c["fail"], c["S"], c["d"], c["Sx"], c["de"], c["Se"], c["dx"], c["nj"]))
print("prompt cases", pn, "cells disagreeing (table vs rule vs errors vs verdict):", pmis)
print("FN63 side by side: %d cases; qp-stock errors identical %d, qp-m2 identical %d" % (ident["n"], ident["stock"], ident["m2"]))
