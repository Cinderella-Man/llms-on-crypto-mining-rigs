#!/usr/bin/env python3
"""r148_tables.py --bitid DIR --prof DIR --out DIR: R148's tables (C793, GGML_CUDA_MMVQ_PASCAL_GLU_MT=3 on a P106).

--bitid  rig_r148.sh's bitid/ folder (multitoken_bitid glu at on (=3), nopf (=4), np1 (=3 _NP=1), m1 (=1), off and fault):
         bitid_lines.tsv (arm, check, pair, n, m, k, verdict, values that differ, values) and bitid_checks.txt: every line
         at Flash-Next's 640 x 2560 bit-identical in every arm but the fault; the lines that differ at =3 and not off; the
         fault's lines at 5-8 tokens of the new pairs (an IQ4_XS, Q4_K or Q5_K product) that do not differ.
--prof   rig_r148_prof.sh's output (sw.tsv, evals.tsv): prof_cases.tsv, per pass, pair and token count, the median over
         the replayed evaluations of the first K launch groups' GPU time in each arm, off - on, and the time a served
         graph saves, off - on + 2 x sw (sw: one SWIGLU's time without the profiler's events, the median of graph mode's
         eval_us / eval_groups at that token count) for a new pair, off - on for a Q2_0/Q3_K pair (the same launch in
         both arms); prof_est_ms.txt, per pass and token count, the est. ms a verify pass saves: the sum over the file's
         layers of layers x saved us.
Standard library only.
"""
import argparse, collections, csv, os, re, statistics

# Qwen3.8-Flash-Next's (Swift GSQ-RCO Q2_0) unfused shared-expert layers by gate+up pair (C757's shapes.py)
LAYERS = {"q2_0+q3_K": 8, "q3_K+q3_K": 6, "q3_K+q2_0": 3, "q2_0+q4_K": 3, "q2_0+iq4_xs": 3, "iq4_xs+q3_K": 3,
          "q4_K+q4_K": 2, "q4_K+q3_K": 2, "q4_K+q2_0": 2, "q3_K+iq4_xs": 2, "iq4_xs+q2_0": 2, "q5_K+q3_K": 1,
          "q3_K+q5_K": 1, "iq4_xs+q4_K": 1}
NEW = re.compile(r"q4_K|q5_K|iq4_xs")
LINE = re.compile(r"^(GLU/\S+)\s+(\S+)\s+n=(\d+)\s+m=(\d+)\s+k=(\d+)\s*:\s*(.*)$")
ARMS = ["on", "nopf", "np1", "m1", "off", "fault"]


def bitid(src, out):
    rows, summary = [], {}
    for arm in ARMS:
        path = os.path.join(src, arm + ".txt")
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                m = LINE.match(line)
                if m:
                    check, pair, n, mm, k, rest = m.groups()
                    d = re.search(r"\((\d+) of (\d+) values differ", rest)
                    verdict = rest.split(" (")[0].strip()
                    rows.append([arm, check, pair, int(n), int(mm), int(k), verdict,
                                 int(d.group(1)) if d else "", int(d.group(2)) if d else ""])
                elif "comparisons" in line:
                    summary[arm] = line.strip()
    with open(os.path.join(out, "bitid_lines.tsv"), "w") as f:
        f.write("arm\tcheck\tpair\tn\tm\tk\tverdict\tdiffer\tvalues\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    key = lambda r: (r[1], r[2], r[3], r[4], r[5])
    differs = lambda r: r[7] != "" and r[7] > 0
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r[0]].append(r)
    # the harness prints its "declined" one-token lines (a product the read-once kernels do not take) like the others;
    # only its summary line counts them apart, so the sets below hold every printed line
    lines = []
    for arm in ARMS:
        a = by_arm[arm]
        shape = [r for r in a if r[4] == 640 and r[5] == 2560]
        shapes = collections.Counter((r[4], r[5]) for r in a if differs(r))
        lines.append(f"{arm}: {len(a)} lines, {sum(differs(r) for r in a)} differ (by shape: "
                     + ", ".join(f"{m} x {k}: {c}" for (m, k), c in sorted(shapes.items()))
                     + f"); at 640 x 2560: {len(shape)} lines, {sum(differs(r) for r in shape)} differ; "
                     f"harness: {summary.get(arm, '?')}")
    diff = {arm: {key(r) for r in by_arm[arm] if differs(r)} for arm in ARMS}
    for arm in ["on", "nopf", "np1"]:
        lines.append(f"{arm} against m1 (=1): lines that differ only here {len(diff[arm] - diff['m1'])}, "
                     f"only at m1 {len(diff['m1'] - diff[arm])}")
    for arm in ["on", "nopf", "np1", "m1"]:
        only, only_off = sorted(diff[arm] - diff["off"]), sorted(diff["off"] - diff[arm])
        lines.append(f"{arm} against off: lines that differ only here {len(only)}, only off {len(only_off)}; "
                     f"their checks, tokens and shapes: here "
                     + ", ".join(f"{c} n={n} {m} x {k}: {v}" for (c, n, m, k), v in sorted(collections.Counter(
                         (x[0], x[2], x[3], x[4]) for x in only).items()))
                     + "; off " + ", ".join(f"{c} n={n} {m} x {k}: {v}" for (c, n, m, k), v in sorted(
                         collections.Counter((x[0], x[2], x[3], x[4]) for x in only_off).items())))
    fault = by_arm["fault"]
    fault_new = [r for r in fault if 5 <= r[3] <= 8 and NEW.search(r[2])]
    caught = [r for r in fault_new if differs(r)]
    big = [r for r in fault_new if r[4] == 640 and r[5] == 2560]
    pairs = sorted({r[2] for r in fault_new})
    missed_pairs = sorted({r[2] for r in fault_new} - {r[2] for r in caught})
    lines.append(f"fault: lines at 5-8 tokens of the new pairs: {len(fault_new)}, differ: {len(caught)}; "
                 f"pairs {len(pairs)}, pairs with no differing line: {len(missed_pairs)} {' '.join(missed_pairs)}; "
                 f"at 640 x 2560: {len(big)} lines of {len({r[2] for r in big})} pairs, {sum(differs(r) for r in big)} differ")
    lines.append("fault: lines at 5-8 tokens of the new pairs that do not differ (by check, shape): "
                 + ", ".join(f"{c} {m} x {k}: {n}" for (c, m, k), n in sorted(collections.Counter(
                     (r[1], r[4], r[5]) for r in fault_new if not differs(r)).items())))
    lines.append("on: the same lines (new pairs, 5-8 tokens, 33 x 8192, GLU/apart): "
                 + ", ".join(f"{v}: {c}" for v, c in sorted(collections.Counter(
                     r[6] for r in by_arm["on"] if 5 <= r[3] <= 8 and NEW.search(r[2]) and r[1] == "GLU/apart"
                     and r[4] == 33 and r[5] == 8192).items())))
    with open(os.path.join(out, "bitid_checks.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def prof(src, out):
    sw = collections.defaultdict(list)
    with open(os.path.join(src, "sw.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sw[int(r["n"])].append(float(r["eval_us"]) / int(r["eval_groups"]))
    swm = {n: statistics.median(v) for n, v in sw.items()}
    ev = collections.defaultdict(list)
    lab = {}
    with open(os.path.join(src, "evals.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["arm"], int(r["pass"]), r["pair"], int(r["n"]))
            ev[k].append(float(r["sum_k_us"]))
            lab.setdefault(k, set()).add(r["g0_label"].startswith("MUL_MAT+MUL_MAT+SWIGLU"))
    cases, est = [], collections.defaultdict(lambda: [0.0, 0.0])
    for (arm, p, pair, n) in sorted(k for k in ev if k[0] == "off"):
        on = ("on", p, pair, n)
        if on not in ev:
            continue
        o, m = statistics.median(ev[("off", p, pair, n)]), statistics.median(ev[on])
        new = bool(NEW.search(pair))
        d = o - m
        saved = d + 2 * swm[n] if new else d
        L = LAYERS.get(pair, 0)
        est[(p, n)][0 if new else 1] += L * saved / 1000
        cases.append([p, pair, L, n, int(new), "/".join(str(int(x)) for x in sorted(lab[("off", p, pair, n)])),
                      "/".join(str(int(x)) for x in sorted(lab[on])), len(ev[("off", p, pair, n)]), len(ev[on]),
                      f"{o:.2f}", f"{m:.2f}", f"{d:.2f}", f"{swm[n]:.3f}", f"{saved:.2f}"])
    with open(os.path.join(out, "prof_cases.tsv"), "w") as f:
        f.write("pass\tpair\tlayers\tn\tnew_pair\toff_fused\ton_fused\toff_evals\ton_evals\toff_med_us\ton_med_us"
                "\toff_minus_on_us\tsw_us\tsaved_us\n")
        for c in cases:
            f.write("\t".join(str(x) for x in c) + "\n")
    with open(os.path.join(out, "prof_est_ms.txt"), "w") as f:
        for n in sorted(swm):
            f.write(f"sw, {n} tokens: {swm[n]:.3f} us (median of {len(sw[n])} graph-mode evaluations)\n")
        for (p, n) in sorted(est):
            a, b = est[(p, n)]
            f.write(f"est. ms a verify pass, pass {p}, {n} tokens: {a:.3f} saved by the 22 new-pair layers; "
                    f"{b:.3f} by the 17 Q2_0/Q3_K layers (the same launch in both arms: the spread)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitid")
    ap.add_argument("--prof")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.bitid:
        bitid(a.bitid, a.out)
    if a.prof:
        prof(a.prof, a.out)
