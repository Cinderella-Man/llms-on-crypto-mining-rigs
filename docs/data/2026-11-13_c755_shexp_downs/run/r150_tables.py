#!/usr/bin/env python3
"""r150_tables.py: R150 (Q1456), C755 on a P106. The perf and served tables of rig_c755.sh's output.

  python3 -I r150_tables.py --perf <out/perf/us_run.tsv> --parts <dir> --replies <replies.tsv> --out <dir> > r150.md

--perf     rig_c755.sh's us_run.tsv (arm, pass, type, m, k, n, us a call)
--parts    R49's verify_parts.py --out folder per profiled load, named <k>-<arm> (1-off, 2-on, 3-on, 4-off)
--replies  2026-11-11_r79_c722_p106/run/served_table.py's replies.tsv (the server's own timings of every request)

perf: per type and token count, each arm's us a call in each pass, mc - kcol, and the est. ms a verify pass that
difference makes for the served file's shared-expert downs (16 Q4_0 and 7 Q5_0 layers, one product each a pass). At
1 token both arms run the one-token kernel, so its mc - kcol is the spread of a pair.
served: the launch groups of labels.tsv (both card groups summed), ms a verify pass: the Q4_0 and Q5_0 shared-expert
downs (MUL_MAT ffn_down_shexp:q4_0 / q5_0), the rest, all groups; and the warm loads' ms a step from the server.
A unit's class, "on" against "off" over the values of the two loads each (two requests a load): the spread is the
larger of the two arms' ranges; gain when every "on" value is below every "off" value and the means differ by more
than the spread, loss the other way, negligible when the means differ by no more than the spread, mixed otherwise
(served_stack.py's rule). Standard library only."""
import argparse, collections, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--perf", required=True); ap.add_argument("--parts", required=True)
ap.add_argument("--replies", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bitid", default="")  # rig_c755.sh's out/bitid: one <arm>.txt per arm
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LAYERS = {"q4_0": 16, "q5_0": 7}

# bitid: per arm, input and type, the comparisons and those with differences (values differing, summed)
if a.bitid:
    BL = re.compile(r"^(MUL_MAT\S*)\s+(q4_0|q5_0)\s+n=(\d+) m=(\d+)\s+k=(\d+)\s+: (bit-identical|DIFFERENT) \((\d+) of (\d+)")
    print("## bitid: multitoken_bitid shdown, 2-4 tokens against one-token products\n")
    print("| arm | the hook takes q4_0, q5_0 | input | type | comparisons | with differences | values differing | values |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    with open(os.path.join(a.out, "bitid.tsv"), "w") as w:
        w.write("arm\thook_takes\tinput\ttype\tcomparisons\tdiffering\tvalues_differing\tvalues\n")
        for arm in ("on", "table", "off", "mc0"):
            p = os.path.join(a.bitid, arm + ".txt")
            if not os.path.exists(p):
                continue
            c = collections.OrderedDict()
            hook = {}
            for line in open(p):
                m = re.match(r"^(q4_0|q5_0): the k-column hook (takes|leaves)", line)
                if m:
                    hook[m.group(1)] = m.group(2)
                m = BL.match(line)
                if m:
                    k = (m.group(1), m.group(2))
                    x = c.setdefault(k, [0, 0, 0, 0])
                    x[0] += 1; x[1] += m.group(6) == "DIFFERENT"; x[2] += int(m.group(7)); x[3] += int(m.group(8))
            hk = ", ".join(hook.get(t, "?") for t in ("q4_0", "q5_0"))
            for (inp, t), x in c.items():
                row = [arm, hk, inp, t] + [str(v) for v in x]
                w.write("\t".join(row) + "\n")
                print("| " + " | ".join(row) + " |")
    print()


def klass(on, off):
    sp = max(max(on) - min(on), max(off) - min(off))
    d = statistics.mean(on) - statistics.mean(off)
    if abs(d) <= sp:
        return "negligible", d, sp
    if max(on) < min(off):
        return "gain", d, sp
    if min(on) > max(off):
        return "loss", d, sp
    return "mixed", d, sp


# perf
us = {}  # (arm, pass, type, n) -> us a call
for line in open(a.perf):
    f = line.rstrip("\n").split("\t")
    if len(f) == 7:
        us[(f[0], int(f[1]), f[2], int(f[5]))] = float(f[6])
passes = sorted({p for _, p, _, _ in us})
types = [t for t in ("q4_0", "q5_0") if any(k[2] == t for k in us)]
ns = sorted({n for _, _, _, n in us})
print("## perf: us a call, m = 2,560, k = 640\n")
hdr = ["type", "n"] + [f"{arm} p{p}" for p in passes for arm in ("kcol", "mc")] + [f"mc-kcol p{p}" for p in passes] \
    + [f"est ms a pass p{p}" for p in passes]
print("| " + " | ".join(hdr) + " |\n|" + "---|" * 2 + "---:|" * (len(hdr) - 2))
with open(os.path.join(a.out, "perf.tsv"), "w") as w:
    w.write("\t".join(h.replace(" ", "_") for h in hdr) + "\n")
    tot = collections.defaultdict(float)  # (n, pass) -> est ms a pass, both types
    for t in types:
        for n in ns:
            v = [us.get((arm, p, t, n)) for p in passes for arm in ("kcol", "mc")]
            d = [us[("mc", p, t, n)] - us[("kcol", p, t, n)] for p in passes]
            e = [x * LAYERS[t] / 1000 for x in d]
            for p, x in zip(passes, e):
                tot[(n, p)] += x
            row = [t, str(n)] + [f"{x:.2f}" for x in v] + [f"{x:+.2f}" for x in d] + [f"{x:+.3f}" for x in e]
            w.write("\t".join(row) + "\n")
            print("| " + " | ".join(row) + " |")
print("\nBoth types, est. ms a pass (mc - kcol; 16 Q4_0 + 7 Q5_0 products):\n")
print("| n | " + " | ".join(f"pass {p}" for p in passes) + " |\n|---:|" + "---:|" * len(passes))
with open(os.path.join(a.out, "perf_pass.tsv"), "w") as w:
    w.write("n\t" + "\t".join(f"est_ms_pass{p}" for p in passes) + "\n")
    for n in ns:
        w.write(f"{n}\t" + "\t".join(f"{tot[(n, p)]:+.3f}" for p in passes) + "\n")
        print(f"| {n} | " + " | ".join(f"{tot[(n, p)]:+.3f}" for p in passes) + " |")

# served
runs = sorted((d for d in os.listdir(a.parts) if re.fullmatch(r"\d+-\w+", d)), key=lambda d: int(d.split("-")[0]))
UNITS = ["Q4_0 shexp down", "Q5_0 shexp down", "both downs", "the rest", "all groups"]
val = collections.defaultdict(float)
groups = collections.defaultdict(float)
traces = []
for r in runs:
    for line in open(os.path.join(a.parts, r, "labels.tsv")):
        f = line.rstrip("\n").split("\t")
        if f[0] == "trace" or len(f) < 5:
            continue
        if (r, f[0]) not in traces:
            traces.append((r, f[0]))
        ms = float(f[4])
        m = re.fullmatch(r"MUL_MAT ffn_down_shexp:(q4_0|q5_0):640x2560", f[2])
        if m:
            u = m.group(1).upper() + " shexp down"
            val[(r, f[0], u)] += ms
            val[(r, f[0], "both downs")] += ms
            groups[(r, f[0], u)] += float(f[3])
        else:
            val[(r, f[0], "the rest")] += ms
        val[(r, f[0], "all groups")] += ms
print("\n## served: the verify pass, all cards, ms a pass\n")
cols = [f"{r} {t}" for r, t in traces]
print("| unit | " + " | ".join(cols) + " | on - off (means) | spread | class |\n|---|" + "---:|" * (len(cols) + 2) + "---|")
with open(os.path.join(a.out, "served.tsv"), "w") as w:
    w.write("unit\t" + "\t".join(c.replace(" ", "_") for c in cols) + "\ton_minus_off_ms\tspread_ms\tclass\n")
    for u in UNITS:
        on = [val[(r, t, u)] for r, t in traces if r.endswith("-on")]
        off = [val[(r, t, u)] for r, t in traces if r.endswith("-off")]
        k, d, sp = klass(on, off)
        vs = [f"{val[(r, t, u)]:.3f}" for r, t in traces]
        w.write(f"{u}\t" + "\t".join(vs) + f"\t{d:+.3f}\t{sp:.3f}\t{k}\n")
        print(f"| {u} | " + " | ".join(vs) + f" | {d:+.3f} | {sp:.3f} | {k} |")
print("\nLaunch groups a pass (Q4_0, Q5_0): " + "; ".join(
    f"{r} {t} {groups[(r, t, 'Q4_0 shexp down')]:.0f}, {groups[(r, t, 'Q5_0 shexp down')]:.0f}" for r, t in traces))

# the server's ms a step, warm loads
rep = [l.rstrip("\n").split("\t") for l in open(a.replies)]
h = rep[0]
rows = [dict(zip(h, f)) for f in rep[1:] if len(f) == len(h)]
print("\n## served: the server's own timings\n")
print("| run | load | request | same_text_as_first | steps_server | tokens_a_step | predicted_ms | ms_a_step_server | tok_s |")
print("|---|---|---|---|---:|---:|---:|---:|---:|")
for x in rows:
    print("| " + " | ".join(x[c] for c in ("run", "load", "request", "same_text_as_first", "steps_server", "tokens_a_step",
                                           "predicted_ms", "ms_a_step_server", "tok_s")) + " |")
print("\n| load | request | on - off ms a step (means) | spread | class |\n|---|---|---:|---:|---|")
with open(os.path.join(a.out, "step.tsv"), "w") as w:
    w.write("load\trequest\ton_values\toff_values\ton_minus_off_ms\tspread_ms\tclass\n")
    for load in ("warm", "prof"):
        for req in sorted({x["request"] for x in rows}):
            on = [float(x["ms_a_step_server"]) for x in rows if x["load"] == load and x["request"] == req and x["run"].endswith("-on")]
            off = [float(x["ms_a_step_server"]) for x in rows if x["load"] == load and x["request"] == req and x["run"].endswith("-off")]
            if not on or not off:
                continue
            k, d, sp = klass(on, off)
            w.write(f"{load}\t{req}\t{','.join(map(str, on))}\t{','.join(map(str, off))}\t{d:+.2f}\t{sp:.2f}\t{k}\n")
            print(f"| {load} | {req} | {d:+.2f} | {sp:.2f} | {k} |")
