#!/usr/bin/env python3
"""served_stack.py: R120b (Q1391). The served verify pass of rig_kern.sh's "served" step split by the six verify-pass
switches of its "on" arm: each switch's products in ms a pass, per load and request, from R49's verify_parts.py output
(labels.tsv and parts.tsv of each profiled load), and each unit's class, "on" against "off".

  python3 -I served_stack.py PARTS --out DIR > stack.md

PARTS holds verify_parts.py's --out folder per load, named <k>-<arm> (1-off, 2-on, 3-on, 4-off). The launch groups of
labels.tsv (both card groups summed) are put in units by their label, the first rule that matches winning:
  GLU_MT   (C757) the shared expert's gate, up and SWIGLU: one-op MUL_MAT of ffn_gate_shexp or ffn_up_shexp, the fused
           MUL_MAT+MUL_MAT+SWIGLU launches, SWIGLU ffn_swiglu (not the routed experts' launches that hold a shared
           expert's q2_0 gate and up: those run the same in both arms and stay in "the rest")
  F32_MT   (C754) MUL_MAT ffn_gate_inp:f32 (the router, 2,560 -> 512) and ffn_gate_inp_shexp:f32 (2,560 -> 1), apart
  Q3K_MT   (C726) every other one-op MUL_MAT of a q3_K weight
  Q6K_MT   (C727) one-op MUL_MAT of q6_K ssm_out and attn_output (6,144 -> 2,560)
  DMT_KT   (C728) one-op MUL_MAT of iq4_xs and q4_K ssm_out and attn_output (6,144 -> 2,560), by name and type
  RO8      (C729) takes 5-8-token products only; the preset's verify passes are at most 4 tokens, so it has no unit
  the rest every other launch group
"all groups" is labels.tsv's total; "sum of the parts" parts.tsv's (R41's estimator). A unit's class: "on" against "off"
over the four values each (two loads, two requests); the spread is the larger of the two arms' ranges; a gain when
every "on" value is below every "off" value and the means differ by more than the spread, a loss the other way,
negligible when the means differ by no more than the spread, mixed otherwise. Standard library only."""
import argparse, collections, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("parts"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = sorted((d for d in os.listdir(a.parts) if re.fullmatch(r"\d+-\w+", d)), key=lambda d: int(d.split("-")[0]))

ONE = re.compile(r"^MUL_MAT ([A-Za-z0-9_.]+):([A-Za-z0-9_]+):\d+x\d+$")


def unit(label):
    if label.startswith("MUL_MAT+MUL_MAT+SWIGLU ") or label == "SWIGLU ffn_swiglu":
        return "GLU_MT"
    m = ONE.match(label)
    if not m:
        return "the rest"
    name, t = m.group(1), m.group(2)
    if name in ("ffn_gate_shexp", "ffn_up_shexp"):
        return "GLU_MT"
    if name == "ffn_gate_inp" and t == "f32":
        return "F32_MT router"
    if name == "ffn_gate_inp_shexp" and t == "f32":
        return "F32_MT shexp gate"
    if t == "q3_K":
        return "Q3K_MT"
    if t == "q6_K" and name in ("ssm_out", "attn_output"):
        return "Q6K_MT"
    if t in ("iq4_xs", "q4_K") and name in ("ssm_out", "attn_output"):
        return f"DMT_KT {name}:{t}"
    return "the rest"


UNITS = ["Q3K_MT", "GLU_MT", "Q6K_MT", "DMT_KT ssm_out:iq4_xs", "DMT_KT ssm_out:q4_K", "DMT_KT attn_output:q4_K",
         "F32_MT router", "F32_MT shexp gate", "the rest", "all groups", "sum of the parts"]
val = collections.defaultdict(float)  # (run, trace, unit) -> ms a pass
traces = []
for r in runs:
    for line in open(os.path.join(a.parts, r, "labels.tsv")):
        f = line.rstrip("\n").split("\t")
        if f[0] == "trace" or len(f) < 5:
            continue
        if (r, f[0]) not in traces:
            traces.append((r, f[0]))
        ms = float(f[4])
        val[(r, f[0], unit(f[2]))] += ms
        val[(r, f[0], "all groups")] += ms
    for line in open(os.path.join(a.parts, r, "parts.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) == 5 and f[1] == "all cards" and f[3] == "sum of the parts":
            val[(r, f[0], "sum of the parts")] = float(f[4])


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


with open(os.path.join(a.out, "stack.tsv"), "w") as w:
    w.write("unit\t" + "\t".join(f"{r}_{t}" for r, t in traces) + "\ton_minus_off_ms\tspread_ms\tclass\n")
    rows = []
    for u in UNITS:
        on = [val[(r, t, u)] for r, t in traces if r.endswith("-on")]
        off = [val[(r, t, u)] for r, t in traces if r.endswith("-off")]
        k, d, sp = klass(on, off)
        rows.append((u, k, d, sp))
        w.write(f"{u}\t" + "\t".join(f"{val[(r, t, u)]:.3f}" for r, t in traces) + f"\t{d:+.3f}\t{sp:.3f}\t{k}\n")

print("## The verify pass by switch, all cards, ms a pass\n")
print("| unit | " + " | ".join(f"{r} {t}" for r, t in traces) + " | on − off (means) | spread | class |")
print("|---" + "|---:" * (len(traces) + 2) + "|---|")
for (u, k, d, sp) in rows:
    print(f"| {u} | " + " | ".join(f"{val[(r, t, u)]:.3f}" for r, t in traces) + f" | {d:+.3f} | {sp:.3f} | {k} |")
