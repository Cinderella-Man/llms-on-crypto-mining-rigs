#!/usr/bin/env python3
"""types.py: C1146's routed gate/up product per weight type and setting (Q1871).

  python3 -I types.py --arm unset=verify/g0 --arm c=verify/g1 --arm ptx=verify/g2 \
      --ref r125=<R125's verify/gg> [--requests d0p2,d8192p2,d0p1,d8192p1] [--tsv types.tsv] > types.md

Each folder holds one subfolder per request written by R49's verify_parts.py --out (labels.tsv: ms a verify pass by
launch-group label, per card group; parts.tsv: R41's parts). A routed gate/up launch group of type T is the label
"MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:T:2560x640 ffn_up_exps:T:2560x640"; its ms a pass and groups a pass are
summed over the card groups, and ms a layer = ms a pass / groups a pass. Tables per request:
  1. ms a layer per type: the reference (R125's gg load) and every arm, each arm against the first arm, and against
     the same arm's Q2_0 layer;
  2. ms a pass per type and what each arm saves against the first arm (ms a layer saved x layers);
  3. the verify pass by R41's part, all cards, per arm.
--tsv writes every value in long form: table, request, key, arm, groups_per_pass, ms.
Standard library only.
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict

TYPES = ["iq2_xxs", "iq2_s", "iq2_xs", "iq1_m", "q2_0"]
SHAPE = "2560x640"


def label_of(t):
    return f"MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps:{t}:{SHAPE} ffn_up_exps:{t}:{SHAPE}"


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load(folder, req):
    d = os.path.join(folder, req)
    if not os.path.isfile(os.path.join(d, "labels.tsv")):
        return None
    lab = {t: [0.0, 0.0] for t in TYPES}  # type -> [ms a pass, groups a pass]
    want = {label_of(t): t for t in TYPES}
    for r in read_tsv(os.path.join(d, "labels.tsv")):
        if r["trace"].startswith("verify") and r["label"] in want:
            v = lab[want[r["label"]]]
            v[0] += float(r["ms_per_pass"])
            v[1] += float(r["groups_per_pass"])
    parts = OrderedDict()
    passes = 0
    for r in read_tsv(os.path.join(d, "parts.tsv")):
        if r["trace"].startswith("verify") and r["cards"] == "all cards" and not r["part"].startswith("by position"):
            parts[r["part"]] = float(r["ms_per_pass"])
            passes = int(r["passes"])
    return lab, parts, passes


def per_layer(v):
    return v[0] / v[1] if v[1] else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=folder, the first is the base")
    ap.add_argument("--ref", help="name=folder of a reference load (R125's gg)")
    ap.add_argument("--requests", default="d0p2,d8192p2,d0p1,d8192p1")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    arms = [x.split("=", 1) for x in a.arm]
    ref = a.ref.split("=", 1) if a.ref else None
    base = arms[0][0]
    long = []
    P = print
    P("# The routed gate/up product by type and setting\n")
    P("ms of the four-token verify pass, the kept passes of the request's typical steps (R49's verify_parts.py, the"
      " profiler's cost subtracted per span), all nine cards; one load per arm.\n")
    for req in a.requests.split(","):
        data = OrderedDict()
        for name, folder in ([ref] if ref else []) + arms:
            got = load(folder, req)
            if got is not None:
                data[name] = got
        if base not in data:
            P(f"## {req}: no data for `{base}`\n")
            continue
        names = list(data)
        P(f"## Request {req}\n")
        P("Kept verify passes: " + ", ".join(f"`{n}` {data[n][2]}" for n in names) + ".\n")
        P("### 1. ms a layer (one routed gate/up launch group, 2560 × 640 per expert, gate and up)\n")
        hdr = "| Type | layers | " + " | ".join(f"`{n}`" for n in names)
        hdr += " | " + " | ".join(f"`{n}` − `{base}`" for n, _ in arms[1:])
        hdr += " | " + " | ".join(f"`{n}` / its Q2_0" for n, _ in arms) + " |"
        P(hdr)
        P("|---|---:|" + "---:|" * (len(names) + len(arms) - 1 + len(arms)))
        for t in TYPES:
            layers = data[base][0][t][1]
            row = [t.upper(), f"{layers:.0f}"]
            for n in names:
                v = data[n][0][t]
                row.append(f3(per_layer(v)))
                long.append(("per_layer", req, t, n, f"{v[1]:.2f}", f"{v[0]:.4f}"))
            b = per_layer(data[base][0][t])
            for n, _ in arms[1:]:
                x = per_layer(data[n][0][t]) if n in data else None
                row.append("—" if x is None or b is None else f"{x - b:+.3f} ({(x - b) / b * 100:+.1f} %)")
            for n, _ in arms:
                x = per_layer(data[n][0][t]) if n in data else None
                q = per_layer(data[n][0]["q2_0"]) if n in data else None
                row.append("—" if x is None or not q else f"{x / q:.2f}")
            P("| " + " | ".join(row) + " |")
        P("\n### 2. ms a pass per type, and what each setting saves against the first\n")
        P("| Type | layers | " + " | ".join(f"`{n}` ms a pass" for n in names) + " | "
          + " | ".join(f"saved by `{n}`" for n, _ in arms[1:]) + " |")
        P("|---|---:|" + "---:|" * (len(names) + len(arms) - 1))
        tot = {n: 0.0 for n in names}
        for t in TYPES:
            row = [t.upper(), f"{data[base][0][t][1]:.0f}"]
            for n in names:
                row.append(f"{data[n][0][t][0]:.3f}")
                tot[n] += data[n][0][t][0]
            for n, _ in arms[1:]:
                row.append("—" if n not in data else f"{data[base][0][t][0] - data[n][0][t][0]:+.3f}")
            P("| " + " | ".join(row) + " |")
        row = ["**the five types**", ""] + [f"**{tot[n]:.3f}**" for n in names]
        row += ["—" if n not in data else f"**{tot[base] - tot[n]:+.3f}**" for n, _ in arms[1:]]
        P("| " + " | ".join(row) + " |")
        P("\n### 3. The verify pass by R41's part, all cards\n")
        P("| Part | " + " | ".join(f"`{n}`" for n in names) + " | "
          + " | ".join(f"`{n}` − `{base}`" for n, _ in arms[1:]) + " |")
        P("|---|" + "---:|" * (len(names) + len(arms) - 1))
        for k in data[base][1]:
            row = [k] + [f3(data[n][1].get(k)) for n in names]
            for n, _ in arms[1:]:
                x = data[n][1].get(k) if n in data else None
                row.append("—" if x is None else f"{x - data[base][1][k]:+.3f}")
                long.append(("parts", req, k, n, "", f3(x)))
            long.append(("parts", req, k, base, "", f3(data[base][1][k])))
            P("| " + " | ".join(row) + " |")
        P()
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("table\trequest\tkey\tarm\tgroups_per_pass\tms\n")
            for r in long:
                f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
