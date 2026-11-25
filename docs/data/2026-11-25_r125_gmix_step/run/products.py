#!/usr/bin/env python3
"""products.py: R125's verify pass of two model files compared part by part and product by product (Q1409).

  python3 -I products.py --base pmix=verify/pg --other gmix=verify/gg [--requests d0p2,d8192p2] \
      [--min-ms 0.02] [--tsv products.tsv] > products.md

Each folder holds one subfolder per request (d0p2, d8192p2, ...) written by R49's verify_parts.py --out over that
request's piece of a group-mode GGML_CUDA_PROFILE trace (R121's split_trace.py): parts.tsv (R41's parts) and
labels.tsv (ms a verify pass by launch-group label, cards 0 to 7 and the last card). A label is the launch group's ops
and its weights as name:type:shape. Tables per request:
  1. the verify pass by R41's part, all cards: each file's ms and the difference;
  2. by product: a product is the label without the types and shapes (the same weights in either file), summed over
     both card groups; each file's ms, launch groups a pass, the types it holds, and the difference, largest first;
     products under --min-ms difference are summed in one row;
  3. every label (with its types) of the products in table 2 that differ, each file's ms and groups a pass.
--tsv writes every value in long form: table, request, key, file, groups_per_pass, ms.
Standard library only.
"""
import argparse
import csv
import os
import re
import sys
from collections import OrderedDict, defaultdict


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def product_of(label):
    """-> (product, types): the label's op chain and weight names, and the types it holds."""
    head, _, rest = label.partition(" ")
    names, types = [], []
    for tok in rest.split():
        m = re.match(r"^([^:]+):([^:]+):(\S+)$", tok)
        if m:
            names.append(m.group(1))
            types.append(m.group(2))
        else:
            names.append(tok)
    return head + " " + " ".join(names), types


def load(folder, request):
    d = os.path.join(folder, request)
    parts = OrderedDict()
    for r in read_tsv(os.path.join(d, "parts.tsv")):
        if r["trace"].startswith("verify") and r["cards"] == "all cards" and not r["part"].startswith("by position"):
            parts[r["part"]] = float(r["ms_per_pass"])
            parts["_passes"] = int(r["passes"])
    labels = defaultdict(lambda: [0.0, 0.0])  # label -> [ms, groups]
    for r in read_tsv(os.path.join(d, "labels.tsv")):
        if not r["trace"].startswith("verify"):
            continue
        v = labels[r["label"]]
        v[0] += float(r["ms_per_pass"])
        v[1] += float(r["groups_per_pass"])
    return parts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--requests", default="d0p2,d8192p2")
    ap.add_argument("--min-ms", type=float, default=0.02)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    bn, bdir = a.base.split("=", 1)
    on, odir = a.other.split("=", 1)
    long = []
    P = print
    P(f"# The verify pass by part and product: `{bn}` against `{on}`\n")
    P("ms a verify pass (four tokens), the kept passes of the request's typical steps (verify_parts.py), the profiler's"
      " cost subtracted per span; one load per file.\n")
    for req in a.requests.split(","):
        if not (os.path.isdir(os.path.join(bdir, req)) and os.path.isdir(os.path.join(odir, req))):
            P(f"## {req}: missing in one file's folder\n")
            continue
        bp, bl = load(bdir, req)
        op, ol = load(odir, req)
        P(f"## Request {req}\n")
        P(f"Kept verify passes: `{bn}` {bp.get('_passes', 0)}, `{on}` {op.get('_passes', 0)}.\n")
        P(f"### 1. By part (R41's parts), all cards\n")
        P(f"| Part | `{bn}` | `{on}` | `{on}` − `{bn}` |")
        P("|---|---:|---:|---:|")
        for k in bp:
            if k.startswith("_"):
                continue
            b, o = bp[k], op.get(k, 0.0)
            P(f"| {k} | {b:.3f} | {o:.3f} | {o - b:+.3f} |")
            long += [("parts", req, k, bn, "", f"{b:.3f}"), ("parts", req, k, on, "", f"{o:.3f}")]
        # products
        prod = defaultdict(lambda: {bn: [0.0, 0.0, set()], on: [0.0, 0.0, set()]})
        members = defaultdict(set)
        for name, labs in ((bn, bl), (on, ol)):
            for lab, (ms, n) in labs.items():
                p, types = product_of(lab)
                e = prod[p][name]
                e[0] += ms
                e[1] += n
                e[2].update(types)
                members[p].add(lab)
        rows = sorted(prod.items(), key=lambda kv: -abs(kv[1][on][0] - kv[1][bn][0]))
        tb = sum(v[bn][0] for v in prod.values())
        to = sum(v[on][0] for v in prod.values())
        P(f"\n### 2. By product, all cards (the label without types and shapes)\n")
        P(f"| Product | `{bn}` types | `{on}` types | `{bn}` groups | `{on}` groups | `{bn}` ms | `{on}` ms | difference |")
        P("|---|---|---|---:|---:|---:|---:|---:|")
        rest_b = rest_o = 0.0
        rest_n = 0
        shown = []
        for p, v in rows:
            b, o = v[bn], v[on]
            long += [("products", req, p, bn, f"{b[1]:.2f}", f"{b[0]:.4f}"), ("products", req, p, on, f"{o[1]:.2f}", f"{o[0]:.4f}")]
            if abs(o[0] - b[0]) < a.min_ms:
                rest_b += b[0]
                rest_o += o[0]
                rest_n += 1
                continue
            shown.append(p)
            P(f"| `{p[:120]}` | {', '.join(sorted(b[2])) or '-'} | {', '.join(sorted(o[2])) or '-'} | {b[1]:.2f} | "
              f"{o[1]:.2f} | {b[0]:.3f} | {o[0]:.3f} | {o[0] - b[0]:+.3f} |")
        P(f"| the other {rest_n} products (each under {a.min_ms} ms apart) | | | | | {rest_b:.3f} | {rest_o:.3f} | {rest_o - rest_b:+.3f} |")
        P(f"| **the pass (sum of the labels)** | | | | | **{tb:.3f}** | **{to:.3f}** | **{to - tb:+.3f}** |")
        P(f"\n### 3. The labels of the products above\n")
        P(f"| Label | `{bn}` groups | `{on}` groups | `{bn}` ms | `{on}` ms |")
        P("|---|---:|---:|---:|---:|")
        for p in shown:
            for lab in sorted(members[p]):
                b = bl.get(lab, [0.0, 0.0])
                o = ol.get(lab, [0.0, 0.0])
                P(f"| `{lab[:140]}` | {b[1]:.2f} | {o[1]:.2f} | {b[0]:.3f} | {o[0]:.3f} |")
                long += [("labels", req, lab, bn, f"{b[1]:.2f}", f"{b[0]:.4f}"), ("labels", req, lab, on, f"{o[1]:.2f}", f"{o[0]:.4f}")]
        P()
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("table\trequest\tkey\tfile\tgroups_per_pass\tms\n")
            for r in long:
                f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
