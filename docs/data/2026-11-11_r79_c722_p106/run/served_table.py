#!/usr/bin/env python3
"""served_table.py: R79 (Q1277). The served step of 2026-11-11_c722_short_batch_products/run/rig_c722.sh as tables: the
replies of every load (the server's own timings, and whether every reply carries the first one's text), and the
weight products' GPU time per verify pass by type and by name, from 2026-11-10_served_verify_parts/run/verify_parts.py's
output on each profiled load.

  python3 -I served_table.py SERVED --parts PARTS --out DIR [--loads warm,prof] [--requests 2] > served.md

SERVED is rig_c722.sh's served/ folder: one verify_prof.sh OUT per arm load, named <k>-<arm> (1-off, 2-on, 3-on, 4-off),
each with <load>/reply_greedy-<i>.json. PARTS holds verify_parts.py's --out folder for each, under the same name
(PARTS/<k>-<arm>/labels.tsv and parts.tsv).
Steps by the server's counts, as 2026-11-10_served_verify_parts/run/replies.py: generated tokens minus accepted draft
tokens; ms a step = the reply's generation time over that count.
The weight products: labels.tsv's launch groups whose label is one MUL_MAT of a named weight (`MUL_MAT <name>:<type>:KxM`),
summed over the card groups (cards 0 to 7 and card 8). "The attention and recurrent projections" are attn_qkv,
attn_gate, attn_q, ssm_out and attn_output; "the switch's types" are IQ4_XS and Q4_K, the products C722's kernel takes
(whatever their name). parts.tsv's "all cards" rows give the pass's parts and its sum.

Out: DIR/replies.tsv; DIR/products.tsv (per arm load and trace: ms a pass of the one-op weight products by type, the
switch's types, the attention and recurrent projections, the sum of the parts); DIR/by_name.tsv (per arm load, trace,
name and type of the switch's types: ms a pass and launch groups a pass); DIR/parts.tsv (per arm load and trace: the
all-cards parts). Standard library only."""
import argparse, collections, hashlib, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("served"); ap.add_argument("--parts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--loads", default="warm,prof"); ap.add_argument("--requests", type=int, default=2)
# the switch's types (ggml type names): C722's by default; C726's rig run passes q3_K
ap.add_argument("--types", default="iq4_xs,q4_K")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = sorted((d for d in os.listdir(a.served) if re.fullmatch(r"\d+-\w+", d) and os.path.isdir(os.path.join(a.served, d))),
              key=lambda d: int(d.split("-")[0]))

# the replies
cols = ["run", "load", "request", "text_md5", "same_text_as_first", "prompt_n", "predicted_n", "draft_n",
        "draft_n_accepted", "steps_server", "tokens_a_step", "predicted_ms", "ms_a_step_server", "tok_s"]
reps, first = [], None
for r in runs:
    for ld in a.loads.split(","):
        for i in range(1, a.requests + 1):
            p = os.path.join(a.served, r, ld, f"reply_greedy-{i}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            msg = j["choices"][0]["message"]  # the reasoning and the answer (--reasoning-format deepseek splits them)
            text = (msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "")
            t = j.get("timings", {})
            pn, dn, da = t.get("predicted_n", 0), t.get("draft_n", 0), t.get("draft_n_accepted", 0)
            h = hashlib.md5(text.encode()).hexdigest()
            first = first or h
            steps = pn - da
            reps.append(dict(run=r, load=ld, request=f"greedy-{i}", text_md5=h[:12], same_text_as_first="yes" if h == first else "NO",
                             prompt_n=t.get("prompt_n", 0), predicted_n=pn, draft_n=dn, draft_n_accepted=da, steps_server=steps,
                             tokens_a_step=pn / steps if steps else float("nan"), predicted_ms=t.get("predicted_ms", 0.0),
                             ms_a_step_server=t.get("predicted_ms", 0.0) / steps if steps else float("nan"),
                             tok_s=t.get("predicted_per_second", 0.0)))
fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
with open(os.path.join(a.out, "replies.tsv"), "w") as w:
    w.write("\t".join(cols) + "\n")
    for x in reps:
        w.write("\t".join(fmt(x[c]) for c in cols) + "\n")

# the weight products and the parts
WMM = re.compile(r"^MUL_MAT ([A-Za-z0-9_.]+):([A-Za-z0-9_]+):\d+x\d+$")
prod = collections.defaultdict(float)  # (run, trace, name, type) -> ms a pass
groups = collections.defaultdict(float)  # (run, trace, name, type) -> launch groups a pass
parts = {}  # (run, trace, part) -> ms a pass, all cards
traces = collections.OrderedDict()
for r in runs:
    lp = os.path.join(a.parts, r, "labels.tsv")
    if not os.path.exists(lp):
        continue
    for line in open(lp):
        f = line.rstrip("\n").split("\t")
        if f[0] == "trace" or len(f) < 5:
            continue
        traces[(r, f[0])] = 1
        m = WMM.match(f[2])
        if m:
            prod[(r, f[0], m.group(1), m.group(2))] += float(f[4])
            groups[(r, f[0], m.group(1), m.group(2))] += float(f[3])
    for line in open(os.path.join(a.parts, r, "parts.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) == 5 and f[1] == "all cards":
            parts[(r, f[0], f[3])] = float(f[4])
            parts[(r, f[0], "passes")] = int(f[2])
types = sorted({t for (_, _, _, t) in prod})
ATT = ("attn_qkv", "attn_gate", "attn_q", "ssm_out", "attn_output")
SW = tuple(a.types.split(","))
first_run = next(iter(traces))[0] if traces else ""
names = sorted({n for (r, _, n, t) in prod if t in SW}, key=lambda n: -sum(prod[(first_run, tr, n, t)] for (rr, tr) in traces
                                                                            if rr == first_run for t in SW))


def row(r, tr):
    by_t = {t: sum(v for (rr, tt, _, ty), v in prod.items() if rr == r and tt == tr and ty == t) for t in types}
    sw = sum(by_t.get(t, 0.0) for t in SW)
    att = sum(v for (rr, tt, n, _), v in prod.items() if rr == r and tt == tr and n in ATT)
    return by_t, sw, att


with open(os.path.join(a.out, "products.tsv"), "w") as w:
    w.write("run\ttrace\tpasses\t" + "\t".join(types) + "\tswitch_types\tattention_and_recurrent_projections\tsum_of_parts\n")
    for (r, tr) in traces:
        by_t, sw, att = row(r, tr)
        w.write(f"{r}\t{tr}\t{parts.get((r, tr, 'passes'), '')}\t" + "\t".join(f"{by_t[t]:.3f}" for t in types)
                + f"\t{sw:.3f}\t{att:.3f}\t{parts.get((r, tr, 'sum of the parts'), float('nan')):.3f}\n")
with open(os.path.join(a.out, "by_name.tsv"), "w") as w:
    w.write("run\ttrace\tname\ttype\tgroups_per_pass\tms_per_pass\n")
    for (r, tr) in traces:
        for n in names:
            for t in SW:
                if (r, tr, n, t) in prod:
                    w.write(f"{r}\t{tr}\t{n}\t{t}\t{groups[(r, tr, n, t)]:.2f}\t{prod[(r, tr, n, t)]:.3f}\n")
pnames = list(dict.fromkeys(p for (_, _, p) in parts if p != "passes"))
with open(os.path.join(a.out, "parts.tsv"), "w") as w:
    w.write("run\ttrace\tpart\tms_per_pass\n")
    for (r, tr) in traces:
        for p in pnames:
            if (r, tr, p) in parts:
                w.write(f"{r}\t{tr}\t{p}\t{parts[(r, tr, p)]:.3f}\n")

# the report
print("## The replies\n")
print("| " + " | ".join(cols) + " |")
print("|---|---|---|---|---" + "|---:" * (len(cols) - 5) + "|")
for x in reps:
    print("| " + " | ".join(fmt(x[c]) for c in cols) + " |")
print("\n## The one-op weight products per verify pass, all cards, ms\n")
print("| run | trace | passes | " + " | ".join(types) + f" | {' + '.join(t.upper() for t in SW)} | attention and recurrent projections | sum of the parts |")
print("|---|---|---:" + "|---:" * (len(types) + 3) + "|")
for (r, tr) in traces:
    by_t, sw, att = row(r, tr)
    print(f"| {r} | {tr} | {parts.get((r, tr, 'passes'), '')} | " + " | ".join(f"{by_t[t]:.3f}" for t in types)
          + f" | {sw:.3f} | {att:.3f} | {parts.get((r, tr, 'sum of the parts'), float('nan')):.3f} |")
print("\n## The switch's products by name, ms a pass\n")
cols2 = [(n, t) for n in names for t in SW if any((r, tr, n, t) in prod for (r, tr) in traces)]
print("| run | trace | " + " | ".join(f"{n}:{t}" for n, t in cols2) + " |")
print("|---|---" + "|---:" * len(cols2) + "|")
for (r, tr) in traces:
    print(f"| {r} | {tr} | " + " | ".join(f"{prod[(r, tr, n, t)]:.3f}" for n, t in cols2) + " |")
print("\n## The pass's parts, all cards, ms\n")
print("| part | " + " | ".join(f"{r} {tr}" for (r, tr) in traces) + " |")
print("|---" + "|---:" * len(traces) + "|")
for p in pnames:
    print(f"| {p} | " + " | ".join(f"{parts.get((r, tr, p), float('nan')):.3f}" for (r, tr) in traces) + " |")
