#!/usr/bin/env python3
"""served_table.py: R83 (Q1300). The served step of 2026-11-11_c691_sampling_launches/run/rig_c691.sh as tables: the
switch's line in each load's server log, the replies of every load (the server's own timings, and whether every reply
carries the first one's text), the step of the warm loads with its class, and the head card's sampling chain per
verify pass from 2026-11-10_served_verify_parts/run/verify_parts.py's output on each profiled load.

  python3 -I served_table.py SERVED --parts PARTS --out DIR [--loads warm,prof] [--requests 2] [--class-load warm] > served.md

SERVED is rig_c691.sh's served/ folder: one verify_prof.sh OUT per arm load, named <k>-<arm> (1-off, 2-on, 3-on, 4-off),
each with <load>/reply_greedy-<i>.json and <load>/server-<load>.log. PARTS holds verify_parts.py's --out folder for
each, under the same name (PARTS/<k>-<arm>/accounting.tsv, by_op.tsv, last_card_groups.tsv, parts.tsv, cut.tsv).
- The switch's line: the engine's log line "one top-k over the output rows" (one per context made with the switch on).
- Steps by the server's counts, as 2026-11-10_served_verify_parts/run/replies.py: generated tokens minus accepted draft
  tokens; ms a step = the reply's generation time over that count.
- The class, per request, on --class-load's ms a step: the `on` loads against the `off` loads. The spread is the
  larger of the two `off` loads' difference and the two `on` loads' difference; gain when every `on` value is below
  every `off` value and the means differ by more than the spread, loss the other way round, negligible otherwise.
- The sampling chain: verify_parts.py's bucket "the sampling chain after the product" on the last card (every launch
  group after the output head's product), beside the head's product; the launch groups after the head by op, from
  last_card_groups.tsv (the label's first word).
Out: DIR/switch.tsv, DIR/replies.tsv, DIR/steps.tsv, DIR/chain.tsv, DIR/after_ops.tsv. Standard library only."""
import argparse, collections, hashlib, json, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("served"); ap.add_argument("--parts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--loads", default="warm,prof"); ap.add_argument("--requests", type=int, default=2)
ap.add_argument("--class-load", default="warm")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
runs = sorted((d for d in os.listdir(a.served) if re.fullmatch(r"\d+-\w+", d) and os.path.isdir(os.path.join(a.served, d))),
              key=lambda d: int(d.split("-")[0]))
arm = lambda r: r.split("-", 1)[1]
fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)


def tsv(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


# the switch's line
sw = []
for r in runs:
    for ld in a.loads.split(","):
        p = os.path.join(a.served, r, ld, f"server-{ld}.log")
        if not os.path.exists(p):
            continue
        lines = open(p, errors="replace").read().splitlines()
        eng = sum("one top-k over the output rows" in x for x in lines)
        env = any(x.startswith("===") and "LLAMA_SAMPLING_TOP_K_ROWS=1" in x for x in lines)
        sw.append((r, ld, eng, "yes" if env else "no"))
with open(os.path.join(a.out, "switch.tsv"), "w") as w:
    w.write("run\tload\tengine_lines\tswitch_in_start_line\n")
    for x in sw:
        w.write("\t".join(map(str, x)) + "\n")

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
                             tokens_a_step=pn / steps if steps else float("nan"), predicted_ms=float(t.get("predicted_ms", 0.0)),
                             ms_a_step_server=t.get("predicted_ms", 0.0) / steps if steps else float("nan"),
                             tok_s=float(t.get("predicted_per_second", 0.0))))
with open(os.path.join(a.out, "replies.tsv"), "w") as w:
    w.write("\t".join(cols) + "\n")
    for x in reps:
        w.write("\t".join(fmt(x[c]) for c in cols) + "\n")

# the step's class
steps = []
for i in range(1, a.requests + 1):
    v = {x["run"]: x["ms_a_step_server"] for x in reps if x["load"] == a.class_load and x["request"] == f"greedy-{i}"}
    off = [v[r] for r in runs if arm(r) == "off" and r in v]
    on = [v[r] for r in runs if arm(r) == "on" and r in v]
    if not off or not on:
        continue
    spread = max(max(off) - min(off), max(on) - min(on))
    d = statistics.mean(on) - statistics.mean(off)
    cls = ("gain" if max(on) < min(off) and -d > spread else
           "loss" if min(on) > max(off) and d > spread else "negligible")
    steps.append((f"greedy-{i}", off, on, d, spread, cls))
with open(os.path.join(a.out, "steps.tsv"), "w") as w:
    w.write("request\toff_ms_a_step\ton_ms_a_step\tmean_on_minus_off\tspread\tclass\n")
    for q, off, on, d, s, c in steps:
        w.write(f"{q}\t{', '.join(f'{x:.2f}' for x in off)}\t{', '.join(f'{x:.2f}' for x in on)}\t{d:+.2f}\t{s:.2f}\t{c}\n")

# the sampling chain
chain, after = [], collections.OrderedDict()
for r in runs:
    d = os.path.join(a.parts, r)
    if not os.path.isdir(d):
        continue
    acc = tsv(os.path.join(d, "accounting.tsv"))
    last = next((x["cards"] for x in acc if x["cards"].startswith("card ")), None)
    traces = list(dict.fromkeys(x["trace"] for x in acc if x["trace"].startswith("verify")))
    ops = tsv(os.path.join(d, "by_op.tsv"))
    parts = tsv(os.path.join(d, "parts.tsv"))
    cut = {x["request"]: x for x in tsv(os.path.join(d, "cut.tsv"))}
    lcg = tsv(os.path.join(d, "last_card_groups.tsv"))
    for tr in traces:
        g = lambda where: float(next((x["ms_per_pass"] for x in acc if x["trace"] == tr and x["cards"] == last and x["where"] == where), "nan"))
        op = lambda name, col: float(next((x[col] for x in ops if x["trace"] == tr and x["cards"] == last and x["first_op"] == name), 0.0))
        passes = next((x["passes"] for x in acc if x["trace"] == tr and x["cards"] == last), "")
        tot = float(next((x["ms_per_pass"] for x in parts if x["trace"] == tr and x["cards"] == "all cards"
                          and x["part"] == "sum of the parts"), "nan"))
        aft = [x for x in lcg if x["trace"] == tr and x["position"] == "after"]
        for x in aft:
            k = (r, tr, x["label"].split(" ")[0])
            n, ms = after.get(k, (0, 0.0))
            after[k] = (n + 1, ms + float(x["ms"]))
        req = "r" + tr.split("-r")[-1] if "-r" in tr else ""
        chain.append(dict(run=r, trace=tr, card=last, passes=passes, head_ms=g("the head's product"),
                          chain_ms=g("the sampling chain after the product"), groups_after=len(aft),
                          top_k_after=sum(1 for x in aft if x["label"].startswith("TOP_K")),
                          pad_after=sum(1 for x in aft if x["label"].startswith("PAD")),
                          top_k_card_groups=op("TOP_K", "groups_per_pass"), top_k_card_ms=op("TOP_K", "ms_per_pass"),
                          pad_card_groups=op("PAD", "groups_per_pass"), pad_card_ms=op("PAD", "ms_per_pass"),
                          sum_of_parts=tot, median_raw_ms=cut.get(req, {}).get("median_raw_ms_typical_verify", "")))
ccols = ["run", "trace", "card", "passes", "head_ms", "chain_ms", "groups_after", "top_k_after", "pad_after",
         "top_k_card_groups", "top_k_card_ms", "pad_card_groups", "pad_card_ms", "sum_of_parts", "median_raw_ms"]
f3 = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
with open(os.path.join(a.out, "chain.tsv"), "w") as w:
    w.write("\t".join(ccols) + "\n")
    for x in chain:
        w.write("\t".join(f3(x[c]) for c in ccols) + "\n")
with open(os.path.join(a.out, "after_ops.tsv"), "w") as w:
    w.write("run\ttrace\top\tgroups\tms_per_pass\n")
    for (r, tr, o), (n, ms) in after.items():
        w.write(f"{r}\t{tr}\t{o}\t{n}\t{ms:.4f}\n")

# the report
print("## The switch's line in the server logs\n")
print("| Run | Load | The engine's line | The switch in rig_server.sh's start line |")
print("|---|---|---:|---|")
for x in sw:
    print("| " + " | ".join(map(str, x)) + " |")
print("\n## The replies\n")
print("| " + " | ".join(cols) + " |")
print("|---|---|---|---|---" + "|---:" * (len(cols) - 5) + "|")
for x in reps:
    print("| " + " | ".join(fmt(x[c]) for c in cols) + " |")
print(f"\n## The step ({a.class_load} loads), ms a step\n")
print("| Request | `off` | `on` | Mean `on` − `off` | Spread | Class |")
print("|---|---|---|---:|---:|---|")
for q, off, on, d, s, c in steps:
    print(f"| {q} | {', '.join(f'{x:.2f}' for x in off)} | {', '.join(f'{x:.2f}' for x in on)} | {d:+.2f} | {s:.2f} | {c} |")
print("\n## The head card's sampling chain per verify pass, ms\n")
print("| " + " | ".join(ccols) + " |")
print("|---|---|---" + "|---:" * (len(ccols) - 3) + "|")
for x in chain:
    print("| " + " | ".join(f3(x[c]) for c in ccols) + " |")
print("\n## The launch groups after the head's product, by op (groups a pass, ms a pass)\n")
names = list(dict.fromkeys(o for (_, _, o) in after))
keys = list(dict.fromkeys((r, tr) for (r, tr, _) in after))
print("| Op | " + " | ".join(f"{r} {tr}" for r, tr in keys) + " |")
print("|---" + "|---:" * len(keys) + "|")
for o in names:
    print(f"| {o} | " + " | ".join((lambda v: f"{v[0]}: {v[1]:.4f}" if v else "")(after.get((r, tr, o))) for r, tr in keys) + " |")
