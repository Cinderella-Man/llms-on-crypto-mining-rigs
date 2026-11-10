#!/usr/bin/env python3
"""arms.py: the loads of S37 side by side (Q961): the server's own timings per reply, the step and its parts per load
from step_split.py's TSVs, each switch arm against the reference loads with its class, and the greedy request's
steps paired with the reference's.

  python3 -I arms.py DIR --loads warm,d1off,d1cache,d1carry,d1on,d1off-2 --ref d1off,d1off-2 \
      [--requests long,greedy,smoke] [--tsv arms.tsv] > arms.md

DIR/<load>/ holds reply_<request>.json (llama-server's replies) and, for profiled loads, steps.tsv (step_split.py
--tsv). A load's name up to its "-" is its arm.
The class of an arm's value v against the two reference loads a and b (lower is faster): gain when v is below both
and their mean minus v is more than |a - b|; loss when v is above both and v minus their mean is more than |a - b|;
negligible otherwise. A load's one value is the mean of its requests' median typical steps, so that a longer reply
does not weigh more.
The greedy request has no random draw, so loads that reply with the same text and draft counts ran the same steps:
step i of one load is paired with step i of the other, and the mean difference has a standard error.
"""
import argparse
import csv
import json
import math
import os
import statistics

PARTS = [("step_ms", "step"), ("verify_span_ms", "verify_span"), ("verify_gpu_ms", "verify_gpu"),
         ("verify_rest_ms", "verify_rest"), ("draft_span_ms", "the draft's span"), ("draft_gpu_ms", "the draft's GPU time"),
         ("lead_sum_ms", "the leads"), ("turn_mid_ms", "the turns before the last"), ("turn_last_ms", "the last turn")]


def read_steps(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n = int(r["n_dp"])
            x = dict(request=r["request"], i=int(r["i"]), typical=int(r["typical"]), n_dp=n)
            for k in ["step_ms", "verify_span_ms", "verify_gpu_ms", "verify_rest_ms"]:
                x[k] = float(r[k])
            x["draft_span_ms"] = x["step_ms"] - x["verify_span_ms"]
            x["draft_gpu_ms"] = sum(float(r[f"gpu{k}_ms"]) for k in range(n))
            x["lead_sum_ms"] = sum(float(r[f"lead{k}_ms"]) for k in range(n))
            x["turn_mid_ms"] = sum(float(r[f"turn{k}_ms"]) for k in range(n - 1))
            x["turn_last_ms"] = float(r[f"turn{n - 1}_ms"]) if n else 0.0
            out.append(x)
    return out


def read_reply(path):
    try:
        with open(path) as f:
            r = json.load(f)
        m = r["choices"][0]["message"]
        t = r.get("timings", {})
        return dict(text=(m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), t=t,
                    finish=r["choices"][0].get("finish_reason"))
    except (OSError, ValueError, KeyError, IndexError):
        return None


def klass(v, a, b):
    lo, hi, mean, spread = min(a, b), max(a, b), (a + b) / 2.0, abs(a - b)
    if v < lo and mean - v > spread:
        return "gain"
    if v > hi and v - mean > spread:
        return "loss"
    return "negligible"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--loads", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--requests", default="long,greedy,smoke")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    loads, ref, reqs = a.loads.split(","), a.ref.split(","), a.requests.split(",")
    steps, replies = {}, {}
    for ld in loads:
        p = os.path.join(a.dir, ld, "steps.tsv")
        if os.path.exists(p):
            steps[ld] = read_steps(p)
        replies[ld] = {r: read_reply(os.path.join(a.dir, ld, f"reply_{r}.json")) for r in reqs}

    print("## The server's own timings per reply")
    print()
    print("Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one "
          "sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.")
    print()
    print("| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | "
          "tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for ld in loads:
        for r in reqs:
            rp = replies[ld][r]
            if not rp or not rp["t"]:
                print(f"| {ld} | {r} | no reply | | | | | | | | | | | |")
                continue
            t = rp["t"]
            pn, dn, da = t.get("predicted_n", 0), t.get("draft_n", 0), t.get("draft_n_accepted", 0)
            n_steps = pn - da
            prof = sum(1 for x in steps.get(ld, []) if x["request"] == r)
            row = dict(load=ld, request=r, prompt_n=t.get("prompt_n", 0), predicted_n=pn, draft_n=dn, draft_n_accepted=da,
                       accepted_share=da / dn if dn else float("nan"), steps_server=n_steps,
                       tokens_a_step=pn / n_steps if n_steps else float("nan"), predicted_ms=t.get("predicted_ms", 0.0),
                       ms_a_step_server=t.get("predicted_ms", 0.0) / n_steps if n_steps else float("nan"),
                       tok_s=t.get("predicted_per_second", 0.0), prompt_ms=t.get("prompt_ms", 0.0), steps_profile=prof)
            rows.append(row)
            print(f"| {ld} | {r} | {row['prompt_n']} | {pn} | {dn} | {da} | {row['accepted_share']:.3f} | {n_steps} | "
                  f"{row['tokens_a_step']:.3f} | {row['predicted_ms']:.1f} | {row['ms_a_step_server']:.2f} | "
                  f"{row['tok_s']:.2f} | {row['prompt_ms']:.1f} | {prof if ld in steps else '—'} |")
    print()

    print("## The step per load, from the profile (typical steps)")
    print()
    val = {}
    for key, label in PARTS:
        print(f"**{label}, ms: median (mean) per request, and the mean of the three medians**")
        print()
        print("| load | " + " | ".join(reqs) + " | mean of the medians |")
        print("|---|" + "---:|" * (len(reqs) + 1))
        for ld in loads:
            if ld not in steps:
                continue
            meds, cells = [], []
            for r in reqs:
                xs = [x[key] for x in steps[ld] if x["request"] == r and x["typical"]]
                if xs:
                    meds.append(statistics.median(xs))
                    cells.append(f"{statistics.median(xs):.2f} ({statistics.fmean(xs):.2f}; {len(xs)})")
                else:
                    cells.append("—")
            v = statistics.fmean(meds) if len(meds) == len(reqs) else float("nan")
            val[(ld, key)] = v
            for r, m in zip(reqs, meds):
                val[(ld, key, r)] = m
            print(f"| {ld} | " + " | ".join(cells) + f" | {v:.2f} |")
        print()

    print("## Each arm against the reference loads")
    print()
    ra, rb = ref
    print(f"Reference: {ra} and {rb}. Difference = the load's value minus the mean of the two; spread = their difference.")
    print()
    print("| load | value | step, ms | minus the reference | reference spread | class |")
    print("|---|---|---:|---:|---:|---|")
    arm_rows = []
    for ld in loads:
        if ld not in steps or ld in ref:
            continue
        for sel in [None] + reqs:
            k = lambda l: (l, "step_ms") if sel is None else (l, "step_ms", sel)
            if k(ld) not in val or k(ra) not in val or k(rb) not in val:
                continue
            v, x, y = val[k(ld)], val[k(ra)], val[k(rb)]
            c = klass(v, x, y)
            name = "mean of the medians" if sel is None else f"{sel}, median"
            arm_rows.append(dict(load=ld, value=name, step_ms=v, ref_a=x, ref_b=y, diff=v - (x + y) / 2.0, cls=c))
            print(f"| {ld} | {name} | {v:.2f} | {v - (x + y) / 2.0:+.2f} | {abs(x - y):.2f} ({x:.2f}, {y:.2f}) | {c} |")
    print()

    print("## The greedy request, step by step against " + ra)
    print()
    base = replies[ra].get("greedy")
    print("| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | "
          "median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    pair_rows = []
    for ld in loads:
        rp = replies[ld].get("greedy")
        if not rp or not base:
            continue
        same = rp["text"] == base["text"] and all(rp["t"].get(k) == base["t"].get(k) for k in
                                                   ["draft_n", "draft_n_accepted", "predicted_n"])
        cells = [ld, "yes" if same else "NO", str(rp["t"].get("draft_n")), str(rp["t"].get("draft_n_accepted"))]
        if ld in steps and ld != ra and same:
            A = {x["i"]: x for x in steps[ra] if x["request"] == "greedy"}
            B = {x["i"]: x for x in steps[ld] if x["request"] == "greedy"}
            ids = [i for i in sorted(A) if i in B and A[i]["typical"] and B[i]["typical"] and A[i]["n_dp"] == B[i]["n_dp"]]
            if len(ids) > 2 and len(A) == len(B):
                d = {key: [B[i][key] - A[i][key] for i in ids] for key, _ in PARTS}
                se = statistics.stdev(d["step_ms"]) / math.sqrt(len(ids))
                cells += [str(len(ids)), f"{statistics.fmean(d['step_ms']):+.2f}", f"{se:.2f}",
                          f"{statistics.median(d['step_ms']):+.2f}"]
                cells += [f"{statistics.fmean(d[k]):+.2f}" for k in ["draft_span_ms", "verify_span_ms", "lead_sum_ms",
                                                                     "turn_mid_ms", "turn_last_ms"]]
                pair_rows.append(dict(load=ld, n=len(ids), mean=statistics.fmean(d["step_ms"]), se=se,
                                      median=statistics.median(d["step_ms"]),
                                      **{k: statistics.fmean(d[k]) for k, _ in PARTS if k != "step_ms"}))
            else:
                cells += [f"not paired ({len(A)} and {len(B)} steps)"] + [""] * 8
        else:
            cells += ["—"] + [""] * 8
        print("| " + " | ".join(cells) + " |")
    print()

    if a.tsv:
        with open(a.tsv, "w") as f:
            cols = ["load", "request", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted", "accepted_share",
                    "steps_server", "tokens_a_step", "predicted_ms", "ms_a_step_server", "tok_s", "prompt_ms", "steps_profile",
                    "typical_steps"] + [f"{k}_median" for k, _ in PARTS] + [f"{k}_mean" for k, _ in PARTS]
            f.write("\t".join(cols) + "\n")
            for row in rows:
                xs = [x for x in steps.get(row["load"], []) if x["request"] == row["request"] and x["typical"]]
                v = [row[c] if not isinstance(row[c], float) else f"{row[c]:.3f}" for c in cols[:14]] + [len(xs)]
                v += [f"{statistics.median(x[k] for x in xs):.3f}" if xs else "" for k, _ in PARTS]
                v += [f"{statistics.fmean(x[k] for x in xs):.3f}" if xs else "" for k, _ in PARTS]
                f.write("\t".join(str(c) for c in v) + "\n")
        with open(os.path.splitext(a.tsv)[0] + "_class.tsv", "w") as f:
            f.write("load\tvalue\tstep_ms\tref_a_ms\tref_b_ms\tminus_ref_mean_ms\tclass\n")
            for r in arm_rows:
                f.write(f"{r['load']}\t{r['value']}\t{r['step_ms']:.3f}\t{r['ref_a']:.3f}\t{r['ref_b']:.3f}\t{r['diff']:.3f}\t{r['cls']}\n")
        with open(os.path.splitext(a.tsv)[0] + "_paired.tsv", "w") as f:
            keys = [k for k, _ in PARTS if k != "step_ms"]
            f.write("load\tsteps_paired\tstep_mean_diff_ms\tstandard_error_ms\tstep_median_diff_ms\t"
                    + "\t".join(k.replace("_ms", "_mean_diff_ms") for k in keys) + "\n")
            for r in pair_rows:
                f.write(f"{r['load']}\t{r['n']}\t{r['mean']:.3f}\t{r['se']:.3f}\t{r['median']:.3f}\t"
                        + "\t".join(f"{r[k]:.3f}" for k in keys) + "\n")


if __name__ == "__main__":
    main()
