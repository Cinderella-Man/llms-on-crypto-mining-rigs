#!/usr/bin/env python3
"""ladder.py: R122's ladder in one table (Q1400), from the tables tables.sh writes with S37's arms.py.

  python3 -I ladder.py TABLES --rungs base,a,b,c,d,e,f [--tsv ladder.tsv] > ladder.md

TABLES holds rung_<r>_class.tsv and rung_<r>_paired.tsv (arms.py: rung r's two loads against the rung before it),
stack_class.tsv and stack_paired.tsv (the last rung against the first), arms_all.tsv (every load's server timings),
vm.tsv and arms/<load>/reply_{greedy,long}.json.
Each load is classed by its own value (arms.py: the mean of the two requests' median typical steps) against the two
loads of the rung before: gain when below both and the reference mean minus the value is more than their spread, loss
the mirror, negligible otherwise. The rung's own line classes the mean of its two loads the same way.
The greedy request (temperature 0, top-k 1, seed 1) is compared with base's: the text (reasoning and content) and the
drafted, accepted and generated counts. The long request samples at temperature 1.0 and is not compared.
A third table puts each reply's prompt and generation time (the server's timings) next to the sum of its profiled
generation steps, so that time the server spends outside the steps shows.
"""
import argparse
import csv
import json
import os

SWITCH = {
    "a": "LLAMA_DRAFT_PREFETCH=1 (C615)",
    "b": "LLAMA_VERIFY_PREPARE=2 (C738)",
    "c": "LLAMA_MTP_GRAPH_KEEP=1 (C619)",
    "d": "LLAMA_KV_SEQ_RM_BOUND=1 (C740)",
    "e": "LLAMA_SERVER_YIELD=2 (C741)",
    "f": "LLAMA_MTP_EMBD_FIRST=1, LLAMA_MTP_DRAFT_CAP=1, LLAMA_MTP_SKIP_LAST_ROW=1 (C771, C744, C742)",
}


def klass(v, a, b):
    lo, hi, mean, spread = min(a, b), max(a, b), (a + b) / 2.0, abs(a - b)
    if v < lo and mean - v > spread:
        return "gain"
    if v > hi and v - mean > spread:
        return "loss"
    return "negligible"


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def reply(d, load, req):
    try:
        with open(os.path.join(d, "arms", load, f"reply_{req}.json")) as f:
            r = json.load(f)
        m, t = r["choices"][0]["message"], r.get("timings", {})
        return dict(text=(m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""),
                    finish=r["choices"][0].get("finish_reason"),
                    **{k: t.get(k) for k in ["predicted_n", "draft_n", "draft_n_accepted"]})
    except (OSError, ValueError, KeyError, IndexError):
        return None


def timings(d, load, req):
    try:
        with open(os.path.join(d, "arms", load, f"reply_{req}.json")) as f:
            return json.load(f).get("timings", {}) or {}
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rungs", default="base,a,b,c,d,e,f")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    d, rungs = a.dir, a.rungs.split(",")
    loads = rungs + [f"{r}-2" for r in rungs]
    vm = {r["load"]: r for r in rows(os.path.join(d, "vm.tsv"))}
    allrows = {(r["load"], r["request"]): r for r in rows(os.path.join(d, "arms_all.tsv"))}
    g0 = reply(d, rungs[0], "greedy")

    def same(load):
        g = reply(d, load, "greedy")
        if not g or not g0:
            return "no reply"
        if g["text"] == g0["text"] and all(g[k] == g0[k] for k in ["predicted_n", "draft_n", "draft_n_accepted"]):
            return "yes"
        return "NO" if g["text"] != g0["text"] else "same text, other counts"

    out = []
    print("## Each rung against the rung before it")
    print()
    print("Step, ms: the mean of the greedy and the long request's median typical step (S37's `arms.py`). Reference: the "
          "two loads of the rung before; spread = their difference. The rung's line: the mean of its two loads. Paired: "
          "the greedy request's steps against the rung before's first load, step by step (mean difference and its "
          "standard error), where the replies are the same.")
    print()
    print("| rung | switch added | load | step, ms | greedy, ms | long, ms | reference loads, ms | minus their mean | "
          "spread | class | greedy paired, ms | greedy reply = base's | lowest MemAvailable, MiB |")
    print("|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---|---:|")
    pairs = [(r, rungs[i - 1], f"rung_{r}") for i, r in enumerate(rungs) if i > 0]
    pairs.append(("stack", rungs[0], "stack"))
    for name, ref, stem in pairs:
        cls = rows(os.path.join(d, f"{stem}_class.tsv"))
        paired = {r["load"]: r for r in rows(os.path.join(d, f"{stem}_paired.tsv"))}
        cur = rungs[-1] if name == "stack" else name
        mine = [cur, f"{cur}-2"]
        vals, refab = {}, None
        for r in cls:
            vals[(r["load"], r["value"])] = float(r["step_ms"])
            if r["value"] == "mean of the medians":
                refab = (float(r["ref_a_ms"]), float(r["ref_b_ms"]))
        label = "the stack (" + "+".join(rungs[1:]) + ") against " + ref if name == "stack" else SWITCH.get(name, "")
        for ld in mine:
            k = (ld, "mean of the medians")
            if k not in vals or refab is None:
                print(f"| {name} | {label} | {ld} | no value | | | | | | | | {same(ld)} | {vm.get(ld, {}).get('min_mem_available_mib', '')} |")
                continue
            v, (x, y) = vals[k], refab
            c = klass(v, x, y)
            gr, lo = vals.get((ld, "greedy, median")), vals.get((ld, "long, median"))
            p = paired.get(ld)
            pc = f"{float(p['step_mean_diff_ms']):+.2f} ± {float(p['standard_error_ms']):.2f} ({p['steps_paired']})" if p else "—"
            print(f"| {name} | {label} | {ld} | {v:.2f} | {gr:.2f} | {lo:.2f} | {ref} {x:.2f}, {ref}-2 {y:.2f} | "
                  f"{v - (x + y) / 2:+.2f} | {abs(x - y):.2f} | {c} | {pc} | {same(ld)} | "
                  f"{vm.get(ld, {}).get('min_mem_available_mib', '')} |")
            out.append(dict(rung=name, load=ld, step_ms=v, greedy_ms=gr, long_ms=lo, ref_a_ms=x, ref_b_ms=y,
                            minus_ref_mean_ms=v - (x + y) / 2, spread_ms=abs(x - y), cls=c,
                            paired_mean_ms=float(p["step_mean_diff_ms"]) if p else "",
                            paired_se_ms=float(p["standard_error_ms"]) if p else "", greedy_same=same(ld),
                            min_mem_available_mib=vm.get(ld, {}).get("min_mem_available_mib", "")))
        if all((ld, "mean of the medians") in vals for ld in mine) and refab:
            m = sum(vals[(ld, "mean of the medians")] for ld in mine) / 2
            x, y = refab
            c = klass(m, x, y)
            print(f"| **{name}** | | **both** | **{m:.2f}** | | | {(x + y) / 2:.2f} (mean) | **{m - (x + y) / 2:+.2f}** | "
                  f"{abs(x - y):.2f} | **{c}** | | | |")
            out.append(dict(rung=name, load="both", step_ms=m, greedy_ms="", long_ms="", ref_a_ms=x, ref_b_ms=y,
                            minus_ref_mean_ms=m - (x + y) / 2, spread_ms=abs(x - y), cls=c, paired_mean_ms="",
                            paired_se_ms="", greedy_same="", min_mem_available_mib=""))
    print()

    print("## Every load: the server's counts and the host")
    print()
    print("| load | greedy: generated, drafted, accepted | greedy: tokens a step | greedy: ms a step (server) | "
          "long: generated | long: tokens a step | long: ms a step (server) | greedy reply = base's | "
          "lowest MemAvailable, MiB | swap at start / most, MiB | major faults | pswpin | pswpout |")
    print("|---|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|")
    for ld in loads:
        g, lg = allrows.get((ld, "greedy"), {}), allrows.get((ld, "long"), {})
        v = vm.get(ld, {})
        print(f"| {ld} | {g.get('predicted_n', '')}, {g.get('draft_n', '')}, {g.get('draft_n_accepted', '')} | "
              f"{g.get('tokens_a_step', '')} | {g.get('ms_a_step_server', '')} | {lg.get('predicted_n', '')} | "
              f"{lg.get('tokens_a_step', '')} | {lg.get('ms_a_step_server', '')} | {same(ld)} | "
              f"{v.get('min_mem_available_mib', '')} | {v.get('swap_start_mib', '')} / {v.get('swap_max_mib', '')} | "
              f"{v.get('major_faults', '')} | {v.get('pswpin', '')} | {v.get('pswpout', '')} |")
    print()

    print("## The server's time outside the profiled steps")
    print()
    print("Prompt and generation: the server's `prompt_ms` and `predicted_ms` for the reply. Step 0: the request's first "
          "generation step in the host profile, shown apart because the long request's step 0 is not an ordinary step "
          "(it lasts about 7 s, and the server books part of it as prompt time). The rest: the sum of the request's "
          "other generation steps (typical or not) and their count. Outside: generation minus the rest, the time the "
          "server counts as generation that the steps after the first do not cover (it holds step 0 too where that is "
          "an ordinary step, as in the greedy request). The greedy request is the first after the server starts.")
    print()
    print("| load | greedy: prompt, ms | greedy: generation, ms | greedy: step 0, ms | greedy: the rest, ms (steps) | "
          "greedy: outside, ms | long: prompt, ms | long: generation, ms | long: step 0, ms | "
          "long: the rest, ms (steps) | long: outside, ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ld in loads:
        st = rows(os.path.join(d, f"steps_{ld}.tsv"))
        cells = [ld]
        for req in ["greedy", "long"]:
            t = timings(d, ld, req)
            xs = [(int(r["i"]), float(r["step_ms"])) for r in st if r["request"] == req]
            if t.get("predicted_ms") is None or not xs:
                cells += ["", "", "", "", ""]
                continue
            s0 = sum(v for i, v in xs if i == 0)
            rest = [v for i, v in xs if i != 0]
            cells += [f"{t.get('prompt_ms', 0):.1f}", f"{t['predicted_ms']:.1f}", f"{s0:.1f}",
                      f"{sum(rest):.1f} ({len(rest)})", f"{t['predicted_ms'] - sum(rest):.1f}"]
        print("| " + " | ".join(cells) + " |")
    print()

    if a.tsv:
        cols = ["rung", "load", "step_ms", "greedy_ms", "long_ms", "ref_a_ms", "ref_b_ms", "minus_ref_mean_ms",
                "spread_ms", "cls", "paired_mean_ms", "paired_se_ms", "greedy_same", "min_mem_available_mib"]
        with open(a.tsv, "w") as f:
            f.write("\t".join(c if c != "cls" else "class" for c in cols) + "\n")
            for r in out:
                f.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
