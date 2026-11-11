#!/usr/bin/env python3
"""R81 (Q1295): the readouts of the ubatch trace for one server load, from trace_table.py's tables per request.

Usage: python3 -I r81_read.py <tables dir> <out dir> [--prompt-req N]

<tables dir> holds requests.tsv (split_requests.py) and one folder req<NN>/ per request with trace_table.py's
calls.tsv, ubatch.tsv, splits.tsv and replan.tsv. Writes to <out dir>:
  prompt_calls.tsv    request N's main-model prompt calls (the first decode calls of the main context, `dec`, whose
                      tokens add up to the request's prompt), each with its micro-batches, the host's prep and wait
  prompt_ubatch.tsv   each of those calls' micro-batches: graph, why, re-plans, pre, build, alloc, inputs, submit,
                      card0, the evaluation's card0, the CPU split's w, c and l
  prompt_splits.tsv   each of those micro-batches' splits on the cards: w, c, l, inputs, nodes
  prompt_sums.tsv     sums and medians over those micro-batches (of at least 128 tokens): the host's parts, card 0's
                      w and c, the splits on cards 1 to 8 that hold the host over 20 ms in l; and per draft call
                      (catch-up) of the prompt: the wait for the target's rows, the parts, the re-plans' drain and plan
  replans.tsv         every re-plan of every request (context from the micro-batch it came before), with its triggers
  replan_counts.tsv   per request and context: re-plans, re-plans with a size change, triggers by kind
  steps.tsv           the generated tokens' main-model evaluations of up to 8 tokens (after each request's prompt
                      calls), per request and token count: steps, graph reused/rebuilt by reason, medians of pre,
                      build, alloc, inputs, submit, card0 and the evaluation's card0, and per backend the medians of
                      w, c and l
  steps_all.tsv       the same, every request together
  steps_detail.tsv    each of those steps: the host's parts, the evaluation's card0, the CPU split's l, card 0's w and c
  prompt_inputs.tsv   per request, its main-model micro-batches of more than 8 tokens: tokens and the sum of `inputs`
"""
import csv
import os
import re
import statistics
import sys
from collections import defaultdict


def rd(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else ""


def write(path, rows, cols):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    tdir, out = sys.argv[1], sys.argv[2]
    preq = int(sys.argv[sys.argv.index("--prompt-req") + 1]) if "--prompt-req" in sys.argv else 7
    os.makedirs(out, exist_ok=True)
    reqs = rd(f"{tdir}/requests.tsv")

    data = {}
    for r in reqs:
        n = int(r["req"])
        d = f"{tdir}/req{n:02d}"
        calls = rd(f"{d}/calls.tsv")
        ubs = rd(f"{d}/ubatch.tsv")
        sp = defaultdict(list)
        for s in rd(f"{d}/splits.tsv"):
            sp[s["ubatch"]].append(s)
        rp = rd(f"{d}/replan.tsv")
        # the prompt's main-model calls: the first `dec` calls whose tokens add up to the prompt
        pn = int(r["prompt_n"]) if r["prompt_n"] else 0
        pcalls, acc = [], 0
        for c in calls:
            if c["ctx"] != "dec":
                continue
            if acc >= pn:
                break
            pcalls.append(c["call"])
            acc += int(c["n_tokens"])
        data[n] = dict(calls=calls, ubs=ubs, sp=sp, rp=rp, pcalls=pcalls, prompt_n=pn)

    # request N's prompt
    if preq in data:
        D = data[preq]
        crow, urow, srow = [], [], []
        for k, cid in enumerate(D["pcalls"], 1):
            c = next(x for x in D["calls"] if x["call"] == cid)
            mine = [u for u in D["ubs"] if u["call"] == cid]
            crow.append(dict(call=k, n_tokens=c["n_tokens"], ubatches=len(mine), prep=c["prep"], init=c["init"],
                             wait=c["wait"], rebuilt=c["rebuilt"], reused=c["reused"], replans=c["replans"],
                             card0=c["card0"], submit=c["submit"], w=c["w"], l=c["l"]))
            for j, u in enumerate(mine, 1):
                cpu = [s for s in D["sp"][u["ubatch"]] if s["backend"].startswith("CPU")]
                row = dict(call=k, ub=j, ubatch=u["ubatch"], n_tokens=u["n_tokens"], graph=u["graph"], why=u["why"],
                           replans=u["replans"], pre=u["pre"], build=u["build"], alloc=u["alloc"],
                           inputs=u["inputs"], submit=u["submit"], card0=u["card0"], total=u["total"],
                           eval_card0=u["s_card0"], copy=u["copy"], n_splits=u["n_splits"],
                           cpu_w=sum(int(s["w"]) for s in cpu), cpu_c=sum(int(s["c"]) for s in cpu),
                           cpu_l=sum(int(s["l"]) for s in cpu), cpu_first=" ".join(s["first"] for s in cpu))
                urow.append(row)
                for s in D["sp"][u["ubatch"]]:
                    if s["backend"].startswith("CPU"):
                        continue
                    srow.append(dict(call=k, ub=j, n_tokens=u["n_tokens"], split=s["split"], backend=s["backend"],
                                     inputs=s["inputs"], nodes=s["nodes"], w=s["w"], c=s["c"], l=s["l"]))
        write(f"{out}/prompt_calls.tsv", crow, ["call", "n_tokens", "ubatches", "prep", "init", "wait", "rebuilt",
                                                "reused", "replans", "card0", "submit", "w", "l"])
        write(f"{out}/prompt_ubatch.tsv", urow, ["call", "ub", "ubatch", "n_tokens", "graph", "why", "replans", "pre",
                                                 "build", "alloc", "inputs", "submit", "card0", "eval_card0", "total",
                                                 "copy", "n_splits", "cpu_w", "cpu_c", "cpu_l", "cpu_first"])
        write(f"{out}/prompt_splits.tsv", srow, ["call", "ub", "n_tokens", "split", "backend", "inputs", "nodes", "w",
                                                 "c", "l"])
        # sums over the prompt's main-model micro-batches, and the draft's calls between them
        big = [u for u in urow if int(u["n_tokens"]) >= 128]
        sums = []
        for f in ["pre", "build", "alloc", "inputs", "cpu_l", "eval_card0", "card0", "submit", "total"]:
            v = [int(u[f]) for u in big]
            sums.append(dict(part=f"main {f}", n=len(v), sum_us=sum(v), median_us=med(v), min_us=min(v), max_us=max(v)))
        c0 = [s for s in srow if s["backend"] == "CUDA0" and int(s["n_tokens"]) >= 128]
        for f in "wc":
            v = [int(s[f]) for s in c0]
            sums.append(dict(part=f"main CUDA0 {f}", n=len(v), sum_us=sum(v), median_us=med(v), min_us=min(v), max_us=max(v)))
        blk = [s for s in srow if s["backend"] != "CUDA0" and int(s["l"]) > 20000]
        sums.append(dict(part="main splits on cards 1-8 with l > 20 ms", n=len(blk), sum_us=sum(int(s["l"]) for s in blk),
                         median_us=med([int(s["l"]) for s in blk]), min_us=min([int(s["l"]) for s in blk], default=""),
                         max_us=max([int(s["l"]) for s in blk], default="")))
        sums.append(dict(part="  of them w", n=len(blk), sum_us=sum(int(s["w"]) for s in blk)))
        sums.append(dict(part="  of them c", n=len(blk), sum_us=sum(int(s["c"]) for s in blk)))
        ubi = {u["ubatch"]: u for u in D["ubs"]}
        drains = defaultdict(int)
        plans = defaultdict(int)
        for r in D["rp"]:
            m = re.search(r"drain = (-?\d+) plan = (-?\d+)", r["text"])
            u = ubi.get(r["ubatch"])
            if m and u:
                drains[u["call"]] += int(m.group(1))
                plans[u["call"]] += int(m.group(2))
        last = max(int(x) for x in D["pcalls"]) if D["pcalls"] else -1
        for c in D["calls"]:
            if c["ctx"] != "mtp" or int(c["call"]) > last + 1 or int(c["n_tokens"]) < 8:
                continue
            mine = [u for u in D["ubs"] if u["call"] == c["call"]]
            sums.append(dict(part=f"draft call of {c['n_tokens']} rows: decode-trace wait", n=1, sum_us=c["wait"]))
            for f in ["pre", "build", "alloc", "inputs", "submit", "total"]:
                v = [int(u[f]) for u in mine]
                sums.append(dict(part=f"draft call of {c['n_tokens']} rows: {f}", n=len(v), sum_us=sum(v), median_us=med(v)))
            for b in sorted({s["backend"] for u in mine for s in D["sp"][u["ubatch"]]}):
                for f in "wcl":
                    v = [sum(int(s[f]) for s in D["sp"][u["ubatch"]] if s["backend"] == b) for u in mine
                         if any(s["backend"] == b for s in D["sp"][u["ubatch"]])]
                    sums.append(dict(part=f"draft call of {c['n_tokens']} rows: {b} {f}", n=len(v), sum_us=sum(v),
                                     median_us=med(v), min_us=min(v, default=""), max_us=max(v, default="")))
            sums.append(dict(part=f"draft call of {c['n_tokens']} rows: re-plans' drain", n=c["replans"],
                             sum_us=drains[c["call"]]))
            sums.append(dict(part=f"draft call of {c['n_tokens']} rows: re-plans' plan", n=c["replans"],
                             sum_us=plans[c["call"]]))
        write(f"{out}/prompt_sums.tsv", sums, ["part", "n", "sum_us", "median_us", "min_us", "max_us"])

    # per request: the main model's micro-batches of more than 8 tokens, their tokens and the host's `inputs`
    pin = []
    for n, D in sorted(data.items()):
        mine = [u for u in D["ubs"] if u["ctx"] == "dec" and int(u["n_tokens"]) > 8]
        if not mine:
            continue
        t = sum(int(u["n_tokens"]) for u in mine)
        s = sum(int(u["inputs"]) for u in mine)
        cl = []
        for u in mine:
            cl += [int(x["l"]) for x in D["sp"][u["ubatch"]] if x["backend"].startswith("CPU")]
        pin.append(dict(req=n, prompt_n=D["prompt_n"], ubatches=len(mine), tokens=t, inputs_sum_us=s,
                        inputs_us_a_token=round(s / t, 1), inputs_median_us=med([int(u["inputs"]) for u in mine]),
                        cpu_l_sum_us=sum(cl), cpu_l_us_a_token=round(sum(cl) / t, 1)))
    write(f"{out}/prompt_inputs.tsv", pin, ["req", "prompt_n", "ubatches", "tokens", "inputs_sum_us", "inputs_us_a_token",
                                            "inputs_median_us", "cpu_l_sum_us", "cpu_l_us_a_token"])

    # re-plans
    rrow, counts = [], defaultdict(lambda: defaultdict(int))
    for n, D in sorted(data.items()):
        ubi = {u["ubatch"]: u for u in D["ubs"]}
        for r in D["rp"]:
            u = ubi.get(r["ubatch"], {})
            ctx = u.get("ctx", "?")
            size = "size_changed = 1" in r["text"]
            rrow.append(dict(req=n, ctx=ctx, n_tokens=u.get("n_tokens", ""), call=u.get("call", ""),
                             ubatch=r["ubatch"], text=r["text"], triggers=r["triggers"]))
            k = counts[(n, ctx)]
            k["replans"] += 1
            k["size_changed"] += size
            k["backends_changed"] += "backends_changed = 1" in r["text"]
            for t in (r["triggers"] or "").split(" | "):
                if not t:
                    continue
                m = re.match(r"galloc-(trigger|grow): (\w+)", t)
                k[f"{m.group(1)}:{m.group(2)}" if m else "other"] += 1
            m = re.search(r"drain = (-?\d+) plan = (-?\d+)", r["text"])
            if m:
                k["drain_us"] += int(m.group(1))
                k["plan_us"] += int(m.group(2))
    write(f"{out}/replans.tsv", rrow, ["req", "ctx", "n_tokens", "call", "ubatch", "text", "triggers"])
    keys = sorted({kk for v in counts.values() for kk in v})
    write(f"{out}/replan_counts.tsv", [dict(req=n, ctx=c, **v) for (n, c), v in sorted(counts.items())],
          ["req", "ctx"] + keys)

    # the generated tokens' main-model steps
    groups = defaultdict(list)
    for n, D in sorted(data.items()):
        pc = set(D["pcalls"])
        for u in D["ubs"]:
            if u["ctx"] != "dec" or u["call"] in pc or int(u["n_tokens"]) > 8 or u["call"] == "-1":
                continue
            groups[(n, int(u["n_tokens"]))].append((u, D["sp"][u["ubatch"]]))
            groups[("all", int(u["n_tokens"]))].append((u, D["sp"][u["ubatch"]]))
    backends = sorted({s["backend"] for v in groups.values() for _, sp in v for s in sp},
                      key=lambda b: (not b.startswith("CPU"), int(re.sub(r"\D", "", b) or 0)))
    cols = ["req", "n_tokens", "steps", "reused", "rebuilt", "pre", "build", "alloc", "inputs", "submit", "card0",
            "eval_card0", "total"]
    for b in backends:
        cols += [f"{b}_w", f"{b}_c", f"{b}_l"]
    rows_req, rows_all = [], []
    for (n, nt), v in sorted(groups.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        why = defaultdict(int)
        for u, _ in v:
            if u["graph"] == "rebuilt":
                why[u["why"]] += 1
        row = dict(req=n, n_tokens=nt, steps=len(v), reused=sum(1 for u, _ in v if u["graph"] == "reused"),
                   rebuilt=", ".join(f"{k} {c}" for k, c in sorted(why.items())) or "0")
        for f in ["pre", "build", "alloc", "inputs", "submit", "card0", "total"]:
            row[f] = med([int(u[f]) for u, _ in v])
        row["eval_card0"] = med([int(u["s_card0"]) for u, _ in v if u["s_card0"] != ""])
        for b in backends:
            for f in "wcl":
                row[f"{b}_{f}"] = med([sum(int(s[f]) for s in sp if s["backend"] == b) for _, sp in v
                                       if any(s["backend"] == b for s in sp)])
        (rows_all if n == "all" else rows_req).append(row)
    write(f"{out}/steps.tsv", rows_req, cols)
    write(f"{out}/steps_all.tsv", rows_all, cols)

    # every one of those steps: the host's parts and the evaluation before card 0
    det = []
    for n, D in sorted(data.items()):
        pc = set(D["pcalls"])
        k = 0
        for u in D["ubs"]:
            if u["ctx"] != "dec" or u["call"] in pc or int(u["n_tokens"]) > 8 or u["call"] == "-1":
                continue
            k += 1
            sp = D["sp"][u["ubatch"]]
            cpu = [s for s in sp if s["backend"].startswith("CPU")]
            c0 = [s for s in sp if s["backend"] == "CUDA0"]
            det.append(dict(req=n, step=k, ubatch=u["ubatch"], n_tokens=u["n_tokens"], graph=u["graph"], why=u["why"],
                            pre=u["pre"], build=u["build"], alloc=u["alloc"], inputs=u["inputs"], card0=u["card0"],
                            eval_card0=u["s_card0"], cpu_l=sum(int(s["l"]) for s in cpu),
                            cpu_first=" ".join(s["first"] for s in cpu),
                            cpu_nodes=" ".join(s["nodes"] for s in cpu),
                            cuda0_w=sum(int(s["w"]) for s in c0), cuda0_c=sum(int(s["c"]) for s in c0),
                            cuda0_inputs=" ".join(s["inputs"] for s in c0), submit=u["submit"], total=u["total"]))
    write(f"{out}/steps_detail.tsv", det, ["req", "step", "ubatch", "n_tokens", "graph", "why", "pre", "build", "alloc",
                                           "inputs", "card0", "eval_card0", "cpu_l", "cpu_first", "cpu_nodes", "cuda0_w",
                                           "cuda0_c", "cuda0_inputs", "submit", "total"])


if __name__ == "__main__":
    main()
