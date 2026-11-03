#!/usr/bin/env python3
# verify_split.py OUT_DIR [CONSOLE]: Q583's tables from verify_cost.sh's output folder (prof-draft.csv, prof-bench.csv,
# bench-prof.json, bench-wall.json, reply-draft.json) and its console (CONSOLE, default OUT_DIR/../verify_cost.out,
# which marks where the drafting request starts in the trace). Standard library only.
#
# A GGML_CUDA_PROFILE trace (device,eval,host_ms,group,label,us) has one line per launch group of every graph
# evaluation on a card. The evaluations, in host-time order, are cut into passes: a pass ends when the next evaluation
# runs on a lower-numbered card (a layer split runs the cards in order; a card's second evaluation in a row joins its
# pass). A pass over five or more cards is the main model's; a shorter one is the draft's (its weights on one card,
# its token embeddings on another). Each launch group goes into one part of the model by its label (PARTS below).
#
# Writes in OUT_DIR: bench_by_k.csv (ms of GPU time per k-token pass of the main model, per part, the median over the
# repetitions, and llama-bench's wall time with and without the profiler), draft_request.csv (the drafting request's
# main passes and draft passes per step), passes.csv (every pass of both traces), and prints the Markdown tables.
import collections, csv, json, re, statistics, sys

PARTS = [  # (part, regex on the launch group's label); the first match wins
    ("routed experts (Q2_0)", r"^MUL_MAT_ID|_exps:"),
    ("shared expert", r"_shexp:"),
    ("BF16 hyper-connections", r"\bhc_|DSV4_HC"),
    ("router (F32)", r"ffn_gate_inp:"),
    ("attention and recurrent projections", r" (attn_[a-z_]+|ssm_out|ssm_ba|ssm_alpha|ssm_beta|ssm_in|ssm_dt|eh_proj):"),
    ("recurrent state (delta net, conv)", r"GATED_DELTA_NET|SSM_CONV|ssm_conv|conv_states|ssm_states|_state"),
    ("flash attention", r"FLASH_ATTN_EXT"),
    ("output layer", r" output:"),
    ("token embeddings", r"^GET_ROWS.* (token_embd|inp_embd)"),
]
PART_NAMES = [p for p, _ in PARTS] + ["other ops without weights", "other weights"]


def part_of(label):
    for name, rx in PARTS:
        if re.search(rx, label):
            return name
    ops, _, weights = label.partition(" ")
    return "other weights" if re.search(r":[a-z0-9_]+:\d+x\d+", weights) else "other ops without weights"


def passes_of(path, skip_lines=0):
    """[(t0_ms, [devices], {part: us}, {label: us}, n_line_first)] in host-time order."""
    evals = collections.OrderedDict()
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            key = (int(row["device"]), int(row["eval"]))
            e = evals.setdefault(key, {"t": float(row["host_ms"]), "line": i, "groups": []})
            e["groups"].append((row["label"], float(row["us"])))
    order = sorted(evals.items(), key=lambda kv: kv[1]["t"])
    out, cur, last_dev = [], None, None
    for (dev, _), e in order:
        if e["line"] < skip_lines:
            continue
        if cur is None or dev < last_dev:
            cur = {"t0": e["t"], "devs": [], "parts": collections.Counter(), "labels": collections.Counter(),
                   "line": e["line"]}
            out.append(cur)
        if not cur["devs"] or cur["devs"][-1] != dev:
            cur["devs"].append(dev)
        for label, us in e["groups"]:
            cur["parts"][part_of(label)] += us
            cur["labels"][label] += us
        last_dev = dev
    return out


def ms(p):
    return sum(p["parts"].values()) / 1000


def fmt_row(cells):
    return "| " + " | ".join(cells) + " |"


def main():
    d = sys.argv[1].rstrip("/")
    log = open(sys.argv[2] if len(sys.argv) > 2 else f"{d}/../verify_cost.out").read()
    m = re.search(r"request starts after (\d+) profile lines", log)
    req_line = int(m.group(1)) if m else 0
    rows_all = []

    # llama-bench: k-token passes, warm-up first, then the repetitions, test after test
    bench = {s: json.load(open(f"{d}/bench-{s}.json")) for s in ("prof", "wall")}
    ks = [t["n_prompt"] for t in bench["prof"]]
    reps = len(bench["prof"][0]["samples_ns"])
    bp = [p for p in passes_of(f"{d}/prof-bench.csv") if len(p["devs"]) >= 5]
    print(f"bench trace: {len(bp)} main passes for {len(ks)} tests x (1 + {reps}); device patterns "
          f"{collections.Counter(tuple(p['devs']) for p in bp).most_common(3)}")
    per_k = {}
    for i, k in enumerate(ks):
        chunk = bp[i * (1 + reps):(i + 1) * (1 + reps)][1:]
        per_k[k] = chunk
        for j, p in enumerate(chunk):
            rows_all.append(("bench", k, j, p))
    wall = {s: {t["n_prompt"]: t["avg_ns"] / 1e6 for t in bench[s]} for s in bench}
    with open(f"{d}/bench_by_k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "part", "gpu_ms_per_pass_median"])
        for k, chunk in per_k.items():
            for part in PART_NAMES + ["total"]:
                vals = [ms(p) if part == "total" else p["parts"][part] / 1000 for p in chunk]
                w.writerow([k, part, f"{statistics.median(vals):.2f}"])
            w.writerow([k, "wall ms per pass (llama-bench, profiler on)", f"{wall['prof'][k]:.2f}"])
            w.writerow([k, "wall ms per pass (llama-bench, profiler off)", f"{wall['wall'][k]:.2f}"])

    # the drafting request: main passes (the prompt, then the verifies) and the draft passes between them
    dp = passes_of(f"{d}/prof-draft.csv", req_line)
    steps, cur = [], None
    for p in dp:
        if len(p["devs"]) >= 5:
            cur = {"main": p, "drafts": []}
            steps.append(cur)
        elif cur is not None:
            cur["drafts"].append(p)
    print(f"draft trace: {len(dp)} passes from line {req_line}, {len(steps)} main passes; draft device patterns "
          f"{collections.Counter(tuple(q['devs']) for s in steps for q in s['drafts']).most_common(3)}; drafts per step "
          f"{collections.Counter(len(s['drafts']) for s in steps).most_common(5)}")
    # the prompt's passes (and the last step) are followed by one draft pass; a drafting step by the n_max drafts and
    # the draft's pass over the tokens the verify accepted
    prompt, verify = steps[0], [s for s in steps if len(s["drafts"]) > 1]
    with open(f"{d}/draft_request.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "main_gpu_ms", "draft_passes", "draft_gpu_ms", "step_wall_ms_profiled"]
                   + [f"main:{p}" for p in PART_NAMES] + [f"draft:{p}" for p in PART_NAMES])
        for i, s in enumerate(steps):
            nxt = steps[i + 1]["main"]["t0"] if i + 1 < len(steps) else None
            wall_ms = f"{nxt - s['main']['t0']:.1f}" if nxt else ""
            dsum = collections.Counter()
            for q in s["drafts"]:
                dsum.update(q["parts"])
            w.writerow([i, f"{ms(s['main']):.2f}", len(s["drafts"]), f"{sum(dsum.values()) / 1000:.2f}", wall_ms]
                       + [f"{s['main']['parts'][p] / 1000:.2f}" for p in PART_NAMES]
                       + [f"{dsum[p] / 1000:.2f}" for p in PART_NAMES])
            rows_all.append(("draft-main", "", i, s["main"]))
            for j, q in enumerate(s["drafts"]):
                rows_all.append(("draft-draft", "", f"{i}.{j}", q))
    with open(f"{d}/passes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "k", "index", "devices", "gpu_ms"] + PART_NAMES)
        for trace, k, idx, p in rows_all:
            w.writerow([trace, k, idx, " ".join(map(str, p["devs"])), f"{ms(p):.2f}"]
                       + [f"{p['parts'][x] / 1000:.2f}" for x in PART_NAMES])

    # Markdown: GPU ms per pass by part and k, the verify passes of the drafting request, the draft passes
    vmed = lambda f: statistics.median([f(s) for s in verify]) if verify else 0.0
    dparts = lambda s, x: sum(q["parts"][x] for q in s["drafts"]) / 1000
    hdr = ["part"] + [f"k = {k}" for k in ks] + ["verify in the draft request (median)", "draft passes per step (median)"]
    print("\n" + fmt_row(hdr) + "\n" + fmt_row(["---"] + ["---:"] * (len(hdr) - 1)))
    for x in PART_NAMES:
        cells = [f"{statistics.median([p['parts'][x] / 1000 for p in per_k[k]]):.1f}" for k in ks]
        cells += [f"{vmed(lambda s: s['main']['parts'][x] / 1000):.1f}", f"{vmed(lambda s: dparts(s, x)):.1f}"]
        print(fmt_row([x] + cells))
    cells = [f"**{statistics.median([ms(p) for p in per_k[k]]):.1f}**" for k in ks]
    cells += [f"**{vmed(lambda s: ms(s['main'])):.1f}**", f"**{vmed(lambda s: sum(ms(q) for q in s['drafts'])):.1f}**"]
    print(fmt_row(["**GPU total**"] + cells))
    print(fmt_row(["wall, profiler on"] + [f"{wall['prof'][k]:.1f}" for k in ks] + ["", ""]))
    print(fmt_row(["wall, profiler off"] + [f"{wall['wall'][k]:.1f}" for k in ks] + ["", ""]))
    sw = [steps[i + 1]["main"]["t0"] - s["main"]["t0"] for i, s in enumerate(steps[:-1]) if len(s["drafts"]) > 1]
    if sw:
        print(f"\ndrafting request: {len(verify)} verify passes; step wall time (profiled, host clock) median "
              f"{statistics.median(sw):.1f} ms, mean {statistics.mean(sw):.1f}; prompt pass {ms(prompt['main']):.1f} ms GPU")
    try:
        t = json.load(open(f"{d}/reply-draft.json"))["timings"]
        print("reply timings:", {k: t[k] for k in t if k.startswith(("predicted", "prompt", "draft"))})
    except Exception as e:  # the reply is optional for the tables
        print("reply timings: not read:", e)


main()
