#!/usr/bin/env python3
"""timeline.py: R42 (Q1156), one prompt's per-card timeline from the fork's two light profiles.

  python3 -I timeline.py --load <name>=<load dir> [--load ...] [--pair <graph load>=<host load> ...] --out <dir>

A load dir is one rig_taper.sh OUT (r42_run.sh's <OUT>/<load>): load-A/server.log (-lv 5), load-A/requests.tsv
(taper_replay.py's requests in order) and, for a profiled load, prof-graph.csv (GGML_CUDA_PROFILE_MODE=graph) or
prof-host.csv (GGML_CUDA_PROFILE_HOST).

The two profiles, as the fork writes them (ggml/src/ggml-cuda/ggml-cuda.cu):
  graph  one line per graph evaluation and card: host_ms when the host had queued the evaluation (the end of its
         graph_compute: inputs copied, every kernel launched), eval_us the GPU time between two events around its
         work, how: replay, capture or eager (a captured CUDA graph replayed, captured anew, or every launch group
         launched on its own). Nothing waits: an evaluation is read when the same graph is next evaluated on its
         card; one whose end event has not completed by then is a "(not ready)" line without a GPU time, and the
         last evaluation of each graph on each card is never written.
  host   one line per graph evaluation and card: the host's entry to and exit from graph_compute and the GPU time.
         Before it takes the entry time the profile waits for the card's previous evaluation to end
         (cudaEventSynchronize), so the host never runs more than one evaluation ahead of a card.
A micro-batch of the main model is one evaluation on each of the nine cards in card order (the CPU split before it
is in neither file); an evaluation out of that order is the MTP draft's (on its cards 4 and 8).

Clocks: the draft context logs "graph reallocation: ..." when it allocates a graph, right before it queues the
graph's first split, so each such line sits within a ms of a draft evaluation; the offset (log = trace + offset)
that matches the most lines within --tol ms wins.

A request's prompt is the main micro-batches queued between the request's launch and launch + prompt_ms (the
server's prompt time). Its window runs from the first of them on card 0 to the first evaluation after the last of
them (the host has the prompt's output then). Per card: micro-batches, GPU time, busy share = GPU time / window.
--pair g=h takes the GPU time of each micro-batch from load h (the host profile has all of them) and the window
from load g (the graph profile does not change the host's pace), request by request.
The host's time before a split: from the previous evaluation in the trace (any card) to this one (graph: queued to
queued; host: exit to entry). The phases of a request: runs of main micro-batches (a call), the host's wait for the
cards after a call (from its last queued split to the next evaluation), runs of draft evaluations, and the pause
before the next call (a checkpoint save, the graph build and the CPU split).

Writes <out>/requests.tsv, busy.tsv, splits.tsv, phases.tsv, counts.tsv and prints a summary. Standard library only.
"""
import argparse
import bisect
import csv
import os
import re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="append", required=True, help="name=dir")
ap.add_argument("--pair", action="append", default=[], help="graph load=host load")
ap.add_argument("--out", required=True)
ap.add_argument("--cards", type=int, default=9)
ap.add_argument("--tol", type=float, default=1.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LAST = a.cards - 1


def stamp(line):  # "M.SS.mmm.uuu ..." -> ms since the server's start
    m = re.match(r"(\d+)\.(\d+)\.(\d+)\.(\d+) ", line)
    return (int(m[1]) * 60 + int(m[2])) * 1000 + int(m[3]) + int(m[4]) / 1000 if m else None


def parse_log(path):
    tasks, reallocs, info = [], [], {}
    cur = None
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        t = stamp(line)
        if t is None:
            continue
        if "pipeline parallelism enabled" in line:
            info["pipeline"] = line.split(":", 2)[-1].strip()[:80]
        if "graph reallocation:" in line:
            m = re.search(r"nodes = (\d+) \(changed = (\d+)\), splits = (\d+)", line)
            reallocs.append((t, int(m[1]), int(m[2]), int(m[3])))
            if cur is not None:
                cur["reallocs"].append((int(m[1]), int(m[2]), int(m[3])))
        if "launch_slot_" in line and "processing task" in line:
            cur = {"launch": t, "calls": [], "ckpt": [], "prompt_ms": None, "reused": None, "reallocs_main": None,
                   "reallocs": [], "taper": []}
            tasks.append(cur)
        elif cur is None:
            continue
        elif "prompt processing, n_tokens" in line:
            cur["calls"].append((int(re.search(r"n_tokens =\s*(\d+)", line)[1]), t))
        elif "created context checkpoint" in line:
            m = re.search(r"n_tokens = (\d+).*?([\d.]+) ms\)", line)
            cur["ckpt"].append((t, int(m[1]), float(m[2])))
        elif "ubatch taper:" in line:
            cur["taper"].append(line.split("ubatch taper:", 1)[1].strip()[:120])
        elif "prompt eval time" in line:
            cur["prompt_ms"] = float(re.search(r"prompt eval time =\s*([\d.]+) ms", line)[1])
        elif "graphs reused" in line:
            cur["reused"] = int(re.search(r"graphs reused =\s*(\d+)", line)[1])
        elif "graph reallocs =" in line:
            m = re.search(r"graph reallocs =\s*(\d+)\s*/\s*(\d+)", line)
            cur["reallocs_main"] = (int(m[1]), int(m[2]))
    return tasks, reallocs, info


def parse_trace(path):
    recs = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        host = "host_enter_ms" in rd.fieldnames
        for r in rd:
            if host:
                recs.append({"dev": int(r["device"]), "eval": int(r["eval"]), "t": float(r["host_enter_ms"]),
                             "t_exit": float(r["host_exit_ms"]), "gpu": float(r["gpu_us"]) / 1000, "how": "",
                             "groups": -1, "graph": -1})
            else:
                ok = int(r["group"]) >= 0
                recs.append({"dev": int(r["device"]), "eval": int(r["eval"]), "t": float(r["host_ms"]),
                             "t_exit": float(r["host_ms"]), "gpu": float(r["eval_us"]) / 1000 if ok else None,
                             "how": r["how"] if ok else "not ready", "groups": int(r["eval_groups"]),
                             "graph": int(r["graph"])})
    recs.sort(key=lambda x: (x["t"], x["dev"]))
    return recs, ("host" if host else "graph")


def classify(recs):
    mb, expect = -1, 0
    for r in recs:
        if r["dev"] == expect:
            if expect == 0:
                mb += 1
            r["main"], r["mb"] = True, mb
            expect = (expect + 1) % a.cards
        else:
            r["main"], r["mb"] = False, None


def align(recs, reallocs):
    ev = sorted([r["t"] for r in recs if not r["main"]] + [r["t_exit"] for r in recs if not r["main"]])
    lines = [x[0] for x in reallocs]
    if not ev or not lines:
        return None, 0, len(lines)

    def matched(off):
        n = 0
        for L in lines:
            j = bisect.bisect_left(ev, L - off - a.tol)
            if j < len(ev) and ev[j] <= L - off + a.tol:
                n += 1
        return n

    cands = [L - e for L in lines[:12] for e in ev]
    best = max(cands, key=matched)
    return best, matched(best), len(lines)


loads, req_rows, busy_rows, split_rows, phase_rows, count_rows, summary = {}, [], [], [], [], [], []
for spec in a.load:
    name, d = spec.split("=", 1)
    tasks, reallocs, info = parse_log(os.path.join(d, "load-A", "server.log"))
    reqs = list(csv.DictReader(open(os.path.join(d, "load-A", "requests.tsv")), delimiter="\t"))
    trace = next((os.path.join(d, f) for f in ("prof-graph.csv", "prof-host.csv") if os.path.exists(os.path.join(d, f))), None)
    recs, kind = parse_trace(trace) if trace else ([], "none")
    classify(recs)
    off, n_match, n_lines = align(recs, reallocs) if recs else (None, 0, len(reallocs))
    summary.append(f"{name}: {kind} profile, {len(recs)} evaluations, {sum(1 for r in recs if r['gpu'] is None)} without a "
                   f"GPU time; clock offset {('%.1f ms' % off) if off is not None else '-'} ({n_match} of {n_lines} draft "
                   f"allocation lines matched); {len(tasks)} requests in the log, {len(reqs)} in requests.tsv; "
                   f"{info.get('pipeline', 'no pipeline line')}")
    per_req = {}
    for i, (t, r) in enumerate(zip(tasks, reqs)):
        row = {"load": name, "request": i + 1, "kind": r["kind"], "task": r["task"], "arm": r["arm"],
               "prompt_n": int(r["prompt_n"]), "prompt_ms": t["prompt_ms"],
               "calls_logged": " ".join(str(c[0]) for c in t["calls"]),
               "checkpoint_saves_ms": " ".join("%.1f" % c[2] for c in t["ckpt"]),
               "graphs_reused": t["reused"], "graph_reallocs_main": "%d/%d" % t["reallocs_main"] if t["reallocs_main"] else "",
               "draft_reallocs": len(t["reallocs"]), "draft_reallocs_changed": sum(x[1] for x in t["reallocs"]),
               "taper": " ; ".join(t["taper"])}
        if off is not None and t["prompt_ms"] is not None:
            w0, w1 = t["launch"] - off, t["launch"] - off + t["prompt_ms"]
            mine = [x for x in recs if x["main"] and w0 <= x["t"] <= w1]
            mbs = sorted({x["mb"] for x in mine})
            mine = [x for x in recs if x["main"] and x["mb"] in set(mbs)]
            if mine:
                first = min(x["t"] for x in mine)
                j_last = max(k for k, x in enumerate(recs) if x["main"] and x["mb"] == mbs[-1])
                end = recs[j_last + 1]["t"] if j_last + 1 < len(recs) else w1
                win = end - first
                hows = Counter(x["how"] for x in mine)
                row.update({"micro_batches": len(mbs), "window_ms": round(win, 1),
                            "window_vs_prompt_ms": round(win - t["prompt_ms"], 1),
                            "draft_evals_in_window": sum(1 for x in recs if not x["main"] and first <= x["t"] <= end),
                            "main_how": " ".join(f"{k}:{v}" for k, v in sorted(hows.items()))})
                per_req[i + 1] = {"arm": r["arm"], "kind": r["kind"], "n": int(r["prompt_n"]), "window": win,
                                  "gpu": {(mbs.index(x["mb"]), x["dev"]): x["gpu"] for x in mine}, "n_mb": len(mbs)}
                for dev in range(a.cards):
                    xs = [x for x in mine if x["dev"] == dev]
                    g = [x["gpu"] for x in xs if x["gpu"] is not None]
                    busy_rows.append({"load": name, "gpu_from": name, "request": i + 1, "kind": r["kind"], "arm": r["arm"],
                                      "prompt_n": r["prompt_n"], "card": dev, "micro_batches": len(xs),
                                      "with_gpu_time": len(g), "gpu_ms": round(sum(g), 1), "window_ms": round(win, 1),
                                      "busy_share": round(sum(g) / win, 4) if win > 0 else ""})
                # the host's time before each split, and the phases
                span = [x for x in recs if first <= x["t"] <= end]
                prev = None
                for k, x in enumerate(recs):
                    if x["t"] < first or x["t"] > end:
                        prev = x
                        continue
                    if x["main"] and x["mb"] in mbs:
                        hb = x["t"] - prev["t_exit"] if prev is not None else None
                        split_rows.append({"load": name, "request": i + 1, "kind": r["kind"], "arm": r["arm"],
                                           "prompt_n": r["prompt_n"], "micro_batch": mbs.index(x["mb"]), "card": x["dev"],
                                           "t_ms": round(x["t"] - first, 1),
                                           "host_before_ms": round(hb, 2) if hb is not None else "",
                                           "after": ("main card %d" % prev["dev"]) if prev and prev["main"] else
                                                    (("draft card %d" % prev["dev"]) if prev else ""),
                                           "gpu_ms": round(x["gpu"], 2) if x["gpu"] is not None else "",
                                           "how": x["how"]})
                    prev = x
                # phases: runs of main (prompt) and draft evaluations in time order
                runs = []
                for x in span:
                    tag = "main" if x["main"] else "draft"
                    if runs and runs[-1]["tag"] == tag:
                        runs[-1]["recs"].append(x)
                    else:
                        runs.append({"tag": tag, "recs": [x]})
                for q, ru in enumerate(runs):
                    xs = ru["recs"]
                    nxt = runs[q + 1]["recs"][0]["t"] if q + 1 < len(runs) else end
                    g = [x["gpu"] for x in xs if x["gpu"] is not None]
                    phase_rows.append({"load": name, "request": i + 1, "kind": r["kind"], "arm": r["arm"],
                                       "prompt_n": r["prompt_n"], "run": q, "what": ru["tag"],
                                       "micro_batches": len({x["mb"] for x in xs}) if ru["tag"] == "main" else "",
                                       "evaluations": len(xs), "start_ms": round(xs[0]["t"] - first, 1),
                                       "queued_ms": round(xs[-1]["t_exit"] - xs[0]["t"], 1),
                                       "until_next_ms": round(nxt - xs[-1]["t_exit"], 1),
                                       "gpu_ms_known": round(sum(g), 1), "cards": "".join(sorted({str(x["dev"]) for x in xs}))})
        req_rows.append(row)
    # every evaluation of the load, by kind and how
    c = Counter(("main" if x["main"] else "draft", x["how"]) for x in recs)
    for (w, h), v in sorted(c.items()):
        count_rows.append({"load": name, "evaluations": w, "how": h or "(host profile)", "n": v})
    loads[name] = per_req

for spec in a.pair:
    g, h = spec.split("=", 1)
    for i, rg in loads.get(g, {}).items():
        rh = loads.get(h, {}).get(i)
        if not rh or rh["arm"] != rg["arm"] or rh["n"] != rg["n"]:
            summary.append(f"pair {g}={h} request {i}: not paired")
            continue
        if rh["n_mb"] != rg["n_mb"]:
            # a graph trace never writes the last evaluation of each graph: the window still holds, the micro-batches
            # cannot be matched one to one
            summary.append(f"pair {g}={h} request {i}: {rg['n_mb']} micro-batches in {g}, {rh['n_mb']} in {h}; GPU time "
                           f"from {h}, window from {g}, no ratio per micro-batch")
        for dev in range(a.cards):
            gs = [rh["gpu"].get((m, dev)) for m in range(rh["n_mb"])]
            gs = [x for x in gs if x is not None]
            gg = [(rg["gpu"].get((m, dev)), rh["gpu"].get((m, dev))) for m in range(rg["n_mb"])] if rh["n_mb"] == rg["n_mb"] else []
            both = [(x, y) for x, y in gg if x is not None and y is not None]
            busy_rows.append({"load": g, "gpu_from": h, "request": i, "kind": rg["kind"], "arm": rg["arm"],
                              "prompt_n": rg["n"], "card": dev, "micro_batches": rh["n_mb"], "with_gpu_time": len(gs),
                              "gpu_ms": round(sum(gs), 1), "window_ms": round(rg["window"], 1),
                              "busy_share": round(sum(gs) / rg["window"], 4),
                              "graph_over_host_gpu": round(sum(x for x, _ in both) / sum(y for _, y in both), 4) if both else ""})


# where the host's time in a prompt's window goes, per request: the parts add up to the window
wait_rows = []
ckpt_ms = {(r["load"], r["request"]): sum(float(x) for x in r["checkpoint_saves_ms"].split()) for r in req_rows}
for (ld, rq) in sorted({(s["load"], s["request"]) for s in split_rows}):
    ss = [s for s in split_rows if s["load"] == ld and s["request"] == rq]
    ps = [p for p in phase_rows if p["load"] == ld and p["request"] == rq]
    win = next(r["window_ms"] for r in req_rows if r["load"] == ld and r["request"] == rq)
    hb = [s for s in ss if s["host_before_ms"] != ""]
    c0_in = [s["host_before_ms"] for s in hb if s["card"] == 0 and s["after"].startswith("main")]
    c0_first = [s["host_before_ms"] for s in hb if s["card"] == 0 and not s["after"].startswith("main")]
    blocked = [s["host_before_ms"] for s in hb if s["card"] > 0 and s["host_before_ms"] > 20]
    normal = [s["host_before_ms"] for s in hb if s["card"] > 0 and s["host_before_ms"] <= 20]
    mains = [p for p in ps if p["what"] == "main"]
    drafts = [p for p in ps if p["what"] == "draft"]
    first_row = next(s for s in ss if s["micro_batch"] == 0 and s["card"] == 0)
    parts = {
        "card0_within_call_ms": sum(c0_in),
        "blocked_splits_ms": sum(blocked),
        "other_splits_ms": sum(normal),
        "wait_for_cards_after_call_ms": sum(p["until_next_ms"] for p in mains),
        "draft_catch_up_ms": sum(p["queued_ms"] for p in drafts),
        "after_draft_ms": sum(p["until_next_ms"] for p in drafts),
    }
    # the first micro-batch's own host time lies before the window; the later calls' first ones are in after_draft_ms
    wait_rows.append({"load": ld, "request": rq, "kind": ss[0]["kind"], "arm": ss[0]["arm"], "prompt_n": ss[0]["prompt_n"],
                      "window_ms": win, "micro_batches": len({s["micro_batch"] for s in ss}), "calls": len(mains),
                      **{k: round(v, 1) for k, v in parts.items()},
                      "sum_ms": round(sum(parts.values()), 1),
                      "card0_within_call_median_ms": round(sorted(c0_in)[len(c0_in) // 2], 1) if c0_in else "",
                      "card0_first_of_call_ms": " ".join("%.1f" % x for x in c0_first),
                      "checkpoint_saves_ms": round(ckpt_ms.get((ld, rq), 0.0), 1),
                      "blocked_splits": len(blocked),
                      "blocked_split_ms_list": " ".join("%.0f" % x for x in blocked)})


def write(fn, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(os.path.join(a.out, fn), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


write("requests.tsv", req_rows)
write("busy.tsv", busy_rows)
write("splits.tsv", split_rows)
write("phases.tsv", phase_rows)
write("counts.tsv", count_rows)
write("waits.tsv", wait_rows)
print("\n".join(summary))
