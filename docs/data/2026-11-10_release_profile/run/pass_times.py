#!/usr/bin/env python3
"""pass_times.py: the unprofiled time of a k-token pass from release_prof.sh's loads, two ways.

  python3 -I pass_times.py --requests OUT/requests.tsv --log warm=OUT/server-warm.log --log prof=OUT/server-prof.log \
      --log wall=OUT/server-wall.log --passes <release_parts.py's passes.tsv> --out <dir>

1. C120r's T(k): a request's prompt time / (N / k), from requests.tsv. It holds everything the request does besides its
   passes: its start, a context checkpoint after each call but the last, and in a load's first request at a size the
   one-off launches.
2. The time of the passes of one call. llama-server reads a prompt of N tokens in calls of N - 516, 512 and 4 tokens
   (N - 4 and 4 for 340) and logs the time when each of the first ones ends ("prompt processing, n_tokens = ..."). Per
   request, from the log's own time stamps: the first call (launch to its line), the second (line to line: its passes
   and one checkpoint) and the tail (the last line to the end of the prompt: a checkpoint and the last 4 tokens).
   Pass time = the first call / its passes, a last pass of fewer tokens (336 = 67 x 5 + 1) taken out at that size's
   own time in the same load. The first call still holds the request's start. With --passes (release_parts.py's
   table of every profiled pass) the profiled requests show how long that is: the log's time of each first call's end
   fixes the trace's clock, and the table gives each request's start (launch to its first pass), its first pass, and
   the pause between its first call and its last (the checkpoint).
Writes <out>/calls.tsv (every request of every load), <out>/prof_requests.tsv (the profiled requests' start, first
pass, checkpoint pause) and <out>/pass_times.tsv, and prints them. Standard library only."""
import argparse, csv, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--requests", required=True); ap.add_argument("--log", action="append", required=True)
ap.add_argument("--passes", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def stamp(line):  # "M.SS.mmm.uuu ..." -> ms
    m = re.match(r"(\d+)\.(\d+)\.(\d+)\.(\d+) ", line)
    return (int(m[1]) * 60 + int(m[2])) * 1000 + int(m[3]) + int(m[4]) / 1000 if m else None


reqs = list(csv.DictReader(open(a.requests), delimiter="\t"))
calls = []  # one dict per request
for spec in a.log:
    load, path = spec.split("=", 1)
    tasks, cur = [], None
    for line in open(path, errors="replace"):
        t = stamp(line)
        if t is None:
            continue
        if "launch_slot_" in line and "processing task" in line:
            cur = {"launch": t, "lines": [], "prompt_ms": None}
            tasks.append(cur)
        elif cur is not None and "prompt processing, n_tokens" in line:
            cur["lines"].append((int(re.search(r"n_tokens =\s*(\d+)", line)[1]), t))
        elif cur is not None and "prompt eval time" in line:
            cur["prompt_ms"] = float(re.search(r"prompt eval time =\s*([\d.]+) ms", line)[1])
    mine = [r for r in reqs if r["arm"] == load or r["arm"].startswith(load) and r["arm"][len(load):].isdigit()]
    if len(mine) != len(tasks):
        raise SystemExit(f"{load}: {len(tasks)} requests in the log, {len(mine)} in {a.requests}")
    for i, (r, t) in enumerate(zip(mine, tasks)):
        k, n = int(r["k"]), int(r["prompt_n"])
        c = {"load": load, "arm": r["arm"], "request": i + 1, "k": k, "n": n, "prompt_ms": t["prompt_ms"],
             "t_c120r": t["prompt_ms"] / (n / k), "launch": t["launch"], "lines": t["lines"]}
        c["call1_tokens"], c["call1_ms"] = t["lines"][0][0], t["lines"][0][1] - t["launch"]
        c["call2_tokens"] = t["lines"][1][0] - t["lines"][0][0] if len(t["lines"]) > 1 else 0
        c["call2_ms"] = t["lines"][1][1] - t["lines"][0][1] if len(t["lines"]) > 1 else float("nan")
        c["tail_ms"] = t["launch"] + t["prompt_ms"] - t["lines"][-1][1]
        calls.append(c)

with open(os.path.join(a.out, "calls.tsv"), "w") as f:
    f.write("load\tarm\trequest\tk\tprompt_n\tprompt_ms\tprompt_ms_per_pass\tcall1_tokens\tcall1_ms\tcall2_tokens\tcall2_ms\ttail_ms\n")
    for c in calls:
        f.write(f"{c['load']}\t{c['arm']}\t{c['request']}\t{c['k']}\t{c['n']}\t{c['prompt_ms']:.1f}\t{c['t_c120r']:.2f}\t"
                f"{c['call1_tokens']}\t{c['call1_ms']:.1f}\t{c['call2_tokens']}\t{c['call2_ms']:.1f}\t{c['tail_ms']:.1f}\n")
print(open(os.path.join(a.out, "calls.tsv")).read())

if a.passes:
    passes = {}  # k -> [(pass, first_card_host_ms, last_card_host_ms, last_card_raw_ms)]
    for r in csv.DictReader(open(a.passes), delimiter="\t"):
        passes.setdefault(int(r["k"]), []).append((int(r["pass"]), float(r["first_card_host_ms"]),
                                                   float(r["last_card_host_ms"]), float(r["last_card_raw_ms"])))
    prof = [c for c in calls if c["load"] == "prof"]
    rows = []
    for c in prof:
        ps = passes[c["k"]]
        # the passes of the first call: its tokens in micro-batches of k, the remainder as one more
        n1 = c["call1_tokens"] // c["k"] + (1 if c["call1_tokens"] % c["k"] else 0)
        end = ps[n1 - 1][2] + ps[n1 - 1][3]          # the last card ends the call's last pass
        origin = c["lines"][0][1] - end              # the log's time of the trace's host_ms 0
        launch = c["launch"] - origin
        rows.append({"k": c["k"], "request": c["request"], "origin": origin, "start": ps[0][1] - launch,
                     "first_pass": ps[1][1] - ps[0][1], "step": statistics.median(b[1] - x[1] for x, b in zip(ps[2:n1 - 1], ps[3:n1])),
                     "pause": ps[n1][1] - ps[n1 - 1][1], "passes_call1": n1})
    with open(os.path.join(a.out, "prof_requests.tsv"), "w") as f:
        f.write("k\trequest\ttrace_origin_in_log_ms\tstart_ms\tfirst_pass_ms\tpass_to_pass_ms\tcall1_passes\tlast_pass_to_final_call_ms\n")
        for r in rows:
            f.write(f"{r['k']}\t{r['request']}\t{r['origin']:.1f}\t{r['start']:.1f}\t{r['first_pass']:.1f}\t{r['step']:.2f}\t"
                    f"{r['passes_call1']}\t{r['pause']:.1f}\n")
    print(open(os.path.join(a.out, "prof_requests.tsv")).read())

with open(os.path.join(a.out, "pass_times.tsv"), "w") as f:
    f.write("load\tarm\tk\tprompt_n\tprompt_ms_per_pass\tcall1_ms_per_pass\tcall2_ms_per_pass_with_checkpoint\n")
    known = {}  # (load, n) -> {k: pass time}, for a call's last pass of fewer tokens
    for c in sorted(calls, key=lambda c: (c["load"], c["n"], c["k"])):
        k = c["k"]
        tk = known.setdefault((c["load"], c["n"]), {})

        def per_pass(ms, tokens):
            # a last pass of fewer tokens: taken out at that size's own time when the load measured it, else counted
            # as its share of a pass
            full, odd = tokens // k, tokens % k
            if odd and odd in tk:
                return (ms - tk[odd]) / full
            return ms / (full + odd / k)

        raw = per_pass(c["call1_ms"], c["call1_tokens"])
        tk[k] = raw
        c2 = per_pass(c["call2_ms"], c["call2_tokens"]) if c["call2_tokens"] else float("nan")
        f.write(f"{c['load']}\t{c['arm']}\t{k}\t{c['n']}\t{c['t_c120r']:.2f}\t{raw:.2f}\t{c2:.2f}\n")
print(open(os.path.join(a.out, "pass_times.tsv")).read())
