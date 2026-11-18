#!/usr/bin/env python3
"""server_turns.py ARM SERVER_LOG CLIENT_LOG: one run's per-request table rebuilt from its server log, as turns.csv.

For an agent-check run whose client tables are not at hand: the server log (llama-server's, one slot, the agent check
runs one task at a time) holds every request in order, with its prompt and decode times and the cache size when it
ended; the client log (release_check_s12.sh's <arm>.log) holds the tasks in the order they ran and each task's turns.
The requests are assigned to tasks and turns in that order. Per request:
  prompt_n, prompt_ms       the "prompt eval time" line (tokens read, milliseconds)
  predicted_n, predicted_ms the "eval time" line (tokens generated, milliseconds)
  cache_n                   the release line's n_tokens less prompt_n and predicted_n: the tokens taken from the cache
A request the server stopped without timing lines (the client's cap) is left out. Prints CSV with turns.csv's column
names for same_prompts.py and report.py work (the client's columns ttft_s, t_end_s, finish_reason and n_tool_calls
empty). Checked against R90's v3 turns.csv (66 requests): prompt_n, prompt_ms, predicted_n and predicted_ms equal on
all 65 timed requests, cache_n equal on 18 and one less on 47 (the last generated token is in n_tokens on some
requests only), so a rebuilt cache count is within one token.
Exits 1 when the requests in the log and the turns in the client log differ in number. Standard library only.
"""
import csv
import re
import sys

arm, slog, clog = sys.argv[1:4]
tasks = []
for line in open(clog, errors="replace"):
    m = re.search(r" \S+?/(\S+)/a(\d+)/t\d+ (?:PASS|FAIL|pass|fail) turns (\d+)", line)
    if m:
        tasks.append((m.group(1), m.group(2), int(m.group(3))))

launch = re.compile(r"slot launch_slot_: id +\d+ \| task (\d+) \| processing task")
pe = re.compile(r"\| task (\d+) \| prompt eval time = +([0-9.]+) ms / +(\d+) tokens")
ev = re.compile(r"\| task (\d+) \| +eval time = +([0-9.]+) ms / +(\d+) tokens")
rel = re.compile(r"\| task (\d+) \| stop processing: n_tokens = (\d+)")
order, req = [], {}
for line in open(slog, errors="replace"):
    m = launch.search(line)
    if m:
        order.append(m.group(1))
        req[m.group(1)] = {}
        continue
    for name, rx in (("p", pe), ("e", ev), ("r", rel)):
        m = rx.search(line)
        if m and m.group(1) in req:
            req[m.group(1)][name] = m.groups()[1:]

if len(order) != sum(t[2] for t in tasks):
    sys.exit(f"{len(order)} requests in {slog}, {sum(t[2] for t in tasks)} turns in {clog}")
w = csv.writer(sys.stdout)
w.writerow(["arm", "task_id", "attempt", "turn", "ttft_s", "t_end_s", "prompt_n", "cache_n", "prompt_ms", "predicted_n",
            "predicted_ms", "finish_reason", "n_tool_calls"])
i = 0
for task, att, n in tasks:
    for turn in range(1, n + 1):
        r = req[order[i]]
        i += 1
        if not all(k in r for k in "per"):
            continue
        pn, gn = int(r["p"][1]), int(r["e"][1])
        w.writerow([arm, task, att, turn, "", "", pn, max(0, int(r["r"][0]) - pn - gn), r["p"][0], gn, r["e"][0], "", ""])
