#!/usr/bin/env python3
"""server_graph_log.py: per request of one llama-server log, the CUDA backend's graph decisions that
GGML_CUDA_GRAPH_DEBUG=1 logs (at debug level: the server needs -lv 5) and the keep-alive's counts that the server
prints after each reply while GGML_CUDA_KEEPALIVE_EAGER is set (R192, Q1561).

  python3 -I server_graph_log.py SERVER.log [--requests requests-LOAD.csv] [--tsv tasks.tsv] > tasks.md

A request is one slot task: from its "launch_slot_: ... task N" line to the next task's. Per task, the debug lines
of ggml_backend_cuda_graph_compute and its helpers (ggml-cuda.cu), each one evaluation of one card:
  reused       "CUDA Graph id N reused": the graph llama.cpp built is the one this card ran last (same uid), so its
               properties are not compared again
  warm-up done "CUDA graph warmup complete": the second unchanged call after a change; this evaluation is captured
  warm-up reset "CUDA graph warmup reset": the properties changed after a completed warm-up; this evaluation and the
               ones until two unchanged calls run kernel by kernel
  update failed, disabled: an executable that could not be updated, a graph with a node CUDA graphs cannot take
An evaluation of a graph GGML_CUDA_GRAPHS_FORCE takes (more than 8 tokens) is decided without these lines, and an
evaluation whose properties changed before its warm-up completed logs nothing; so the counts name causes, while the
graph-mode profile (graph_passes.py) gives every evaluation's outcome.
The keep-alive line ("keep-alive = ... EAGER=1: E evaluations kernel by kernel, A acted on (S stops), ...") counts
since the start, over the cards; per task the difference to the task before is printed: evaluations run kernel by
kernel, those at shapes new to their card (on which the switch ended the card's block before it ran: "acted on"),
and the stops of a running block among them. With --requests the tasks are named by the deep lane's requests CSV in
order (kind, pass, depth).
Standard library only.
"""
import argparse
import csv
import re

LAUNCH = re.compile(r"launch_slot_: id +\d+ \| task (\d+) \|")
KEEP = re.compile(r"task (\d+) \| +keep-alive = .*EAGER=\d+: (\d+) evaluations kernel by kernel, (\d+) acted on \((\d+) stops\), "
                  r"(\d+) new sizes outside a graph \((\d+) stops\)")
REUSED_GRAPHS = re.compile(r"task (\d+) \| +graphs reused = +(\d+)")
GEN = re.compile(r"task (\d+) \| +eval time = +([0-9.]+) ms / +(\d+) tokens")
KINDS = [("reused", re.compile(r"CUDA Graph id \d+ reused")),
         ("warmup_done", re.compile(r"CUDA graph warmup complete")),
         ("warmup_reset", re.compile(r"CUDA graph warmup reset")),
         ("update_failed", re.compile(r"CUDA graph update failed")),
         ("disabled", re.compile(r"disabling CUDA graphs"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--requests")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    tasks = []
    cur = {"task": "(load)", **{k: 0 for k, _ in KINDS}}
    tasks.append(cur)
    with open(a.log, errors="replace") as f:
        for line in f:
            m = LAUNCH.search(line)
            if m:
                cur = {"task": m.group(1), **{k: 0 for k, _ in KINDS}}
                tasks.append(cur)
                continue
            for k, rx in KINDS:
                if rx.search(line):
                    cur[k] += 1
                    break
            else:
                m = KEEP.search(line)
                if m and m.group(1) == cur["task"]:
                    cur["ka"] = tuple(int(x) for x in m.groups()[1:])
                    continue
                m = REUSED_GRAPHS.search(line)
                if m and m.group(1) == cur["task"]:
                    cur["graphs_reused"] = int(m.group(2))
                    continue
                m = GEN.search(line)
                if m and m.group(1) == cur["task"]:
                    cur["gen_ms"], cur["gen_n"] = float(m.group(2)), int(m.group(3))
    names = []
    if a.requests:
        with open(a.requests, newline="") as f:
            names = [f"{r['kind']} {r['pass']} at {r['depth']}" for r in csv.DictReader(f)]
    prev = (0, 0, 0, 0, 0)
    out = []
    for i, t in enumerate(tasks):
        ka = t.get("ka")
        d = tuple(x - y for x, y in zip(ka, prev)) if ka else None
        if ka:
            prev = ka
        name = names[i - 1] if 0 < i <= len(names) else ""
        out.append((t, name, d))
    print(f"# Graph decisions and keep-alive counts per request ({a.log})\n")
    print("| task | request | gen tokens | gen ms | graphs reused (llama) | debug: reused | warm-up done (captured) | "
          "warm-up reset (kernel by kernel) | update failed | disabled | kernel by kernel | at new shapes | block stops |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, name, d in out:
        dd = d if d else ("", "", "", "", "")
        print(f"| {t['task']} | {name} | {t.get('gen_n', '')} | {t.get('gen_ms', '')} | {t.get('graphs_reused', '')} | "
              f"{t['reused']} | {t['warmup_done']} | {t['warmup_reset']} | {t['update_failed']} | {t['disabled']} | "
              f"{dd[0]} | {dd[1]} | {dd[2]} |")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("task\trequest\tgen_n\tgen_ms\tgraphs_reused\tdbg_reused\tdbg_warmup_done\tdbg_warmup_reset\t"
                    "dbg_update_failed\tdbg_disabled\tka_eager\tka_new_shapes\tka_stops\tka_new_sizes_outside\t"
                    "ka_outside_stops\n")
            for t, name, d in out:
                dd = d if d else ("", "", "", "", "")
                f.write(f"{t['task']}\t{name}\t{t.get('gen_n', '')}\t{t.get('gen_ms', '')}\t{t.get('graphs_reused', '')}\t"
                        f"{t['reused']}\t{t['warmup_done']}\t{t['warmup_reset']}\t{t['update_failed']}\t{t['disabled']}\t"
                        + "\t".join(str(x) for x in dd) + "\n")


if __name__ == "__main__":
    main()
