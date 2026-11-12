#!/usr/bin/env python3
"""r130_tables.py: R130's tables from run/draft_trace.py parse's rows and each load's llama-server log (-lv 5,
LLAMA_UBATCH_TRACE=1): the MTP draft's re-plans split by scheduler, and the draft's compute buffers per card.

    python3 -I r130_tables.py replans trace.tsv NAME=server.log [...] > replans.tsv
    python3 -I r130_tables.py buffers NAME=server.log [...] > buffers.tsv
    python3 -I r130_tables.py memory NAME=load-A [...] > memory.tsv

replans: per load, phase (prompt, gen, load) and scheduler, the draft's re-plans, those after a change of cards, and
  the sums of their drains and plans in ms (draft_trace.py's `trace.tsv` rows with replan = 1, context draft). The
  scheduler is `per-size` for a pass of up to 8 tokens in a load whose log names per-size schedulers
  (`process_ubatch_graph: scheduler for ...`, LLAMA_MTP_GRAPH_KEEP=1), else `shared` (the context's own scheduler).
buffers: per load and buffer (CUDA0..CUDA8, CUDA_Host), the draft's compute buffers at the end of the log in MiB: the
  shared scheduler's, the per-size schedulers' sum, and both; plus the count of per-size schedulers made, of their
  plans (`sched_small_plan:` lines, LLAMA_SCHED_PER_SIZE_RESERVE=1), of the draft's schedulers allocated (the shared
  one and each per-size one) and of their buffers allocated again larger (one per buffer; `-> B` may round to `A`).
  From the allocator's `galloc-grow: <buffer> buffer A -> B MiB` lines (LLAMA_UBATCH_TRACE=1): consecutive lines
  are one scheduler's allocation; a run of them from 0.00 with a zero CUDA0 buffer is a draft
  scheduler's (the target's has hundreds of MiB on every card): a per-size one when a per-size scheduler was made
  since the previous run, else the shared one planned again (each context reserves twice at load); a lone line grows
  the scheduler whose buffer it starts from (the draft's first, else the target's).
memory: per load (rig_taper.sh's load folder) and card, the MiB in use after loading and at the end
  (`memory_loaded.txt`, `memory_after.txt`), the free MiB in llama-server's memory breakdown at exit
  (`common_memory_breakdown_print`, the main model's context; the draft's memory is in its `unaccounted` column), and
  the load's lowest MemAvailable in MiB (`memavail.txt`). Standard library only.
"""
import csv
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GROW = re.compile(r"galloc-grow: (\S+) buffer ([0-9.]+) -> ([0-9.]+) MiB")
PERSIZE = re.compile(r"process_ubatch_graph: scheduler for ")
SMALL = re.compile(r"sched_small_plan: ")
BUFS = [f"CUDA{i}" for i in range(9)] + ["CUDA_Host"]


def has_per_size(path):
    with open(path, errors="replace") as f:
        return any(PERSIZE.search(line) for line in f)


def replans(argv):
    trace, logs = argv[0], dict(a.split("=", 1) for a in argv[1:])
    per_size = {name: has_per_size(path) for name, path in logs.items()}
    acc = defaultdict(lambda: [0, 0, 0, 0])
    with open(trace, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["context"] != "draft" or row["replan"] != "1" or row["log"] not in logs:
                continue
            sched = "per-size" if per_size[row["log"]] and int(row["n_tokens"]) <= 8 else "shared"
            a = acc[(row["log"], row["phase"], sched)]
            a[0] += 1
            a[1] += int(row["cards_changed"])
            a[2] += int(row["drain_us"])
            a[3] += int(row["plan_us"])
    print("load\tphase\tscheduler\treplans\tafter_a_change_of_cards\tdrains_ms\tplans_ms")
    for name in logs:
        for phase in ("load", "prompt", "gen"):
            for sched in ("shared", "per-size"):
                if sched == "per-size" and not per_size[name]:
                    continue
                n, c, d, p = acc[(name, phase, sched)]
                print(f"{name}\t{phase}\t{sched}\t{n}\t{c}\t{d / 1000:.1f}\t{p / 1000:.3f}")
    return 0


def buffers(argv):
    print("load\tbuffer\tshared_mib\tper_size_mib\tdraft_mib\tper_size_schedulers\tper_size_plans\t"
          "draft_allocations\tdraft_growths")
    for spec in argv:
        name, path = spec.split("=", 1)
        runs, prev_no, n_sizes, n_small = [], None, 0, 0
        with open(path, errors="replace") as f:
            for no, raw in enumerate(f):
                line = ANSI.sub("", raw)
                if PERSIZE.search(line):
                    n_sizes += 1
                    runs.append("per-size")
                if SMALL.search(line):
                    n_small += 1
                m = GROW.search(line)
                if not m:
                    continue
                if prev_no is None or no != prev_no + 1 or not isinstance(runs[-1], list):
                    runs.append([])
                runs[-1].append((m.group(1), float(m.group(2)), float(m.group(3))))
                prev_no = no
        # draft[0] is the shared scheduler: an allocation from zero with no per-size scheduler made since the last one
        # is that scheduler planned again (each context reserves twice at load), so it replaces it
        draft, target, grows, pending, n_alloc = [], [], 0, 0, 0
        for run in runs:
            if run == "per-size":
                pending += 1
                continue
            frm = {b: a for b, a, _ in run}
            to = {b: c for b, _, c in run}
            if len(run) >= len(BUFS) - 1 and all(v == 0.0 for v in frm.values()):
                if to.get("CUDA0", 0.0) > 0.0:
                    target[:] = [dict(to)]
                elif pending and draft:
                    draft.append(dict(to))
                    pending -= 1
                    n_alloc += 1
                elif draft:
                    draft[0] = dict(to)
                else:
                    draft.append(dict(to))
                    n_alloc += 1
                continue
            for b, a, c in run:
                for s in draft + target:
                    if abs(s.get(b, 0.0) - a) < 0.005:
                        s[b] = c
                        if any(s is d for d in draft):
                            grows += 1
                        break
        shared = draft[0] if draft else {}
        for b in BUFS:
            sh = shared.get(b, 0.0)
            ps = sum(s.get(b, 0.0) for s in draft[1:])
            print(f"{name}\t{b}\t{sh:.2f}\t{ps:.2f}\t{sh + ps:.2f}\t{n_sizes}\t{n_small}\t{n_alloc}\t{grows}")
    return 0


def memory(argv):
    card = re.compile(r"^\s*(\d+), (\d+) MiB")
    brk = re.compile(r"common_memory_breakdown_print: \|\s+- CUDA(\d+) \([^)]*\)\s+\|\s+\d+ =\s+(\d+) \+ \(\s*(\d+) =\s*\d+ \+"
                     r"\s+\d+ \+\s+\d+\) \+\s+(\d+) \|")
    print("load\tcard\tmib_loaded\tmib_end\tbreakdown_free_mib\tbreakdown_unaccounted_mib\tlowest_memavailable_mib")
    for spec in argv:
        name, d = spec.split("=", 1)
        used = {}
        for which in ("loaded", "after"):
            with open(f"{d}/memory_{which}.txt", errors="replace") as f:
                for line in f:
                    m = card.match(line)
                    if m:
                        used.setdefault(int(m.group(1)), {})[which] = int(m.group(2))
        free = {}
        with open(f"{d}/server.log", errors="replace") as f:
            for raw in f:
                m = brk.search(ANSI.sub("", raw))
                if m:
                    free[int(m.group(1))] = (m.group(2), m.group(4))
        with open(f"{d}/memavail.txt") as f:
            low = min(int(line.split()[1]) for line in f if len(line.split()) == 2)
        for c in sorted(used):
            fr, un = free.get(c, ("-", "-"))
            print(f"{name}\t{c}\t{used[c].get('loaded', '-')}\t{used[c].get('after', '-')}\t{fr}\t{un}\t{low}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"replans": replans, "buffers": buffers, "memory": memory}.get(
        mode, lambda a: print(__doc__) or 2)(sys.argv[2:]))
