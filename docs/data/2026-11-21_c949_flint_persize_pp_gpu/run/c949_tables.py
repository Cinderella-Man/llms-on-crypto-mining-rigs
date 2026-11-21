#!/usr/bin/env python3
"""C949 (Q1687): the tables from run.sh's output folders. Counts and token hashes only: no time from the traces.

Usage: python3 -I c949_tables.py <out folder> <run folder> [<run folder> ...]
Each run folder holds one sub-folder per arm (pp0, pp1, off, base) with server.log, session.tsv and mem.csv.
Writes to <out folder>:
  tokens.tsv     every request of every round: run, arm, round, request, kind, generated tokens, prompt tokens read,
                 cached tokens, md5 of the token ids (12 hex digits)
  same.tsv       per request: how many (run, arm, round) replies it has and how many distinct token lists
  arms.tsv       per run and arm: the pipeline parallelism line, per-size schedulers made and their input copies, passes
                 of at most 8 tokens (on a per-size scheduler when they are on) by input copy, those rebuilt while an
                 earlier graph was in flight (why = in_flight), prompt calls of several micro-batches whose last has at
                 most 8 tokens, and MiB per card after the session
  persize.txt    every per-size scheduler's creation line, per run and arm
  small.tsv      the passes of at most 8 tokens: run, arm, tokens, outputs, graph (reused, rebuilt and why), input
                 copy, count
  tails.tsv      the prompt calls of several micro-batches whose last has at most 8 tokens: run, arm, the call's tokens,
                 its micro-batches, the last one's graph and why, and the input copy each micro-batch used
"""
import collections
import csv
import os
import re
import sys

OUT, RUNS = sys.argv[1], sys.argv[2:]
ARMS = ["pp0", "pp1", "off", "base"]
SWITCHES = {
    "pp0": "LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=0",
    "pp1": "LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=1",
    "off": "-",
    "base": "LLAMA_SCHED_PER_SIZE=1 (base build)",
}

R_SCHED = re.compile(r"sched-trace: copy = (\d+) ")
R_UB = re.compile(r"ubatch-trace: (\w+) n_tokens = (\d+) n_seqs = \d+ n_outputs = (\d+) graph = (\w+)(?: why = (\w+))?")
R_DEC = re.compile(r"decode-trace: (\w+) n_tokens = (\d+) ")
R_MADE = re.compile(r"process_ubatch_graph: (scheduler for \d+-token batches .*)$")
R_COPIES = re.compile(r"(\d+) input copies")


def parse(log):
    made, small, tails = [], collections.Counter(), []
    copies_all = collections.Counter()
    pp_line = 0
    last = None  # the kind of the previous trace line
    copy = None
    call = None  # [call tokens, [(tokens, outputs, graph, why, copy), ...]]

    def close(c):
        if c and len(c[1]) > 1 and c[1][-1][0] <= 8:
            tails.append(c)

    with open(log, errors="replace") as f:
        for line in f:
            if "pipeline parallelism enabled" in line:
                pp_line += 1
            m = R_MADE.search(line)
            if m:
                made.append(m.group(1).strip())
                continue
            m = R_SCHED.search(line)
            if m:
                copy = int(m.group(1))
                copies_all[copy] += 1
                last = "sched"
                continue
            m = R_DEC.search(line)
            if m:
                close(call)
                call = [int(m.group(2)), []] if m.group(1) == "dec" else None
                last = "dec"
                continue
            m = R_UB.search(line)
            if m:
                kind, n, outs, graph, why = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5) or ""
                c = copy if last == "sched" else None
                if kind == "dec":
                    if n <= 8:
                        small[(n, outs, graph, why, c)] += 1
                    if call is not None:
                        call[1].append((n, outs, graph, why, c))
                last = "ub"
                copy = None
    close(call)
    return made, small, tails, copies_all, pp_line


def main():
    os.makedirs(OUT, exist_ok=True)
    tok_rows, arm_rows, small_rows, tail_rows, persize = [], [], [], [], []
    for run_dir in RUNS:
        run = os.path.basename(run_dir.rstrip("/"))
        for arm in ARMS:
            d = os.path.join(run_dir, arm)
            if not os.path.isdir(d):
                continue
            with open(os.path.join(d, "session.tsv")) as f:
                for r in csv.reader(f, delimiter="\t"):
                    tok_rows.append([run, arm] + r[:3] + r[4:8])
            made, small, tails, copies_all, pp_line = parse(os.path.join(d, "server.log"))
            mem = []
            if os.path.exists(os.path.join(d, "mem.csv")):
                with open(os.path.join(d, "mem.csv")) as f:
                    mem = [ln.split(",")[1].strip() for ln in f if ln.strip()]
            copies = sorted({int(R_COPIES.search(x).group(1)) for x in made if R_COPIES.search(x)})
            by_copy = collections.Counter()
            in_flight = 0
            for (n, outs, graph, why, c), k in small.items():
                by_copy[c] += k
                in_flight += k if why == "in_flight" else 0
                small_rows.append([run, arm, n, outs, graph + (f" ({why})" if why else ""), "" if c is None else c, k])
            for c in tails:
                ubs = c[1]
                tail_rows.append([run, arm, c[0], "+".join(str(u[0]) for u in ubs),
                                  ubs[-1][2] + (f" ({ubs[-1][3]})" if ubs[-1][3] else ""),
                                  ",".join("" if u[4] is None else str(u[4]) for u in ubs)])
            arm_rows.append([
                run, arm, SWITCHES[arm], pp_line, len(made), ",".join(map(str, copies)) or "-",
                sum(small.values()), " ".join(f"{c}:{by_copy[c]}" for c in sorted(by_copy, key=lambda x: (x is None, x))),
                in_flight, len(tails), " / ".join(mem),
            ])
            persize += [f"{run} {arm}: {x}" for x in made]
    with open(os.path.join(OUT, "tokens.tsv"), "w") as f:
        f.write("run\tarm\tround\trequest\tkind\tgenerated_tokens\tprompt_tokens_read\tcached_tokens\ttokens_md5_12\n")
        for r in tok_rows:
            f.write("\t".join(map(str, r)) + "\n")
    groups = collections.OrderedDict()
    for r in tok_rows:
        groups.setdefault(r[3], []).append(r[8])
    with open(os.path.join(OUT, "same.tsv"), "w") as f:
        f.write("request\treplies\tdistinct_token_lists\ttokens_md5_12\n")
        for req, hs in groups.items():
            f.write(f"{req}\t{len(hs)}\t{len(set(hs))}\t{','.join(sorted(set(hs)))}\n")
    with open(os.path.join(OUT, "arms.tsv"), "w") as f:
        f.write("run\tarm\tswitches\tpipeline_parallelism_line\tper_size_schedulers_made\tinput_copies\t"
                "passes_le8_tokens\tby_input_copy\tle8_rebuilt_in_flight\tcalls_ending_le8_behind_others\tMiB_per_card\n")
        for r in arm_rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(OUT, "small.tsv"), "w") as f:
        f.write("run\tarm\ttokens\toutputs\tgraph\tinput_copy\tpasses\n")
        for r in sorted(small_rows, key=lambda r: (r[0], ARMS.index(r[1]), r[2], r[3], r[4], str(r[5]))):
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(OUT, "tails.tsv"), "w") as f:
        f.write("run\tarm\tcall_tokens\tmicro_batches\tlast_graph\tinput_copies\n")
        for r in tail_rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(OUT, "persize.txt"), "w") as f:
        f.write("\n".join(persize) + "\n")
    print(open(os.path.join(OUT, "same.tsv")).read())
    print(open(os.path.join(OUT, "arms.tsv")).read())
    print(open(os.path.join(OUT, "tails.tsv")).read())


main()
