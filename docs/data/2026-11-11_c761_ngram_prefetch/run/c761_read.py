#!/usr/bin/env python3
"""C761 (Q1339): the readouts of llama-server loads run under LLAMA_UBATCH_TRACE=1 with the n-gram row prefetch off and
on: each prompt micro-batch's inputs split per graph input, each call's n-gram rows (helper and inline), and the host's
memory and swap per load.

Usage: python3 -I c761_read.py <out dir> <arm>=<server.log>[,<vm.tsv>] ...

A request starts at the server's `launch_slot_ ... processing task` line (request 0 holds the load and the warm-up). A
main-model decode call is a `decode-trace: dec` line; the `ubatch-trace: dec` lines after it are its micro-batches, and
its `ngram-prefetch: call` line ends it. A call of more than 8 tokens is a prompt call. In each ubatch-trace line,
`inputs_by` gives every input type's time in set_input (us; qwen4exp_ple is the n-gram input) and `ngram` the n-gram
input's own prefetch call (us), the rows it issued there and the rows the helper had issued. vm.tsv is vm_sampler.sh's.

Writes to <out dir>:
  ubatch.tsv    every prompt micro-batch: arm, request, call, micro-batch, tokens, graph, pre, build, alloc, inputs,
                the n-gram input's time and the other inputs' time, the n-gram prefetch call's time and rows (inline,
                helper), submit, card0, total (us)
  inputs_by.tsv every prompt micro-batch's inputs per input type (us)
  calls.tsv     every main-model call with an ngram-prefetch line: arm, request, call, its fields
  requests.tsv  per arm and request with a prompt: prompt tokens and ms (the server's), the prompt calls' sizes, and over
                the micro-batches of at least 128 tokens: their count, the sums of inputs, the n-gram input, its prefetch
                call and card0, the medians of inputs and of the n-gram input for each call's first micro-batch and for
                the later ones, the helper's and set_input's rows, the most bytes the helper had issued ahead
  vm.tsv        per arm: MemAvailable's lowest (MiB), swap used at the start, highest and end (MiB), and the swap-in,
                swap-out (MiB) and major faults over the load
"""
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_CALL = re.compile(r"decode-trace: (\w+) n_tokens = (\d+) n_outputs = (\d+)")
RE_UB = re.compile(r"ubatch-trace: (\w+) n_tokens = (\d+) n_seqs = (\d+) n_outputs = (\d+) graph = (\w+)(?: why = (\S+))? "
                   r"pre = (-?\d+) build = (-?\d+) alloc = (-?\d+) inputs = (-?\d+) submit = (-?\d+) card0 = (-?\d+) "
                   r"total = (-?\d+) us(?: inputs_by = (.*))?$")
RE_NG = re.compile(r"ngram-prefetch: call n_tokens = (\d+) ubatches = (\d+) (\w+) rows = (\d+) helper = (\d+) inline = (\d+) "
                   r"mismatch = (\d+) pages = (\d+) bytes = (\d+) madv_bytes = (\d+) digest = ([0-9a-f]+) window_max = (\d+) "
                   r"cap = (\d+) helper_us = (-?\d+) inline_us = (-?\d+)")
NG_FIELDS = ["n_tokens", "ubatches", "mode", "rows", "helper", "inline", "mismatch", "pages", "bytes", "madv_bytes", "digest",
             "window_max", "cap", "helper_us", "inline_us"]
RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def med(v):
    return statistics.median(v) if v else ""


def parse_by(s):
    """'a:1,b:2x3 ngram = 5/6/7' -> ({a: 1, b: 2}, (5, 6, 7) or None)"""
    ng = None
    if " ngram = " in s:
        s, t = s.split(" ngram = ", 1)
        ng = tuple(int(x) for x in t.strip().split("/"))
    by = {}
    for part in filter(None, s.strip().split(",")):
        name, us = part.rsplit(":", 1)
        by[name] = int(us.split("x")[0])
    return by, ng


def read_log(arm, path):
    reqs = [dict(req=0, prompt_n="", prompt_ms="", calls=[])]
    cur = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw).rstrip("\n")
        if "launch_slot_" in line and "processing task" in line:
            reqs.append(dict(req=len(reqs), prompt_n="", prompt_ms="", calls=[]))
            cur = None
        r = reqs[-1]
        m = RE_PROMPT.search(line)
        if m:
            r["prompt_ms"], r["prompt_n"] = m.group(1), m.group(2)
            continue
        m = RE_CALL.search(line)
        if m:
            if m.group(1) == "dec":
                cur = dict(n_tokens=int(m.group(2)), ub=[], ng=None)
                r["calls"].append(cur)
            continue
        m = RE_UB.search(line)
        if m:
            if m.group(1) == "dec" and cur is not None:
                by, ng = parse_by(m.group(14) or "")
                cur["ub"].append(dict(n=int(m.group(2)), graph=m.group(5), pre=int(m.group(7)), build=int(m.group(8)),
                                      alloc=int(m.group(9)), inputs=int(m.group(10)), submit=int(m.group(11)),
                                      card0=int(m.group(12)), total=int(m.group(13)), by=by, ng=ng))
            continue
        m = RE_NG.search(line)
        if m and cur is not None:
            cur["ng"] = dict(zip(NG_FIELDS, m.groups()))
    return reqs


def read_vm(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            v = line.rstrip("\n").split("\t")
            if len(v) == len(head) and "-" not in v:
                rows.append({k: float(x) for k, x in zip(head, v)})
    if not rows:
        return None
    used = [(r["swap_total_kb"] - r["swap_free_kb"]) / 1024 for r in rows]
    return dict(samples=len(rows), seconds=f"{rows[-1]['wall'] - rows[0]['wall']:.1f}",
                mem_available_min_mib=f"{min(r['mem_available_kb'] for r in rows) / 1024:.1f}",
                swap_start_mib=f"{used[0]:.1f}", swap_max_mib=f"{max(used):.1f}", swap_end_mib=f"{used[-1]:.1f}",
                swap_in_mib=f"{(rows[-1]['pswpin'] - rows[0]['pswpin']) * 4 / 1024:.1f}",
                swap_out_mib=f"{(rows[-1]['pswpout'] - rows[0]['pswpout']) * 4 / 1024:.1f}",
                major_faults=int(rows[-1]["pgmajfault"] - rows[0]["pgmajfault"]))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fu = open(f"{out}/ubatch.tsv", "w", encoding="utf-8")
    fu.write("arm\treq\tcall\tub\tn_tokens\tgraph\tpre\tbuild\talloc\tinputs\tngram_input\tother_inputs\tngram_prefetch_us\t"
             "ngram_rows_inline\tngram_rows_helper\tsubmit\tcard0\ttotal\n")
    fb = open(f"{out}/inputs_by.tsv", "w", encoding="utf-8")
    fb.write("arm\treq\tcall\tub\tn_tokens\tinput\tus\n")
    fc = open(f"{out}/calls.tsv", "w", encoding="utf-8")
    fc.write("arm\treq\tcall\t" + "\t".join(NG_FIELDS) + "\n")
    fr = open(f"{out}/requests.tsv", "w", encoding="utf-8")
    fr.write("arm\treq\tprompt_n\tprompt_ms\tprompt_calls\tubatches_128\tinputs_sum_ms\tngram_input_sum_ms\t"
             "ngram_prefetch_sum_ms\tcard0_sum_ms\tinputs_med_first_ms\tinputs_med_later_ms\tngram_input_med_first_ms\t"
             "ngram_input_med_later_ms\trows_helper\trows_inline\trows_mismatch\twindow_max_mib\n")
    fv = open(f"{out}/vm.tsv", "w", encoding="utf-8")
    vm_cols = ["samples", "seconds", "mem_available_min_mib", "swap_start_mib", "swap_max_mib", "swap_end_mib",
               "swap_in_mib", "swap_out_mib", "major_faults"]
    fv.write("arm\t" + "\t".join(vm_cols) + "\n")

    for spec in sys.argv[2:]:
        arm, paths = spec.split("=", 1)
        log, _, vm = paths.partition(",")
        for r in read_log(arm, log):
            prompt_calls = [c for c in r["calls"] if c["n_tokens"] > 8]
            for ci, c in enumerate(r["calls"]):
                if c["ng"]:
                    fc.write(f"{arm}\t{r['req']}\t{ci}\t" + "\t".join(c["ng"][k] for k in NG_FIELDS) + "\n")
            if not prompt_calls:
                continue
            big = []
            for ci, c in enumerate(prompt_calls):
                for ui, u in enumerate(c["ub"]):
                    ple = u["by"].get("qwen4exp_ple", "")
                    ng = u["ng"] or ("", "", "")
                    other = u["inputs"] - ple if ple != "" else ""
                    fu.write("\t".join(str(x) for x in [arm, r["req"], ci, ui, u["n"], u["graph"], u["pre"], u["build"],
                                                        u["alloc"], u["inputs"], ple, other, ng[0], ng[1], ng[2],
                                                        u["submit"], u["card0"], u["total"]]) + "\n")
                    for name, us in u["by"].items():
                        fb.write(f"{arm}\t{r['req']}\t{ci}\t{ui}\t{u['n']}\t{name}\t{us}\n")
                    if u["n"] >= 128:
                        big.append((ui == 0, u, ple, ng))
            first = [b for b in big if b[0]]
            later = [b for b in big if not b[0]]
            ngs = [c["ng"] for c in prompt_calls if c["ng"]]
            fr.write("\t".join(str(x) for x in [
                arm, r["req"], r["prompt_n"], r["prompt_ms"], " ".join(str(c["n_tokens"]) for c in prompt_calls), len(big),
                f"{sum(b[1]['inputs'] for b in big) / 1e3:.1f}",
                f"{sum(b[2] for b in big if b[2] != '') / 1e3:.1f}",
                f"{sum(b[3][0] for b in big if b[3][0] != '' and b[3][0] >= 0) / 1e3:.1f}",
                f"{sum(b[1]['card0'] for b in big if b[1]['card0'] >= 0) / 1e3:.1f}",
                f"{med([b[1]['inputs'] for b in first]) / 1e3:.1f}" if first else "",
                f"{med([b[1]['inputs'] for b in later]) / 1e3:.1f}" if later else "",
                f"{med([b[2] for b in first if b[2] != '']) / 1e3:.1f}" if first and first[0][2] != "" else "",
                f"{med([b[2] for b in later if b[2] != '']) / 1e3:.1f}" if later and later[0][2] != "" else "",
                sum(int(n["helper"]) for n in ngs), sum(int(n["inline"]) for n in ngs),
                sum(int(n["mismatch"]) for n in ngs),
                f"{max((int(n['window_max']) for n in ngs), default=0) / 1048576:.1f}"]) + "\n")
        if vm:
            v = read_vm(vm)
            if v:
                fv.write(arm + "\t" + "\t".join(str(v[k]) for k in vm_cols) + "\n")
    for f in (fu, fb, fc, fr, fv):
        f.close()


if __name__ == "__main__":
    main()
