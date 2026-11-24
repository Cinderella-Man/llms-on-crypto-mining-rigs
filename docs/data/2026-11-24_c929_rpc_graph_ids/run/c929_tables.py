#!/usr/bin/env python3
"""c929_tables.py: C929's tables from run.sh's work folder (or a copy of it without sources, builds and models).

    python3 -I c929_tables.py OUTDIR WORK

Reads WORK/server-<build>/out (c929_server.sh's output per build: ARM.Qn.json, ARM.server.log, ARM.rpc-PORT.log),
WORK/server-<build>/console.txt, WORK/cpu/summary.tsv (cpu_tests.sh's verdicts), WORK/checks/ctest.tsv and
WORK/rpc-<build>/RC. Writes:
  OUTDIR/server_tokens.tsv  per build, arm and request: tokens generated, the server's prompt tokens, the md5 of the
                            token list (first 12 hex digits, as the check's console prints them), and whether they are
                            head's cpu_off tokens for that request
  OUTDIR/server_rpc.tsv     per build and arm: llama-server alive after the four requests, its aborts ("Remote RPC server
                            crashed"), the trace's re-plans and buffer growths, and per RPC server the commands its log
                            lists: graphs sent whole (graph_compute), of them under a uid that server had run before (sent
                            whole again after a drop), graphs run by their uid (graph_recompute), buffers allocated and
                            freed, and uids it rejected ("graph with uid ... not found")
  OUTDIR/checks.tsv         cpu_tests.sh's verdicts (build, ctest, test-llama-archs, head's logits against base's) and
                            the three server builds' exit codes
  OUTDIR/ctest.tsv          ctest test by test, base and head
Standard library only.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BUILDS = ["head", "base", "c901"]
ARMS = ["cpu_off", "cpu_on", "rpc_noreuse_off", "rpc_noreuse_on", "rpc_reuse_off", "rpc_reuse_on"]
UID = re.compile(r"\[graph_compute\] device: [0-9]+, uid: ([0-9]+),")


def tokens_md5(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None, "", ""
    toks = d.get("tokens") or []
    # as the console prints it: `jq -c .tokens | md5sum`, the compact list and a newline
    md5 = hashlib.md5((json.dumps(toks, separators=(",", ":")) + "\n").encode()).hexdigest()[:12]
    return md5, len(toks), d.get("timings", {}).get("prompt_n", "")


def rpc_counts(path):
    n = dict(compute=0, again=0, recompute=0, alloc=0, free=0, rejected=0)
    seen = set()
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw)
        m = UID.search(line)
        if m:
            n["compute"] += 1
            u = m.group(1)
            if u != "0" and u in seen:
                n["again"] += 1
            seen.add(u)
        elif "[graph_recompute] device:" in line:
            n["recompute"] += 1
        elif "[alloc_buffer] device:" in line:
            n["alloc"] += 1
        elif "[free_buffer] remote_ptr" in line:
            n["free"] += 1
        if re.search(r"graph with uid [0-9]+ not found", line):
            n["rejected"] += 1
    return n


def main():
    outdir, work = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    tok = ["build\tarm\trequest\ttokens\tprompt_n\ttokens_md5\tsame_as_head_cpu_off"]
    rpc = ["build\tarm\tllama_server_alive\tclient_aborts\tsched_replans\tgalloc_grow\trpc_server\tgraph_compute\t"
           "sent_whole_again\tgraph_recompute\talloc_buffer\tfree_buffer\trejected_uids"]
    ref = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        ref[q] = tokens_md5(os.path.join(work, "server-head", "out", f"cpu_off.{q}.json"))[0]
    for b in BUILDS:
        sdir = os.path.join(work, f"server-{b}", "out")
        console = os.path.join(work, f"server-{b}", "console.txt")
        alive = {}
        if os.path.exists(console):
            for line in open(console, errors="replace"):
                m = re.match(r"^(\S+): env .*alive after the four requests: (yes|no)", line)
                if m:
                    alive[m.group(1)] = m.group(2)
        for arm in ARMS:
            log = os.path.join(sdir, f"{arm}.server.log")
            if not os.path.exists(log):
                continue
            for q in ("Q1", "Q2", "Q3", "Q4"):
                md5, n, pn = tokens_md5(os.path.join(sdir, f"{arm}.{q}.json"))
                if md5 is None:
                    tok.append(f"{b}\t{arm}\t{q}\t\t\tno reply\tno")
                else:
                    tok.append(f"{b}\t{arm}\t{q}\t{n}\t{pn}\t{md5}\t{'yes' if md5 == ref[q] else 'no'}")
            text = ANSI.sub("", open(log, errors="replace").read())
            aborts = text.count("Remote RPC server crashed")
            replans = text.count("sched-replan:")
            grows = text.count("galloc-grow:")
            ports = sorted(re.sub(r"^.*\.rpc-([0-9]+)\.log$", r"\1", p)
                           for p in glob.glob(os.path.join(sdir, f"{arm}.rpc-*.log")))
            if not ports:
                rpc.append(f"{b}\t{arm}\t{alive.get(arm, '')}\t{aborts}\t{replans}\t{grows}\t\t\t\t\t\t\t")
            for p in ports:
                c = rpc_counts(os.path.join(sdir, f"{arm}.rpc-{p}.log"))
                rpc.append(f"{b}\t{arm}\t{alive.get(arm, '')}\t{aborts}\t{replans}\t{grows}\t{p}\t{c['compute']}\t"
                           f"{c['again']}\t{c['recompute']}\t{c['alloc']}\t{c['free']}\t{c['rejected']}")
    open(os.path.join(outdir, "server_tokens.tsv"), "w").write("\n".join(tok) + "\n")
    open(os.path.join(outdir, "server_rpc.tsv"), "w").write("\n".join(rpc) + "\n")

    chk = ["tree\tcheck\tverdict\tdetail"]
    summ = os.path.join(work, "cpu", "summary.tsv")
    if os.path.exists(summ):
        chk += [l for l in open(summ).read().splitlines()[1:] if l]
    for b in BUILDS:
        rc = os.path.join(work, f"rpc-{b}", "RC")
        if os.path.exists(rc):
            line = open(rc).read().strip()
            m = re.search(r"build rc=([0-9]+) s=([0-9]+); (.*)$", line)
            if m:
                chk.append(f"rpc-{b}\tbuild\t{'OK' if m.group(1) == '0' else 'FAIL'}\trc={m.group(1)}; {m.group(2)} s; "
                           f"llama-server ggml-rpc-server test-llama-archs; {m.group(3).strip()}")
    open(os.path.join(outdir, "checks.tsv"), "w").write("\n".join(chk) + "\n")
    ct = os.path.join(work, "checks", "ctest.tsv")
    if os.path.exists(ct):
        shutil.copyfile(ct, os.path.join(outdir, "ctest.tsv"))


if __name__ == "__main__":
    main()
