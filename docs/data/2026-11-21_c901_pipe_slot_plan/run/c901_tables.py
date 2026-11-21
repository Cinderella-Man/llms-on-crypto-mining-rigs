#!/usr/bin/env python3
"""c901_tables.py: the server check's tables from run/c901_server.sh's output folder.

    python3 -I c901_tables.py OUTDIR SERVER_OUT [ARM ...]

SERVER_OUT is c901_server.sh's OUT/out (ARM.server.log and ARM.Qn.json per arm); the arms default to cpu_off cpu_on
rpc_off rpc_on. Writes:
  OUTDIR/server_tokens.tsv   per arm and request: tokens generated, the server's prompt tokens, the md5 of the token list
                             (its first 12 hex digits, as the check's console prints them)
  OUTDIR/server_replans.tsv  per arm and request (in the order of the log): the pipe's guesses, the pipe slots' re-plans
                             after no change of device and after one, the plans over the whole memory (from the server's
                             running counts, `pipe reallocs = S / D ..., F plans`), and the context's own graph reallocs
  OUTDIR/server_trace.tsv    per arm: the trace's sched-replan lines (all, after a change of device), their drain and plan
                             time summed (us), and the pipe_slot_plan_full lines by outcome and their time summed (ms)
Standard library only.
"""
import hashlib
import json
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TASK = re.compile(r"\| task ([0-9]+) \|")
GUESS = re.compile(r"pipelined =\s*([0-9]+) guesses, ([0-9]+) right, ([0-9]+) wrong")
GR = re.compile(r"graph reallocs =\s*([0-9]+)\s*/\s*([0-9]+)")
PR = re.compile(r"pipe reallocs =\s*([0-9]+)\s*/\s*([0-9]+).*?,\s*([0-9]+) plans")
REPLAN = re.compile(r"sched-replan: #[0-9]+ backends_changed = ([01]).* drain = ([0-9]+) plan = ([0-9]+) us")
PLAN = re.compile(r"pipe_slot_plan_full: slot ([0-9]+) re-planned( after a change of device)?; .*: ([0-9]+) nodes, "
                  r"([0-9.]+) ms, (.*) \(plan [0-9]+\)")


def main():
    outdir, sdir = sys.argv[1], sys.argv[2]
    arms = sys.argv[3:] or ["cpu_off", "cpu_on", "rpc_off", "rpc_on"]
    os.makedirs(outdir, exist_ok=True)
    tok = ["arm\trequest\ttokens\tprompt_n\ttokens_md5"]
    rep = ["arm\trequest\ttask\tguesses\tguesses_right\tpipe_replans_same_devices\tpipe_replans_changed_devices\t"
           "plans_whole_memory\tgraph_reallocs_same\tgraph_reallocs_changed"]
    tr = ["arm\tsched_replan_lines\tafter_device_change\tdrain_us\tplan_us\tplan_full_lines\tplan_full_fits\t"
          "plan_full_given_up\tplan_full_ms"]
    for arm in arms:
        for q in ("Q1", "Q2", "Q3", "Q4"):
            p = os.path.join(sdir, f"{arm}.{q}.json")
            try:
                d = json.load(open(p))
                toks = d.get("tokens") or []
                # as the check's console prints it: `jq -c .tokens | md5sum`, the compact list and a newline
                md5 = hashlib.md5((json.dumps(toks, separators=(",", ":")) + "\n").encode()).hexdigest()[:12]
                tok.append(f"{arm}\t{q}\t{len(toks)}\t{d.get('timings', {}).get('prompt_n', '')}\t{md5}")
            except (OSError, ValueError):
                tok.append(f"{arm}\t{q}\t\t\tno reply")
        log = os.path.join(sdir, f"{arm}.server.log")
        prev_p, prev_g, cur, i = (0, 0, 0), (0, 0), {}, 0
        n_rp = n_dev = drain = plan = 0
        n_pf = n_fit = n_off = 0
        pf_ms = 0.0
        for raw in open(log, errors="replace"):
            line = ANSI.sub("", raw)
            m = REPLAN.search(line)
            if m:
                n_rp += 1
                n_dev += int(m.group(1))
                drain += int(m.group(2))
                plan += int(m.group(3))
                continue
            m = PLAN.search(line)
            if m:
                n_pf += 1
                pf_ms += float(m.group(4))
                if "fits it" in m.group(5):
                    n_fit += 1
                else:
                    n_off += 1
                continue
            t = TASK.search(line)
            m = GUESS.search(line)
            if m and t:
                cur = dict(task=t.group(1), guesses=m.group(1), right=m.group(2))
                continue
            m = GR.search(line)
            if m and t:
                g = (int(m.group(1)), int(m.group(2)))
                cur.setdefault("task", t.group(1))
                cur["gr"] = (g[0] - prev_g[0], g[1] - prev_g[1])
                prev_g = g
                continue
            m = PR.search(line)
            if m and t:
                c = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                d = [x - y for x, y in zip(c, prev_p)]
                prev_p = c
                i += 1
                gr = cur.get("gr", ("", ""))
                rep.append(f"{arm}\tQ{i}\t{cur.get('task', t.group(1))}\t{cur.get('guesses', '')}\t{cur.get('right', '')}\t"
                           f"{d[0]}\t{d[1]}\t{d[2]}\t{gr[0]}\t{gr[1]}")
                cur = {}
        tr.append(f"{arm}\t{n_rp}\t{n_dev}\t{drain}\t{plan}\t{n_pf}\t{n_fit}\t{n_off}\t{pf_ms:.3f}")
    for name, rows in (("server_tokens.tsv", tok), ("server_replans.tsv", rep), ("server_trace.tsv", tr)):
        with open(os.path.join(outdir, name), "w") as f:
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
