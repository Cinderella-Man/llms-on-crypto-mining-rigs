#!/usr/bin/env python3
"""C615 (Q1100): the readout of one llama-server load's log: every main-model evaluation whose n-gram input set rows
(its `draft-prefetch:` line, logged with LLAMA_UBATCH_TRACE or LLAMA_DRAFT_PREFETCH_STATS=1), paired with the same
evaluation's `sched-trace:` and `ubatch-trace: dec` lines when C701's LLAMA_UBATCH_TRACE is on. The n-gram input sets
its rows while the micro-batch's inputs are set, so for one micro-batch the lines come in this order: draft-prefetch,
sched-trace, ubatch-trace.

Usage: python3 -I c615_read.py <server.log> <out dir> [--load NAME]

Writes to <out dir>:
  evals.tsv    one row per evaluation: n_tokens; rows, ahead (rows a drafter had issued in the window), pages, resident
               (pages in the page cache once the input set its rows, -1 = not checked) and the window's prefetch calls,
               tokens, rows issued, rows already issued, tokens over the cap, page bytes and the calls' time; the
               counting's own time (us); the evaluation's card0 (its start to its first launch off the CPU), the CPU
               split's w, c and l and its first node, the first card split's w and c; the micro-batch's inputs, submit
               and card0
  summary.tsv  per kind: verify (2 to 8 tokens), verify4 (4 tokens), one (1 token), prompt (more than 8): evaluations,
               the medians of the CPU split's l, of card0, of the card split's c, of inputs and of the calls' time, the
               rows ahead / rows, the resident pages / pages checked; and the context's C615 line
"""
import os
import re
import statistics
import sys

PF = re.compile(r"draft-prefetch: (.*)$")
SCHED = re.compile(r"sched-trace: (.*?) \| (.*)$")
UB = re.compile(r"ubatch-trace: dec (.*)$")
SPLIT = re.compile(r"^(\S+?):i(\d+):n(\d+):w(-?\d+):c(-?\d+):l(-?\d+)(?::(\S+))?$")
KV = re.compile(r"(\w+) = (-?\d+)")
C615 = re.compile(r"(C615: .*)$")

COLS = ["eval", "n_tokens", "rows", "ahead", "pages", "resident", "calls", "tokens", "issued", "dup", "capped",
        "bytes", "call_us", "us", "card0", "cpu_w", "cpu_c", "cpu_l", "cpu_node", "card_w", "card_c", "ub_inputs",
        "ub_submit", "ub_card0"]


def kv(s):
    return {k: int(v) for k, v in KV.findall(s)}


def med(xs):
    xs = [x for x in xs if x is not None and x != ""]
    return round(statistics.median(xs), 1) if xs else ""


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    log, out = args[0], args[1]
    load = args[args.index("--load") + 1] if "--load" in args else os.path.basename(os.path.dirname(log))
    os.makedirs(out, exist_ok=True)

    rows, cur, c615 = [], None, ""

    def emit():
        nonlocal cur
        if cur is not None:
            rows.append(cur)
        cur = None

    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PF.search(line)
            if m:
                emit()
                d = kv(m.group(1))
                cur = {c: "" for c in COLS}
                cur.update({k: d.get(k, "") for k in COLS[1:14]})
                cur["eval"] = len(rows) + 1
                continue
            m = SCHED.search(line)
            if m and cur is not None and cur["card0"] == "":
                head = kv(m.group(1))
                cur["card0"] = head.get("card0", "")
                cpu = card = None
                for s in m.group(2).split():
                    sm = SPLIT.match(s)
                    if not sm:
                        continue
                    be = sm.group(1)
                    if be.startswith("CPU"):
                        cpu = cpu or sm
                    elif card is None:
                        card = sm
                        break
                if cpu:
                    cur.update(cpu_w=int(cpu.group(4)), cpu_c=int(cpu.group(5)), cpu_l=int(cpu.group(6)),
                               cpu_node=cpu.group(7) or "")
                if card:
                    cur.update(card_w=int(card.group(4)), card_c=int(card.group(5)))
                continue
            m = UB.search(line)
            if m and cur is not None and cur["card0"] != "":
                d = kv(m.group(1))
                cur.update(ub_inputs=d.get("inputs", ""), ub_submit=d.get("submit", ""), ub_card0=d.get("card0", ""))
                emit()
                continue
            m = C615.search(line)
            if m:
                c615 = m.group(1)
    emit()

    with open(f"{out}/evals.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLS) + "\n")

    kinds = {
        "verify": lambda r: 2 <= r["n_tokens"] <= 8,
        "verify4": lambda r: r["n_tokens"] == 4,
        "one": lambda r: r["n_tokens"] == 1,
        "prompt": lambda r: r["n_tokens"] > 8,
    }
    scol = ["load", "kind", "evals", "cpu_l_med", "card0_med", "card_c_med", "inputs_med", "call_us_med",
            "rows", "ahead", "pages_checked", "resident"]
    with open(f"{out}/summary.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(scol) + "\n")
        for k, sel in kinds.items():
            rs = [r for r in rows if sel(r)]
            chk = [r for r in rs if isinstance(r["resident"], int) and r["resident"] >= 0]
            v = [load, k, len(rs), med([r["cpu_l"] for r in rs]), med([r["card0"] for r in rs]),
                 med([r["card_c"] for r in rs]), med([r["ub_inputs"] for r in rs]), med([r["call_us"] for r in rs]),
                 sum(r["rows"] for r in rs), sum(r["ahead"] for r in rs), sum(r["pages"] for r in chk),
                 sum(r["resident"] for r in chk)]
            f.write("\t".join(str(x) for x in v) + "\n")
        f.write(f"# {c615}\n")
    print(f"{load}: {len(rows)} evaluations; {c615 or 'no C615 line'}")


if __name__ == "__main__":
    main()
