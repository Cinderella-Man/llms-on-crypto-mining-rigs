#!/usr/bin/env python3
"""C741 (Q1310): the yields to the task queue in llama-server's steps, from LLAMA_STEP_TRACE files.

  python3 -I yields.py [--tsv FILE] LOAD=step_trace.jsonl [LOAD=step_trace.jsonl ...] > yields.md
  python3 -I yields.py --two-slots [--tsv FILE] LOAD=DIR [LOAD=DIR ...] > two_slots.md (one slot generating before B)

Each trace line after the header is one step of the server (C405's trace). The first form takes the generation steps
(the steps that drafted, n.drf > 0) and gives, per load:
  - the yields ("yld") and the queue worker's wakes ("wake") of each step, as the number of steps with each value
    (a build without C741 counts neither: "—");
  - the median of each host stage a yield makes, in ms, over the steps that have both marks:
      s.drf -> s.drf.in        the draft's yield start
      s.drf.out -> s.drf.end   the draft's yield end
      s.dec.y -> s.dec.in      the verify pass's yield start
      s.dec.out -> s.dec.end   the verify pass's yield end
      s.mpd -> s.cu.in         the catch-up's yield start
      s.cu.out -> s.cu.end     the catch-up's yield end
    and the median over steps of the six together.
The second form reads two_slots.py's folders (DIR/step_trace.jsonl, DIR/client.json, DIR/reads.tsv): when B was posted
(the "q.cmpl" mark: the step it came in), the first step after it with B's prompt in its batch (more tokens than A's
sampled token and its draft: n.tok > n.drf + 1), the steps between; the reads the server saw posted in a step ("q.read")
and, paired with them in order, the worker's answers in a yield of the same step ("q.take"): the rest are left to the
step's end, where the loop takes every task in the queue; and for every read the client made, how many steps ended
between its sending and its answer on the client's side. Standard library only.
"""
import json
import statistics
import sys

STAGES = [
    ("s.drf", "s.drf.in"),
    ("s.drf.out", "s.drf.end"),
    ("s.dec.y", "s.dec.in"),
    ("s.dec.out", "s.dec.end"),
    ("s.mpd", "s.cu.in"),
    ("s.cu.out", "s.cu.end"),
]


def read_trace(path):
    t0 = None
    steps = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if i == 0 and "t0_steady_ms" in d:
                t0 = d["t0_steady_ms"]
                continue
            steps.append(d)
    return t0, steps


def stage(marks, a, b):
    ta = tb = None
    for name, t in marks:
        if name == a and ta is None:
            ta = t
        elif name == b and ta is not None and tb is None:
            tb = t
    return None if ta is None or tb is None else tb - ta


def first(marks, name):
    for n, t in marks:
        if n == name:
            return t
    return None


def dist(values):
    if not values:
        return "—"
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return ", ".join(f"{k}: {out[k]}" for k in sorted(out))


def med(v):
    return f"{statistics.median(v):.4f}" if v else "—"


def gen_table(loads, tsv):
    print("| load | steps | yields a step (steps) | wakes a step (steps) | " +
          " | ".join(f"`{a} → {b}`" for a, b in STAGES) + " | the six, median of sums |")
    print("|---|---:|---|---|" + "---:|" * len(STAGES) + "---:|")
    rows = []
    for name, path in loads:
        _, steps = read_trace(path)
        gen = [s for s in steps if s.get("n", {}).get("drf", 0) > 0]
        yld = [s["n"]["yld"] for s in gen if "yld" in s.get("n", {})]
        wake = [s["n"].get("wake", 0) for s in gen if "yld" in s.get("n", {})]
        per = [[] for _ in STAGES]
        sums = []
        for s in gen:
            vals = [stage(s.get("t", []), a, b) for a, b in STAGES]
            for k, v in enumerate(vals):
                if v is not None:
                    per[k].append(v)
            if all(v is not None for v in vals):
                sums.append(sum(vals))
        print(f"| {name} | {len(gen)} | {dist(yld)} | {dist(wake)} | " + " | ".join(med(p) for p in per) +
              f" | {med(sums)} |")
        rows.append([name, len(gen), sum(yld) if yld else "", sum(wake) if yld else "",
                     len(yld)] + [med(p) for p in per] + [med(sums)])
    print()
    print("ms on the trace's steady clock; generation steps are the steps that drafted (n.drf > 0); a stage the build "
          "does not run (no yield there) still has both marks, back to back")
    if tsv:
        with open(tsv, "w") as f:
            f.write("load\tsteps\tyields\twakes\tsteps_counted\t" + "\t".join(f"{a}->{b}" for a, b in STAGES) +
                    "\tsum_median\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")


def two_slots(loads, tsv):
    print("| load | B came during A | B posted in step | its prompt in step | steps between | reads by the client | "
          "failed | reads posted in a step (`q.read`) | of them answered in a yield of that step | left to the step's end | "
          "answered in a yield, posted before the step | step ends between a read's sending and its answer, client side "
          "(reads) | read ms, median | read ms, max | step ms, median | step ms, max |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    rows = []
    for name, d in loads:
        t0, steps = read_trace(d + "/step_trace.jsonl")
        with open(d + "/client.json") as f:
            client = json.load(f)
        reads = []
        with open(d + "/reads.tsv") as f:
            next(f)
            for line in f:
                k, s, r, ms, st, _ = line.rstrip("\n").split("\t")
                reads.append((k, float(s), float(r), float(ms), int(st)))
        i_post = next((i for i, s in enumerate(steps) if first(s.get("t", []), "q.cmpl") is not None), None)
        # B's prompt: the first step after B's post whose batch holds more than A's sampled token and its draft
        i_prompt = None
        if i_post is not None:
            i_prompt = next((i for i in range(i_post + 1, len(steps))
                             if steps[i].get("n", {}).get("tok", 0) > steps[i].get("n", {}).get("drf", 0) + 1), None)
        between = (i_prompt - i_post) if (i_post is not None and i_prompt is not None) else None
        # the reads the server saw posted in a step ("q.read"), paired in order with the worker's answers in a yield of
        # the same step ("q.take"); a read left unpaired is taken by the loop when its step ends, a take left unpaired
        # answers a read posted before its step began (between steps, where the trace keeps no mark)
        n_read = n_paired = n_left = n_early = 0
        for s in steps:
            pending = 0
            for n, _ in s.get("t", []):
                if n == "q.read":
                    n_read += 1
                    pending += 1
                elif n == "q.take":
                    if pending > 0:
                        pending -= 1
                        n_paired += 1
                    else:
                        n_early += 1
            n_left += pending
        ends = []
        begins = []
        for s in steps:
            tb = first(s.get("t", []), "s.upd")
            te = first(s.get("t", []), "s.upd.end")
            if tb is not None and te is not None:
                begins.append(tb)
                ends.append(te)
        dur = [e - b for b, e in zip(begins, ends)]
        crossed = []
        if t0 is not None:
            for k, s, r, ms, st in reads:
                a, b = s - t0, r - t0
                crossed.append(sum(1 for e in ends if a < e < b))
        lat = [r[3] for r in reads if r[4] == 200]
        print(f"| {name} | {'yes' if client.get('b_during_a') else 'NO'} | {i_post if i_post is not None else '—'} | "
              f"{i_prompt if i_prompt is not None else '—'} | {between if between is not None else '—'} | {len(reads)} | "
              f"{client.get('reads_failed', 0)} | {n_read} | {n_paired} | {n_left} | {n_early} | {dist(crossed)} | "
              f"{med(lat)} | {max(lat):.4f} | {med(dur)} | {max(dur):.4f} |" if lat and dur else f"| {name} | (no data) |")
        rows.append([name, int(bool(client.get("b_during_a"))), i_post, i_prompt, between, len(reads),
                     client.get("reads_failed", 0), n_read, n_paired, n_left, n_early, max(crossed) if crossed else "",
                     dist(crossed), med(lat), f"{max(lat):.4f}" if lat else "", med(dur), f"{max(dur):.4f}" if dur else ""])
    print()
    print("Steps are the trace's lines, counted from 0; a step ends at its s.upd.end mark. The server's columns come from "
          "its marks (a build without them shows 0); the loop takes every task left in the queue when a step ends. Reads "
          "are the client's /slots and /metrics GETs while A or B ran, their times on the client (the HTTP threads' and "
          "the client's own delays included); ms on the steady clock")
    if tsv:
        with open(tsv, "w") as f:
            f.write("load\tb_during_a\tb_post_step\tb_prompt_step\tsteps_between\treads\treads_failed\tq_read\t"
                    "answered_in_step\tleft_to_step_end\tanswered_posted_before\tmax_step_ends_crossed\tstep_ends_crossed\t"
                    "read_ms_median\tread_ms_max\tstep_ms_median\tstep_ms_max\n")
            for r in rows:
                f.write("\t".join("" if x is None else str(x) for x in r) + "\n")


def main(argv):
    tsv = None
    mode = "gen"
    loads = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--tsv":
            tsv = argv[i + 1]
            i += 2
            continue
        if a == "--two-slots":
            mode = "two"
        elif "=" in a:
            loads.append(tuple(a.split("=", 1)))
        else:
            print(__doc__)
            return 2
        i += 1
    if not loads:
        print(__doc__)
        return 2
    if mode == "two":
        two_slots(loads, tsv)
    else:
        gen_table(loads, tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
