#!/usr/bin/env python3
"""fn85.py: where a served reply's generation time goes outside its profiled steps (FN85, Q1410), from R122's raw
files: each load's GGML_CUDA_PROFILE_HOST file and its llama-server log.

  python3 -I fn85.py OUT --loads base,a,...,f-2 --step-split STEP_SPLIT_DIR [--tsv-dir DIR] > fn85.md

OUT/<load>/host_profile.csv and OUT/<load>/server-<load>.log (R122's run/r122_loads.sh). STEP_SPLIT_DIR holds S37's
step_split.py, whose cutting into requests, main passes and steps this script reuses, so that "the rest" here is
R122's ladder.md column of that name.

The server's generation time (predicted_ms) runs from the first token's sample (T_s) to the acceptance after the
reply's last verify pass (T_end), and its prompt time from the slot's start to T_s (tools/server/server-common.h,
server_slot_stats). Its log and the profile have different zeros, so the slot's start is placed on the profile's clock
by an offset: the slot starts `delta` ms before the request's first evaluation, T_s = first entry - delta + prompt_ms,
T_end = T_s + predicted_ms. delta is found from the server's progress lines ("n_gen = N, tg = R t/s", printed every
3 s right after a step's acceptance): in a generation step the server accepts right after the draft's catch-up over
the verified tokens (the first evaluation on the draft's last card with more than 1 ms of GPU time after a main
pass), so each line's log time, taken from the slot's start, must equal (that catch-up's end + delta - first entry)
for some step, within --tol ms. The delta that holds for every line of the request, and puts T_end no earlier than
--tol ms before the request's last evaluation ends, is used; the largest such if several. "anchor" says how it was
found; "fallback" (no line fits) puts T_end 3 ms after the request's last evaluation, which is where it falls in
every request that has a fit.

Waits. The fork's keep-alive (GGML_CUDA_KEEPALIVE, keepalive.cu) runs one block per card until no graph has been
computed on the card for GGML_CUDA_KEEPALIVE_MS (default 1,000 ms); the controller beats the card's block right after
each evaluation's exit. An evaluation is "timeout-ended" when it ends between 1,000 and 1,000 + --win ms after its
card's last beat (the latest exit of an earlier evaluation on that card, or its own exit when the host left it before
the GPU finished): it waited for the block to end by itself. --timeout sets the block's timeout when the run set
GGML_CUDA_KEEPALIVE_MS (R128's fka250: 250), --min-wait the shortest wait counted (default 300 ms; R128 uses 100 with
--timeout 250). Its wait runs from its entry (or from the start of a host
idle of 100 ms or more right before it, when one precedes it) to its end. An evaluation ends at entry + the longer of
its host span and its GPU time.

Where the outside time goes: "outside" = predicted_ms - the rest = head + tail, head = [T_s, M_1) (M_1: the entry of
the main pass that starts step_split's step 1, the first step of "the rest"), tail = [M_n, T_end] (M_n: the entry of
the request's last main pass). Each is split into the waits inside it, host idles of 100 ms or more that are not part
of a wait, and the remainder (the draft's passes, step 0's verify, the last verify pass's own work).
"""
import argparse
import csv
import os
import re
import statistics
import sys

LOG_T = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")
TIMEOUT_MS = 1000.0
IDLE_MS = 100.0


def log_time_ms(line):
    m = LOG_T.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return (mi * 60 + s) * 1000.0 + ms + us / 1000.0


def parse_log(path):
    """The server's tasks in order: launch time, prompt_ms, predicted_ms, n_gen, the 3-second progress lines."""
    tasks, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            t = log_time_ms(line)
            if t is None:
                continue
            m = re.search(r"task (\d+) \| processing task", line)
            if m:
                cur = dict(task=int(m.group(1)), launch=t, prompt_ms=None, predicted_ms=None, n_gen=None, tg=[],
                           release=None)
                tasks.append(cur)
                continue
            if cur is None:
                continue
            m = re.search(r"prompt eval time = +([\d.]+) ms", line)
            if m:
                cur["prompt_ms"] = float(m.group(1))
            m = re.search(r" eval time = +([\d.]+) ms / +(\d+) tokens", line)
            if m and "prompt" not in line:
                cur["predicted_ms"], cur["n_gen"] = float(m.group(1)), int(m.group(2))
            m = re.search(r"n_gen = +(\d+), tg = +([\d.]+) t/s", line)
            if m:
                cur["tg"].append((t, int(m.group(1)), float(m.group(2))))
            if "stop processing" in line:
                cur["release"] = t
    return [x for x in tasks if x["predicted_ms"] is not None]


def end_of(r):
    en, ex, dev, gpu = r
    return en + max(ex - en, gpu)


def anchor(s, ps, task, draft_last, tol):
    """delta (ms): the slot's start is delta ms before the request's first evaluation."""
    p0 = s[0][0]
    acc = []
    for j, p in enumerate(ps):
        nxt = ps[j + 1][0] if j + 1 < len(ps) else len(s)
        for i in range(p[1] + 1, nxt):
            if s[i][2] == draft_last and s[i][3] > 1.0:
                acc.append(end_of(s[i]))
                break
    last_end = max(end_of(r) for r in s)
    t_end0 = p0 + task["prompt_ms"] + task["predicted_ms"]
    upper = t_end0 - last_end + tol
    lines = task["tg"]
    if not lines or not acc:
        return t_end0 - last_end - 3.0, "fallback (no progress line)"
    cands = [[p0 + (lt - task["launch"]) - a for a in acc] for (lt, n, tg) in lines]
    ok = [c0 for c0 in cands[0] if 0.0 <= c0 <= upper and all(any(abs(c - c0) <= tol for c in cl) for cl in cands[1:])]
    if not ok:
        return t_end0 - last_end - 3.0, "fallback (no fit)"
    return max(ok), f"{len(lines)} progress lines" + ("" if len(ok) == 1 else f", {len(ok)} fits (the largest)")


# waits.tsv's columns, written as the header alone when no load has a wait (R137, 2026-11-13: EAGER=1 left none)
WAIT_KEYS = ["load", "request", "card", "evaluation", "where", "start_ms", "enter_ms", "host_ms", "gpu_ms", "end_ms",
             "wait_ms", "idle_before_ms", "after_beat_ms", "from_t_s_ms", "to_t_end_ms", "in_gen_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--loads", required=True)
    ap.add_argument("--requests", default="greedy,long")
    ap.add_argument("--step-split", required=True)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_MS)
    ap.add_argument("--min-wait", type=float, default=300.0)
    ap.add_argument("--tsv-dir")
    a = ap.parse_args()
    sys.path.insert(0, a.step_split)
    import step_split as ss  # noqa: E402

    names = a.requests.split(",")
    main_devs = list(range(9))
    req_rows, wait_rows = [], []
    for load in a.loads.split(","):
        d = os.path.join(a.out, load)
        rows = ss.load_rows(os.path.join(d, "host_profile.csv"))
        segs = ss.cut(rows, 2000.0)
        seg_passes = [ss.main_passes(s, main_devs) for s in segs]
        req = [(s, p) for s, p in zip(segs, seg_passes) if len(p) >= 10][-len(names):]
        tasks = parse_log(os.path.join(d, f"server-{load}.log"))[-len(names):]
        between = set()
        for s, ps in req:
            for p, q in zip(ps, ps[1:]):
                between.update(x[2] for x in s[p[1] + 1:q[0]])
        draft_last = max(between)
        for name, (s, ps), task in zip(names, req, tasks):
            st = ss.steps_of(s, ps, draft_last)
            med = statistics.median(x["verify_gpu_ms"] for x in st)
            g0 = 0
            while g0 < len(st) and st[g0]["verify_gpu_ms"] > 1.5 * med:
                g0 += 1
            m1 = s[ps[g0 + 1][0]][0]
            mn = s[ps[-1][0]][0]
            rest = sum(x["step_ms"] for x in st[g0 + 1:])
            delta, how = anchor(s, ps, task, draft_last, a.tol)
            p0 = s[0][0]
            t_s = p0 - delta + task["prompt_ms"]
            t_end = t_s + task["predicted_ms"]
            in_main = {}
            for j, p in enumerate(ps):
                for i in range(p[0], p[1] + 1):
                    in_main[i] = j
            # host idles: stretches of IDLE_MS or more with no evaluation running on any card
            idles, run_end = [], None
            for i, r in enumerate(s):
                if run_end is not None and r[0] - run_end >= IDLE_MS:
                    idles.append((run_end, r[0], i))
                run_end = end_of(r) if run_end is None else max(run_end, end_of(r))
            idle_before = {i: (b, e) for (b, e, i) in idles}
            # timeout-ended evaluations and their waits
            waits = []
            for i, r in enumerate(s):
                en, ex, dev, gpu = r
                end = end_of(r)
                beats = [x[1] for x in s[:i + 1] if x[2] == dev and x[1] <= end - 5.0]
                if not beats:
                    continue
                since = end - max(beats)
                if not (a.timeout <= since <= a.timeout + a.win):
                    continue
                w0 = idle_before[i][0] if i in idle_before else en
                if end - w0 < a.min_wait:
                    continue  # an ordinary evaluation that happens to end near a timeout
                j = in_main.get(i)
                if end <= t_s:
                    where = "prompt"
                elif w0 < m1:
                    where = "head"
                elif w0 < mn:
                    where = "steps"
                elif w0 <= t_end:
                    where = "tail"
                else:
                    where = "after T_end"
                waits.append(dict(load=load, request=name, card=dev,
                                  evaluation=("main pass %d of %d" % (j + 1, len(ps))) if j is not None else "draft",
                                  where=where, start_ms=w0, enter_ms=en, host_ms=ex - en, gpu_ms=gpu, end_ms=end,
                                  wait_ms=end - w0, idle_before_ms=(en - w0), after_beat_ms=since,
                                  from_t_s_ms=w0 - t_s, to_t_end_ms=t_end - end,
                                  in_gen_ms=max(0.0, min(end, t_end) - max(w0, t_s))))
            wspans = [(w["start_ms"], w["end_ms"]) for w in waits]

            def overlap(lo, hi, spans):
                return sum(max(0.0, min(hi, e) - max(lo, b)) for b, e in spans)

            def idle_not_wait(lo, hi):
                tot = 0.0
                for (b, e, i) in idles:
                    if any(wb <= b and e <= we for wb, we in wspans):
                        continue
                    tot += max(0.0, min(hi, e) - max(lo, b))
                return tot

            head_w, tail_w = overlap(t_s, m1, wspans), overlap(mn, t_end, wspans)
            head_i, tail_i = idle_not_wait(t_s, m1), idle_not_wait(mn, t_end)
            req_rows.append(dict(
                load=load, request=name, delta_ms=delta, anchor=how, prompt_ms=task["prompt_ms"],
                predicted_ms=task["predicted_ms"], rest_ms=rest, outside_ms=task["predicted_ms"] - rest,
                head_ms=m1 - t_s, head_waits_ms=head_w, head_idle_ms=head_i, head_other_ms=m1 - t_s - head_w - head_i,
                tail_ms=t_end - mn, tail_waits_ms=tail_w, tail_idle_ms=tail_i, tail_other_ms=t_end - mn - tail_w - tail_i,
                waits_in_gen=sum(1 for w in waits if w["in_gen_ms"] > 0), waits_in_gen_ms=head_w + tail_w,
                waits_in_prompt=sum(1 for w in waits if w["where"] == "prompt"),
                outside_less_waits_ms=task["predicted_ms"] - rest - head_w - tail_w,
                last_eval_to_t_end_ms=t_end - max(end_of(r) for r in s)))
            wait_rows += waits

    print("## Each request: the generation time outside the profiled steps, part by part (ms)")
    print()
    cols = ["load", "request", "delta_ms", "prompt_ms", "predicted_ms", "rest_ms", "outside_ms", "head_ms",
            "head_waits_ms", "head_idle_ms", "head_other_ms", "tail_ms", "tail_waits_ms", "tail_idle_ms",
            "tail_other_ms", "waits_in_gen", "outside_less_waits_ms", "waits_in_prompt", "last_eval_to_t_end_ms",
            "anchor"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in req_rows:
        print("| " + " | ".join(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    print()
    print(f"## Every wait for a keep-alive block (an evaluation that ended {a.timeout:,.0f} to {a.timeout:,.0f} + "
          f"{a.win:.0f} ms after its card's last beat)")
    print()
    wcols = ["load", "request", "card", "evaluation", "where", "from_t_s_ms", "to_t_end_ms", "idle_before_ms",
             "host_ms", "gpu_ms", "wait_ms", "after_beat_ms", "in_gen_ms"]
    print("| " + " | ".join(wcols) + " |")
    print("|" + "---|" * len(wcols))
    for r in wait_rows:
        print("| " + " | ".join(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in wcols) + " |")
    if a.tsv_dir:
        os.makedirs(a.tsv_dir, exist_ok=True)
        for fn, rows, cs in [("requests.tsv", req_rows, list(req_rows[0].keys())),
                             ("waits.tsv", wait_rows, list(wait_rows[0].keys()) if wait_rows else WAIT_KEYS)]:
            with open(os.path.join(a.tsv_dir, fn), "w") as f:
                w = csv.writer(f, delimiter="\t", lineterminator="\n")
                w.writerow(cs)
                for r in rows:
                    w.writerow([f"{r[c]:.3f}" if isinstance(r[c], float) else r[c] for c in cs])


if __name__ == "__main__":
    main()
