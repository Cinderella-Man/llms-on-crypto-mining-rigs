#!/usr/bin/env python3
"""step_gap.py: a served speculative step, stage by stage, from llama-server's LLAMA_STEP_TRACE file (C405, Q960),
optionally beside the fork's GGML_CUDA_PROFILE_HOST file of the same load.

  python3 -I step_gap.py TRACE.jsonl [--profile HOST_PROFILE.csv (--server-log SERVER.log | --profile-t0 MS)]
      [--main-devices 0,1,2,3,4,5,6,7,8] [--draft-device 8] [--skip 3] [--check] [--tsv steps.tsv] > gap.md

The trace (LLAMA_STEP_TRACE=<file>, Miner Llama C405): a header line {"trace":"llama-step","t0_steady_ms":...}, then
one JSON line per llama-server update_slots call that decoded: {"step","write_us","n":{counts},"t":[[mark, ms],...]}.
Marks "s.*" are the server's, "mtp.*" the MTP drafter's, "v.*" and "d.*" the main and the draft context's stages
(dec, dec.bal, dec.mem, dec.out, ub.apply, ub.reuse or ub.build, ub.in, ub.comp, ub.cp, dec.end, and sync/sync.end
around every wait for the cards). One line holds a step's drafting passes, then the verify pass they feed, the draft's
catch-up over the verified tokens and the acceptance.

A generation step is a line with drafted tokens (n.drf > 0) and an acceptance (s.acc). A typical step has the commonest
sequence of mark names among them; the first --skip generation steps after each prompt line are left out (a request's
first steps pay one-off costs). Tables are over typical steps:
  waterfall  every pair of consecutive marks, repeated names numbered by occurrence (mtp.dp#2); the rows add up to the
             line, from s.upd to s.upd.end
  gap        the stages between the last drafting pass's wait (its last d.sync.end) and the verify pass's inputs set
             (v.ub.in): the host's part of what S37 measured as the server between the last draft pass and the next
             verify pass. With the profile, two rows more: from the end of the last drafting pass's work on the draft
             card (its graph_compute entry + its GPU time, as step_split.py counts it) to that wait's end, and from
             v.ub.in to the first main card's graph_compute entry (the scheduler's input copies for that card)
  between    from a line's s.upd.end to the next line's s.upd (the line's own write, the server's task loop)
--check verifies the file (header, every line JSON, times non-decreasing in a line, the counts) and exits 1 on a
problem. The profile's times move onto the trace's scale by t0_profile - t0_trace (the server log's line "its 0 ms is
<t> ms on the steady clock" from GGML_CUDA_PROFILE_HOST, or --profile-t0).
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict

DESC = {
    ("d.sync.end", "mtp.dp.smp"): "the drafted id and candidates read, the draft's host sampler",
    ("mtp.dp.smp", "mtp.dp.row"): "the h row read, the draft token kept, the next batch",
    ("mtp.dp.row", "mtp.drf.end"): "the drafter's end (shortlist and carry reset, length check)",
    ("mtp.drf.end", "s.drf.out"): "back in the server",
    ("s.drf.out", "s.drf.end"): "the yield's end: the queue's worker thread stops",
    ("s.drf.end", "s.ckpt"): "draft bookkeeping, the draft context's seq_rm, checkpoints",
    ("s.ckpt", "s.ckgen"): "the generation checkpoint check",
    ("s.ckgen", "s.bat"): "the sampled and drafted tokens into the batch",
    ("s.bat", "s.pre.end"): "the prompt loop over the slots, the rest of pre_decode",
    ("s.pre.end", "s.slots.end"): "TTS and realtime slots, LoRA, embeddings flag, taper",
    ("s.slots.end", "s.render"): "the batch rendered for llama_process",
    ("s.render", "s.dec"): "into decode()",
    ("s.dec", "s.dec.y"): "metrics, output rows, screened head, head bound, taper hint",
    ("s.dec.y", "s.dec.in"): "the yield's start: the queue's worker thread woken",
    ("s.dec.in", "v.dec"): "llama_process into llama_context::decode",
    ("v.dec", "v.dec.bal"): "pipe drain, batch checks, the batch allocator",
    ("v.dec.bal", "v.dec.mem"): "shortlist, scheduler reserve, memory update, the memory slot (init_batch)",
    ("v.dec.mem", "v.dec.out"): "output buffer, the samplers' begin, the cross-context waits",
    ("v.dec.out", "v.ub.apply"): "the ubatch, the memory context applied",
    ("v.ub.apply", "v.ub.reuse"): "graph parameters, the graph reused",
    ("v.ub.apply", "v.ub.build"): "graph parameters, the graph built and allocated",
    ("v.ub.reuse", "v.ub.in"): "the graph's inputs set (host side)",
    ("v.ub.build", "v.ub.in"): "the graph's inputs set (host side)",
}


def desc(a, b):
    return DESC.get((re.sub(r"\(\d+\)$", "", a), re.sub(r"\(\d+\)$", "", b)), "")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def load_trace(path, problems):
    header, lines = None, []
    with open(path) as f:
        for k, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                o = json.loads(raw)
            except json.JSONDecodeError:
                problems.append(f"line {k + 1}: not JSON (a line cut at the server's exit?)")
                continue
            if k == 0:
                if o.get("trace") != "llama-step" or "t0_steady_ms" not in o:
                    problems.append("line 1: no llama-step header")
                header = o
                continue
            if not isinstance(o.get("t"), list) or not o["t"]:
                problems.append(f"line {k + 1}: no marks")
                continue
            ts = [m[1] for m in o["t"]]
            if any(b < a for a, b in zip(ts, ts[1:])):
                problems.append(f"line {k + 1}: times go backwards")
            o["t"] = fold_syncs(o["t"])
            o["names"] = [m[0] for m in o["t"]]
            lines.append(o)
    return header, lines


def fold_syncs(marks):
    """A run of back-to-back waits of one context (sync, sync.end, sync, sync.end, ...) becomes one pair, named with
    the run's length: <r>.sync(k) and <r>.sync.end(k). The getters that read one output each synchronize."""
    out, i = [], 0
    while i < len(marks):
        name = marks[i][0]
        if name.endswith(".sync") and i + 1 < len(marks) and marks[i + 1][0] == name + ".end":
            j, k = i, 0
            while j + 1 < len(marks) and marks[j][0] == name and marks[j + 1][0] == name + ".end":
                j += 2
                k += 1
            label = f"({k})" if k > 1 else ""
            out.append([name + label, marks[i][1]])
            out.append([name + ".end" + label, marks[j - 1][1]])
            i = j
            continue
        out.append(marks[i])
        i += 1
    return out


def numbered(names):
    seen, out = Counter(), []
    for n in names:
        seen[n] += 1
        out.append(f"{n}#{seen[n]}" if seen[n] > 1 else n)
    return out


def anchor_from_log(path):
    pat = re.compile(r"timing every graph evaluation into .* \(its 0 ms is ([0-9.]+) ms on the steady clock\)")
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return None


def load_profile(path, shift):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["host_enter_ms"]) + shift, int(r["device"]), float(r["gpu_us"]) / 1000.0))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort()
    return rows


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def stats_row(label, xs, desc=""):
    return [label, f"{statistics.median(xs):.3f}", f"{statistics.mean(xs):.3f}", f"{pct(xs, 0.1):.3f}",
            f"{pct(xs, 0.9):.3f}", desc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--profile")
    ap.add_argument("--server-log")
    ap.add_argument("--profile-t0", type=float)
    ap.add_argument("--main-devices", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--draft-device", type=int, default=8)
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    problems = []
    header, lines = load_trace(a.trace, problems)
    if header is None:
        problems.append("no header")

    gen, prompt_lines, since_prompt = [], 0, 0
    for o in lines:
        is_gen = o["n"].get("drf", 0) > 0 and "s.acc" in o["names"]
        if not is_gen:
            prompt_lines += 1
            since_prompt = 0
            continue
        since_prompt += 1
        o["first"] = since_prompt <= a.skip
        gen.append(o)
        # a generation line's own consistency: one verify decode, at least one drafting pass per drafted token (the
        # MTP drafter runs its full length even where the slot keeps fewer tokens, at a reply's end)
        n_dp = o["names"].count("mtp.dp")
        if n_dp and n_dp < o["n"]["drf"]:
            problems.append(f"step {o['step']}: {n_dp} drafting passes for {o['n']['drf']} drafted tokens")
        o["spare"] = max(0, n_dp - o["n"]["drf"])
        if o["names"].count("v.dec") != 1:
            problems.append(f"step {o['step']}: {o['names'].count('v.dec')} main decodes")

    print(f"# {a.trace}\n")
    print(f"- lines: {len(lines)} ({prompt_lines} without a draft, {len(gen)} generation steps); drafted "
          f"{sum(o['n'].get('drf', 0) for o in gen)}, accepted {sum(o['n'].get('acc', 0) for o in gen)}; "
          f"drafting passes beyond the drafted tokens {sum(o['spare'] for o in gen)} (in "
          f"{sum(1 for o in gen if o['spare'])} steps); line writes: median "
          f"{statistics.median([o['write_us'] for o in lines if o['write_us'] >= 0] or [0]):.0f} us")
    n_drf = Counter(o["n"]["drf"] for o in gen).most_common(1)[0][0] if gen else 0
    sig = Counter(tuple(o["names"]) for o in gen if not o["first"] and o["n"]["drf"] == n_drf)
    if not sig:
        print("- no typical generation step")
        for p in problems:
            print(f"- PROBLEM: {p}")
        return 1 if (a.check and problems) else 0
    common, n_common = sig.most_common(1)[0]
    typ = [o for o in gen if not o["first"] and o["n"]["drf"] == n_drf and tuple(o["names"]) == common]
    print(f"- typical steps: {len(typ)} of {len(gen)} (the commonest of {len(sig)} mark sequences among the "
          f"steps that drafted {n_drf} tokens, {common.count('mtp.dp')} drafting passes); the first {a.skip} after "
          f"each prompt left out")

    names = numbered(list(common))
    per_pair = defaultdict(list)
    for o in typ:
        ts = [m[1] for m in o["t"]]
        for i in range(len(ts) - 1):
            per_pair[i].append(ts[i + 1] - ts[i])
    print("\n## Waterfall (ms between consecutive marks; the rows add up to the line)\n")
    rows = []
    for i in range(len(names) - 1):
        d = desc(common[i], common[i + 1])
        rows.append(stats_row(f"{names[i]} → {names[i + 1]}", per_pair[i], d))
    span = [o["t"][-1][1] - o["t"][0][1] for o in typ]
    rows.append(stats_row("**the line (s.upd → s.upd.end)**", span))
    print(table(rows, ["from → to", "median", "mean", "p10", "p90", "what runs there"]))

    # the gap: from the last drafting pass's wait to the verify pass's inputs set
    try:
        i_drf_end = common.index("s.drf.out")
        i0 = max(i for i in range(i_drf_end) if common[i].startswith("d.sync.end"))
        i1 = common.index("v.ub.in", i_drf_end)
    except ValueError:
        i0 = i1 = None
    prof = None
    if a.profile:
        t0p = a.profile_t0 if a.profile_t0 is not None else (anchor_from_log(a.server_log) if a.server_log else None)
        if t0p is None:
            print("\n- the profile's 0 ms is unknown: pass --server-log or --profile-t0")
        else:
            prof = load_profile(a.profile, t0p - header["t0_steady_ms"])
    if i0 is not None:
        print("\n## The gap: the last drafting pass's wait → the verify pass's inputs set\n")
        rows = []
        main_dev0 = int(a.main_devices.split(",")[0])
        extra_pre, extra_post = [], []
        if prof:
            import bisect
            enters = [r[0] for r in prof]
            for o in typ:
                t_sync_end, t_vin = o["t"][i0][1], o["t"][i1][1]
                t_drf = o["t"][common.index("s.drf")][1]
                j = bisect.bisect_left(enters, t_drf)
                last = None
                while j < len(prof) and prof[j][0] < t_vin:
                    if prof[j][1] == a.draft_device:
                        last = prof[j]
                    j += 1
                k = bisect.bisect_left(enters, t_vin - 0.2)
                while k < len(prof) and prof[k][1] != main_dev0:
                    k += 1
                if last is not None and k < len(prof):
                    extra_pre.append(t_sync_end - (last[0] + last[2]))
                    extra_post.append(prof[k][0] - t_vin)
            if extra_pre:
                rows.append(stats_row("draft card's work end → d.sync.end (last pass)", extra_pre,
                                      "the drafted id's way back, the host's wake"))
        for i in range(i0, i1):
            rows.append(stats_row(f"{names[i]} → {names[i + 1]}", per_pair[i], desc(common[i], common[i + 1])))
        host = [o["t"][i1][1] - o["t"][i0][1] for o in typ]
        rows.append(stats_row("**host: last d.sync.end → v.ub.in**", host))
        if extra_post:
            rows.append(stats_row("v.ub.in → first main card's graph_compute entry", extra_post,
                                  "the scheduler's input copies for that card"))
            tot = [x + h + y for x, h, y in zip(extra_pre, host, extra_post)]
            rows.append(stats_row("**the gap, as S37 counts it**", tot))
        print(table(rows, ["from → to", "median", "mean", "p10", "p90", "what runs there"]))

    btw, typ_ids, gen_ids = [], {id(o) for o in typ}, {id(o) for o in gen}
    for p, q in zip(lines, lines[1:]):
        if id(p) in typ_ids and id(q) in gen_ids:
            btw.append(q["t"][0][1] - p["t"][-1][1])
    if btw:
        print("\n## Between lines\n")
        print(table([stats_row("s.upd.end → next line's s.upd", btw, "the line's write, the task loop")],
                    ["from → to", "median", "mean", "p10", "p90", "what runs there"]))

    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("step\ttypical\tdrf\tacc\tline_ms\tgap_host_ms\n")
            for o in gen:
                ok = tuple(o["names"]) == common and not o["first"]
                g = (o["t"][i1][1] - o["t"][i0][1]) if (ok and i0 is not None) else float("nan")
                f.write(f"{o['step']}\t{int(ok)}\t{o['n'].get('drf', 0)}\t{o['n'].get('acc', 0)}\t"
                        f"{o['t'][-1][1] - o['t'][0][1]:.3f}\t{g:.3f}\n")

    for p in problems:
        print(f"- PROBLEM: {p}")
    if a.check:
        print(f"\ncheck: {'OK' if not problems else 'FAIL'} ({len(problems)} problems)")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
