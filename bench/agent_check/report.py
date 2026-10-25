"""Tables and the pilot's go/no-go checks from run.py's results and the proxy log.

    python3 report.py table RESULTS [--log RESULTS/proxy.jsonl …] [--csv RESULTS/attempts.csv]
    python3 report.py gonogo RESULTS --log … [--replay REPLAY.jsonl --replay-label v2-ud --live-arm v2-ud]
                      [--meminfo RIG/meminfo.log] [--server-log RIG/server-*.log …] [--expect-min v2-ud=3.9,v1-ud=5.5]
    python3 report.py compare RESULTS --log … [--arms stock,rel] [--turns-csv RESULTS/turns.csv]

`table` writes one row per scored attempt in the column layout of ../task_results.py (lane agent_polyglot, file =
arm, category = track, cluster = task), so ../paired_tasks.py reads it, followed by the agent columns: turns, requests,
wall-clock split into prefill, decode, fixed per-request cost and the rest (tools and client), prompt tokens processed
and cached, cache hit rate, peak context, time to first token, and the process failures. It prints a summary per arm.

`compare` puts the arms side by side (the first arm is the reference): per task the result, minutes, turns and tool
calls (all, failed, malformed) with the time ratio; per arm the tasks passed, minutes, time to first token over all
turns, the server's prompt and generation speeds and the tool-call failures; the ratio of total and per-task times on
the tasks every arm scored; and the median time to first token per turn. Every request's row goes to turns.csv.

`gonogo` applies the pilot's checks (agent_workload_check.md §6.3):
- harness: at least 5 of 6 attempts finish without an infrastructure failure; every request with a proxy row and timings; the proxy's
  time to first token against the server's prefill time (their difference is the request's fixed cost plus the
  network round trip);
- tool calls: malformed calls under 5 % of all calls, none parsed from inside the reasoning;
- cache: from turn 2 on, the median cache hit rate ≥ 80 %, and each turn's cache path: "exact" (cache_n reaches the
  previous request's prompt plus its generated tokens), "reply re-read" (cache_n stops at the previous prompt's end,
  so the previous reply is processed again), "older checkpoint" (cache_n falls short of the previous prompt: Q132's
  eviction case);
- memory (rig): lowest MemAvailable ≥ 600 MB and no swap growth, from rig_server.sh's meminfo.log;
- time: median minutes per task within 2× of the note's estimate per arm;
- replay: its ms per prompt and per generated token within 5 % of the live rows of the same traces;
- server logs: counts of the lines that show backend sampling off, checkpoint restores, full re-processing, the
  screened head's decisions, NaN or watchdog kills.
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from task_results import COLUMNS  # noqa: E402
from replay import costs, load_log, load_replay  # noqa: E402

AGENT_COLUMNS = ["track", "turns", "requests", "wall_s", "prefill_s", "decode_s", "fixed_s", "other_s",
                 "prompt_new", "prompt_cached", "cache_hit", "peak_context", "ttft_p50", "ttft_max",
                 "tool_calls", "tool_errors", "unknown_tool", "invalid_args", "markup_in_text", "length_stops",
                 "runaway_kind", "compactions", "hidden_pass", "tests_untouched", "n_pass", "n_total", "tag"]
SLACK = 16  # tokens of template between one request's end and the next request's new part


def load_results(root):
    rows = []
    for p in sorted(Path(root).glob("*/*/a*/result.json")):
        r = json.loads(p.read_text())
        if r.get("status") == "scored":
            rows.append(r)
    infra = sorted(str(p.relative_to(root)) for p in Path(root).glob("*/*/a*-t*-infra"))
    exhausted = [json.loads(p.read_text()) for p in Path(root).glob("*/*/a*/result.json")
                 if json.loads(p.read_text()).get("status") == "infra_exhausted"]
    return rows, infra, exhausted


def requests_by_attempt(logs):
    traces, _ = load_log(logs) if logs else ({}, {})
    return traces


def cache_paths(reqs):
    """Cache path of every request after the first: exact, reply re-read or older checkpoint."""
    out = []
    for prev, cur in zip(reqs, reqs[1:]):
        if None in (prev.get("cache_n"), prev.get("prompt_n"), cur.get("cache_n")):
            out.append("unknown")
            continue
        prev_prompt = prev["cache_n"] + prev["prompt_n"]
        if cur["cache_n"] >= prev_prompt + (prev.get("predicted_n") or 0) - SLACK:
            out.append("exact")
        elif cur["cache_n"] >= prev_prompt - SLACK:
            out.append("reply re-read")
        else:
            out.append("older checkpoint")
    return out


def attempt_row(r, reqs):
    pre = sum(q.get("prompt_ms") or 0 for q in reqs) / 1000
    dec = sum(q.get("predicted_ms") or 0 for q in reqs) / 1000
    req_s = sum(q.get("t_end") or 0 for q in reqs)
    pn = sum(q.get("prompt_n") or 0 for q in reqs)
    cn = sum(q.get("cache_n") or 0 for q in reqs)
    gn = sum(q.get("predicted_n") or 0 for q in reqs)
    ttft = [q["t_first_token"] for q in reqs if q.get("t_first_token") is not None]
    stop = "runaway_" + r["runaway"] if r["runaway"] else ("length" if r["length_stops"] else "stop")
    row = dict(lane="agent_polyglot", task_id=r["task_id"], cluster=r["task_id"], category=r["track"], file=r["arm"],
               attempt=r["attempt"], seed="", error_type=r.get("error_type") or ("" if r["success"] else "tests"),
               prompt_tokens=pn + cn if reqs else r["usage_input"] + r["usage_cache_read"],
               output_tokens=gn if reqs else r["usage_output"], reasoning_chars="", stop_type=stop,
               n_calls=r["tool_calls"], undecodable=r["markup_in_text"], unknown_function=r["unknown_tool"],
               unknown_arg=r["invalid_args"], call_in_think=r["call_in_think"], extra_think_close="",
               track=r["track"], turns=r["turns"], requests=len(reqs), wall_s=r["wall_s"],
               prefill_s=round(pre, 2), decode_s=round(dec, 2), fixed_s=round(req_s - pre - dec, 2),
               other_s=round(r["wall_s"] - req_s, 2), prompt_new=pn, prompt_cached=cn,
               cache_hit=round(cn / (cn + pn), 4) if cn + pn else "",
               peak_context=max(((q.get("cache_n") or 0) + (q.get("prompt_n") or 0) for q in reqs), default=""),
               ttft_p50=round(median(ttft), 3) if ttft else "", ttft_max=round(max(ttft), 3) if ttft else "",
               tool_calls=r["tool_calls"], tool_errors=r["tool_errors"], unknown_tool=r["unknown_tool"],
               invalid_args=r["invalid_args"], markup_in_text=r["markup_in_text"], length_stops=r["length_stops"],
               runaway_kind=r["runaway"], compactions=r["compactions"], hidden_pass=r["hidden_pass"],
               tests_untouched=r["tests_untouched"], n_pass=r["n_pass"], n_total=r["n_total"], tag=r["tag"])
    row["pass"] = r["success"]
    row["malformed"] = int(r["malformed_calls"] > 0)
    row["runaway"] = int(bool(r["runaway"]) or r["length_stops"] > 0)
    return row


def build(a):
    results, infra, exhausted = load_results(a.results)
    traces = requests_by_attempt(a.log)
    rows = [attempt_row(r, traces.get(r["tag"], [])) for r in results]
    return results, rows, infra, exhausted, traces


def cmd_table(a):
    _, rows, infra, exhausted, _ = build(a)
    out = a.csv or Path(a.results) / "attempts.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + AGENT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scored attempts -> {out}; {len(infra)} infrastructure tries re-run, "
          f"{len(exhausted)} attempts given up\n")
    by = defaultdict(list)
    for r in rows:
        by[r["file"]].append(r)
    print("| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | "
          "turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, rr in by.items():
        mins = sorted(r["wall_s"] / 60 for r in rr)
        p90 = mins[min(len(mins) - 1, int(0.9 * len(mins)))]
        mean = lambda k: sum(float(r[k] or 0) for r in rr) / len(rr)
        calls = sum(r["tool_calls"] for r in rr)
        bad = sum(r["unknown_tool"] + r["invalid_args"] + r["markup_in_text"] for r in rr)
        hit = [r["cache_hit"] for r in rr if r["cache_hit"] != ""]
        ttft = [r["ttft_p50"] for r in rr if r["ttft_p50"] != ""]
        print(f"| {arm} | {len(rr)} | {100 * sum(r['pass'] for r in rr) / len(rr):.1f} % | {median(mins):.1f}, "
              f"{p90:.1f} | {mean('prefill_s'):.0f} / {mean('decode_s'):.0f} / {mean('fixed_s'):.0f} / "
              f"{mean('other_s'):.0f} | {median(r['turns'] for r in rr)} | {mean('prompt_new'):.0f} | "
              f"{100 * sum(hit) / len(hit):.1f} % | {median(ttft) if ttft else float('nan'):.2f} | "
              f"{bad}/{calls} | {sum(r['runaway'] for r in rr)} |" if hit else
              f"| {arm} | {len(rr)} | {100 * sum(r['pass'] for r in rr) / len(rr):.1f} % | {median(mins):.1f}, "
              f"{p90:.1f} | (no proxy rows) | {median(r['turns'] for r in rr)} | | | | {bad}/{calls} | "
              f"{sum(r['runaway'] for r in rr)} |")


def verdict(ok):
    return "GO" if ok else "NO-GO"


def cmd_gonogo(a):
    results, rows, infra, exhausted, traces = build(a)
    lines = []
    allok = True

    def item(name, ok, detail):
        nonlocal allok
        allok &= ok is not False
        lines.append(f"- **{name}: {'n/a' if ok is None else verdict(ok)}** — {detail}")

    # harness
    hit = {tuple(p.split("/")[:2]) + (p.split("/")[2].split("-")[0],) for p in infra}
    clean = [r for r in results if (r["arm"], r["task_id"], f"a{r['attempt']}") not in hit]
    n_att = len(results) + len(exhausted)
    item("harness: infrastructure", n_att > 0 and len(clean) >= 5 / 6 * n_att,
         f"{len(clean)} of {n_att} attempts finished without an infrastructure failure; {len(infra)} tries re-run "
         f"({', '.join(infra) or 'none'}), {len(exhausted)} attempts given up")
    missing, no_timings, resid = [], 0, []
    for r in results:
        reqs = traces.get(r["tag"], [])
        n_assist = sum(r["stop_reasons"].values())
        if len(reqs) != n_assist:
            missing.append(f"{r['tag']} ({len(reqs)} rows / {n_assist} replies)")
        for q in reqs:
            if q.get("prompt_ms") is None:
                no_timings += 1
            elif q.get("t_first_token") is not None:
                resid.append(q["t_first_token"] - q["prompt_ms"] / 1000)
    item("harness: proxy rows", not missing and not no_timings,
         f"{sum(len(traces.get(r['tag'], [])) for r in results)} requests logged; attempts with a row count "
         f"different from pi's replies: {', '.join(missing) or 'none'}; rows without timings: {no_timings}")
    if resid:
        rs = sorted(resid)
        item("harness: first token vs prefill", rs[int(0.9 * (len(rs) - 1))] < 0.5 + a.rtt,
             f"proxy first token minus server prefill: median {median(rs):.2f} s, p90 {rs[int(0.9 * (len(rs) - 1))]:.2f} s "
             f"(limit 0.5 s + round trip {a.rtt:.2f} s; includes the per-request fixed cost before prefill)")
    # tool calls
    calls = sum(r["tool_calls"] for r in results)
    bad = sum(r["malformed_calls"] for r in results)
    think = sum(r["call_in_think"] for r in results)
    item("tool calls", (calls == 0 or bad / calls < 0.05) and think == 0,
         f"malformed {bad} of {calls} calls ({100 * bad / max(1, calls):.1f} %: unknown tool "
         f"{sum(r['unknown_tool'] for r in results)}, invalid arguments {sum(r['invalid_args'] for r in results)}, "
         f"markup in text {sum(r['markup_in_text'] for r in results)}); calls inside the reasoning {think}")
    # cache
    hits, paths = [], defaultdict(int)
    for r in results:
        reqs = traces.get(r["tag"], [])
        for q in reqs[1:]:
            if q.get("cache_n") is not None and q.get("prompt_n") is not None and q["cache_n"] + q["prompt_n"]:
                hits.append(q["cache_n"] / (q["cache_n"] + q["prompt_n"]))
        for p in cache_paths(reqs):
            paths[p] += 1
    if hits:
        item("cache", median(hits) >= 0.8 and not paths["older checkpoint"],
             f"turns ≥ 2: median cache hit {100 * median(hits):.1f} %, min {100 * min(hits):.1f} %; cache paths "
             + ", ".join(f"{k} {v}" for k, v in sorted(paths.items())))
    else:
        item("cache", None, "no proxy rows")
    # memory
    if a.meminfo:
        avail, swap = [], []
        for p in a.meminfo:
            for l in Path(p).read_text().splitlines():
                f = l.split()
                if len(f) >= 3 and f[1].isdigit():
                    avail.append(int(f[1]) / 1024)
                    swap.append(int(f[2]) / 1024)
        if avail:
            item("memory", min(avail) >= 600 and max(swap) - swap[0] <= 16,
                 f"lowest MemAvailable {min(avail):.0f} MB; swap used {swap[0]:.0f} → {max(swap):.0f} MB")
    # time
    if a.expect_min:
        exp = dict((k, float(v)) for k, v in (x.split("=") for x in a.expect_min.split(",")))
        for arm, e in exp.items():
            mins = [r["wall_s"] / 60 for r in results if r["arm"] == arm]
            if mins:
                m = median(mins)
                item(f"time ({arm})", e / 2 <= m <= 2 * e, f"median {m:.1f} min per task against {e:.1f} est.")
    # replay
    if a.replay:
        rep = [x for x in load_replay(a.replay) if x["label"] == a.replay_label]
        tr = sorted({x["trace"] for x in rep})
        live = [q for t in tr for q in traces.get(t, [])]
        if rep and live:
            cr, cl = costs(rep), costs(live)
            dp, dg = cr["ms_prompt"] / cl["ms_prompt"] - 1, cr["ms_gen"] / cl["ms_gen"] - 1
            item("replay", abs(dp) <= 0.05 and abs(dg) <= 0.05,
                 f"{a.replay_label} on {len(tr)} traces: ms per prompt token {cr['ms_prompt']:.2f} vs live "
                 f"{cl['ms_prompt']:.2f} ({100 * dp:+.1f} %), per generated token {cr['ms_gen']:.2f} vs "
                 f"{cl['ms_gen']:.2f} ({100 * dg:+.1f} %)")
    # server logs
    if a.server_log:
        pats = {"backend sampling off": r"backend sampling.*(not compatible|disabled)|disabling backend sampling",
                "checkpoint restored": r"restored context checkpoint|restoring.*checkpoint",
                "checkpoint created": r"created context checkpoint",
                "checkpoint erased": r"erased.*checkpoint|erasing.*checkpoint",
                "full re-processing": r"forcing full prompt re-processing",
                "screened head": r"screen",
                "NaN / inf": r"\bnan\b|\binf\b",
                "watchdog kill": r"watchdog"}
        for p in a.server_log:
            text = Path(p).read_text(errors="replace")
            counts = {k: len(re.findall(v, text, re.I)) for k, v in pats.items()}
            lines.append(f"- server log {Path(p).name}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("\n".join(lines))
    print(f"\n**Overall: {verdict(allok)}**")


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def cmd_compare(a):
    results, rows, infra, exhausted, traces = build(a)
    arms = a.arms.split(",") if a.arms else list(dict.fromkeys(r["file"] for r in rows))
    rows = [r for r in rows if r["file"] in arms]
    by = {(r["file"], r["task_id"]): r for r in rows}
    first = {}  # tasks in the order they first ran
    for r in results:
        if r["arm"] in arms:
            first[r["task_id"]] = min(first.get(r["task_id"], r["started"]), r["started"])
    tasks = sorted({r["task_id"] for r in rows}, key=lambda t: first.get(t, ""))
    reqs = {r["tag"]: traces.get(r["tag"], []) for r in rows}
    turn_of = lambda q: int(q["tag"].rsplit("/", 1)[1])

    out = a.turns_csv or Path(a.results) / "turns.csv"
    cols = ["arm", "task_id", "attempt", "turn", "ttft_s", "t_end_s", "prompt_n", "cache_n", "prompt_ms", "predicted_n",
            "predicted_ms", "finish_reason", "n_tool_calls"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            for q in reqs[r["tag"]]:
                w.writerow([r["file"], r["task_id"], r["attempt"], turn_of(q), q.get("t_first_token"), q.get("t_end"),
                            q.get("prompt_n"), q.get("cache_n"), q.get("prompt_ms"), q.get("predicted_n"),
                            q.get("predicted_ms"), q.get("finish_reason"), q.get("n_tool_calls")])

    def result(r):
        if r is None:
            return "—"
        if r["runaway_kind"]:
            return f"runaway ({r['runaway_kind']})"
        return "pass" if r["pass"] else f"fail ({r['n_pass']}/{r['n_total']})"

    print(f"{len(rows)} scored attempts, arms {', '.join(arms)}; {len(infra)} infrastructure tries re-run, "
          f"{len(exhausted)} attempts given up; per-request rows in {out}\n")
    print("### Per task\n")
    print("Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed "
          "(unknown tool, arguments that fail the schema, or call markup in the text).\n")
    head = "| task | " + " | ".join(f"{arm}: result, min, turns, tool calls" for arm in arms)
    if len(arms) > 1:
        head += " | " + " | ".join(f"time {arms[0]} / {arm}" for arm in arms[1:])
    print(head + " |")
    print("|---" * (1 + len(arms) + max(0, len(arms) - 1)) + "|")
    for t in tasks:
        cells = []
        for arm in arms:
            r = by.get((arm, t))
            cells.append("—" if r is None else
                         f"{result(r)}, {r['wall_s'] / 60:.1f}, {r['turns']}, {r['tool_calls']} / {r['tool_errors']} / "
                         f"{r['unknown_tool'] + r['invalid_args'] + r['markup_in_text']}")
        for arm in arms[1:]:
            r0, r1 = by.get((arms[0], t)), by.get((arm, t))
            cells.append(f"{r0['wall_s'] / r1['wall_s']:.2f}×" if r0 and r1 else "—")
        print(f"| {t} | " + " | ".join(cells) + " |")

    print("\n### Per arm\n")
    print("| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | "
          "prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        rr = [r for r in rows if r["file"] == arm]
        if not rr:
            continue
        qs = [q for r in rr for q in reqs[r["tag"]]]
        ttft = [q["t_first_token"] for q in qs if q.get("t_first_token") is not None]
        pn, pms = sum(q.get("prompt_n") or 0 for q in qs), sum(q.get("prompt_ms") or 0 for q in qs)
        gn, gms = sum(q.get("predicted_n") or 0 for q in qs), sum(q.get("predicted_ms") or 0 for q in qs)
        bad = sum(r["unknown_tool"] + r["invalid_args"] + r["markup_in_text"] for r in rr)
        print(f"| {arm} | {sum(r['pass'] for r in rr)} / {len(rr)} | {sum(r['wall_s'] for r in rr) / 60:.1f} | "
              f"{median(r['wall_s'] / 60 for r in rr):.1f} | {median(ttft) if ttft else float('nan'):.2f}, "
              f"{pct(ttft, 0.9):.2f} | {1000 * pn / pms if pms else float('nan'):.0f} | "
              f"{1000 * gn / gms if gms else float('nan'):.1f} | {sum(r['tool_calls'] for r in rr)} / "
              f"{sum(r['tool_errors'] for r in rr)} / {bad} | {sum(r['runaway'] for r in rr)} |")
    if len(arms) > 1:
        both = [t for t in tasks if all((arm, t) in by for arm in arms)]
        print(f"\nTasks scored on every arm: {len(both)}.")
        for arm in arms[1:]:
            m0 = sum(by[(arms[0], t)]["wall_s"] for t in both)
            m1 = sum(by[(arm, t)]["wall_s"] for t in both)
            ratios = [by[(arms[0], t)]["wall_s"] / by[(arm, t)]["wall_s"] for t in both]
            gm = 2 ** (sum(math.log2(x) for x in ratios) / len(ratios)) if ratios else float("nan")
            print(f"- {arm} against {arms[0]}: {m1 / 60:.1f} against {m0 / 60:.1f} minutes in all "
                  f"({m0 / m1 if m1 else float('nan'):.2f}× faster), per task {gm:.2f}× (geometric mean, "
                  f"range {min(ratios, default=float('nan')):.2f}–{max(ratios, default=float('nan')):.2f}); passed "
                  f"{sum(by[(arm, t)]['pass'] for t in both)} against {sum(by[(arms[0], t)]['pass'] for t in both)}")

    print("\n### Time to first token per turn\n")
    print("Median over tasks of the seconds from the request to its first token as the client sees it (network "
          "included), and of the prompt tokens the server processed for it (the rest came from its cache).\n")
    nmax = max((turn_of(q) for qq in reqs.values() for q in qq), default=0)
    print("| turn | " + " | ".join(f"{arm}: tasks, TTFT s, new prompt tokens" for arm in arms) + " |")
    print("|---" * (1 + len(arms)) + "|")
    for k in range(1, nmax + 1):
        cells = []
        for arm in arms:
            qs = [q for r in rows if r["file"] == arm for q in reqs[r["tag"]] if turn_of(q) == k]
            tt = [q["t_first_token"] for q in qs if q.get("t_first_token") is not None]
            pn = [q["prompt_n"] for q in qs if q.get("prompt_n") is not None]
            cells.append(f"{len(qs)}, {median(tt):.2f}, {median(pn):.0f}" if tt and pn else f"{len(qs)}, —, —")
        print(f"| {k} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("table", "gonogo", "compare"):
        p = sub.add_parser(name)
        p.add_argument("results")
        p.add_argument("--log", nargs="*", default=[])
        if name == "table":
            p.add_argument("--csv", type=Path)
        elif name == "compare":
            p.add_argument("--arms", help="arm,arm,… in this order; the first is the reference (default: all)")
            p.add_argument("--turns-csv", type=Path, help="default RESULTS/turns.csv")
        else:
            p.add_argument("--replay", nargs="*")
            p.add_argument("--replay-label")
            p.add_argument("--meminfo", nargs="*")
            p.add_argument("--server-log", nargs="*")
            p.add_argument("--expect-min", help="arm=minutes,… (the note's estimate per task)")
            p.add_argument("--rtt", type=float, default=0.1, help="network round trip to the server, s")
    a = ap.parse_args()
    {"table": cmd_table, "gonogo": cmd_gonogo, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
