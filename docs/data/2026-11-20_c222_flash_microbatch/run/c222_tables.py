#!/usr/bin/env python3
"""c222_tables.py: C222's (Q775) tables from one load of rig_taper.sh (taper_replay.py's requests.tsv and the
server's log): prompt seconds of every pass by value, by the size of a turn's new prompt, each taper setting against
the control passes, the replies, and which micro-batches each pass's calls were read in.

    python3 -I c222_tables.py --load c222/load-A --control short=600:64 --out tables
    python3 -I c222_tables.py --load c222/load-A --control short=600:64 --drop-pass 0 --out tables_warm

Only the replayed turns count (kind "turn"); the unmeasured prefix requests ("reset") are left out of the seconds but
kept in the order that maps the server's log lines to requests. A turn is binned by its planned new tokens (n_new):
under 600 (the control's short rule covers these calls), 600-1,023, 1,024-1,999, 2,000 and more.

Class of a setting in a bucket: the mean of its passes minus the mean of the control passes; gain when below minus
the spread, loss when above it, negligible otherwise; the spread is the largest difference between the control
passes in that bucket (spread_s). R60's rule also took the setting's own passes' difference (own_spread_s); the
class by that rule is class_r60.

Writes, in --out:
  passes.tsv    per pass: setting, turns, new tokens, prompt tokens, prompt seconds, tok/s, the requests' wall
                seconds, turns whose prompt_n differs from the planned new tokens, graph reallocations, the server's
                micro-batch plans (calls logged, the largest micro-batch of a call of 601 to 2,048 tokens)
  buckets.tsv   per setting and bucket: turns, new tokens, seconds by pass, mean, the control's by pass and mean,
                the difference in s and %, the spread, the class
  turns.tsv     every replayed turn: task, turn, new tokens, the tokens its call read (prompt_n, the same in every
                pass), prompt ms in every pass (columns p<pass>:<setting>)
  replies.tsv   per setting: turns whose greedy reply equals the control's in every pass
  fit.tsv       per setting: least squares of a turn's prompt seconds on its new tokens over the turns of 512 to
                1,999 new tokens (the review's fixed cost a turn and ms a token), every pass's turns together
  plans.tsv     per pass and call-size class: calls, and the micro-batch sizes the server logged (most common plans)
  decision.md   the classes in words
Standard library only.
"""
import argparse, collections, csv, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True, help="the load folder (requests.tsv, server.log or taper_lines.txt)")
ap.add_argument("--control", default="short=600:64")
ap.add_argument("--out", required=True)
ap.add_argument("--drop-pass", type=int, action="append", default=[],
                help="leave this pass out of every table (e.g. 0, the load's first pass, which warms the caches)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

BUCKETS = ((0, 600), (600, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"under {hi}" if lo == 0 else f"{lo}-{hi - 1}")


reqs = list(csv.DictReader(open(os.path.join(a.load, "requests.tsv")), delimiter="\t"))
for r in reqs:
    r["pass"], r["n_new"] = int(r["pass"]), int(r["n_new"])
    r["prompt_ms"], r["wall_s"] = float(r["prompt_ms"]), float(r["wall_s"])
    r["prompt_n"] = int(r["prompt_n"]) if r["prompt_n"] not in ("", "None") else 0
    r["calls"], r["realloc"] = [], 0

# the server's log in request order: taper lines before each request's timing line, and the running count of graph
# reallocations after it (rig_taper.sh's realloc_count reads the same lines)
log = os.path.join(a.load, "server.log")
if os.path.exists(log):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pend, k, prev = [], 0, (0, 0)
    cur = None
    for line in open(log, errors="replace"):
        line = ansi.sub("", line)
        m = re.search(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$", line)
        if m:
            pend.append((int(m.group(1)), m.group(5).strip()))
            continue
        if "prompt eval time =" in line:
            if k < len(reqs):
                reqs[k]["calls"] = pend
                cur = reqs[k]
            pend, k = [], k + 1
            continue
        m = re.search(r"graph reallocs =\s+(\d+)\s+/\s+(\d+)", line)
        if m and cur is not None:
            s, d = int(m.group(1)), int(m.group(2))
            cur["realloc"] = (s - prev[0]) + (d - prev[1])
            prev, cur = (s, d), None
    if k != len(reqs):
        print(f"warning: {k} timing lines in the server log, {len(reqs)} requests")


def sizes(plan):
    """'64x4 124 5' -> [64, 64, 64, 64, 124, 5]"""
    out = []
    for t in plan.split():
        n, _, c = t.partition("x")
        out += [int(n)] * (int(c) if c else 1)
    return out


turns = [r for r in reqs if r["kind"] == "turn" and r["pass"] not in a.drop_pass]
order = sorted({r["pass"] for r in turns})
setting_of = {p: next(r["arm"] for r in turns if r["pass"] == p) for p in order}
settings = [a.control] + sorted({s for s in setting_of.values() if s != a.control})
by_pass = {p: {(r["task"], int(r["turn"])): r for r in turns if r["pass"] == p} for p in order}
keys = sorted(set.intersection(*(set(v) for v in by_pass.values())), key=lambda k: next(
    i for i, r in enumerate(turns) if (r["task"], int(r["turn"])) == k))
n_new = {k: by_pass[order[0]][k]["n_new"] for k in keys}
passes_of = {s: [p for p in order if setting_of[p] == s] for s in settings}
ctrl = passes_of[a.control]


def secs(p, lo, hi):
    return sum(by_pass[p][k]["prompt_ms"] for k in keys if lo <= n_new[k] < hi) / 1000


# passes.tsv
with open(f"{a.out}/passes.tsv", "w") as f:
    f.write("pass\tsetting\tturns\tnew_tokens\tprompt_tokens\tprompt_s\ttok_s\twall_s\tturns_prompt_n_differs\t"
            "graph_reallocs\tcalls_logged\tlargest_ubatch_601_2048\n")
    for p in order:
        rs = [by_pass[p][k] for k in keys]
        pt = sum(r["prompt_n"] for r in rs)
        ps = sum(r["prompt_ms"] for r in rs) / 1000
        allp = [r for r in reqs if r["pass"] == p]
        calls = [c for r in rs for c in r["calls"]]
        mid = [max(sizes(pl)) for n, pl in calls if 601 <= n <= 2048]
        f.write(f"{p}\t{setting_of[p]}\t{len(rs)}\t{sum(n_new[k] for k in keys)}\t{pt}\t{ps:.2f}\t{pt / ps:.2f}\t"
                f"{sum(r['wall_s'] for r in allp):.2f}\t{sum(1 for r in rs if r['prompt_n'] != r['n_new'])}\t"
                f"{sum(r['realloc'] for r in allp)}\t{len(calls)}\t{max(mid) if mid else '-'}\n")

# buckets.tsv and the classes
cls = {}
with open(f"{a.out}/buckets.tsv", "w") as f:
    f.write("setting\tnew_tokens\tturns\ttokens\tprompt_s_by_pass\tmean_s\tcontrol_s_by_pass\tcontrol_mean_s\t"
            "delta_s\tdelta_pct\tspread_s\town_spread_s\tclass\tclass_r60\n")
    for s in settings:
        for lo, hi in BUCKETS:
            kb = [k for k in keys if lo <= n_new[k] < hi]
            if not kb:
                continue
            v = [secs(p, lo, hi) for p in passes_of[s]]
            o = [secs(p, lo, hi) for p in ctrl]
            m, om = statistics.mean(v), statistics.mean(o)
            spread, own = max(o) - min(o), max(v) - min(v)
            d = m - om
            c = "control" if s == a.control else ("gain" if d < -spread else "loss" if d > spread else "negligible")
            sr = max(spread, own)
            c60 = "control" if s == a.control else ("gain" if d < -sr else "loss" if d > sr else "negligible")
            cls[(s, bname(lo, hi))] = (c, v, m, o, om, d, spread, own, c60)
            f.write(f"{s}\t{bname(lo, hi)}\t{len(kb)}\t{sum(n_new[k] for k in kb)}\t{', '.join(f'{x:.2f}' for x in v)}\t"
                    f"{m:.2f}\t{', '.join(f'{x:.2f}' for x in o)}\t{om:.2f}\t{d:+.2f}\t{d / om * 100:+.2f}\t"
                    f"{spread:.2f}\t{own:.2f}\t{c}\t{c60}\n")

# turns.tsv
with open(f"{a.out}/turns.tsv", "w") as f:
    f.write("task\tturn\tnew_tokens\tcall_tokens\t" + "\t".join(f"p{p}:{setting_of[p]}" for p in order) + "\n")
    for k in keys:
        f.write(f"{k[0]}\t{k[1]}\t{n_new[k]}\t{by_pass[order[0]][k]['prompt_n']}\t"
                + "\t".join(f"{by_pass[p][k]['prompt_ms']:.1f}" for p in order) + "\n")

# replies.tsv
with open(f"{a.out}/replies.tsv", "w") as f:
    f.write("setting\tpasses\tturns\tsame_reply_as_control_in_every_pass\tcontrol_passes_agree\n")
    want = {k: {by_pass[p][k]["gen_md5"] for p in ctrl} for k in keys}
    agree = sum(1 for k in keys if len(want[k]) == 1)
    for s in settings:
        n = sum(1 for k in keys if all(by_pass[p][k]["gen_md5"] in want[k] for p in passes_of[s]))
        f.write(f"{s}\t{len(passes_of[s])}\t{len(keys)}\t{n}\t{agree}\n")

# fit.tsv: seconds = fixed + ms a token x new tokens, turns of 512 to 1,999 new tokens
with open(f"{a.out}/fit.tsv", "w") as f:
    f.write("setting\tturns\tpoints\tfixed_s\tms_per_token\n")
    for s in settings:
        kb = [k for k in keys if 512 <= n_new[k] < 2000]
        xs = [n_new[k] for p in passes_of[s] for k in kb]
        ys = [by_pass[p][k]["prompt_ms"] / 1000 for p in passes_of[s] for k in kb]
        if len(xs) > 2:
            mx, my = statistics.mean(xs), statistics.mean(ys)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            f.write(f"{s}\t{len(kb)}\t{len(xs)}\t{my - b * mx:.2f}\t{b * 1000:.2f}\n")

# plans.tsv: which micro-batches each pass's turn calls were read in
CLASSES = ((9, 600), (601, 1023), (1024, 2048), (2049, 10 ** 9))
with open(f"{a.out}/plans.tsv", "w") as f:
    f.write("pass\tsetting\tcall_tokens\tcalls_logged\tlargest_ubatch\tmost_common_plans\n")
    for p in order:
        calls = [c for k in keys for c in by_pass[p][k]["calls"]]
        for lo, hi in CLASSES:
            cs = [(n, pl) for n, pl in calls if lo <= n <= hi]
            if not cs:
                continue
            common = collections.Counter(f"{n}: {pl}" for n, pl in cs).most_common(2)
            f.write(f"{p}\t{setting_of[p]}\t{lo}-{hi if hi < 10 ** 9 else 'up'}\t{len(cs)}\t"
                    f"{max(max(sizes(pl)) for _, pl in cs)}\t{' | '.join(c for c, _ in common)}\n")

# decision.md
bk = [bname(lo, hi) for lo, hi in BUCKETS if (lo, hi) != (0, 10 ** 9)]
with open(f"{a.out}/decision.md", "w") as f:
    f.write(f"{len(keys)} replayed turns in every pass; the control is {a.control}, passes "
            f"{', '.join(str(p) for p in ctrl)}.\n\n")
    for s in settings[1:]:
        c = cls[(s, "all")]
        per = ", ".join(f"{b} {cls[(s, b)][0]} ({cls[(s, b)][5]:+.2f} s, spread {cls[(s, b)][6]:.2f})"
                        for b in bk if (s, b) in cls)
        f.write(f"- {s} (passes {', '.join(str(p) for p in passes_of[s])}): {', '.join(f'{x:.2f}' for x in c[1])} s a "
                f"pass against the control's {', '.join(f'{x:.2f}' for x in c[3])} ({c[5]:+.2f} s, "
                f"{c[5] / c[4] * 100:+.2f} %, spread {c[6]:.2f}) -> {c[0]} (R60's rule: {c[8]}); by new tokens: {per}\n")
print(open(f"{a.out}/decision.md").read())
