# C956: compares the loads of run.sh. Per load: sizes.jsonl (client.py) and replay/requests.tsv (taper_replay.py),
# and the "ubatch taper:" lines of its server log (one per planned prompt call), matched to the requests in order by
# the server's "prompt eval time =" lines (split.txt: where the replay's part of the log starts). Checks:
#   pairs of passes (SIZE_PAIRS and REPLAY_PAIRS below, with their rules): replies, top-3 log-probabilities,
#   counters and every call's pieces;
#   the floor: in every pass with cuts=1 no piece under 9 tokens that is not a whole stretch between the in-call cuts
#   (516 and 4 tokens before the call's end) and the call's ends;
#   the plans: each logged call against the pieces C905's CPU plan check ran for its setting and size (plans-rs3.tsv
#   beside plans.tsv, Flash-Next's recurrent rule; plans.tsv's "before"/"after" where it has no row), and a call
#   without cuts=1 also against the rig's logged pieces (plans.tsv's "rig", C222's and R225's replays on nine P106s).
# Nothing timed: the replay's time columns are not read.
# Usage: python3 -I compare.py <load dir> <C905's plans.tsv> <out dir> > verdicts.tsv
#   writes <out>/sizes.tsv, replies.tsv, replay.tsv, calls.tsv
import csv
import hashlib
import json
import os
import re
import sys

ld, plans_path, out = sys.argv[1:4]
MIN = 9
N_UBATCH = 512
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"ubatch taper: (\d+) tokens, head (\d), tail (\d), (\d+) ubatches: (.*)$")


def expand(plan):
    res = []
    for t in plan.split():
        n, _, c = t.partition("x")
        res += [int(n)] * (int(c) if c else 1)
    return res


def bad_pieces(n, plan):
    # pieces under MIN that are not a whole stretch between the cuts (n - 516, n - 4) and the call's ends
    bounds = sorted({0, n} | {n - off for off in (4 + N_UBATCH, 4) if n > off})
    bad, pos = [], 0
    for s in expand(plan):
        if s < MIN:
            whole = pos in bounds and pos + s in bounds and not any(pos < b < pos + s for b in bounds)
            if not whole:
                bad.append(f"{s}@{pos}")
        pos += s
    return bad


def log_parts(load):
    path = os.path.join(ld, load, "server.log")
    if not os.path.exists(path):
        return [], []
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    try:
        s = int(open(os.path.join(ld, load, "split.txt")).read().split()[0])
    except (OSError, ValueError, IndexError):
        s = len(lines)

    def reqs(seg):
        res, pend = [], []
        for line in seg:
            line = ansi.sub("", line)
            m = pat.search(line)
            if m:
                pend.append((int(m.group(1)), m.group(5).strip()))
            elif "prompt eval time =" in line:
                res.append(pend)
                pend = []
        return res
    return reqs(lines[:s]), reqs(lines[s:])


def setting(spec, server_cuts):
    # the effective (taper, cuts) of a request
    if spec in ("-", None):
        return "short=1400:64", server_cuts
    return spec.replace(",cuts=1", ""), int(",cuts=1" in spec)


plans = {}  # (setting, call size) -> plans.tsv's row (the rig's logged pieces in "rig")
with open(plans_path) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        plans[(r["setting"], int(r["call_tokens"]))] = r
cpu = {}  # (setting, cuts, call size) -> the pieces C905's plan check ran (plans-rs3.tsv beside plans.tsv, "logged")
with open(os.path.join(os.path.dirname(plans_path), "plans-rs3.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        cpu[(r["spec"].replace(",cuts=1", ""), int(",cuts=1" in r["spec"]), int(r["call_tokens"]))] = r["logged"]


def cpu_plan(tap, cuts, n):
    if (tap, cuts, n) in cpu:
        return cpu[(tap, cuts, n)]
    r = plans.get((tap, n))
    return (r["after"] if cuts else r["before"]) if r else None


def rig_plan(tap, cuts, n):
    r = plans.get((tap, n))
    return r["rig"] if r and not cuts and r["rig"] not in ("", "-") else None


def plan_check(tap, cuts, calls):
    # per call: "ok", "differs" or "-" (no CPU plan for that size)
    res = []
    for n, p in calls:
        want = cpu_plan(tap, cuts, n)
        if want is None:
            res.append("-")
            continue
        res.append("ok" if expand(want) == expand(p) else f"differs (CPU {want})")
    return res


warn = []
loads = sorted(d for d in os.listdir(ld) if os.path.isdir(os.path.join(ld, d)))
S, R = {}, {}  # (load, pass) -> {request key: row}
for load in loads:
    lg_s, lg_r = log_parts(load)
    server_cuts = 1 if load == "headc" else 0
    p = os.path.join(ld, load, "sizes.jsonl")
    rows = [json.loads(x) for x in open(p)] if os.path.exists(p) else []
    if len(rows) != len(lg_s):
        warn.append(f"{load}: {len(rows)} sized requests, {len(lg_s)} timing lines")
    for i, r in enumerate(rows):
        r["calls"] = lg_s[i] if i < len(lg_s) else None
        r["tap"], r["cuts"] = setting(r["spec"], server_cuts)
        S.setdefault((load, r["pass"]), {})[r["req"]] = r
    p = os.path.join(ld, load, "replay", "requests.tsv")
    rows = list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []
    if rows and len(rows) != len(lg_r):
        warn.append(f"{load}: {len(rows)} replay requests, {len(lg_r)} timing lines")
    for i, r in enumerate(rows):
        r["calls"] = lg_r[i] if i < len(lg_r) else None
        r["tap"], r["cuts"] = setting(r["arm"], 0)
        R.setdefault((load, r["pass"]), {})[(r["kind"], r["task"], r["turn"])] = r


def md5(x):
    return hashlib.md5(json.dumps(x).encode()).hexdigest()[:10]


def counts(r, draft):
    c = r.get("counts") or {}
    return {k: v for k, v in c.items() if k.startswith("draft") == draft}


# tables
with open(os.path.join(out, "sizes.tsv"), "w") as f, open(os.path.join(out, "replies.tsv"), "w") as g:
    f.write("load\tpass\ttaper\treq\tn_prompt\treply_md5\tcounters\tdraft\tcalls\tpieces_under_9_beside_a_cut\tcpu_plan\n")
    g.write("load\tpass\treq\treply_token_ids\n")
    for (load, ps), rs in sorted(S.items()):
        for name, r in rs.items():
            calls = r["calls"] or []
            bad = [b for n, pl in calls for b in bad_pieces(n, pl)]
            pc = plan_check(r["tap"], r["cuts"], calls)
            f.write(f"{load}\t{ps}\t{r['tap']}{',cuts=1' if r['cuts'] else ''}\t{name}\t{r['n_prompt']}\t"
                    f"{md5(r.get('tokens')) if 'error' not in r else 'ERROR'}\t{json.dumps(counts(r, False))}\t"
                    f"{json.dumps(counts(r, True))}\t{' | '.join(f'{n}: {pl}' for n, pl in calls) or '-'}\t"
                    f"{' '.join(bad) or '-'}\t{' '.join(pc) or '-'}\n")
            g.write(f"{load}\t{ps}\t{name}\t{' '.join(map(str, r.get('tokens') or []))}\n")
with open(os.path.join(out, "replay.tsv"), "w") as f:
    f.write("load\tpass\ttaper\tkind\ttask\tturn\tn_new\tprompt_n\tcache_n\tpredicted_n\tgen_md5\tcalls\t"
            "pieces_under_9_beside_a_cut\tcpu_plan\n")
    for (load, ps), rs in sorted(R.items()):
        for (kind, task, turn), r in rs.items():
            calls = r["calls"] or []
            bad = [b for n, pl in calls for b in bad_pieces(n, pl)]
            pc = plan_check(r["tap"], r["cuts"], calls)
            f.write(f"{load}\t{ps}\t{r['arm']}\t{kind}\t{task}\t{turn}\t{r['n_new']}\t{r['prompt_n']}\t{r['cache_n']}\t"
                    f"{r['predicted_n']}\t{r['gen_md5']}\t{' | '.join(f'{n}: {pl}' for n, pl in calls) or '-'}\t"
                    f"{' '.join(bad) or '-'}\t{' '.join(pc) or '-'}\n")
# every distinct logged call: setting, size, the cards' pieces, C905's CPU plan, the rig's
calls = {}
for group in (S, R):
    for rs in group.values():
        for r in rs.values():
            for n, pl in r["calls"] or []:
                calls.setdefault((r["tap"], r["cuts"], n), set()).add(pl)
with open(os.path.join(out, "calls.tsv"), "w") as f:
    f.write("taper\tcuts\tcall_tokens\tcards\tcpu_plan_check\trig\tsame_as_cpu\tsame_as_rig\tpieces_under_9_beside_a_cut\n")
    for (tap, cuts, n), pls in sorted(calls.items()):
        want, rig = cpu_plan(tap, cuts, n), rig_plan(tap, cuts, n)
        same = "-" if want is None else ("yes" if all(expand(p) == expand(want) for p in pls) else "no")
        same_rig = "-" if rig is None else ("yes" if all(expand(p) == expand(rig) for p in pls) else "no")
        bad = sorted({b for p in pls for b in bad_pieces(n, p)})
        f.write(f"{tap}\t{cuts}\t{n}\t{' / '.join(sorted(pls))}\t{want or '-'}\t{rig or '-'}\t{same}\t{same_rig}\t"
                f"{' '.join(bad) or '-'}\n")

# verdicts
print("check\tagainst\twhat has to hold\tverdict\tdetail")
fail = 0
# Rules. "same": every request the same reply, top-3 log-probabilities, counters and pieces. "split": every request
# the same counters; a request whose prompt call (and the call that built the cache it reads on) is cut into the same
# pieces in both the same reply and top-3 log-probabilities bit for bit; a request whose pieces differ is listed, not
# judged (on the cards a call read in
# other pieces gives other low bits, and a 64-token greedy reply can then part at a near-tie: divergence.py sizes
# each one, and the pair head/s2048 against head/srv shows the same for two settings without cuts=1). The drafted and
# accepted counts are listed, not judged (a load's first request drafts differently: Q1715).
SIZE_PAIRS = [
    ("head/srv", "base/srv", "same", "short=1400:64, cuts unset: the base's replies, log-probabilities, counters and pieces"),
    ("head/s2048", "base/s2048", "same", "short=2048:64, cuts unset: the base's replies, log-probabilities, counters and pieces"),
    ("head/srv2", "head/srv", "same", "the first pass again on the same load: everything the same"),
    ("headc/srv", "head/v5c", "same", "cuts=1 set on the server against sent per request: everything the same"),
    ("head/v5c", "head/srv", "split", "short=1400:64,cuts=1 against short=1400:64: counters the same; where the pieces are the same, everything"),
    ("head/s2048c", "head/s2048", "split", "short=2048:64,cuts=1 against short=2048:64: counters the same; where the pieces are the same, everything"),
    ("head/s2048", "head/srv", "split", "two settings without cuts=1 (short=2048:64 against short=1400:64), the control for the rule"),
]
for run, ref, rule, what in SIZE_PAIRS:
    a, b = S.get(tuple(run.split("/"))), S.get(tuple(ref.split("/")))
    if not a or not b:
        print(f"sizes {run}\t{ref}\t{what}\tMISSING\t")
        fail = 1
        continue
    errs = [k for k, x in list(a.items()) + list(b.items()) if "error" in x]
    keys = sorted(set(a) & set(b), key=lambda k: list(b).index(k))
    # a request reading on top of an earlier one's cache (the continuation, the edit that restores a checkpoint)
    # counts that request's pieces too
    deps = {"cont-577": "fresh-577", "edit-1534": "fresh-1534"}
    diff_cal = [k for k in keys if a[k]["calls"] != b[k]["calls"]
                or (k in deps and deps[k] in keys and a[deps[k]]["calls"] != b[deps[k]]["calls"])]
    same_cal = [k for k in keys if k not in diff_cal]
    diff_tok = [k for k in keys if a[k].get("tokens") != b[k].get("tokens")]
    diff_cnt = [k for k in keys if counts(a[k], False) != counts(b[k], False)]
    diff_drf = [k for k in keys if counts(a[k], True) != counts(b[k], True)]
    diff_prb = [k for k in keys if a[k].get("probs") != b[k].get("probs")]
    bad_same = [k for k in same_cal if k in diff_tok or k in diff_prb]
    ok = not errs and not diff_cnt and len(keys) == len(a) == len(b)
    if rule == "same":
        ok = ok and not diff_tok and not diff_prb and not diff_cal
        detail = (f"{len(keys)} requests; replies differ: {','.join(diff_tok) or 'none'}; top-3 log-probabilities "
                  f"differ: {','.join(diff_prb) or 'none'}; counters differ: {','.join(diff_cnt) or 'none'}; pieces "
                  f"differ: {','.join(diff_cal) or 'none'}")
    else:
        ok = ok and not bad_same
        moved = [k for k in diff_cal if k in diff_tok]
        detail = (f"{len(keys)} requests; counters differ: {','.join(diff_cnt) or 'none'}; pieces the same in "
                  f"{len(same_cal)}, of which replies or top-3 log-probabilities differ in {len(bad_same)} "
                  f"({','.join(bad_same) or '-'}); pieces differ in {len(diff_cal)} ({','.join(diff_cal) or '-'}), of "
                  f"which replies differ in {len(moved)} and are the same in {len(diff_cal) - len(moved)}")
    detail += f"; drafted/accepted differ (listed only): {','.join(diff_drf) or 'none'}"
    if errs:
        detail += "; errors: " + ",".join(errs)
    print(f"sizes {run}\t{ref}\t{what}\t{'OK' if ok else 'FAIL'}\t{detail}")
    fail |= not ok

REPLAY_PAIRS = [
    ("head/0", "base/0", "same", "warm-up pass, short=1400:64: the base's replies, counters and pieces"),
    ("head/1", "base/1", "same", "short=1400:64: the base's replies, counters and pieces"),
    ("head/4", "head/1", "same", "short=1400:64, last pass against the first: everything the same"),
    ("head/3", "head/2", "same", "short=1400:64,cuts=1, second pass against the first: everything the same"),
    ("head/2", "head/1", "tokens", "short=1400:64,cuts=1 against short=1400:64: the same replies and counters"),
    ("head/3", "head/4", "tokens", "short=1400:64,cuts=1 against short=1400:64: the same replies and counters"),
]
for run, ref, rule, what in REPLAY_PAIRS:
    a, b = R.get(tuple(run.split("/"))), R.get(tuple(ref.split("/")))
    if not a or not b:
        print(f"replay {run}\t{ref}\t{what}\tMISSING\t")
        fail = 1
        continue
    keys = [k for k in a if k in b]
    dg = [k for k in keys if a[k]["gen_md5"] != b[k]["gen_md5"]]
    dc = [k for k in keys if (a[k]["prompt_n"], a[k]["cache_n"], a[k]["predicted_n"]) != (b[k]["prompt_n"], b[k]["cache_n"], b[k]["predicted_n"])]
    dp = [k for k in keys if a[k]["calls"] != b[k]["calls"]]
    ok = not dg and not dc and len(keys) == len(a) == len(b)
    if rule == "same":
        ok = ok and not dp
    fmt = lambda ks: ",".join(f"{t}/{u}" for _, t, u in ks[:8]) + (" ..." if len(ks) > 8 else "")
    detail = (f"{len(keys)} requests ({sum(1 for k in keys if k[0] == 'turn')} turns); replies differ: "
              f"{len(dg)} {fmt(dg)}; counters differ: {len(dc)} {fmt(dc)}; pieces differ: {len(dp)} requests")
    print(f"replay {run}\t{ref}\t{what}\t{'OK' if ok else 'FAIL'}\t{detail}")
    fail |= not ok

# the floor and the plans, per pass
for name, group in (("sizes", S), ("replay", R)):
    for (load, ps), rs in sorted(group.items()):
        cs = [(n, pl) for r in rs.values() for n, pl in (r["calls"] or [])]
        cuts = {r["cuts"] for r in rs.values()}
        bad = [(n, bad_pieces(n, pl)) for n, pl in cs if bad_pieces(n, pl)]
        pcs = [x for r in rs.values() for x in plan_check(r["tap"], r["cuts"], r["calls"] or [])]
        nok, ndiff = sum(1 for x in pcs if x == "ok"), sum(1 for x in pcs if x.startswith("differs"))
        rg = [(rig_plan(r["tap"], r["cuts"], n), pl) for r in rs.values() for n, pl in (r["calls"] or [])]
        rg = [(w, pl) for w, pl in rg if w is not None]
        rok = sum(1 for w, pl in rg if expand(w) == expand(pl))
        tap = sorted({r["tap"] + (",cuts=1" if r["cuts"] else "") for r in rs.values()})
        where = ", ".join("%d (%s)" % (n, " ".join(b)) for n, b in sorted(set((n, tuple(b)) for n, b in bad)))
        detail = (f"{'/'.join(tap)}: {len(rs)} requests, {len(cs)} taper lines; calls with a piece under 9 beside a "
                  f"cut: {len(bad)} ({where or '-'}); C905's CPU plan check: {nok} the same, {ndiff} differ, "
                  f"{len(pcs) - nok - ndiff} sizes it did not run"
                  + (f"; the rig's logged pieces: {rok} of {len(rg)} calls the same" if rg else ""))
        if cuts == {1}:
            ok = not bad and ndiff == 0 and len(cs) > 0
            print(f"{name} {load}/{ps}\t-\tcuts=1: no piece under 9 tokens beside a cut, every call C905's plan\t{'OK' if ok else 'FAIL'}\t{detail}")
        else:
            ok = ndiff == 0 and rok == len(rg)
            print(f"{name} {load}/{ps}\t-\twithout cuts=1: every call C905's plan and the rig's pieces\t{'OK' if ok else 'FAIL'}\t{detail}")
        fail |= not ok
for w in warn:
    print(f"warning\t-\t-\tINFO\t{w}")
print(f"all\t-\tverdict\t{'PASS' if not fail else 'FAIL'}\t")
