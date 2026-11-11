#!/usr/bin/env python3
"""Tables of run_draft_count.sh's loads: what each draft length yields, by draft position (registry Q1395).

    python3 draft_table.py WORK [WORK ...] --out DIR [--ref d3] [--cost COST.tsv] [--boot 2000] [--prompt-bins 4096,8192]
    python3 draft_table.py --selftest

WORK is a chain's work folder: loads.tsv and, per load <label>, replay-<label>.jsonl (bench/agent_check/replay.py's
rows with the server's timings counters), metrics-<label>.txt (the server's /metrics after the load),
server-<label>.log (LLAMA_TRACE=1's line per verify step, "accepted a/d draft tokens") and, when the dump was on,
dump-<label>.jsonl (LLAMA_SPEC_DUMP's line per verify step). Labels must differ across the folders given.

Counts, three ways and checked against each other:
- tokens a step from the timings: generated tokens over steps, a step being every token the target sampled on its own
  (generated minus accepted), pooled over a load's requests; the share of drafted tokens accepted;
- by draft position from the trace lines (each verify step's accepted a of drafted d): at position i, the steps that
  drafted i or more (drafted_i), those that reached i with every earlier drafted token accepted (reached_i), and those
  that accepted i or more (accepted_i); P_i = accepted_i / drafted_i (the first i all accepted) and the conditional
  c_i = accepted_i / reached_i;
- the server's own counter per position (/metrics, llamacpp:spec_decode_num_accepted_tokens_per_pos_total), which
  must equal accepted_i, and its verify steps; the dump's n_acc and draft length per step, which must equal the trace.
A step that rolled back through a checkpoint is counted apart (none is expected: the MTP line keeps n_rs_seq = the
draft length). Requests are matched to the trace's tasks in order, by their drafted and accepted sums.

Output in DIR: loads.tsv (one row per load), positions.tsv (per load and position), arms.tsv (per arm: each load by
value, the arm pooled), compare.tsv (each arm against --ref: the pooled difference in tokens a step with 95 % bootstrap
intervals over requests and over traces, and both arms' spreads between loads), by_context.tsv (per arm and bin of the
request's context length: tokens a step, P_i and c_i), cut.tsv (per load, the accepted
tokens + 1 per verify step of the load's draft cut to each shorter length: an estimate on the same steps, to read
against the shorter arms' own loads), requests.tsv, same_reply.tsv (requests with the same reply between each two
loads), check.tsv and summary.md. With --cost (a TSV with columns n_max and ms_step: an estimated time per step at that
draft length, e.g. from the rig's step map), arms.tsv adds est. tokens per second. Standard library only.
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

TRACE_RE = re.compile(r"\| task (\d+) \| accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?")
METRIC_RE = re.compile(r"^llamacpp:(\w+)(?:\{position=\"(\d+)\"\})?\s+(\S+)$")


# ---------------------------------------------------------------- reading


def read_loads(work):
    """loads.tsv: label, arm, draft length, start, line."""
    out = []
    p = Path(work) / "loads.tsv"
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        f = l.split("\t")
        out.append({"label": f[0], "arm": f[1], "n_max": int(f[2]) if f[2].isdigit() else None, "start": f[3],
                    "line": f[4] if len(f) > 4 else "", "work": Path(work)})
    return out


def read_metrics(path):
    m, pos = {}, {}
    if not path.exists():
        return None
    for l in path.read_text().splitlines():
        r = METRIC_RE.match(l.strip())
        if not r:
            continue
        name, p, v = r.groups()
        v = float(v)
        if p is not None:
            pos[int(p)] = int(v)
        else:
            m[name] = v
    m["per_pos"] = [pos[i] for i in sorted(pos)]
    return m


def read_trace(path):
    """Verify steps in log order: (task, accepted, drafted, restored)."""
    steps = []
    if not path.exists():
        return steps
    with path.open(errors="replace") as f:
        for l in f:
            r = TRACE_RE.search(l)
            if r:
                steps.append((int(r.group(1)), int(r.group(2)), int(r.group(3)), bool(r.group(4))))
    return steps


def read_dump(path):
    if not path.exists():
        return None
    out = []
    with path.open() as f:
        for l in f:
            if l.strip():
                s = json.loads(l)
                out.append((s.get("task"), int(s["n_acc"]), len(s["draft"]), s.get("verify"), s.get("type"),
                            len(s.get("cand") or [])))
    return out


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


# ---------------------------------------------------------------- per load


def positions(steps, n):
    """drafted_i, reached_i, accepted_i for i = 1..n from (accepted, drafted) pairs."""
    rows = []
    for i in range(1, n + 1):
        drafted = sum(1 for a, d in steps if d >= i)
        reached = sum(1 for a, d in steps if d >= i and a >= i - 1)
        accepted = sum(1 for a, d in steps if a >= i)
        rows.append({"position": i, "drafted": drafted, "reached": reached, "accepted": accepted,
                     "P": accepted / drafted if drafted else float("nan"),
                     "conditional": accepted / reached if reached else float("nan")})
    return rows


def tok_step(gen, acc):
    return gen / (gen - acc) if gen > acc else float("nan")


def load_table(ld):
    w, label = ld["work"], ld["label"]
    rows = read_rows(w / f"replay-{label}.jsonl")
    met = read_metrics(w / f"metrics-{label}.txt")
    trace = read_trace(w / f"server-{label}.log")
    dump = read_dump(w / f"dump-{label}.jsonl")
    ok_rows = [r for r in rows if not r.get("error") and r.get("status") == 200]
    gen = sum(r.get("predicted_n") or 0 for r in ok_rows)
    drafted = sum(r.get("draft_n") or 0 for r in ok_rows)
    acc = sum(r.get("draft_n_accepted") or 0 for r in ok_rows)
    live = [(a, d) for _, a, d, rs in trace if not rs]
    restores = sum(1 for *_, rs in trace if rs)
    n = ld["n_max"] or max([d for _, d in live] or [0])
    pos = positions(live, n)
    out = {"label": label, "arm": ld["arm"], "n_max": n, "requests": len(rows), "errors": len(rows) - len(ok_rows),
           "generated": gen, "drafted": drafted, "accepted": acc, "steps": gen - acc,
           "tokens_a_step": tok_step(gen, acc), "share_accepted": acc / drafted if drafted else float("nan"),
           "verify_steps": len(live), "accepted_per_verify_step": (sum(a for a, _ in live) / len(live)) if live else float("nan"),
           "restores": restores,
           "sampled_on_device": sum(r.get("sampled_on_device") or 0 for r in ok_rows),
           "grammar_rejected": sum(r.get("grammar_rejected_n") or 0 for r in ok_rows),
           "dump_steps": len(dump) if dump is not None else None}
    for p in pos:
        out[f"P{p['position']}"] = p["P"]
    checks = []

    def chk(name, ok, detail):
        checks.append({"label": label, "check": name, "verdict": "OK" if ok else "FAIL", "detail": detail})

    chk("rows", out["errors"] == 0 and len(rows) > 0, f"{len(rows)} rows, {out['errors']} with an error")
    chk("trace sums = timings", sum(a for a, _ in live) == acc and sum(d for _, d in live) == drafted,
        f"trace accepted {sum(a for a, _ in live)} drafted {sum(d for _, d in live)}; timings {acc} {drafted}")
    chk("no restore inside a step", restores == 0, f"{restores} steps restored a checkpoint")
    if met is None:
        chk("metrics", False, "no metrics file")
    else:
        mp = met["per_pos"]
        tp = [p["accepted"] for p in pos]
        mp_cut = mp + [0] * (len(tp) - len(mp))
        chk("metrics per position = trace", mp_cut[:len(tp)] == tp and all(x == 0 for x in mp_cut[len(tp):]),
            f"metrics {mp}; trace {tp}")
        chk("metrics steps and sums = trace",
            int(met.get("spec_decode_num_drafts_total", -1)) == len(live)
            and int(met.get("spec_decode_num_accepted_tokens_total", -1)) == acc
            and int(met.get("spec_decode_num_draft_tokens_total", -1)) == drafted,
            f"metrics steps {met.get('spec_decode_num_drafts_total')} accepted "
            f"{met.get('spec_decode_num_accepted_tokens_total')} drafted {met.get('spec_decode_num_draft_tokens_total')}")
    if dump is not None:
        ds = [(a, d) for _, a, d, *_ in dump]
        chk("dump = trace", ds == live, f"dump {len(ds)} steps, trace {len(live)}; the first difference at "
            f"{next((i for i, (x, y) in enumerate(zip(ds, live)) if x != y), min(len(ds), len(live)))}")
        types = sorted({t for *_, t, _ in dump if t})
        verify = sorted({v for *_, v, _, _ in dump if v})
        out["dump_type"] = ",".join(types)
        out["dump_verify"] = ",".join(verify)
        out["dump_cand_rows"] = sum(c for *_, c in dump)
    # requests to tasks: in order, by their drafted and accepted sums
    tasks, order = defaultdict(lambda: [0, 0, []]), []
    for t, a, d, rs in trace:
        if rs:
            continue
        if t not in tasks:
            order.append(t)
        tasks[t][0] += a
        tasks[t][1] += d
        tasks[t][2].append((a, d))
    req = []
    k = 0
    aligned = True
    for r in rows:
        dn, an = r.get("draft_n") or 0, r.get("draft_n_accepted") or 0
        st = []
        if dn > 0:
            if k < len(order) and tasks[order[k]][0] == an and tasks[order[k]][1] == dn:
                st = tasks[order[k]][2]
                k += 1
            else:
                aligned = False
        g = r.get("predicted_n") or 0
        rr = {"label": label, "arm": ld["arm"], "trace": r.get("trace"), "idx": r.get("idx"), "rep": r.get("rep", 0),
              "prompt_n": r.get("prompt_n"), "context": (r.get("prompt_n") or 0) + (r.get("cache_n") or 0),
              "generated": g, "drafted": dn, "accepted": an, "tokens_a_step": tok_step(g, an), "verify_steps": len(st),
              "grammar_rejected": r.get("grammar_rejected_n") or 0, "reply_md5": r.get("reply_md5")}
        for p in positions(st, n):
            rr[f"acc{p['position']}"] = p["accepted"]
            rr[f"rch{p['position']}"] = p["reached"]
            rr[f"drf{p['position']}"] = p["drafted"]
        req.append(rr)
    chk("requests = tasks", aligned and k == len(order), f"{k} of {len(order)} tasks matched to requests in order")
    for p in pos:
        p.update(label=label, arm=ld["arm"])
    cut = []
    for L in range(1, n + 1):
        cut.append({"label": label, "arm": ld["arm"], "n_max": n, "cut_to": L, "verify_steps": len(live),
                    "tokens_a_verify_step": (1 + sum(min(a, L) for a, _ in live) / len(live)) if live else float("nan")})
    return out, pos, req, cut, checks


# ---------------------------------------------------------------- arms and comparisons


def pooled(reqs):
    g = sum(r["generated"] for r in reqs)
    a = sum(r["accepted"] for r in reqs)
    return tok_step(g, a)


def boot_diff(ref, arm, key, n_boot, rng):
    """95 % interval of pooled(arm) - pooled(ref), resampling the keys both have."""
    kr, ka = defaultdict(list), defaultdict(list)
    for r in ref:
        kr[key(r)].append(r)
    for r in arm:
        ka[key(r)].append(r)
    keys = sorted(set(kr) & set(ka), key=str)
    if len(keys) < 2:
        return float("nan"), float("nan"), len(keys)
    diffs = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(pooled([r for k in pick for r in ka[k]]) - pooled([r for k in pick for r in kr[k]]))
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1], len(keys)


def analyse(works, ref=None, cost=None, n_boot=2000, seed=1, prompt_bins=(4096, 8192)):
    loads = [ld for w in works for ld in read_loads(w)]
    if len({ld["label"] for ld in loads}) != len(loads):
        raise SystemExit("two loads share a label: give each chain's folder its own labels")
    T = {"loads": [], "positions": [], "requests": [], "cut": [], "check": []}
    for ld in loads:
        out, pos, req, cut, checks = load_table(ld)
        T["loads"].append(out)
        T["positions"] += pos
        T["requests"] += req
        T["cut"] += cut
        T["check"] += checks
    arms = []
    for ld in loads:
        if ld["arm"] not in arms:
            arms.append(ld["arm"])
    ref = ref or (arms[0] if arms else None)
    by_arm = defaultdict(list)
    for r in T["requests"]:
        by_arm[r["arm"]].append(r)
    rows = []
    for arm in arms:
        ls = [o for o in T["loads"] if o["arm"] == arm]
        ts = [o["tokens_a_step"] for o in ls]
        row = {"arm": arm, "n_max": ls[0]["n_max"], "loads": len(ls),
               "tokens_a_step_each": " ".join(f"{x:.3f}" for x in ts),
               "tokens_a_step_pooled": pooled(by_arm[arm]),
               "spread": (max(ts) - min(ts)) if len(ts) > 1 else float("nan"),
               "share_accepted_each": " ".join(f"{o['share_accepted']:.4f}" for o in ls),
               "accepted_per_verify_step_each": " ".join(f"{o['accepted_per_verify_step']:.3f}" for o in ls)}
        live = [p for p in T["positions"] if p["arm"] == arm]
        for i in range(1, (row["n_max"] or 0) + 1):
            ps = [p for p in live if p["position"] == i]
            drafted = sum(p["drafted"] for p in ps)
            reached = sum(p["reached"] for p in ps)
            accepted = sum(p["accepted"] for p in ps)
            row[f"P{i}"] = accepted / drafted if drafted else float("nan")
            row[f"c{i}"] = accepted / reached if reached else float("nan")
        if cost and row["n_max"] in cost:
            row["ms_step_est"] = cost[row["n_max"]]
            row["tok_s_est"] = row["tokens_a_step_pooled"] / cost[row["n_max"]] * 1000.0
        rows.append(row)
    T["arms"] = rows
    rng = random.Random(seed)
    cmp_rows = []
    spread = {r["arm"]: r["spread"] for r in rows}
    for arm in arms:
        if arm == ref:
            continue
        d = pooled(by_arm[arm]) - pooled(by_arm[ref])
        lo_r, hi_r, n_r = boot_diff(by_arm[ref], by_arm[arm], lambda r: (r["trace"], r["idx"], r["rep"]), n_boot, rng)
        lo_t, hi_t, n_t = boot_diff(by_arm[ref], by_arm[arm], lambda r: r["trace"], n_boot, rng)
        base = pooled(by_arm[ref])
        cmp_rows.append({"arm": arm, "ref": ref, "diff": d, "diff_pct": 100.0 * d / base if base == base and base else float("nan"),
                         "lo_requests": lo_r, "hi_requests": hi_r, "requests": n_r,
                         "lo_traces": lo_t, "hi_traces": hi_t, "traces": n_t,
                         "spread_ref": spread.get(ref), "spread_arm": spread.get(arm)})
    T["compare"] = cmp_rows
    # by the request's context length (prompt tokens, cached or not)
    bins = sorted(prompt_bins or [])

    def bin_of(c):
        lo = 0
        for e in bins:
            if c < e:
                return f"{lo}-{e - 1}"
            lo = e
        return f">={lo}"

    by_ctx = []
    for arm in arms:
        n = next(r["n_max"] for r in rows if r["arm"] == arm) or 0
        groups = defaultdict(list)
        for r in by_arm[arm]:
            groups[bin_of(r["context"])].append(r)
        for key in sorted(groups, key=lambda k: min(r["context"] for r in groups[k])):
            g = groups[key]
            row = {"arm": arm, "context": key, "requests": len(g), "tokens_a_step": pooled(g),
                   "verify_steps": sum(r["verify_steps"] for r in g)}
            for i in range(1, n + 1):
                d = sum(r.get(f"drf{i}", 0) for r in g)
                c = sum(r.get(f"rch{i}", 0) for r in g)
                a = sum(r.get(f"acc{i}", 0) for r in g)
                row[f"P{i}"] = a / d if d else float("nan")
                row[f"c{i}"] = a / c if c else float("nan")
            by_ctx.append(row)
    T["by_context"] = by_ctx
    # the same reply between each two loads
    md5 = defaultdict(dict)
    for r in T["requests"]:
        md5[r["label"]][(r["trace"], r["idx"], r["rep"])] = r["reply_md5"]
    labels = [o["label"] for o in T["loads"]]
    same = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ks = set(md5[a]) & set(md5[b])
            same.append({"load_a": a, "load_b": b, "requests": len(ks),
                         "same_reply": sum(1 for k in ks if md5[a][k] == md5[b][k])})
    T["same_reply"] = same
    return T


# ---------------------------------------------------------------- output


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.4f}"
    return str(x)


def write_tsv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def summary(T):
    s = ["# Draft length by position (run_draft_count.sh)", ""]
    s.append("## Loads\n")
    s.append(md_table(T["loads"], ["label", "n_max", "requests", "generated", "drafted", "accepted", "tokens_a_step",
                                   "share_accepted", "verify_steps", "accepted_per_verify_step", "restores",
                                   "grammar_rejected"]))
    s.append("\n## Arms (pooled over their loads; each load by value)\n")
    nm = max([r["n_max"] or 0 for r in T["arms"]] or [0])
    s.append(md_table(T["arms"], ["arm", "loads", "tokens_a_step_each", "tokens_a_step_pooled", "spread"]
                      + [f"P{i}" for i in range(1, nm + 1)] + (["tok_s_est"] if any("tok_s_est" in r for r in T["arms"]) else [])))
    s.append("\n## Conditional acceptance by position (pooled)\n")
    s.append(md_table(T["arms"], ["arm"] + [f"c{i}" for i in range(1, nm + 1)]))
    if T["compare"]:
        s.append("\n## Against the reference arm\n")
        s.append(md_table(T["compare"], ["arm", "ref", "diff", "diff_pct", "lo_requests", "hi_requests", "lo_traces",
                                         "hi_traces", "spread_ref", "spread_arm"]))
    if T["by_context"]:
        s.append("\n## By the request's context length (prompt tokens, cached or not)\n")
        s.append(md_table(T["by_context"], ["arm", "context", "requests", "verify_steps", "tokens_a_step"]
                          + [f"P{i}" for i in range(1, nm + 1)]))
    s.append("\n## Checks\n")
    bad = [c for c in T["check"] if c["verdict"] != "OK"]
    s.append(f"{len(T['check']) - len(bad)} of {len(T['check'])} checks OK.\n")
    if bad:
        s.append(md_table(bad, ["label", "check", "verdict", "detail"]))
    return "\n".join(s) + "\n"


def write_all(T, out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("loads", "positions", "arms", "compare", "by_context", "cut", "requests", "same_reply", "check"):
        write_tsv(out / f"{name}.tsv", T[name])
    (out / "summary.md").write_text(summary(T))


# ---------------------------------------------------------------- self-test


def selftest(tmp):
    """Synthetic loads with known counts: every table must reproduce them, and a broken counter must be caught."""
    rng = random.Random(7)
    fails = []

    def expect(cond, what):
        if not cond:
            fails.append(what)

    def make(work, label, arm, n, reqs, accept_p, broken=None):
        work.mkdir(parents=True, exist_ok=True)
        with (work / "loads.tsv").open("a") as f:
            f.write(f"{label}\t{arm}\t{n}\t2026-01-01T00:00:00\tline\n")
        rows, log, dump = [], ["=== start"], []
        per_pos = [0] * n
        steps = drafted = accepted = 0
        task = 100
        known = []
        for t in range(reqs):
            task += 1
            g = 0
            dn = an = 0
            target = rng.randint(20, 60)
            while g < target:
                d = n if target - g > n else max(0, target - g - 1)
                if d == 0:
                    g += 1
                    continue
                a = 0
                while a < d and rng.random() < accept_p:
                    a += 1
                log.append(f"slot update_slots: id  0 | task {task} | accepted {a:2d}/{d:2d} draft tokens")
                dump.append({"task": task, "n_acc": a, "draft": [1] * d, "sampled": [1] * (a + 1), "verify": "reject",
                             "type": "draft-mtp", "cand": [[1, 2, 3]] * d})
                known.append((a, d))
                for i in range(a):
                    per_pos[i] += 1
                steps += 1
                g += a + 1
                dn += d
                an += a
            drafted += dn
            accepted += an
            rows.append({"label": label, "trace": f"x/t{t // 2}/a1/t1", "idx": t % 2, "rep": 0, "status": 200,
                         "predicted_n": g, "draft_n": dn, "draft_n_accepted": an, "reply_md5": f"m{t}",
                         "prompt_n": 100})
        if broken == "metrics":
            per_pos[0] += 1
        met = [f"llamacpp:spec_decode_num_drafts_total {steps}", f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}",
               f"llamacpp:spec_decode_num_draft_tokens_total {drafted}"]
        met += [f'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}} {v}' for i, v in enumerate(per_pos)]
        (work / f"metrics-{label}.txt").write_text("\n".join(met) + "\n")
        (work / f"server-{label}.log").write_text("\n".join(log) + "\n")
        (work / f"replay-{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        if broken == "dump":
            dump[3]["n_acc"] = dump[3]["n_acc"] + 1 if dump[3]["n_acc"] < len(dump[3]["draft"]) else 0
        (work / f"dump-{label}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in dump))
        return known, sum(r["predicted_n"] for r in rows), accepted

    w = Path(tmp) / "ok"
    k3a, g3a, a3a = make(w, "d3-1", "d3", 3, 8, 0.7)
    k5a, g5a, a5a = make(w, "d5-1", "d5", 5, 8, 0.7)
    k3b, g3b, a3b = make(w, "d3-2", "d3", 3, 8, 0.7)
    T = analyse([w], n_boot=200)
    expect(all(c["verdict"] == "OK" for c in T["check"]), f"clean loads fail a check: {[c for c in T['check'] if c['verdict'] != 'OK']}")
    L = {o["label"]: o for o in T["loads"]}
    expect(abs(L["d3-1"]["tokens_a_step"] - g3a / (g3a - a3a)) < 1e-12, "tokens a step of d3-1")
    for i in range(1, 6):
        want = sum(1 for a, d in k5a if a >= i) / sum(1 for a, d in k5a if d >= i)
        expect(abs(L["d5-1"][f"P{i}"] - want) < 1e-12, f"P{i} of d5-1")
    pr = [p for p in T["positions"] if p["label"] == "d5-1" and p["position"] == 3][0]
    expect(pr["reached"] == sum(1 for a, d in k5a if d >= 3 and a >= 2), "reached_3 of d5-1")
    arm3 = [r for r in T["arms"] if r["arm"] == "d3"][0]
    expect(abs(arm3["tokens_a_step_pooled"] - (g3a + g3b) / (g3a + g3b - a3a - a3b)) < 1e-12, "pooled d3")
    expect(arm3["loads"] == 2 and arm3["spread"] == abs(L["d3-1"]["tokens_a_step"] - L["d3-2"]["tokens_a_step"]), "d3 spread")
    cmp5 = T["compare"][0]
    arm5 = [r for r in T["arms"] if r["arm"] == "d5"][0]
    expect(cmp5["arm"] == "d5" and cmp5["ref"] == "d3"
           and abs(cmp5["diff"] - (arm5["tokens_a_step_pooled"] - arm3["tokens_a_step_pooled"])) < 1e-12
           and cmp5["lo_traces"] <= cmp5["hi_traces"], "compare row")
    cut = [c for c in T["cut"] if c["label"] == "d5-1" and c["cut_to"] == 2][0]
    expect(abs(cut["tokens_a_verify_step"] - (1 + sum(min(a, 2) for a, _ in k5a) / len(k5a))) < 1e-12, "cut to 2")
    expect(all(r["verify_steps"] > 0 for r in T["requests"]), "every request matched to its task")
    bc5 = [r for r in T["by_context"] if r["arm"] == "d5"]
    expect(len(bc5) == 1 and bc5[0]["requests"] == 8 and all(abs(bc5[0][f"P{i}"] - arm5[f"P{i}"]) < 1e-12
                                                             and abs(bc5[0][f"c{i}"] - arm5[f"c{i}"]) < 1e-12
                                                             for i in range(1, 6)),
           "one context bin holds the arm's whole count")
    sr = [s for s in T["same_reply"] if s["load_a"] == "d3-1" and s["load_b"] == "d3-2"][0]
    expect(sr["same_reply"] == sr["requests"] == 8, "same replies by md5")
    for broken, name in (("metrics", "metrics per position = trace"), ("dump", "dump = trace")):
        wb = Path(tmp) / broken
        make(wb, "d4-1", "d4", 4, 6, 0.6, broken=broken)
        Tb = analyse([wb], n_boot=50)
        expect(any(c["check"] == name and c["verdict"] == "FAIL" for c in Tb["check"]), f"a broken {broken} is not caught")
    print(f"selftest: {'OK' if not fails else 'FAILED'} ({len(fails)} failures)")
    for f in fails:
        print("  " + f)
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ref", help="the arm the others are compared with (default: the first loaded)")
    ap.add_argument("--cost", type=Path, help="TSV with n_max and ms_step (est.)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--prompt-bins", default="4096,8192", help="context-length bins of by_context.tsv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(0 if selftest(tmp) else 1)
    if not a.works or not a.out:
        ap.error("give the work folders and --out")
    cost = None
    if a.cost:
        cost = {}
        lines = [l.split("\t") for l in a.cost.read_text().splitlines() if l.strip()]
        h = lines[0]
        for f in lines[1:]:
            r = dict(zip(h, f))
            cost[int(r["n_max"])] = float(r["ms_step"])
    T = analyse(a.works, ref=a.ref, cost=cost, n_boot=a.boot,
                prompt_bins=[int(x) for x in a.prompt_bins.split(",") if x.strip()])
    write_all(T, a.out)
    bad = [c for c in T["check"] if c["verdict"] != "OK"]
    print(f"{len(T['loads'])} loads; {len(T['check']) - len(bad)} of {len(T['check'])} checks OK; tables in {a.out}")
    for c in bad:
        print(f"  {c['label']}: {c['check']}: {c['detail']}")


if __name__ == "__main__":
    main()
