#!/usr/bin/env python3
"""ckpt_compare.py: R85's checkpoint check (Q1306): the context checkpoints and restores of two rig_taper.sh loads,
request by request, from their server logs (-lv 5), with each request's label from the load's requests.tsv.

  python3 -I ckpt_compare.py --a v=c217v/load-A --b i=c217i/load-A --out .

Per request, a checkpoint is the token count the server took it at, in the order it took it: a checkpoint before a
call when it is created ("created context checkpoint N of M (... n_tokens = T"), one inside a call when it joins its
batch ("context checkpoint at n_tokens = T goes inside the call"; a later checkpoint of the same batch can evict it
before its call runs, and then it is never filled, as the base creates it and then evicts it). A restore is
"restored context checkpoint (... n_tokens = T". Requests pair in order; a request's prompt_n in requests.tsv must be
its log's "prompt eval time" count. Writes ckpt_requests.tsv (every request) and ckpt_summary.tsv (per kind and arm),
and writes and prints ckpt_summary.txt with the log-wide counts (the switch's start-up line, checkpoints not taken
inside a call, erased, moved to the host, C703's catch-up lines) and the error and warning lines by text. Standard
library only.
"""
import argparse, csv, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="name=load folder of the reference (switch off)")
ap.add_argument("--b", required=True, help="name=load folder compared with it (switch on)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CREATED = re.compile(r"created context checkpoint \d+ of \d+ \([^)]*n_tokens = (\d+)")
JOINED = re.compile(r"context checkpoint at n_tokens = (\d+) goes inside the call")
FILLED = re.compile(r"created context checkpoint \d+ of \d+ inside the call \([^)]*n_tokens = (\d+)[^)]*, ([0-9.]+) ms\)")
RESTORED = re.compile(r"restored context checkpoint \([^)]*n_tokens = (\d+)")
COUNTS = {
    "switch_on": "context checkpoints inside a prompt's decode calls",
    "switch_off_warning": "--ctx-checkpoints-incall: ",
    "not_taken_inside": "was not taken inside the call",
    "no_free_store": "inside the call - none taken",
    "erased": " erased: ",
    "moved_to_host": "moved to the host",
    "catchup_overlap": "LLAMA_MTP_CATCHUP_OVERLAP",
}
LEVEL = re.compile(r"^\s*[0-9.]+ ([EW]) (.*)$")  # -lv 5's prefix: time, level letter


def load(spec):
    name, path = spec.split("=", 1)
    rows = list(csv.DictReader(open(os.path.join(path, "requests.tsv")), delimiter="\t"))
    reqs, counts, errs = [], {k: 0 for k in COUNTS}, {}
    for line in open(os.path.join(path, "server.log"), errors="replace"):
        line = ANSI.sub("", line)
        for k, s in COUNTS.items():
            if s in line:
                counts[k] += 1
        m = LEVEL.match(line)
        if m:  # error and warning lines, counted by their text with the numbers taken out
            key = m.group(1) + " " + re.sub(r"[0-9]+(\.[0-9]+)?", "#", m.group(2).strip())[:160]
            errs[key] = errs.get(key, 0) + 1
        if "| new prompt, " in line:
            reqs.append(dict(ck=[], joined=0, filled=[], fill_ms=0.0, rs=[], pe_n=None))
            continue
        if not reqs:
            continue
        r = reqs[-1]
        m = CREATED.search(line)
        if m:
            r["ck"].append(int(m.group(1)))
        m = JOINED.search(line)
        if m:
            r["ck"].append(int(m.group(1)))
            r["joined"] += 1
        m = FILLED.search(line)
        if m:
            r["filled"].append(int(m.group(1)))
            r["fill_ms"] += float(m.group(2))
        m = RESTORED.search(line)
        if m:
            r["rs"].append(int(m.group(1)))
        if "prompt eval time =" in line:
            m = re.search(r"/\s*([0-9]+) tokens", line)
            r["pe_n"] = int(m.group(1)) if m else None
    ok = len(reqs) == len(rows) and all(q["pe_n"] in (None, int(r["prompt_n"])) for q, r in zip(reqs, rows))
    print(f"{name}: {len(reqs)} prompts in the server log, {len(rows)} requests, matched: {ok}")
    assert ok, f"{name}: the server log's prompts do not match requests.tsv"
    for q, r in zip(reqs, rows):
        q.update(kind=r["kind"], arm=r["arm"], p=r["pass"], task=r["task"], turn=r["turn"], prompt_n=r["prompt_n"])
    return name, reqs, counts, errs


na, ra, ca, ea = load(a.a)
nb, rb, cb, eb = load(a.b)
assert len(ra) == len(rb), "the two loads have different request counts"
fmt = lambda v: ",".join(map(str, v))
summ = {}
with open(f"{a.out}/ckpt_requests.tsv", "w") as f:
    f.write(f"i\tkind\tarm\tpass\ttask\tturn\tprompt_n_{na}\tprompt_n_{nb}\tcheckpoints_{na}\tcheckpoints_{nb}\t"
            f"same_checkpoints\tjoined_{nb}\tfilled_{nb}\tfill_ms_{nb}\trestored_{na}\trestored_{nb}\tsame_restores\n")
    for i, (x, y) in enumerate(zip(ra, rb)):
        assert (x["kind"], x["arm"], x["p"], x["task"], x["turn"]) == (y["kind"], y["arm"], y["p"], y["task"], y["turn"])
        sc, sr = x["ck"] == y["ck"], x["rs"] == y["rs"]
        f.write(f"{i}\t{x['kind']}\t{x['arm']}\t{x['p']}\t{x['task']}\t{x['turn']}\t{x['prompt_n']}\t{y['prompt_n']}\t"
                f"{fmt(x['ck'])}\t{fmt(y['ck'])}\t{int(sc)}\t{y['joined']}\t{len(y['filled'])}\t{y['fill_ms']:.1f}\t"
                f"{fmt(x['rs'])}\t{fmt(y['rs'])}\t{int(sr)}\n")
        s = summ.setdefault((x["kind"], x["arm"]), dict(n=0, sc=0, ck_a=0, ck_b=0, joined=0, filled=0, fill_ms=0.0,
                                                        sr=0, rs_a=0, rs_b=0, pn=0))
        s["n"] += 1
        s["sc"] += sc
        s["sr"] += sr
        s["ck_a"] += len(x["ck"])
        s["ck_b"] += len(y["ck"])
        s["joined"] += y["joined"]
        s["filled"] += len(y["filled"])
        s["fill_ms"] += y["fill_ms"]
        s["rs_a"] += len(x["rs"])
        s["rs_b"] += len(y["rs"])
        s["pn"] += x["prompt_n"] == y["prompt_n"]
lines = []
with open(f"{a.out}/ckpt_summary.tsv", "w") as f:
    f.write(f"kind\tarm\trequests\tsame_prompt_n\tsame_checkpoints\tcheckpoints_{na}\tcheckpoints_{nb}\tjoined_{nb}\t"
            f"filled_{nb}\tfill_ms_{nb}\tsame_restores\trestored_{na}\trestored_{nb}\n")
    for (k, arm), s in sorted(summ.items()):
        f.write(f"{k}\t{arm}\t{s['n']}\t{s['pn']}\t{s['sc']}\t{s['ck_a']}\t{s['ck_b']}\t{s['joined']}\t{s['filled']}\t"
                f"{s['fill_ms']:.1f}\t{s['sr']}\t{s['rs_a']}\t{s['rs_b']}\n")
        lines.append(f"{k} {arm}: {s['n']} requests; prompt_n the same in {s['pn']}; checkpoints at the same token "
                     f"counts in the same order in {s['sc']} ({s['ck_a']} in {na}, {s['ck_b']} in {nb}, of them "
                     f"{s['joined']} inside a call, {s['filled']} filled, {s['fill_ms']:.1f} ms logged); restores the "
                     f"same in {s['sr']} ({s['rs_a']} and {s['rs_b']})")
with open(f"{a.out}/ckpt_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n\n")
    for n, c, e in ((na, ca, ea), (nb, cb, eb)):
        f.write(f"{n}: " + ", ".join(f"{k} {v}" for k, v in c.items()) +
                f"; error lines {sum(v for k, v in e.items() if k[0] == 'E')}, warning lines "
                f"{sum(v for k, v in e.items() if k[0] == 'W')}, by text:\n")
        for k, v in sorted(e.items(), key=lambda kv: -kv[1])[:30]:
            f.write(f"  {v} x {k}\n")
print(open(f"{a.out}/ckpt_summary.txt").read())
