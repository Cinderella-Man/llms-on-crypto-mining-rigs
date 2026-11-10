#!/usr/bin/env python3
"""lane_tables.py: R43's tables (Q1155) from rig_taper.sh's load folders (requests.tsv, checkpoints.txt,
memavail.txt, summary.txt of taper_replay.py).

  python3 -I lane_tables.py --base run1/load-A --load "A=run1/load-A" --load "B=run1/load-B" \
      --load "vram4=run2/load-A" --load "tail4096=run3/load-A" --out tables

Every measured request is a replayed turn (kind "turn") or a fresh prompt (kind "fresh"); the unmeasured prefix
requests ("reset") are left out. A turn is keyed by (task, turn) and binned by its planned new tokens (n_new) in the
agent check's size bins, so the same turns pair across loads and arms. Per load and arm a pass's prompt time is the sum
of its turns' prompt_ms. Against the base (--base, its taper-off arm), each load and arm gets:
  delta = (mean prompt time of its passes) - (the base's), per pass, and a class:
  gain when delta < -spread, loss when delta > +spread, negligible otherwise, where spread is the larger of the two
  arms' own difference between their passes (the base's off arm has two passes, off at the start and at the end).
Writes passes.tsv, bins.tsv, fresh.tsv, versus.tsv, versus_bins.tsv, loads.tsv and summary.md. Standard library only.
"""
import argparse, csv, os, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="append", required=True, help="name=folder of one load (load-A, load-B)")
ap.add_argument("--base", required=True, help="the name of the base load (its taper-off arm is the base)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

BINS = [(1, 64), (65, 255), (256, 511), (512, 1023), (1024, 1999), (2000, 10 ** 9)]


def bin_of(n):
    for lo, hi in BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10 ** 9 else 'up'}"
    return "0"


def med(v):
    return statistics.median(v) if v else float("nan")


loads = {}  # name -> dict
for spec in a.load:
    name, path = spec.split("=", 1)
    rows = list(csv.DictReader(open(os.path.join(path, "requests.tsv")), delimiter="\t"))
    for r in rows:
        for k in ("pass", "turn", "n_new", "prompt_n", "cache_n", "predicted_n"):
            r[k] = int(r[k]) if r[k] not in ("", "None") else 0
        for k in ("prompt_ms", "predicted_ms", "wall_s"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else 0.0
    ck = open(os.path.join(path, "checkpoints.txt")).read().splitlines() if os.path.exists(
        os.path.join(path, "checkpoints.txt")) else []
    mem = []
    if os.path.exists(os.path.join(path, "memavail.txt")):
        for line in open(os.path.join(path, "memavail.txt")):
            p = line.split()
            if len(p) == 2:
                mem.append(int(p[1]))
    # the server log, when there: the checkpoint time of each request, matched to requests.tsv in order
    slog = os.path.join(path, "server.log")
    if os.path.exists(slog):
        reqs = []
        for line in open(slog, errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "| new prompt, " in line:
                reqs.append(dict(ck_n=0, ck_ms=0.0, rs_n=0, rs_ms=0.0, pe_n=None))
            elif reqs and "created context checkpoint" in line:
                m = re.search(r"([0-9.]+) ms\)\s*$", line)
                reqs[-1]["ck_n"] += 1
                reqs[-1]["ck_ms"] += float(m.group(1)) if m else 0.0
            elif reqs and "restored context checkpoint" in line:
                m = re.search(r"([0-9.]+) ms\)\s*$", line)
                reqs[-1]["rs_n"] += 1
                reqs[-1]["rs_ms"] += float(m.group(1)) if m else 0.0
            elif reqs and "prompt eval time =" in line:
                m = re.search(r"/\s*([0-9]+) tokens", line)
                reqs[-1]["pe_n"] = int(m.group(1)) if m else None
        ok = len(reqs) == len(rows) and all(q["pe_n"] in (None, r["prompt_n"]) for q, r in zip(reqs, rows))
        print(f"{name}: {len(reqs)} prompts in the server log, {len(rows)} requests, matched: {ok}")
        if ok:
            for q, r in zip(reqs, rows):
                r.update(ck_n=q["ck_n"], ck_ms=q["ck_ms"], rs_n=q["rs_n"], rs_ms=q["rs_ms"])
    loads[name] = dict(path=path, rows=rows, ck=ck, mem=mem)
base = a.base
assert base in loads, f"--base {base} is not a --load"


def turns(name, arm):
    """per pass: {(task, turn): row}"""
    out = {}
    for r in loads[name]["rows"]:
        if r["kind"] == "turn" and r["arm"] == arm:
            out.setdefault(r["pass"], {})[(r["task"], r["turn"])] = r
    return out


def arms(name):
    return sorted({r["arm"] for r in loads[name]["rows"] if r["kind"] == "turn"}, key=lambda x: (x != "off", x))


# passes.tsv: one row per load, arm and pass
with open(f"{a.out}/passes.tsv", "w") as f:
    f.write("load\tarm\tpass\tturns\tprompt_tokens\tprompt_s\ttok_s\tturns_reprocessed\tcheckpoints_created\t"
            "created_s\tcheckpoints_restored\trestored_s\n")
    for name in loads:
        for arm in arms(name):
            for p, ts in sorted(turns(name, arm).items()):
                n = sum(r["prompt_n"] for r in ts.values())
                ms = sum(r["prompt_ms"] for r in ts.values())
                rep = sum(1 for r in ts.values() if r["prompt_n"] > r["n_new"] + 16)  # re-read beyond the new tokens
                ck = [sum(r.get(k, 0) for r in ts.values()) for k in ("ck_n", "ck_ms", "rs_n", "rs_ms")]
                f.write(f"{name}\t{arm}\t{p}\t{len(ts)}\t{n}\t{ms / 1000:.2f}\t{n / ms * 1000:.2f}\t{rep}\t"
                        f"{ck[0]}\t{ck[1] / 1000:.2f}\t{ck[2]}\t{ck[3] / 1000:.2f}\n")

# bins.tsv: per load, arm, bin: turns, the median prompt time over all passes, the sum per pass
with open(f"{a.out}/bins.tsv", "w") as f:
    f.write("load\tarm\tbin\tturns_per_pass\tpasses\tmedian_s\tsum_s_per_pass\tsum_s_by_pass\tnew_tokens_per_pass\n")
    for name in loads:
        for arm in arms(name):
            tp = turns(name, arm)
            for lo, hi in BINS:
                b = bin_of(lo)
                per_pass = {p: [r for r in ts.values() if bin_of(r["n_new"]) == b] for p, ts in tp.items()}
                allv = [r["prompt_ms"] for v in per_pass.values() for r in v]
                if not allv:
                    continue
                sums = [sum(r["prompt_ms"] for r in v) / 1000 for _, v in sorted(per_pass.items())]
                ntok = sum(r["n_new"] for r in next(iter(per_pass.values())))
                f.write(f"{name}\t{arm}\t{b}\t{len(next(iter(per_pass.values())))}\t{len(sums)}\t"
                        f"{med(allv) / 1000:.2f}\t{statistics.mean(sums):.2f}\t"
                        f"{', '.join(f'{s:.2f}' for s in sums)}\t{ntok}\n")

# fresh.tsv: every fresh prompt by value
with open(f"{a.out}/fresh.tsv", "w") as f:
    f.write("load\tarm\tprompt\tprompt_n\tprompt_s\ttok_s\tcheckpoints_created\tcreated_s\n")
    for name in loads:
        for r in loads[name]["rows"]:
            if r["kind"] == "fresh":
                f.write(f"{name}\t{r['arm']}\t{r['task']}\t{r['prompt_n']}\t{r['prompt_ms'] / 1000:.2f}\t"
                        f"{r['prompt_n'] / max(r['prompt_ms'], 1) * 1000:.2f}\t{r.get('ck_n', '')}\t"
                        f"{r.get('ck_ms', 0) / 1000:.2f}\n")

# loads.tsv: checkpoint lines (their count and the time each line logs) and memory
MS = re.compile(r"([0-9.]+) ms\)\s*$")
DEV = re.compile(r"on the devices ([0-9.]+) MiB")


def ck_stats(lines, kind):
    v = [float(m.group(1)) for x in lines if x.startswith(kind) for m in [MS.search(x)] if m]
    d = [float(m.group(1)) for x in lines if x.startswith(kind) for m in [DEV.search(x)] if m]
    n = sum(1 for x in lines if x.startswith(kind))
    return n, (f"{med(v):.1f}" if v else ""), (f"{sum(v) / 1000:.1f}" if v else ""), (f"{max(d):.1f}" if d else "")


with open(f"{a.out}/loads.tsv", "w") as f:
    f.write("load\tcheckpoints_created\tcreated_median_ms\tcreated_sum_s\tcreated_on_devices_mib\t"
            "checkpoints_restored\trestored_median_ms\trestored_sum_s\tlowest_memavailable_mib\tmemavail_samples\n")
    for name, L in loads.items():
        c = ck_stats(L["ck"], "created")
        r = ck_stats(L["ck"], "restored")
        f.write(f"{name}\t{c[0]}\t{c[1]}\t{c[2]}\t{c[3]}\t{r[0]}\t{r[1]}\t{r[2]}\t"
                f"{min(L['mem']) if L['mem'] else ''}\t{len(L['mem'])}\n")


# versus: each load's arms against the base's off arm, paired by (task, turn) over the turns every pass has
def pass_sums(name, arm, keys, b=None):
    out = []
    for p, ts in sorted(turns(name, arm).items()):
        out.append(sum(ts[k]["prompt_ms"] for k in keys if k in ts and (b is None or bin_of(ts[k]["n_new"]) == b)) / 1000)
    return out


bp = turns(base, "off")
keys = set.intersection(*[set(ts) for ts in bp.values()])
for name in loads:
    for arm in arms(name):
        for ts in turns(name, arm).values():
            keys &= set(ts)
keys = sorted(keys)
base_s = pass_sums(base, "off", keys)
base_mean = statistics.mean(base_s)
base_spread = max(base_s) - min(base_s)
lines = []
with open(f"{a.out}/versus.tsv", "w") as f, open(f"{a.out}/versus_bins.tsv", "w") as g:
    f.write("load\tarm\tturns\tprompt_s_by_pass\tmean_s\tbase_s_by_pass\tdelta_s\tdelta_pct\tspread_s\tclass\t"
            "greedy_same_as_base\n")
    g.write("load\tarm\tbin\tturns\tmean_s\tbase_mean_s\tdelta_s\tdelta_pct_of_bin\tdelta_pct_of_total\n")
    for name in loads:
        for arm in arms(name):
            s = pass_sums(name, arm, keys)
            m = statistics.mean(s)
            spread = max(base_spread, max(s) - min(s))
            d = m - base_mean
            cls = "base" if (name == base and arm == "off") else (
                "gain" if d < -spread else "loss" if d > spread else "negligible")
            bmd5 = {k: {ts[k]["gen_md5"] for ts in bp.values()} for k in keys}
            same = sum(1 for k in keys if all(ts[k]["gen_md5"] in bmd5[k] for ts in turns(name, arm).values()))
            f.write(f"{name}\t{arm}\t{len(keys)}\t{', '.join(f'{x:.2f}' for x in s)}\t{m:.2f}\t"
                    f"{', '.join(f'{x:.2f}' for x in base_s)}\t{d:+.2f}\t{d / base_mean * 100:+.2f}\t{spread:.2f}\t"
                    f"{cls}\t{same}/{len(keys)}\n")
            lines.append(f"- {name} {arm}: {', '.join(f'{x:.2f}' for x in s)} s per pass over {len(keys)} turns "
                         f"against the base's {', '.join(f'{x:.2f}' for x in base_s)} s: {d:+.2f} s "
                         f"({d / base_mean * 100:+.2f} %), spread {spread:.2f} s → {cls}; greedy replies as the "
                         f"base's on {same} of {len(keys)} turns")
            for lo, hi in BINS:
                b = bin_of(lo)
                kb = [k for k in keys if bin_of(bp[next(iter(bp))][k]["n_new"]) == b]
                if not kb:
                    continue
                mb = statistics.mean(pass_sums(name, arm, kb, b))
                bb = statistics.mean(pass_sums(base, "off", kb, b))
                g.write(f"{name}\t{arm}\t{b}\t{len(kb)}\t{mb:.2f}\t{bb:.2f}\t{mb - bb:+.2f}\t"
                        f"{(mb - bb) / bb * 100:+.2f}\t{(mb - bb) / base_mean * 100:+.2f}\n")

with open(f"{a.out}/summary.md", "w") as f:
    f.write(f"Base: {base}, taper off, {len(keys)} turns in every pass of every arm.\n\n")
    f.write("\n".join(lines) + "\n\n")
    for name in loads:
        sp = os.path.join(loads[name]["path"], "summary.txt")
        if os.path.exists(sp):
            f.write(f"taper_replay.py's summary, {name}:\n\n```\n{open(sp).read()}```\n\n")
print(open(f"{a.out}/summary.md").read())
