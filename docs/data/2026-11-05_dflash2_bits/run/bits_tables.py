#!/usr/bin/env python3
"""bits_tables.py: C528's (Q1017) tables from rig_bits.sh's folder.

  python3 -I bits_tables.py [--gap-s 20] [--card 8] [--boost-mhz 1600] [--main-devices 0,...,8] OUT

Reads OUT/<arm>/ for every arm folder with a phases.txt (in load order) and writes OUT/tables/:
  requests.tsv  one row per arm and request: the reply's prompt and generated counts and speeds, drafted and accepted
                tokens, the acceptance; from the adapt log the drafted steps and tokens per step (1 + accepted per
                step); from the host profile the draft pass's GPU and host time per step, the verify pass's GPU time and
                the whole step (median, p10, p90 and mean over the request's decoding steps); the steps by clock: boost
                steps (below 0.9 of the median: the first steps after a long prompt, the cards still at boost), slow
                ones (above 1.5 of it: stalls, with their excess over the median summed) and base steps (the rest:
                their median step, verify and draft GPU time, and the boost steps' draft GPU time); the time per token
                (the median, the mean and the base step over tokens per step); md5 of the reply's text and whether it
                equals the first arm's
  arms.tsv      one row per arm: the same over all its requests (speeds as sums of tokens over sums of time; time per
                token as the requests' steps times their step over their tokens), the load's time, and
                the memory: host MemAvailable and MemAvailable minus CmaFree at their lowest during the load and during
                the requests (the RAM watchdog's one-second samples), swap at its highest, the watchdog's kills, and the
                drafter's card's memory once loaded, at its highest (nvidia-smi every 10 s) and at the end; the share of
                nvidia-smi's samples during the requests with the SM clock at --boost-mhz or more, all cards and the
                drafter's card
The host profile (GGML_CUDA_PROFILE_HOST) has one line per card and graph evaluation. A main-model pass is one
evaluation on each of --main-devices in order; the evaluations between two passes are the drafter's (the line puts it
on the last card) and make the step a drafted step. Passes longer than 1 s of GPU time (prompt micro-batches) are not
steps. The profile is split by request where no evaluation starts for more than half of --gap-s seconds
(rig_bits.sh idles GAP_S before each request), the first part being the load's warm-up. Standard library only.
"""
import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def phases(d):
    ph = {"req": []}
    for line in (d / "phases.txt").read_text().split("\n"):
        f = line.split()
        if not f:
            continue
        if f[0] in ("req_start", "req_end"):
            ph["req"].append((f[0], f[1], float(f[2])))
        else:
            ph[f[0]] = float(f[1])
    return ph


def reply(d, name):
    p = d / f"reply_{name}.json"
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}, ""
    if "choices" in r:
        m = r["choices"][0].get("message", {})
        text = (m.get("reasoning_content") or "") + "\n<<content>>\n" + (m.get("content") or "")
    else:
        text = r.get("content", "")
    return r.get("timings", {}), text


def adapt_steps(d):
    """per task, in order of first appearance: (steps, drafted, accepted)"""
    tasks = {}
    p = d / "adapt.jsonl"
    if not p.exists():
        return []
    for line in p.read_text().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "step" not in r:
            continue
        t = tasks.setdefault(r["task"], [0, 0, 0])
        t[0] += 1
        t[1] += r.get("n_draft", 0)
        t[2] += r.get("n_acc", 0)
    return list(tasks.values())


def profile_steps(d, main_devs, gap_ms):
    """the profile's drafted steps, split into segments by idle gaps"""
    p = d / "host_profile.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["host_enter_ms"]), float(r["host_exit_ms"]), int(r["device"]), float(r["gpu_us"]) / 1000.0))
    rows.sort()
    seg_of, seg = [], 0
    for i, r in enumerate(rows):
        if i and r[0] - rows[i - 1][1] > gap_ms:
            seg += 1
        seg_of.append(seg)
    passes, i, n = [], 0, len(main_devs)
    while i < len(rows):
        if rows[i][2] == main_devs[0] and i + n <= len(rows) and [rows[i + k][2] for k in range(n)] == main_devs:
            passes.append((i, i + n - 1))
            i += n
        else:
            i += 1
    segs = [[] for _ in range(seg + 1)]
    for (s, e), (nxt, _) in zip(passes, passes[1:]):
        if seg_of[s] != seg_of[nxt]:
            continue
        between = rows[e + 1:nxt]
        main_gpu = sum(rows[k][3] for k in range(s, e + 1))
        if not between or main_gpu > 1000.0:
            continue
        segs[seg_of[s]].append(dict(step=rows[nxt][0] - rows[s][0], verify_gpu=main_gpu,
                                    draft_gpu=sum(b[3] for b in between), draft_host=between[-1][1] - between[0][0],
                                    evals=len(between)))
    return segs


def watchdog(d, arm):
    p = d / f"ram_watchdog-{arm}.tsv"
    rows, kills = [], 0
    if p.exists():
        for line in p.read_text().split("\n"):
            if line.startswith("#"):
                kills += " KILL " in line
                continue
            f = line.split("\t")
            if len(f) >= 6:
                rows.append((int(f[0]), int(f[1]), int(f[3]), int(f[4]), int(f[5])))
    return rows, kills


def card_mem(d, card):
    """card's memory.used (MiB) per nvidia-smi sample: [(epoch, MiB)], and once loaded / at the end"""
    out = []
    p = d / "gpu.csv"
    if p.exists():
        for line in p.read_text().split("\n"):
            f = [x.strip() for x in line.split(",")]
            if len(f) < 6 or not f[1].isdigit() or int(f[1]) != card:
                continue
            try:
                t = time.mktime(time.strptime(f[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
                out.append((t, int(f[5].split()[0])))
            except ValueError:
                pass

    def at(name):
        q = d / name
        if not q.exists():
            return ""
        for line in q.read_text().split("\n"):
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 2 and f[0] == str(card):
                return f[1].split()[0]
        return ""
    return out, at("vram_loaded.csv"), at("vram_after.csv")


def clock_share(d, t0, t1, card, boost_mhz):
    """the share of nvidia-smi's samples between t0 and t1 with the SM clock at boost_mhz or more: all cards, the card"""
    n = b = nc = bc = 0
    p = d / "gpu.csv"
    if p.exists():
        for line in p.read_text().split("\n"):
            f = [x.strip() for x in line.split(",")]
            if len(f) < 5 or not f[1].isdigit():
                continue
            try:
                t = time.mktime(time.strptime(f[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
                mhz = int(f[4].split()[0])
            except ValueError:
                continue
            if not t0 < t <= t1:
                continue
            n += 1
            b += mhz >= boost_mhz
            if int(f[1]) == card:
                nc += 1
                bc += mhz >= boost_mhz
    return (b / n if n else float("nan")), (bc / nc if nc else float("nan"))


def fmt(x, nd=1):
    if isinstance(x, float):
        return "" if x != x else f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--gap-s", type=float, default=20)
    ap.add_argument("--card", type=int, default=8)
    ap.add_argument("--boost-mhz", type=int, default=1600)
    ap.add_argument("--main-devices", default="0,1,2,3,4,5,6,7,8")
    a = ap.parse_args()
    main_devs = [int(x) for x in a.main_devices.split(",")]
    arms = sorted((d for d in a.out.iterdir() if (d / "phases.txt").exists()), key=lambda d: phases(d).get("load_start", 0))
    tdir = a.out / "tables"
    tdir.mkdir(exist_ok=True)
    first_text = {}
    rq_cols = ["arm", "request", "prompt_n", "prompt_tok_s", "predicted_n", "decode_tok_s", "draft_n", "draft_accepted",
               "acceptance", "steps", "tokens_per_step", "profiled_steps", "draft_gpu_ms", "draft_gpu_p10", "draft_gpu_p90",
               "draft_host_ms", "verify_gpu_ms", "step_ms", "step_p10", "step_p90", "step_mean_ms", "boost_steps",
               "slow_steps", "slow_excess_ms", "base_step_ms", "base_verify_gpu_ms", "base_draft_gpu_ms",
               "boost_draft_gpu_ms", "ms_per_token_median", "ms_per_token_mean", "ms_per_token_base", "text_md5",
               "same_text_as_first_arm"]
    arm_cols = ["arm", "load_s", "predicted_n", "decode_tok_s", "prompt_tok_s", "draft_n", "draft_accepted", "acceptance",
                "steps", "tokens_per_step", "profiled_steps", "draft_gpu_ms", "draft_gpu_p10", "draft_gpu_p90",
                "draft_host_ms", "verify_gpu_ms", "step_ms", "step_mean_ms", "boost_steps", "slow_steps", "slow_excess_ms",
                "base_draft_gpu_ms", "boost_draft_gpu_ms", "ms_per_token_median", "ms_per_token_mean", "ms_per_token_base",
                "texts_same_as_first_arm",
                "load_min_avail_mib", "load_min_usable_mib", "req_min_avail_mib", "req_min_usable_mib", "max_swap_used_mib",
                "max_swapout_10s_mib", "watchdog_kills", f"card{a.card}_loaded_mib", f"card{a.card}_max_mib",
                f"card{a.card}_after_mib", "boost_share_all_cards", f"boost_share_card{a.card}"]
    rq_rows, arm_rows = [], []
    for d in arms:
        arm = d.name
        ph = phases(d)
        names = [n for k, n, _ in ph["req"] if k == "req_start"]
        tasks = adapt_steps(d)
        segs = profile_steps(d, main_devs, a.gap_s * 500.0)
        req_segs = segs[-len(names):] if names and len(segs) >= len(names) else [[] for _ in names]
        tot = dict(pn=0, pms=0.0, gn=0, gms=0.0, dn=0, da=0, steps=0, acc=0, same=0)
        agg = dict(med_ms=0.0, mean_ms=0.0, base_ms=0.0, tokens=0, boost=0, slow=0, excess=0.0)
        all_steps, base_all, boost_all = [], [], []
        for i, name in enumerate(names):
            t, text = reply(d, name)
            md5 = hashlib.md5(text.encode()).hexdigest() if text else ""
            first_text.setdefault(name, md5)
            same = "yes" if md5 and md5 == first_text[name] else "no"
            st, nd, na = tasks[i] if i < len(tasks) else (0, 0, 0)
            ps = req_segs[i]
            all_steps += ps
            dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
            pn, pms, gn, gms = t.get("prompt_n", 0), t.get("prompt_ms", 0.0), t.get("predicted_n", 0), t.get("predicted_ms", 0.0)
            tot.update(pn=tot["pn"] + pn, pms=tot["pms"] + pms, gn=tot["gn"] + gn, gms=tot["gms"] + gms, dn=tot["dn"] + dn,
                       da=tot["da"] + da, steps=tot["steps"] + st, acc=tot["acc"] + na, same=tot["same"] + (same == "yes"))
            col = lambda k: [s[k] for s in ps]
            # the steps by clock: boost steps are below 0.9 of the request's median step (the first steps after a long
            # prompt, the cards still at boost), slow ones above 1.5 of it (stalls, their excess over it summed), base
            # steps the rest; time per token is a step (median, mean, base median) over the request's tokens per step,
            # and per arm the requests' steps (adapt log) times it over their tokens
            tps = 1 + na / st if st else float("nan")
            sm = med(col("step"))
            mean = sum(col("step")) / len(ps) if ps else float("nan")
            boost = [s for s in ps if s["step"] < 0.9 * sm]
            slow = [s["step"] for s in ps if s["step"] > 1.5 * sm]
            base = [s for s in ps if 0.9 * sm <= s["step"] <= 1.5 * sm]
            base_all += base
            boost_all += boost
            bm = med([s["step"] for s in base])
            if ps and st:
                agg.update(med_ms=agg["med_ms"] + st * sm, mean_ms=agg["mean_ms"] + st * mean, base_ms=agg["base_ms"] + st * bm,
                           tokens=agg["tokens"] + st + na, boost=agg["boost"] + len(boost), slow=agg["slow"] + len(slow),
                           excess=agg["excess"] + sum(x - sm for x in slow))
            rq_rows.append([arm, name, pn, fmt(1000.0 * pn / pms if pms else float("nan")), gn,
                            fmt(1000.0 * gn / gms if gms else float("nan"), 2), dn, da, fmt(da / dn if dn else float("nan"), 3),
                            st, fmt(tps, 3), len(ps), fmt(med(col("draft_gpu"))),
                            fmt(pct(col("draft_gpu"), 0.1)), fmt(pct(col("draft_gpu"), 0.9)), fmt(med(col("draft_host"))),
                            fmt(med(col("verify_gpu"))), fmt(sm), fmt(pct(col("step"), 0.1)),
                            fmt(pct(col("step"), 0.9)), fmt(mean), len(boost), len(slow), fmt(sum(x - sm for x in slow), 0),
                            fmt(bm), fmt(med([s["verify_gpu"] for s in base])), fmt(med([s["draft_gpu"] for s in base])),
                            fmt(med([s["draft_gpu"] for s in boost])), fmt(sm / tps, 2), fmt(mean / tps, 2), fmt(bm / tps, 2),
                            md5, same])
        wd, kills = watchdog(d, arm)
        ls, le = ph.get("load_start", 0), ph.get("load_end", 0)
        end = ph.get("end", float("inf"))
        load = [r for r in wd if ls <= r[0] <= le]
        reqs = [r for r in wd if le < r[0] <= end]
        mem, loaded, after = card_mem(d, a.card)
        req_mem = [m for t_, m in mem if le < t_ <= end]
        col = lambda k: [s[k] for s in all_steps]
        arm_rows.append([arm, fmt(le - ls, 0), tot["gn"], fmt(1000.0 * tot["gn"] / tot["gms"] if tot["gms"] else float("nan"), 2),
                         fmt(1000.0 * tot["pn"] / tot["pms"] if tot["pms"] else float("nan")), tot["dn"], tot["da"],
                         fmt(tot["da"] / tot["dn"] if tot["dn"] else float("nan"), 3), tot["steps"],
                         fmt(1 + tot["acc"] / tot["steps"] if tot["steps"] else float("nan"), 3), len(all_steps),
                         fmt(med(col("draft_gpu"))), fmt(pct(col("draft_gpu"), 0.1)), fmt(pct(col("draft_gpu"), 0.9)),
                         fmt(med(col("draft_host"))), fmt(med(col("verify_gpu"))), fmt(med(col("step"))),
                         fmt(sum(col("step")) / len(all_steps) if all_steps else float("nan")), agg["boost"], agg["slow"],
                         fmt(agg["excess"], 0), fmt(med([s["draft_gpu"] for s in base_all])),
                         fmt(med([s["draft_gpu"] for s in boost_all])),
                         *(fmt(agg[k] / agg["tokens"] if agg["tokens"] else float("nan"), 2) for k in ("med_ms", "mean_ms", "base_ms")),
                         f"{tot['same']}/{len(names)}",
                         min((r[1] for r in load), default=""), min((r[2] for r in load), default=""),
                         min((r[1] for r in reqs), default=""), min((r[2] for r in reqs), default=""),
                         max((r[3] for r in wd), default=""), max((r[4] for r in wd), default=""), kills,
                         loaded, max(req_mem, default=""), after,
                         *(fmt(x, 2) for x in clock_share(d, le, end, a.card, a.boost_mhz))])
        print(f"{arm}: {len(names)} requests, {len(tasks)} adapt-log tasks, {len(segs)} profile segments "
              f"(drafted steps per segment {[len(s) for s in segs]}; adapt-log steps {[t[0] for t in tasks]})")
    for fname, cols, rows in (("requests.tsv", rq_cols, rq_rows), ("arms.tsv", arm_cols, arm_rows)):
        with open(tdir / fname, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
        print(f"wrote {tdir / fname}")


if __name__ == "__main__":
    main()
