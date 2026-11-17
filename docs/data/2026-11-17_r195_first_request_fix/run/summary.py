#!/usr/bin/env python3
"""summary.py: R195's answer (Q1562), per arm (e1: GGML_CUDA_KEEPALIVE_EAGER=1, k1: =3) and depth, the first request
at the depth against its repeat, from tables.sh's tables and the deep lane's requests.csv.

  python3 -I summary.py --res RES --requests requests.csv [--depths 4096,6144,8192] [--loads e1,k1] [--tsv summary.tsv]

RES holds, per load L and depth D: passes-L-dD.tsv and .md (R192's graph_passes.py on the depth's cut of the profile),
gp-L-dD.csv (that cut, for the graph ids), tasks-L.tsv (server_graph_log.py) and first-L.md (R121's first_request.py).

The passes are paired by what they read, not by their order: in a load with several depths the first request's
60-token prompt micro-batch is written to the profile (its graph is evaluated again at the next depth), so the first
request then has one main pass more at its start than the repeat, which reads only the prompt's last 4 tokens. The
leading main passes of the first request whose graph on card 0 differs from the repeat's first pass are that
micro-batch: listed apart, and the rest paired in order. Then pass 0 is the prompt's last 4 tokens, pass 1 the first
verify pass, pass 2 the second; "three passes" is their sum. The first request's last pass (one token, kernel by
kernel) is listed apart: the profiler never writes the repeat's last pass.

Per arm and depth, also: generation ms (the server's), the generation steps of each request and first_request.py's
paired-step difference (all, steps 0 to 2, steps 3 on); the keep-alive's stops per request; the slowest evaluation of
the MTP draft that ran kernel by kernel or was captured (a wait of about 1 s shows there); the largest difference of
one paired main pass; whether the replies are the same in the pair and across the arms. Standard library only.
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def lead_offset(gp_path, main_dev=0, gap_ms=3000.0):
    rows = []
    with open(gp_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(r["device"]) == main_dev:
                    rows.append((float(r["host_ms"]), int(r["eval"]), r["graph"]))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort()
    pieces, cur, tmax = [], [], None
    for t, ev, g in rows:
        if cur and t > tmax + gap_ms:
            pieces.append(cur)
            cur, tmax = [], None
        cur.append((ev, g))
        tmax = t if tmax is None else max(tmax, t)
    if cur:
        pieces.append(cur)
    p1, p2 = sorted(pieces[0]), sorted(pieces[1])
    k = 0
    while k < len(p1) and p1[k][1] != p2[0][1]:
        k += 1
    return k


def draft_max(md_path, name):
    best = float("nan")
    rx = re.compile(r"^\| " + re.escape(name) + r" \| (\d+) \| (\w+) \| (\d+) \| ([0-9.]+) \| (-?\d+) \|$")
    with open(md_path) as f:
        for line in f:
            m = rx.match(line.strip())
            if m:
                v = float(m.group(4))
                best = v if math.isnan(best) else max(best, v)
    return best


def tasks(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            m = re.match(r"decode (\d) at (\d+)", r.get("request", ""))
            if m:
                out[(int(m.group(2)), int(m.group(1)))] = r
    return out


def first_steps(path):
    out = {}
    d = None
    with open(path) as f:
        for line in f:
            m = re.match(r"### Depth (\d+):", line)
            if m:
                d = int(m.group(1))
                continue
            if d is None:
                continue
            m = re.match(r"- generation lines: (\d+) and (\d+)", line)
            if m:
                out.setdefault(d, {}).update(steps_1=int(m.group(1)), steps_2=int(m.group(2)))
            m = re.match(r"- the span, pass 1 − pass 2, ms: all (\d+) paired steps (-?[0-9.]+); steps 0-2 (-?[0-9.]+) "
                         r"\(.*\); steps 3 on (-?[0-9.]+)", line)
            if m:
                out.setdefault(d, {}).update(paired=int(m.group(1)), paired_ms=float(m.group(2)),
                                             s02_ms=float(m.group(3)), s3_ms=float(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--depths", default="4096,6144,8192")
    ap.add_argument("--loads", default="e1,k1")
    ap.add_argument("--tsv")
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    loads = a.loads.split(",")
    req = {}
    with open(a.requests, newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "decode" and r["load"] in loads:
                req[(r["load"], int(r["depth"]), int(r["pass"]))] = r
    out = []
    for L in loads:
        tk = tasks(os.path.join(a.res, f"tasks-{L}.tsv"))
        fs = first_steps(os.path.join(a.res, f"first-{L}.md"))
        for d in depths:
            base = os.path.join(a.res, f"passes-{L}-d{d}")
            n1, n2 = f"d{d}p1", f"d{d}p2"
            k = lead_offset(os.path.join(a.res, f"gp-{L}-d{d}.csv"))
            p1, p2 = [], []
            last1 = ("", float("nan"))
            with open(base + ".tsv", newline="") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r[f"how_{n1}"]:
                        p1.append((r[f"how_{n1}"], fnum(r[f"gpu_ms_{n1}"]), int(r[f"cards_{n1}"] or 0)))
                    if r[f"how_{n2}"]:
                        p2.append((r[f"how_{n2}"], fnum(r[f"gpu_ms_{n2}"]), int(r[f"cards_{n2}"] or 0)))
                    if f"{n1} last" in r["tag"]:
                        last1 = (r[f"how_{n1}"], fnum(r[f"gpu_ms_{n1}"]))
            lead = p1[:k]
            q1 = p1[k:]
            three = [(q1[j], p2[j]) for j in range(3)]
            maxdiff, maxat = float("-inf"), -1
            for j in range(min(len(q1), len(p2))):
                if q1[j][2] == 9 and p2[j][2] == 9 and q1[j][1] - p2[j][1] > maxdiff:
                    maxdiff, maxat = q1[j][1] - p2[j][1], j
            r1, r2 = req.get((L, d, 1), {}), req.get((L, d, 2), {})
            same_pair = bool(r1.get("text_md5")) and r1.get("text_md5") == r2.get("text_md5")
            same_arms = all(req.get((o, d, p), {}).get("text_md5") == req.get((L, d, p), {}).get("text_md5")
                            for o in loads for p in (1, 2))
            t1, t2 = tk.get((d, 1), {}), tk.get((d, 2), {})
            st = fs.get(d, {})
            out.append({
                "load": L, "depth": d,
                "gen_ms_1": fnum(r1.get("gen_ms")), "gen_ms_2": fnum(r2.get("gen_ms")),
                "steps_1": st.get("steps_1", ""), "steps_2": st.get("steps_2", ""),
                "paired_steps": st.get("paired", ""), "paired_ms": st.get("paired_ms", float("nan")),
                "steps02_ms": st.get("s02_ms", float("nan")), "steps3on_ms": st.get("s3_ms", float("nan")),
                "prompt_ms_1": fnum(r1.get("prompt_ms")), "prompt_ms_2": fnum(r2.get("prompt_ms")),
                "microbatch_how": "+".join(h for h, _, _ in lead), "microbatch_ms": sum(m for _, m, _ in lead),
                "tail_how_1": three[0][0][0], "tail_ms_1": three[0][0][1], "tail_how_2": three[0][1][0],
                "tail_ms_2": three[0][1][1],
                "verify1_how_1": three[1][0][0], "verify1_ms_1": three[1][0][1], "verify1_how_2": three[1][1][0],
                "verify1_ms_2": three[1][1][1],
                "verify2_how_1": three[2][0][0], "verify2_ms_1": three[2][0][1], "verify2_how_2": three[2][1][0],
                "verify2_ms_2": three[2][1][1],
                "three_ms_1": sum(x[0][1] for x in three), "three_ms_2": sum(x[1][1] for x in three),
                "last_how_1": last1[0], "last_ms_1": last1[1],
                "max_pass_diff_ms": maxdiff, "max_pass_diff_at": maxat,
                "draft_max_ms_1": draft_max(base + ".md", n1), "draft_max_ms_2": draft_max(base + ".md", n2),
                "ka_eager_1": t1.get("ka_eager", ""), "ka_eager_2": t2.get("ka_eager", ""),
                "ka_stops_1": t1.get("ka_stops", ""), "ka_stops_2": t2.get("ka_stops", ""),
                "same_reply_pair": same_pair, "same_reply_arms": same_arms, "text_md5": r1.get("text_md5", ""),
            })
    print("# R195: the first request at a new depth against its repeat, per keep-alive arm\n")
    print("e1: GGML_CUDA_KEEPALIVE_EAGER=1 (the v4 preset as released); k1: =3 (C866). One load per arm, every value "
          "one measured request.\n")
    print("## Generation\n")
    print("| arm | depth | first, ms | repeat, ms | first − repeat | steps first / repeat | paired steps: first − repeat "
          "| of which steps 0 to 2 | steps 3 on | block stops first / repeat | same reply |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|")
    for o in out:
        same = "yes" if o["same_reply_pair"] and o["same_reply_arms"] else ("pair only" if o["same_reply_pair"] else "no")
        print(f"| {o['load']} | {o['depth']:,} | {o['gen_ms_1']:,.2f} | {o['gen_ms_2']:,.2f} | "
              f"{o['gen_ms_1'] - o['gen_ms_2']:+.2f} | {o['steps_1']} / {o['steps_2']} | {o['paired_ms']:+.2f} | "
              f"{o['steps02_ms']:+.2f} | {o['steps3on_ms']:+.2f} | {o['ka_stops_1']} / {o['ka_stops_2']} | {same} |")
    print("\n## The passes that run kernel by kernel or are captured (GPU ms summed over the nine cards)\n")
    print("How on cards 0 to 8: E kernel by kernel, C captured, R replayed.\n")
    print("| arm | depth | prompt's last 4 tokens, first | repeat | first verify, first | repeat | second verify, first "
          "| repeat | the three, first − repeat | first's last pass (1 token) | first's 60-token micro-batch | "
          "slowest draft evaluation, first, ms | largest one-pass difference, ms (pass) |")
    print("|---|---:|---|---|---|---|---|---|---:|---|---|---:|---|")
    for o in out:
        last = f"{o['last_how_1']} {o['last_ms_1']:.2f}" if o["last_how_1"].count("E") == 9 else "not written"
        mb = f"{o['microbatch_how']} {o['microbatch_ms']:,.2f}" if o["microbatch_how"] else "not written"
        print(f"| {o['load']} | {o['depth']:,} | {o['tail_how_1']} {o['tail_ms_1']:.2f} | {o['tail_how_2']} "
              f"{o['tail_ms_2']:.2f} | {o['verify1_how_1']} {o['verify1_ms_1']:.2f} | {o['verify1_how_2']} "
              f"{o['verify1_ms_2']:.2f} | {o['verify2_how_1']} {o['verify2_ms_1']:.2f} | {o['verify2_how_2']} "
              f"{o['verify2_ms_2']:.2f} | {o['three_ms_1'] - o['three_ms_2']:+.2f} | {last} | {mb} | "
              f"{o['draft_max_ms_1']:.2f} | {o['max_pass_diff_ms']:+.2f} ({o['max_pass_diff_at']}) |")
    if a.tsv:
        with open(a.tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t")
            w.writeheader()
            for o in out:
                w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in o.items()})


if __name__ == "__main__":
    main()
