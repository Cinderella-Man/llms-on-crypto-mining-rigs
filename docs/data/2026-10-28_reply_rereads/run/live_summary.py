"""R17's tables (Q260, Q263): live agent-check arms against the release-line runs, from each run's own files.

    python3 -I live_summary.py --runs NAME=RUNDIR,TURNS.tsv ... > live.md

RUNDIR holds an agent-check run as release_check.sh leaves it on the client: proxy.jsonl, res/attempts.csv (or
attempts.csv) and rig/server-rel.log; TURNS.tsv is rereads.py's output for it. Per run: tasks passed, minutes per task,
prompt and decode seconds per task, first token per turn, generation speed, the checkpoint restores of the previous
prompt's end and what they re-read, the requests that kept cached ids past a token difference (LLAMA_SERVER_TEXT_REUSE)
and the end-of-prompt checkpoints per request. Then the fixed cost of a short turn: prompt time of requests that
process at most 64 tokens, and of 65-256.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=rundir,turns.tsv")
    a = ap.parse_args()
    rows_a, rows_b = [], []
    for spec in a.runs:
        name, files = spec.split("=", 1)
        rdir, tsv = files.split(",")
        rdir = Path(rdir)
        att_f = rdir / "res" / "attempts.csv" if (rdir / "res" / "attempts.csv").exists() else rdir / "attempts.csv"
        att = list(csv.DictReader(open(att_f)))
        reqs = [json.loads(l) for l in open(rdir / "proxy.jsonl") if l.strip()]
        reqs = [r for r in reqs if r.get("status") == 200 and r.get("prompt_n") is not None]
        log = (rdir / "rig" / "server-rel.log").read_text(errors="replace")
        restores = len(re.findall(r"restored context checkpoint", log))
        kept = [int(m) for m in re.findall(r"same text: (\d+) more cached tokens kept", log)]
        turns = list(csv.DictReader(open(tsv), delimiter="\t"))
        rr = [t for t in turns if t["restored_pos_max"]]
        walls = [float(x["wall_s"]) for x in att]
        n = len(att)
        ttft = [r["t_first_token"] for r in reqs if r.get("t_first_token") is not None]
        gen_n = sum(r["predicted_n"] or 0 for r in reqs)
        gen_ms = sum(r["predicted_ms"] or 0 for r in reqs)
        ck = [(r.get("timings") or {}).get("ckpt_created_n") for r in reqs]
        ck = [c for c in ck if c is not None and c >= 0]
        rows_a.append([name, f"{sum(int(x['pass']) for x in att)}/{n}", f"{pct([w / 60 for w in walls], 0.5):.2f}",
                       f"{sum(walls) / n / 60:.2f}", f"{sum(float(x['prefill_s']) for x in att) / n:.1f}",
                       f"{sum(float(x['decode_s']) for x in att) / n:.1f}", f"{1000 * gen_n / gen_ms:.1f}" if gen_ms else "",
                       f"{pct(ttft, 0.5):.2f}", f"{pct(ttft, 0.9):.2f}", f"{max(ttft):.1f}", str(restores),
                       f"{sum(int(t['reread_tokens']) for t in rr):,}", f"{sum(float(t['reread_ms']) for t in rr) / 1000:.1f}",
                       f"{len(kept)} ({sum(kept):,})", f"{sum(ck) / len(ck):.2f}" if ck else ""])
        small = [r["prompt_ms"] for r in reqs if r["prompt_n"] <= 64]
        mid = [r["prompt_ms"] for r in reqs if 64 < r["prompt_n"] <= 256]
        rows_b.append([name, str(len(small)), f"{pct(small, 0.5):.0f}", str(len(mid)), f"{pct(mid, 0.5):.0f}"])
    print("| run | passed | min per task (median) | min per task (mean) | prompt s per task | decode s per task | "
          "generation tok/s | first token p50 s | p90 s | max s | restores of the previous prompt's end | re-read tokens | "
          "re-read s | requests that kept cached ids (tokens kept) | end-of-prompt checkpoints per request |")
    print("|" + "---|" * 15)
    for r in rows_a:
        print("| " + " | ".join(r) + " |")
    print()
    print("| run | requests processing ≤ 64 tokens | their prompt ms (median) | requests processing 65–256 | their prompt ms (median) |")
    print("|---|---|---|---|---|")
    for r in rows_b:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
