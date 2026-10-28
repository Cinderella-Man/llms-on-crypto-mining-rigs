"""Tables for Q260 from rereads.py's and retok.py's output.

    python3 -I summary.py --runs d4=out/d4.tsv,runs/d4/attempts.csv ... --retok out/retok.tsv > summary.md

A re-read is a turn whose next request restored the checkpoint at the previous prompt's end, so it processed the
whole reply again. "If exact" is the next request's first-token time with only the re-read tokens taken out
(at that request's own prompt speed).
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=turns.tsv,attempts.csv")
    ap.add_argument("--retok", required=True)
    a = ap.parse_args()
    print("| run | turn pairs | re-reads (checkpoint restores) | where the reply diverged | re-read tokens | re-read s"
          " | task time s (12 tasks) | share of task time | share of prompt time | worst first token s | same, if exact |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in a.runs:
        name, files = spec.split("=")
        tsv, att = files.split(",")
        rows = list(csv.DictReader(open(tsv), delimiter="\t"))
        rr = [r for r in rows if r["restored_pos_max"]]
        segs = sorted({r["segment"] for r in rr})
        attempts = list(csv.DictReader(open(att)))
        wall = sum(float(x["wall_s"]) for x in attempts)
        prefill = sum(float(x["prefill_s"]) for x in attempts)
        tok = sum(int(r["reread_tokens"]) for r in rr)
        sec = sum(float(r["reread_ms"]) for r in rr) / 1000
        worst = max(rows, key=lambda r: float(r["next_ttft_s"] or 0))
        w_after = float(worst["next_ttft_s"]) - float(worst["reread_ms"]) / 1000
        print(f"| {name} | {len(rows)} | {len(rr)} | {', '.join(segs)} | {tok:,} | {sec:.1f} | {wall:.0f} |"
              f" {100 * sec / wall:.1f} % | {100 * sec / prefill:.0f} % | {float(worst['next_ttft_s']):.1f} |"
              f" {w_after:.1f} |")
    print()
    rows = list(csv.DictReader(open(a.retok), delimiter="\t"))
    buckets = [(0, 256), (256, 1024), (1024, 4096), (4096, 10 ** 9)]
    print("| reply length (tokens) | turns | generated tokens | ids change in a text round trip | first change ≥ 256 tokens"
          " before the end | in the reasoning | in a tool call |")
    print("|---|---|---|---|---|---|---|")
    for lo, hi in buckets + [(0, 10 ** 9)]:
        b = [r for r in rows if lo <= int(r["n_gen"]) < hi]
        d = [r for r in b if int(r["first_diff"]) >= 0]
        far = [r for r in d if int(r["diff_from_end"]) >= 256]
        label = "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}–{hi - 1}" if hi < 10 ** 9 else f"≥ {lo}")
        print(f"| {label} | {len(b):,} | {sum(int(r['n_gen']) for r in b):,} | {len(d)} ({100 * len(d) / max(1, len(b)):.1f} %)"
              f" | {len(far)} | {sum(r['part'] == 'reasoning' for r in d)} | {sum(r['part'] == 'tool-call' for r in d)} |")


if __name__ == "__main__":
    main()
