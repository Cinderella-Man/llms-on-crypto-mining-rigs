#!/usr/bin/env python3
"""R227 (Q1641): the verdict's tables, standard library only.

    python3 -I docs/data/2026-11-20_r227_flash_taper1400_agent/run/verdict.py OUT CONTROL_A=turns.csv \
        TAPER_A=turns.csv CONTROL_B=turns.csv TAPER_B=turns.csv

The four runs in the order they ran: the two controls are the first and third, the two `short=1400:64` runs the second
and fourth. Writes, in OUT:

  verdict_buckets.tsv  per run, every turn of the task's first attempt by its new tokens (prompt_n): up to 600 (read
                       in micro-batches of 64 by both arms), 601 to 1,400 (64s with `short=1400:64`, 512s in the
                       control), 1,401 to 2,047 and 2,048 and more (512s in both); turns, new tokens, prompt seconds,
                       ms a prompt token
  verdict_matched.tsv  the same buckets over the requests that read the same prompt on the same cached prefix
                       (prompt_n and cache_n equal, the same task and turn) in all four runs: prompt seconds per run
  verdict_turns.tsv    the matched requests one by one, prompt ms per run
  verdict_close.tsv    the close requests one by one: the same task and turn in all four runs, new tokens and cached
                       tokens within 2 % of each other (cached tokens within 2 % + 2) and the new tokens in one bucket;
                       their new tokens, cached tokens and prompt ms per run (the matched requests are among them)
  verdict_first_token.tsv  per run, the first token of every turn: median, mean and the six waits around the median

and prints, per bucket, both tables' figures with the arms' means, their difference and the controls' spread.
"""
import csv
import sys

EDGES = [(0, 600, "up to 600"), (601, 1400, "601-1,400"), (1401, 2047, "1,401-2,047"), (2048, 10**9, "2,048+")]


def load(path):
    return [r for r in csv.DictReader(open(path)) if r["attempt"] == "1" and r.get("prompt_ms")]


def bucket(n):
    for lo, hi, b in EDGES:
        if lo <= n <= hi:
            return b
    raise ValueError(n)


def main():
    out, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    if len(runs) != 4:
        sys.exit("name four runs: control a, 1400 a, control b, 1400 b")
    names = [n for n, _ in runs]
    rows = {n: load(p) for n, p in runs}
    labels = [b for _, _, b in EDGES] + ["all turns"]

    # every turn by bucket
    allb = {}
    with open(f"{out}/verdict_buckets.tsv", "w") as fh:
        fh.write("run\tbucket\tturns\tnew_tokens\tprompt_s\tms_a_token\n")
        for n in names:
            for b in labels:
                sel = [r for r in rows[n] if b == "all turns" or bucket(int(r["prompt_n"])) == b]
                tok = sum(int(r["prompt_n"]) for r in sel)
                s = sum(float(r["prompt_ms"]) for r in sel) / 1000
                ms = s * 1000 / tok if tok else 0.0
                allb[n, b] = (len(sel), tok, s, ms)
                fh.write(f"{n}\t{b}\t{len(sel)}\t{tok}\t{s:.2f}\t{ms:.3f}\n")

    # the requests matched in all four runs
    keyed = {n: {(r["task_id"], r["turn"]): r for r in rows[n]} for n in names}
    keys = sorted(set.intersection(*(set(k) for k in keyed.values())))
    matched = [k for k in keys
               if len({(keyed[n][k]["prompt_n"], keyed[n][k]["cache_n"]) for n in names}) == 1]
    with open(f"{out}/verdict_turns.tsv", "w") as fh:
        fh.write("task\tturn\tprompt_n\tcache_n\tbucket\t" + "\t".join(f"{n}_prompt_ms" for n in names) + "\n")
        for k in matched:
            r0 = keyed[names[0]][k]
            fh.write(f"{k[0]}\t{k[1]}\t{r0['prompt_n']}\t{r0['cache_n']}\t{bucket(int(r0['prompt_n']))}\t"
                     + "\t".join(f"{float(keyed[n][k]['prompt_ms']):.1f}" for n in names) + "\n")
    mb = {}
    with open(f"{out}/verdict_matched.tsv", "w") as fh:
        fh.write("bucket\trequests\tnew_tokens\t" + "\t".join(f"{n}_prompt_s" for n in names) + "\n")
        for b in labels:
            sel = [k for k in matched if b == "all turns" or bucket(int(keyed[names[0]][k]["prompt_n"])) == b]
            tok = sum(int(keyed[names[0]][k]["prompt_n"]) for k in sel)
            secs = [sum(float(keyed[n][k]["prompt_ms"]) for k in sel) / 1000 for n in names]
            mb[b] = (len(sel), tok, secs)
            fh.write(f"{b}\t{len(sel)}\t{tok}\t" + "\t".join(f"{s:.2f}" for s in secs) + "\n")

    # the close requests: the same task and turn in all four runs, new tokens and cached tokens each within 2 % of
    # each other, and the new tokens in one bucket (so both arms read them in the same micro-batches but for the taper)
    close = []
    for k in keys:
        pn = [int(keyed[n][k]["prompt_n"]) for n in names]
        cn = [int(keyed[n][k]["cache_n"]) for n in names]
        if (max(pn) <= 1.02 * min(pn) and max(cn) <= 1.02 * min(cn) + 2
                and len({bucket(p) for p in pn}) == 1):
            close.append(k)
    with open(f"{out}/verdict_close.tsv", "w") as fh:
        fh.write("task\tturn\tbucket\t" + "\t".join(f"{n}_prompt_n\t{n}_cache_n\t{n}_prompt_ms" for n in names) + "\n")
        for k in close:
            fh.write(f"{k[0]}\t{k[1]}\t{bucket(int(keyed[names[0]][k]['prompt_n']))}\t" + "\t".join(
                f"{keyed[n][k]['prompt_n']}\t{keyed[n][k]['cache_n']}\t{float(keyed[n][k]['prompt_ms']):.1f}"
                for n in names) + "\n")
    cb_ = {}
    for b in labels:
        sel = [k for k in close if b == "all turns" or bucket(int(keyed[names[0]][k]["prompt_n"])) == b]
        cb_[b] = (len(sel), [sum(float(keyed[n][k]["prompt_ms"]) for k in sel) / 1000 for n in names],
                  [sum(int(keyed[n][k]["prompt_n"]) for k in sel) for n in names])

    # the first token of every turn: median, mean, and the six waits around the median
    with open(f"{out}/verdict_first_token.tsv", "w") as fh:
        fh.write("run\tturns\tmedian_s\tmean_s\taround_the_median_s\n")
        for n in names:
            t = sorted(float(r["ttft_s"]) for r in rows[n] if r.get("ttft_s"))
            m = len(t) // 2
            med = t[m] if len(t) % 2 else (t[m - 1] + t[m]) / 2
            fh.write(f"{n}\t{len(t)}\t{med:.2f}\t{sum(t) / len(t):.3f}\t"
                     + " ".join(f"{x:.2f}" for x in t[max(0, m - 3):m + 3]) + "\n")

    ca, ta, cb, tb = names
    print("bucket | every turn, ms a prompt token: " + ", ".join(names)
          + " | 1400 - control, means | controls' spread")
    for b in labels:
        v = {n: allb[n, b][3] for n in names}
        d = (v[ta] + v[tb]) / 2 - (v[ca] + v[cb]) / 2
        print(f"{b} | " + ", ".join(f"{v[n]:.3f} ({allb[n, b][0]})" for n in names)
              + f" | {d:+.3f} | {abs(v[ca] - v[cb]):.3f}")
    print()
    print("bucket | matched requests, prompt s: " + ", ".join(names) + " | 1400 - control, means | controls' spread")
    for b in labels:
        k, tok, secs = mb[b]
        v = dict(zip(names, secs))
        d = (v[ta] + v[tb]) / 2 - (v[ca] + v[cb]) / 2
        print(f"{b} ({k} requests, {tok} tokens) | " + ", ".join(f"{v[n]:.2f}" for n in names)
              + f" | {d:+.2f} | {abs(v[ca] - v[cb]):.2f}")
    print()
    print("bucket | close requests, prompt s (new tokens): " + ", ".join(names)
          + " | 1400 - control, means | controls' spread")
    for b in labels:
        k, secs, toks = cb_[b]
        v = dict(zip(names, secs))
        d = (v[ta] + v[tb]) / 2 - (v[ca] + v[cb]) / 2
        print(f"{b} ({k} requests) | " + ", ".join(f"{s:.2f} ({t})" for s, t in zip(secs, toks))
              + f" | {d:+.2f} | {abs(v[ca] - v[cb]):.2f}")


if __name__ == "__main__":
    main()
