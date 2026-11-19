#!/usr/bin/env python3
"""r216_prompts.py: prompt time by prompt size, from the requests.tsv that r181_replans.py or r216_replans.py writes.

    python3 -I r216_prompts.py LABEL=requests.tsv ... > prompts_by_size.tsv
    python3 -I r216_prompts.py --pairs A=requests.tsv B=requests.tsv > prompt_pairs.tsv

For each run and each size of prompt (new tokens read: 1-63, 64-511, 512-1,999, 2,000 and more): the requests, their
prompt tokens and seconds, ms a token, and the re-plans of the prompt scheduler with the cards busy. With --pairs, two
runs: every prompt of 256 new tokens or more whose count of new tokens occurs once in each run (in the agent check's
fixed tasks that is most often the same turn of the same task; the text is not compared), with both runs' ms and busy
re-plans. Standard library only.
"""
import csv
import sys

SIZES = [(1, 63), (64, 511), (512, 1999), (2000, None)]


def pairs(a, b):
    (la, pa), (lb, pb) = a.split("=", 1), b.split("=", 1)
    qa = [q for q in csv.DictReader(open(pa), delimiter="\t") if q["prompt_ms"]]
    qb = [q for q in csv.DictReader(open(pb), delimiter="\t") if q["prompt_ms"]]

    def once(qs):
        by = {}
        for q in qs:
            by.setdefault(int(q["prompt_tokens"]), []).append(q)
        return {k: v[0] for k, v in by.items() if len(v) == 1 and k >= 256}
    oa, ob = once(qa), once(qb)
    print(f"prompt_tokens\t{la}_task\t{la}_ms\t{la}_busy\t{lb}_task\t{lb}_ms\t{lb}_busy\t{la}_minus_{lb}_ms")
    for k in sorted(set(oa) & set(ob)):
        x, y = oa[k], ob[k]
        print(f"{k}\t{x['task']}\t{x['prompt_ms']}\t{x['replans_cards_busy']}\t{y['task']}\t{y['prompt_ms']}\t"
              f"{y['replans_cards_busy']}\t{float(x['prompt_ms']) - float(y['prompt_ms']):.2f}")


def main():
    if sys.argv[1] == "--pairs":
        pairs(sys.argv[2], sys.argv[3])
        return
    print("run\tprompt_tokens\trequests\ttokens\tprompt_s\tms_a_token\treplans_cards_busy")
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        qs = [q for q in csv.DictReader(open(path), delimiter="\t") if q["prompt_ms"]]
        for lo, hi in SIZES + [(0, None)]:
            s = [q for q in qs if int(q["prompt_tokens"]) >= lo and (hi is None or int(q["prompt_tokens"]) <= hi)]
            tok = sum(int(q["prompt_tokens"]) for q in s)
            ms = sum(float(q["prompt_ms"]) for q in s)
            busy = sum(int(q["replans_cards_busy"]) for q in s)
            name = "all" if lo == 0 else f"{lo}-{hi}" if hi else f"{lo}+"
            print(f"{label}\t{name}\t{len(s)}\t{tok}\t{ms / 1000:.1f}\t{ms / tok:.3f}\t{busy}" if tok else
                  f"{label}\t{name}\t0\t0\t\t\t")


if __name__ == "__main__":
    main()
