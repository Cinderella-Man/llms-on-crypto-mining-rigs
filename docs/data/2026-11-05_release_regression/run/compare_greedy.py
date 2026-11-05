#!/usr/bin/env python3
"""Compares greedy_requests.py outputs of engine builds: the first file is the reference.

Per prompt and mode: whether the rendered prompt is the same (sha1, token count), the generated tokens' count, the
first position where a build's tokens leave the reference's, and from the probs requests the reference's top-2 margin
(log-probability of its first choice minus its second) at that position: a margin near zero is a near tie that a
different order of additions can flip, a large one is a real change. Also the largest difference in the chosen
token's log-probability over the shared prefix, and per build whether its pipe tokens equal its probs tokens (the
pipelined path against plain decoding). The last line is the verdict: IDENTICAL or DIFFERS.
Standard library only. usage: compare_greedy.py REF.jsonl OTHER.jsonl [...] > identity.md
"""
import json
import sys


def load(path):
    with open(path) as f:
        return {(r["name"], r["mode"]): r for r in map(json.loads, filter(str.strip, f))}


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def margin(rec, i):
    if not rec or i is None or i >= len(rec.get("probs", [])):
        return None
    top = rec["probs"][i]
    return round(top[0][1] - top[1][1], 4) if len(top) > 1 else None


def max_dlp(ra, rb):
    if not ra or not rb:
        return None
    pa, pb = ra.get("probs", []), rb.get("probs", [])
    d = first_diff(ra["tokens"], rb["tokens"])
    n = min(len(pa), len(pb), d if d is not None else len(pa))
    vals = [abs(pa[i][0][1] - pb[i][0][1]) for i in range(n) if pa[i] and pb[i]]
    return round(max(vals), 4) if vals else 0.0


def main():
    paths = sys.argv[1:]
    runs = [load(p) for p in paths]
    names = [p.rsplit("/", 1)[-1].removeprefix("greedy-").removesuffix(".jsonl") for p in paths]
    ref, rname = runs[0], names[0]
    same = True
    print(f"Reference: `{rname}`\n")
    print("| prompt | mode | build | prompt tokens | template = ref | tokens | = ref | first difference | "
          "ref margin there | max abs d logprob (shared prefix) |")
    print("|---|---|---|---:|---|---:|---|---:|---:|---:|")
    for key in sorted(ref, key=lambda k: (list(ref).index(k))):
        r0 = ref[key]
        for run, bname in zip(runs[1:], names[1:]):
            r = run.get(key)
            if r is None:
                print(f"| {key[0]} | {key[1]} | {bname} | missing | | | no | | | |")
                same = False
                continue
            tmpl = r["prompt_sha1"] == r0["prompt_sha1"] and r["prompt_n"] == r0["prompt_n"]
            d = first_diff(r0["tokens"] or [], r["tokens"] or [])
            same = same and tmpl and d is None
            m = margin(ref.get((key[0], "probs")), d) if d is not None else None
            dl = max_dlp(ref.get((key[0], "probs")), run.get((key[0], "probs")))
            print(f"| {key[0]} | {key[1]} | {bname} | {r['prompt_n']} | {'yes' if tmpl else 'NO'} | "
                  f"{len(r['tokens'] or [])} | {'yes' if d is None else 'NO'} | {'' if d is None else d} | "
                  f"{'' if m is None else m} | {'' if dl is None else dl} |")
    print("\n| build | prompt | pipe tokens = probs tokens | first difference |")
    print("|---|---|---|---:|")
    for run, bname in zip(runs, names):
        for nm in dict.fromkeys(k[0] for k in run):
            a, b = run.get((nm, "pipe")), run.get((nm, "probs"))
            if not a or not b:
                continue
            d = first_diff(a["tokens"] or [], b["tokens"] or [])
            same = same and d is None
            print(f"| {bname} | {nm} | {'yes' if d is None else 'NO'} | {'' if d is None else d} |")
    print(f"\nverdict: {'IDENTICAL' if same else 'DIFFERS'} ({len(ref)} requests, {len(runs) - 1} builds against "
          f"{rname})")


if __name__ == "__main__":
    main()
