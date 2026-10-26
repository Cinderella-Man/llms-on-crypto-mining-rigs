"""A readable report of a texts.json (agent.py texts) for review: every text with its kind, how it was written (first
try, other seed, Flint's rewording), the longest run of words it shares with any text the pipeline's authors wrote
(texts_spec.json's instructions and facts, texts_spec_v1.json, the tool runner's default messages) and Flint's judge
note. Standard library only.

usage: python texts_report.py texts.json --spec texts_spec.json > report.md
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402


def longest_run(text, authored):
    """The longest run of words text shares with any of the authored strings, and its length."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    best = ""
    for s in authored:
        v = re.findall(r"[a-z0-9]+", s.lower())
        prev = [0] * (len(v) + 1)
        for i in range(1, len(w) + 1):
            cur = [0] * (len(v) + 1)
            for j in range(1, len(v) + 1):
                if w[i - 1] == v[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > len(best.split()):
                        best = " ".join(w[i - cur[j]:i])
            prev = cur
    return best, len(best.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts")
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    t = json.load(open(a.texts))
    spec = json.load(open(a.spec))
    authored = []

    def walk(x):
        if isinstance(x, str):
            authored.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    v1 = os.path.join(os.path.dirname(os.path.abspath(a.spec)), "texts_spec_v1.json")
    if os.path.exists(v1):
        walk(json.load(open(v1)))
    walk(agent.agx.DEFAULT_MESSAGES)
    authored = [s for s in authored if len(s.split()) >= 2]
    rows = [("system", s) for s in t["system"]] + [("tool", v) for v in t["tools"].values()] + \
        [("param", v) for v in t["params"].values()] + \
        [(k, v) for k in sorted(t["templates"]) for v in t["templates"][k]] + \
        [("message", v) for v in t.get("messages", {}).values()]
    worst = 0
    print(f"# Harness texts: {os.path.basename(a.texts)}\n")
    print(f"Writer: {t['writer'].get('file')} ({t['writer'].get('id')}); spec version "
          f"{t.get('spec', {}).get('version')}; failed: {[x['id'] for x in t.get('failed', [])]}\n")
    print("| kind | id | text | written | longest run shared with authored text | judge |")
    print("|---|---|---|---|---|---|")
    for kind, r in rows:
        if not r.get("text"):
            continue
        run, n = longest_run(r["text"], authored)
        worst = max(worst, n)
        how = r.get("how", "first") if len(r.get("tries", [])) <= 1 else f"{r.get('how', 'write')}, try {len(r['tries'])}"
        cell = r["text"].replace("|", "\\|").replace("\n", " ")
        judge = (r.get("judge") or "").split("\n")[-1].replace("|", "\\|")[:120]
        print(f"| {kind} | {r['id']} | {cell} | {how} | {n}: {run} | {judge} |")
    print(f"\nLongest run shared with authored text, over all texts: {worst} words.")


if __name__ == "__main__":
    main()
