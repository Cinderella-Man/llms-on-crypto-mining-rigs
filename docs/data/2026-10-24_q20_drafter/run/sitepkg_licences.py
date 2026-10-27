"""Q20 (2026-10-25): the third-party packages behind the site-packages agent/edit prompts: answers and answer tokens per
package (from the generation files) and each package's licence as its own installed METADATA states it
(License-Expression, License, or License :: classifiers). Run inside python:3.12-slim after
`pip install huggingface_hub pyarrow` (the environment prompts.py built the prompts in).

usage: python3 -I sitepkg_licences.py WORK_DIR
  WORK_DIR holds the drafter's prompt files and answer files (gen-<label>.jsonl).
"""
import collections
import importlib.metadata as md
import json
import sys

W = sys.argv[1]
pkg_of = {}
for f in ("prompts.jsonl", "prompts-ext.jsonl", "prompts-ext2.jsonl"):
    for line in open(f"{W}/{f}"):
        r = json.loads(line)
        s = r.get("source", "")
        if r["category"] in ("edit", "agent") and s.startswith("site-packages/"):
            pkg_of[r["id"]] = s.split("/")[1]
n, t = collections.Counter(), collections.Counter()
for g in "A B B2 B3 C C2 C2s E1 E2 F1 F2 FV3 V1".split():
    for line in open(f"{W}/gen-{g}.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        p = pkg_of.get(r["id"])
        if p:
            n[p] += 1
            t[p] += len(r["gen_tokens"])

# top-level import name -> distribution
dist_of = {}
for d in md.distributions():
    tops = (d.read_text("top_level.txt") or "").split()
    files = {str(f).split("/")[0] for f in (d.files or [])}
    for name in set(tops) | files:
        dist_of.setdefault(name.removesuffix(".py"), d)


def licence(d):
    m = d.metadata
    exp = m.get("License-Expression")
    if exp:
        return exp
    lic = (m.get("License") or "").strip().splitlines()
    cls = [c.split("::")[-1].strip() for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
    return "; ".join(x for x in [lic[0][:60] if lic else "", ", ".join(cls)] if x) or "?"


print("| Package (dir) | Distribution | Licence (METADATA) | Answers | Answer tokens |")
print("|---|---|---|---|---|")
for p, k in t.most_common():
    d = dist_of.get(p)
    print(f"| {p} | {d.metadata['Name'] + ' ' + d.version if d else '?'} | {licence(d) if d else '?'} | {n[p]} | {k / 1e6:.3f} M |")
print(f"total: {sum(n.values())} answers, {sum(t.values()) / 1e6:.2f} M answer tokens")
