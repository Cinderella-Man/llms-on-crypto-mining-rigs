#!/usr/bin/env python3
"""Drops texts from a texts.json so that `agent.py texts --extend` writes them again: every template of the given kinds
(after a change to their facts in texts_spec.json), the given ids, and, with --braces, every text that has braces besides
its placeholders (the check agent.py texts applies since 2026-10-26). Writes the rest to OUT and prints what it dropped.

  texts_drop.py texts-v2d/texts.json texts-v2e/base.json --kinds write_tests,fix_bug,implement --ids lines_shown --braces
"""
import argparse
import json
import re

PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def stray_braces(text):
    rest = PLACEHOLDER.sub("", text or "")
    return "{" in rest or "}" in rest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--kinds", default="", help="template kinds to drop, comma-separated")
    p.add_argument("--ids", default="", help="text ids to drop, comma-separated")
    p.add_argument("--braces", action="store_true", help="drop every text with braces besides its placeholders")
    a = p.parse_args()
    t = json.load(open(a.src))
    kinds = {k for k in a.kinds.split(",") if k}
    ids = {i for i in a.ids.split(",") if i}
    dropped = []

    def drop(r):
        return r["id"] in ids or (a.braces and stray_braces(r.get("text")))
    for k in ("tools", "params", "messages"):
        for i, r in t.get(k, {}).items():
            if r.get("text") and drop(r):
                dropped.append(i)
                t[k][i] = {"id": i, "kind": r.get("kind"), "text": None, "dropped_text": r["text"]}
    keep = []
    for r in t["system"]:
        if drop(r):
            dropped.append(r["id"])
        else:
            keep.append(r)
    t["system"] = keep
    for k, rs in t["templates"].items():
        keep = []
        for r in rs:
            if k in kinds or drop(r):
                dropped.append(r["id"])
            else:
                keep.append(r)
        t["templates"][k] = keep
    t["dropped"] = {"from": a.src, "kinds": sorted(kinds), "ids": sorted(ids), "braces": a.braces, "texts": dropped}
    with open(a.out, "w") as f:
        json.dump(t, f, ensure_ascii=False, indent=1)
    print(f"dropped {len(dropped)}: {dropped}")


if __name__ == "__main__":
    main()
