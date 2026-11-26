#!/usr/bin/env python3
"""Count what each preset line does to the model's replies.

Every PRESET_ENV and PRESET_ARGS line of a preset (release/<model>-<version>/settings.env) carries a tag at the start of
its comment, `# [tag] ...`. The tags are explained in docs/reply_effects.md:

  same      the same replies as without it (checked, or it only moves work, memory, clocks or timing)
  draft     changes only the draft's guesses or how they are checked: greedy replies the same, sampled replies drawn
            from the same distribution with other random draws
  order     the same arithmetic in another order or kernel: logits move by rounding, replies can part at near-ties
  input     changes the tokens the model reads (the same text)
  lossy     rounds numbers to fewer bits by design: a small real error, measured on its own
  sampling  sampling values the replies depend on by design
  mode      a mode the replies depend on by design (for example reasoning off)

Usage:
  python3 -I scripts/preset_effects.py                 # the v5 presets, a Markdown table per preset
  python3 -I scripts/preset_effects.py --check FILE... # exit 1 if a line has no tag or an unknown one
"""
import argparse
import glob
import os
import re
import sys

TAGS = ["mode", "lossy", "input", "sampling", "order", "draft", "same"]
LINE = re.compile(r'^PRESET_(ENV|ARGS)=(.*)$')
SKIP = re.compile(r'^PRESET_(ENV|ARGS)=(\$\{PRESET_(ENV|ARGS)# \})?\s*$')
TAG = re.compile(r'^\[([a-z]+)\]\s*(.*)$')


def split_comment(rest):
    """The value and the comment of a line: the comment starts at the first '#' after whitespace outside quotes."""
    quote = None
    for i, c in enumerate(rest):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#" and i > 0 and rest[i - 1].isspace():
            return rest[:i].rstrip(), rest[i + 1:].strip()
    return rest.rstrip(), ""


def setting(kind, value):
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] == '"':
        v = v[1:-1]
    v = re.sub(r'^\$PRESET_%s\s*' % kind, "", v)
    return v


def read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            m = LINE.match(line)
            if not m or SKIP.match(line):
                continue
            value, comment = split_comment(m.group(2))
            t = TAG.match(comment)
            rows.append({
                "line": n,
                "kind": m.group(1),
                "setting": setting(m.group(1), value),
                "tag": t.group(1) if t else None,
                "note": t.group(2) if t else comment,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--check", action="store_true", help="only check the tags")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = a.files or sorted(glob.glob(os.path.join(root, "release", "*-v5", "settings.env")))
    bad = 0
    out = []
    for path in files:
        rows = read(path)
        name = os.path.relpath(path, root) if path.startswith(root) else path
        for r in rows:
            if r["tag"] not in TAGS:
                bad += 1
                print("%s:%d: %s: %s" % (name, r["line"],
                      "no tag" if r["tag"] is None else "unknown tag [%s]" % r["tag"], r["setting"][:80]),
                      file=sys.stderr)
        if a.check:
            continue
        counts = {t: sum(1 for r in rows if r["tag"] == t) for t in TAGS}
        out.append("## %s\n" % name)
        out.append("| " + " | ".join("`%s`" % t for t in TAGS) + " | lines |")
        out.append("|" + "---:|" * (len(TAGS) + 1))
        out.append("| " + " | ".join(str(counts[t]) for t in TAGS) + " | %d |\n" % len(rows))
        changing = [r for r in rows if r["tag"] in TAGS and r["tag"] != "same"]
        if changing:
            out.append("| Tag | Setting | Note |")
            out.append("|---|---|---|")
            for r in sorted(changing, key=lambda r: (TAGS.index(r["tag"]), r["line"])):
                s = r["setting"].replace("|", "\\|")
                out.append("| `%s` | `%s` | %s |" % (r["tag"], s, r["note"].replace("|", "\\|")))
            out.append("")
    if out:
        print("\n".join(out))
    if bad:
        print("%d line(s) without a valid tag" % bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
