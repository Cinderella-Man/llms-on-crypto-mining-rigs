#!/usr/bin/env python3
"""Joins the hard-wrapped lines of Markdown paragraphs and list items into one line each, for text that GitHub renders
with every line break kept (release notes, issues). Headings, tables, code blocks and HTML comments stay as they are.

  python3 scripts/md_unwrap.py < notes.md > notes.unwrapped.md
"""
import re
import sys

ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s")


def unwrap(lines):
    out, fence = [], False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        keep = (fence or not line.strip() or line.lstrip().startswith(("#", "|", "<!--", ">"))
                or ITEM.match(line))
        prev = out[-1] if out else ""
        joinable = (prev.strip() and not prev.lstrip().startswith(("#", "|", "```", "<!--", ">")))
        if not keep and joinable and not fence:
            out[-1] = prev.rstrip() + " " + line.strip()
        else:
            out.append(line)
    return out


if __name__ == "__main__":
    sys.stdout.write("\n".join(unwrap(sys.stdin.readlines())) + "\n")
