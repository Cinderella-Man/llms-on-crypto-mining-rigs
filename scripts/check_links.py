#!/usr/bin/env python3
"""Every relative link in the repository's Markdown files: the target exists, and an anchor (#section) names a heading
of the target the way GitHub builds it. Links inside fenced code blocks and absolute URLs are skipped.

  python3 scripts/check_links.py [ROOT]     prints MISSING / ANCHOR lines and a count; exit status 1 if any
"""
import os
import re
import sys


def slug(heading):
    heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


_anchors = {}


def anchors(path):
    if path not in _anchors:
        found, seen = set(), {}
        try:
            for line in open(path, encoding="utf-8"):
                m = re.match(r"^#{1,6}\s+(.*)", line)
                if m:
                    s = slug(m.group(1))
                    n = seen.get(s, 0)
                    seen[s] = n + 1
                    found.add(s if n == 0 else f"{s}-{n}")
        except (OSError, UnicodeDecodeError):
            pass
        _anchors[path] = found
    return _anchors[path]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    bad = 0
    for d, _, files in os.walk(root):
        if "/.git" in d:
            continue
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            p = os.path.join(d, f)
            text = re.sub(r"```.*?```", "", open(p, encoding="utf-8", errors="replace").read(), flags=re.S)
            for m in re.finditer(r"\]\(([^)\s]+)\)", text):
                link = m.group(1)
                if re.match(r"[a-z]+:", link):
                    continue
                target, _, anchor = link.partition("#")
                tp = os.path.normpath(os.path.join(d, target)) if target else p
                if target and not os.path.exists(tp):
                    print("MISSING", p, link)
                    bad += 1
                elif anchor and tp.endswith(".md") and os.path.isfile(tp) and anchor not in anchors(tp):
                    print("ANCHOR", p, link)
                    bad += 1
    print(f"{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
