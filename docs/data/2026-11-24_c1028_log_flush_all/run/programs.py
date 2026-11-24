#!/usr/bin/env python3
# C1028 (2026-11-24): programs.tsv, the programs the change touches, from a llama.cpp clone holding BASE, COMMIT and
# UPSTREAM. Per changed file under tools/, examples/ and tests/: its kind, its program (the CMake target that builds it),
# whether the flush guard was added, how many exit() calls got a flush, whether upstream has the file and whether
# upstream's copy is the base's. Reads only git objects (no checkout).
# Usage: python3 -I programs.py <llama.cpp clone> <base> <commit> <upstream ref> > programs.tsv
import re
import subprocess
import sys
from pathlib import PurePosixPath

repo, base, commit, upstream = sys.argv[1:5]


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout if r.returncode == 0 else ""


def blob(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}", check=False)


def program(path: str) -> str:
    p = PurePosixPath(path)
    if p.parts[0] == "tests":
        return p.stem
    # the CMakeLists.txt of the file's folder or the nearest folder above it under tools/ or examples/
    for d in [p.parent, *p.parent.parents]:
        if len(d.parts) < 2:
            break
        text = blob(commit, f"{d}/CMakeLists.txt")
        if not text:
            continue
        rel = str(p.relative_to(d))
        target = None
        for ln in text.splitlines():
            m = re.match(r"\s*set\(TARGET\s+(\S+)\)", ln)
            if m:
                target = m.group(1)
            m = re.match(r"\s*add_(executable|library)\s*\(\s*(\S+)\s+(.*)", ln)
            if m and re.search(rf"(^|\s){re.escape(rel)}(\s|\)|$)", m.group(3)):
                name = target if m.group(2) == "${TARGET}" else m.group(2)
                return name[:-5] if name and name.endswith("-impl") else (name or "?")
    return "?"


files = [f for f in git("diff", "--name-only", base, commit).split() if f.split("/")[0] in ("tools", "examples", "tests")]
print("kind\tprogram\tfile\tguard added\texits flushed\tupstream has it\tupstream copy")
kinds = {"tools": "tool", "examples": "example", "tests": "test"}
for f in sorted(files, key=lambda x: (x.split("/")[0], x)):
    diff = git("diff", base, commit, "--", f)
    added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    guard = any("common_log_flush_guard" in ln for ln in added)
    exits = sum(1 for ln in added if ln.strip() == "common_log_flush(common_log_main());")
    up = blob(upstream, f)
    same = up == blob(base, f) if up else False
    print(f"{kinds[f.split('/')[0]]}\t{program(f)}\t{f}\t{'yes' if guard else 'no'}\t{exits}\t{'yes' if up else 'no (fork)'}\t"
          f"{'-' if not up else ('the base' + chr(39) + 's' if same else 'differs from the base')}")
