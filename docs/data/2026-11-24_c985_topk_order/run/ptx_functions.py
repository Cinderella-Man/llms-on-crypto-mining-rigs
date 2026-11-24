#!/usr/bin/env python3
"""C985: compare two builds' PTX function by function.

Usage: python3 -I ptx_functions.py <base ptx dump> <head ptx dump>
A dump is `cuobjdump -ptx <binary>`. Each function (`.entry` or `.func`, from its header line to the `}` that closes it
at column 0) is hashed with SHA-1; every other line (module headers, `.version`, `.target`, global and shared
declarations) counts as a module-level line. Prints how many functions each build has, the functions (name and hash)
only in one build, and the module-level lines only in one build (as multisets).
"""
import collections
import hashlib
import re
import sys

HEAD = re.compile(r"^\s*(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z0-9_$.]+)")


def parse(path):
    funcs = collections.Counter()
    other = collections.Counter()
    name, body, opened = None, [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if name is None:
                m = HEAD.match(line)
                if m and not line.rstrip().endswith(";"):
                    name, body, opened = m.group(1), [line], False
                    continue
                other[line] += 1
                continue
            body.append(line)
            if not opened:
                if line == "{":
                    opened = True  # a definition: its body runs to the "}" at column 0
                elif line.rstrip().endswith(";"):
                    for ln in body:  # a declaration over several lines, ended by ";"
                        other[ln] += 1
                    name, body = None, []
            elif line == "}":
                funcs[(name, hashlib.sha1("\n".join(body).encode()).hexdigest())] += 1
                name, body = None, []
    return funcs, other


def main():
    fb, ob = parse(sys.argv[1])
    fh, oh = parse(sys.argv[2])
    print(f"PTX functions: base {sum(fb.values())}, head {sum(fh.values())}")
    for label, a, b in (("only in base", fb, fh), ("only in head", fh, fb)):
        d = a - b
        print(f"functions {label}: {sum(d.values())}")
        for (n, h), c in sorted(d.items()):
            print(f"  {n}\t{h[:12]}\tx{c}")
    for label, a, b in (("only in base", ob, oh), ("only in head", oh, ob)):
        d = a - b
        print(f"module-level lines {label}: {sum(d.values())}")
        for line, c in sorted(d.items()):
            print(f"  {line[:160]}\tx{c}")


if __name__ == "__main__":
    main()
