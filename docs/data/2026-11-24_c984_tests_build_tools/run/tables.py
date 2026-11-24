#!/usr/bin/env python3
"""C984: the tables of this folder from builds.sh's $WORK/out/ (copied off the build machine).

Usage: python3 -I run/tables.py <out-dir> <this folder>
Writes tarballs.tsv (one row per build), binaries.tsv (every file in each tarball's bin/ with its bytes and md5),
compare.tsv (each binary of the old script's build against the three later builds of the same commit) and
buildinfo.txt (each build's BUILDINFO.txt lines that name it, its commit, its web page and its check tools).
"""
import pathlib
import re
import sys

ARMS = [
    ("old", "the v5 tip, the script before the change"),
    ("new", "the v5 tip, the changed script"),
    ("off", "the v5 tip, the changed script, CHECK_TOOLS=0"),
    ("dup", "the v5 tip, the changed script, EXTRA_TARGETS=\"test-llama-archs test-cuda-keepalive\""),
    ("v3", "miner-llama-v3.0, the changed script"),
    ("v2", "miner-llama-v2.0, the changed script"),
]


def pairs(path):
    """md5sum or 'stat -c "%s %n"' lines -> {name: value}"""
    out = {}
    for line in path.read_text().splitlines():
        a, b = line.split(None, 1)
        out[b.strip()] = a
    return out


def main():
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    tarballs = {}
    for line in (src / "tarballs.txt").read_text().splitlines():
        f = line.split()
        tarballs[f[0]] = f[1:]
    rows = ["arm\tbuild\trc\tseconds\ttarball\tbytes\tmd5\tfiles_in_bin\tcheck_tools_line"]
    brows = ["arm\tfile\tbytes\tmd5"]
    info = []
    md5s = {}
    for arm, what in ARMS:
        rc = (src / f"{arm}.rc").read_text()
        m = re.match(r"rc=(\d+) seconds=(\d+)", rc)
        size, md5, name = tarballs[arm]
        bm = pairs(src / f"{arm}.bin.md5")
        bs = pairs(src / f"{arm}.bin.size")
        md5s[arm] = bm
        bi = (src / f"{arm}.buildinfo").read_text()
        ct = re.search(r"^check tools: (.*)$", bi, re.M)
        rows.append("\t".join([arm, what, m.group(1), m.group(2), name, size, md5, str(len(bm)),
                               ct.group(1) if ct else "(no such line: the script before the change)"]))
        for f in sorted(bm):
            brows.append(f"{arm}\t{f}\t{bs[f]}\t{bm[f]}")
        info.append(f"== {arm}: {what}\n{bi.rstrip()}\n")
    crows = ["file\told_md5\tnew\toff\tdup"]
    names = sorted(set().union(*(md5s[a] for a in ("old", "new", "off", "dup"))))
    for f in names:
        o = md5s["old"].get(f)
        cells = []
        for a in ("new", "off", "dup"):
            v = md5s[a].get(f)
            cells.append("-" if v is None else ("only here" if o is None else ("same" if v == o else "differs")))
        crows.append("\t".join([f, o or "-"] + cells))
    (dst / "tarballs.tsv").write_text("\n".join(rows) + "\n")
    (dst / "binaries.tsv").write_text("\n".join(brows) + "\n")
    (dst / "compare.tsv").write_text("\n".join(crows) + "\n")
    (dst / "buildinfo.txt").write_text("\n".join(info))


if __name__ == "__main__":
    main()
