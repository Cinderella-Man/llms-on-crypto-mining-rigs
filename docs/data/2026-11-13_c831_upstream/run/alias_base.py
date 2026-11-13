#!/usr/bin/env python3
"""R153 (Q1463): a set recorded with the v4-rc1 build's test-backend-ops (upstream b12360 merged), made replayable by
the build before the merge (80c94fe0d), so that both builds replay the same stored inputs and CPU references and their
errors can be compared case by case.

A record names each case by its id, the first 20 hex digits of md5("<class>|<parameters>#<n>"), and the replay checks
the meta's name line against its own case string, then the tensor layout and every blob's md5. Upstream b12360 changed
two things in those names, not in the tensors: test_mul_mat's parameters end in ",offset_a=0" (the byte offset of
the weights' view, new upstream; at 0 the view is built as before), and the fused matrix-vector test class lost "vec" in its name (#26048: the op name
MUL_MAT_VEC_FUSION stays). This tool writes, for each case of SRC, the same .in and .ref (hard links) and a .meta with
the base's name, key and class under the base's id, into DST. Cases whose parameters hold another offset_a (new
upstream cases) are left out. Every source id is checked against md5 of its own key first.

Usage: python3 -I alias_base.py SRC DST    (SRC: a set folder with cases/ and manifest.tsv; DST: a new folder)
Output: DST/cases/, DST/manifest.tsv (the base's ids and names), DST/alias.tsv (source id, base id, base name),
DST/record.txt (where it came from); runs.tsv and runs-*.tsv are copied unchanged (their filters do not name offset_a).
"""
import hashlib
import os
import re
import shutil
import sys

CLASS = {  # test class (typeid name) in the v4-rc1 build -> in 80c94fe0d's
    "19test_mul_mat_fusion": "23test_mul_mat_vec_fusion",
    "24test_mul_mat_fusion_w4a8": "28test_mul_mat_vec_fusion_w4a8",
    "27test_mul_mat_fusion_overlap": "31test_mul_mat_vec_fusion_overlap",
}
OFFSET = re.compile(r",offset_a=(\d+)$")


def cid(key):
    return hashlib.md5(key.encode()).hexdigest()[:20]


def main(src, dst):
    os.makedirs(os.path.join(dst, "cases"), exist_ok=True)
    out_manifest, out_alias, left = [], [], 0
    for line in open(os.path.join(src, "manifest.tsv")):
        f = line.rstrip("\n").split("\t")
        sid = f[0]
        meta = open(os.path.join(src, "cases", sid + ".meta")).read().split("\n")
        head = {l.split("\t", 1)[0]: l.split("\t", 1)[1] for l in meta[:8] if "\t" in l}
        key, name, cls = head["key"], head["name"], head["class"]
        if cid(key) != sid:
            sys.exit(f"{sid}: md5 of its key is {cid(key)}: not this build's id scheme")
        kcls, rest = key.split("|", 1)
        params, n = rest.rsplit("#", 1)
        m = OFFSET.search(params)
        if m and m.group(1) != "0":
            left += 1
            continue
        bparams = OFFSET.sub("", params)
        bcls = CLASS.get(kcls, kcls)
        bkey = f"{bcls}|{bparams}#{n}"
        bname = OFFSET.sub("", name[:-1]) + ")" if name.endswith(")") else name
        bid = cid(bkey)
        for e in ("in", "ref"):
            p = os.path.join(dst, "cases", f"{bid}.{e}")
            if not os.path.exists(p):
                os.link(os.path.join(src, "cases", f"{sid}.{e}"), p)
        new = []
        for l in meta:
            k = l.split("\t", 1)[0]
            new.append({"name": f"name\t{bname}", "key": f"key\t{bkey}", "class": f"class\t{CLASS.get(cls, cls)}"}.get(k, l))
        open(os.path.join(dst, "cases", bid + ".meta"), "w").write("\n".join(new))
        out_manifest.append("\t".join([bid, bname] + f[2:]))
        out_alias.append(f"{sid}\t{bid}\t{bname}")
    open(os.path.join(dst, "manifest.tsv"), "w").write("".join(l + "\n" for l in sorted(out_manifest)))
    open(os.path.join(dst, "alias.tsv"), "w").write("".join(l + "\n" for l in out_alias))
    for r in os.listdir(src):
        if r.startswith("runs") and r.endswith(".tsv"):
            shutil.copy(os.path.join(src, r), os.path.join(dst, r))
    open(os.path.join(dst, "record.txt"), "w").write(
        f"alias of {src} for the base build's case names (alias_base.py): {len(out_alias)} cases, {left} left out "
        f"(an offset_a other than 0)\n" + open(os.path.join(src, "record.txt")).read())
    print(f"{dst}: {len(out_alias)} cases under the base's names, {left} left out")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
