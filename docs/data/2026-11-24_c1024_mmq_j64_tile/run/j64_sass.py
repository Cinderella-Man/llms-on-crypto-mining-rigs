#!/usr/bin/env python3
"""C1024: the 64-column prompt kernels (mul_mat_q with J=64) of the 11 types GGML_CUDA_MMQ_J64_I64 covers, compared
across three builds: the v5 candidate (C, upstream #31352's 128-row tile), the switch's branch (H) and the reference (R,
the candidate with the tile back at 64 rows).

  j64_sass.py extract < dump > j64.txt
      dump: `cuobjdump -sass <llama-server> | c++filt`, one build's sm_61 code. Every copy of mul_mat_q<type, 64, ...> of
      the 11 types: a line '### <kernel name>\t<sha1>' (the SHA-1 sass_compare.py hash gives the copy) and then its
      instructions without addresses and encodings.
  j64_sass.py compare <dir>
      dir holds h-C.tsv, h-H.tsv, h-R.tsv (sass_compare.py hash of each build) and j64-C.txt, j64-H.txt, j64-R.txt. Writes
      <dir>/j64.tsv, one row per kernel name, and prints the verdicts. The switch's kernels are the names without the
      fused gate/up and with every later flag false: the head should hold two copies of each, the candidate's and the
      reference's.
"""
import collections
import hashlib
import os
import re
import sys

# ggml_type values (ggml/include/ggml.h at the candidate)
TYPES = {41: "Q1_0", 42: "Q2_0", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
         16: "IQ2_XXS", 18: "IQ3_XXS", 20: "IQ4_NL", 23: "IQ4_XS"}
ORDER = [41, 42, 2, 3, 6, 7, 8, 16, 18, 20, 23]
# mul_mat_q<type, J, fallback, prec_src1, has_fusion[, kq_fp32[, force_tiling]]>
KRE = re.compile(r"^void mul_mat_q<\(ggml_type\)(\d+), (\d+), (true|false), \(ggml_prec\)(\d+), (true|false)((?:, (?:true|false))*)>\(")
ENC = re.compile(r"/\*\s*0x[0-9a-f]+\s*\*/")
ADDR = re.compile(r"^/\*[0-9a-f]{4,}\*/\s*")


def parse(name):
    m = KRE.match(name)
    if not m or int(m.group(2)) != 64 or int(m.group(1)) not in TYPES:
        return None
    rest = [x.strip() for x in m.group(6).split(",") if x.strip()]
    return {"type": int(m.group(1)), "fallback": m.group(3), "prec": m.group(4), "fusion": m.group(5),
            "rest": ",".join(rest) or "-", "switch": m.group(5) == "false" and all(x == "false" for x in rest)}


def extract():
    name, lines = None, []

    def flush():
        if name is not None and parse(name):
            h = hashlib.sha1("\n".join(re.sub(r"\s+", " ", s) for s in lines).encode()).hexdigest()
            print(f"### {name}\t{h}")
            for s in lines:
                t = ENC.sub("", ADDR.sub("", s)).strip()
                if t:
                    print(t)

    for raw in sys.stdin:
        s = raw.strip()
        if s.startswith("Function : "):
            flush()
            name, lines = s[len("Function : "):], []
        elif s.startswith("code for sm_") or s.startswith("Fatbin ") or s.startswith("....."):
            flush()
            name, lines = None, []
        elif name is not None and s:
            lines.append(s)
    flush()


def load_h(path):
    out = collections.defaultdict(list)
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            r = dict(zip(hdr, ln.rstrip("\n").split("\t")))
            if parse(r["kernel"]):
                out[r["kernel"]].append((r["sha1"], int(r["instructions"])))
    return out


def load_extract(path):
    out = {}
    if not os.path.exists(path):
        return out
    cur = None
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("### "):
            name, h = ln[4:].rsplit("\t", 1)
            cur = out.setdefault(h, [])
        elif cur is not None:
            cur.append(ln)
    return out


def compare(d):
    C, H, R = (load_h(os.path.join(d, f"h-{x}.tsv")) for x in "CHR")
    XH, XR = load_extract(os.path.join(d, "j64-H.txt")), load_extract(os.path.join(d, "j64-R.txt"))
    names = sorted(set(C) | set(H) | set(R),
                   key=lambda n: (ORDER.index(parse(n)["type"]), parse(n)["fallback"], parse(n)["fusion"], n))
    cols = ["type", "row_check", "prec_src1", "fused_gate_up", "later_flags", "switch_kernel", "copies_C", "copies_H",
            "copies_R", "instructions_C", "instructions_H", "instructions_R", "H_is_C_plus_R", "H_is_C", "C_differs_from_R"]
    rows, notes = [], []
    tally = collections.Counter()
    for n in names:
        p = parse(n)
        c, h, r = C.get(n, []), H.get(n, []), R.get(n, [])
        hs, cs, rs = sorted(x[0] for x in h), sorted(x[0] for x in c), sorted(x[0] for x in r)
        plus = hs == sorted(cs + rs)
        same = hs == cs
        differ = cs != rs
        rows.append([TYPES[p["type"]], p["fallback"], p["prec"], p["fusion"], p["rest"], "yes" if p["switch"] else "no",
                     len(c), len(h), len(r), "/".join(str(x[1]) for x in c), "/".join(str(x[1]) for x in sorted(h, key=lambda x: x[1])),
                     "/".join(str(x[1]) for x in r), "yes" if plus else "no", "yes" if same else "no", "yes" if differ else "no"])
        if p["switch"]:
            tally["switch"] += 1
            tally["switch_plus"] += plus
            tally["switch_differ"] += differ
            if not plus:
                # the head's copies that are not the candidate's, line by line against the reference's copy
                for sha in set(hs) - set(cs):
                    a, b = XH.get(sha, []), XR.get(rs[0], []) if rs else []
                    nd = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
                    ex = [(x, y) for x, y in zip(a, b) if x != y][:3]
                    notes.append(f"{TYPES[p['type']]} row_check={p['fallback']}: head copy {sha[:10]} against the reference's "
                                 f"{rs[0][:10] if rs else 'none'}: {len(a)} and {len(b)} lines, {nd} differ; first: {ex}")
        else:
            tally["other"] += 1
            tally["other_same"] += same
    with open(os.path.join(d, "j64.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"J=64 kernel names of the 11 types: {len(names)}")
    print(f"the switch's kernels (no fused gate/up, later flags false): {tally['switch']}; the head holds the candidate's copy "
          f"and the reference's copy, byte for byte: {tally['switch_plus']} of {tally['switch']}; the candidate's copy "
          f"differs from the reference's: {tally['switch_differ']} of {tally['switch']}")
    print(f"other J=64 names (fused gate/up, kq_fp32, forced tiling): {tally['other']}; the head's the candidate's: "
          f"{tally['other_same']} of {tally['other']}")
    for r in rows:
        if r[5] == "yes":
            print(f"  {r[0]:8} row check {r[1]:5}: instructions C {r[9]}, H {r[10]}, R {r[11]}; H = C + R: {r[12]}")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    else:
        compare(sys.argv[2])
