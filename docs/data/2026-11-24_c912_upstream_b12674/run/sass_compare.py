#!/usr/bin/env python3
# C861 (a): does GGML_CUDA_MMQ_KQ_FP32SCALE leave the switched-off code as it was, and what does the FP32 form change?
# (C912, 2026-11-24: norm() also reads mul_mat_q's seventh template argument, upstream b12674's force_tiling.)
#   sass_compare.py hash < dump > table.tsv
#       dump: `cuobjdump -sass <binary> | c++filt` (one binary's sm_61 code). One row per kernel: its name (for
#       mul_mat_q, the kq_fp32 template flag split off into its own column), a SHA-1 of its SASS text, the instruction
#       count and the counts of XMAD, I2F, IDP (dp4a), FFMA, FMUL and FADD.
#   sass_compare.py compare base.tsv new.tsv OUT_PREFIX
#       OUT_PREFIX.identity.tsv: every base kernel against the new build's kernel of the same name with kq_fp32 off
#       (same SHA-1 or not); OUT_PREFIX.kq.tsv: each K-quant mul_mat_q kernel, its counts off and on.
#       Prints a summary.
import sys, re, hashlib, collections

OPS = ("XMAD", "I2F", "IDP", "FFMA", "FMUL", "FADD")
INSTR = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)")
KQ_TYPES = {"(ggml_type)11": "Q3_K", "(ggml_type)12": "Q4_K", "(ggml_type)13": "Q5_K", "(ggml_type)14": "Q6_K"}


def split_args(s):
    """Split a template argument list at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def norm(name):
    """mul_mat_q<type, J, fallback, prec, has_fusion[, kq_fp32[, force_tiling]]>(...) -> (name without kq_fp32, kq_fp32).
    force_tiling (upstream b12674's seventh argument, used on GCN only) stays in the name when true, so its kernels
    count as new ones; false, the name is the one a six-argument build gives."""
    m = re.match(r"^(void mul_mat_q<)(.*?)(>\(.*)$", name)
    if not m:
        return name, "-"
    args = split_args(m.group(2))
    if len(args) == 7:
        tail = ", force_tiling=true" if args[6] == "true" else ""
        return m.group(1) + ", ".join(args[:5]) + tail + m.group(3), args[5]
    if len(args) == 6:
        return m.group(1) + ", ".join(args[:5]) + m.group(3), args[5]
    return name, "false"


def hash_mode():
    rows = []
    name, lines = None, []

    def flush():
        if name is None:
            return
        counts = collections.Counter()
        n = 0
        for ln in lines:
            mm = INSTR.match(ln)
            if mm:
                n += 1
                op = mm.group(1)
                for o in OPS:
                    if op == o or op.startswith(o):
                        counts[o] += 1
        h = hashlib.sha1("\n".join(lines).encode()).hexdigest()
        base, kq = norm(name)
        rows.append([base, kq, h, str(n)] + [str(counts[o]) for o in OPS])

    for raw in sys.stdin:
        ln = raw.rstrip("\n")
        s = ln.strip()
        if s.startswith("Function : "):
            flush()
            name, lines = s[len("Function : "):], []
        elif s.startswith("code for sm_") or s.startswith("Fatbin ") or s.startswith("....."):
            flush()
            name, lines = None, []
        elif name is not None and s:
            # cuobjdump pads the encoding column to a width that varies between builds: compare text, not padding
            lines.append(re.sub(r"\s+", " ", s))
    flush()
    print("\t".join(["kernel", "kq_fp32", "sha1", "instructions"] + list(OPS)))
    for r in rows:
        print("\t".join(r))


def load(path):
    rows = []
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    return rows


def compare_mode(base_p, new_p, out):
    base, new = load(base_p), load(new_p)
    new_off = collections.defaultdict(list)
    new_on = {}
    for r in new:
        if r["kq_fp32"] == "true":
            new_on[r["kernel"]] = r
        else:
            new_off[r["kernel"]].append(r)
    base_by = collections.defaultdict(list)
    for r in base:
        base_by[r["kernel"]].append(r)

    same = differ = missing = 0
    with open(out + ".identity.tsv", "w") as f:
        f.write("kernel\tcopies_base\tcopies_new\tidentical\n")
        for k, rs in sorted(base_by.items()):
            hb = sorted(r["sha1"] for r in rs)
            hn = sorted(r["sha1"] for r in new_off.get(k, []))
            if not hn:
                missing += 1
                verdict = "missing in new"
            elif hb == hn:
                same += 1
                verdict = "yes"
            else:
                differ += 1
                verdict = "NO"
            if verdict != "yes" or k.startswith("void mul_mat_q<"):
                f.write(f"{k}\t{len(hb)}\t{len(hn)}\t{verdict}\n")
    extra_off = sorted(set(new_off) - set(base_by))

    with open(out + ".kq.tsv", "w") as f:
        cols = ["instructions"] + list(OPS)
        f.write("type\tJ\tfallback\thas_fusion\t" + "\t".join(f"{c}_off\t{c}_on" for c in cols) + "\n")
        for k, on in sorted(new_on.items()):
            m = re.match(r"^void mul_mat_q<(.*?)>\(", k)
            a = split_args(m.group(1))
            off = new_off.get(k, [None])[0]
            if off is None:
                continue
            vals = "\t".join(f"{off[c]}\t{on[c]}" for c in cols)
            f.write(f"{KQ_TYPES.get(a[0], a[0])}\t{a[1]}\t{a[2]}\t{a[4]}\t{vals}\n")

    print(f"base kernels (names): {len(base_by)}; identical in new with kq_fp32 off: {same}; different: {differ}; "
          f"missing in new: {missing}; names only in new (kq_fp32 off): {len(extra_off)}; kq_fp32 on kernels: {len(new_on)}")
    for k in extra_off[:20]:
        print("only in new:", k)


ENC = re.compile(r"/\*\s*0x[0-9a-f]+\s*\*/")
ADDR = re.compile(r"^/\*[0-9a-f]{4,}\*/\s*")
MOV32I = re.compile(r"^MOV32I (R\d+), (0x[0-9a-f]+) ;$")


def extract_mode(list_path):
    """Write the listed kernels' SASS (kq_fp32 off) without addresses and encodings: '### name' then its lines."""
    want = set(ln.rstrip("\n") for ln in open(list_path) if ln.strip())
    name, keep = None, False
    for raw in sys.stdin:
        s = raw.strip()
        if s.startswith("Function : "):
            base, kq = norm(s[len("Function : "):])
            keep = base in want and kq != "true"
            if keep:
                print("### " + base)
        elif s.startswith("code for sm_") or s.startswith("Fatbin "):
            keep = False
        elif keep:
            s = ENC.sub("", ADDR.sub("", s)).strip()
            if s:
                print(s)


def read_extract(path):
    out = collections.defaultdict(list)
    cur = None
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("### "):
            cur = []
            out[ln[4:]].append(cur)
        else:
            cur.append(ln)
    return out


def linediff_mode(a_path, b_path):
    """Every line that differs between two extracts, classified: a MOV32I constant (old -> new) or anything else."""
    import difflib
    a, b = read_extract(a_path), read_extract(b_path)
    pairs = collections.Counter()
    only_const, other = 0, []
    for k in sorted(a):
        if k not in b or len(a[k]) != len(b[k]):
            other.append((k, "copies differ"))
            continue
        bad = False
        for la, lb in zip(sorted(a[k]), sorted(b[k])):
            if len(la) != len(lb):
                bad = True
                break
            for x, y in zip(la, lb):
                if x == y:
                    continue
                mx, my = MOV32I.match(x), MOV32I.match(y)
                if mx and my and mx.group(1) == my.group(1):
                    pairs[(int(mx.group(2), 16), int(my.group(2), 16))] += 1
                else:
                    bad = True
        if bad:
            other.append((k, "other lines differ"))
        else:
            only_const += 1
    print(f"kernels compared: {len(a)}; differing only in MOV32I constants: {only_const}; differing otherwise: {len(other)}")
    for (x, y), n in sorted(pairs.items()):
        print(f"MOV32I {x} -> {y} (+{y - x}): {n} lines")
    for k, why in other[:20]:
        print("OTHER:", why, k)


if __name__ == "__main__":
    if sys.argv[1] == "hash":
        hash_mode()
    elif sys.argv[1] == "extract":
        extract_mode(sys.argv[2])
    elif sys.argv[1] == "linediff":
        linediff_mode(sys.argv[2], sys.argv[3])
    else:
        compare_mode(sys.argv[2], sys.argv[3], sys.argv[4])
