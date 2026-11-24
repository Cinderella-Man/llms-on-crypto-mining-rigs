#!/usr/bin/env python3
# C993 (Q1726): tables for the fault switch's factor (GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE).
#   c993_tables.py hashcmp h-B.tsv h-H.tsv differ.list
#       h-*.tsv: sass_compare.py hash tables of the base and the head. Compares each kernel name with each value of its
#       kq_fp32 flag (mul_mat_q's copies with and without the FP32 scale are two kernels) by the multiset of its copies'
#       SHA-1s; prints the counts and writes the names whose copies differ (with kq_fp32 off, as sass_compare.py
#       extract reads them) to differ.list.
#   c993_tables.py sassdiff ext-B.txt ext-H.txt lines.tsv
#       ext-*.txt: sass_compare.py extracts ('### kernel' then its SASS lines, no addresses or encodings) of the
#       kernels whose SASS differs between the base and the head. Writes every differing line (lines.tsv) and prints a
#       summary: per kernel its copies and instruction counts, and each kind of change with registers and predicates
#       written as R and P.
#   c993_tables.py tables W OUT
#       W: run.sh's work folder (check/arms.txt, check/<arm>.tsv and .log, checks/sass/). Writes into OUT:
#       arms.tsv      per arm: binary, switches, cases, OK, FAIL, the cases the fork's kernels took (their error changes
#                     under the fault), and for those the verdicts and the error against ((factor - 1)/factor)^2: the
#                     test's error is NMSE = sum((a - b)^2) / sum(a^2) with a the card's output, so a card output scaled
#                     by f against an exact reference gives ((f - 1)/f)^2, and the test's band of 5e-4 is crossed at
#                     f = 1/(1 - sqrt(5e-4)) = 1.0229
#       identity.tsv  the arm pairs that must give the same error on every case, and whether they do
#       cases.tsv     every case: the kernel that takes it with the fault on, and its error in every arm
#       logs.tsv      per arm: the kernel and fault lines its log printed (once each)
#       live.tsv      the arms with a live CPU check: cases, live verdicts, references identical, verdicts that differ
#       sass.tsv      every SASS line that differs between the base and the head (from checks/sass/lines.tsv)
#       sass_kernels.tsv  per kernel whose SASS differs: its instructions in the base and the head, the lines that differ
import collections
import difflib
import math
import os
import re
import statistics
import sys

REG = re.compile(r"\bU?R\d+\b")
PRED = re.compile(r"\bU?P\d\b")


def read_extract(path):
    out = collections.defaultdict(list)
    cur = None
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("### "):
            cur = []
            out[ln[4:]].append(cur)
        elif cur is not None:
            cur.append(ln)
    return out


def norm(line):
    return PRED.sub("P", REG.sub("R", line))


def short(name):
    """The kernel's name with its template arguments, without the parameter list."""
    name = name.replace("void ", "", 1)
    return name[:name.find(">(") + 1] if ">(" in name else name.split("(")[0]


def sassdiff(a_path, b_path, out_path):
    a, b = read_extract(a_path), read_extract(b_path)
    kinds = collections.Counter()
    per_kernel = []
    with open(out_path, "w") as f:
        f.write("kernel\tcopy\tinstructions_base\tinstructions_head\tchange\tbase\thead\n")
        for k in sorted(set(a) | set(b)):
            ca, cb = a.get(k, []), b.get(k, [])
            n_lines = 0
            for i in range(max(len(ca), len(cb))):
                la = ca[i] if i < len(ca) else []
                lb = cb[i] if i < len(cb) else []
                sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op == "equal":
                        continue
                    xa, xb = la[i1:i2], lb[j1:j2]
                    for m in range(max(len(xa), len(xb))):
                        x = xa[m] if m < len(xa) else ""
                        y = xb[m] if m < len(xb) else ""
                        f.write(f"{k}\t{i}\t{len(la)}\t{len(lb)}\t{op}\t{x}\t{y}\n")
                        kinds[(norm(x), norm(y))] += 1
                        n_lines += 1
            per_kernel.append((k, len(ca), len(cb), [len(c) for c in ca], [len(c) for c in cb], n_lines))
    print(f"kernels that differ: {len(per_kernel)}; differing lines: {sum(p[5] for p in per_kernel)}")
    for k, na, nb, ia, ib, n in per_kernel:
        print(f"{short(k)}: copies {na}/{nb}, instructions base {ia} head {ib}, lines that differ {n}  [{k[:160]}]")
    print("kinds of change (registers as R, predicates as P), base -> head:")
    for (x, y), n in kinds.most_common():
        print(f"  {n:4d}  {x or '(none)'}  ->  {y or '(none)'}")


def hashcmp(b_path, h_path, out_list):
    def load(p):
        d = collections.defaultdict(list)
        with open(p) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for ln in f:
                r = dict(zip(hdr, ln.rstrip("\n").split("\t")))
                d[(r["kernel"], r["kq_fp32"])].append(r["sha1"])
        return d
    b, h = load(b_path), load(h_path)
    same = [k for k in b if k in h and sorted(b[k]) == sorted(h[k])]
    differ = sorted(k for k in b if k in h and sorted(b[k]) != sorted(h[k]))
    missing = [k for k in b if k not in h]
    extra = [k for k in h if k not in b]
    print(f"kernels (name and kq_fp32 flag): base {len(b)} ({sum(map(len, b.values()))} copies), head {len(h)} "
          f"({sum(map(len, h.values()))} copies); the same bytes {len(same)}; different {len(differ)} "
          f"(with kq_fp32 on: {sum(k[1] == 'true' for k in differ)}); missing in head {len(missing)}; only in head {len(extra)}")
    with open(out_list, "w") as f:
        for k, kq in differ:
            if kq != "true":
                f.write(k + "\n")


def parse_vars(case_name):
    inner = case_name[case_name.find("(") + 1:case_name.rfind(")")]
    return dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^,]*)", inner))


def read_log(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if len(ln) > 400:
                continue
            out.setdefault(ln, 1)
    return out


def errmax(s):
    vals = [float(x) for x in s.split(",") if x.strip()]
    return max(vals) if vals else float("nan")


def factor(envs):
    """The factor the arm's switches give fault 1 (None when fault 1 is off)."""
    d = dict(e.split("=", 1) for e in envs.split() if "=" in e)
    if d.get("GGML_CUDA_FATTN_TILE_PASCAL_FAULT") != "1":
        return None
    s = d.get("GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE")
    if s is None:
        return 1.05
    try:
        v = float(s)
        return v if math.isfinite(v) else 1.05
    except ValueError:
        return 1.05


def g(x, p=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}g}"


def tables(w, out):
    c = os.path.join(w, "check")
    os.makedirs(out, exist_ok=True)
    arms = []
    for ln in open(os.path.join(c, "arms.txt")):
        ln = ln.rstrip("\n")
        if ln.strip():
            tag, b, envs = (ln.split("|", 2) + ["", ""])[:3]
            arms.append((tag, b, envs.strip()))
    res = {}   # arm -> case id -> (verdict, err)
    order = []
    names = {}  # case id -> case name (two pairs of the cases share a name)
    for tag, _, _ in arms:
        res[tag] = {}
        p = os.path.join(c, tag + ".tsv")
        if not os.path.exists(p):
            continue
        for ln in open(p):
            f = ln.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            cid = f[0]
            names.setdefault(cid, f[1])
            if cid not in res[tag]:
                res[tag][cid] = (f[2], errmax(f[3]) if len(f) > 3 and f[2] != "NOREC" else float("nan"))
                if cid not in order:
                    order.append(cid)

    # the cases each family of arms has the fork's kernels take: their error changes with the fault on
    def taken(off, on):
        return {n for n in order if n in res.get(off, {}) and n in res.get(on, {}) and res[off][n][1] != res[on][n][1]}

    taken_prompt = taken("head-unset", "head-f1")
    taken_dec = taken("base-dec-unset", "base-dec-f1")

    def kernel_of(n, dec):
        v = parse_vars(names[n])
        if dec and v.get("type_K") == "q8_0" and v.get("sinks") == "0" and int(v.get("nb", "99")) <= 4:
            return "decode"
        return "prompt"

    with open(os.path.join(out, "arms.tsv"), "w") as f:
        f.write("arm\tbinary\tswitches (with GGML_CUDA_FATTN_TILE_PASCAL=2)\tcases\tOK\tFAIL\tfork kernel cases\t"
                "their OK\ttheir FAIL\tfactor\t((factor-1)/factor)^2\terror median\terror min\terror max\t"
                "sqrt(error) median\terror/((factor-1)/factor)^2 median\n")
        for tag, b, envs in arms:
            r = res[tag]
            dec = "dec" in tag
            tk = [n for n in order if n in r and n in (taken_dec if dec else taken_prompt)]
            fac = factor(envs)
            errs = [r[n][1] for n in tk if not math.isnan(r[n][1])]
            e2 = ((fac - 1) / fac) ** 2 if fac is not None else None
            med = statistics.median(errs) if errs else None
            row = [tag, b, envs or "(none)", len(r), sum(v[0] == "OK" for v in r.values()),
                   sum(v[0] == "FAIL" for v in r.values()), len(tk), sum(r[n][0] == "OK" for n in tk),
                   sum(r[n][0] == "FAIL" for n in tk), "" if fac is None else f"{fac:g}", g(e2),
                   g(med), g(min(errs) if errs else None), g(max(errs) if errs else None),
                   g(math.sqrt(med) if med is not None else None),
                   g(statistics.median(e / e2 for e in errs) if errs and e2 else None)]
            f.write("\t".join(str(x) for x in row) + "\n")

    pairs = [("base-unset", "head-unset", "the switch unset: the head as the base"),
             ("base-f1", "head-f1", "_FAULT=1 alone: the head's x 1.05 as the base's"),
             ("head-f1", "head-s1.05", "_FAULT_SCALE=1.05: the default"),
             ("head-f1", "head-sbad", "_FAULT_SCALE=x1.03 (not a number): the default"),
             ("head-unset", "head-s1.00", "_FAULT_SCALE=1.0: no error"),
             ("head-unset", "head-snof", "_FAULT_SCALE=1.03 without _FAULT: nothing"),
             ("base-dec-unset", "head-dec-unset", "decode kernel on, the switch unset: the head as the base")]
    with open(os.path.join(out, "identity.tsv"), "w") as f:
        f.write("arm\tagainst\twhat it shows\tcases in both\tthe same error\tdifferent error\tthe same verdict\n")
        for x, y, what in pairs:
            if x not in res or y not in res:
                continue
            both = [n for n in order if n in res[x] and n in res[y]]
            same = sum(res[x][n][1] == res[y][n][1] or (math.isnan(res[x][n][1]) and math.isnan(res[y][n][1])) for n in both)
            sv = sum(res[x][n][0] == res[y][n][0] for n in both)
            f.write(f"{x}\t{y}\t{what}\t{len(both)}\t{same}\t{len(both) - same}\t{sv}\n")

    with open(os.path.join(out, "cases.tsv"), "w") as f:
        f.write("case id\tcase\tkernel with the fault on (prompt arms)\tkernel with the fault on (decode arms)\t" +
                "\t".join(t for t, _, _ in arms) + "\n")
        for n in order:
            kp = "prompt" if n in taken_prompt else "card's own"
            kd = kernel_of(n, True) if n in taken_dec else "card's own"
            cells = []
            for t, _, _ in arms:
                v = res[t].get(n)
                cells.append("" if v is None else f"{v[0]} {g(v[1], 6)}")
            f.write(f"{n}\t{names[n]}\t{kp}\t{kd}\t" + "\t".join(cells) + "\n")

    keep = re.compile(r"(compute-6\.1 (prompt|decode) attention|GGML_CUDA_FATTN_TILE_PASCAL_FAULT|is not a number)")
    with open(os.path.join(out, "logs.tsv"), "w") as f:
        f.write("arm\tkernel and fault lines (once each)\n")
        for t, _, _ in arms:
            lines = [re.sub(r"^.*?(ggml_cuda_flash_attn_ext_tile_pascal|ftp_fault_scale)", r"\1", ln)
                     for ln in read_log(os.path.join(c, t + ".log")) if keep.search(ln)]
            f.write(f"{t}\t{' | '.join(dict.fromkeys(lines)) or '(none)'}\n")

    with open(os.path.join(out, "live.tsv"), "w") as f:
        f.write("arm\tcases checked live\tlive OK\tlive FAIL\treferences identical\treferences differ\t"
                "live and replayed verdicts differ\n")
        for t, _, _ in arms:
            p = os.path.join(c, t + ".log")
            if not os.path.exists(p):
                continue
            ls = [ln for ln in open(p, errors="replace") if "cpu-check: live" in ln]
            if not ls:
                continue
            f.write(f"{t}\t{len(ls)}\t{sum('live OK' in x for x in ls)}\t{sum('live FAIL' in x for x in ls)}\t"
                    f"{sum('reference identical' in x for x in ls)}\t{sum('reference differs' in x for x in ls)}\t"
                    f"{sum('verdicts differ' in x for x in ls)}\n")

    src = os.path.join(w, "checks", "sass", "lines.tsv")
    if os.path.exists(src):
        per = collections.OrderedDict()
        with open(src) as fi, open(os.path.join(out, "sass.tsv"), "w") as fo:
            hdr = fi.readline()
            fo.write(hdr)
            for ln in fi:
                f = ln.rstrip("\n").split("\t")
                f[0] = short(f[0])
                fo.write("\t".join(f) + "\n")
                k = per.setdefault((f[0], f[1]), [f[2], f[3], 0])
                k[2] += 1
        with open(os.path.join(out, "sass_kernels.tsv"), "w") as fo:
            fo.write("kernel\tcopy\tinstructions base\tinstructions head\tlines that differ\n")
            for (k, cp), (ib, ih, n) in per.items():
                fo.write(f"{k}\t{cp}\t{ib}\t{ih}\t{n}\n")
    print(f"tables in {out}: {len(order)} cases, prompt-kernel cases {len(taken_prompt)}, "
          f"cases changed with the decode kernel on {len(taken_dec)}")


if __name__ == "__main__":
    if sys.argv[1] == "hashcmp":
        hashcmp(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "sassdiff":
        sassdiff(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "tables":
        tables(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: c993_tables.py hashcmp h-B h-H differ.list | sassdiff ext-B ext-H lines.tsv | tables W OUT")
