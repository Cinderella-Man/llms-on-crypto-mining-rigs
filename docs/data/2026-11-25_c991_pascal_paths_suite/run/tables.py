#!/usr/bin/env python3
"""C991 (Q1724): the tables from a copy of run.sh's work folder (standard library only; run with python3 -I).

Usage: python3 -I tables.py <work copy> <out folder> [<C947's bops-coverage.tsv>]

Writes into <out folder>:
  bops.tsv       every test-backend-ops run: run, operators, extra environment, cases, passed, failed, not supported,
                 exit code, failing cases
  coverage.tsv   per operator: share, cases in the dry listing, cases the live pass ran, passed, failed, C947's cases
  kernels.tsv    per share: each kernel the live pass ran (Nsight Systems' kernel summary) and its launch count; fa-ornith
                 the fa share under Ornith's attention switches (STEPS=ornith)
  reach.tsv      per fork path: its switch, its kernels, their launches per share, and the cases a probe shows on it
  fa-reach.tsv   FLASH_ATTN_EXT by head sizes, K and V types and query tokens: cases, how many the Pascal kernels computed
                 (failed with GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1, which scales their output by 1.05), with the forced set
                 and with Ornith's attention switches added
  moe.tsv        each MOE_LAYER case: live verdict, verdict with GGML_CUDA_PERSISTENT_MOE_FAULT=3; and the check line
  cpy.tsv        cpy_reps.sh's repeats of the hop cast's copies: setting, repeat, cases, failed, each failing copy's
                 target type with its error and bound
"""
import collections
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# a case line; a failing one carries its error first ("[CPY] ERR = 0.000000004 > 0.000000003   CPY(...): FAIL")
CASE = re.compile(r"^(.*?)([A-Z][A-Z0-9_]*)\((.*)\): (.*?)(OK|FAIL)$")
UNSUP = re.compile(r"^  ([A-Za-z0-9_]+)\((.*)\): not supported")
DRY = re.compile(r"^  ([A-Za-z0-9_]+)\((.*)\): in [0-9]+ B, ref [0-9]+ B$")

# the fork's compute-6.1-only paths: (path, switch, kernel-name test)
PATHS = [
    ("Pascal flash-attention tile kernel (prompt)", "GGML_CUDA_FATTN_TILE_PASCAL",
     lambda k: k.startswith("flash_attn_tile_pascal")),
    ("Pascal decode attention kernel (pull request 9)", "GGML_CUDA_FATTN_DEC_PASCAL (with FATTN_TILE_PASCAL)",
     lambda k: k.startswith("flash_attn_dec_pascal")),
    ("persistent MoE kernel", "GGML_CUDA_PERSISTENT_MOE",
     lambda k: k.startswith("k_moe_layer")),
    ("persistent MoE self-check comparison", "GGML_CUDA_PERSISTENT_MOE_CHECK",
     lambda k: k.startswith("k_moep_compare")),
    ("attention block: Q/K/V products", "GGML_CUDA_ATTN_BLOCK",
     lambda k: k.startswith("mul_mat_vec_q_pascal_qkv")),
    ("attention block: epilogue (norms, RoPE, cache writes)", "GGML_CUDA_ATTN_BLOCK",
     lambda k: k.startswith("attn_epilogue_f32")),
    ("older Pascal decode attention kernel", "GGML_CUDA_FATTN_PASCAL",
     lambda k: k.startswith("flash_attn_pascal") or k.startswith("fap_")),
    ("matrix-vector kernels: one token and multi-token (C968)", "GGML_CUDA_MMVQ_PASCAL",
     lambda k: k.startswith("mul_mat_vec_q_pascal") and not k.startswith("mul_mat_vec_q_pascal_qkv")),
    ("matrix-vector kernels: k-column (C968)", "GGML_CUDA_MMVQ_PASCAL",
     lambda k: "kcol" in k),
    ("matrix-vector kernels: F16/BF16 (C968)", "GGML_CUDA_MMVF_PASCAL",
     lambda k: k.startswith("mul_mat_vec_f16_pascal") or k.startswith("mul_mat_vec_bf16_pascal")
     or ("pascal" in k and k.startswith("mul_mat_vec_f"))),
]
SHARES = ["fa", "rest", "fused"]
TRACES = SHARES + ["fa-ornith"]  # kern-<trace>.tsv


def lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                yield ANSI.sub("", ln.rstrip("\n"))
    except FileNotFoundError:
        return


def cases(path):
    """(op, params, verdict, detail) for each case line, and the not-supported count"""
    out, unsup = [], 0
    for ln in lines(path):
        m = CASE.match(ln)
        if m and (m.group(1) == "  " or m.group(1).startswith("[")):
            out.append((m.group(2), m.group(3), m.group(5), (m.group(1) + m.group(4)).strip()))
        elif UNSUP.match(ln):
            unsup += 1
    return out, unsup


def read_kv(path):
    d = {}
    for ln in lines(path):
        if ": " in ln:
            k, v = ln.split(": ", 1)
            d[k] = v
    return d


def main():
    work, out = sys.argv[1], sys.argv[2]
    c947 = sys.argv[3] if len(sys.argv) > 3 else None
    b, lst = os.path.join(work, "bops"), os.path.join(work, "list")
    os.makedirs(out, exist_ok=True)
    forced = read_kv(os.path.join(work, "settings.txt")).get("PASCAL_FORCED", "").split()

    rc = {}
    for ln in lines(os.path.join(b, "rc.txt")):
        if " rc=" in ln:
            t, r = ln.split(" rc=")
            rc[t] = r

    runs = sorted(f[:-4] for f in os.listdir(b) if f.endswith(".log") and not f.startswith("stats-"))
    res = {}
    with open(os.path.join(out, "bops.tsv"), "w", encoding="utf-8") as f:
        f.write("run\toperators\textra_env\tcases\tpassed\tfailed\tnot_supported\texit_code\tfailing_cases\n")
        for r in runs:
            cs, unsup = cases(os.path.join(b, r + ".log"))
            res[r] = cs
            cmd = read_kv(os.path.join(b, r + ".cmd"))
            env = [e for e in cmd.get("env", "").split() if e not in forced]
            m = re.search(r"-o '([^']*)'", cmd.get("args", ""))
            nops = len(m.group(1).split(",")) if m else 0
            ok = sum(1 for c in cs if c[2] == "OK")
            fail = [c for c in cs if c[2] == "FAIL"]
            f.write("\t".join([r, str(nops), " ".join(env) or "-", str(len(cs)), str(ok), str(len(fail)), str(unsup),
                               rc.get(r, "?"), "; ".join(f"{c[0]}({c[1]})" for c in fail) or "-"]) + "\n")

    # coverage: dry listing against the live passes
    support = set(lines(os.path.join(lst, "ops-support.txt")))
    fused = set(lines(os.path.join(lst, "ops-fused.txt")))
    dry = collections.Counter()
    for ln in lines(os.path.join(lst, "dry-cases.txt")):
        m = DRY.match(ln)
        if m:
            dry[m.group(1)] += 1
    live = collections.defaultdict(lambda: [0, 0])
    for s in SHARES:
        for c in res.get(s + "-live", []):
            live[c[0]][0 if c[2] == "OK" else 1] += 1
    ref = {}
    if c947:
        for ln in list(lines(c947))[1:]:
            p = ln.split("\t")
            if len(p) >= 3 and p[0] != "total":
                ref[p[0]] = p[2]

    def share(op):
        if op in ("MUL_MAT", "MUL_MAT_ID"):
            return "not run (C968)"
        if op == "FLASH_ATTN_EXT":
            return "fa"
        if op in support:
            return "rest"
        if op in fused:
            return "fused"
        return "none"

    with open(os.path.join(out, "coverage.tsv"), "w", encoding="utf-8") as f:
        f.write("operator\tshare\tdry_cases\tlive_cases\tpassed\tfailed\tc947_cases\n")
        tot = [0, 0, 0, 0]
        for op in sorted(set(dry) | set(live)):
            ok, fl = live[op]
            f.write("\t".join([op, share(op), str(dry[op]), str(ok + fl), str(ok), str(fl), ref.get(op, "-")]) + "\n")
            if share(op) in SHARES:
                tot = [tot[0] + dry[op], tot[1] + ok + fl, tot[2] + ok, tot[3] + fl]
        f.write("\t".join(["total of the three shares", "fa, rest, fused"] + [str(x) for x in tot] + ["-"]) + "\n")

    # kernels per share
    kern = {}
    with open(os.path.join(out, "kernels.tsv"), "w", encoding="utf-8") as f:
        f.write("share\tlaunches\tkernel\n")
        for s in TRACES:
            kern[s] = []
            for ln in lines(os.path.join(b, f"kern-{s}.tsv")):
                p = ln.split("\t")
                if len(p) == 2 and p[0].isdigit():
                    kern[s].append((int(p[0]), p[1]))
                    f.write(f"{s}\t{p[0]}\t{p[1]}\n")

    def keyed(cs, op=None):
        """{(params, n): verdict}: the n-th case with these params (a few cases share their params string)"""
        seen, d = collections.Counter(), {}
        for c in cs:
            if op is None or c[0] == op:
                d[(c[1], seen[c[1]])] = c[2]
                seen[c[1]] += 1
        return d

    # FLASH_ATTN_EXT: the fault probe
    fa_live = keyed(res.get("fa-live", []))
    fa_fault = keyed(res.get("fa-fault", []))
    fa_orn = keyed(res.get("fa-ornith", []))
    fa_ornf = keyed(res.get("fa-ornith-fault", []))

    def par(params, key):
        m = re.search(r"(?:^|,)" + key + r"=(\[[^\]]*\]|[^,]*)", params)
        return m.group(1) if m else "-"

    def num(v):
        return int(v) if v.isdigit() else -1

    groups = {}
    for p in fa_live:
        key = (par(p[0], "hsk"), par(p[0], "hsv"), par(p[0], "type_K") + "/" + par(p[0], "type_V"), par(p[0], "nb"))
        g = groups.setdefault(key, [0, 0, 0, 0, 0])
        g[0] += 1
        g[1] += fa_live[p] == "OK"
        g[2] += fa_fault.get(p) == "FAIL"
        g[3] += fa_orn.get(p) == "OK"
        g[4] += fa_ornf.get(p) == "FAIL"
    groups = dict(sorted(groups.items(), key=lambda kv: (num(kv[0][0]), num(kv[0][1]), kv[0][2], num(kv[0][3]))))
    with open(os.path.join(out, "fa-reach.tsv"), "w", encoding="utf-8") as f:
        f.write("hsk\thsv\ttype_K/type_V\tquery_tokens\tcases\tpassed_live\ton_pascal_kernels\t"
                "passed_with_ornith_switches\ton_pascal_kernels_with_ornith_switches\n")
        for k, g in groups.items():
            f.write("\t".join(list(k) + [str(x) for x in g]) + "\n")
        f.write("\t".join(["all", "", "", ""] + [str(sum(g[i] for g in groups.values())) for i in range(5)]) + "\n")

    # MOE_LAYER: live, check, fault
    moe_live = keyed(res.get("fused-live", []), "MOE_LAYER")
    moe_chk = keyed(res.get("moe-check", []))
    moe_flt = keyed(res.get("moe-fault", []))
    chk_line = [ln.strip() for ln in lines(os.path.join(b, "moe-check.log")) if "persistent MoE kernel ran" in ln]
    with open(os.path.join(out, "moe.tsv"), "w", encoding="utf-8") as f:
        f.write("case\tlive\twith_check\twith_fault3\n")
        for p in sorted(moe_live):
            f.write(f"MOE_LAYER({p[0]})\t{moe_live[p]}\t{moe_chk.get(p, '-')}\t{moe_flt.get(p, '-')}\n")
        for ln in chk_line:
            f.write(f"# check: {ln}\n")

    # reach per path
    probe = {
        "GGML_CUDA_FATTN_TILE_PASCAL": f"{sum(1 for v in fa_fault.values() if v == 'FAIL')} of {len(fa_fault)} "
                                       "FLASH_ATTN_EXT cases fail with its fault",
        "GGML_CUDA_FATTN_DEC_PASCAL (with FATTN_TILE_PASCAL)":
            f"with Ornith's attention switches {sum(1 for v in fa_ornf.values() if v == 'FAIL')} of {len(fa_ornf)} "
            "FLASH_ATTN_EXT cases fail with the fault (both Pascal attention kernels)",
        "GGML_CUDA_PERSISTENT_MOE": f"{sum(1 for v in moe_flt.values() if v == 'FAIL')} of {len(moe_flt)} "
                                    "MOE_LAYER cases fail with its fault",
    }
    with open(os.path.join(out, "reach.tsv"), "w", encoding="utf-8") as f:
        f.write("path\tswitch\tkernels\t" + "\t".join(f"launches_{s}" for s in TRACES) + "\tprobe\n")
        for name, sw, test in PATHS:
            names = sorted({re.sub(r"<.*", "", k) for s in TRACES for _, k in kern[s] if test(k)})
            n = [sum(c for c, k in kern[s] if test(k)) for s in TRACES]
            f.write("\t".join([name, sw, ", ".join(names) or "-"] + [str(x) for x in n] + [probe.get(sw, "-")]) + "\n")

    # the hop cast's copies, repeated (cpy_reps.sh)
    cd = os.path.join(work, "cpy")
    if os.path.isdir(cd):
        reps = []
        for fn in os.listdir(cd):
            m = re.match(r"(forced|unset)-([0-9]+)\.log$", fn)
            if m:
                reps.append((int(m.group(2)), m.group(1), fn))
        with open(os.path.join(out, "cpy.tsv"), "w", encoding="utf-8") as f:
            f.write("setting\trepeat\tcases\tfailed\tfailing (target type: error > bound)\n")
            for n, st, fn in sorted(reps, key=lambda x: (x[1] != "forced", x[0])):
                cs, _ = cases(os.path.join(cd, fn))
                fl = [c for c in cs if c[2] == "FAIL"]
                desc = []
                for c in fl:
                    t = re.search(r"type_dst=([a-z0-9_]+)", c[1])
                    e = re.search(r"ERR = ([0-9.]+) > ([0-9.]+)", c[3])
                    desc.append(f"{t.group(1) if t else '?'}: {e.group(1)} > {e.group(2)}" if e else (t.group(1) if t else "?"))
                f.write(f"{st}\t{n}\t{len(cs)}\t{len(fl)}\t{'; '.join(desc) or '-'}\n")

    for r in runs:
        cs = res[r]
        print(f"{r}: {len(cs)} cases, {sum(1 for c in cs if c[2] == 'OK')} OK, {sum(1 for c in cs if c[2] == 'FAIL')} FAIL")


if __name__ == "__main__":
    main()
