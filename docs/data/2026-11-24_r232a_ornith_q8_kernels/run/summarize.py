#!/usr/bin/env python3
"""Tables from r232a.sh's output folder (R232 (a), Q1706).

Usage: python3 -I summarize.py OUT_DIR DEST_DIR

Writes to DEST_DIR:
  tests.tsv     one row per correctness run: arm, case set, exit code, passed/run, failures, not supported, kernels logged
  perf.tsv      one row per Ornith case and arm: each repeat's us/run by value, their spread, and the run order
  verdict.tsv   each switch arm against base per case: mean difference, the larger repeat spread, the class
  clocks.tsv    card clocks before and after each run (nvidia-smi)

Class rule (fixed before the run): gain if every repeat of the arm is faster than every repeat of base and the means
differ by more than the larger of the two repeat spreads; loss if the same holds the other way; otherwise negligible.
Added after the run: pairs whose kernel cannot differ (KERNEL below, from the routing in fattn-tile-pascal.cu and
fattn.cu) differed by up to 1.71 % at 1 to 8 query tokens, more than their repeat spread, so the final class also asks
the change to exceed that same-kernel floor (floor.tsv lists every same-kernel pair).
"""
import os
import re
import sys
from collections import defaultdict

ARMS = ["base", "min1", "dec4", "both"]
ARM_ENV = {
    "base": "nothing set",
    "min1": "GGML_CUDA_FATTN_TILE_PASCAL_MIN=1",
    "dec4": "GGML_CUDA_FATTN_DEC_PASCAL=4",
    "both": "GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4",
}
def kernel(t, nt, arm):
    """The kernel an arm's process runs for a case, by the routing of pascal-p106-v5 af00bb13b on compute 6.1."""
    mn = arm in ("min1", "both")
    dec = arm in ("dec4", "both")
    if t == "q8_0":
        if nt <= 4 and dec and (nt >= 3 or mn):
            return "Pascal decode kernel"
        if nt >= 3 or mn:
            return "Pascal tile kernel"
        return "stock vector kernel"
    if nt >= 9 or mn:
        return "Pascal tile kernel"
    return "stock kernel"


CASE_RE = re.compile(r"name=ornith_(q8_0|f16)_kv(\d+)_nt(\d+)")
PERF_RE = re.compile(r"^\s*FLASH_ATTN_EXT\((.*)\):\s+(\d+) runs -\s+([\d.]+) us/run")
KERNEL_RE = re.compile(r"compute-6\.1 (\w+) attention kernel: ([^(]*)")


def kernel_paths(text):
    seen = []
    for m in KERNEL_RE.finditer(text):
        s = f"{m.group(1)}: {m.group(2).strip().rstrip(',')}"
        if s not in seen:
            seen.append(s)
    return "; ".join(seen) if seen else "none"


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def tests(out, dest):
    rows = []
    for arm in ARMS:
        for cs in ("suite256", "ornith"):
            name = f"test-{cs}-{arm}"
            log = os.path.join(out, name + ".log")
            if not os.path.exists(log):
                continue
            text = read(log)
            rc = read(os.path.join(out, name + ".rc")).strip()
            m = re.search(r"(\d+)/(\d+) tests passed", text)
            passed, run = (m.group(1), m.group(2)) if m else ("?", "?")
            fails = len(re.findall(r"^\s*FLASH_ATTN_EXT\(.*\): .*FAIL", text, re.M))
            nsup = len(re.findall(r"^\s*FLASH_ATTN_EXT\(.*\): not supported", text, re.M))
            rows.append([arm, ARM_ENV[arm], "suite FLASH_ATTN_EXT, head 256" if cs == "suite256" else "Ornith cases (cases.txt)",
                         rc, passed, run, str(fails), str(nsup), kernel_paths(text)])
    with open(os.path.join(dest, "tests.tsv"), "w") as f:
        f.write("arm\tswitches set\tcase set\texit code\tpassed\trun\tfailed\tnot supported on CUDA (skipped)\tPascal kernel paths that logged\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def perf(out, dest):
    vals = defaultdict(dict)        # (type, kv, nt) -> {(arm, r): us}
    order = []
    for line in read(os.path.join(out, "progress.log")).splitlines():
        m = re.search(r"start (perf-ornith-(\w+)-r(\d+))", line)
        if m:
            order.append(m.group(1))
    paths = {}
    for run in order:
        _, _, arm, r = run.split("-")
        r = int(r[1:])
        text = read(os.path.join(out, run + ".log"))
        paths[(arm, r)] = kernel_paths(text)
        for line in text.splitlines():
            m = PERF_RE.match(line)
            if not m:
                continue
            c = CASE_RE.search(m.group(1))
            key = (c.group(1), int(c.group(2)), int(c.group(3)))
            vals[key][(arm, r)] = float(m.group(3))
    reps = sorted({r for v in vals.values() for (_, r) in v})
    keys = sorted(vals, key=lambda k: (k[0] != "q8_0", k[2], k[1]))
    with open(os.path.join(dest, "perf.tsv"), "w") as f:
        f.write("K/V type\tn_kv\tquery tokens\tarm\t" + "\t".join(f"us/run, repeat {r}" for r in reps) +
                "\tmean\tspread (max - min)\n")
        for k in keys:
            for arm in ARMS:
                xs = [vals[k].get((arm, r)) for r in reps]
                if any(x is None for x in xs):
                    continue
                f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{arm}\t" + "\t".join(f"{x:.2f}" for x in xs) +
                        f"\t{sum(xs) / len(xs):.2f}\t{max(xs) - min(xs):.2f}\n")
    def mean(k, arm):
        xs = [vals[k].get((arm, r)) for r in reps]
        return None if any(x is None for x in xs) else xs

    # the same-kernel floor: every pair of arms whose process runs the same kernel for a case
    same = []
    for k in keys:
        for i, x in enumerate(ARMS):
            for y in ARMS[i + 1:]:
                a, b = mean(k, y), mean(k, x)
                if a and b and kernel(k[0], k[2], x) == kernel(k[0], k[2], y):
                    same.append((k, y, x, sum(a) / len(a) / (sum(b) / len(b)) - 1))
    with open(os.path.join(dest, "floor.tsv"), "w") as f:
        f.write("K/V type\tn_kv\tquery tokens\tarm\tagainst\tkernel (both)\tarm / against - 1\n")
        for k, y, x, d in sorted(same, key=lambda s: -abs(s[3])):
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{y}\t{x}\t{kernel(k[0], k[2], x)}\t{d * 100:+.2f} %\n")
    floor = max(abs(d) for k, _, _, d in same if k[2] <= 8)
    with open(os.path.join(dest, "verdict.tsv"), "w") as f:
        f.write("K/V type\tn_kv\tquery tokens\tarm\tkernel, base\tkernel, arm\tbase mean us\tarm mean us\t"
                "arm - base us\tarm / base\tlarger repeat spread us\tclass by the repeat rule\t"
                f"class (also beyond the {floor * 100:.2f} % same-kernel floor)\n")
        for k in keys:
            b = mean(k, "base")
            if not b:
                continue
            for arm in ARMS[1:]:
                a = mean(k, arm)
                if not a:
                    continue
                mb, ma = sum(b) / len(b), sum(a) / len(a)
                spread = max(max(b) - min(b), max(a) - min(a))
                if max(a) < min(b) and mb - ma > spread:
                    cls = "gain"
                elif min(a) > max(b) and ma - mb > spread:
                    cls = "loss"
                else:
                    cls = "negligible"
                kb, ka = kernel(k[0], k[2], "base"), kernel(k[0], k[2], arm)
                fin = cls if abs(ma / mb - 1) > floor else "negligible"
                if kb == ka:
                    fin = "same kernel"
                f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{arm}\t{kb}\t{ka}\t{mb:.2f}\t{ma:.2f}\t{ma - mb:+.2f}\t"
                        f"{ma / mb:.4f}\t{spread:.2f}\t{cls}\t{fin}\n")
    with open(os.path.join(dest, "perf_runs.tsv"), "w") as f:
        f.write("order\trun\tPascal kernel paths that logged\n")
        for i, run in enumerate(order, 1):
            _, _, arm, r = run.split("-")
            f.write(f"{i}\t{run}\t{paths[(arm, int(r[1:]))]}\n")


def clocks(out, dest):
    with open(os.path.join(dest, "clocks.tsv"), "w") as f:
        f.write("run\twhen\ttime\tcard\tpstate\tSM MHz\tmemory MHz\tmax SM MHz\tmax memory MHz\tmemory used\n")
        for name in sorted(os.listdir(out)):
            if not name.endswith(".smi"):
                continue
            for line in read(os.path.join(out, name)).splitlines():
                when, t, rest = line.split(" ", 2)
                f.write(f"{name[:-4]}\t{when}\t{t}\t" + "\t".join(x.strip() for x in rest.split(",")) + "\n")


def main():
    out, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    tests(out, dest)
    if os.path.exists(os.path.join(out, "progress.log")):
        perf(out, dest)
    clocks(out, dest)


if __name__ == "__main__":
    main()
