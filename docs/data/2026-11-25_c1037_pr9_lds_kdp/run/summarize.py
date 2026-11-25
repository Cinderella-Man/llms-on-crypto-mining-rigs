#!/usr/bin/env python3
"""Tables from c1037.sh's output folder (C1037, Q1770).

Usage: python3 -I summarize.py OUT_DIR DEST_DIR

Writes to DEST_DIR:
  tests.tsv      one row per correctness run: arm, case set, exit code, passed/run, failures, kernels logged
  perf.tsv       one row per Ornith case and arm: each repeat's us/run by value, mean, spread
  verdict.tsv    each switch arm against the Q8_0 preset (q8) and against the F16 cache (base, F16 cases), per case
  floor.tsv      the same-kernel pair (kdp and ldskdp run one kernel at head 256) per case
  perf_runs.tsv  the order of the timing runs and the Pascal paths each logged
  clocks.tsv     card clocks before and after each run (nvidia-smi)

Class rule, fixed on 2026-11-25 before any timing run ended: an arm is a gain against its reference if every repeat
of the arm is faster than every repeat of the reference, the means differ by more than the larger of the two repeat
spreads, and the change exceeds the same-kernel floor for that token count; a loss if the same holds the other way;
otherwise negligible. The floor is the larger of R232a's same-kernel differences (1.71 % at 1 to 8 tokens, 2.69 % at
512 tokens; docs/data/2026-11-24_r232a_ornith_q8_kernels/floor.tsv) and this run's own largest |ldskdp / kdp - 1| at
that token count (the two arms run the same kernel at head 256).
"""
import os
import re
import sys
from collections import defaultdict

ARMS = ["base", "q8", "lds", "kdp", "ldskdp"]
TEST_ARMS = ["q8", "lds", "kdp", "ldskdp"]
SWITCH_ARMS = ["lds", "kdp", "ldskdp"]
Q8 = "GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
ARM_ENV = {
    "base": "nothing set",
    "q8": Q8,
    "lds": Q8 + " GGML_CUDA_FATTN_TILE_PASCAL_LDS=1",
    "kdp": Q8 + " GGML_CUDA_FATTN_TILE_PASCAL_KDP=1",
    "ldskdp": Q8 + " GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1",
}
R232A_FLOOR = {8: 0.0171, 512: 0.0269}


def kernel(t, nt, arm):
    """The kernel an arm's process runs for a case at head 256, by the routing of pascal-p106-v5 eee615cdb."""
    if t == "f16":
        if arm == "base" and nt < 9:
            return "stock kernel"
        return "Pascal tile kernel, F16"
    if nt <= 4 and arm != "base":
        return "Pascal decode kernel"
    if arm == "base" and nt < 3:
        return "stock vector kernel"
    if arm in ("kdp", "ldskdp"):
        return "Pascal tile kernel, Q8_0 K scored in int8 (dp4a), loads decoded at store"
    if arm == "lds":
        return "Pascal tile kernel, Q8_0 loads decoded at store"
    return "Pascal tile kernel, Q8_0 loads decoded at load"


CASE_RE = re.compile(r"name=ornith_(q8_0|f16)_kv(\d+)_nt(\d+)")
PERF_RE = re.compile(r"^\s*FLASH_ATTN_EXT\((.*)\):\s+(\d+) runs -\s+([\d.]+) us/run")
KERNEL_RE = re.compile(r"compute-6\.1 ([\w ]+?) attention(?: kernel)?: ([^(]*)")


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
    for arm in TEST_ARMS:
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


def classify(a, b, floor):
    """a: the arm's repeats, b: the reference's repeats (us)."""
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    spread = max(max(a) - min(a), max(b) - min(b))
    change = ma / mb - 1
    if max(a) < min(b) and mb - ma > spread and -change > floor:
        return "gain"
    if min(a) > max(b) and ma - mb > spread and change > floor:
        return "loss"
    return "negligible"


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
        path = os.path.join(out, run + ".log")
        if not os.path.exists(path):
            continue
        text = read(path)
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

    def rv(k, arm):
        xs = [vals.get(k, {}).get((arm, r)) for r in reps]
        return None if not xs or any(x is None for x in xs) else xs

    def mean(xs):
        return sum(xs) / len(xs)

    with open(os.path.join(dest, "perf.tsv"), "w") as f:
        f.write("K/V type\tn_kv\tquery tokens\tarm\tkernel\t" + "\t".join(f"us/run, repeat {r}" for r in reps) +
                "\tmean\tspread (max - min)\n")
        for k in keys:
            for arm in ARMS:
                xs = rv(k, arm)
                if not xs:
                    continue
                f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{arm}\t{kernel(k[0], k[2], arm)}\t" +
                        "\t".join(f"{x:.2f}" for x in xs) + f"\t{mean(xs):.2f}\t{max(xs) - min(xs):.2f}\n")

    # the same-kernel floor: kdp and ldskdp run one kernel at head 256
    own = defaultdict(float)
    with open(os.path.join(dest, "floor.tsv"), "w") as f:
        f.write("K/V type\tn_kv\tquery tokens\tldskdp mean us\tkdp mean us\tldskdp / kdp - 1\n")
        for k in keys:
            a, b = rv(k, "ldskdp"), rv(k, "kdp")
            if a and b:
                d = mean(a) / mean(b) - 1
                own[k[2]] = max(own[k[2]], abs(d))
                f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{mean(a):.2f}\t{mean(b):.2f}\t{d * 100:+.2f} %\n")
    floor = {nt: max(R232A_FLOOR.get(nt, 0.0171), own.get(nt, 0.0)) for nt in sorted({k[2] for k in keys})}

    with open(os.path.join(dest, "verdict.tsv"), "w") as f:
        f.write("n_kv\tquery tokens\tarm\tkernel\tarm repeats us\tarm mean us\t"
                "Q8_0 preset (q8) repeats us\tq8 mean us\tarm / q8\tlarger repeat spread vs q8 us\tclass vs q8\t"
                "F16 (base) repeats us\tF16 mean us\tarm / F16\tlarger repeat spread vs F16 us\tclass vs F16\t"
                "floor used\t12 layers, arm - q8, ms a call (est.)\n")
        for k in keys:
            if k[0] != "q8_0":
                continue
            q = rv(k, "q8")
            fk = ("f16", k[1], k[2])
            fb = rv(fk, "base")
            for arm in ["q8"] + SWITCH_ARMS:
                a = rv(k, arm)
                if not a:
                    continue
                fl = floor[k[2]]
                cols = [str(k[1]), str(k[2]), arm, kernel(k[0], k[2], arm), ", ".join(f"{x:.2f}" for x in a), f"{mean(a):.2f}"]
                if q and arm != "q8":
                    sp = max(max(a) - min(a), max(q) - min(q))
                    cols += [", ".join(f"{x:.2f}" for x in q), f"{mean(q):.2f}", f"{mean(a) / mean(q):.4f}", f"{sp:.2f}",
                             classify(a, q, fl)]
                else:
                    cols += ["", "", "", "", ""]
                if fb:
                    sp = max(max(a) - min(a), max(fb) - min(fb))
                    cols += [", ".join(f"{x:.2f}" for x in fb), f"{mean(fb):.2f}", f"{mean(a) / mean(fb):.4f}", f"{sp:.2f}",
                             classify(a, fb, fl)]
                else:
                    cols += ["", "", "", "", ""]
                cols += [f"{fl * 100:.2f} %", f"{12 * (mean(a) - mean(q)) / 1000:+.3f}" if q and arm != "q8" else ""]
                f.write("\t".join(cols) + "\n")

    with open(os.path.join(dest, "perf_runs.tsv"), "w") as f:
        f.write("order\trun\tPascal kernel paths that logged\n")
        for i, run in enumerate(order, 1):
            _, _, arm, r = run.split("-")
            f.write(f"{i}\t{run}\t{paths.get((arm, int(r[1:])), 'not run')}\n")


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
