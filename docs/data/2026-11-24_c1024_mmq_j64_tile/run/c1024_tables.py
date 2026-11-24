#!/usr/bin/env python3
"""C1024: tables from run.sh's test-backend-ops arms and SASS step.

Usage: python3 -I c1024_tables.py <work dir> <out dir>
Reads <work dir>/bops/rc.txt and <arm>.log, <work dir>/checks/sass/j64.tsv. Writes into <out dir>:
  bops.tsv       one row per arm: exit code, the 'N/N tests passed' line, OK, FAIL and 'not supported' counts, how
                 many of the C1024 cases ran and passed, the switch's log line, how many first-launch lines
  j64_cases.tsv  one row per C1024 case (MUL_MAT of the 11 types at 49 to 512 columns, test-backend-ops' C1024 block):
                 its verdict in each arm
  launches.tsv   the first-launch lines of the switch's kernels: arm, type, row check, tile
  j64_sass.tsv   a copy of the SASS step's j64.tsv
and prints a summary.
"""
import os
import re
import shutil
import sys

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
B = os.path.join(W, "bops")
CASE = re.compile(r"^\s*(MUL_MAT(?:_ID)?\((.*)\)): (OK|FAIL|not supported.*)$")
J64 = re.compile(r"^type_a=(q1_0|q2_0|q4_0|q4_1|q5_0|q5_1|q8_0|iq2_xxs|iq3_xxs|iq4_nl|iq4_xs),type_b=f32,"
                 r"m=(64|128|192|256|1000|4096),n=(49|64|65|128|512),k=(512|1024),")
LAUNCH = re.compile(r"first launch of (\S+)'s J=64 kernel \((row check|no row check)\), tile (\d+) x (\d+)")

rcs = {}
if os.path.exists(os.path.join(B, "rc.txt")):
    for ln in open(os.path.join(B, "rc.txt")):
        a, _, rc = ln.strip().partition(" rc=")
        rcs[a] = rc
arms = [a for a in ("mul_mat-unset", "mul_mat-j64i64", "mul_mat_id-unset", "mul_mat_id-j64i64", "mul_mat-j64-mode1")
        if os.path.exists(os.path.join(B, a + ".log"))]

cases = {}   # case -> {arm: verdict}
order = []
launches = []
with open(os.path.join(OUT, "bops.tsv"), "w") as f:
    f.write("arm\trc\tpassed\tOK\tFAIL\tnot_supported\tC1024_cases\tC1024_OK\tswitch_line\tfirst_launch_lines\n")
    for a in arms:
        ok = fail = ns = jn = jok = 0
        passed, switch, nl = "", "", 0
        for ln in open(os.path.join(B, a + ".log"), errors="replace"):
            ln = ln.rstrip("\n")
            m = CASE.match(ln)
            if m:
                v = m.group(3)
                ok += v == "OK"
                fail += v == "FAIL"
                ns += v.startswith("not supported")
                if m.group(1).startswith("MUL_MAT(") and J64.match(m.group(2)):
                    jn += 1
                    jok += v == "OK"
                    if m.group(1) not in cases:
                        cases[m.group(1)] = {}
                        order.append(m.group(1))
                    cases[m.group(1)][a] = v
                continue
            p = re.search(r"(\d+/\d+) tests passed", ln)
            if p:
                passed = p.group(1)
            if "GGML_CUDA_MMQ_J64_I64=" in ln and "keep 64 rows" in ln:
                switch = ln.split(": ", 1)[-1].strip()
            lm = LAUNCH.search(ln)
            if lm:
                nl += 1
                launches.append([a, lm.group(1), lm.group(2), f"{lm.group(3)} x {lm.group(4)}"])
        f.write(f"{a}\t{rcs.get(a, '')}\t{passed}\t{ok}\t{fail}\t{ns}\t{jn}\t{jok}\t{switch or '-'}\t{nl}\n")

with open(os.path.join(OUT, "j64_cases.tsv"), "w") as f:
    f.write("case\t" + "\t".join(a for a in arms if a.startswith("mul_mat-")) + "\n")
    for c in order:
        f.write(c + "\t" + "\t".join(cases[c].get(a, "-") for a in arms if a.startswith("mul_mat-")) + "\n")

with open(os.path.join(OUT, "launches.tsv"), "w") as f:
    f.write("arm\ttype\trow_check\ttile_rows_x_cols\n")
    for r in launches:
        f.write("\t".join(r) + "\n")

sj = os.path.join(W, "checks", "sass", "j64.tsv")
if os.path.exists(sj):
    shutil.copy(sj, os.path.join(OUT, "j64_sass.tsv"))

for ln in open(os.path.join(OUT, "bops.tsv")):
    print(ln.rstrip("\n"))
print(f"C1024 cases: {len(order)}; every arm OK on every one: "
      f"{sum(all(v == 'OK' for v in cases[c].values()) for c in order)}")
print(f"first-launch lines: {len(launches)} "
      f"({', '.join(sorted(set(f'{r[0]}: {r[3]}' for r in launches)))})")
