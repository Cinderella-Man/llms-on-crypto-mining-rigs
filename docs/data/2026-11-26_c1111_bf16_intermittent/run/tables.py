#!/usr/bin/env python3
"""C1111 (Q1844): tables from run/c1111.sh's logs, copied off the build machine.

  python3 -I tables.py <work dir> <out dir>

<work dir> is c1111.sh's W (or a copy of its paths/ and san/ folders). Writes:
  kernels.tsv          the kernels each Nsight Systems run launched (cuda_gpu_kern_sum, per build): the BF16 case and its
                       F16 sibling alone (paths/), the sibling set with and without CUDA graphs (kernels/)
  initcheck_kernels.tsv  per build and sanitizer run with every report printed: initcheck's reports per kernel
  initcheck_cases.tsv    the same per test case (est.: a report is counted for the next case line the log prints
                       after it, since the sanitizer's reports and the test's lines share one log)
"""
import collections
import csv
import os
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CASE = re.compile(r'(MUL_MAT\([^)]*\)): (OK|FAIL)$')


def lines(path):
    with open(path, 'rb') as f:
        for raw in f:
            yield ANSI.sub('', raw.decode('utf-8', 'replace')).rstrip('\n')


def short_kernel(name):
    # "void mul_mat_q<(ggml_type)2, (int)32, ...>(const char *, ...)+0x1af0" -> "mul_mat_q<(ggml_type)2, (int)32, ...>"
    name = re.sub(r'\+0x[0-9a-f]+$', '', name.strip())
    name = re.sub(r'^void ', '', name)
    depth, out = 0, []
    for ch in name:
        if ch == '(' and depth == 0 and out:  # the parameter list (casts in template arguments sit at depth > 0)
            break
        depth += ch == '<'
        depth -= ch == '>'
        out.append(ch)
    return ''.join(out)


def kern_sum(path):
    """(instances, kernel name) rows of an Nsight Systems --stats log's cuda_gpu_kern_sum table"""
    on = False
    for ln in lines(path):
        if "Executing 'cuda_gpu_kern_sum'" in ln:
            on = True
            continue
        if on and ln.startswith('[') and 'Executing' in ln:
            break
        if on and re.match(r'^\s+[0-9.]+\s+[0-9]+\s+[0-9]+\s', ln):
            f = ln.split()
            # Time (%), Total Time (ns), Instances, Avg, Med, Min, Max, StdDev, Name
            yield f[2], ' '.join(f[8:])


def kernels(work, out):
    # paths/<build>-nsys-<case>.log: one case alone; kernels/<build>-<arm>-<set>-cc<n>.log: a sibling set, graphs traced
    rows = []
    for sub, pat in (('paths', r'(.+)-nsys-(bf16|f16)\.log$'), ('kernels', r'(base|head)-(.+)\.log$')):
        d = os.path.join(work, sub)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            m = re.match(pat, fn)
            if not m:
                continue
            run = ('alone-' if sub == 'paths' else '') + m.group(2)
            for n, k in kern_sum(os.path.join(d, fn)):
                rows.append([m.group(1), run, n, k])
    with open(os.path.join(out, 'kernels.tsv'), 'w', newline='') as fo:
        w = csv.writer(fo, delimiter='\t', lineterminator='\n')
        w.writerow(['build', 'run', 'instances', 'kernel'])
        w.writerows(rows)


def initcheck(work, out):
    sdir = os.path.join(work, 'san')
    per_kernel, per_case = [], []
    for fn in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        m = re.match(r'(.+)-initcheck-(.+)-noapi-all\.log$', fn)
        if not m:
            continue
        build, mode = m.group(1), m.group(2)
        kern = collections.Counter()
        case = collections.Counter()
        case_kern = collections.defaultdict(set)
        pending, kind = [], None
        for ln in lines(os.path.join(sdir, fn)):
            if ln.startswith('========= Uninitialized'):
                kind = ln[len('========= '):]
                continue
            if kind and ln.startswith('=========     at '):
                k = short_kernel(ln[len('=========     at '):])
                kern[k] += 1
                pending.append(k)
                kind = None
                continue
            c = CASE.search(ln)
            if c and pending:
                case[c.group(1)] += len(pending)
                case_kern[c.group(1)].update(pending)
                pending = []
        for k, n in kern.most_common():
            per_kernel.append([build, mode, n, k])
        for c, n in case.most_common():
            per_case.append([build, mode, n, c, '; '.join(sorted(case_kern[c]))])
    with open(os.path.join(out, 'initcheck_kernels.tsv'), 'w', newline='') as fo:
        w = csv.writer(fo, delimiter='\t', lineterminator='\n')
        w.writerow(['build', 'mode', 'reports', 'kernel'])
        w.writerows(per_kernel)
    with open(os.path.join(out, 'initcheck_cases.tsv'), 'w', newline='') as fo:
        w = csv.writer(fo, delimiter='\t', lineterminator='\n')
        w.writerow(['build', 'mode', 'reports', 'case', 'kernels'])
        w.writerows(per_case)


def main():
    work, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    kernels(work, out)
    initcheck(work, out)


if __name__ == '__main__':
    main()
