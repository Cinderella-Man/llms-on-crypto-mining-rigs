#!/usr/bin/env python3
"""C727: compare two builds' device code, function by function, from cuobjdump's output (no GPU). From C767's
compare_sass.py (2026-11-11_c767_q5k_one_row/run/), with C727's expectations: the tip (pascal-p106-v4-b12246-c727) adds
the Q6_K instantiations mul_mat_vec_q_pascal_dmt<GGML_TYPE_Q6_K, NT, R, PF> (NT 2-4, R 1-2, PF 0-1: 12) to
mmvq-pascal-dmt.cu, and every other function keeps the base's code, IQ4_XS's and Q4_K's instantiations of the same
kernel among them.

Usage: compare_sass.py <base llama-server.sass> <tip llama-server.sass> <tip mmvq-pascal-dmt.cu.res> <out dir>
The .sass files are cuobjdump -sass of each rig tarball's bin/llama-server, which links every kernel; the .res file is
cuobjdump -res-usage of the tip's mmvq-pascal-dmt.cu.o.

Names inside anonymous namespaces and static functions carry a hash of their translation unit (_GLOBAL__N__<8 hex>_..._cu_
<8 hex>, _INTERNAL_<8 hex>_..._cu_<8 hex>); both hashes are replaced by H before comparing, in names and in code. A
function's code is every line from its "Function :" line to the next function or fatbin section, blanks collapsed.

Writes to <out dir>:
  sass_binary.tsv  bin/llama-server: functions in each build, the same, different, only in one
  sass_dmt.tsv     every instantiation of mul_mat_vec_q_pascal_dmt: type, NT, R, PF, code hash in each build, the same
                   or new, and the tip's registers, stack and local memory
  sass_differ.txt  every function that differs or is in one build only, and whether C727 expects it
Exit status 1 for anything unexpected.
"""
import hashlib
import os
import re
import sys

HASH = re.compile(r'(_GLOBAL__N__|_INTERNAL_)[0-9a-f]{8}(_\d+_\w+?_cu_)[0-9a-f]{8}')
FUNC = re.compile(r'^\s*Function : (\S+)')
SECTION = re.compile(r'^(Fatbin |\s*code for sm_|\s*arch = |\s*code version|\s*host = |\s*compile_size|\s*identifier)')
DMT = re.compile(r'mul_mat_vec_q_pascal_dmtIL9ggml_type(\d+)ELi(\d)ELi(\d)ELb([01])E')
TYPES = {12: 'q4_K', 14: 'q6_K', 23: 'iq4_xs'}


def norm(s):
    return HASH.sub(r'\1H\2H', s)


def functions(path):
    """{(normalized name, occurrence): md5 of the normalized code}"""
    out, seen = {}, {}
    key, h = None, None
    with open(path, errors='replace') as f:
        for line in f:
            m = FUNC.match(line)
            if m or SECTION.match(line):
                if key is not None:
                    out[key] = h.hexdigest()
                key, h = None, None
                if m:
                    n = norm(m.group(1))
                    k = seen.get(n, 0)
                    seen[n] = k + 1
                    key, h = (n, k), hashlib.md5()
                continue
            if key is not None:
                t = ' '.join(norm(line).split())
                if t:
                    h.update(t.encode() + b'\n')
    if key is not None:
        out[key] = h.hexdigest()
    return out


def resources(path):
    """{normalized name: (REG, STACK, LOCAL)} from cuobjdump -res-usage"""
    res, name = {}, None
    with open(path, errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('Function '):
                name = norm(s.split()[1].rstrip(':'))
            elif name and 'REG:' in s:
                g = lambda k: int(re.search(k + r':(\d+)', s).group(1)) if re.search(k + r':(\d+)', s) else -1
                res[name] = (g('REG'), g('STACK'), g('LOCAL'))
                name = None
    return res


def q6k_new(name):
    m = DMT.search(name)
    return m is not None and int(m.group(1)) == 14


def main():
    base_sass, tip_sass, tip_res, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    bad = 0
    lines = []
    a = functions(base_sass)
    b = functions(tip_sass)
    same = sum(1 for k in a if k in b and a[k] == b[k])
    differ = [k for k in a if k in b and a[k] != b[k]]
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    for k in differ:
        lines.append(f'bin/llama-server\tdiffers\t{k[0]}#{k[1]}\tUNEXPECTED')
        bad += 1
    for k in only_a:
        lines.append(f'bin/llama-server\tbase only\t{k[0]}#{k[1]}\tUNEXPECTED')
        bad += 1
    for k in only_b:
        ok = q6k_new(k[0])
        lines.append(f'bin/llama-server\ttip only\t{k[0]}#{k[1]}\t{"expected (C727)" if ok else "UNEXPECTED"}')
        bad += not ok
    with open(os.path.join(out, 'sass_binary.tsv'), 'w') as f:
        f.write('binary\tfunctions_base\tfunctions_tip\tsame\tdiffer\tbase_only\ttip_only\n')
        f.write(f'bin/llama-server\t{len(a)}\t{len(b)}\t{same}\t{len(differ)}\t{len(only_a)}\t{len(only_b)}\n')

    res = resources(tip_res)
    inst = {}
    for (n, k) in list(a) + list(b):
        m = DMT.search(n)
        if m and k == 0:
            inst[(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))] = n
    n_new = n_spill = 0
    with open(os.path.join(out, 'sass_dmt.tsv'), 'w') as f:
        f.write('type\tNT\tR\tPF\tcode_base\tcode_tip\tsame\treg_tip\tstack_tip\tlocal_tip\n')
        for t in sorted(inst, key=lambda t: (TYPES.get(t[0], str(t[0])), t[1:])):
            n = inst[t]
            ha, hb = a.get((n, 0), '-'), b.get((n, 0), '-')
            state = 'yes' if ha == hb else ('new' if ha == '-' and t[0] == 14 else 'NO')
            n_new += state == 'new'
            r = res.get(n, (-1, -1, -1))
            n_spill += t[0] == 14 and (r[1] != 0 or r[2] != 0)
            f.write(f'{TYPES.get(t[0], t[0])}\t{t[1]}\t{t[2]}\t{t[3]}\t{ha[:12]}\t{hb[:12]}\t{state}\t{r[0]}\t{r[1]}\t{r[2]}\n')
    if n_new != 12:
        bad += 1
        lines.append(f'mmvq-pascal-dmt.cu\tQ6_K instantiations\t{n_new} new, 12 expected\tUNEXPECTED')
    if n_spill:
        bad += 1
        lines.append(f'mmvq-pascal-dmt.cu\tQ6_K instantiations\t{n_spill} with stack or local memory\tUNEXPECTED')

    with open(os.path.join(out, 'sass_differ.txt'), 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    regs = sorted(r[0] for (n, r) in res.items() if q6k_new(n))
    print(f'llama-server functions {len(a)} / {len(b)}, same {same}, differ {len(differ)}, base only {len(only_a)}, '
          f'tip only {len(only_b)}; dmt instantiations {len(inst)} ({n_new} new Q6_K), Q6_K registers '
          f'{regs[0] if regs else "-"}-{regs[-1] if regs else "-"}, {n_spill} with stack or local memory; unexpected {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
