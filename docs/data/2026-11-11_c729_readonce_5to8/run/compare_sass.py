#!/usr/bin/env python3
"""C729: compare two builds' device code, function by function, from cuobjdump's output (no GPU). From C767's
compare_sass.py (2026-11-11_c767_q5k_one_row/run/), with C729's expectations: the tip (pascal-p106-v4-b12246-c729) adds
the instantiations of 5 to 8 tokens at R = 1, mul_mat_vec_q_pascal_dmt<type, 5..8, 1, PF> to mmvq-pascal-dmt.cu and
mul_mat_vec_q5_K_pascal_mt<5..8, 1, PF> to mmvq-pascal-q5k.cu, and every other function of the build keeps the base's
code.

Usage: compare_sass.py <base dir> <tip dir> <out dir>
Each build dir holds what checks.sh's sass step dumps with CUDA 12.9's cuobjdump: obj/<object>.sass, .res and .ptx for
every CUDA object of the build tree (cuobjdump -sass, -res-usage, -ptx), and llama-server.sass (cuobjdump -sass of the
rig tarball's bin/llama-server, which links every kernel).

Names inside anonymous namespaces and static functions carry a hash of their translation unit (_GLOBAL__N__<8 hex>_..._cu_
<8 hex>, _INTERNAL_<8 hex>_..._cu_<8 hex>); both hashes are replaced by H before comparing, in names and in code. A
function's code is every line from its "Function :" line to the next function or fatbin section, blanks collapsed.

Writes to <out dir>:
  sass_objects.tsv  per CUDA object: functions in each build, the same, different, only in one; the PTX the same or not
  sass_binary.tsv   the same for bin/llama-server
  sass_kernels.tsv  every instantiation of both read-once kernels: code hash and registers in each build, and in the tip
                    the stack, local memory and spill instructions (LDL, STL), the dp4a (IDP), shared loads (LDS),
                    global loads (LDG), FFMA, FADD and I2F, and the last two FFMA of the super-block's expression
  sass_differ.txt   every function that differs or is in one build only, and whether C729 expects it
Expected: tip-only functions that are the new instantiations (in their two objects and in llama-server), and those two
objects' PTX (the new functions are in it). Exit status 1 for anything else.
"""
import hashlib
import os
import re
import sys

HASH = re.compile(r'(_GLOBAL__N__|_INTERNAL_)[0-9a-f]{8}(_\d+_\w+?_cu_)[0-9a-f]{8}')
FUNC = re.compile(r'^\s*Function : (\S+)')
SECTION = re.compile(r'^(Fatbin |\s*code for sm_|\s*arch = |\s*code version|\s*host = |\s*compile_size|\s*identifier)')
DMT = re.compile(r'mul_mat_vec_q_pascal_dmtIL9ggml_type(\d+)ELi(\d)ELi(\d)ELb([01])E')
Q5K = re.compile(r'mul_mat_vec_q5_K_pascal_mtILi(\d)ELi(\d)ELb([01])E')
OBJS = {'mmvq-pascal-dmt.cu': DMT, 'mmvq-pascal-q5k.cu': Q5K}
TYPES = {'12': 'q4_K', '13': 'q5_K', '23': 'iq4_xs'}
OP = re.compile(r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*?);')


def norm(s):
    return HASH.sub(r'\1H\2H', s)


def functions(path, keep=False):
    """{(normalized name, occurrence): md5 of the normalized code} (keep: also the code lines)"""
    out, code, seen = {}, {}, {}
    key, h, lines = None, None, None
    with open(path, errors='replace') as f:
        for line in f:
            m = FUNC.match(line)
            if m or SECTION.match(line):
                if key is not None:
                    out[key] = h.hexdigest()
                    if keep:
                        code[key] = lines
                key, h, lines = None, None, None
                if m:
                    n = norm(m.group(1))
                    k = seen.get(n, 0)
                    seen[n] = k + 1
                    key, h, lines = (n, k), hashlib.md5(), []
                continue
            if key is not None:
                t = ' '.join(norm(line).split())
                if t:
                    h.update(t.encode() + b'\n')
                    lines.append(line)
    if key is not None:
        out[key] = h.hexdigest()
        if keep:
            code[key] = lines
    return (out, code) if keep else out


def resources(path):
    """{normalized name: {REG, STACK, LOCAL, SHARED}} from cuobjdump -res-usage"""
    res, name = {}, None
    with open(path, errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('Function '):
                name = norm(s.split()[1].rstrip(':'))
            elif name and 'REG:' in s:
                res[name] = {k: int(v) for k, v in re.findall(r'(REG|STACK|LOCAL|SHARED):(\d+)', s)}
                name = None
    return res


def text_md5(path):
    with open(path, errors='replace') as f:
        return hashlib.md5(' '.join(norm(f.read()).split()).encode()).hexdigest()


def compare(a, b):
    same = sum(1 for k in a if k in b and a[k] == b[k])
    differ = [k for k in a if k in b and a[k] != b[k]]
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    return same, differ, only_a, only_b


def new_inst(name):
    """an instantiation C729 adds: NT 5 to 8 at R = 1"""
    for rx in OBJS.values():
        m = rx.search(name)
        if m:
            g = m.groups()
            nt, r = (int(g[1]), int(g[2])) if rx is DMT else (int(g[0]), int(g[1]))
            return 5 <= nt <= 8 and r == 1
    return False


def counts(lines):
    c = {k: 0 for k in ('LDL', 'STL', 'IDP', 'LDS', 'LDG', 'FFMA', 'FADD', 'I2F')}
    ffma = []
    for line in lines:
        m = OP.match(line)
        if not m:
            continue
        op = m.group(1).split('.')[0]
        if op in c:
            c[op] += 1
        if op == 'FFMA':
            ffma.append(' '.join((m.group(1) + m.group(2)).split()))
    return c, ffma


def main():
    base, tip, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    bad = 0
    differ_lines = []
    tot = [0, 0, 0, 0, 0, 0]
    rows = []
    objs = sorted(f[:-5] for f in os.listdir(os.path.join(tip, 'obj')) if f.endswith('.sass'))
    base_objs = sorted(f[:-5] for f in os.listdir(os.path.join(base, 'obj')) if f.endswith('.sass'))
    for o in sorted(set(objs) | set(base_objs)):
        pa, pb = os.path.join(base, 'obj', o + '.sass'), os.path.join(tip, 'obj', o + '.sass')
        a = functions(pa) if os.path.exists(pa) else {}
        b = functions(pb) if os.path.exists(pb) else {}
        same, differ, only_a, only_b = compare(a, b)
        xa, xb = os.path.join(base, 'obj', o + '.ptx'), os.path.join(tip, 'obj', o + '.ptx')
        ptx = 'same' if os.path.exists(xa) and os.path.exists(xb) and text_md5(xa) == text_md5(xb) else 'differs'
        rows.append((o, len(a), len(b), same, len(differ), len(only_a), len(only_b), ptx))
        for k in differ:
            differ_lines.append(f'{o}\tdiffers\t{k[0]}#{k[1]}\tUNEXPECTED')
            bad += 1
        for k in only_a:
            differ_lines.append(f'{o}\tbase only\t{k[0]}#{k[1]}\tUNEXPECTED')
            bad += 1
        for k in only_b:
            ok = o in OBJS and new_inst(k[0])
            differ_lines.append(f'{o}\ttip only\t{k[0]}#{k[1]}\t{"expected (C729)" if ok else "UNEXPECTED"}')
            bad += not ok
        if ptx != 'same':
            ok = o in OBJS and only_b and not differ and not only_a
            differ_lines.append(f'{o}\tPTX differs\t-\t{"expected (C729: the new instantiations)" if ok else "UNEXPECTED"}')
            bad += not ok
        for i, v in enumerate((len(a), len(b), same, len(differ), len(only_a), len(only_b))):
            tot[i] += v
    with open(os.path.join(out, 'sass_objects.tsv'), 'w') as f:
        f.write('object\tfunctions_base\tfunctions_tip\tsame\tdiffer\tbase_only\ttip_only\tptx\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
        nptx = sum(1 for r in rows if r[7] == 'same')
        f.write(f'all {len(rows)} objects\t' + '\t'.join(map(str, tot)) + f'\t{nptx} of {len(rows)} same\n')

    a = functions(os.path.join(base, 'llama-server.sass'))
    b = functions(os.path.join(tip, 'llama-server.sass'))
    same, differ, only_a, only_b = compare(a, b)
    for k in differ:
        differ_lines.append(f'bin/llama-server\tdiffers\t{k[0]}#{k[1]}\tUNEXPECTED')
        bad += 1
    for k in only_a:
        differ_lines.append(f'bin/llama-server\tbase only\t{k[0]}#{k[1]}\tUNEXPECTED')
        bad += 1
    for k in only_b:
        ok = new_inst(k[0])
        differ_lines.append(f'bin/llama-server\ttip only\t{k[0]}#{k[1]}\t{"expected (C729)" if ok else "UNEXPECTED"}')
        bad += not ok
    with open(os.path.join(out, 'sass_binary.tsv'), 'w') as f:
        f.write('binary\tfunctions_base\tfunctions_tip\tsame\tdiffer\tbase_only\ttip_only\n')
        f.write(f'bin/llama-server\t{len(a)}\t{len(b)}\t{same}\t{len(differ)}\t{len(only_a)}\t{len(only_b)}\n')

    # every instantiation of both kernels
    n_new = 0
    n_inst = 0
    with open(os.path.join(out, 'sass_kernels.tsv'), 'w') as f:
        f.write('kernel\ttype\tNT\tR\tPF\tcode_base\tcode_tip\tsame\treg_base\treg_tip\tstack\tlocal\tLDL\tSTL\tIDP\tLDS\t'
                'LDG\tFFMA\tFADD\tI2F\tlast_two_FFMA\n')
        for o, rx in OBJS.items():
            qa = functions(os.path.join(base, 'obj', o + '.sass'))
            qb, code = functions(os.path.join(tip, 'obj', o + '.sass'), keep=True)
            ra = resources(os.path.join(base, 'obj', o + '.res'))
            rb = resources(os.path.join(tip, 'obj', o + '.res'))
            inst = {}
            for (n, k) in list(qa) + list(qb):
                m = rx.search(n)
                if m and k == 0:
                    g = m.groups()
                    t = (TYPES.get(g[0], g[0]), int(g[1]), int(g[2]), int(g[3])) if rx is DMT else \
                        ('q5_K', int(g[0]), int(g[1]), int(g[2]))
                    inst[t] = n
            for t in sorted(inst):
                n = inst[t]
                n_inst += 1
                ha, hb = qa.get((n, 0), '-'), qb.get((n, 0), '-')
                state = 'yes' if ha == hb else ('new' if ha == '-' and new_inst(n) else 'NO')
                n_new += state == 'new'
                bad += state == 'NO'
                c, ffma = counts(code.get((n, 0), []))
                r = rb.get(n, {})
                last = '; '.join(ffma[-2:])
                kern = 'mul_mat_vec_q_pascal_dmt' if rx is DMT else 'mul_mat_vec_q5_K_pascal_mt'
                f.write(f'{kern}\t{t[0]}\t{t[1]}\t{t[2]}\t{t[3]}\t{ha[:12]}\t{hb[:12]}\t{state}\t'
                        f'{ra.get(n, {}).get("REG", "-")}\t{r.get("REG", "-")}\t{r.get("STACK", "-")}\t'
                        f'{r.get("LOCAL", "-")}\t' + '\t'.join(str(c[k]) for k in c) + f'\t{last}\n')
    # 2 types x 4 token counts x 2 prefetch settings, and Q5_K's 4 x 2
    if n_new != 24:
        bad += 1
        differ_lines.append(f'read-once kernels\tnew instantiations\t{n_new} new, 24 expected\tUNEXPECTED')

    with open(os.path.join(out, 'sass_differ.txt'), 'w') as f:
        f.write('\n'.join(differ_lines) + ('\n' if differ_lines else ''))
    print(f'objects {len(rows)}, functions base {tot[0]} tip {tot[1]}, same {tot[2]}, differ {tot[3]}, '
          f'base only {tot[4]}, tip only {tot[5]}; llama-server functions {len(a)} / {len(b)}, same {same}, tip only '
          f'{len(only_b)}; read-once instantiations {n_inst} ({n_new} new); unexpected {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
