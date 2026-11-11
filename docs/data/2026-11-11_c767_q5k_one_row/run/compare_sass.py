#!/usr/bin/env python3
"""C767: compare two builds' device code, function by function, from cuobjdump's output (no GPU). From C766's
compare_sass.py (2026-11-11_c766_q5k_default/run/), with C767's expectations: the tip (pascal-p106-v4-b12246-c767) adds
the one-token instantiations mul_mat_vec_q5_K_pascal_mt<1, R, PF> to mmvq-pascal-q5k.cu, and every other function of
the build keeps the base's code.

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
  sass_q5k.tsv      every instantiation mul_mat_vec_q5_K_pascal_mt<NT, R, PF>: code hash and registers in each build
  sass_differ.txt   every function that differs or is in one build only, and whether C767 expects it
Expected: tip-only functions that are the NT = 1 instantiations (in mmvq-pascal-q5k.cu.o and in llama-server), and that
object's PTX (the new functions are in it). Exit status 1 for anything else.
"""
import hashlib
import os
import re
import sys

HASH = re.compile(r'(_GLOBAL__N__|_INTERNAL_)[0-9a-f]{8}(_\d+_\w+?_cu_)[0-9a-f]{8}')
FUNC = re.compile(r'^\s*Function : (\S+)')
SECTION = re.compile(r'^(Fatbin |\s*code for sm_|\s*arch = |\s*code version|\s*host = |\s*compile_size|\s*identifier)')
Q5K = re.compile(r'mul_mat_vec_q5_K_pascal_mtILi(\d)ELi(\d)ELb([01])E')
Q5K_OBJ = 'mmvq-pascal-q5k.cu'


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


def registers(path):
    """{normalized name: REG} from cuobjdump -res-usage"""
    regs, name = {}, None
    with open(path, errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('Function '):
                name = norm(s.split()[1].rstrip(':'))
            elif name and 'REG:' in s:
                regs[name] = int(re.search(r'REG:(\d+)', s).group(1))
                name = None
    return regs


def text_md5(path):
    with open(path, errors='replace') as f:
        return hashlib.md5(' '.join(norm(f.read()).split()).encode()).hexdigest()


def compare(a, b):
    same = sum(1 for k in a if k in b and a[k] == b[k])
    differ = [k for k in a if k in b and a[k] != b[k]]
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    return same, differ, only_a, only_b


def one_token(name):
    m = Q5K.search(name)
    return m is not None and m.group(1) == '1'


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
            ok = o == Q5K_OBJ and one_token(k[0])
            differ_lines.append(f'{o}\ttip only\t{k[0]}#{k[1]}\t{"expected (C767)" if ok else "UNEXPECTED"}')
            bad += not ok
        if ptx != 'same':
            ok = o == Q5K_OBJ and only_b and not differ and not only_a
            differ_lines.append(f'{o}\tPTX differs\t-\t{"expected (C767: the new instantiations)" if ok else "UNEXPECTED"}')
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
        ok = one_token(k[0])
        differ_lines.append(f'bin/llama-server\ttip only\t{k[0]}#{k[1]}\t{"expected (C767)" if ok else "UNEXPECTED"}')
        bad += not ok
    with open(os.path.join(out, 'sass_binary.tsv'), 'w') as f:
        f.write('binary\tfunctions_base\tfunctions_tip\tsame\tdiffer\tbase_only\ttip_only\n')
        f.write(f'bin/llama-server\t{len(a)}\t{len(b)}\t{same}\t{len(differ)}\t{len(only_a)}\t{len(only_b)}\n')

    # every instantiation, from the Q5_K object
    qa = functions(os.path.join(base, 'obj', Q5K_OBJ + '.sass'))
    qb = functions(os.path.join(tip, 'obj', Q5K_OBJ + '.sass'))
    ra = registers(os.path.join(base, 'obj', Q5K_OBJ + '.res'))
    rb = registers(os.path.join(tip, 'obj', Q5K_OBJ + '.res'))
    inst = {}
    for (n, k), h in list(qa.items()) + list(qb.items()):
        m = Q5K.search(n)
        if m and k == 0:
            inst[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = n
    n_new = 0
    with open(os.path.join(out, 'sass_q5k.tsv'), 'w') as f:
        f.write('NT\tR\tPF\tcode_base\tcode_tip\tsame\treg_base\treg_tip\n')
        for t in sorted(inst):
            n = inst[t]
            ha, hb = qa.get((n, 0), '-'), qb.get((n, 0), '-')
            state = 'yes' if ha == hb else ('new' if ha == '-' and t[0] == 1 else 'NO')
            n_new += state == 'new'
            f.write(f'{t[0]}\t{t[1]}\t{t[2]}\t{ha[:12]}\t{hb[:12]}\t{state}\t{ra.get(n, "-")}\t{rb.get(n, "-")}\n')
    if n_new != 4:
        bad += 1
        differ_lines.append(f'{Q5K_OBJ}\tone-token instantiations\t{n_new} new, 4 expected\tUNEXPECTED')

    with open(os.path.join(out, 'sass_differ.txt'), 'w') as f:
        f.write('\n'.join(differ_lines) + ('\n' if differ_lines else ''))
    print(f'objects {len(rows)}, functions base {tot[0]} tip {tot[1]}, same {tot[2]}, differ {tot[3]}, '
          f'base only {tot[4]}, tip only {tot[5]}; llama-server functions {len(a)} / {len(b)}, same {same}, tip only '
          f'{len(only_b)}; q5k instantiations {len(inst)} ({n_new} new); unexpected {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
