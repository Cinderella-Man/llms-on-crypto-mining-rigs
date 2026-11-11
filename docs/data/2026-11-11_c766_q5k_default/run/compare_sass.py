#!/usr/bin/env python3
"""C766: compare two builds' device code, function by function, from cuobjdump's output (no GPU).

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
  sass_q5k.tsv      the 12 instantiations mul_mat_vec_q5_K_pascal_mt<NT, R, PF>: code hash and registers in each build
  sass_settings.tsv for each setting of GGML_CUDA_MMVQ_PASCAL_Q5K_MT (_R, _PF), the instantiation each build launches
                    (from the host code of mmvq-pascal-q5k.cu before and after C766) and whether the two are one code,
                    with the base setting whose kernel the tip's setting runs
  sass_differ.txt   every function that differs or is in one build only
Exit status 1 if any function or PTX differs.
"""
import hashlib
import os
import re
import sys

HASH = re.compile(r'(_GLOBAL__N__|_INTERNAL_)[0-9a-f]{8}(_\d+_\w+?_cu_)[0-9a-f]{8}')
FUNC = re.compile(r'^\s*Function : (\S+)')
SECTION = re.compile(r'^(Fatbin |\s*code for sm_|\s*arch = |\s*code version|\s*host = |\s*compile_size|\s*identifier)')
Q5K = re.compile(r'mul_mat_vec_q5_K_pascal_mtILi(\d)ELi(\d)ELb([01])E')


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
            differ_lines.append(f'{o}\tdiffers\t{k[0]}#{k[1]}')
        for k in only_a:
            differ_lines.append(f'{o}\tbase only\t{k[0]}#{k[1]}')
        for k in only_b:
            differ_lines.append(f'{o}\ttip only\t{k[0]}#{k[1]}')
        bad += len(differ) + len(only_a) + len(only_b) + (ptx != 'same')
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
    bad += len(differ) + len(only_a) + len(only_b)
    for k in differ:
        differ_lines.append(f'bin/llama-server\tdiffers\t{k[0]}#{k[1]}')
    for k in only_a + only_b:
        differ_lines.append(f'bin/llama-server\tone build only\t{k[0]}#{k[1]}')
    with open(os.path.join(out, 'sass_binary.tsv'), 'w') as f:
        f.write('binary\tfunctions_base\tfunctions_tip\tsame\tdiffer\tbase_only\ttip_only\n')
        f.write(f'bin/llama-server\t{len(a)}\t{len(b)}\t{same}\t{len(differ)}\t{len(only_a)}\t{len(only_b)}\n')

    # the 12 instantiations, from the Q5_K object
    qa = functions(os.path.join(base, 'obj', 'mmvq-pascal-q5k.cu.sass'))
    qb = functions(os.path.join(tip, 'obj', 'mmvq-pascal-q5k.cu.sass'))
    ra = registers(os.path.join(base, 'obj', 'mmvq-pascal-q5k.cu.res'))
    rb = registers(os.path.join(tip, 'obj', 'mmvq-pascal-q5k.cu.res'))
    inst = {}
    for (n, k), h in qb.items():
        m = Q5K.search(n)
        if m and k == 0:
            inst[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = n
    with open(os.path.join(out, 'sass_q5k.tsv'), 'w') as f:
        f.write('NT\tR\tPF\tcode_base\tcode_tip\tsame\treg_base\treg_tip\n')
        for t in sorted(inst):
            n = inst[t]
            ha, hb = qa.get((n, 0), '-'), qb.get((n, 0), '-')
            f.write(f'{t[0]}\t{t[1]}\t{t[2]}\t{ha[:12]}\t{hb[:12]}\t{"yes" if ha == hb else "NO"}\t'
                    f'{ra.get(n, "-")}\t{rb.get(n, "-")}\n')

    # which instantiation a setting launches: before C766 R = 1 only with _R=1 (else 2) and the prefetch unless =2;
    # since C766 R = 2 only with _R=2 (else 1) and the prefetch only with _PF=1 and not =2
    def launched(build, mode, r_env, pf_env):
        if build == 'base':
            return (1 if r_env == 1 else 2), int(mode != 2)
        return (2 if r_env == 2 else 1), int(mode != 2 and pf_env == 1)

    settings = [('=1', 1, None, None), ('=2', 2, None, None), ('=1 _R=1', 1, 1, None), ('=2 _R=1', 2, 1, None),
                ('=1 _R=2', 1, 2, None), ('=2 _R=2', 2, 2, None), ('=1 _PF=1', 1, None, 1),
                ('=1 _R=1 _PF=1', 1, 1, 1), ('=1 _R=2 _PF=1', 1, 2, 1), ('=2 _R=2 _PF=1', 2, 2, 1)]
    with open(os.path.join(out, 'sass_settings.tsv'), 'w') as f:
        f.write('setting\tNT\tbase_kernel\ttip_kernel\tsame_code_as_base_setting\tbase_setting_with_tip_kernel\n')
        for s, mode, r_env, pf_env in settings:
            for nt in (2, 3, 4):
                rb_, pb_ = launched('base', mode, r_env, pf_env)
                rt_, pt_ = launched('tip', mode, r_env, pf_env)
                hb_ = qa.get((inst.get((nt, rb_, pb_), ''), 0))
                ht_ = qb.get((inst.get((nt, rt_, pt_), ''), 0))
                twins = [s2 for s2, m2, r2, p2 in settings
                         if qa.get((inst.get((nt,) + launched('base', m2, r2, p2), ''), 0)) == ht_ and ht_ is not None]
                f.write(f'{s}\t{nt}\t<{nt},{rb_},{pb_}>\t<{nt},{rt_},{pt_}>\t{"yes" if hb_ == ht_ else "no"}\t'
                        f'{" | ".join(twins) if twins else "NONE"}\n')
                if not twins:
                    bad += 1

    with open(os.path.join(out, 'sass_differ.txt'), 'w') as f:
        f.write('\n'.join(differ_lines) + ('\n' if differ_lines else ''))
    print(f'objects {len(rows)}, functions base {tot[0]} tip {tot[1]}, same {tot[2]}, differ {tot[3]}, '
          f'base only {tot[4]}, tip only {tot[5]}; llama-server functions {len(a)} / {len(b)}, same {same}; '
          f'q5k instantiations {len(inst)}; problems {bad}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
