#!/usr/bin/env python3
# Q671: per kernel and arm, from the outputs of compile_arms.sh: registers, stack frame, spills, load instruction forms,
# instruction count and theoretical occupancy on a 5-SM GP106 (sm_61).
#   registers, stack, local, shared   cuobjdump -res-usage (REG, STACK, LOCAL, SHARED; static shared memory only)
#   spill_st, spill_ld                ptxas -v "spill stores/loads" (bytes); it does not count the stack frame
#   stl, ldl                          STL and LDL instructions in the SASS: spills and stack frame together
#   ldg_forms                         every load mnemonic with its modifiers and how often it occurs, e.g. LDG.E.CI.128:12
#   blocks_sm, occupancy              resident blocks of the file's thread count per SM (registers, static shared
#                                     memory, 2,048 threads, 32 blocks) and active warps over the SM's 64
#   same_as_base                      the kernel's instruction text (without addresses and encodings) equals the
#                                     base arm's
# Usage: analyze.py <out dir> <work dir> [<work dir> ...]   (several compile_arms.sh runs; the arm "base" and the build's own
#   objects come from the first)
# Writes <out>/kernels.tsv.gz (one row per arm and kernel), summary_family_arm.tsv, forms_family_arm.tsv,
# changes_vs_base.tsv (kernels whose registers, stack or load forms an arm changed), baseline_check.tsv (base arm
# against the build's own objects) and half_math.tsv (FP16 arithmetic and conversions per file and arm).
import csv
import gzip
import hashlib
import os
import re
import sys
from collections import defaultdict, Counter

FUNC = re.compile(r'Function : (\S+)')
INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;')
RES_FN = re.compile(r'^\s*Function (\S+):')
RES_KV = re.compile(r'(REG|STACK|SHARED|LOCAL):(\d+)')
LOADS = {'LDG', 'LD', 'LDS', 'LDL', 'LDC', 'TLD', 'LDSM'}
STORES = {'STG', 'ST', 'STS', 'STL'}
HALF_OPS = {'HADD2', 'HMUL2', 'HFMA2', 'HSETP2', 'HMNMX2', 'HADD2_32I', 'HMUL2_32I', 'HFMA2_32I'}

# sm_61 limits (CUDA occupancy calculator): per SM 65,536 registers, 2,048 threads, 32 blocks, 96 KB shared memory;
# registers are allocated per warp in units of 256, shared memory in units of 256 bytes; 48 KB shared memory per block
SM_REGS, SM_THREADS, SM_BLOCKS, SM_SMEM, SM_WARPS = 65536, 2048, 32, 98304, 64


def blocks_per_sm(regs, smem, threads):
    warps = -(-threads // 32)
    regs_warp = -(-(max(regs, 1) * 32) // 256) * 256
    by_regs = SM_REGS // (regs_warp * warps)
    smem_b = -(-smem // 256) * 256
    by_smem = SM_SMEM // smem_b if smem_b else SM_BLOCKS
    return max(0, min(SM_BLOCKS, SM_THREADS // threads, by_regs, by_smem))


ANON = re.compile(r"(?<!\d)(\d+)_GLOBAL__N_")


def anon(name):
    """The mangled name with each anonymous-namespace identifier (a length, then that many characters: a hash, the file
    name and a counter that varies between compiles) replaced by <anon:file>."""
    out, i = [], 0
    for m in ANON.finditer(name):
        if m.start() < i:
            continue
        n = int(m.group(1))
        ident = name[m.start() + len(m.group(1)): m.start() + len(m.group(1)) + n]
        f = re.search(r"_\d+_(\w+?)_[0-9a-f]{8}(?:_\d+)?$", ident)
        out += [name[i:m.start()], "<anon:" + (f.group(1) if f else "") + ">"]
        i = m.start() + len(m.group(1)) + n
    return "".join(out) + name[i:]


def parse_sass(path):
    """name -> (insn count, Counter of mnemonics, digest of the instruction text)"""
    out = {}
    name, ops, h, n = None, None, None, 0
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            m = FUNC.search(line)
            if m:
                if name:
                    out[name] = (n, ops, h.hexdigest())
                name, ops, h, n = anon(m.group(1)), Counter(), hashlib.md5(), 0
                continue
            if name is None:
                continue
            m = INSN.match(line)
            if not m:
                continue
            text = m.group(1)
            h.update(text.encode() + b'\n')
            toks = text.split()
            op = toks[1] if text.startswith('@') and len(toks) > 1 else toks[0]
            ops[op.rstrip(',')] += 1
            if op.startswith('HADD2.F32') and '-RZ' in text:
                ops['HADD2_CVT'] += 1  # a conversion done as half arithmetic (Q672)
            n += 1
    if name:
        out[name] = (n, ops, h.hexdigest())
    return out


def parse_res(path):
    out, name = {}, None
    for line in open(path, errors='replace'):
        m = RES_FN.match(line)
        if m:
            name = anon(m.group(1))
            continue
        if name and "REG:" in line:
            out[name] = {k: int(v) for k, v in RES_KV.findall(line)}
            name = None
    return out


def parse_ptxas(path):
    out, cur = {}, None
    for line in open(path, errors='replace'):
        m = re.search(r"Compiling entry function '(\S+)'", line) or re.search(r'Function properties for (\S+)', line)
        if m:
            cur = anon(m.group(1))
            continue
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if m and cur:
            out[cur] = tuple(int(x) for x in m.groups())
    return out


def parse_dem(path, names):
    try:
        dem = [l.rstrip('\n') for l in open(path, errors='replace')]
    except OSError:
        return {}
    return dict(zip(names, dem)) if len(dem) == len(names) else {}


def main():
    out, works = sys.argv[1], sys.argv[2:]
    work = works[0]
    os.makedirs(out, exist_ok=True)
    files = {}
    for r in csv.DictReader(open(os.path.join(work, 'run', 'files.tsv')), delimiter='\t'):
        files[r['file']] = (r['family'], int(r['threads']))
    arm_work, times, arms = {}, {}, []
    for w in works:
        for l in open(os.path.join(w, "jobs.tsv")):
            a = l.split("\t")[0]
            if a not in arm_work:
                arm_work[a] = w
                arms.append(a)
        for l in open(os.path.join(w, "time.tsv")):
            a, f, rc, s = l.split("\t")
            times[(a, f)] = (rc, s)

    rows = []
    base_digest = {}  # (file, kernel) -> digest
    base_row = {}
    for arm in arms:
        for f, (fam, thr) in files.items():
            sp = os.path.join(arm_work[arm], 'sass', arm, f + '.sass.gz')
            if not os.path.exists(sp):
                continue
            sass = parse_sass(sp)
            res = parse_res(os.path.join(arm_work[arm], 'res', arm, f + '.res.txt'))
            spill = parse_ptxas(os.path.join(arm_work[arm], 'ptxas', arm, f + '.txt'))
            names = list(sass)
            dem = parse_dem(os.path.join(arm_work[arm], 'res', arm, f + '.dem.txt'), list(res))
            for k in names:
                n, ops, digest = sass[k]
                r = res.get(k, {})
                regs, stack, shared, local = r.get('REG', 0), r.get('STACK', 0), r.get('SHARED', 0), r.get('LOCAL', 0)
                forms = {o: c for o, c in ops.items() if o.split('.')[0] in LOADS}
                ldg = {o: c for o, c in forms.items() if o.startswith('LDG')}
                ld_glob = sum(c for o, c in forms.items() if o.split('.')[0] in ('LDG', 'LD'))
                b = blocks_per_sm(regs, shared, thr)
                row = dict(arm=arm, file=f, family=fam, kernel=anon(k), name=anon(dem.get(k, k)),
                           regs=regs, stack=stack, local=local, shared=shared,
                           spill_st=spill.get(k, (0, 0, 0))[1], spill_ld=spill.get(k, (0, 0, 0))[2],
                           stl=ops.get('STL', 0), ldl=ops.get('LDL', 0), insns=n,
                           ldg=sum(ldg.values()), ldg_ci=sum(c for o, c in ldg.items() if '.CI' in o),
                           ldg_128=sum(c for o, c in ldg.items() if o.endswith('.128')),
                           lds=sum(c for o, c in forms.items() if o.startswith('LDS')),
                           ld_generic=sum(c for o, c in forms.items() if o.split('.')[0] == 'LD'),
                           half_math=sum(c for o, c in ops.items() if o.split('.')[0] in HALF_OPS) - ops.get('HADD2_CVT', 0),
                           hadd2_cvt=ops.get('HADD2_CVT', 0),
                           f2f_up=sum(c for o, c in ops.items() if o.startswith('F2F.F32.F16')),
                           f2f_down=sum(c for o, c in ops.items() if o.startswith('F2F.F16.F32')),
                           blocks_sm=b, occupancy=round(100 * b * (-(-thr // 32)) / SM_WARPS, 1),
                           forms=';'.join(f'{o}:{c}' for o, c in sorted(forms.items())), digest=digest)
                rows.append(row)
                if arm == 'base':
                    base_digest[(f, row['kernel'])] = digest
                    base_row[(f, row['kernel'])] = row
    for r in rows:
        b = base_row.get((r['file'], r['kernel']))
        r['same_as_base'] = int(b is not None and b['digest'] == r['digest'])
        r['d_regs'] = r['regs'] - b['regs'] if b else ''
        r['d_insns'] = r['insns'] - b['insns'] if b else ''
        r['d_blocks'] = r['blocks_sm'] - b['blocks_sm'] if b else ''
        r['forms_changed'] = int(b is not None and b['forms'] != r['forms'])
    cols = ['arm', 'file', 'family', 'kernel', 'name', 'regs', 'stack', 'local', 'shared', 'spill_st', 'spill_ld', 'stl',
            'ldl', 'insns', 'ldg', 'ldg_ci', 'ldg_128', 'lds', 'ld_generic', 'half_math', 'hadd2_cvt', 'f2f_up', 'f2f_down', 'blocks_sm',
            'occupancy', 'forms', 'same_as_base', 'd_regs', 'd_insns', 'd_blocks', 'forms_changed']
    with gzip.open(os.path.join(out, 'kernels.tsv.gz'), 'wt') as fh:
        w = csv.DictWriter(fh, cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # per family and arm
    agg = defaultdict(list)
    for r in rows:
        agg[(r['family'], r['arm'])].append(r)
    with open(os.path.join(out, 'summary_family_arm.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['family', 'arm', 'kernels', 'regs_mean', 'regs_max', 'kernels_regs_ge_168', 'kernels_with_stl_ldl',
                    'stl_total', 'ldl_total', 'insns_total', 'insns_vs_base_pct', 'blocks_sm_mean', 'occupancy_mean',
                    'kernels_changed', 'kernels_forms_changed', 'ldg_total', 'ldg_ci_total', 'ldg_128_total'])
        for fam in dict.fromkeys(v[0] for v in files.values()):
            base_ins = sum(r['insns'] for r in agg.get((fam, 'base'), []))
            for arm in arms:
                rs = agg.get((fam, arm))
                if not rs:
                    continue
                ins = sum(r['insns'] for r in rs)
                w.writerow([fam, arm, len(rs), round(sum(r['regs'] for r in rs) / len(rs), 1), max(r['regs'] for r in rs),
                            sum(r['regs'] >= 168 for r in rs), sum(r['stl'] + r['ldl'] > 0 for r in rs),
                            sum(r['stl'] for r in rs), sum(r['ldl'] for r in rs), ins,
                            round(100 * (ins - base_ins) / base_ins, 2) if base_ins else '',
                            round(sum(r['blocks_sm'] for r in rs) / len(rs), 2),
                            round(sum(r['occupancy'] for r in rs) / len(rs), 1),
                            sum(not r['same_as_base'] for r in rs), sum(r['forms_changed'] for r in rs),
                            sum(r['ldg'] for r in rs), sum(r['ldg_ci'] for r in rs), sum(r['ldg_128'] for r in rs)])

    # load forms per family and arm
    with open(os.path.join(out, 'forms_family_arm.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['family', 'arm', 'form', 'count'])
        for (fam, arm), rs in sorted(agg.items()):
            tot = Counter()
            for r in rs:
                for it in r['forms'].split(';'):
                    if it:
                        o, c = it.rsplit(':', 1)
                        tot[o] += int(c)
            for o, c in sorted(tot.items()):
                w.writerow([fam, arm, o, c])

    # what an arm changed
    with open(os.path.join(out, 'changes_vs_base.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['arm', 'file', 'kernel', 'name', 'regs_base', 'regs', 'stack_base', 'stack', 'stl_ldl_base', 'stl_ldl',
                    'insns_base', 'insns', 'blocks_base', 'blocks', 'forms_base', 'forms'])
        for r in rows:
            b = base_row.get((r['file'], r['kernel']))
            if r['arm'] == 'base' or not b or r['same_as_base']:
                continue
            if r['regs'] != b['regs'] or r['stack'] != b['stack'] or r['forms'] != b['forms'] \
                    or r['stl'] + r['ldl'] != b['stl'] + b['ldl'] or r['blocks_sm'] != b['blocks_sm']:
                w.writerow([r['arm'], r['file'], r['kernel'], r['name'], b['regs'], r['regs'], b['stack'], r['stack'],
                            b['stl'] + b['ldl'], r['stl'] + r['ldl'], b['insns'], r['insns'], b['blocks_sm'],
                            r['blocks_sm'], b['forms'], r['forms']])

    # the base arm against the objects of the build itself
    with open(os.path.join(out, 'baseline_check.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['file', 'kernels_base_cubin', 'kernels_build', 'identical', 'different', 'only_in_one'])
        for f in files:
            bp = os.path.join(work, 'sass', 'build', f + '.sass.gz')
            cp = os.path.join(work, 'sass', 'base', f + '.sass.gz')
            if not (os.path.exists(bp) and os.path.exists(cp)):
                continue
            a, b = parse_sass(cp), parse_sass(bp)
            same = sum(1 for k in a if k in b and a[k][2] == b[k][2])
            diff = sum(1 for k in a if k in b and a[k][2] != b[k][2])
            w.writerow([f, len(a), len(b), same, diff, len(set(a) ^ set(b))])

    # FP16 arithmetic and conversions per file and arm
    with open(os.path.join(out, 'half_math.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['arm', 'file', 'kernels', 'kernels_with_half_math', 'half_math_insns', 'f2f_f16_to_f32_insns', 'hadd2_cvt', 'f2f_f32_to_f16_insns'])
        per = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        for r in rows:
            p = per[(r['arm'], r['file'])]
            p[0] += 1
            p[1] += r['half_math'] > 0
            p[2] += r['half_math']
            p[3] += r['f2f_up']
            p[4] += r['hadd2_cvt']
            p[5] += r['f2f_down']
        for (arm, f), p in sorted(per.items()):
            w.writerow([arm, f, p[0], p[1], p[2], p[3], p[4], p[5]])

    with open(os.path.join(out, 'compile_times.tsv'), 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['arm', 'file', 'rc', 'seconds'])
        for (a, f), (rc, s) in sorted(times.items()):
            w.writerow([a, f, rc, s.strip()])
    print(f'{len(rows)} kernel rows, {len(arms)} arms')


main()
