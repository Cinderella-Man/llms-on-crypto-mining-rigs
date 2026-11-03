#!/usr/bin/env python3
# Q672, the sm_61 FP16-conversion trap: how each CUDA kernel of a build converts FP16 to FP32 in its sm_61 SASS.
# On compute 6.1 (GP10x consumer Pascal, the P106) FP16 arithmetic runs at a small fraction of the FP32 rate, so a
# conversion that ptxas emits as half arithmetic (`HADD2.F32 Rd, -RZ, Rs.H0_H0`, an add of -0 that writes FP32) costs far
# more than the conversion instruction `F2F.F32.F16`. Reads the gzipped `cuobjdump -sass -arch sm_61` of a binary
# (sass_identity.sh writes one per build) and counts per kernel:
#   hadd2_cvt   HADD2.F32 with an -RZ operand: a conversion done as half arithmetic
#   hadd2_cvt_after_ld  those within 8 instructions after a load (LDG, LDS, LDL, LD, LDC): the load-and-convert
#               pattern of the forum report (a crude proxy: the converted register is not traced)
#   half_math   other HADD2, HMUL2, HFMA2, HSETP2 (real FP16 arithmetic)
#   f2f_f16     F2F.F32.F16 (conversions by the conversion unit)
# Usage: hadd2_audit.py <tag>.sass.gz <out.tsv> [<name regex> ...]
# Writes one row per kernel with any of these (or every kernel matching a regex), sorted by hadd2_cvt; prints totals per
# family (the regexes, default: the BF16/F16 matrix-vector kernels, the Pascal quantized kernels, the k-column and
# multi-token kernels, the persistent MoE kernel, the rest).
import gzip
import re
import sys
from collections import defaultdict

FUNC = re.compile(r'Function : (\S+)')
INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;')
ANON = re.compile(r'\d+_GLOBAL__N__[0-9a-f]+_\d+_(\w+?)_[0-9a-f]{8}(?=\d|E)')
FAMILIES = [
    ('BF16/F16 matrix-vector (mmvf, Pascal BF16)', r'mul_mat_vec_f|mmvf'),
    ('Pascal quantized (mmvq-pascal, 1-4 tokens)', r'mul_mat_vec_q_pascal'),
    ('k-column (mmvq-pascal-kcol)', r'kcol'),
    ('multi-token experts (mmvq-pascal-mtok)', r'mtok'),
    ('persistent MoE (router included)', r'moe_persistent|moep_'),
    ('stock mul_mat_vec_q', r'mul_mat_vec_q(?!_pascal)'),
]


def main():
    path, out = sys.argv[1], sys.argv[2]
    fams = [(r, r) for r in sys.argv[3:]] or FAMILIES
    rows = {}
    name = None
    recent = []  # instruction indexes of the last loads

    def counts(n):
        return rows.setdefault(n, defaultdict(int))

    i = 0
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            m = FUNC.search(line)
            if m:
                name = ANON.sub(lambda a: f'<anon:{a.group(1)}>', m.group(1))
                recent = []
                i = 0
                continue
            if name is None:
                continue
            m = INSN.match(line)
            if not m:
                continue
            text = m.group(1)
            op = text.split()[0] if not text.startswith('@') else text.split()[1]
            i += 1
            c = counts(name)
            if op.startswith(('LDG', 'LDS', 'LDL', 'LDC', 'LD.')):
                recent.append(i)
                recent = recent[-4:]
            if op.startswith('HADD2.F32') and '-RZ' in text:
                c['hadd2_cvt'] += 1
                if any(i - j <= 8 for j in recent):
                    c['hadd2_cvt_after_ld'] += 1
            elif op.startswith(('HADD2', 'HMUL2', 'HFMA2', 'HSETP2')):
                c['half_math'] += 1
            elif op.startswith('F2F.F32.F16'):
                c['f2f_f16'] += 1
            c['insns'] += 1

    keys = ['hadd2_cvt', 'hadd2_cvt_after_ld', 'half_math', 'f2f_f16', 'insns']
    fam_tot = defaultdict(lambda: defaultdict(int))
    fam_n = defaultdict(int)

    def family(n):
        for label, rx in fams:
            if re.search(rx, n):
                return label
        return 'other kernels'

    with open(out, 'w') as o:
        o.write('family\tkernel\t' + '\t'.join(keys) + '\n')
        for n, c in sorted(rows.items(), key=lambda kv: (-kv[1]['hadd2_cvt'], kv[0])):
            fam = family(n)
            fam_n[fam] += 1
            for k in keys:
                fam_tot[fam][k] += c[k]
            if c['hadd2_cvt'] or c['half_math'] or c['f2f_f16']:
                o.write(f'{fam}\t{n}\t' + '\t'.join(str(c[k]) for k in keys) + '\n')
    print('family\tkernels\t' + '\t'.join(keys))
    for fam in [label for label, _ in fams] + ['other kernels']:
        t = fam_tot[fam]
        print(f'{fam}\t{fam_n[fam]}\t' + '\t'.join(str(t[k]) for k in keys))


if __name__ == '__main__':
    main()
