#!/usr/bin/env python3
# For the kernels sass_identity.py classed DIFFERS: the opcodes whose counts differ between a reference build and the
# merged build, so a reader sees whether the arithmetic, loads, stores and shuffles changed or only the integer address
# arithmetic and scheduling around them did.
# Usage: sass_opdiff.py <dir> <reference tag> <merged tag> > opdiff.tsv   (<dir> as for sass_identity.py, with its
# kernels.tsv; run beside sass_identity.py, which it imports)
import gzip
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_identity as si  # noqa: E402

# opcodes of integer address arithmetic, moves and scheduling: everything else is arithmetic, memory or control flow
ADDR = ('IADD', 'IADD32I', 'IADD3', 'MOV', 'MOV32I', 'XMAD', 'SHL', 'SHR', 'LEA', 'ISCADD', 'LOP', 'LOP32I', 'BFE',
        'NOP', 'DEPBAR')


def opcodes(path, want):
    out = {}
    name = None
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            m = si.FUNC.search(line)
            if m:
                name = si.norm(m.group(1))
                continue
            if name in want:
                m = si.INSN.match(line)
                if m:
                    t = si.ENC.sub('', m.group(1)).strip()
                    o = t.split()[1] if t.startswith('@') else t.split()[0]
                    out.setdefault(name, Counter())[o.rstrip(';')] += 1
    return out


def main():
    d, ref, merged = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(f'{d}/kernels.tsv') as f:
        head = f.readline().rstrip('\n').split('\t')
        ci = head.index('class')
        want = {r[0] for r in (line.rstrip('\n').split('\t') for line in f) if r[ci] == 'DIFFERS'}
    a, b = opcodes(f'{d}/{ref}.sass.gz', want), opcodes(f'{d}/{merged}.sass.gz', want)
    print(f'kernel\tinstructions_{ref}\tinstructions_{merged}\topcode count changes\tonly address arithmetic and scheduling')
    for n in sorted(want):
        ca, cb = a.get(n, Counter()), b.get(n, Counter())
        diff = {k: cb[k] - ca[k] for k in sorted(set(ca) | set(cb)) if cb[k] != ca[k]}
        only = all(k.split('.')[0] in ADDR for k in diff)
        print(f'{n}\t{sum(ca.values())}\t{sum(cb.values())}\t' + ' '.join(f'{k} {v:+d}' for k, v in diff.items()) +
              f'\t{"yes" if only else "NO"}')


if __name__ == '__main__':
    main()
