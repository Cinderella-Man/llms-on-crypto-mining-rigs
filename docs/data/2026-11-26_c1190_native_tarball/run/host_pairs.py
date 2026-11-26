#!/usr/bin/env python3
"""C1190: do the host symbols found in one build only pair up once nvcc's path hashes are taken out?

    python3 -I host_pairs.py OUT.tsv host_<program>.tsv [...]

Each host_<program>.tsv is compare.sh's list of the defined symbols whose size differs or that only one build has
(symbol, kind, sizes in A, sizes in B). Names of anonymous namespaces and internal functions carry a hash of the
source's path (card_compare.py's unhash renames it), so the same function gets another name in a build made in another
folder. For each program: the symbols only in A and only in B, how many carry such a hash, and whether, renamed, the
two lists are the same names with the same sizes.
"""
import collections
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location('cc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'card_compare.py'))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def main():
    out = sys.argv[1]
    with open(out, 'w') as f:
        f.write('program\tsize differs\tonly in A\tonly in B\twith a path hash A\twith a path hash B\t'
                'pair up renamed (same names, kinds and sizes)\n')
        for path in sys.argv[2:]:
            prog = os.path.basename(path)[len('host_'):-len('.tsv')]
            a, b = collections.Counter(), collections.Counter()
            differs = hashed_a = hashed_b = 0
            for line in open(path, errors='replace').read().splitlines()[1:]:
                name, kind, sa, sb = line.split('\t')
                seen = set()
                renamed = cc.unhash(name, seen)
                if sa != '-' and sb == '-':
                    a[(renamed, kind, sa)] += 1
                    hashed_a += bool(seen)
                elif sb != '-' and sa == '-':
                    b[(renamed, kind, sb)] += 1
                    hashed_b += bool(seen)
                else:
                    differs += 1
            f.write(f'{prog}\t{differs}\t{sum(a.values())}\t{sum(b.values())}\t{hashed_a}\t{hashed_b}\t'
                    f'{"yes" if a == b else "no"}\n')
    print(open(out).read(), end='')


if __name__ == '__main__':
    main()
