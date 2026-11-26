#!/usr/bin/env python3
"""C1190: one program's card code in two builds, function by function (standard library only).

    python3 -I card_compare.py A.sass B.sass A.ptx B.ptx OUT_PREFIX

The inputs are `cuobjdump -sass <program>` and `cuobjdump -ptx <program>` of the same program from two builds (A: the
Docker build, B: the native one). A linked program holds one fatbin per CUDA object and names no source file in it,
so functions are matched by architecture, name and copy number (the n-th function of that name in the listing; both
builds link their objects in the same order), as C1118's sass_compare.py did. Each function in both builds is:

  identical  the same listing, the instruction words and Pascal's scheduling control words included (SASS), or the
             same text (PTX);
  same       the same instructions once the encodings are left out (SASS only: the control words differ);
  const      the instructions differ only in integer constants (each pair of constants is counted);
  code       an instruction added, removed or changed.

The last three follow C1179's code_compare.py. nvcc names anonymous namespaces and internal functions after a hash
(_GLOBAL__N__<8 hex>_<n>_<file>_<8 hex>, _INTERNAL_<8 hex>_<n>_<file>_<8 hex>); both hashes are replaced by xxxxxxxx in
names and bodies, and the number of names whose hash differs between the builds is reported, since a hash that depends
on the build folder would otherwise make every such function look new.

Writes OUT_PREFIX.tsv (per kind and verdict: functions), OUT_PREFIX_families.tsv (per kind, family and verdict, for
every verdict but identical), OUT_PREFIX_pairs.tsv (constant pairs and their count) and OUT_PREFIX_code.tsv (every
"code" function and every function in one build only), and prints the totals.
"""
import collections
import hashlib
import re
import sys

COMMENT = re.compile(r'/\*.*?\*/')
NUM = re.compile(r'\b(0x[0-9a-fA-F]+|[0-9]+)\b')
PTX_START = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+)?\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)')
HASH = re.compile(r'(\d*)(_GLOBAL__N__|_INTERNAL_)([0-9a-f]{8})_(\d+)_')


def unhash(text, seen):
    """replace both hashes of each hashed namespace name by xxxxxxxx; seen collects the original names.

    In a mangled name the identifier's length comes first, and a static local's identifier can end in _<number>,
    which differs from compile to compile (_INTERNAL_<hash>_<n>_<file>_<hash>_5339 in one build, ..._161637 in the
    other); that number becomes N and the length is written anew."""
    out, pos = [], 0
    for m in HASH.finditer(text):
        # a nested name follows the one before it at once, and the digits in front of it may start inside that one's
        # hash, so only the digits after the last rewrite count
        if m.start(2) < pos:
            continue
        n = int(m.group(4))
        tail = m.end() + n
        if text[tail:tail + 1] != '_' or not re.fullmatch(r'[0-9a-f]{8}', text[tail + 1:tail + 9]):
            continue
        end, suffix = tail + 9, ''
        digits_start, ident_start = max(m.start(1), pos), m.start(2)
        digits = text[digits_start:ident_start]
        for k in range(len(digits)):  # the length prefix: the longest digit run that ends the identifier sensibly
            length = int(digits[k:])
            stop = ident_start + length
            if stop == end or (stop > end and re.fullmatch(r'_\d+', text[end:stop])):
                suffix = '_N' if stop > end else ''
                end = stop
                digits = digits[:k] + '{len}'
                break
        seen.add(text[ident_start:end])
        ident = f'{m.group(2)}xxxxxxxx_{m.group(4)}_{text[m.end():tail]}_xxxxxxxx{suffix}'
        out.append(text[pos:digits_start])
        out.append(digits.replace('{len}', str(len(ident))) + ident)
        pos = end
    out.append(text[pos:])
    return ''.join(out)


def sass_functions(path, seen):
    funcs = collections.OrderedDict()
    copies = collections.Counter()
    arch, key, raw, body = '', None, [], []

    def flush():
        if key is not None:
            funcs[key] = (hashlib.md5('\n'.join(raw).encode()).digest(), '\n'.join(body))

    with open(path, errors='replace') as f:
        for line in f:
            line = unhash(line.rstrip('\n'), seen)
            if line.startswith('Fatbin '):
                flush()
                key, raw, body = None, [], []
                continue
            m = re.match(r'\s*arch = (\S+)', line)
            if m:
                arch = m.group(1)
                continue
            m = re.match(r'\s*Function : (\S+)', line)
            if m:
                flush()
                copies[(arch, m.group(1))] += 1
                key, raw, body = (arch, m.group(1), copies[(arch, m.group(1))]), [], []
                continue
            if key is None:
                continue
            raw.append(line.strip())
            s = COMMENT.sub('', line).strip()
            if not s or s.startswith('.headerflags') or set(s) <= {'.'}:
                continue
            body.append(s)
    flush()
    return funcs


def ptx_functions(path, seen):
    funcs = collections.OrderedDict()
    copies = collections.Counter()
    key, body = None, []

    def flush():
        if key is not None:
            text = '\n'.join(body)
            funcs[key] = (hashlib.md5(text.encode()).digest(), text)

    with open(path, errors='replace') as f:
        for line in f:
            s = unhash(line.strip(), seen)
            if s.startswith('Fatbin '):
                flush()
                key, body = None, []
                continue
            m = PTX_START.match(s)
            if m and not s.endswith(';'):
                flush()
                copies[m.group(1)] += 1
                key, body = ('ptx', m.group(1), copies[m.group(1)]), []
                continue
            if key is None or not s or s.startswith('//'):
                continue
            body.append(s)
    flush()
    return funcs


def number(tok):
    return int(tok, 16) if tok.lower().startswith('0x') else int(tok, 10)


def compare(a, b):
    if a[0] == b[0]:
        return 'identical', []
    if a[1] == b[1]:
        return 'same', []
    la, lb = a[1].split('\n'), b[1].split('\n')
    if len(la) != len(lb):
        return 'code', []
    pairs = []
    for x, y in zip(la, lb):
        if x == y:
            continue
        if NUM.sub('#', x) != NUM.sub('#', y):
            return 'code', []
        for p, q in zip(NUM.findall(x), NUM.findall(y)):
            if p != q:
                pairs.append((number(p), number(q)))
    return 'const', pairs


def family(name):
    """the kernel's name without its template arguments and parameter types: the mangled name's first identifier"""
    if not name.startswith('_Z'):
        return name
    i, parts = 2, []
    nested = name[i:i + 1] == 'N'
    if nested:
        i += 1
    while i < len(name):
        if name[i] == 'L':
            i += 1
            continue
        m = re.match(r'(\d+)', name[i:])
        if not m:
            break
        n = int(m.group(1))
        i += len(m.group(1))
        parts.append(name[i:i + n])
        i += n
        if not nested:
            break
    return '::'.join(parts)[:100] or name


def main():
    a_sass, b_sass, a_ptx, b_ptx, prefix = sys.argv[1:6]
    totals = collections.Counter()
    fam = collections.Counter()
    pairs_all = collections.Counter()
    code_rows = []
    hashed = {}
    for kind, parse, pa, pb in (('sass', sass_functions, a_sass, b_sass), ('ptx', ptx_functions, a_ptx, b_ptx)):
        seen_a, seen_b = set(), set()
        fa, fb = parse(pa, seen_a), parse(pb, seen_b)
        hashed[kind] = (len(seen_a), len(seen_b), len(seen_a & seen_b))
        for key in list(fa) + [k for k in fb if k not in fa]:
            if key not in fb:
                v = 'only in A'
            elif key not in fa:
                v = 'only in B'
            else:
                v, pairs = compare(fa[key], fb[key])
                for pq in set(pairs):
                    pairs_all[(kind,) + pq] += 1
            totals[(kind, v)] += 1
            if v != 'identical':
                fam[(kind, family(key[1]), v)] += 1
            if v in ('code', 'only in A', 'only in B'):
                code_rows.append((kind, v, key[0], key[1], str(key[2])))
    verdicts = ('identical', 'same', 'const', 'code', 'only in A', 'only in B')
    with open(prefix + '.tsv', 'w') as f:
        f.write('kind\t' + '\t'.join(verdicts) + '\thashed names A\thashed names B\thashed names in both\n')
        for kind in ('sass', 'ptx'):
            f.write(kind + '\t' + '\t'.join(str(totals[(kind, v)]) for v in verdicts) + '\t'
                    + '\t'.join(map(str, hashed[kind])) + '\n')
    with open(prefix + '_families.tsv', 'w') as f:
        f.write('kind\tfamily\tverdict\tfunctions\n')
        for (kind, fm, v), n in sorted(fam.items()):
            f.write(f'{kind}\t{fm}\t{v}\t{n}\n')
    with open(prefix + '_pairs.tsv', 'w') as f:
        f.write('kind\tA constant\tB constant\tfunctions\n')
        for (kind, p, q), n in sorted(pairs_all.items()):
            f.write(f'{kind}\t{p}\t{q}\t{n}\n')
    with open(prefix + '_code.tsv', 'w') as f:
        f.write('kind\twhat\tarch\tfunction\tcopy\n')
        for r in code_rows:
            f.write('\t'.join(r) + '\n')
    for kind in ('sass', 'ptx'):
        print(kind, ', '.join(f'{v} {totals[(kind, v)]}' for v in verdicts),
              f'(hashed names A {hashed[kind][0]}, B {hashed[kind][1]}, in both {hashed[kind][2]})')
    print('constant pairs:', ', '.join(f'{k} {p}->{q}: {n}' for (k, p, q), n in sorted(pairs_all.items())) or 'none')


if __name__ == '__main__':
    main()
