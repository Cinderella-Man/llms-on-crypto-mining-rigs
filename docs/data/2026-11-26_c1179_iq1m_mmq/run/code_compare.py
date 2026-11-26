#!/usr/bin/env python3
"""C1179: compare two builds' CUDA code function by function.

    python3 -I code_compare.py BASE_DIR BRANCH_DIR OUT_PREFIX

Each folder holds <object>.sass and <object>.ptx, the raw `cuobjdump -sass` and `cuobjdump -ptx` output of one
ggml-cuda object (code_check.sh writes them). For every function present in both builds (SASS: each "Function : <name>"
per architecture; PTX: each .entry and .func) the result is "same", "const" (the instructions differ only in integer
constants; each (base, branch) pair of constants is counted) or "code" (an instruction added, removed or changed).

Writes OUT_PREFIX.tsv (per object and kind: functions in both, same, const, code, only in base, only in branch),
OUT_PREFIX_pairs.tsv (each constant pair with its count) and OUT_PREFIX_code.tsv (every "code" function and every
function in one build only), and prints the totals.
"""
import collections
import os
import re
import sys

COMMENT = re.compile(r'/\*.*?\*/')
NUM = re.compile(r'\b(0x[0-9a-fA-F]+|[0-9]+)\b')
PTX_START = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+)?\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)')


def sass_functions(text):
    funcs = {}
    arch, name, body = '', None, []
    for line in text.splitlines():
        m = re.match(r'\s*arch = (\S+)', line)
        if m:
            arch = m.group(1)
            continue
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            if name is not None:
                funcs[(arch, name)] = body
            name, body = m.group(1), []
            continue
        if name is None:
            continue
        s = COMMENT.sub('', line).strip()
        if not s or s.startswith('.headerflags') or set(s) <= {'.'}:
            continue
        if s.startswith('Fatbin') or s.startswith('=====') or s.startswith('code for') or '=' in s and s.split('=')[0].strip() in (
                'arch', 'code version', 'host', 'compile_size', 'identifier'):
            continue
        body.append(s)
    if name is not None:
        funcs[(arch, name)] = body
    return funcs


def ptx_functions(text):
    funcs = {}
    name, body = None, []
    for line in text.splitlines():
        s = line.strip()
        m = PTX_START.match(s)
        if m and not s.endswith(';'):
            if name is not None:
                funcs[('ptx', name)] = body
            name, body = m.group(1), []
            continue
        if name is None or not s or s.startswith('//'):
            continue
        body.append(s)
    if name is not None:
        funcs[('ptx', name)] = body
    return funcs


def number(tok):
    return int(tok, 16) if tok.lower().startswith('0x') else int(tok, 10)


def compare(a, b):
    if a == b:
        return 'same', []
    if len(a) != len(b):
        return 'code', []
    pairs = []
    for x, y in zip(a, b):
        if x == y:
            continue
        if NUM.sub('#', x) != NUM.sub('#', y):
            return 'code', []
        for p, q in zip(NUM.findall(x), NUM.findall(y)):
            if p != q:
                pairs.append((number(p), number(q)))
    return 'const', pairs


def main():
    base_dir, branch_dir, prefix = sys.argv[1:4]
    names = sorted({f.rsplit('.', 1)[0] for f in os.listdir(base_dir)} | {f.rsplit('.', 1)[0] for f in os.listdir(branch_dir)})
    rows, code_rows = [], []
    pairs_all = collections.Counter()
    totals = collections.Counter()
    for obj in names:
        for kind, parse in (('sass', sass_functions), ('ptx', ptx_functions)):
            texts = []
            for d in (base_dir, branch_dir):
                p = os.path.join(d, f'{obj}.{kind}')
                texts.append(open(p, errors='replace').read() if os.path.exists(p) else '')
            fa, fb = parse(texts[0]), parse(texts[1])
            if not fa and not fb:
                continue
            c = collections.Counter()
            for key in sorted(set(fa) | set(fb)):
                if key not in fb:
                    c['only_base'] += 1
                    code_rows.append((obj, kind, 'only in base', key[0], key[1]))
                    continue
                if key not in fa:
                    c['only_branch'] += 1
                    code_rows.append((obj, kind, 'only in branch', key[0], key[1]))
                    continue
                c['both'] += 1
                verdict, pairs = compare(fa[key], fb[key])
                c[verdict] += 1
                if verdict == 'code':
                    code_rows.append((obj, kind, 'code', key[0], key[1]))
                for pq in set(pairs):
                    pairs_all[(kind,) + pq] += 1
            rows.append((obj, kind, c['both'], c['same'], c['const'], c['code'], c['only_base'], c['only_branch']))
            totals.update({f'{kind}_{k}': v for k, v in c.items()})
    with open(prefix + '.tsv', 'w') as f:
        f.write('object\tkind\tin_both\tsame\tconst_only\tcode\tonly_base\tonly_branch\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    with open(prefix + '_pairs.tsv', 'w') as f:
        f.write('kind\tbase_constant\tbranch_constant\tfunctions\n')
        for (kind, p, q), n in sorted(pairs_all.items()):
            f.write(f'{kind}\t{p}\t{q}\t{n}\n')
    with open(prefix + '_code.tsv', 'w') as f:
        f.write('object\tkind\twhat\tarch\tfunction\n')
        for r in code_rows:
            f.write('\t'.join(r) + '\n')
    for k in sorted(totals):
        print(f'{k}\t{totals[k]}')
    print('constant pairs (kind, base, branch: functions):',
          ', '.join(f'{k} {p}->{q}: {n}' for (k, p, q), n in sorted(pairs_all.items())) or 'none')


if __name__ == '__main__':
    main()
