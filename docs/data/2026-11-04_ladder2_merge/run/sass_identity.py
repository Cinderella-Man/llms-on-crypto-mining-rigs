#!/usr/bin/env python3
# Per-kernel SASS identity of a merged build against the builds of the branches it merges (sass_identity.sh writes the
# inputs). For every CUDA kernel of the merged build's test-backend-ops it reports which of the other builds compiled
# exactly the same sm_61 machine code, so that a merge which changes a kernel's code shows up without a GPU.
#
# Usage: sass_identity.py <dir> <merged tag> <base tag> <branch tag>[:<tag of the tree it was cut from>] ...
#   <dir>/<tag>.sass.gz   cuobjdump -sass -arch sm_61 of the build's test-backend-ops
#   <dir>/<tag>.res.gz    cuobjdump -res-usage -arch sm_61 of the merged binary
# The base is the tree the merge starts from; a branch cut from an older tree names that tree's build after a colon.
# Output: <dir>/kernels.tsv (one row per kernel of the merged build: the tags with identical code, the "loose", "same
# mix" and "same ops" matches, registers/shared/stack/local bytes), <dir>/carried.tsv (every kernel a branch changed or
# added against the tree it was cut from: carried into the merged build unchanged, changed on both sides, or not), and
# a summary on stdout.
#
# Matching: kernels by mangled name, with an anonymous namespace's per-build name (_GLOBAL__N__<hash>_<n>_<file>_<hash>)
# replaced by its file's. A kernel compiled in several translation units has several bodies; their sorted list is
# compared at four levels:
#   strict    the instructions and their encodings, offsets stripped
#   loose     also without the encodings and the constant-bank operands (c[bank][offset]: kernel parameters and
#             globals), which move when a kernel's parameter struct gains a field, not the arithmetic
#   same mix  the loose instructions sorted, with registers and predicates renamed: the same instructions in another
#             order and register allocation, as when ptxas schedules a changed prologue
#   same ops  the same multiset of opcodes (with their modifiers; NOP and DEPBAR left out): every arithmetic, load,
#             store and shuffle instruction the same in kind and number, operands (address arithmetic, register
#             reuse flags, branch targets) and order free
# The Pascal matrix-vector kernels (mul_mat_vec_q_pascal, _mc) gained a last template parameter `tile` (Q2_0's tile
# staging); a merged kernel name with tile = false also matches the name without it in a build that predates it.
import gzip
import hashlib
import re
import sys
from collections import defaultdict

INSN = re.compile(r'^\s*/\*[0-9a-f]{4,}\*/\s*(.*)$')
CTRL = re.compile(r'^\s*(/\* 0x[0-9a-f]{16} \*/)\s*$')
ENC = re.compile(r'/\* 0x[0-9a-f]{16} \*/')
CBANK = re.compile(r'c\[0x[0-9a-f]+\] ?\[0x[0-9a-f]+\]')
REGS = re.compile(r'\b(R\d+|P\d|UR\d+)\b')
SCHED = ('NOP', 'DEPBAR')  # scheduling only: left out of the opcode multiset
FUNC = re.compile(r'Function : (\S+)')
RES_FUNC = re.compile(r'^\s*Function (\S+):\s*$')
RES_FIELD = re.compile(r'(REG|STACK|SHARED|LOCAL):(\d+)')
ANON = re.compile(r'\d+_GLOBAL__N__[0-9a-f]+_\d+_(\w+?)_[0-9a-f]{8}(?=\d|E)')
TILE = re.compile(r'^(.*mul_mat_vec_q_pascal(?:_mc)?I.*?)Lb0EEEv(.*)$')


def norm(name):
    return ANON.sub(lambda m: f'<anon:{m.group(1)}>', name)


def load_sass(path):
    strict = defaultdict(list)
    loose = defaultdict(list)
    mix = defaultdict(list)
    ops = defaultdict(list)
    name = None
    s_lines = []
    l_lines = []

    def flush():
        if name is not None and s_lines:
            strict[name].append(hashlib.sha1('\n'.join(s_lines).encode()).hexdigest())
            loose[name].append(hashlib.sha1('\n'.join(l_lines).encode()).hexdigest())
            mix[name].append(hashlib.sha1('\n'.join(sorted(REGS.sub('R', x) for x in l_lines)).encode()).hexdigest())
            opc = sorted(o for o in (x.split()[1] if x.startswith('@') else x.split()[0] for x in l_lines if x)
                         if not o.startswith(SCHED))
            ops[name].append(hashlib.sha1('\n'.join(opc).encode()).hexdigest())

    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            m = FUNC.search(line)
            if m:
                flush()
                name = norm(m.group(1))
                s_lines, l_lines = [], []
                continue
            if name is None:
                continue
            m = INSN.match(line)
            if m:
                text = m.group(1).strip()
                s_lines.append(text)
                l_lines.append(CBANK.sub('c[]', ENC.sub('', text)).strip())
                continue
            if CTRL.match(line):
                s_lines.append(line.strip())
                continue
            if line.strip().startswith('.....') or 'code for sm_' in line:
                flush()
                name = None
                s_lines, l_lines = [], []
    flush()
    return tuple({k: tuple(sorted(v)) for k, v in t.items()} for t in (strict, loose, mix, ops))


def load_res(path):
    res = {}
    name = None
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            m = RES_FUNC.match(line)
            if m:
                name = norm(m.group(1))
                continue
            if name is not None:
                fields = dict(RES_FIELD.findall(line))
                if fields:
                    old = res.get(name)
                    new = tuple(int(fields.get(k, 0)) for k in ('REG', 'SHARED', 'STACK', 'LOCAL'))
                    res[name] = new if old is None else tuple(max(a, b) for a, b in zip(old, new))
                name = None
    return res


def lookup(table, name):
    if name in table:
        return table[name]
    m = TILE.match(name)
    if m:
        return table.get(m.group(1) + 'EEv' + m.group(2))
    return None


def main():
    d, merged, base = sys.argv[1], sys.argv[2], sys.argv[3]
    # <branch>[:<the tree it was cut from>], when that is not the base
    own = {a.split(':')[0]: (a.split(':') + [base])[1] for a in sys.argv[4:]}
    branches = list(own)
    refs = [base] + branches
    sass = {t: load_sass(f'{d}/{t}.sass.gz') for t in set([merged] + refs + list(own.values()))}
    res = load_res(f'{d}/{merged}.res.gz')
    ms, ml, mm, mo = sass[merged]

    counts = defaultdict(int)
    flagged = []
    with open(f'{d}/kernels.tsv', 'w') as out:
        out.write('kernel\tbodies\tidentical_in\tloose_only_in\tsame_mix_in\tsame_ops_in\tclass\tregisters\tshared_bytes\tstack_bytes\tlocal_bytes\n')
        for name in sorted(ms):
            same = [t for t in refs if lookup(sass[t][0], name) == ms[name]]
            loose = [t for t in refs if t not in same and lookup(sass[t][1], name) == ml[name]]
            remix = [t for t in refs if t not in same + loose and lookup(sass[t][2], name) == mm[name]]
            reops = [t for t in refs if t not in same + loose + remix and lookup(sass[t][3], name) == mo[name]]
            present = [t for t in refs if lookup(sass[t][0], name) is not None]
            if base in same:
                cls = 'as base'
            elif same:
                cls = 'as ' + '+'.join(same)
            elif not present:
                cls = 'NEW in merge'
            elif loose:
                cls = 'LOOSE ' + '+'.join(loose)
            elif remix:
                cls = 'SAME MIX ' + '+'.join(remix)
            elif reops:
                cls = 'SAME OPS ' + '+'.join(reops)
            else:
                cls = 'DIFFERS'
            counts[cls if not cls.startswith('as ') or cls == 'as base' else 'as a branch: ' + cls[3:]] += 1
            if cls.startswith(('NEW', 'LOOSE', 'SAME MIX', 'SAME OPS', 'DIFFERS')):
                flagged.append((cls, name))
            r = res.get(name, ('', '', '', ''))
            out.write(f'{name}\t{len(ms[name])}\t{",".join(same)}\t{",".join(loose)}\t{",".join(remix)}\t{",".join(reops)}\t{cls}\t' + '\t'.join(map(str, r)) + '\n')

    # every kernel a branch changed or added against the base must be in the merged build unchanged
    bs = sass[base][0]
    untiled = {}  # a merged Pascal kernel's name without the tile parameter -> its merged name
    for mn in ms:
        m = TILE.match(mn)
        if m:
            untiled[m.group(1) + 'EEv' + m.group(2)] = mn
    carried_bad = 0
    with open(f'{d}/carried.tsv', 'w') as out:
        out.write('branch\tkernel\tin_merged\n')
        for t in branches:
            ts = sass[t][0]
            os_ = sass[own[t]][0]  # the tree the branch was cut from
            n_ok = n_bad = n_both = 0
            for name, h in sorted(ts.items()):
                if os_.get(name) == h:
                    continue  # not changed by this branch
                mn = name if name in ms else untiled.get(name)
                mh = ms.get(mn) if mn else None
                if mh == h:
                    status = 'identical'
                    n_ok += 1
                elif mn is not None and os_.get(name) is not None and bs.get(mn) is not None and \
                        bs.get(mn) != lookup(os_, mn):
                    # changed by this branch and by the base's own commits since the branch's tree: neither version
                    # can be carried unchanged; read the two changes
                    status = 'CHANGED ON BOTH SIDES'
                    n_both += 1
                else:
                    status = 'MISSING or CHANGED'
                    n_bad += 1
                out.write(f'{t}\t{name}\t{status}\n')
            print(f'{t} (cut from {own[t]}): kernels it changed or added: {n_ok} carried identically, {n_bad} not, '
                  f'{n_both} changed on both sides')
            carried_bad += n_bad

    print(f'merged build: {len(ms)} kernels')
    for cls, n in sorted(counts.items()):
        print(f'  {cls}: {n}')
    for cls, name in flagged[:40]:
        print(f'  flagged ({cls}): {name}')
    if len(flagged) > 40:
        print(f'  ... {len(flagged) - 40} more flagged in kernels.tsv')
    print('verdict:', 'every kernel matches a source build' if not flagged and not carried_bad else
          f'{len(flagged)} kernels flagged, {carried_bad} branch kernels not carried')


if __name__ == '__main__':
    main()
