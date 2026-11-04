#!/usr/bin/env python3
# Q671: the nvcc flags a build gives its CUDA files, taken from the build's compile_commands.json, as one argument per
# line, ready to be read by compile_arms.sh. Drops the parts that differ per file or only matter for the host compiler
# and the fatbin (`-c <file>`, `-o <object>`, `-Xcompiler ...`, `-compress-mode=...`, `-x cu`), turns the build's
# `--generate-code=arch=compute_61,code=[compute_61,sm_61]` into `code=sm_61` (one cubin, no PTX) and makes the
# `--options-file` path absolute. Checks that every file listed has the same flags.
# Usage: extract_flags.py <compile_commands.json> <flags.txt> <file name>...   (names as in ggml/src/ggml-cuda/)
import json
import os
import shlex
import sys


def flags_of(entry):
    args = shlex.split(entry['command'])
    out = []
    i = 1  # args[0] is nvcc
    while i < len(args):
        a = args[i]
        if a in ('-c', '-o', '-x', '-Xcompiler'):
            i += 2
            continue
        if a.startswith('-compress-mode='):
            i += 1
            continue
        if a == '--options-file':
            out += [a, os.path.join(entry['directory'], args[i + 1])]
            i += 2
            continue
        if a.startswith('--generate-code=') and 'compute_61' in a:
            out.append('--generate-code=arch=compute_61,code=sm_61')
            i += 1
            continue
        if a.endswith('.cu') and os.path.sep in a:
            i += 1  # the source file itself
            continue
        out.append(a)
        i += 1
    return out


def main():
    cc, dest, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    entries = {os.path.basename(e['file']): e for e in json.load(open(cc))
               if os.path.dirname(e['file']).endswith('ggml-cuda')}
    ref = None
    for n in names:
        if n not in entries:
            sys.exit(f'{n}: not in {cc}')
        f = flags_of(entries[n])
        if ref is None:
            ref = f
        elif f != ref:
            sys.exit(f'{n}: flags differ from {names[0]}\n  {set(f) ^ set(ref)}')
    with open(dest, 'w') as fh:
        fh.write('\n'.join(ref) + '\n')
    print(f'{len(ref)} arguments, identical for {len(names)} files')


main()
