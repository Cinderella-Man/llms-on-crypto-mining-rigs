#!/usr/bin/env python3
"""Turns an existing quant's per-tensor format choices into llama-quantize options using other formats.

Community quants such as Unsloth's "UD" files choose a format per tensor and per layer, which is where much of their
quality comes from, but they use K-quant and I-quant formats that Pascal cards with few SMs decode slowly. This script
reads such a file's header (pure Python, no numpy) and prints `--tensor-type` options that keep the same per-tensor
allocation with formats that decode quickly on these cards.

    scripts/mirror_recipe.py <existing quant.gguf> [MAP]
    MAP: comma-separated source=target pairs; default Q5_K=q5_0,Q6_K=q8_0,IQ4_XS=q4_1,Q4_K=q4_1,IQ3_S=q4_1,Q8_0=q8_0
    EXTRA="$(scripts/mirror_recipe.py flint-UD-Q4_K_XL.gguf)" scripts/quantize_docker.sh ... Q4_1 imatrix.gguf

Tensors whose format isn't in MAP (norms, F32/F16) are left to llama-quantize's defaults. One pattern is printed per
(tensor kind, target format), listing the layer numbers it applies to.
"""
import collections
import re
import struct
import sys

TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
         13: 'Q5_K', 14: 'Q6_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
         22: 'IQ2_S', 23: 'IQ4_XS', 29: 'IQ1_M', 30: 'BF16', 39: 'MXFP4'}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
DEFAULT_MAP = 'Q5_K=q5_0,Q6_K=q8_0,IQ4_XS=q4_1,Q4_K=q4_1,IQ3_S=q4_1,Q8_0=q8_0'


def read_tensor_types(path):
    f = open(path, 'rb')

    def rd(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

    def string():
        return f.read(rd('Q')).decode('utf-8', 'replace')

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = rd('I'), rd('Q')
            if et in (8, 9):
                return [value(et) for _ in range(n)]
            return f.read(struct.calcsize('<' + SCALAR[et]) * n)
        return rd(SCALAR[t])

    if f.read(4) != b'GGUF':
        sys.exit(f'{path}: not a GGUF file')
    rd('I')
    n_tensors, n_kv = rd('Q'), rd('Q')
    for _ in range(n_kv):
        string()
        value(rd('I'))
    result = []
    for _ in range(n_tensors):
        name = string()
        n_dims = rd('I')
        for _ in range(n_dims):
            rd('Q')
        t = rd('I')
        rd('Q')
        result.append((name, TYPES.get(t, f'type{t}')))
    return result


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mapping = dict(pair.split('=') for pair in (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MAP).split(','))
    groups = collections.defaultdict(list)   # (kind, target) -> layer numbers
    singles = []
    for name, src in read_tensor_types(sys.argv[1]):
        if src not in mapping:
            continue
        m = re.match(r'blk\.(\d+)\.(.+)$', name)
        if m:
            groups[(m.group(2), mapping[src])].append(int(m.group(1)))
        else:
            singles.append((name, mapping[src]))
    opts = []
    for (kind, target), layers in sorted(groups.items()):
        alternation = '|'.join(str(n) for n in sorted(layers))
        opts.append(f'--tensor-type blk\\.({alternation})\\.{re.escape(kind)}={target}')
    for name, target in singles:
        if name == 'output.weight':
            opts.append(f'--output-tensor-type {target}')
        elif name == 'token_embd.weight':
            opts.append(f'--token-embedding-type {target}')
        else:
            opts.append(f'--tensor-type {re.escape(name)}={target}')
    print(' '.join(opts))


if __name__ == '__main__':
    main()
