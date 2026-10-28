#!/usr/bin/env python3
"""MD5 of named tensors' data in a GGUF file, to check whether two files hold byte-identical tensors.

  python3 -I run/tensor_md5.py FILE.gguf TENSOR_NAME ...

Prints one line per tensor: name, type, MD5 of its data (block bytes only, no alignment padding). Types with a known
block layout: F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K. Standard library only.
"""
import hashlib
import struct
import sys

# type id: (name, weights per block, bytes per block)
TYPES = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 30: ('BF16', 1, 2), 2: ('Q4_0', 32, 18), 6: ('Q5_0', 32, 22),
         8: ('Q8_0', 32, 34), 12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210)}
SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


def main():
    f = open(sys.argv[1], 'rb')
    want = sys.argv[2:]

    def rd(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr():
        return f.read(rd('<Q')).decode('utf-8', 'replace')

    def skip(t):
        if t in SCALAR:
            f.read(struct.calcsize(SCALAR[t]))
        elif t == 8:
            rstr()
        elif t == 9:
            et, n = rd('<I'), rd('<Q')
            for _ in range(n):
                skip(et)
        else:
            raise ValueError(f'type {t}')

    assert f.read(4) == b'GGUF'
    _, nt, nkv = rd('<I'), rd('<Q'), rd('<Q')
    align = 32
    for _ in range(nkv):
        k, t = rstr(), rd('<I')
        if k == 'general.alignment':
            align = rd(SCALAR[t])
        else:
            skip(t)
    infos = {}
    for _ in range(nt):
        name = rstr()
        dims = [rd('<Q') for _ in range(rd('<I'))]
        infos[name] = (dims, rd('<I'), rd('<Q'))
    data0 = (f.tell() + align - 1) // align * align
    for name in want:
        if name not in infos:
            print(f'{name}\tmissing')
            continue
        dims, t, off = infos[name]
        tname, per, nbytes = TYPES.get(t, (str(t), 0, 0))
        n = 1
        for d in dims:
            n *= d
        if not per:
            print(f'{name}\t{tname}\tunknown block size')
            continue
        f.seek(data0 + off)
        h, left = hashlib.md5(), n // per * nbytes
        while left:
            chunk = f.read(min(left, 1 << 24))
            h.update(chunk)
            left -= len(chunk)
        print(f'{name}\t{tname}\t{h.hexdigest()}')


if __name__ == '__main__':
    main()
