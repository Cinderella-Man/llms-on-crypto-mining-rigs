#!/usr/bin/env python3
"""Zero named tensors of a GGUF file in place (their data bytes; for Q8_0, Q4_0, Q5_0 and K-quants that is every weight
0): e.g. an MTP block's routed down projection, so the block's routed experts add nothing and its shared expert is all
that is left. Work on a copy.

  python3 -I tensor_zero.py FILE.gguf TENSOR_NAME ...

Standard library only.
"""
import struct
import sys

SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


def main():
    path, want = sys.argv[1], set(sys.argv[2:])
    with open(path, 'r+b') as f:
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
        infos = []
        for _ in range(nt):
            name = rstr()
            dims = [rd('<Q') for _ in range(rd('<I'))]
            infos.append((name, rd('<I'), rd('<Q')))
        data0 = (f.tell() + align - 1) // align * align
        f.seek(0, 2)
        end = f.tell() - data0
        offs = sorted(o for _, _, o in infos)
        for name, t, off in infos:
            if name not in want:
                continue
            nxt = next((o for o in offs if o > off), end)
            f.seek(data0 + off)
            left = nxt - off
            while left:
                n = min(left, 1 << 24)
                f.write(b'\0' * n)
                left -= n
            print(f'{name}: {nxt - off} bytes zeroed')
            want.discard(name)
    if want:
        print('not found: ' + ' '.join(sorted(want)))
        sys.exit(1)


if __name__ == '__main__':
    main()
