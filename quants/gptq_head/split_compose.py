#!/usr/bin/env python3
# Variants of a GGUF that differ only in a few tensors, as two-file split GGUFs sharing one large first file:
# llama.cpp loads <name>-00001-of-00002.gguf and finds <name>-00002-of-00002.gguf next to it.
#
#   split_compose.py base SRC.gguf OUT-base.gguf REGEX            SRC without the tensors REGEX matches, as split 1 of 2
#   split_compose.py shard BASE.gguf SRC.gguf DIR NAME REGEX [DONOR.gguf]
#       DIR/NAME-00001-of-00002.gguf = link to BASE; DIR/NAME-00002-of-00002.gguf = every tensor REGEX matches, from
#       DONOR when it has the tensor, else from SRC
#   split_compose.py verify SRC.gguf DIR/NAME-00001-of-00002.gguf [DONOR.gguf]
#       the two files together hold every tensor of SRC exactly once, each byte for byte SRC's or the donor's
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import os, re, sys
import numpy as np
import gguf


def writer(r, path):
    w = gguf.GGUFWriter(path, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    return w


def split_kv(w, no, n_tensors):
    w.add_uint16('split.no', no)
    w.add_uint16('split.count', 2)
    w.add_int32('split.tensors.count', n_tensors)


def base(src, out, regex):
    r = gguf.GGUFReader(src)
    pat = re.compile(regex)
    keep = [t for t in r.tensors if not pat.search(t.name)]
    w = writer(r, out)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.') or f.name.startswith('split.'):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    split_kv(w, 0, len(r.tensors))
    for t in keep:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in keep:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f'{out}: {len(keep)} of {len(r.tensors)} tensors ({len(r.tensors) - len(keep)} left for the second files)')


def shard(base_path, src, d, name, regex, donor=None):
    r = gguf.GGUFReader(src)
    pat = re.compile(regex)
    dt = {t.name: t for t in gguf.GGUFReader(donor).tensors} if donor else {}
    take = [dt.get(t.name, t) for t in r.tensors if pat.search(t.name)]
    os.makedirs(d, exist_ok=True)
    p1 = os.path.join(d, f'{name}-00001-of-00002.gguf')
    p2 = os.path.join(d, f'{name}-00002-of-00002.gguf')
    if os.path.lexists(p1):
        os.remove(p1)
    os.symlink(os.path.relpath(base_path, d), p1)
    w = writer(r, p2)
    split_kv(w, 1, len(r.tensors))
    for t in take:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in take:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f'{p2}: {len(take)} tensors, {sum(t.name in dt for t in take)} from the donor')


def verify(src, first, donor=None):
    second = first.replace('-00001-of-00002', '-00002-of-00002')
    have = {}
    for p in (first, second):
        for t in gguf.GGUFReader(p).tensors:
            if t.name in have:
                sys.exit(f'{t.name} is in both files')
            have[t.name] = t
    dt = {t.name: t for t in gguf.GGUFReader(donor).tensors} if donor else {}
    rs = gguf.GGUFReader(src).tensors
    if len(have) != len(rs):
        sys.exit(f'{len(have)} tensors in the split, {len(rs)} in {src}')
    n_don = 0
    for t in rs:
        ref = dt.get(t.name, t)
        o = have[t.name]
        if o.tensor_type != ref.tensor_type or not np.array_equal(np.asarray(o.data).reshape(-1), np.asarray(ref.data).reshape(-1)):
            sys.exit(f'{t.name} is neither the source\'s nor the donor\'s')
        n_don += t.name in dt
    print(f'{first}: all {len(rs)} tensors present once, {n_don} from the donor, the rest the source\'s, byte for byte')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    {'base': base, 'shard': shard, 'verify': verify}[cmd](*args)
