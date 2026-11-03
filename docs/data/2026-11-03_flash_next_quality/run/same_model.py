#!/usr/bin/env python3
"""Is a GGUF the same model as a folder of BF16 safetensors? For each pair of names it converts the BF16 tensor (or one
expert's slice of a fused [experts, rows, columns] tensor) the way the GGUF stores it and compares the bytes with the
GGUF's tensor: Q8_0 bit for bit as ggml's quantize_row_q8_0_ref (gguf_retype.py's q8_0, which is what llama-quantize
and the converter's q8_0 both write), F32 by widening, F16 by rounding to nearest even, BF16 as is. Equal bytes on a
few tensors from different layers and kinds (a hyper-connection, an expert, the router) mean the file was made from
these weights; a different fine-tune or revision differs in every block.
  python3 -I same_model.py <safetensors folder> <GGUF shard>... --pair ST_NAME=GGUF_NAME[@EXPERT] [--pair ...]
Prints one line per pair and exits with status 1 if any pair differs. Standard library only; needs gguf_reshard.py
and gguf_retype.py from the repository's scripts/ (found next to this file or in ../../../../scripts).
"""
import argparse, json, math, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header  # noqa: E402
from gguf_retype import BF16, F16, F32, NAMES, Q8_0, f16, floats, nbytes, q8_0  # noqa: E402


def st_tensor(folder, name):
    """(file, byte offset of the data, dtype, shape) of a tensor in a sharded safetensors folder."""
    index = json.load(open(os.path.join(folder, "model.safetensors.index.json")))["weight_map"]
    path = os.path.join(folder, index[name])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    return path, 8 + n + meta["data_offsets"][0], meta["dtype"], meta["shape"]


def pread(path, n, start):
    with open(path, "rb") as f:
        raw = os.pread(f.fileno(), n, start)
    assert len(raw) == n, f"{path}: short read at {start}"
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--pair", action="append", required=True)
    a = ap.parse_args()
    where = {}
    for p in a.shards:
        h = Header(p)
        for name, dims, typ, off in h.tensors:
            where[name] = (p, h.data0 + off, dims, typ)
    bad = 0
    for pair in a.pair:
        st, gg = pair.split("=")
        gg, _, expert = gg.partition("@")
        path, start, dtype, shape = st_tensor(a.folder, st)
        assert dtype == "BF16", f"{st}: {dtype}, not BF16"
        gpath, gstart, dims, typ = where[gg]
        assert typ in (Q8_0, F16, F32, BF16), f"{gg}: type {typ} cannot be compared"
        if expert:
            e = int(expert)
            n_exp, rows, cols = shape
            assert list(dims) == [cols, rows, n_exp], f"{gg}: dims {dims} against {st}: {shape}"
            n = rows * cols
            start += e * n * 2
            gstart += e * nbytes(n, typ)
        else:
            assert list(dims) == shape[::-1], f"{gg}: dims {dims} against {st}: {shape}"
            n = math.prod(shape)
        raw = pread(path, n * 2, start)
        x = floats(raw, BF16)
        want = q8_0(x) if typ == Q8_0 else f16(x) if typ == F16 else bytes(x) if typ == F32 else raw
        got = pread(gpath, len(want), gstart)
        same = got == want
        bad += not same
        where_diff = ""
        if not same:
            step = 34 if typ == Q8_0 else 64
            first = next(i for i in range(0, len(want), step) if got[i:i + step] != want[i:i + step])
            diff = sum(got[i:i + step] != want[i:i + step] for i in range(0, len(want), step))
            where_diff = f", first difference in byte {first:,}, {diff:,} of {len(want) // step:,} blocks differ"
        print(f"{'same' if same else 'DIFFERENT'}  {st} -> {gg}{'@' + expert if expert else ''}  "
              f"{NAMES[typ]}, {n:,} values{where_diff}")
        sys.stdout.flush()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
