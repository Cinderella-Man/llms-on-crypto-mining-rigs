#!/usr/bin/env python3
"""Whether several Flash-Next files hold the same n-gram table (per_layer_token_embd), so that one copy of it on a
rig's SSD serves all of them. For each NAME=SHARD[,SHARD...] it finds the shard that holds the tensor and prints the
shard, the tensor's type code, shape, bytes, its data offset, the md5 of its data, the shard's header size, the md5 of
the shard's bytes before the tensor's data when the shard holds only it (its header), and whether it does.
  python3 -I table_identity.py [--tensor per_layer_token_embd.weight] NAME=SHARD[,SHARD...] ... > table_md5.tsv
Reads the headers, then the tensor's bytes once (28.8 GB for Flash-Next's table). Standard library only.
"""
import argparse, hashlib, os, struct

# ggml type -> (block weights, block bytes) for the types a table is stored in; others hash up to the next tensor
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 20: (32, 18), 23: (256, 136)}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def header(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    rstr = lambda: f.read(rd("<Q")).decode("utf-8", "replace")

    def rval(t):
        if t in SCALAR:
            return rd(SCALAR[t])
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [rval(et) for _ in range(n)]
        raise ValueError(f"type {t}")

    assert f.read(4) == b"GGUF"
    _ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k, t = rstr(), rd("<I")
        kv[k] = rval(t)
    infos = []
    for _ in range(nt):
        name = rstr()
        dims = [rd("<Q") for _ in range(rd("<I"))]
        infos.append((name, rd("<I"), rd("<Q"), dims))
    align = kv.get("general.alignment", 32)
    data0 = (f.tell() + align - 1) // align * align
    f.close()
    return kv, infos, data0


def md5_range(path, start, n, chunk=64 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        f.seek(start)
        while n > 0:
            b = f.read(min(chunk, n))
            if not b:
                raise IOError(f"{path}: short read")
            h.update(b)
            n -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tensor", default="per_layer_token_embd.weight")
    a = ap.parse_args()
    print("file\tshard\ttype\tshape\tbytes\tdata_offset\tdata_md5\theader_bytes\tbefore_md5\talone")
    for spec in a.files:
        name, shards = spec.split("=", 1)
        for path in shards.split(","):
            kv, infos, data0 = header(path)
            hit = [x for x in infos if x[0] == a.tensor]
            if not hit:
                continue
            _, typ, off, dims = hit[0]
            ends = sorted(x[2] for x in infos if x[2] > off)
            end = (data0 + ends[0]) if ends else os.path.getsize(path)
            n = end - (data0 + off)
            if typ in TYPE_SIZE:
                blk, size = TYPE_SIZE[typ]
                count = 1
                for d in dims:
                    count *= d
                n = count // blk * size
            print(f"{name}\t{os.path.basename(path)}\t{typ}\t{'x'.join(map(str, dims))}\t{n}\t{data0 + off}"
                  f"\t{md5_range(path, data0 + off, n)}\t{data0}\t{md5_range(path, 0, data0 + off) if len(infos) == 1 else "-"}\t"
                  f"{'yes' if len(infos) == 1 else f'no ({len(infos) - 1} others)'}", flush=True)


if __name__ == "__main__":
    main()
