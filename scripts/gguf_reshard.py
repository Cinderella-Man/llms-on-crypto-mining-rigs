#!/usr/bin/env python3
# gguf_reshard.py: moves the tensors that match a pattern out of a split GGUF into one more shard of their own, so
# that one big tensor (Flash-Next's 28.8 GB n-gram table, `per_layer_token_embd`) can sit on a faster disk than the
# rest. Tensor bytes are copied unchanged; only the headers change (split.count, the tensor lists, the offsets).
#   gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir <dir> [--isolated-dir <fast disk>] SHARD...
#     writes <prefix>-0000k-of-0000N.gguf, N = old count + 1: the old shards keep their numbers (minus the moved
#     tensors) and the new last shard holds the moved ones. With --isolated-dir it is written there and symlinked into
#     --out-dir, beside the others, where llama.cpp looks for it.
#     --headers-only writes only the headers, each file sized as if the data were there (a check of the layout with
#     scripts/fit_probe.sh on header-only copies).
#   gguf_reshard.py verify OLD_SHARD... --new NEW_SHARD...
#     rebuilds each old shard's bytes from its own header and the new shards' tensor data, and prints the SHA-256 of
#     each: equal to the old file's hash, it proves the data is the same byte for byte. It reads the new shards whole.
# Every tensor keeps its slot (its bytes plus the padding up to the next tensor), so offsets stay aligned. Standard
# library only.
import argparse, hashlib, os, re, struct, sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
CHUNK = 64 << 20


class Header:
    """A GGUF header: the raw bytes of every key/value pair, the tensor infos and where the data starts."""

    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        rstr = lambda: f.read(rd("<Q"))

        def skip(t):
            if t in SCALAR:
                f.read(struct.calcsize(SCALAR[t]))
            elif t == 8:
                rstr()
            elif t == 9:
                et, n = rd("<I"), rd("<Q")
                for _ in range(n):
                    skip(et)
            else:
                raise ValueError(f"{path}: value type {t}")

        assert f.read(4) == b"GGUF", f"{path}: not a GGUF"
        self.version, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
        self.kv = []  # (key, type, raw value bytes)
        for _ in range(nkv):
            key, t = rstr().decode(), rd("<I")
            start = f.tell()
            skip(t)
            end = f.tell()
            f.seek(start)
            self.kv.append((key, t, f.read(end - start)))
        self.tensors = []  # [name, dims, type, offset]
        for _ in range(nt):
            name = rstr().decode()
            dims = [rd("<Q") for _ in range(rd("<I"))]
            self.tensors.append([name, dims, rd("<I"), rd("<Q")])
        self.align = self.value("general.alignment", 32)
        self.data0 = (f.tell() + self.align - 1) // self.align * self.align
        self.size = os.path.getsize(path)
        order = sorted(self.tensors, key=lambda x: x[3])
        ends = [x[3] for x in order[1:]] + [self.size - self.data0]
        self.slot = {x[0]: e - x[3] for x, e in zip(order, ends)}  # bytes from a tensor's offset to the next one's

    def value(self, key, default=None):
        for k, t, raw in self.kv:
            if k == key:
                return struct.unpack(SCALAR[t], raw)[0] if t in SCALAR else raw
        return default


def write_header(out, version, kv, tensors, align):
    """Writes the header for `tensors` laid out in order, each in its slot padded to the alignment."""
    buf = bytearray()
    wstr = lambda s: buf.extend(struct.pack("<Q", len(s)) + s)
    buf += b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kv))
    for key, t, raw in kv:
        wstr(key.encode())
        buf += struct.pack("<I", t) + raw
    off = 0
    for name, dims, typ, _old, slot in tensors:
        wstr(name.encode())
        buf += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
        off += slot + (-slot) % align
    buf += b"\0" * ((-len(buf)) % align)
    write_all(out.fileno(), bytes(buf))


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def copy_range(src, dst, start, n, align):
    """Copies n bytes at `start` of src to dst's position, then zeros up to the alignment."""
    pad = (-n) % align
    while n > 0:
        try:
            done = os.copy_file_range(src.fileno(), dst.fileno(), min(n, CHUNK), start)
        except (AttributeError, OSError):  # older Python, or across file systems (EXDEV): plain reads
            done = 0
        if not done:
            data = os.pread(src.fileno(), min(n, CHUNK), start)
            assert data, f"short read at {start}"
            write_all(dst.fileno(), data)
            done = len(data)
        start += done
        n -= done
    write_all(dst.fileno(), b"\0" * pad)


def split_name(path, no, count):
    return re.sub(r"-\d{5}-of-\d{5}\.gguf$", f"-{no:05d}-of-{count:05d}.gguf", os.path.basename(path))


def cmd_split(a):
    olds = [Header(p) for p in a.shards]
    count = olds[0].value("split.count")
    assert count, "not a split GGUF (no split.count): split it first with llama-gguf-split"
    assert count == len(olds), f"{len(olds)} shards given, split.count is {count}"
    pat, new_count = re.compile(a.isolate), count + 1
    align = olds[0].align

    def write_shard(path, version, kv, tensors):  # tensors: (source header, [name, dims, type, offset, slot])
        with open(path, "wb") as dst:
            write_header(dst, version, kv, [t for _h, t in tensors], align)
            if a.headers_only:  # the right size, no data: for checking that llama.cpp reads the layout
                os.ftruncate(dst.fileno(), os.lseek(dst.fileno(), 0, 1) + sum(s + (-s) % align for *_x, s in
                                                                               (t for _h, t in tensors)))
                return
            for h, (_n, _d, _t, off, slot) in tensors:
                with open(h.path, "rb") as src:
                    copy_range(src, dst, h.data0 + off, slot, align)

    moved = []
    os.makedirs(a.out_dir, exist_ok=True)
    for i, h in enumerate(olds):
        assert h.align == align, "shards with different alignments"
        keep = [(h, t + [h.slot[t[0]]]) for t in sorted(h.tensors, key=lambda x: x[3]) if not pat.search(t[0])]
        moved += [(h, t + [h.slot[t[0]]]) for t in sorted(h.tensors, key=lambda x: x[3]) if pat.search(t[0])]
        kv = [(k, t, struct.pack("<H", new_count) if k == "split.count" else raw) for k, t, raw in h.kv]
        path = os.path.join(a.out_dir, split_name(h.path, i + 1, new_count))
        write_shard(path, h.version, kv, keep)
        print(f"{path}: {len(keep)} tensors, {os.path.getsize(path):,} bytes")
    assert moved, f"no tensor matches {a.isolate}"
    kv = [("split.no", 2, struct.pack("<H", count)), ("split.count", 2, struct.pack("<H", new_count)),
          ("split.tensors.count", 5, struct.pack("<i", olds[0].value("split.tensors.count")))]
    name = split_name(olds[0].path, new_count, new_count)
    path = os.path.join(a.isolated_dir or a.out_dir, name)
    write_shard(path, olds[0].version, kv, moved)
    print(f"{path}: {len(moved)} tensors ({', '.join(t[0] for _h, t in moved)}), {os.path.getsize(path):,} bytes")
    if a.isolated_dir:
        link = os.path.join(a.out_dir, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(path), link)
        print(f"{link} -> {os.path.abspath(path)}")


def cmd_verify(a):
    where = {}  # tensor name -> (file, offset in it)
    for p in a.new:
        h = Header(p)
        for name, _d, _t, off in h.tensors:
            where[name] = (p, h.data0 + off)
    files = {}
    for p in a.old:
        h = Header(p)
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            sha.update(f.read(h.data0))
        for name, _d, _t, off in sorted(h.tensors, key=lambda x: x[3]):
            src, pos = where[name]
            f = files.setdefault(src, open(src, "rb"))
            f.seek(pos)
            left = h.slot[name]
            while left:
                buf = f.read(min(left, CHUNK))
                assert buf, f"{src}: short read for {name}"
                sha.update(buf)
                left -= len(buf)
        print(f"{sha.hexdigest()}  {p} (rebuilt from the new shards)")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--isolate", required=True, help="regex of the tensor names that get their own shard")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--isolated-dir", help="write the new shard here and symlink it into --out-dir")
    s.add_argument("--headers-only", action="store_true", help="write the headers and size the files, copy no data")
    s.add_argument("shards", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("old", nargs="+")
    v.add_argument("--new", nargs="+", required=True)
    a = ap.parse_args()
    cmd_split(a) if a.cmd == "split" else cmd_verify(a)


if __name__ == "__main__":
    main()
