#!/usr/bin/env python3
# gguf_reshard_test.py DIR: a self-test of gguf_reshard.py (beside it). Writes a two-shard GGUF with random tensors of
# odd sizes (so slots end unaligned at the end of a file), moves one tensor from the middle of the first shard into a
# third shard in another folder, and checks that (1) `verify` rebuilds both old shards to their SHA-256, (2) every
# tensor's bytes are the same in the new shards, (3) the new headers carry split.count 3, the right split.no and all
# tensors. Prints "ok" or fails an assertion. Uses DIR for its files and removes them. Standard library only.
import hashlib, os, random, shutil, struct, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

ALIGN = 32


def kv_u16(key, v):
    return (key, 2, struct.pack("<H", v))


def write_gguf(path, kv, tensors):  # tensors: (name, bytes)
    with open(path, "wb") as f:
        buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for key, t, raw in kv:
            buf += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", t) + raw
        off = 0
        for name, data in tensors:
            buf += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", 1) + struct.pack("<Q", len(data))
            buf += struct.pack("<IQ", 24, off)  # type I8: one byte per element
            off += len(data) + (-len(data)) % ALIGN
        buf += b"\0" * ((-len(buf)) % ALIGN)
        for i, (_name, data) in enumerate(tensors):
            buf += data + (b"\0" * ((-len(data)) % ALIGN) if i < len(tensors) - 1 else b"")
        f.write(buf)


def main():
    d = os.path.abspath(sys.argv[1])
    old, new, fast = (os.path.join(d, x) for x in ("old", "new", "fast"))
    for x in (old, new, fast):
        os.makedirs(x, exist_ok=True)
    rnd = random.Random(213)
    t1 = [(f"blk.{i}.w.weight", rnd.randbytes(rnd.randint(1, 5000))) for i in range(5)]
    t1.insert(2, ("per_layer_token_embd.weight", rnd.randbytes(70001)))
    t2 = [(f"blk.{i}.w.weight", rnd.randbytes(rnd.randint(1, 5000))) for i in range(5, 9)]
    total = len(t1) + len(t2)
    meta = [("general.architecture", 8, struct.pack("<Q", 4) + b"test"), kv_u16("split.no", 0), kv_u16("split.count", 2),
            ("split.tensors.count", 5, struct.pack("<i", total))]
    p1, p2 = os.path.join(old, "m-00001-of-00002.gguf"), os.path.join(old, "m-00002-of-00002.gguf")
    write_gguf(p1, meta, t1)
    write_gguf(p2, [kv_u16("split.no", 1), kv_u16("split.count", 2), meta[3]], t2)
    sha = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (p1, p2)}

    tool = os.path.join(here, "gguf_reshard.py")
    subprocess.run([sys.executable, "-I", tool, "split", "--isolate", r"^per_layer_token_embd\.", "--out-dir", new,
                    "--isolated-dir", fast, p1, p2], check=True, stdout=subprocess.DEVNULL)
    names = [os.path.join(new, f"m-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    assert os.path.islink(names[2]) and os.path.realpath(names[2]).startswith(fast)
    out = subprocess.run([sys.executable, "-I", tool, "verify", p1, p2, "--new", *names], check=True,
                         capture_output=True, text=True).stdout.split("\n")
    got = {line.split()[1]: line.split()[0] for line in out if line}
    assert got == sha, f"rebuilt hashes differ: {got} against {sha}"

    want = dict(t1 + t2)
    seen = 0
    for i, p in enumerate(names):
        h = g.Header(p)
        assert h.value("split.count") == 3 and h.value("split.no") == i and h.value("split.tensors.count") == total
        assert h.data0 % ALIGN == 0
        raw = open(p, "rb").read()
        for name, _dims, _typ, off in h.tensors:
            assert off % ALIGN == 0
            data = want[name]
            assert raw[h.data0 + off:h.data0 + off + len(data)] == data, name
            seen += 1
    assert g.Header(names[0]).value("general.architecture") is not None
    assert [t[0] for t in g.Header(names[2]).tensors] == ["per_layer_token_embd.weight"]
    assert seen == total
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
