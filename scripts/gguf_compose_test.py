#!/usr/bin/env python3
# gguf_compose_test.py DIR: a self-test of gguf_compose.py (beside it). Writes a base GGUF in three shards (the last
# holding one table tensor alone, as Flash-Next's files do) and a donor of the same model in two shards of another
# layout, whose tensors have other types and sizes; tensors of odd sizes, so slots end unaligned. Then:
#   (1) layout base, tensors taken by name, glob and regex, --link-unchanged: every taken tensor has the donor's type
#       and bytes, every other the base's, each shard keeps its key/value pairs and offsets are aligned, the shard with
#       nothing taken is a link to the base's, and `verify` passes; without the link that shard has the base's SHA-256;
#   (2) layout donor, (3) one file (no split keys), (4) a re-split by size with the table isolated in a last shard:
#       the same bytes, shard 1 with the base's first key/value pairs, split keys that number the shards, `verify`;
#   (5) --headers-only: the same headers and file sizes as the full files;
#   and negative controls, which must fail: a donor with one shape changed, a donor of another architecture, a
#   pattern that matches nothing, an output path that is an input, and `verify` of an output with one byte changed
#   in a taken tensor and in a base tensor.
# Prints "ok" or fails an assertion. Uses DIR for its files and removes them. Standard library only.
import hashlib, os, random, shutil, struct, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

ALIGN = 32
TOOL = os.path.join(here, "gguf_compose.py")


def arch(name):
    return ("general.architecture", 8, struct.pack("<Q", len(name)) + name.encode())


def split_kv(no, count, total):
    return [("split.no", 2, struct.pack("<H", no)), ("split.count", 2, struct.pack("<H", count)),
            ("split.tensors.count", 5, struct.pack("<i", total))]


def write_gguf(path, kv, tensors, pad_last=False):  # tensors: (name, dims, type, bytes)
    with open(path, "wb") as f:
        buf = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for key, t, raw in kv:
            buf += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", t) + raw
        off = 0
        for name, dims, typ, data in tensors:
            buf += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims))
            buf += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off)
            off += len(data) + (-len(data)) % ALIGN
        buf += b"\0" * ((-len(buf)) % ALIGN)
        f.write(buf)
        for i, (_n, _d, _t, data) in enumerate(tensors):
            f.write(data + (b"\0" * ((-len(data)) % ALIGN) if pad_last or i < len(tensors) - 1 else b""))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(*args, ok=True):
    r = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, f"{args[0]} {'failed' if ok else 'passed'}: {r.stdout}{r.stderr}"
    return r.stdout + r.stderr


def tensors_of(paths):
    """name -> (type, dims, data bytes up to the next tensor) across the shards."""
    out = {}
    for p in paths:
        h = g.Header(p)
        raw = open(p, "rb").read()
        for name, dims, typ, off in h.tensors:
            assert off % ALIGN == 0 and h.data0 % ALIGN == 0, f"{p}: {name} unaligned"
            out[name] = (typ, dims, raw[h.data0 + off:h.data0 + off + h.slot[name]])
    return out


def main():
    d = os.path.abspath(sys.argv[1])
    rnd = random.Random(597)
    names = ["token_embd.weight"] + [f"blk.{i}.{k}.weight" for i in range(4)
                                     for k in ("attn_qkv", "attn_norm", "ffn_up_exps", "ffn_down_shexp", "hc_attn_up")]
    names += ["output.weight", "per_layer_token_embd.weight"]
    dims = {n: [rnd.choice([32, 64, 96]), rnd.randint(1, 9)] for n in names}
    # the base: types 18 (IQ3_XXS) etc. with arbitrary bytes; the donor: type 8 (Q8_0) with other sizes
    base_t = {n: (rnd.choice([12, 18, 23]), rnd.randbytes(rnd.randint(1, 3000))) for n in names}
    donor_t = {n: (8, rnd.randbytes(rnd.randint(1, 3000))) for n in names}
    total = len(names)
    b_dir, d_dir = os.path.join(d, "base"), os.path.join(d, "donor")
    for x in (b_dir, d_dir):
        os.makedirs(x, exist_ok=True)
    row = lambda src, n: (n, dims[n], *src[n])
    b_groups = [names[:12], names[12:-1], names[-1:]]  # the table alone in shard 3
    base = [os.path.join(b_dir, f"b-0000{i + 1}-of-00003.gguf") for i in range(3)]
    for i, grp in enumerate(b_groups):
        kv = ([arch("test"), ("test.block_count", 4, struct.pack("<I", 4))] if i == 0 else []) + split_kv(i, 3, total)
        write_gguf(base[i], kv, [row(base_t, n) for n in grp], pad_last=i == 2)
    d_order = names[-1:] + names[:-1]  # the table first, in a shard of its own
    donor = [os.path.join(d_dir, f"q-0000{i + 1}-of-00002.gguf") for i in range(2)]
    write_gguf(donor[0], [arch("test")] + split_kv(0, 2, total), [row(donor_t, n) for n in d_order[:1]])
    write_gguf(donor[1], split_kv(1, 2, total), [row(donor_t, n) for n in d_order[1:]])

    takes = ["--take", "output.weight", "--take", "glob:blk.*.attn_qkv.weight", "--take", r"re:ffn_down_shexp"]
    take = {n for n in names if n == "output.weight" or "attn_qkv" in n or "ffn_down_shexp" in n}
    io = ["--base", *base, "--donor", *donor]

    def check(paths, layout):
        got = tensors_of(paths)
        assert set(got) == set(names), layout
        for n in names:
            typ, data = (donor_t if n in take else base_t)[n]
            assert got[n][0] == typ and got[n][1] == dims[n], f"{layout}: {n} type or shape"
            assert got[n][2][:len(data)] == data and not got[n][2][len(data):].strip(b"\0"), f"{layout}: {n} bytes"
        out = run("verify", *io, *takes, "--new", *paths)
        assert f"{len(take)} identical to the donor's" in out, out

    # (1) the base's layout, the untouched table shard linked, then copied
    o1 = os.path.join(d, "o1")
    run("compose", *io, *takes, "--out-dir", o1, "--prefix", "c", "--link-unchanged")
    p1 = [os.path.join(o1, f"c-0000{i}-of-00003.gguf") for i in (1, 2, 3)]
    assert os.path.islink(p1[2]) and os.path.samefile(p1[2], base[2])
    for p, b in zip(p1, base):
        assert g.Header(p).kv == g.Header(b).kv, f"{p}: key/value pairs"
        assert [t[0] for t in g.Header(p).tensors] == [t[0] for t in g.Header(b).tensors], f"{p}: tensor order"
    check(p1, "base")
    o1b = os.path.join(d, "o1b")
    run("compose", *io, *takes, "--out-dir", o1b)
    p1b = [os.path.join(o1b, os.path.basename(b)) for b in base]
    assert not os.path.islink(p1b[2]) and sha(p1b[2]) == sha(base[2]), "the shard with nothing taken changed"
    check(p1b, "base, copied")

    # (2) the donor's layout, (3) one file, (4) re-split by size with the table isolated
    for layout, extra, count in (("donor", [], 2), ("one", [], 1), ("0.000012", ["--isolate", "^per_layer_token_embd"], None)):
        o = os.path.join(d, f"o-{layout}")
        run("compose", *io, *takes, "--out-dir", o, "--prefix", "c", "--layout", layout, *extra)
        paths = sorted(os.path.join(o, f) for f in os.listdir(o))
        assert count is None or len(paths) == count, f"{layout}: {len(paths)} shards"
        assert len(paths) > 2 or count, f"{layout}: not re-split"
        hs = [g.Header(p) for p in paths]
        plain = lambda kv: [x for x in kv if not x[0].startswith("split.")]
        assert plain(hs[0].kv) == plain(g.Header(base[0]).kv), f"{layout}: shard 1's key/value pairs"
        for i, h in enumerate(hs):
            want = (i, len(hs), total) if len(hs) > 1 else (None, None, None)
            assert (h.value("split.no"), h.value("split.count"), h.value("split.tensors.count")) == want, layout
            assert i == 0 or not plain(h.kv), f"{layout}: shard {i + 1} has more than the split keys"
        if layout == "donor":
            assert [t[0] for t in hs[0].tensors] == ["per_layer_token_embd.weight"]
        if extra:
            assert [t[0] for t in hs[-1].tensors] == ["per_layer_token_embd.weight"], "the table is not alone last"
        check(paths, layout)
        if layout == "one":
            assert paths == [os.path.join(o, "c.gguf")]

    # (5) headers only: the same headers and sizes
    o5 = os.path.join(d, "o5")
    run("compose", *io, *takes, "--out-dir", o5, "--prefix", "c", "--headers-only")
    for p in p1:
        q = os.path.join(o5, os.path.basename(p))
        assert os.path.getsize(q) == os.path.getsize(os.path.realpath(p)), f"{q}: size"
        h, hq = g.Header(os.path.realpath(p) if os.path.islink(p) else p), g.Header(q)
        assert h.kv == hq.kv and h.tensors == hq.tensors, f"{q}: header"

    # negative controls
    bad_dir = os.path.join(d, "bad")
    os.makedirs(bad_dir, exist_ok=True)
    shape = os.path.join(bad_dir, "shape.gguf")
    t = [row(donor_t, n) for n in names]
    t[3] = (t[3][0], [t[3][1][0], t[3][1][1] + 1], *t[3][2:])
    write_gguf(shape, [arch("test")], t)
    out = run("compose", "--base", *base, "--donor", shape, *takes, "--out-dir", bad_dir, ok=False)
    assert "shapes differ" in out, out
    other = os.path.join(bad_dir, "arch.gguf")
    write_gguf(other, [arch("other")], [row(donor_t, n) for n in names])
    out = run("compose", "--base", *base, "--donor", other, *takes, "--out-dir", bad_dir, ok=False)
    assert "architectures differ" in out, out
    out = run("compose", *io, "--take", "glob:blk.*.nothing", "--out-dir", bad_dir, ok=False)
    assert "matches no tensor" in out, out
    out = run("compose", *io, *takes, "--out-dir", b_dir, ok=False)
    assert "is one of the inputs" in out, out
    for name in ("blk.3.attn_qkv.weight", "blk.2.ffn_up_exps.weight"):  # one from the donor, one from the base
        h = g.Header(p1b[1])
        off = h.data0 + next(t[3] for t in h.tensors if t[0] == name)
        with open(p1b[1], "r+b") as f:
            f.seek(off)
            b = f.read(1)
            f.seek(off)
            f.write(bytes([b[0] ^ 1]))
        out = run("verify", *io, *takes, "--new", *p1b, ok=False)
        assert f"{name}: bytes differ" in out, out
        with open(p1b[1], "r+b") as f:
            f.seek(off)
            f.write(b)
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
