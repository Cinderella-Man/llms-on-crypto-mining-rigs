#!/usr/bin/env python3
# gguf_prune_test.py DIR: a self-test of gguf_prune.py (beside it). Builds a three-shard GGUF of a qwen4exp-like model
# with 8 experts, 2 used, in three layers: separate and fused gate/up expert tensors in block types of 32, 64 and 256
# values (Q8_0, Q2_0, IQ3_S, Q6_K: random bytes), F32 and F16 routers, `exp_probs_b`, `.scale` and `.input_scale`
# tensors on the expert axis, tensors whose last dimension is also 8 but that are not experts (an indexer projection,
# the shared expert's gate), odd sizes that need padding, and a third shard with nothing to change; plus the same
# model in one file. A ranking table gives each layer counts with ties at the cut. Checks that
#   (1) keeping all 8 by the ranking writes every shard again byte for byte, and the one-file model too;
#   (2) keeping 5 by the ranking writes the same bytes as this test's own writer given the sliced tensors (each layer's
#       5 largest counts, ties to the lower id, in id order) and expert_count 5, with kept.tsv listing those ids and
#       --link-unchanged making the third shard a link to its source; `verify` passes;
#   (3) an id list in its own order (not ascending) gives the slabs in that order, and `verify` passes;
#   (4) --headers-only writes the same headers and file sizes with no data, `verify --headers-only` passes and a full
#       `verify` of it fails;
#   (5) --force-ids with the ranking keeps each layer's forced ids (one ranked last, one tied, a layer left out) and
#       fills the other places by rank: the bytes this test's writer makes for those ids, and `verify` passes.
# Negative controls, which must fail: `verify` with one byte changed in a kept slab, in a tensor that is not sliced,
# or with two kept ids swapped in kept.tsv; `prune` keeping fewer experts than are used, an id list with a repeated
# id, and one that leaves a layer out; more forced ids than kept, a forced id out of range or repeated, a forced layer
# the model lacks, and --force-ids with --ids. Prints "ok" or fails an assertion. Uses DIR for its files and removes them.
# Standard library only.
import hashlib, os, random, shutil, struct, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

TOOL = os.path.join(here, "gguf_prune.py")
ALIGN = 32
F32, F16, Q8_0, Q6_K, IQ3_S, Q2_0 = 0, 1, 8, 14, 21, 42
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q6_K: (256, 210), IQ3_S: (256, 110), Q2_0: (64, 18)}
N_EXPERT, N_USED, KEEP = 8, 2, 5
rng = random.Random(20261104)


def nbytes(dims, typ):
    n = 1
    for d in dims:
        n *= d
    return n // BLOCK[typ][0] * BLOCK[typ][1]


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_bytes(key, typ, val):
    raw = {2: lambda v: struct.pack("<H", v), 4: lambda v: struct.pack("<I", v), 5: lambda v: struct.pack("<i", v),
           6: lambda v: struct.pack("<f", v), 8: gstr,
           9: lambda v: struct.pack("<IQ", 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)}[typ](val)
    return gstr(key) + struct.pack("<I", typ) + raw


def write_gguf(path, kvs, tensors):
    """kvs: [(key, type, value)]; tensors: [(name, dims, type, data)], laid out in order, each padded to ALIGN."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kv_bytes(*kv) for kv in kvs)
    data, off = b"", 0
    for name, dims, typ, raw in tensors:
        assert len(raw) == nbytes(dims, typ)
        head += gstr(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + \
            struct.pack("<IQ", typ, off)
        pad = raw + b"\0" * ((-len(raw)) % ALIGN)
        data += pad
        off += len(pad)
    head += b"\0" * ((-len(head)) % ALIGN)
    with open(path, "wb") as f:
        f.write(head + data)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def ok(*args):
    r = run(*args)
    assert r.returncode == 0, f"{args}: {r.stderr}"
    return r.stdout


def fails(*args):
    r = run(*args)
    assert r.returncode != 0, f"{args} passed"
    return r.stderr


def model():
    """The shards' tensors: [[(name, dims, type, data)], ...] for three shards; expert tensors on the last axis."""
    t = lambda name, dims, typ: (name, dims, typ, rng.randbytes(nbytes(dims, typ)))
    s1 = [t("token_embd.weight", [16, 5], F32), t("output_norm.weight", [7], F32)]
    s1 += [t("blk.0.attn_norm.weight", [16], F32), t("blk.0.ffn_gate_inp.weight", [16, 8], F32),
           t("blk.0.ffn_gate_exps.weight", [256, 3, 8], IQ3_S), t("blk.0.ffn_up_exps.weight", [32, 3, 8], Q8_0),
           t("blk.0.ffn_down_exps.weight", [256, 2, 8], Q6_K), t("blk.0.ffn_down_exps.scale", [8], F32),
           t("blk.0.exp_probs_b.bias", [8], F32), t("blk.0.indexer.q_proj.weight", [16, 8], F32),
           t("blk.0.ffn_gate_inp_shexp.weight", [16], F32), t("blk.0.ffn_down_shexp.weight", [32, 16], Q8_0)]
    s1 += [t("blk.1.ffn_gate_inp.weight", [16, 8], F16), t("blk.1.ffn_gate_up_exps.weight", [64, 6, 8], Q2_0),
           t("blk.1.ffn_down_exps.weight", [64, 3, 8], Q2_0), t("blk.1.ffn_down_exps.input_scale", [8], F32)]
    s2 = [t("blk.2.ffn_gate_inp.weight", [16, 8], F32), t("blk.2.ffn_gate_exps.weight", [32, 5, 8], Q8_0),
          t("blk.2.ffn_up_exps.weight", [32, 5, 8], Q8_0), t("blk.2.ffn_down_exps.weight", [256, 1, 8], IQ3_S),
          t("blk.2.ffn_gate_exps.input_scale", [8], F32), t("blk.2.attn_q.weight", [16, 9], F16)]
    s3 = [t("per_layer_token_embd.weight", [4, 9], F32)]
    return [s1, s2, s3]


def kvs(keep, shard=None, count=3):
    """Shard 1 (or a one-file model when shard is None) carries the model's keys; the others only the split's."""
    split = [("split.no", 2, shard), ("split.count", 2, count), ("split.tensors.count", 5, 28)] if shard is not None else []
    if shard:
        return split
    return [("general.architecture", 8, "qwen4exp"), ("general.name", 8, "prune test"),
            ("qwen4exp.block_count", 4, 3), ("qwen4exp.expert_count", 4, keep), ("qwen4exp.expert_used_count", 4, N_USED),
            ("qwen4exp.attention.recurrent_layers", 9, [1, 0, 1]), ("qwen4exp.router_logit_softcapping", 6, 30.0)] + split


def sliced(tensors, ids):
    """The tensors with each expert-axis tensor cut to the slabs of `ids` (layer -> ids), in their order."""
    out = []
    for name, dims, typ, raw in tensors:
        parts = name.split(".")
        expert = parts[0] == "blk" and (parts[2].endswith("_exps") or parts[2] in ("ffn_gate_inp", "exp_probs_b"))
        if expert:
            slab = len(raw) // N_EXPERT
            raw = b"".join(raw[e * slab:(e + 1) * slab] for e in ids[int(parts[1])])
            dims = dims[:-1] + [len(ids[int(parts[1])])]
        out.append((name, dims, typ, raw))
    return out


def top(counts, keep):
    """Each layer's kept ids: an expert is kept when fewer than `keep` experts beat it (more routings, or as many
    and a lower id)."""
    return {l: [e for e in range(N_EXPERT)
                if sum(c[f] > c[e] or (c[f] == c[e] and f < e) for f in range(N_EXPERT)) < keep]
            for l, c in counts.items()}


def main():
    d = os.path.abspath(sys.argv[1])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shards = model()
    names = [os.path.join(d, f"m-0000{i + 1}-of-00003.gguf") for i in range(3)]
    for i, (p, ts) in enumerate(zip(names, shards)):
        write_gguf(p, kvs(N_EXPERT, i), ts)
    one = os.path.join(d, "one.gguf")
    write_gguf(one, kvs(N_EXPERT), [t for s in shards for t in s])

    # counts with ties: layer 0 ties at the cut (experts 2, 5, 6 at 40 for the last two places)
    counts = {0: [90, 10, 40, 70, 80, 40, 40, 5], 1: [3, 3, 3, 3, 3, 3, 3, 3], 2: [1, 2, 3, 4, 50, 60, 70, 80]}
    table = os.path.join(d, "experts.tsv")
    with open(table, "w") as f:
        f.write("layer\texpert\troutings\tshare_pct\n")
        for l, c in counts.items():
            f.writelines(f"{l}\t{e}\t{v}\t0\n" for e, v in enumerate(c))
    want = top(counts, KEEP)
    assert want[0] == [0, 2, 3, 4, 5] and want[1] == [0, 1, 2, 3, 4] and want[2] == [3, 4, 5, 6, 7], want

    # (1) all 8 kept: the same bytes
    ok("prune", "--keep", "8", "--experts", table, "--out-dir", f"{d}/all", *names)
    for p in names:
        assert sha(os.path.join(d, "all", os.path.basename(p))) == sha(p), f"{p}: not the same bytes with all kept"
    ok("prune", "--keep", "8", "--experts", table, "--out-dir", f"{d}/all1", one)
    assert sha(f"{d}/all1/one.gguf") == sha(one), "one file: not the same bytes with all kept"

    # (2) 5 kept by the ranking, against this test's own writer
    out = f"{d}/cut"
    ok("prune", "--keep", str(KEEP), "--experts", table, "--out-dir", out, "--prefix", "p", "--link-unchanged", *names)
    news = [os.path.join(out, f"p-0000{i + 1}-of-00003.gguf") for i in range(3)]
    for i in range(2):
        ref = os.path.join(d, f"ref-{i}.gguf")
        write_gguf(ref, kvs(KEEP, i), sliced(shards[i], want))
        assert sha(news[i]) == sha(ref), f"shard {i + 1}: not the expected bytes"
    assert os.path.islink(news[2]) and os.path.samefile(news[2], names[2]), "shard 3 is not a link to its source"
    with open(f"{out}/kept.tsv") as f:
        assert f.read() == "layer\tkept_ids\n" + "".join(f"{l}\t{','.join(map(str, want[l]))}\n" for l in range(3))
    ok("verify", *names, "--new", *news, "--ids", f"{out}/kept.tsv")
    ok("prune", "--keep", str(KEEP), "--experts", table, "--out-dir", f"{d}/cut1", one)
    ref = os.path.join(d, "ref-one.gguf")
    write_gguf(ref, kvs(KEEP), sliced([t for s in shards for t in s], want))
    assert sha(f"{d}/cut1/one.gguf") == sha(ref), "one file: not the expected bytes"

    # (3) an id list in its own order
    order = {0: [7, 3, 0, 5, 1], 1: [4, 3, 2, 1, 0], 2: [6, 0, 2, 4, 1]}
    ids = os.path.join(d, "ids.tsv")
    with open(ids, "w") as f:
        f.writelines(f"{l}\t{','.join(map(str, v))}\n" for l, v in order.items())
    ok("prune", "--ids", ids, "--out-dir", f"{d}/perm", *names)
    perm = [os.path.join(d, "perm", os.path.basename(p)) for p in names]
    for i in range(3):
        ref = os.path.join(d, f"ref-perm-{i}.gguf")
        write_gguf(ref, kvs(KEEP, i), sliced(shards[i], order))
        assert sha(perm[i]) == sha(ref), f"shard {i + 1} of the id list: not the expected bytes"
    ok("verify", *names, "--new", *perm, "--ids", f"{d}/perm/kept.tsv")

    # (4) header copies
    hdr = f"{d}/hdr"
    ok("prune", "--keep", str(KEEP), "--experts", table, "--out-dir", hdr, "--prefix", "p", "--headers-only", *names)
    hnews = [os.path.join(hdr, os.path.basename(p)) for p in news]
    for full, h in zip(news, hnews):
        assert os.path.getsize(full) == os.path.getsize(h), f"{h}: size"
        n = g.Header(full).data0
        with open(full, "rb") as a, open(h, "rb") as b:
            fa, fb = a.read(), b.read()
        assert fa[:n] == fb[:n] and fb[n:] == bytes(len(fb) - n), f"{h}: not the header and zeros"
    ok("verify", *names, "--new", *hnews, "--ids", f"{hdr}/kept.tsv", "--headers-only")
    fails("verify", *names, "--new", *hnews, "--ids", f"{hdr}/kept.tsv")

    # (5) forced ids with the ranking: layer 0 forces its last-ranked expert 7, layer 2 two of its lowest (1, 0, given
    # out of order), layer 1 is left out; the other places go by rank (layer 0: 0, 4, 3, then 2 of the tie at 40)
    force = os.path.join(d, "force.tsv")
    with open(force, "w") as f:
        f.write("layer\tforced_ids\n0\t7\n2\t1,0\n")
    want_f = {0: [0, 2, 3, 4, 7], 1: [0, 1, 2, 3, 4], 2: [0, 1, 5, 6, 7]}
    ok("prune", "--keep", str(KEEP), "--experts", table, "--force-ids", force, "--out-dir", f"{d}/force", *names)
    forced = [os.path.join(d, "force", os.path.basename(p)) for p in names]
    for i in range(3):
        ref = os.path.join(d, f"ref-force-{i}.gguf")
        write_gguf(ref, kvs(KEEP, i), sliced(shards[i], want_f))
        assert sha(forced[i]) == sha(ref), f"shard {i + 1} with forced ids: not the expected bytes"
    with open(f"{d}/force/kept.tsv") as f:
        assert f.read() == "layer\tkept_ids\n" + "".join(f"{l}\t{','.join(map(str, want_f[l]))}\n" for l in range(3))
    ok("verify", *names, "--new", *forced, "--ids", f"{d}/force/kept.tsv")

    # negative controls
    for rows in ("0\t0,1,2,3,4,5\n", "0\t9\n", "0\t7,7\n", "5\t1\n"):
        with open(force, "w") as f:
            f.write(rows)
        fails("prune", "--keep", str(KEEP), "--experts", table, "--force-ids", force, "--out-dir", f"{d}/x", *names)
    fails("prune", "--ids", ids, "--force-ids", force, "--out-dir", f"{d}/x", *names)

    def corrupt(src, tensor, delta):
        """A copy of `src` with one byte of `tensor` changed, `delta` bytes into its data."""
        h = g.Header(src)
        off = next(t[3] for t in h.tensors if t[0] == tensor)
        p = os.path.join(d, "bad-" + os.path.basename(src))
        shutil.copy(src, p)
        with open(p, "r+b") as f:
            f.seek(h.data0 + off + delta)
            b = f.read(1)
            f.seek(h.data0 + off + delta)
            f.write(bytes([b[0] ^ 1]))
        return p

    slab = nbytes([256, 3, 8], IQ3_S) // N_EXPERT
    bad = corrupt(news[0], "blk.0.ffn_gate_exps.weight", 3 * slab + 7)  # the 4th kept slab
    assert "differ" in fails("verify", *names, "--new", bad, *news[1:], "--ids", f"{out}/kept.tsv")
    bad = corrupt(news[0], "blk.0.indexer.q_proj.weight", 5)
    assert "differ" in fails("verify", *names, "--new", bad, *news[1:], "--ids", f"{out}/kept.tsv")
    swapped = os.path.join(d, "swapped.tsv")
    with open(swapped, "w") as f:
        w = dict(want)
        w[1] = [1, 0, 2, 3, 4]
        f.writelines(f"{l}\t{','.join(map(str, v))}\n" for l, v in w.items())
    assert "differ" in fails("verify", *names, "--new", *news, "--ids", swapped)
    fails("prune", "--keep", "1", "--experts", table, "--out-dir", f"{d}/x", *names)
    for rows in ({0: [0, 0, 1, 2, 3], 1: [0, 1, 2, 3, 4], 2: [0, 1, 2, 3, 4]}, {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 4]}):
        with open(ids, "w") as f:
            f.writelines(f"{l}\t{','.join(map(str, v))}\n" for l, v in rows.items())
        fails("prune", "--ids", ids, "--out-dir", f"{d}/x", *names)
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
