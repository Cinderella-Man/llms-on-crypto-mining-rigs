#!/usr/bin/env python3
# gguf_tier_test.py DIR: a self-test of gguf_tier.py (beside it). Builds a three-shard GGUF of a qwen4exp-like model
# with 8 experts, 2 used: layer 0 in shard 1 with separate gate, up and down expert tensors (IQ3_S, Q8_0, Q6_K: random
# bytes), `.scale` and `.input_scale` on its down experts and an F32 router, layer 1 in shard 2 with a fused gate_up
# (Q2_0) and an F16 router, layer 2 beside it with a `.scale` and a `.bias` on its experts (left untiered), odd sizes
# that need padding, and a third shard with nothing to change; plus a synthetic importance matrix. Checks that
#   (1) split ranks each layer's experts by count, ties to the lower id, and writes hot.tsv and cold.tsv;
#   (2) tier with the cold slabs from the base itself writes what this test's own writer makes (hot slabs in hot order,
#       the cold tensor right after each hot one, the tier map right after the router, split.tensors.count raised in
#       the changed shards), links the untouched third shard, and verify passes;
#   (3) extract writes the cold ids' expert tensors with the first shard's keys (no split keys, expert_count 3) and the
#       importance matrix cut the same way (other entries dropped), both as this test's writer makes them;
#   (4) tier with --cold = extract's file and --cold-in = its ids.tsv writes the same bytes as (2);
#   (5) a cold file in other types (Q4_0 and IQ1_M, random bytes, no scales) and a hot file in its own order (--hot,
#       --hot-in) give the expected bytes (the hot scales from the hot file, no cold ones), and verify passes;
#   (5b) --cold-only 'ffn_(gate|up|gate_up)_exps' takes gate, up and gate_up from the cold file and the down
#       projections, scales included, from the base, or the same bytes from another copy of it with --cold-rest;
#   (1c) two usage tables rank by their summed shares (weights 1,0 give the first table's split; a weight count
#       that does not match the tables is refused);
#   (1b) a --hot spec ("0:5") tiers only its layers and leaves the others' shard a link, and --force-hot puts a
#       last-ranked expert among the hot; a spec naming a layer twice or one the table lacks, a malformed spec and more
#       forced ids than hot places are refused;
#   (6) --headers-only writes the same headers and sizes, verify --headers-only passes and a full verify fails.
# Negative controls, which must fail: verify with one byte changed in a cold slab, in a hot slab, in the tier map or
# in an untouched tensor; tier with an expert in both tiers, one in neither, fewer hot experts than are used, a layer
# with expert biases, a --cold-in that lacks a cold id, and a cold file whose rows differ. Prints "ok" or fails an
# assertion. Uses DIR for its files and removes them. Standard library only.
import hashlib, os, random, shutil, struct, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import gguf_reshard as g  # noqa: E402

TOOL = os.path.join(here, "gguf_tier.py")
ALIGN = 32
F32, F16, Q4_0, Q8_0, Q6_K, IQ1_M, IQ3_S, Q2_0 = 0, 1, 2, 8, 14, 29, 21, 42
BLOCK = {F32: (1, 4), F16: (1, 2), Q4_0: (32, 18), Q8_0: (32, 34), Q6_K: (256, 210), IQ1_M: (256, 56),
         IQ3_S: (256, 110), Q2_0: (64, 18)}
N_EXPERT, N_USED = 8, 2
HOT = {0: [0, 4, 3, 2, 5], 1: [6, 1, 0, 2, 3]}
COLD = {0: [6, 1, 7], 1: [4, 5, 7]}
rng = random.Random(20261104238)


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
        assert len(raw) == nbytes(dims, typ), name
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


def t(name, dims, typ):
    return (name, dims, typ, rng.randbytes(nbytes(dims, typ)))


def model():
    s1 = [t("token_embd.weight", [16, 5], F32), t("blk.0.attn_norm.weight", [16], F32),
          t("blk.0.ffn_gate_inp.weight", [16, 8], F32), t("blk.0.ffn_gate_exps.weight", [256, 3, 8], IQ3_S),
          t("blk.0.ffn_up_exps.weight", [32, 3, 8], Q8_0), t("blk.0.ffn_down_exps.weight", [256, 2, 8], Q6_K),
          t("blk.0.ffn_down_exps.scale", [8], F32), t("blk.0.ffn_down_exps.input_scale", [8], F32),
          t("blk.0.indexer.q_proj.weight", [16, 8], F32), t("blk.0.ffn_down_shexp.weight", [32, 16], Q8_0)]
    s2 = [t("blk.1.ffn_gate_inp.weight", [16, 8], F16), t("blk.1.ffn_gate_up_exps.weight", [64, 6, 8], Q2_0),
          t("blk.1.ffn_down_exps.weight", [64, 3, 8], Q2_0), t("blk.2.ffn_gate_inp.weight", [16, 8], F32), t("blk.2.ffn_gate_exps.weight", [32, 5, 8], Q8_0),
          t("blk.2.ffn_up_exps.weight", [32, 5, 8], Q8_0), t("blk.2.ffn_down_exps.weight", [256, 1, 8], IQ3_S),
          t("blk.2.ffn_down_exps.scale", [8], F32), t("blk.2.ffn_up_exps.bias", [5, 8], F32),
          t("blk.2.attn_q.weight", [16, 9], F16)]
    s3 = [t("per_layer_token_embd.weight", [4, 9], F32)]
    return [s1, s2, s3]


N_TENSORS = 21


def arch_kvs(n_expert):
    return [("general.architecture", 8, "qwen4exp"), ("general.name", 8, "tier test"), ("qwen4exp.block_count", 4, 3),
            ("qwen4exp.expert_count", 4, n_expert), ("qwen4exp.expert_used_count", 4, N_USED),
            ("qwen4exp.attention.recurrent_layers", 9, [1, 0, 1])]


def kvs(shard, n_tensors=N_TENSORS):
    split = [("split.no", 2, shard), ("split.count", 2, 3), ("split.tensors.count", 5, n_tensors)]
    return (arch_kvs(N_EXPERT) if shard == 0 else []) + split


def slabs(raw, ids, n=N_EXPERT):
    slab = len(raw) // n
    return b"".join(raw[e * slab:(e + 1) * slab] for e in ids)


def tiered(tensors, hot, cold, cold_raw=None, hot_raw=None, cold_from_base=()):
    """The tiered tensors: cold_raw / hot_raw map an expert tensor's name to (type, data of its tier in order); the
    names in cold_from_base take the base's own cold slabs."""
    out = []
    for name, dims, typ, raw in tensors:
        parts = name.split(".")
        layer = int(parts[1]) if parts[0] == "blk" else None
        if layer in hot and parts[2].endswith("_exps"):
            # a tier whose source lacks a tensor (scales of a requantized tier) has none
            if hot_raw is None or name in hot_raw:
                ht, hd = hot_raw[name] if hot_raw else (typ, slabs(raw, hot[layer]))
                out.append((name, dims[:-1] + [len(hot[layer])], ht, hd))
            if cold_raw is None or name in cold_raw or name in cold_from_base:
                ct, cd = cold_raw[name] if cold_raw and name not in cold_from_base else (typ, slabs(raw, cold[layer]))
                out.append((name.replace("_exps.", "_exps_cold."), dims[:-1] + [len(cold[layer])], ct, cd))
            continue
        out.append((name, dims, typ, raw))
        if layer in hot and parts[2] == "ffn_gate_inp":
            m = [0.0] * N_EXPERT
            for s, e in enumerate(hot[layer]):
                m[e] = float(s)
            for s, e in enumerate(cold[layer]):
                m[e] = float(len(hot[layer]) + s)
            out.append((f"blk.{layer}.ffn_exps_tier.weight", [N_EXPERT], F32, struct.pack(f"<{N_EXPERT}f", *m)))
    return out


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("layer\tids\n" + "".join(f"{l}\t{','.join(map(str, v))}\n" for l, v in sorted(ids.items())))


def main():
    d = os.path.abspath(sys.argv[1])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shards = model()
    names = [os.path.join(d, f"m-0000{i + 1}-of-00003.gguf") for i in range(3)]
    for i, (p, ts) in enumerate(zip(names, shards)):
        write_gguf(p, kvs(i), ts)
    added = 6 + 3  # five cold tensors (three weights, two scales) and a map in layer 0, two weights and a map in layer 1

    # (1) split: layer 0 ties 2, 5 at 40 (2 first), layer 1 ties everything below 6 and 1; layer 2 is not in the table
    counts = {0: [90, 10, 40, 70, 80, 40, 30, 5], 1: [3, 9, 3, 3, 1, 1, 50, 0]}
    table = os.path.join(d, "experts.tsv")
    with open(table, "w") as f:
        f.write("layer\texpert\troutings\n")
        for l, c in counts.items():
            f.writelines(f"{l}\t{e}\t{v}\n" for e, v in enumerate(c))
    ok("split", "--experts", table, "--hot", "5", "--out-dir", f"{d}/split")
    hot_tsv, cold_tsv = f"{d}/split/hot.tsv", f"{d}/split/cold.tsv"
    for path, want in ((hot_tsv, HOT), (cold_tsv, COLD)):
        with open(path) as f:
            assert f.read() == "layer\tids\n" + "".join(f"{l}\t{','.join(map(str, want[l]))}\n" for l in (0, 1)), path

    # (2) cold from the base: this test's writer, the third shard linked, verify
    out = f"{d}/same"
    ok("tier", "--hot-ids", hot_tsv, "--cold-ids", cold_tsv, "--cold", *names, "--out-dir", out, "--prefix", "p",
       "--link-unchanged", *names)
    news = [os.path.join(out, f"p-0000{i + 1}-of-00003.gguf") for i in range(3)]
    want_same = []
    for i in range(2):
        ref = os.path.join(d, f"ref-same-{i}.gguf")
        write_gguf(ref, kvs(i, N_TENSORS + added), tiered(shards[i], HOT, COLD))
        assert sha(news[i]) == sha(ref), f"shard {i + 1}: not the expected bytes"
        want_same.append(sha(ref))
    assert os.path.islink(news[2]) and os.path.samefile(news[2], names[2]), "shard 3 is not a link to its source"
    tier_opts = ["--hot-ids", hot_tsv, "--cold-ids", cold_tsv]
    ok("verify", *names, "--new", *news, *tier_opts, "--cold", *names)

    # (3) extract the cold ids, with a synthetic importance matrix (an unrelated entry and a stats entry are dropped)
    im_t = [t("blk.0.ffn_gate_exps.weight.in_sum2", [256, 8], F32), t("blk.0.ffn_gate_exps.weight.counts", [1, 8], F32),
            t("blk.0.ffn_up_exps.weight.in_sum", [32, 8], F32), t("blk.0.ffn_up_exps.weight.in_sum2", [32, 8], F32),
            t("blk.0.ffn_up_exps.weight.counts", [1, 8], F32), t("blk.0.ffn_up_exps.weight.stats", [12], F32),
            t("blk.0.ffn_down_exps.weight.in_sum2", [256, 8], F32), t("blk.0.ffn_down_exps.weight.counts", [1, 8], F32),
            t("blk.0.attn_q.weight.in_sum2", [16, 1], F32), t("blk.0.attn_q.weight.counts", [1, 1], F32),
            t("blk.1.ffn_gate_up_exps.weight.in_sum2", [64, 8], F32),
            t("blk.1.ffn_gate_up_exps.weight.counts", [1, 8], F32),
            t("blk.1.ffn_down_exps.weight.in_sum2", [64, 8], F32), t("blk.1.ffn_down_exps.weight.counts", [1, 8], F32)]
    im_kv = [("general.type", 8, "imatrix"), ("imatrix.chunk_count", 4, 10), ("imatrix.chunk_size", 4, 512)]
    im = os.path.join(d, "imatrix.gguf")
    write_gguf(im, im_kv, im_t)
    ok("extract", "--ids", cold_tsv, "--imatrix", im, "--out-dir", f"{d}/ex", *names)
    expert_ts = [x for s in shards for x in s if x[0].split(".")[1] in ("0", "1") and x[0].split(".")[2].endswith("_exps")]
    ref = os.path.join(d, "ref-ex.gguf")
    write_gguf(ref, arch_kvs(3), [(n, dims[:-1] + [3], typ, slabs(raw, COLD[int(n.split(".")[1])]))
                                  for n, dims, typ, raw in expert_ts])
    assert sha(f"{d}/ex/experts.gguf") == sha(ref), "extract: not the expected experts.gguf"
    ref = os.path.join(d, "ref-im.gguf")
    write_gguf(ref, im_kv, [(n, dims[:-1] + [3], typ, slabs(raw, COLD[int(n.split(".")[1])]))
                            for n, dims, typ, raw in im_t if dims[-1] == 8])
    assert sha(f"{d}/ex/imatrix.gguf") == sha(ref), "extract: not the expected imatrix.gguf"

    # (4) the extracted cold file gives the same bytes as the base
    ok("tier", *tier_opts, "--cold", f"{d}/ex/experts.gguf", "--cold-in", f"{d}/ex/ids.tsv", "--out-dir", f"{d}/ex4",
       "--prefix", "p", *names)
    for i in range(2):
        assert sha(f"{d}/ex4/p-0000{i + 1}-of-00003.gguf") == want_same[i], f"shard {i + 1} from extract: other bytes"

    # (5) a cold file in other types, in its own expert order, and a hot file holding its experts in reverse
    cold_order = {0: [7, 1, 6], 1: [5, 7, 4]}
    cold_ts = [("blk.0.ffn_gate_exps.weight", [256, 3, 3], IQ1_M), ("blk.0.ffn_up_exps.weight", [32, 3, 3], Q4_0),
               ("blk.0.ffn_down_exps.weight", [256, 2, 3], IQ1_M), ("blk.1.ffn_gate_up_exps.weight", [64, 6, 3], Q2_0),
               ("blk.1.ffn_down_exps.weight", [64, 3, 3], Q2_0)]
    cold_ts = [t(*x) for x in cold_ts]
    write_gguf(f"{d}/cold.gguf", arch_kvs(3), cold_ts)
    write_ids(f"{d}/cold_in.tsv", cold_order)
    hot_order = {0: [7, 6, 5, 4, 3, 2, 1, 0], 1: [7, 6, 5, 4, 3, 2, 1, 0]}
    hot_ts = [(n, dims, typ, slabs(raw, hot_order[int(n.split(".")[1])])) for n, dims, typ, raw in expert_ts]
    hot_ts = [(n, dims, Q8_0 if typ == Q6_K else typ, raw if typ != Q6_K else rng.randbytes(nbytes(dims, Q8_0)))
              for n, dims, typ, raw in hot_ts]
    write_gguf(f"{d}/hot.gguf", arch_kvs(8), hot_ts)
    write_ids(f"{d}/hot_in.tsv", hot_order)
    cold_raw = {n: (typ, slabs(raw, [cold_order[int(n.split(".")[1])].index(e) for e in COLD[int(n.split(".")[1])]], 3))
                for n, dims, typ, raw in cold_ts}
    hot_raw = {n: (typ, slabs(raw, [hot_order[int(n.split(".")[1])].index(e) for e in HOT[int(n.split(".")[1])]]))
               for n, dims, typ, raw in hot_ts}
    mix_opts = [*tier_opts, "--cold", f"{d}/cold.gguf", "--cold-in", f"{d}/cold_in.tsv", "--hot", f"{d}/hot.gguf",
                "--hot-in", f"{d}/hot_in.tsv"]
    ok("tier", *mix_opts, "--out-dir", f"{d}/mix", *names)
    mix = [f"{d}/mix/{os.path.basename(p)}" for p in names]
    for i in range(2):
        ref = os.path.join(d, f"ref-mix-{i}.gguf")
        # the cold file has no scales: layer 0's two cold scales are not written
        write_gguf(ref, kvs(i, N_TENSORS + added - 2), tiered(shards[i], HOT, COLD, cold_raw, hot_raw))
        assert sha(mix[i]) == sha(ref), f"shard {i + 1} of the mixed file: not the expected bytes"
    assert not os.path.islink(mix[2]) and sha(mix[2]) == sha(names[2]), "shard 3 without --link-unchanged: a copy"
    ok("verify", *names, "--new", *mix, *mix_opts)

    # (5b) --cold-only: gate, up and the fused gate_up from the cold file, the down projections (and their scales) the base's own
    ok("tier", *mix_opts, "--cold-only", "ffn_(gate|up|gate_up)_exps", "--out-dir", f"{d}/only", *names)
    base_down = {"blk.0.ffn_down_exps.weight", "blk.0.ffn_down_exps.scale", "blk.0.ffn_down_exps.input_scale",
                 "blk.1.ffn_down_exps.weight"}
    for i in range(2):
        ref = os.path.join(d, f"ref-only-{i}.gguf")
        write_gguf(ref, kvs(i, N_TENSORS + added), tiered(shards[i], HOT, COLD, cold_raw, hot_raw, base_down))
        assert sha(f"{d}/only/{os.path.basename(names[i])}") == sha(ref), f"shard {i + 1} with --cold-only: other bytes"
    # --cold-rest naming another copy of the base gives the same bytes
    copies = []
    for p in names:
        shutil.copy(p, f"{d}/copy-{os.path.basename(p)}")
        copies.append(f"{d}/copy-{os.path.basename(p)}")
    ok("tier", *mix_opts, "--cold-only", "ffn_(gate|up|gate_up)_exps", "--cold-rest", *copies, "--out-dir", f"{d}/rest", *names)
    for i in range(2):
        assert sha(f"{d}/rest/{os.path.basename(names[i])}") == sha(f"{d}/only/{os.path.basename(names[i])}"), \
            f"shard {i + 1} with --cold-rest: not the bytes of --cold-only alone"
    fails("tier", *mix_opts, "--cold-rest", *copies, "--out-dir", f"{d}/x", *names)
    ok("verify", *names, "--new", *[f"{d}/only/{os.path.basename(p)}" for p in names], *mix_opts, "--cold-only",
       "ffn_(gate|up|gate_up)_exps")

    # (1c) two tables rank by their summed shares: a second table (as many routings in all) that favours experts 1 and
    # 7 in layer 0 moves them up; weights 1,0 give the first table's order again
    table2 = os.path.join(d, "experts2.tsv")
    c2 = {0: [10, 300, 10, 10, 10, 10, 10, 1000], 1: counts[1]}
    with open(table2, "w") as f:
        f.write("layer\texpert\troutings\n")
        for l, c in c2.items():
            f.writelines(f"{l}\t{e}\t{v}\n" for e, v in enumerate(c))
    ok("split", "--experts", table, "--experts", table2, "--hot", "5", "--out-dir", f"{d}/two")
    t0, t2 = sum(counts[0]), sum(c2[0])
    share = {e: counts[0][e] / t0 + c2[0][e] / t2 for e in range(8)}
    want0 = sorted(range(8), key=lambda e: (-share[e], e))
    with open(f"{d}/two/hot.tsv") as f:
        assert f.read().splitlines()[1] == f"0\t{','.join(map(str, want0[:5]))}", "two tables: not the summed-share order"
    ok("split", "--experts", table, "--experts", table2, "--weights", "1,0", "--hot", "5", "--out-dir", f"{d}/two10")
    with open(f"{d}/two10/hot.tsv") as f, open(hot_tsv) as first:
        assert f.read() == first.read(), "weights 1,0: not the first table's split"
    fails("split", "--experts", table, "--experts", table2, "--weights", "1", "--hot", "5", "--out-dir", f"{d}/x")

    # (1b) a --hot spec tiers only its layers (layer 0 here; layer 1 keeps its tensors), and --force-hot puts layer 0's
    # last-ranked expert 7 among the hot (the other four hot places by rank)
    write_ids(f"{d}/force.tsv", {0: [7]})
    ok("split", "--experts", table, "--hot", "0:5", "--force-hot", f"{d}/force.tsv", "--out-dir", f"{d}/spec")
    want_h, want_c = {0: [0, 4, 3, 2, 7]}, {0: [5, 6, 1]}
    for path, want in ((f"{d}/spec/hot.tsv", want_h), (f"{d}/spec/cold.tsv", want_c)):
        with open(path) as f:
            assert f.read() == "layer\tids\n" + "".join(f"{l}\t{','.join(map(str, v))}\n" for l, v in want.items()), path
    ok("tier", "--hot-ids", f"{d}/spec/hot.tsv", "--cold-ids", f"{d}/spec/cold.tsv", "--cold", *names, "--out-dir",
       f"{d}/spec", "--link-unchanged", *names)
    ref = os.path.join(d, "ref-spec-0.gguf")
    write_gguf(ref, kvs(0, N_TENSORS + 6), tiered(shards[0], want_h, want_c))
    assert sha(f"{d}/spec/{os.path.basename(names[0])}") == sha(ref), "the --hot spec's shard 1: other bytes"
    assert os.path.islink(f"{d}/spec/{os.path.basename(names[1])}"), "shard 2 (layer 1 untiered) is not a link"
    for args in (("--hot", "0:5,0:4"), ("--hot", "9:5"), ("--hot", "0-1"), ("--hot", "5", "--force-hot", f"{d}/bad_force.tsv")):
        write_ids(f"{d}/bad_force.tsv", {0: [0, 1, 2, 3, 4, 5]})
        fails("split", "--experts", table, *args, "--out-dir", f"{d}/x")

    # (6) header copies
    ok("tier", *mix_opts, "--out-dir", f"{d}/hdr", "--headers-only", *names)
    hdr = [f"{d}/hdr/{os.path.basename(p)}" for p in names]
    for full, h in zip(mix, hdr):
        assert os.path.getsize(full) == os.path.getsize(h), f"{h}: size"
        n = g.Header(full).data0
        with open(full, "rb") as a, open(h, "rb") as b:
            fa, fb = a.read(), b.read()
        assert fa[:n] == fb[:n] and fb[n:] == bytes(len(fb) - n), f"{h}: not the header and zeros"
    ok("verify", *names, "--new", *hdr, *mix_opts, "--headers-only")
    fails("verify", *names, "--new", *hdr, *mix_opts)

    # negative controls: verify catches one changed byte in a cold slab, a hot slab, a tier map, an untouched tensor
    def corrupt(src, tensor, delta):
        h = g.Header(src)
        off = next(x[3] for x in h.tensors if x[0] == tensor)
        p = os.path.join(d, "bad", os.path.basename(src))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy(src, p)
        with open(p, "r+b") as f:
            f.seek(h.data0 + off + delta)
            b = f.read(1)
            f.seek(h.data0 + off + delta)
            f.write(bytes([b[0] ^ 1]))
        return p

    for tensor, delta in (("blk.0.ffn_up_exps_cold.weight", 40), ("blk.0.ffn_gate_exps.weight", 3),
                          ("blk.0.ffn_exps_tier.weight", 9), ("blk.0.indexer.q_proj.weight", 0)):
        bad = corrupt(mix[0], tensor, delta)
        fails("verify", *names, "--new", bad, mix[1], mix[2], *mix_opts)
    # tier refuses lists that are not a partition, too few hot experts, a tiered layer with scales, a short --cold-in,
    # and a cold file with other rows
    bad_lists = [({0: [0, 4, 3, 2, 5], 1: HOT[1]}, {0: [6, 1, 7, 5], 1: COLD[1]}),
                 ({0: [0, 4, 3, 2], 1: HOT[1]}, {0: [6, 1, 7], 1: COLD[1]}),
                 ({0: [0], 1: HOT[1]}, {0: [1, 2, 3, 4, 5, 6, 7], 1: COLD[1]}),
                 ({0: HOT[0], 1: HOT[1], 2: [0, 1, 2, 3, 4]}, {0: COLD[0], 1: COLD[1], 2: [5, 6, 7]})]
    for hot, cold in bad_lists:
        write_ids(f"{d}/bh.tsv", hot)
        write_ids(f"{d}/bc.tsv", cold)
        fails("tier", "--hot-ids", f"{d}/bh.tsv", "--cold-ids", f"{d}/bc.tsv", "--cold", *names, "--out-dir", f"{d}/x",
              *names)
    write_ids(f"{d}/short.tsv", {0: [7, 1, 5], 1: [5, 7, 4]})
    fails("tier", *tier_opts, "--cold", f"{d}/cold.gguf", "--cold-in", f"{d}/short.tsv", "--out-dir", f"{d}/x", *names)
    write_gguf(f"{d}/rows.gguf", arch_kvs(3), [t(n, [dims[0], dims[1] + 1, 3], typ) for n, dims, typ, _r in cold_ts])
    fails("tier", *tier_opts, "--cold", f"{d}/rows.gguf", "--cold-in", f"{d}/cold_in.tsv", "--out-dir", f"{d}/x", *names)
    shutil.rmtree(d)
    print("ok")


if __name__ == "__main__":
    main()
