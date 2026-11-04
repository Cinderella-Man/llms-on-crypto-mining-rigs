#!/usr/bin/env python3
# gguf_prune.py: keeps N of the routed experts in every layer of a (split) GGUF and drops the rest, so that a file too
# big for the cards fits by losing the experts its routers pick least (Qwen3.8-Flash-Next: 512 a layer, 10 picked).
#   gguf_prune.py prune --keep N (--experts experts.tsv[.gz] [--by routings|energy] [--force-ids FORCED.tsv] | --ids IDS.tsv) \
#       --out-dir <dir> [--prefix <name>] [--headers-only] [--link-unchanged] SHARD...
#     Which experts: --experts takes a table with the columns layer, expert and the --by column (a count per expert,
#     docs/data/2026-11-04_expert_usage/experts.tsv.gz) and keeps each layer's N largest, ties to the lower id, in
#     ascending id order; --ids takes one line per layer, `<layer><TAB><id>,<id>,...`, kept in the order given (a
#     header line is skipped), N ids a layer (--keep may be left out). Every layer with routed experts must be listed.
#     --force-ids (with --experts) takes the --ids format with any number of ids a layer, layers may be left out: each
#     layer keeps its forced ids whatever their rank and fills its other N - (forced) places with its largest others
#     (the experts tool-call text needs, docs/data/2026-11-04_toolcall_experts/run/force_keep.py).
#     What changes: every tensor of a layer whose expert axis is its last dimension (`ffn_gate_exps`, `ffn_up_exps`,
#     `ffn_down_exps`, `ffn_gate_up_exps` with their `.scale` and `.input_scale`, the router `ffn_gate_inp`, and
#     `exp_probs_b`) keeps the slabs of the kept experts, in the kept order. That axis is the outermost one, so an
#     expert's slab is a run of whole rows, whole blocks of any quantized type: the bytes move unchanged. The key
#     `<arch>.expert_count` becomes N. Every other key and tensor is copied byte for byte, each tensor stays in its
#     shard, and the shards keep their names in --out-dir, or take --prefix in place of the part before
#     -0000k-of-0000N.gguf. Tensor data is written in the order of the tensor infos, each tensor padded to the alignment,
#     so keeping all experts in ascending order writes the input again byte for byte. The kept ids go to
#     <out-dir>/kept.tsv (the --ids format). --headers-only writes only the headers, each file sized as if the data were
#     there (sparse: header copies for scripts/fit_probe.sh). --link-unchanged makes a shard with nothing to change a
#     symbolic link to its source (Flash-Next's 28.8 GB n-gram table shard).
#   gguf_prune.py verify OLD_SHARD... --new NEW_SHARD... --ids <out-dir>/kept.tsv [--headers-only]
#     checks each new shard against its old one: the same keys in the same order with the same values but
#     `<arch>.expert_count` = N, the same tensor names, types and order, the same shapes but N on the expert axis of the
#     sliced tensors, every other tensor identical byte for byte, and every sliced tensor's slab k byte-equal to the old
#     slab of its layer's k-th kept id. --headers-only checks the headers and the file sizes only.
# Standard library only.
import argparse, gzip, math, os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reshard import CHUNK, SCALAR, Header, copy_range, write_all, write_header  # noqa: E402

# ggml type -> (values per block, bytes per block), ggml.h and ggml-common.h of Miner Llama's v3 base (37e717ff6)
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
         17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
         35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18), 43: (64, 36)}
SLICED = re.compile(r"^blk\.(\d+)\.(ffn_(gate|up|down|gate_up)_exps\.\w+|ffn_gate_inp\.\w+|exp_probs_b\.\w+)$")


def nbytes(name, dims, typ):
    assert typ in BLOCK, f"{name}: ggml type {typ} is not in the block table"
    k, size = BLOCK[typ]
    assert dims[0] % k == 0, f"{name}: rows of {dims[0]} values, not whole blocks of {k}"
    return math.prod(dims) // k * size


def arch_key(h, key):
    raw = h.value("general.architecture")  # a string's raw bytes: its 8-byte length, then the text
    return f"{raw[8:].decode()}.{key}" if raw else None


def first_value(hs, key):
    """The value of <arch>.<key> in the first shard that has it (the architecture is in shard 1 only)."""
    k = arch_key(hs[0], key)
    return hs[0].value(k) if k else None


def sliced(hs, n_expert):
    """{(shard index, name): (layer, slab bytes)} for every tensor on the expert axis."""
    out = {}
    for i, h in enumerate(hs):
        for name, dims, typ, _off in h.tensors:
            m = SLICED.match(name)
            if not m:
                continue
            assert dims[-1] == n_expert, f"{name}: last dimension {dims[-1]}, expert_count {n_expert}"
            size = nbytes(name, dims, typ)
            assert size <= h.slot[name] < size + h.align, f"{name}: {h.slot[name]} bytes in the file, {size} expected"
            out[i, name] = (int(m[1]), size // n_expert)
    return out


def read_ids(path):
    ids = {}
    with open(path) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if not cols[0].isdigit():
                continue
            assert int(cols[0]) not in ids, f"{path}: layer {cols[0]} twice"
            ids[int(cols[0])] = [int(v) for v in cols[1].split(",")] if len(cols) > 1 and cols[1] else []
    return ids


def rank_ids(path, by, keep, force=None):
    """Each layer's `keep` largest experts by column `by`, ties to the lower id, in ascending id order. With `force`
    (layer -> ids) a layer keeps its forced ids and fills its other places with its largest others."""
    per = {}
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        head = f.readline().rstrip("\n").split("\t")
        li, ei, vi = head.index("layer"), head.index("expert"), head.index(by)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            per.setdefault(int(cols[li]), []).append((-float(cols[vi]), int(cols[ei])))
    force = force or {}
    assert set(force) <= set(per), f"forced ids for layers {sorted(set(force) - set(per))}, which the table lacks"
    out = {}
    for layer, rows in per.items():
        forced = set(force.get(layer, []))
        assert len(forced) == len(force.get(layer, [])), f"layer {layer}: a forced id repeats"
        assert forced <= {e for _v, e in rows}, f"layer {layer}: forced ids {sorted(forced - {e for _v, e in rows})} are not in the table"
        assert len(forced) <= keep, f"layer {layer}: {len(forced)} forced ids, more than the {keep} kept"
        rest = [e for _v, e in sorted(rows) if e not in forced]
        out[layer] = sorted(forced | set(rest[:keep - len(forced)]))
    return out


def check_ids(ids, layers, n_expert, n_used):
    missing = sorted(set(layers) - set(ids))
    assert not missing, f"no kept ids for layers {missing}"
    counts = {len(ids[l]) for l in layers}
    assert len(counts) == 1, f"layers keep different counts {sorted(counts)}: llama.cpp takes one expert_count"
    keep = counts.pop()
    assert (n_used or 0) <= keep <= n_expert, f"keep {keep}: needs expert_used_count {n_used} to expert_count {n_expert}"
    for l in layers:
        assert len(set(ids[l])) == keep and all(0 <= e < n_expert for e in ids[l]), f"layer {l}: ids repeat or out of range"
    return keep


def runs(order):
    """Consecutive ascending stretches of `order`: [(first id, length), ...]."""
    out = []
    for e in order:
        if out and out[-1][0] + out[-1][1] == e:
            out[-1][1] += 1
        else:
            out.append([e, 1])
    return out


def out_name(path, prefix):
    base = os.path.basename(path)
    if not prefix:
        return base
    m = re.search(r"-\d{5}-of-\d{5}\.gguf$", base)
    return prefix + (m[0] if m else ".gguf")


def new_kv(h, keep):
    k = arch_key(h, "expert_count")
    return [(key, t, struct.pack(SCALAR[t], keep) if key == k else raw) for key, t, raw in h.kv]


def cmd_prune(a):
    assert sys.byteorder == "little"
    hs = [Header(p) for p in a.shards]
    count = hs[0].value("split.count")
    assert count in (None, len(hs)), f"{len(hs)} shards given, split.count is {count}"
    n_expert, n_used = first_value(hs, "expert_count"), first_value(hs, "expert_used_count")
    assert n_expert, "no <arch>.expert_count in the first shard"
    cut = sliced(hs, n_expert)
    layers = sorted({layer for layer, _slab in cut.values()})
    assert layers, "no tensor on the expert axis"
    if a.ids:
        assert not a.force_ids, "--force-ids goes with --experts, not --ids"
        ids = read_ids(a.ids)
        assert a.keep is None or all(len(ids[l]) == a.keep for l in layers if l in ids), f"{a.ids}: not {a.keep} a layer"
    else:
        assert a.keep, "--keep N with --experts"
        ids = rank_ids(a.experts, a.by, a.keep, read_ids(a.force_ids) if a.force_ids else None)
    keep = check_ids(ids, layers, n_expert, n_used)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "kept.tsv"), "w") as f:
        f.write("layer\tkept_ids\n" + "".join(f"{l}\t{','.join(map(str, ids[l]))}\n" for l in layers))
    total = [0, 0]
    for i, h in enumerate(hs):
        path = os.path.join(a.out_dir, out_name(h.path, a.prefix))
        assert os.path.islink(path) or not (os.path.exists(path) and os.path.samefile(path, h.path)), \
            f"{path} is its own source"
        kv = new_kv(h, keep)
        tensors, changed = [], 0
        for name, dims, typ, off in h.tensors:  # the header's order, which is also the order the data is written in
            if (i, name) in cut:
                tensors.append((name, dims[:-1] + [keep], typ, off, keep * cut[i, name][1]))
                changed += 1
            else:
                tensors.append((name, dims, typ, off, h.slot[name]))
        if not changed and kv == h.kv and a.link_unchanged:
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(os.path.realpath(h.path), path)
            print(f"{path} -> {os.path.realpath(h.path)}: nothing to change")
            continue
        with open(h.path, "rb") as src, open(path + ".part", "wb") as dst:
            write_header(dst, h.version, kv, tensors, h.align)
            if a.headers_only:
                os.ftruncate(dst.fileno(), os.lseek(dst.fileno(), 0, 1) + sum(s + (-s) % h.align for *_x, s in tensors))
            else:
                for name, _dims, _typ, off, slot in tensors:
                    start = h.data0 + off
                    if (i, name) in cut:
                        slab = cut[i, name][1]
                        for e, n in runs(ids[cut[i, name][0]]):
                            copy_range(src, dst, start + e * slab, n * slab, 1)
                        write_all(dst.fileno(), b"\0" * ((-slot) % h.align))
                    else:
                        copy_range(src, dst, start, slot, h.align)
        os.replace(path + ".part", path)
        before = sum(h.slot[t[0]] for t in h.tensors)
        after = sum(s + (-s) % h.align for *_x, s in tensors)
        total[0], total[1] = total[0] + before, total[1] + after
        print(f"{path}: {len(tensors)} tensors, {changed} sliced, {os.path.getsize(path):,} bytes "
              f"({after - before:+,} of tensor data){' (header only)' if a.headers_only else ''}")
        sys.stdout.flush()
    print(f"kept {keep} of {n_expert} experts in {len(layers)} layers: tensor data {total[0]:,} -> {total[1]:,} bytes "
          f"({total[1] - total[0]:+,})")


def same(fo, po, fn, pn, n):
    while n:
        k = min(n, CHUNK)
        if os.pread(fo.fileno(), k, po) != os.pread(fn.fileno(), k, pn):
            return False
        n, po, pn = n - k, po + k, pn + k
    return True


def cmd_verify(a):
    assert len(a.old) == len(a.new), "as many new shards as old ones"
    hos, hns = [Header(p) for p in a.old], [Header(p) for p in a.new]
    n_expert, n_used = first_value(hos, "expert_count"), first_value(hos, "expert_used_count")
    cut = sliced(hos, n_expert)
    layers = sorted({layer for layer, _slab in cut.values()})
    ids = read_ids(a.ids)
    keep = check_ids(ids, layers, n_expert, n_used)
    for i, (ho, hn) in enumerate(zip(hos, hns)):
        assert hn.kv == new_kv(ho, keep), f"{hn.path}: keys differ from the old ones with expert_count {keep}"
        assert len(ho.tensors) == len(hn.tensors), f"{hn.path}: {len(hn.tensors)} tensors, {len(ho.tensors)} before"
        want, off = 0, 0
        for (name, dims, typ, _oo), (nn, dn, tn, on) in zip(ho.tensors, hn.tensors):
            assert (name, typ) == (nn, tn), f"{hn.path}: tensor {nn} of type {tn} where {name} of type {typ} was"
            assert dn == (dims[:-1] + [keep] if (i, name) in cut else dims), f"{name}: shape {dn}"
            assert on == off, f"{name}: offset {on}, {off} expected"
            slot = keep * cut[i, name][1] if (i, name) in cut else ho.slot[name]
            off += slot + (-slot) % ho.align
        assert hn.size == hn.data0 + off, f"{hn.path}: {hn.size:,} bytes, {hn.data0 + off:,} expected"
        if a.headers_only:
            print(f"{hn.path}: header ok against {ho.path} ({sum(1 for k in cut if k[0] == i)} tensors sliced)")
            continue
        n_same = n_cut = 0
        with open(ho.path, "rb") as fo, open(hn.path, "rb") as fn:
            for (name, _d, _t, oo), (_n, _dn, _tn, on) in zip(ho.tensors, hn.tensors):
                po, pn = ho.data0 + oo, hn.data0 + on
                if (i, name) in cut:
                    layer, slab = cut[i, name]
                    k = 0
                    for e, n in runs(ids[layer]):
                        assert same(fo, po + e * slab, fn, pn + k * slab, n * slab), \
                            f"{name}: the slabs of experts {e}-{e + n - 1} differ"
                        k += n
                    n_cut += 1
                else:
                    assert same(fo, po, fn, pn, ho.slot[name]), f"{name}: bytes differ"
                    n_same += 1
        print(f"{hn.path}: {n_same} tensors identical to {ho.path}, {n_cut} sliced to {keep} experts, every kept slab equal")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prune")
    s.add_argument("--keep", type=int, help="experts kept per layer")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--experts", help="per layer and expert counts (columns layer, expert and --by), .gz or plain")
    g.add_argument("--ids", help="<layer>\\t<id>,<id>,... per layer, kept in that order")
    s.add_argument("--by", default="routings", help="the --experts column to rank by (default routings)")
    s.add_argument("--force-ids", help="<layer>\\t<id>,<id>,... per layer: kept whatever their rank (with --experts)")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--prefix", help="new file name before -0000k-of-0000N.gguf")
    s.add_argument("--headers-only", action="store_true", help="write the headers and size the files, copy no data")
    s.add_argument("--link-unchanged", action="store_true",
                   help="a shard with nothing to change becomes a symbolic link to the old shard")
    s.add_argument("shards", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("old", nargs="+")
    v.add_argument("--new", nargs="+", required=True)
    v.add_argument("--ids", required=True, help="the kept.tsv that prune wrote")
    v.add_argument("--headers-only", action="store_true", help="check the headers and sizes only")
    a = ap.parse_args()
    cmd_prune(a) if a.cmd == "prune" else cmd_verify(a)


if __name__ == "__main__":
    main()
