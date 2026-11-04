#!/usr/bin/env python3
# gguf_tier.py: splits each layer's routed experts into a hot tier and a cold tier held in tensors of their own, so that
# the experts the router picks least can sit in a lower-bit type instead of being cut out (scripts/gguf_prune.py). The
# router keeps scoring all of them. Miner Llama's -coldtier engine reads such a file (Q238); llama.cpp without it
# refuses the file (the cold tensors are unknown to it).
#   gguf_tier.py split --experts experts.tsv[.gz]... [--weights w,...] --hot N|<a>-<b>:N,... [--force-hot ids.tsv]
#       [--by routings] --out-dir <dir>
#     <dir>/hot.tsv and <dir>/cold.tsv (gguf_prune.py's --ids format, `<layer><TAB><id>,<id>,...`): each layer's N
#     experts with the largest --by count are hot, the others cold; each tier most-picked first, ties to the lower id,
#     so slot 0 of a tier is its most used expert (the engine reads slot 0 for the other tier's picks). Several
#     --experts tables (usage on agent, wiki and code text) rank by the sum of each table's share of the layer's
#     routings, each times its --weights entry, so no single text decides which experts go cold. A --hot spec
#     gives some layers (or 8-layer groups) their own N and leaves the others untiered; --force-hot's ids (e.g. the
#     tool-call experts) are hot whatever their rank, the other hot places going by rank.
#   gguf_tier.py extract --ids cold.tsv [--imatrix imatrix.gguf] --out-dir <dir> SHARD...
#     <dir>/experts.gguf: the first shard's keys (no split keys, <arch>.expert_count = the ids per layer) and only the
#     routed expert tensors of the listed layers (weights and per-expert scales), each cut to the listed ids in their
#     order under its own name: the input of `llama-quantize --allow-requantize --imatrix <dir>/imatrix.gguf
#     --tensor-type ...`, which converts the weights and copies the scales. <dir>/imatrix.gguf: the importance matrix's
#     keys and the .in_sum2, .in_sum and .counts of those weights cut the same way (its other entries are dropped), so
#     llama-quantize weighs each expert by its own statistics. <dir>/ids.tsv: the ids.
#   gguf_tier.py tier --hot-ids hot.tsv --cold-ids cold.tsv --cold FILE... [--cold-in ids.tsv] [--cold-only REGEX
#       [--cold-rest FILE...]] [--hot FILE... [--hot-in ids.tsv]] --out-dir <dir> [--prefix <name>] [--headers-only] [--link-unchanged] SHARD...
#     the tiered file: SHARD...'s keys and tensors byte for byte, but in each layer of the id lists
#       ffn_<p>_exps.<s>        the hot ids' slabs in hot.tsv's order, from --hot (default: SHARD... itself)
#       ffn_<p>_exps_cold.<s>   right after it, the cold ids' slabs in cold.tsv's order, from --cold
#       ffn_exps_tier.weight    right after the router ffn_gate_inp (F32, one value per expert id): s for the hot id
#                               in slot s, n_hot + s for the cold id in slot s
#     (<p> = gate, up, down or gate_up; <s> = weight, or scale and input_scale where the tier's source has them, since
#     a requantized tier may have none; with --cold-only REGEX only the cold tensors whose names match come from
#     --cold, the others are the base's own slabs, or those of --cold-rest FILE...), and split.tensors.count raised by the new tensors in the first shard and
#     every shard with tensors to change (llama.cpp reads the first shard's; an untouched shard can stay a link). A
#     --cold or --hot file's expert axis holds the experts its --cold-in / --hot-in list names, in that order
#     (extract's ids.tsv), or all of them in id order without one. A slab is a run of whole rows, whole blocks of any
#     type: it moves unchanged. Each layer's hot and cold ids together are all of its experts, each once; the hot tier
#     keeps at least <arch>.expert_used_count; a tiered layer has no per-expert biases. Shard names as
#     gguf_prune.py's (--prefix, --headers-only, --link-unchanged).
#   gguf_tier.py verify SHARD... --new NEW... --hot-ids ... --cold-ids ... --cold ... [the other tier options]
#       [--headers-only]
#     checks each new shard against what `tier` must write: the keys, the tensor names, types, shapes and order, every
#     other tensor byte for byte, every hot and cold slab byte-equal to its source slab, the tier maps' values (a shard
#     that is its source file, a link, is not read twice); --headers-only checks the headers and file sizes only.
# Standard library only.
import argparse, gzip, os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reshard import Header, copy_range, write_all, write_header  # noqa: E402
from gguf_prune import SCALAR, arch_key, first_value, nbytes, out_name, read_ids, runs, same  # noqa: E402

EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.(\w+)$")
SUFFIXES = ("weight", "scale", "input_scale")  # what a tier can hold: weights and their per-expert scales
F32 = 0
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


def cold_name(name):
    return re.sub(r"_exps\.", "_exps_cold.", name, count=1)


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("layer\tids\n" + "".join(f"{l}\t{','.join(map(str, ids[l]))}\n" for l in sorted(ids)))


def hot_counts(spec, layers):
    """--hot: N for every layer, or "<a>-<b>:N,<c>:N,..." for some (the other layers are not tiered)."""
    if re.fullmatch(r"\d+", spec):
        return {l: int(spec) for l in layers}
    out = {}
    for part in spec.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?:(\d+)", part.strip())
        assert m, f"--hot {spec}: '{part}' is not <layer>[-<layer>]:<count>"
        for l in range(int(m[1]), int(m[2] or m[1]) + 1):
            assert l in layers, f"--hot {spec}: layer {l} is not in the table"
            assert l not in out, f"--hot {spec}: layer {l} twice"
            out[l] = int(m[3])
    return out


def read_counts(path, by):
    """{layer: {expert: count}} from a table with the columns layer, expert and `by`."""
    out = {}
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        head = f.readline().rstrip("\n").split("\t")
        li, ei, vi = head.index("layer"), head.index("expert"), head.index(by)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            out.setdefault(int(cols[li]), {})[int(cols[ei])] = float(cols[vi])
    return out


def cmd_split(a):
    # one table ranks by its count; several by the sum of each one's share of the layer's total, times its weight
    tables = [read_counts(p, a.by) for p in a.experts]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(tables)
    assert len(weights) == len(tables), f"--weights: {len(weights)} weights for {len(tables)} tables"
    layers = set(tables[0])
    assert all(set(t) == layers for t in tables), "the tables cover different layers"
    per = {}
    for layer in layers:
        experts = set(tables[0][layer])
        assert all(set(t[layer]) == experts for t in tables), f"layer {layer}: the tables list different experts"
        if len(tables) == 1:
            per[layer] = [(-tables[0][layer][e], e) for e in experts]
            continue
        totals = [sum(t[layer].values()) or 1.0 for t in tables]
        per[layer] = [(-sum(w * t[layer][e] / n for w, t, n in zip(weights, tables, totals)), e) for e in experts]
    counts = hot_counts(a.hot, set(per))
    force = read_ids(a.force_hot) if a.force_hot else {}
    hot, cold = {}, {}
    for layer in sorted(counts):
        order = [e for _v, e in sorted(per[layer])]
        n = counts[layer]
        assert sorted(order) == list(range(len(order))), f"layer {layer}: the table lacks experts or repeats them"
        assert 0 < n < len(order), f"--hot {n}: needs 1 to {len(order) - 1} of layer {layer}'s {len(order)}"
        # forced ids are hot whatever their rank (in rank order, like the rest); the other places go by rank
        fl = set(force.get(layer, []))
        assert len(fl) == len(force.get(layer, [])) and fl <= set(order), f"layer {layer}: forced ids repeat or are unknown"
        assert len(fl) <= n, f"layer {layer}: {len(fl)} forced ids, {n} hot places"
        free = [e for e in order if e not in fl][:n - len(fl)]
        keep = fl | set(free)
        hot[layer] = [e for e in order if e in keep]
        cold[layer] = [e for e in order if e not in keep]
    unknown = sorted(set(force) - set(counts))
    assert not unknown, f"--force-hot names layers {unknown} that are not tiered"
    os.makedirs(a.out_dir, exist_ok=True)
    write_ids(os.path.join(a.out_dir, "hot.tsv"), hot)
    write_ids(os.path.join(a.out_dir, "cold.tsv"), cold)
    sizes = sorted({len(v) for v in hot.values()})
    print(f"{len(hot)} of {len(per)} layers tiered: {'/'.join(map(str, sizes))} hot, by {a.by}"
          f"{f', {sum(len(v) for v in force.values())} forced' if force else ''}")


class Source:
    """The expert tensors of a GGUF (one or more shards) by name, with the expert ids its expert axis holds."""

    def __init__(self, paths, ids_path):
        self.hs = [Header(p) for p in paths]
        self.where = {}  # name -> (header, dims, type, offset)
        for h in self.hs:
            for name, dims, typ, off in h.tensors:
                self.where[name] = (h, dims, typ, off)
        self.ids = read_ids(ids_path) if ids_path else None

    def has(self, name):
        return name in self.where

    def slabs(self, name, layer, want):
        """(header, dims, type, data start, slab bytes, [position of each wanted id on the expert axis])."""
        assert name in self.where, f"{self.hs[0].path}: no tensor {name}"
        h, dims, typ, off = self.where[name]
        held = self.ids[layer] if self.ids is not None else list(range(dims[-1]))
        assert len(held) == dims[-1], f"{name}: {dims[-1]} experts in {h.path}, its id list names {len(held)}"
        pos = {e: i for i, e in enumerate(held)}
        missing = [e for e in want if e not in pos]
        assert not missing, f"{name}: {h.path} does not hold experts {missing}"
        return h, dims, typ, h.data0 + off, nbytes(name, dims, typ) // dims[-1], [pos[e] for e in want]


def check_tiers(hot, cold, layers, n_expert, n_used):
    assert set(hot) == set(cold), "the hot and cold lists name different layers"
    missing = sorted(set(hot) - set(layers))
    assert not missing, f"layers {missing} have no routed expert tensors"
    for l in hot:
        assert sorted(hot[l] + cold[l]) == list(range(n_expert)), \
            f"layer {l}: the hot and cold ids are not each of the {n_expert} experts once"
        assert (n_used or 0) <= len(hot[l]) and cold[l], f"layer {l}: {len(hot[l])} hot, {len(cold[l])} cold"


def plan(a):
    """Per base shard: (path, new keys, [(name, dims, type, slot, how)]); how = ("copy", file, start) |
    ("slabs", file, start, slab, positions) | ("bytes", data)."""
    hs = [Header(p) for p in a.shards]
    count = hs[0].value("split.count")
    assert count in (None, len(hs)), f"{len(hs)} shards given, split.count is {count}"
    n_expert, n_used = first_value(hs, "expert_count"), first_value(hs, "expert_used_count")
    assert n_expert, "no <arch>.expert_count in the first shard"
    hot, cold = read_ids(a.hot_ids), read_ids(a.cold_ids)
    layers = {}
    for h in hs:
        for name, dims, _typ, _off in h.tensors:
            m = EXPS.match(name)
            if m:
                layers.setdefault(int(m[1]), []).append(name)
    check_tiers(hot, cold, layers, n_expert, n_used)
    hot_src = Source(a.hot, a.hot_in) if a.hot else None
    cold_src = Source(a.cold, a.cold_in)
    cold_only = re.compile(a.cold_only) if a.cold_only else None
    assert not a.cold_rest or cold_only, "--cold-rest needs --cold-only"
    rest_src = Source(a.cold_rest, None) if a.cold_rest else None
    out = []
    for i, h in enumerate(hs):
        entries = []
        for name, dims, typ, off in h.tensors:
            m = EXPS.match(name)
            layer = int(m[1]) if m else None
            if m and layer in hot:
                assert m[3] in SUFFIXES, f"{name}: per-expert {m[3]} is indexed by router id, which a tier does not keep"
                assert dims[-1] == n_expert, f"{name}: last dimension {dims[-1]}, expert_count {n_expert}"
                # each tier takes its weights, and its scales if its source has them, from its own source
                if hot_src is None:
                    hf, hd, ht, hstart, hslab = h, dims, typ, h.data0 + off, nbytes(name, dims, typ) // n_expert
                    hot_slabs = (hd, ht, ("slabs", hf.path, hstart, hslab, hot[layer]), hslab)
                elif m[3] == "weight" or hot_src.has(name):
                    hf, hd, ht, hstart, hslab, hpos = hot_src.slabs(name, layer, hot[layer])
                    hot_slabs = (hd, ht, ("slabs", hf.path, hstart, hslab, hpos), hslab)
                else:
                    hot_slabs = None
                cold_slabs = None
                if cold_only and not cold_only.search(name) and rest_src is None:
                    cslab = nbytes(name, dims, typ) // n_expert
                    cold_slabs = (dims, typ, ("slabs", h.path, h.data0 + off, cslab, cold[layer]), cslab)
                elif cold_only and not cold_only.search(name):
                    if m[3] == "weight" or rest_src.has(name):
                        rf, rd, rt, rstart, rslab, rpos = rest_src.slabs(name, layer, cold[layer])
                        cold_slabs = (rd, rt, ("slabs", rf.path, rstart, rslab, rpos), rslab)
                elif m[3] == "weight" or cold_src.has(name):
                    cf, cd, ct, cstart, cslab, cpos = cold_src.slabs(name, layer, cold[layer])
                    cold_slabs = (cd, ct, ("slabs", cf.path, cstart, cslab, cpos), cslab)
                nh, nc = len(hot[layer]), len(cold[layer])
                for tname, n, tier in ((name, nh, hot_slabs), (cold_name(name), nc, cold_slabs)):
                    if tier:
                        td, tt, how, slab = tier
                        assert td[:-1] == dims[:-1], f"{name}: rows {dims[:-1]} in the base, {td[:-1]} in a tier's source"
                        entries.append((tname, dims[:-1] + [n], tt, n * slab, how))
                continue
            entries.append((name, dims, typ, h.slot[name], ("copy", h.path, h.data0 + off)))
            gm = re.match(r"^blk\.(\d+)\.ffn_gate_inp\.weight$", name)
            if gm and int(gm[1]) in hot:
                l = int(gm[1])
                tier = [0.0] * n_expert
                for s, e in enumerate(hot[l]):
                    tier[e] = float(s)
                for s, e in enumerate(cold[l]):
                    tier[e] = float(len(hot[l]) + s)
                data = struct.pack(f"<{n_expert}f", *tier)
                entries.append((f"blk.{l}.ffn_exps_tier.weight", [n_expert], F32, len(data), ("bytes", data)))
        out.append([h, entries])
    added = sum(len(es) for _h, es in out) - sum(len(h.tensors) for h in hs)
    layer_of = lambda n: int(re.match(r"^blk\.(\d+)\.", n)[1])
    tiered = {layer_of(n) for h, es in out for n, *_x in es if n.endswith("_exps_cold.weight")}
    assert tiered == set(hot), f"layers {sorted(set(hot) - tiered)}: no expert tensors to tier"
    routed = {layer_of(n) for h, es in out for n, *_x in es if n.endswith("ffn_exps_tier.weight")}
    assert routed == set(hot), f"layers {sorted(set(hot) - routed)}: no router ffn_gate_inp.weight for the tier map"
    # llama.cpp reads split.tensors.count from the first shard only; a later shard with no tensor to change keeps its
    # keys, so that --link-unchanged can link it (Flash-Next's n-gram table)
    result = []
    for i, (h, entries) in enumerate(out):
        bump = i == 0 or any(how[0] != "copy" for *_x, how in entries)
        kv = [(k, t, struct.pack(SCALAR[t], struct.unpack(SCALAR[t], raw)[0] + added)
               if k == "split.tensors.count" and bump else raw) for k, t, raw in h.kv]
        result.append((h, kv, entries))
    return result


def unchanged(h, kv, entries):
    return kv == h.kv and all(how[0] == "copy" for *_x, how in entries)


def cmd_tier(a):
    assert sys.byteorder == "little"
    files = {}
    total = [0, 0]
    for h, kv, entries in plan(a):
        path = os.path.join(a.out_dir, out_name(h.path, a.prefix))
        os.makedirs(a.out_dir, exist_ok=True)
        assert os.path.islink(path) or not (os.path.exists(path) and os.path.samefile(path, h.path)), \
            f"{path} is its own source"
        if unchanged(h, kv, entries) and a.link_unchanged:
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(os.path.realpath(h.path), path)
            print(f"{path} -> {os.path.realpath(h.path)}: nothing to change")
            continue
        tensors = [(name, dims, typ, 0, slot) for name, dims, typ, slot, _how in entries]
        with open(path + ".part", "wb") as dst:
            write_header(dst, h.version, kv, tensors, h.align)
            if a.headers_only:
                os.ftruncate(dst.fileno(), os.lseek(dst.fileno(), 0, 1) + sum(s + (-s) % h.align for *_x, s in tensors))
            else:
                for _name, _dims, _typ, slot, how in entries:
                    if how[0] == "bytes":
                        write_all(dst.fileno(), how[1] + b"\0" * ((-slot) % h.align))
                        continue
                    src = files.setdefault(how[1], open(how[1], "rb"))
                    if how[0] == "copy":
                        copy_range(src, dst, how[2], slot, h.align)
                    else:
                        _k, _p, start, slab, pos = how
                        for p, n in runs(pos):
                            copy_range(src, dst, start + p * slab, n * slab, 1)
                        write_all(dst.fileno(), b"\0" * ((-slot) % h.align))
        os.replace(path + ".part", path)
        before = sum(h.slot[t[0]] for t in h.tensors)
        after = sum(s + (-s) % h.align for *_x, s in tensors)
        total[0], total[1] = total[0] + before, total[1] + after
        n_cold = sum(1 for t in tensors if t[0].endswith("_exps_cold.weight"))
        print(f"{path}: {len(tensors)} tensors, {n_cold} cold, {os.path.getsize(path):,} bytes "
              f"({after - before:+,} of tensor data){' (header only)' if a.headers_only else ''}")
        sys.stdout.flush()
    print(f"tensor data {total[0]:,} -> {total[1]:,} bytes ({total[1] - total[0]:+,})")


def cmd_verify(a):
    planned = plan(a)
    assert len(a.new) == len(planned), f"{len(a.new)} new shards, {len(planned)} old ones"
    files = {}
    for (h, kv, entries), path in zip(planned, a.new):
        hn = Header(path)
        assert hn.kv == kv, f"{path}: keys differ from what tier writes"
        assert len(hn.tensors) == len(entries), f"{path}: {len(hn.tensors)} tensors, {len(entries)} expected"
        off = 0
        for (name, dims, typ, slot, _how), (nn, dn, tn, on) in zip(entries, hn.tensors):
            assert (name, dims, typ) == (nn, dn, tn), f"{path}: {nn} {dn} type {tn} where {name} {dims} type {typ}"
            assert on == off, f"{name}: offset {on}, {off} expected"
            off += slot + (-slot) % h.align
        assert hn.size == hn.data0 + off, f"{path}: {hn.size:,} bytes, {hn.data0 + off:,} expected"
        if a.headers_only:
            print(f"{path}: header ok against {h.path}")
            continue
        if os.path.samefile(path, h.path):  # a linked shard (--link-unchanged) is its source's bytes
            print(f"{path}: the same file as {h.path}")
            continue
        counts = {"copy": 0, "slabs": 0, "bytes": 0}
        with open(path, "rb") as fn:
            for (name, _d, _t, slot, how), (_n, _dn, _tn, on) in zip(entries, hn.tensors):
                pn = hn.data0 + on
                if how[0] == "bytes":
                    assert os.pread(fn.fileno(), len(how[1]), pn) == how[1], f"{name}: the tier map differs"
                elif how[0] == "copy":
                    src = files.setdefault(how[1], open(how[1], "rb"))
                    assert same(src, how[2], fn, pn, slot), f"{name}: bytes differ"
                else:
                    _k, p0, start, slab, pos = how
                    src = files.setdefault(p0, open(p0, "rb"))
                    k = 0
                    for p, n in runs(pos):
                        assert same(src, start + p * slab, fn, pn + k * slab, n * slab), \
                            f"{name}: the slabs of source positions {p}-{p + n - 1} differ"
                        k += n
                counts[how[0]] += 1
        print(f"{path}: {counts['copy']} tensors identical to {h.path}, {counts['slabs']} tiered from their sources "
              f"slab for slab, {counts['bytes']} tier maps right")
        sys.stdout.flush()


def cmd_extract(a):
    assert sys.byteorder == "little"
    hs = [Header(p) for p in a.shards]
    n_expert = first_value(hs, "expert_count")
    assert n_expert, "no <arch>.expert_count in the first shard"
    ids = read_ids(a.ids)
    counts = {len(v) for v in ids.values()}
    assert len(counts) == 1, f"layers list different counts {sorted(counts)}: llama.cpp takes one expert_count"
    n = counts.pop()
    for l, v in ids.items():
        assert len(set(v)) == n and all(0 <= e < n_expert for e in v), f"layer {l}: ids repeat or out of range"
    picked = []  # (header, name, dims, type, offset)
    for h in hs:
        for name, dims, typ, off in h.tensors:
            m = EXPS.match(name)
            if m and int(m[1]) in ids:
                assert m[3] in SUFFIXES, f"{name}: per-expert {m[3]} cannot be extracted"
                assert dims[-1] == n_expert, f"{name}: last dimension {dims[-1]}, expert_count {n_expert}"
                picked.append((h, name, dims, typ, off))
    assert {int(EXPS.match(p[1])[1]) for p in picked} == set(ids), "a listed layer has no routed expert tensors"
    k = arch_key(hs[0], "expert_count")
    kv = [(key, t, struct.pack(SCALAR[t], n) if key == k else raw) for key, t, raw in hs[0].kv if key not in SPLIT_KEYS]
    os.makedirs(a.out_dir, exist_ok=True)
    tensors = []
    for h, name, dims, typ, off in picked:
        slab = nbytes(name, dims, typ) // n_expert
        tensors.append((name, dims[:-1] + [n], typ, off, n * slab))
    path = os.path.join(a.out_dir, "experts.gguf")
    files = {}
    with open(path + ".part", "wb") as dst:
        write_header(dst, hs[0].version, kv, tensors, hs[0].align)
        for (h, name, dims, typ, off), (*_x, slot) in zip(picked, tensors):
            src = files.setdefault(h.path, open(h.path, "rb"))
            slab = nbytes(name, dims, typ) // n_expert
            for e, c in runs(ids[int(EXPS.match(name)[1])]):
                copy_range(src, dst, h.data0 + off + e * slab, c * slab, 1)
            write_all(dst.fileno(), b"\0" * ((-slot) % hs[0].align))
    os.replace(path + ".part", path)
    write_ids(os.path.join(a.out_dir, "ids.tsv"), ids)
    print(f"{path}: {len(tensors)} expert tensors of {len(ids)} layers, {n} experts each, {os.path.getsize(path):,} bytes")
    if not a.imatrix:
        return
    im = Header(a.imatrix)
    names = {p[1] for p in picked if p[1].endswith(".weight")}
    keep, dropped = [], 0
    for name, dims, typ, off in im.tensors:
        base, _dot, suffix = name.rpartition(".")
        if base in names and suffix in ("in_sum2", "in_sum", "counts"):
            assert typ == F32 and dims[-1] == n_expert, f"{a.imatrix}: {name} {dims} type {typ}"
            keep.append((name, dims, off, base))
        else:
            dropped += 1
    got = {b for *_x, b in keep}
    assert got == names, f"{a.imatrix}: no statistics for {sorted(names - got)[:4]}"
    tensors = [(name, dims[:-1] + [n], F32, off, nbytes(name, dims, F32) // n_expert * n) for name, dims, off, _b in keep]
    path = os.path.join(a.out_dir, "imatrix.gguf")
    with open(a.imatrix, "rb") as src, open(path + ".part", "wb") as dst:
        write_header(dst, im.version, im.kv, tensors, im.align)
        for (name, dims, off, base), (*_x, slot) in zip(keep, tensors):
            slab = nbytes(name, dims, F32) // n_expert
            for e, c in runs(ids[int(EXPS.match(base)[1])]):
                copy_range(src, dst, im.data0 + off + e * slab, c * slab, 1)
            write_all(dst.fileno(), b"\0" * ((-slot) % im.align))
    os.replace(path + ".part", path)
    print(f"{path}: {len(tensors)} statistics of {len(names)} tensors cut to {n} experts, {dropped} other entries dropped")


def tier_args(p):
    p.add_argument("--hot-ids", required=True, help="<layer>\\t<id>,... per tiered layer (split's hot.tsv)")
    p.add_argument("--cold-ids", required=True, help="the same for the cold tier (split's cold.tsv)")
    p.add_argument("--cold", nargs="+", required=True, help="the file (shards) the cold slabs come from")
    p.add_argument("--cold-in", help="the ids --cold's expert axis holds, per layer (extract's ids.tsv)")
    p.add_argument("--hot", nargs="+", help="the file the hot slabs come from (default: the base shards)")
    p.add_argument("--hot-in", help="the ids --hot's expert axis holds, per layer")
    p.add_argument("--cold-only", help="regex: only cold tensors whose name matches come from --cold, the others "
                   "from the base's own slabs (e.g. 'ffn_(gate|up|gate_up)_exps' keeps the base's down projections)")
    p.add_argument("--cold-rest", nargs="+", help="with --cold-only: the file (all experts, in id order) the other cold "
                   "tensors come from instead of the base, e.g. another copy of the same experts on a faster disk")
    p.add_argument("--headers-only", action="store_true", help="write (or check) the headers and sizes only")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--experts", required=True, action="append",
                   help="per layer and expert counts (columns layer, expert and --by); several: ranked by their shares")
    s.add_argument("--weights", help="with several --experts: a weight per table, comma-separated (default all 1)")
    s.add_argument("--hot", required=True, help="hot experts per layer: N, or <a>-<b>:N,<c>:N for some layers")
    s.add_argument("--force-hot", help="<layer>\\t<id>,... per layer, ids that are hot whatever their rank")
    s.add_argument("--by", default="routings", help="the column to rank by (default routings)")
    s.add_argument("--out-dir", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ids", required=True, help="<layer>\\t<id>,... per layer, the experts to extract in that order")
    e.add_argument("--imatrix", help="an importance matrix (GGUF) to cut the same way")
    e.add_argument("--out-dir", required=True)
    e.add_argument("shards", nargs="+")
    t = sub.add_parser("tier")
    tier_args(t)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--prefix", help="new file name before -0000k-of-0000N.gguf")
    t.add_argument("--link-unchanged", action="store_true", help="a shard with nothing to change becomes a link")
    t.add_argument("shards", nargs="+")
    v = sub.add_parser("verify")
    tier_args(v)
    v.add_argument("shards", nargs="+")
    v.add_argument("--new", nargs="+", required=True)
    a = ap.parse_args()
    {"split": cmd_split, "extract": cmd_extract, "tier": cmd_tier, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
