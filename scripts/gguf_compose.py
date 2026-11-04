#!/usr/bin/env python3
# gguf_compose.py: writes a GGUF whose chosen tensors come byte for byte from another file of the same model (the
# donor) and every other tensor byte for byte from the base. Either input may be split, and the output keeps the
# base's shards or is split again. Made for Q218's Pascal spines of Qwen3.8-Flash-Next: a Swift file with its dense
# matrices taken from a Q8_0 of the same model, its routed experts and n-gram table left as they are.
#   gguf_compose.py compose --base SHARD... --donor SHARD... --take PATTERN [--take PATTERN ...] --out-dir <dir>
#       [--prefix <name>] [--layout base|donor|one|<GB>] [--isolate <regex>] [--link-unchanged] [--headers-only]
#     A PATTERN is a tensor name, `glob:<pattern>` (fnmatch on the whole name) or `re:<regex>` (re.search); each
#     must match at least one tensor. Base and donor must be the same model: the same architecture and the same tensor
#     names with the same shapes; a taken tensor keeps the donor's type and bytes.
#     --layout base (the default) keeps the base's shards, each tensor in its shard and in its order; donor follows the
#     donor's shards; one writes a single file; a number re-splits the base's tensor order into shards of at most
#     that many GB (a bigger tensor alone). --isolate then moves the tensors that match it into one last shard of
#     their own (as scripts/gguf_reshard.py does). With the base's shards each shard keeps its own key/value pairs;
#     otherwise shard 1 gets the base's first key/value pairs and the others only the split keys, with split.no,
#     split.count and split.tensors.count set for the new layout (a single file has none).
#     Shards are named <prefix>-0000k-of-0000N.gguf (a single file <prefix>.gguf); the prefix defaults to the base's.
#     Each tensor keeps its source slot (its bytes plus the padding up to the next tensor), padded to the base's
#     alignment, so a base shard with nothing taken comes out byte for byte as it was; --link-unchanged writes a
#     symbolic link to such a shard instead (Flash-Next's 28.8 GB table shard is not copied). --headers-only writes
#     the headers and sizes each file as if the data were there (a fit check without the data).
#   gguf_compose.py verify --base SHARD... --donor SHARD... --take PATTERN... --new SHARD...
#     checks the new shards against both inputs: every tensor of the base once, a taken tensor with the donor's
#     type and bytes and every other with the base's, the base's first key/value pairs, and split keys that number
#     the shards. It reads every tensor, except where a new shard is its source file itself at the same offset (a
#     shard --link-unchanged linked): those bytes are the source's by construction.
# Standard library only.
import argparse, fnmatch, os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reshard import CHUNK, Header, copy_range, write_all  # noqa: E402

SPLIT = ("split.no", "split.count", "split.tensors.count")


class Model:
    """The shards of one (split) GGUF, and each tensor's shard and info by name."""

    def __init__(self, paths):
        self.shards = [Header(p) for p in paths]
        count = self.shards[0].value("split.count")
        assert count in (None, len(paths)), f"{paths[0]}: {len(paths)} shards given, split.count is {count}"
        nos = [h.value("split.no", 0) for h in self.shards]
        assert nos == list(range(len(paths))), f"{paths[0]}: shards out of order (split.no {nos})"
        self.where = {}  # name -> (header, dims, type, offset)
        for h in self.shards:
            for name, dims, typ, off in h.tensors:
                assert name not in self.where, f"{h.path}: {name} twice"
                self.where[name] = (h, dims, typ, off)
        total = self.shards[0].value("split.tensors.count")
        assert total in (None, len(self.where)), f"{paths[0]}: split.tensors.count {total}, {len(self.where)} found"
        self.arch = self.shards[0].value("general.architecture")


def text(raw):
    return raw[8:].decode(errors="replace") if isinstance(raw, bytes) else str(raw)


def same_model(base, donor):
    assert base.arch == donor.arch, f"architectures differ: {text(base.arch)} and {text(donor.arch)}"
    a = {n: w[1] for n, w in base.where.items()}
    b = {n: w[1] for n, w in donor.where.items()}
    odd = sorted(a.keys() ^ b.keys())
    assert not odd, f"tensor names differ ({len(odd)}), e.g. {odd[0]}"
    bad = [n for n in a if a[n] != b[n]]
    assert not bad, f"shapes differ ({len(bad)}), e.g. {bad[0]}: {a[bad[0]]} and {b[bad[0]]}"


def taken(base, patterns):
    """The base's tensor names that match any pattern; a pattern that matches none stops the tool."""
    names, out = list(base.where), set()
    for p in patterns:
        if p.startswith("re:"):
            hit = [n for n in names if re.search(p[3:], n)]
        elif p.startswith("glob:"):
            hit = [n for n in names if fnmatch.fnmatchcase(n, p[5:])]
        else:
            hit = [p] if p in base.where else []
        assert hit, f"--take {p!r} matches no tensor"
        out.update(hit)
    return out


def data_order(h):
    return [t[0] for t in sorted(h.tensors, key=lambda t: t[3])]


def plan(a, base, donor, take):
    """The output shards: (tensor names in header order, the same in data order)."""
    if a.layout == "base":
        shards = [([t[0] for t in h.tensors], data_order(h)) for h in base.shards]
    elif a.layout == "donor":
        shards = [([t[0] for t in h.tensors], data_order(h)) for h in donor.shards]
    else:
        order = [n for h in base.shards for n in data_order(h)]
        groups = [[]]
        if a.layout != "one":
            limit, used = float(a.layout) * 1e9, 0
            assert limit > 0, "--layout: base, donor, one or a size in GB"
        for n in order:
            if a.layout != "one":
                s = slot(base, donor, take, n)
                if groups[-1] and used + s > limit:
                    groups.append([])
                    used = 0
                used += s
            groups[-1].append(n)
        shards = [(grp, grp) for grp in groups]
    if a.isolate:
        pat = re.compile(a.isolate)
        moved = [n for _hdr, data in shards for n in data if pat.search(n)]
        assert moved, f"--isolate {a.isolate!r} matches no tensor"
        shards = [([n for n in hdr if not pat.search(n)], [n for n in data if not pat.search(n)]) for hdr, data in shards]
        shards = [s for s in shards if s[1]] + [(moved, moved)]
    return shards


def source(base, donor, take, name):
    return (donor if name in take else base).where[name]


def slot(base, donor, take, name):
    h = source(base, donor, take, name)[0]
    return h.slot[name]


def split_kv(kv, no, count, total):
    """kv with the split keys set for shard `no` of `count` (replaced where they are, else appended), or removed
    for a single file."""
    if count == 1:
        return [x for x in kv if x[0] not in SPLIT]
    new = {"split.no": (2, struct.pack("<H", no)), "split.count": (2, struct.pack("<H", count)),
           "split.tensors.count": (5, struct.pack("<i", total))}
    out = [(k, *new.pop(k)) if k in new else (k, t, raw) for k, t, raw in kv]
    return out + [(k, t, raw) for k, (t, raw) in new.items()]


def write_shard(path, version, kv, header, data, base, donor, take, align, headers_only):
    offs, off = {}, 0
    for n in data:
        offs[n] = off
        s = slot(base, donor, take, n)
        off += s + (-s) % align
    buf = bytearray()
    wstr = lambda s: buf.extend(struct.pack("<Q", len(s)) + s)
    buf += b"GGUF" + struct.pack("<IQQ", version, len(header), len(kv))
    for key, t, raw in kv:
        wstr(key.encode())
        buf += struct.pack("<I", t) + raw
    for n in header:
        _h, dims, typ, _off = source(base, donor, take, n)
        wstr(n.encode())
        buf += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, offs[n])
    buf += b"\0" * ((-len(buf)) % align)
    with open(path + ".part", "wb") as dst:
        write_all(dst.fileno(), bytes(buf))
        if headers_only:
            os.ftruncate(dst.fileno(), len(buf) + off)
        else:
            files = {}
            for n in data:
                h, _dims, _typ, src_off = source(base, donor, take, n)
                f = files.setdefault(h.path, open(h.path, "rb"))
                copy_range(f, dst, h.data0 + src_off, h.slot[n], align)
            for f in files.values():
                f.close()
    os.replace(path + ".part", path)


def names(prefix, count):
    return [f"{prefix}.gguf"] if count == 1 else [f"{prefix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def cmd_compose(a):
    base, donor = Model(a.base), Model(a.donor)
    same_model(base, donor)
    take = taken(base, a.take)
    shards = plan(a, base, donor, take)
    prefix = a.prefix or re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", os.path.basename(a.base[0]))
    count, total, align = len(shards), len(base.where), base.shards[0].align
    kept = a.layout == "base" and not a.isolate  # each shard keeps its own key/value pairs
    os.makedirs(a.out_dir, exist_ok=True)
    srcs = {f(p) for p in a.base + a.donor for f in (os.path.abspath, os.path.realpath)}
    for i, ((header, data), name) in enumerate(zip(shards, names(prefix, count))):
        path = os.path.join(a.out_dir, name)
        assert os.path.abspath(path) not in srcs and (os.path.islink(path) or os.path.realpath(path) not in srcs), \
            f"{path} is one of the inputs"  # a link from an earlier run is replaced, not written through
        if kept:
            h = base.shards[i]
            kv = h.kv
            if a.link_unchanged and not take & set(data):
                if os.path.lexists(path):
                    os.remove(path)
                os.symlink(os.path.realpath(h.path), path)
                print(f"{path} -> {os.path.realpath(h.path)}: nothing taken")
                continue
        else:
            kv = split_kv(base.shards[0].kv if i == 0 else [], i, count, total)
        write_shard(path, base.shards[0].version, kv, header, data, base, donor, take, align, a.headers_only)
        n = len(take & set(data))
        print(f"{path}: {len(data)} tensors, {n} from the donor, {os.path.getsize(path):,} bytes")
        sys.stdout.flush()
    before = sum(base.where[n][0].slot[n] for n in take)
    after = sum(donor.where[n][0].slot[n] for n in take)
    print(f"{len(take)} tensors from the donor: {before:,} -> {after:,} bytes ({after - before:+,})")


def cmd_verify(a):
    base, donor, new = Model(a.base), Model(a.donor), Model(a.new)
    same_model(base, donor)
    take = taken(base, a.take)
    odd = sorted(new.where.keys() ^ base.where.keys())
    assert not odd, f"the new file's tensors differ from the base's ({len(odd)}), e.g. {odd[0]}"
    plain = lambda kv: [x for x in kv if x[0] not in SPLIT]
    assert plain(new.shards[0].kv) == plain(base.shards[0].kv), "shard 1's key/value pairs differ from the base's"
    count = len(new.shards)
    for i, h in enumerate(new.shards):
        want = (i, count, len(base.where)) if count > 1 else (None, None, None)
        got = tuple(h.value(k) for k in SPLIT)
        assert got == want, f"{h.path}: split keys {got}, want {want}"
        assert h.align == base.shards[0].align, f"{h.path}: alignment {h.align}"
    files, n_take, n_linked = {}, 0, 0
    for name, (hn, dims, typ, off) in new.where.items():
        hs, sdims, styp, soff = source(base, donor, take, name)
        assert dims == sdims and typ == styp, f"{name}: shape {dims} type {typ}, want {sdims} type {styp}"
        left = hs.slot[name]
        assert hn.slot[name] >= left, f"{name}: {hn.slot[name]} bytes in the new file, {left} in its source"
        n_take += name in take
        if os.path.realpath(hs.path) == os.path.realpath(hn.path) and hs.data0 + soff == hn.data0 + off:
            n_linked += 1  # the same bytes of the same file (a linked shard): nothing to read
            continue
        fs, fn = (files.setdefault(p, open(p, "rb")) for p in (hs.path, hn.path))
        ps, pn = hs.data0 + soff, hn.data0 + off
        while left:
            k = min(left, CHUNK)
            assert os.pread(fs.fileno(), k, ps) == os.pread(fn.fileno(), k, pn), f"{name}: bytes differ"
            left, ps, pn = left - k, ps + k, pn + k
    print(f"{len(new.where)} tensors in {count} shards: {n_take} identical to the donor's, "
          f"{len(new.where) - n_take} to the base's ({n_linked} in a shard that is the source file itself)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("compose", "verify"):
        s = sub.add_parser(cmd)
        s.add_argument("--base", nargs="+", required=True)
        s.add_argument("--donor", nargs="+", required=True)
        s.add_argument("--take", action="append", required=True, help="a name, glob:<pattern> or re:<regex>")
    c = sub.choices["compose"]
    c.add_argument("--out-dir", required=True)
    c.add_argument("--prefix", help="the new file name before -0000k-of-0000N.gguf (default: the base's)")
    c.add_argument("--layout", default="base", help="base (default), donor, one, or a shard size in GB")
    c.add_argument("--isolate", help="regex of tensors that get one last shard of their own")
    c.add_argument("--link-unchanged", action="store_true",
                   help="a base shard with nothing taken becomes a symbolic link to it (layout base)")
    c.add_argument("--headers-only", action="store_true", help="write the headers and size the files, copy no data")
    sub.choices["verify"].add_argument("--new", nargs="+", required=True)
    a = ap.parse_args()
    cmd_compose(a) if a.cmd == "compose" else cmd_verify(a)


if __name__ == "__main__":
    main()
