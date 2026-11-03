#!/usr/bin/env python3
# sizes.py: what the retype changed, per shard: the file sizes before and after, and the tensors converted from each
# type to each type with their bytes before and after. Header only. Standard library and scripts/gguf_reshard.py.
#   SCRIPTS=<repo>/scripts python3 -I sizes.py <label> <first old shard> <first new shard> > sizes.tsv
import glob, math, os, re, sys

sys.path.insert(0, os.environ["SCRIPTS"])
import gguf_reshard as g  # noqa: E402

NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16"}
size = lambda n, t: n // 32 * 34 if t == 8 else n * {0: 4, 1: 2, 30: 2}[t]
shards = lambda p: sorted(glob.glob(re.sub(r"-00001-of-(\d{5})\.gguf$", r"-0000?-of-\1.gguf", p)))
label, olds, news = sys.argv[1], shards(sys.argv[2]), shards(sys.argv[3])
assert len(olds) == len(news) > 0
print("\t".join(["file", "shard", "tensors", "old_bytes", "new_bytes", "change", "converted"]))
for i, (po, pn) in enumerate(zip(olds, news), 1):
    ho, hn = g.Header(po), g.Header(pn)
    conv = {}
    for (name, dims, to_, _o), (_n, _d, tn, _p) in zip(ho.tensors, hn.tensors):
        if to_ != tn:
            n = math.prod(dims)
            c = conv.setdefault(f"{NAMES[to_]}->{NAMES[tn]}", [0, 0, 0])
            c[0], c[1], c[2] = c[0] + 1, c[1] + size(n, to_), c[2] + size(n, tn)
    so, sn = os.path.getsize(po), os.path.getsize(pn)
    desc = "; ".join(f"{k}: {n} tensors, {b:,} -> {a:,} bytes" for k, (n, b, a) in sorted(conv.items())) or "none"
    print("\t".join([label, str(i), str(len(ho.tensors)), str(so), str(sn), str(sn - so), desc]))
