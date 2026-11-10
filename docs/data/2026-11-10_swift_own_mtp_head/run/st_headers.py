#!/usr/bin/env python3
# st_headers.py DIR PATTERN: name, dtype, shape, shard and byte range of every safetensors tensor whose name matches.
import json, os, re, struct, sys
d, pat = sys.argv[1], re.compile(sys.argv[2])
wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
heads = {}
for name in sorted(k for k in wm if pat.search(k)):
    sh = wm[name]
    if sh not in heads:
        with open(os.path.join(d, sh), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            heads[sh] = (8 + n, json.loads(f.read(n)))
    base, h = heads[sh]
    t = h[name]
    print(name, t["dtype"], t["shape"], sh, base + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0], sep="\t")
