#!/usr/bin/env python3
# bytes.py --experts experts.tsv.gz --out bytes.tsv NAME=FIRST_SHARD...: the weight bytes a decoded token reads from
# each file, with a tiered file's experts weighted by how often the router picks them (Q232's agent-text routings).
# Per layer a token reads its expert_used_count experts: in an untiered file every expert of a layer has the same size,
# so the layer costs used x slab; in a tiered one the hot and cold slabs differ, so it costs
# used x (p_hot x hot slab + p_cold x cold slab), p_cold the cold experts' share of the layer's routings (read from the
# file's own ffn_exps_tier map). A cut file (scripts/gguf_prune.py) has no cold tier: its picks all go to kept experts.
# The dense part is every other tensor but the token embeddings and the n-gram table (a token reads one row of each);
# tensor_data_bytes_without_embeddings is every tensor's bytes but those two (the weights that sit on the cards).
# This engine also reads slot 0 of a tier for the other tier's picks: `upper` adds one hot and one cold slab per layer,
# the most it can cost (slot 0 is the tier's most picked expert, often a pick already). Standard library only.
import argparse, gzip, os, struct, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "..", "..", "scripts"))
from gguf_reshard import Header  # noqa: E402
from gguf_prune import nbytes  # noqa: E402

ROW_ONLY = ("token_embd.", "per_layer_token_embd.")


def shards(first):
    stem = first[: first.rindex("-00001-of-")]
    count = int(first[first.rindex("-of-") + 4:].split(".")[0])
    return [f"{stem}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+", help="NAME=first shard")
    a = ap.parse_args()
    routings = {}
    with (gzip.open(a.experts, "rt") if a.experts.endswith(".gz") else open(a.experts)) as f:
        head = f.readline().rstrip("\n").split("\t")
        li, ei, ri = head.index("layer"), head.index("expert"), head.index("routings")
        for line in f:
            c = line.rstrip("\n").split("\t")
            routings.setdefault(int(c[li]), {})[int(c[ei])] = float(c[ri])
    rows = []
    for spec in a.files:
        name, first = spec.split("=", 1)
        hs = [Header(p) for p in shards(first)]
        arch = hs[0].value("general.architecture")[8:].decode()
        used = hs[0].value(f"{arch}.expert_used_count")
        dense = exp_bytes = upper = data = rows_only = 0
        per_layer = {}  # layer -> bytes per hot and per cold expert, the hot tier's count, the tier map
        for h in hs:
            for tname, dims, typ, off in h.tensors:
                size = nbytes(tname, dims, typ)
                data += h.slot[tname]
                parts = tname.split(".")
                d = per_layer.setdefault(int(parts[1]), {"hot": 0, "cold": 0, "n_hot": 0, "map": None}) \
                    if parts[0] == "blk" else None
                if d is not None and parts[2].endswith(("_exps", "_exps_cold")):
                    tier = "cold" if parts[2].endswith("_cold") else "hot"
                    d[tier] += size // dims[-1]
                    if tier == "hot":
                        d["n_hot"] = dims[-1]
                elif d is not None and parts[2] == "ffn_exps_tier":
                    with open(h.path, "rb") as f:
                        f.seek(h.data0 + off)
                        d["map"] = struct.unpack(f"<{dims[0]}f", f.read(4 * dims[0]))
                elif tname.startswith(ROW_ONLY):
                    rows_only += h.slot[tname]
                else:
                    dense += size
        cold_share = []
        for layer, d in sorted(per_layer.items()):
            if not d["hot"]:
                continue
            if d["map"] is None:
                exp_bytes += used * d["hot"]
                upper += used * d["hot"]
                continue
            r = routings[layer]
            p_cold = sum(r[e] for e, v in enumerate(d["map"]) if v >= d["n_hot"]) / sum(r.values())
            cold_share.append(p_cold)
            exp_bytes += used * ((1 - p_cold) * d["hot"] + p_cold * d["cold"])
            upper += used * ((1 - p_cold) * d["hot"] + p_cold * d["cold"]) + d["hot"] + d["cold"]
        layers = sum(1 for d in per_layer.values() if d["hot"])
        rows.append((name, data - rows_only, dense, exp_bytes, upper,
                     sum(cold_share) / len(cold_share) if cold_share else 0.0, layers))
    with open(a.out, "w") as f:
        f.write("file\ttensor_data_bytes_without_embeddings\tdense_bytes_per_token\texpert_bytes_per_token\ttotal_bytes_per_token"
                "\ttotal_upper_bytes_per_token\tcold_routing_share_pct\tlayers\n")
        for name, data, dense, eb, up, share, nl in rows:
            f.write(f"{name}\t{data}\t{dense}\t{eb:.0f}\t{dense + eb:.0f}\t{dense + up:.0f}\t{100 * share:.2f}\t{nl}\n")
    with open(a.out) as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main()
