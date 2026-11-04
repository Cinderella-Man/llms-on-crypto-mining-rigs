#!/usr/bin/env python3
"""sizes.tsv: per file, the bytes of each tensor group and of what the cards hold (every tensor but the n-gram table
and the token embeddings, which llama.cpp keeps on the host), beside the file it was made from. Headers only.
  python3 -I sizes.py <name>=<first shard>[:<name of its input>] ... > sizes.tsv
Standard library only; gguf_reshard.py from the repository's scripts/ (next to this file or in ../../../../scripts).
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header  # noqa: E402

GROUPS = [("hyper-connections", r"hc_"), ("router", r"ffn_gate_inp"), ("experts", r"_exps\."),
          ("dense", r"^blk\.\d+\.(attn_(q|k|v|qkv|gate|output)|ssm_out|ffn_(gate|up|down)_shexp)\.weight$|^output\.weight$"),
          ("host", r"^(per_layer_token_embd|token_embd)\.weight$"), ("other", r"")]


def sizes(first):
    out = dict.fromkeys([g for g, _p in GROUPS], 0)
    for p in sorted(glob.glob(re.sub(r"-00001-of-(\d{5})\.gguf$", r"-0000?-of-\1.gguf", first))):
        h = Header(p)
        for name, _dims, _typ, _off in h.tensors:
            out[next(g for g, pat in GROUPS if re.search(pat, name))] += h.slot[name]
    out["cards"] = sum(v for g, v in out.items() if g != "host")
    return out


def main():
    rows = {}
    keys = [g for g, _p in GROUPS] + ["cards"]
    print("\t".join(["file", "input"] + [f"{k} bytes" for k in keys] + ["cards change"]))
    for arg in sys.argv[1:]:
        name, rest = arg.split("=", 1)
        first, _, src = rest.partition(":")
        rows[name] = sizes(first)
        change = rows[name]["cards"] - rows[src]["cards"] if src else ""
        print("\t".join([name, src] + [str(rows[name][k]) for k in keys] + [f"{change:+d}" if src else ""]))


if __name__ == "__main__":
    main()
