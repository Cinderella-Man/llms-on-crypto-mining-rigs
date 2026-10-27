#!/usr/bin/env python3
"""E1 of docs/research/notes/slq.md: Ornith-1.5-122B's N2P/N3P mix allocated per 8-layer group.

File c is the starting mix: routed experts' gate/up N2P and down N3P, every other quantized tensor Q5_0 (attention
k/v, output head and token embedding Q8_0). Each of 18 switches moves one class of one 8-layer group to another
file's format: expert gate/up to N3P and expert down to N2P from file d (c's recipe with the expert formats swapped),
every other tensor of the group to Q8_0 from file b (c's experts, everything else Q8_0). Each switch is scored alone
(KLD against the Q8_0's bases); the costs are assumed to add, and the best set of switches within a byte budget is
found by trying all 2^18 sets.

  specs                                    the 18 switches: name, class, layers, donor, regex
  compose C B D OUT ITEM...                compose_gguf.py (next to this script) from c with the switches ITEM applied
  deltas C B D OUT.tsv                     bytes each switch adds on the cards, from the three files' headers, plus
                                           file c's bytes on the cards (token embedding and MTP block excluded)
  collect OUT.tsv NAME=LOGPREFIX...        mean KLD and same top token (with their ±) per text from llama-perplexity
                                           logs LOGPREFIX.<text>.kld.log
  solve DELTAS.tsv KLD.tsv OUT.tsv [--budgets B,...] [--texts T,...] [--base NAME]
                                           per budget (bytes on the cards relative to file c), the set of switches
                                           with the least predicted damage: the sum over texts of each switch's KLD
                                           change relative to the base's KLD on that text
Needs gguf-py that knows N2P/N3P (PYTHONPATH) for compose and deltas, numpy for deltas and solve.
"""
import os
import re
import subprocess
import sys

GROUPS = [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 47)]
CLASSES = [  # name, tensors within a block, donor
    ("gu", r"ffn_(?:gate|up)_exps\.weight$", "d"),
    ("dn", r"ffn_down_exps\.weight$", "d"),
    ("dense", r"(?!ffn_(?:gate|up|down)_exps\.)", "b"),
]


def switches():
    out = []
    for cls, pat, donor in CLASSES:
        for g, (lo, hi) in enumerate(GROUPS):
            layers = "|".join(str(i) for i in range(lo, hi + 1))
            out.append((f"{cls}{g}", cls, f"{lo}-{hi}", donor, rf"^blk\.(?:{layers})\.{pat}"))
    return out


def by_name():
    return {s[0]: s for s in switches()}


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def cmd_specs(_):
    for s in switches():
        print("\t".join(s))


def cmd_compose(args):
    c, b, d, out, items = args[0], args[1], args[2], args[3], args[4:]
    donors = {"b": b, "d": d}
    sw = by_name()
    rules = [f"{sw[i][4]}={donors[sw[i][3]]}" for i in items]
    compose = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compose_gguf.py")
    subprocess.run([sys.executable, compose, c, out, *rules], check=True)


def cmd_deltas(args):
    import gguf
    c, b, d, out = args
    sizes = {}
    for key, path in (("c", c), ("b", b), ("d", d)):
        sizes[key] = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(path).tensors}
    mtp = {m.group(1) for n in sizes["c"] for m in [re.match(r"blk\.(\d+)\.nextn\.", n)] if m}
    on_card = sum(v for n, v in sizes["c"].items()
                  if n != "token_embd.weight" and not any(n.startswith(f"blk.{i}.") for i in mtp))
    rows = []
    for name, cls, layers, donor, regex in switches():
        pat = re.compile(regex)
        names = [n for n in sizes["c"] if pat.search(n)]
        delta = sum(sizes[donor][n] - sizes["c"][n] for n in names)
        rows.append((name, cls, layers, donor, len(names), delta))
    rows.append(("c", "base", "", "", len(sizes["c"]), on_card))
    write_tsv(out, ("name", "class", "layers", "donor", "tensors", "delta_bytes"), rows)
    print(f"file c on the cards: {on_card / 1e9:.3f} GB (MTP blocks {sorted(mtp)} and token_embd excluded)")


def parse_log(path):
    text = open(path, errors="replace").read()
    k = re.search(r"Mean\s+KLD:\s+(-?[0-9.]+)\s+±\s+([0-9.]+)", text)
    s = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s*%", text)
    if not k:
        return None
    return (k.group(1), k.group(2), s.group(1) if s else "", s.group(2) if s else "")


def cmd_collect(args):
    out, items = args[0], args[1:]
    rows = []
    for item in items:
        name, prefix = item.split("=", 1)
        for text in ("wiki", "agent", "code"):
            path = f"{prefix}.{text}.kld.log"
            if os.path.exists(path):
                r = parse_log(path)
                if r:
                    rows.append((name, text, *r))
    write_tsv(out, ("name", "text", "kld", "kld_se", "same_top_pct", "same_top_se"), rows)


def cmd_solve(args):
    import numpy as np
    deltas, kld, out = args[0], args[1], args[2]
    opts = dict(zip(args[3::2], args[4::2]))
    budgets = [float(x) for x in opts.get("--budgets", "0,-0.25e9,-0.5e9").split(",")]
    texts = opts.get("--texts", "wiki,agent,code").split(",")
    base = opts.get("--base", "c")
    dbytes = {r["name"]: int(r["delta_bytes"]) for r in read_tsv(deltas) if r["class"] != "base"}
    on_card = [int(r["delta_bytes"]) for r in read_tsv(deltas) if r["class"] == "base"][0]
    k = {(r["name"], r["text"]): float(r["kld"]) for r in read_tsv(kld)}
    names = [n for n in dbytes if all((n, t) in k for t in texts)]
    ref = {t: k[(base, t)] for t in texts}
    dk = np.array([[k[(n, t)] - ref[t] for t in texts] for n in names])
    score = (dk / np.array([ref[t] for t in texts])).sum(axis=1)
    by = np.array([dbytes[n] for n in names], dtype=np.float64)
    masks = ((np.arange(2 ** len(names))[:, None] >> np.arange(len(names))) & 1).astype(np.float64)
    tot_b, tot_s = masks @ by, masks @ score
    rows = []
    for budget in budgets:
        i = int(np.argmin(np.where(tot_b <= budget, tot_s, np.inf)))
        chosen = [names[j] for j in range(len(names)) if masks[i, j]]
        pred = [ref[t] + sum(k[(n, t)] - ref[t] for n in chosen) for t in texts]
        rows.append((f"{budget:.0f}", ",".join(chosen) or "-", f"{tot_b[i]:.0f}", f"{on_card + tot_b[i]:.0f}",
                     f"{tot_s[i]:.4f}", *(f"{p:.6f}" for p in pred)))
    write_tsv(out, ("budget_bytes", "switches", "delta_bytes", "on_card_bytes", "damage", *(f"pred_{t}" for t in texts)),
              rows)
    per = os.path.splitext(out)[0] + "-switches.tsv"
    write_tsv(per, ("name", "delta_bytes", "damage", *(f"dkld_{t}" for t in texts), "damage_per_gb"),
              [(n, dbytes[n], f"{score[j]:.4f}", *(f"{x:.6f}" for x in dk[j]),
                f"{score[j] / (dbytes[n] / 1e9):.4f}" if dbytes[n] else "") for j, n in enumerate(names)])
    print(open(out).read())


def main():
    cmds = {"specs": cmd_specs, "compose": cmd_compose, "deltas": cmd_deltas, "collect": cmd_collect,
            "solve": cmd_solve}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
