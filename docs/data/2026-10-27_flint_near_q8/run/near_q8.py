#!/usr/bin/env python3
"""E3: a Flint file within noise of Q8_0 at fewer bytes, choosing Q5_0 or Q8_0 per layer group and tensor class.

The base is Unsloth's Q8_0 with Flint-Pascal-Q5's F16 routers ("q8r"). Each switch takes one tensor class of one
layer group from Flint-Pascal-Q5 (Q5_0 where that file has it), the rest from the base. Each switch is scored alone
(KLD against BF16); the costs are assumed to add, and dynamic programming over the bytes each switch saves finds the
set with the least predicted KLD at each byte level. Groups follow Q36/Q77: layers 0-8, 9-16, 17-24, 25-32, 33-40,
41-48.

  specs                                 the switches: name, class, layers, regex
  compose BASE DONOR OUT ITEM...        compose_gguf.py (next to this script): BASE with DONOR's routers and the
                                        switches ITEM; no ITEM gives q8r
  deltas BASE DONOR OUT.tsv             per switch: tensors, file bytes and bytes read per token it adds (experts
                                        count n_used/n_exp of their bytes, the token embedding none), from the headers;
                                        a last row "q8r" with the base's totals
  collect OUT.tsv NAME=LOG...           KLD, same top token, PPL ratio and 99th-percentile KLD from llama-perplexity logs
  solve DELTAS.tsv KLD.tsv OUT.tsv [--base q8r] [--targets K,+DK,...] [--budgets GB,...] [--unit MB]
                                        the frontier over bytes per token (OUT) and over file bytes
                                        (OUT stem + -file.tsv), and the picks at each KLD target (+DK: the base's
                                        KLD + DK) and file budget (OUT stem + -picks.tsv)
Needs gguf-py (PYTHONPATH) for compose and deltas.
"""
import os
import re
import subprocess
import sys

GROUPS = [(0, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 48)]
CLASSES = [  # name, tensors within a block
    ("q", r"attn_q\.weight$"),
    ("kv", r"attn_[kv]\.weight$"),
    ("o", r"attn_output\.weight$"),
    ("sh", r"ffn_(?:gate|up|down)_shexp\.weight$"),
    ("gu", r"ffn_(?:gate|up)_exps\.weight$"),
    ("dn", r"ffn_down_exps\.weight$"),
]
GLOBAL = [("head", r"^output\.weight$"), ("embd", r"^token_embd\.weight$")]
ROUTER = r"ffn_gate_inp\.weight$"


def switches():
    out = []
    for cls, pat in CLASSES:
        for g, (lo, hi) in enumerate(GROUPS):
            layers = "|".join(str(i) for i in range(lo, hi + 1))
            out.append((f"{cls}{g}", cls, f"{lo}-{hi}", rf"^blk\.(?:{layers})\.{pat}"))
    out += [(name, name, "", regex) for name, regex in GLOBAL]
    return out


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
    base, donor, out, items = args[0], args[1], args[2], args[3:]
    sw = {s[0]: s for s in switches()}
    rules = [f"{ROUTER}={donor}"] + [f"{sw[i][3]}={donor}" for i in items]
    compose = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compose_gguf.py")
    subprocess.run([sys.executable, compose, base, out, *rules], check=True)


def tensor_costs(path):
    """name -> (file bytes, bytes read per token)"""
    import gguf
    r = gguf.GGUFReader(path)
    arch = r.fields["general.architecture"].contents()
    n_exp = r.fields[f"{arch}.expert_count"].contents()
    n_used = r.fields[f"{arch}.expert_used_count"].contents()
    out = {}
    for t in r.tensors:
        b = int(t.n_bytes)
        per_token = 0 if t.name == "token_embd.weight" else (
            b * n_used / n_exp if "_exps" in t.name and len(t.shape) == 3 else b)
        out[t.name] = (b, per_token)
    return out


def cmd_deltas(args):
    base_path, donor_path, out = args
    donor = tensor_costs(donor_path)
    base = tensor_costs(base_path)
    router = re.compile(ROUTER)
    base = {n: (donor[n] if router.search(n) else v) for n, v in base.items()}  # q8r: the donor's routers
    rows = []
    for name, cls, layers, regex in switches():
        pat = re.compile(regex)
        names = [n for n in base if pat.search(n) and donor[n][0] != base[n][0]]
        rows.append((name, cls, layers, len(names),
                     sum(donor[n][0] - base[n][0] for n in names), round(sum(donor[n][1] - base[n][1] for n in names))))
    rows.append(("q8r", "base", "", len(base), sum(v[0] for v in base.values()), round(sum(v[1] for v in base.values()))))
    write_tsv(out, ("name", "class", "layers", "tensors", "delta_file_bytes", "delta_token_bytes"), rows)


def parse_log(path):
    text = open(path, errors="replace").read()
    k = re.search(r"Mean\s+KLD:\s+(-?[0-9.]+)\s+±\s+([0-9.]+)", text)
    s = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)\s*%", text)
    p = re.search(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s+([0-9.]+)", text)
    k99 = re.search(r"99\.0%\s+KLD:\s+([0-9.]+)", text)
    if not k:
        return None
    return (k.group(1), k.group(2), s.group(1) if s else "", s.group(2) if s else "", p.group(1) if p else "",
            k99.group(1) if k99 else "")


def cmd_collect(args):
    out, items = args[0], args[1:]
    rows = []
    for item in items:
        name, path = item.split("=", 1)
        r = parse_log(path) if os.path.exists(path) else None
        if r:
            rows.append((name, *r))
    write_tsv(out, ("name", "kld", "kld_se", "same_top_pct", "same_top_se", "ppl_ratio", "kld99"), rows)


def frontier(items):
    """items: (name, saving in units >= 0, cost). Returns {saving: (least cost, names)} by 0/1 dynamic programming."""
    best = {0: (0.0, ())}
    for name, save, cost in items:
        for s, (c, names) in list(best.items()):
            s2, c2 = s + save, c + cost
            if s2 not in best or c2 < best[s2][0]:
                best[s2] = (c2, names + (name,))
    return best


def cmd_solve(args):
    deltas, kld, out = args[0], args[1], args[2]
    opts = dict(zip(args[3::2], args[4::2]))
    base = opts.get("--base", "q8r")
    unit = float(opts.get("--unit", "1")) * 1e6
    budgets = [float(x) * 1e9 for x in opts.get("--budgets", "33,36,39").split(",")]
    d = {r["name"]: r for r in read_tsv(deltas)}
    k = {r["name"]: float(r["kld"]) for r in read_tsv(kld)}
    ref = k[base]
    # a target with a leading + is relative to the base's KLD
    targets = [ref + float(x) if x.startswith("+") else float(x)
               for x in opts.get("--targets", "+0.003,+0.0065,+0.010").split(",")]
    tot_file, tot_tok = int(d[base]["delta_file_bytes"]), int(d[base]["delta_token_bytes"])
    sw = [n for n in d if n != base and int(d[n]["delta_file_bytes"]) != 0 and n in k]
    picks = []
    stem = os.path.splitext(out)[0]
    for resource, total, path in (("delta_token_bytes", tot_tok, out), ("delta_file_bytes", tot_file, stem + "-file.tsv")):
        items = [(n, round(-int(d[n][resource]) / unit), k[n] - ref) for n in sw]
        fr = frontier(items)
        pareto, best_c = [], float("inf")
        for s in sorted(fr, reverse=True):  # keep the points that no larger saving matches on KLD
            c, names = fr[s]
            if c < best_c:
                pareto.append((s, c, names))
                best_c = c
        pareto.sort()
        write_tsv(path, ("saved_mb", "bytes", "pred_kld", "switches"),
                  [(s, round(total - s * unit), f"{ref + c:.5f}", ",".join(names) or "-") for s, c, names in pareto])
        if resource == "delta_token_bytes":
            for t in targets:
                ok = [(s, c, names) for s, c, names in pareto if ref + c <= t]
                if ok:
                    s, c, names = max(ok)
                    picks.append((f"kld<={t:.4f}", "token", round(total - s * unit), f"{ref + c:.5f}", ",".join(names)))
        else:
            for b in budgets:
                ok = [(s, c, names) for s, c, names in pareto if total - s * unit <= b]
                if ok:
                    s, c, names = min(ok, key=lambda x: x[1])
                    picks.append((f"file<={b / 1e9:g}GB", "file", round(total - s * unit), f"{ref + c:.5f}",
                                  ",".join(names)))
    write_tsv(stem + "-picks.tsv", ("pick", "resource", "bytes", "pred_kld", "switches"), picks)
    print(open(stem + "-picks.tsv").read())


def main():
    cmds = {"specs": cmd_specs, "compose": cmd_compose, "deltas": cmd_deltas, "collect": cmd_collect,
            "solve": cmd_solve}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
