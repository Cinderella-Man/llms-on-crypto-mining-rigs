#!/usr/bin/env python3
"""Which Flash-Next file to run on nine P106 cards: quality (KLD), fit and an estimated decode speed per file, from the
header dumps of dump.py, the fit grid of fit.py and llama-perplexity's KLD logs.
  python3 -I model.py --dumps DIR --fit fit.tsv --kld DIR --out DIR [--eff efficiency.tsv] [--base swift-q2_0]
      [--floor q8_0-ub256] [--nonweight 33.48] [--margin 150]
Writes to --out:
  layout.tsv       per file: experts, shards, which shard holds the n-gram table (alone or not), MiB on the cards (all
                   but the table and the token embeddings, which llama.cpp keeps in host memory), bytes a token reads
  token_bytes.tsv  per file, role and weight type: MB a decoded token reads (dense tensors whole, routed experts
                   used / experts), the GB/s assumed on a P106 and the est. ms
  speed.tsv        per file: est. ms of GPU time per token, summed over the cards, with the kernels timed on a P106
                   ("timed") and with the not-yet-timed kernels at their expected speed ("expected"), and per role
  fit.md           per file and context: the largest micro-batch that leaves --margin MiB on every card, on whole
                   layers and with routed-expert tensors moved (-ot), F16 and q8_0 cache, with and without a draft
  ranking.md       the quality / fit / speed table
The speed is a model, not a measurement: bytes per token by type over the GB/s that type's kernel reached on a P106
(efficiency.tsv, one row per type and role with its source), plus the GPU time R13's profile spent outside weight
products (--nonweight). Standard library only.
"""
import argparse, collections, csv, glob, json, os, re

ROLES = ("routed_gu", "routed_down", "shared", "router", "hc", "attention", "recurrent", "ple", "output", "other")
TEXTS = ("agent", "wiki", "code")


def role(name):
    if re.match(r"^(per_layer_token_embd|token_embd)\.", name):
        return "host"
    if re.search(r"\.ffn_(gate|up|gate_up)_exps\.", name):
        return "routed_gu"
    if re.search(r"\.ffn_down_exps\.", name):
        return "routed_down"
    if ".ffn_gate_inp" in name:
        return "router"
    if "shexp" in name:
        return "shared"
    if "hc_" in name:
        return "hc"
    if re.search(r"\.(attn_|indexer)", name):
        return "attention"
    if ".ssm_" in name:
        return "recurrent"
    if ".ple_" in name:
        return "ple"
    if name == "output.weight":
        return "output"
    return "other"


def label(name):
    maker, rest = name.split("-", 1)
    rest = rest.upper().replace("-HCQ8", "-hcq8").replace("CODER-", "Coder ")
    return {"swift": "Swift", "ista": "ISTA"}.get(maker, maker) + " " + rest


def load_eff(path):
    eff = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        eff[(r["type"], r["role"])] = (float(r["timed_gbps"]), float(r["expected_gbps"]))
    return eff


def gbps(eff, typ, rl):
    for key in ((typ, rl), (typ, "any")):
        if key in eff:
            return eff[key]
    raise SystemExit(f"efficiency.tsv has no row for type {typ}, role {rl}")


def kld(path):
    try:
        t = open(path, errors="replace").read()
    except OSError:
        return None
    m = re.search(r"Mean\s+KLD:\s+(-?[\d.]+)\s+±\s+([\d.]+)", t)
    p = re.search(r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)", t)
    return (float(m.group(1)), float(m.group(2)), float(p.group(1)) if p else None) if m else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eff", default=os.path.join(here, "efficiency.tsv"))
    ap.add_argument("--base", default="swift-q2_0")
    ap.add_argument("--floor", default="q8_0-ub256")
    ap.add_argument("--nonweight", type=float, default=33.48, help="ms of GPU time per token outside weight products")
    ap.add_argument("--margin", type=float, default=150, help="MiB every card must keep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eff = load_eff(a.eff)
    MiB = 2 ** 20

    files, lay, tok, spd = [], {}, {}, {}
    for path in sorted(glob.glob(os.path.join(a.dumps, "*.json"))):
        d = json.load(open(path))
        n = d["name"]
        files.append(n)
        ne, nu = d["keys"]["expert_count"], d["keys"]["expert_used_count"]
        cards = host_embd = 0
        table = None
        t = collections.Counter()
        for name, typ, dims, size, shard in d["tensors"]:
            rl = role(name)
            if rl == "host":
                if name.startswith("per_layer_token_embd."):
                    table = (shard, d["shards"][shard - 1]["tensors"], size)
                else:
                    host_embd += size
                continue
            cards += size
            t[(rl, typ)] += size * nu / ne if rl.startswith("routed") else size
        tok[n] = t
        lay[n] = dict(experts=f"{nu} of {ne}", file_type=d["file_type"], shards=len(d["shards"]),
                      shard_gb=" + ".join(f"{s['bytes'] / 1e9:.2f}" for s in d["shards"]),
                      table_shard=table[0], table_alone="yes" if table[1] == 1 else f"no ({table[1] - 1} others)",
                      reshard="no" if table[1] == 1 else "yes (C160)", table_gb=round(table[2] / 1e9, 2),
                      cards_mib=round(cards / MiB), token_embd_mib=round(host_embd / MiB),
                      token_mb=round(sum(t.values()) / 1e6, 1),
                      routed_mb=round(sum(v for (r, _), v in t.items() if r.startswith("routed")) / 1e6, 1),
                      shared_mb=round(sum(v for (r, _), v in t.items() if r == "shared") / 1e6, 1))
        lay[n]["dense_mb"] = round(lay[n]["token_mb"] - lay[n]["routed_mb"] - lay[n]["shared_mb"], 1)
        s = {"timed": collections.Counter(), "expected": collections.Counter()}
        for (rl, typ), b in t.items():
            g = gbps(eff, typ, rl)
            s["timed"][rl] += b / 1e6 / g[0]
            s["expected"][rl] += b / 1e6 / g[1]
        spd[n] = s

    with open(os.path.join(a.out, "layout.tsv"), "w") as f:
        cols = list(next(iter(lay.values())).keys())
        f.write("file\t" + "\t".join(cols) + "\n")
        for n in files:
            f.write(n + "\t" + "\t".join(str(lay[n][c]) for c in cols) + "\n")
    with open(os.path.join(a.out, "token_bytes.tsv"), "w") as f:
        f.write("file\trole\ttype\tmb_per_token\ttimed_gbps\texpected_gbps\ttimed_ms\texpected_ms\n")
        for n in files:
            for (rl, typ), b in sorted(tok[n].items(), key=lambda x: (ROLES.index(x[0][0]), -x[1])):
                g = gbps(eff, typ, rl)
                f.write(f"{n}\t{rl}\t{typ}\t{b / 1e6:.1f}\t{g[0]}\t{g[1]}\t{b / 1e6 / g[0]:.2f}\t{b / 1e6 / g[1]:.2f}\n")

    total = {n: {k: sum(spd[n][k].values()) + a.nonweight for k in ("timed", "expected")} for n in files}
    base = total.get(a.base)
    with open(os.path.join(a.out, "speed.tsv"), "w") as f:
        f.write("file\ttoken_mb\tnonweight_ms\ttimed_ms\texpected_ms\ttimed_tok_s\texpected_tok_s\t"
                "timed_vs_base_ms\texpected_vs_base_ms\t" + "\t".join(f"timed_{r}_ms" for r in ROLES) + "\t"
                + "\t".join(f"expected_{r}_ms" for r in ROLES) + "\n")
        for n in files:
            T, E = total[n]["timed"], total[n]["expected"]
            f.write(f"{n}\t{lay[n]['token_mb']}\t{a.nonweight}\t{T:.1f}\t{E:.1f}\t{1000 / T:.2f}\t{1000 / E:.2f}\t"
                    f"{T - base['timed']:+.1f}\t{E - base['expected']:+.1f}\t"
                    + "\t".join(f"{spd[n]['timed'][r]:.2f}" for r in ROLES) + "\t"
                    + "\t".join(f"{spd[n]['expected'][r]:.2f}" for r in ROLES) + "\n")

    # fit: the largest -ub with the margin, per file, context, cache, draft and mode
    fit = collections.defaultdict(list)
    for r in csv.DictReader(open(a.fit), delimiter="\t"):
        if r["smallest_free_mib"] in ("", "None"):
            continue
        fit[(r["file"], int(r["ctx"]), r["kv"], r["draft"], r["mode"])].append(
            (int(r["ub"]), int(r["smallest_free_mib"]), int(r["moved"]), r["ts"]))

    def best(n, ctx, kv, draft, mode):
        rows = fit.get((n, ctx, kv, draft, mode), [])
        ok = [x for x in rows if x[1] >= a.margin]
        if ok:
            ub, free, moved, ts = max(ok)
            return True, f"ub {ub}: {free}" + (f" ({moved} moved)" if moved else "")
        if not rows:
            return False, "—"
        return False, f"no ({max(x[1] for x in rows)})"

    def cell(n, ctx, kv="f16", draft="none"):
        ok_w, w = best(n, ctx, kv, draft, "whole")
        ok_o, o = best(n, ctx, kv, draft, "ot")
        if ok_w:
            return w
        return f"-ot {o}" if ok_o else w

    ctxs = sorted({k[1] for k in fit})
    ctxlabel = lambda c: "262k" if c == 262144 else f"{c // 1024}k"
    qual = {n: {t: kld(os.path.join(a.kld, f"{n}.{t}.log")) for t in TEXTS} for n in files + [a.floor]}
    # by agent KLD; a file without one goes after the scored file closest to it on wiki text
    scored = [n for n in files if qual[n]["agent"]]

    def rank(n):
        if qual[n]["agent"]:
            return (qual[n]["agent"][0], 0)
        w = (qual[n]["wiki"] or (9,))[0]
        near = min(scored, key=lambda m: abs((qual[m]["wiki"] or (9,))[0] - w))
        return (qual[near]["agent"][0], 1)

    order = sorted(files, key=rank)
    with open(os.path.join(a.out, "fit.md"), "w") as f:
        f.write(f"Largest `-ub` that leaves {a.margin:.0f} MiB on every card (free MiB on the fullest card; `no (x)`: "
                "the best of any `-ub` falls short), nine P106-090 cards. Whole layers / with routed-expert tensors "
                "moved by `-ot` (how many).\n")
        for draft in ("none", "dflash2", "mtp", "mtp-shr"):
            f.write(f"\n### Draft: {draft}\n\n| File | Cache | " + " | ".join(ctxlabel(c) for c in ctxs) + " |\n")
            f.write("|---|---|" + "---|" * len(ctxs) + "\n")
            for n in order:
                for kv in ("f16", "q8_0"):
                    cells = []
                    for c in ctxs:
                        _, w = best(n, c, kv, draft, "whole")
                        _, o = best(n, c, kv, draft, "ot")
                        cells.append(f"{w} / {o}")
                    f.write(f"| {label(n)} | {kv} | " + " | ".join(cells) + " |\n")

    def q(n, t):
        x = qual[n][t]
        return f"{x[0]:.3f}" if x else "—"

    with open(os.path.join(a.out, "ranking.md"), "w") as f:
        fl = qual[a.floor]
        f.write(f"KLD against Swift-1.5 Q8_0 (noise floor, the reference at another micro-batch: agent "
                f"{fl['agent'][0]:.3f}, wiki {fl['wiki'][0]:.3f}, code {fl['code'][0]:.3f}). Fit: F16 cache, "
                f"largest `-ub` with {a.margin:.0f} MiB left on every card and the fullest card's free MiB, whole "
                "layers unless marked `-ot`. Speed: est. ms of GPU time per decoded token at short context, no "
                "draft, kernels timed on a P106 / with the not-yet-timed kernels at their expected speed.\n\n")
        f.write("| File | Agent KLD (same top) | Wiki, code KLD | On the cards | Table | 64k | 128k | 262k | "
                "64k + DFlash2 | est. ms/token (tok/s) | vs Q2_0 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for n in order:
            ag = qual[n]["agent"]
            agent = f"{ag[0]:.3f} ± {ag[1]:.3f} ({ag[2]:.1f} %)" if ag else "not scored"
            T, E = total[n]["timed"], total[n]["expected"]
            vs = "—" if n == a.base else f"{T - base['timed']:+.1f} / {E - base['expected']:+.1f}"
            tab = f"shard {lay[n]['table_shard']}" + ("" if lay[n]["table_alone"] == "yes" else ", reshard")
            f.write(f"| {label(n)} | {agent} | {q(n, 'wiki')}, {q(n, 'code')} | {lay[n]['cards_mib']:,} MiB | {tab} | "
                    f"{cell(n, 65536)} | {cell(n, 131072)} | {cell(n, 262144)} | {cell(n, 65536, draft='dflash2')} | "
                    f"{T:.1f} ({1000 / T:.1f}) / {E:.1f} ({1000 / E:.1f}) | {vs} |\n")


if __name__ == "__main__":
    main()
