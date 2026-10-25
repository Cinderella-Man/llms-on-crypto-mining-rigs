#!/usr/bin/env python3
"""Helpers and README tables for the release matrix (bench/release_matrix/README.md). Standard library only.

  make_table.py bench FILE.jsonl     llama-bench's JSON lines -> "ppN<TAB>tok/s" and "tgN<TAB>tok/s" lines
  make_table.py sha FILE.json        the greedy request's SHA-256 (client.py greedy)
  make_table.py served FILE.jsonl    chat requests (client.py chat) -> "median<TAB>min<TAB>max<TAB>count"
  make_table.py tables OUT           OUT/results.tsv and OUT/meta.tsv -> Markdown tables for the README: one per block
                                     (untuned, tuned), never mixing the two, then what the tuning adds per engine
"""
import json
import statistics
import sys
from collections import defaultdict

ENGINE_NAMES = {"stock": "Stock llama.cpp", "v1": "v1 engine", "v2": "v2 engine"}


def engine_name(cfg, pipe_args):
    base, pipe = (cfg[:-5], True) if cfg.endswith("+pipe") else (cfg, False)
    name = ENGINE_NAMES.get(base, base)
    return f"{name}, pipelined speculation (`{pipe_args}`)" if pipe else name


def cmd_bench(path):
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d.get("n_gen", 0) == 0 and d.get("n_prompt", 0) > 0:
            print(f"pp{d['n_prompt']}\t{d['avg_ts']:.2f}")
        elif d.get("n_prompt", 0) == 0 and d.get("n_gen", 0) > 0:
            print(f"tg{d['n_gen']}\t{d['avg_ts']:.2f}")


def cmd_served(path):
    v = [json.loads(l).get("tok_s") for l in open(path) if l.strip()]
    v = [x for x in v if x]
    if v:
        print(f"{statistics.median(v):.2f}\t{min(v):.2f}\t{max(v):.2f}\t{len(v)}")


def fmt(x):
    return f"{x:.1f}" if x is not None else "—"


def cmd_tables(out):
    meta = {}
    for line in open(f"{out}/meta.tsv"):
        k, _, v = line.rstrip("\n").partition("\t")
        meta[k] = v
    vals = defaultdict(list)  # (block, model, config, metric) -> values
    text = {}
    models, configs, blocks = [], [], []
    for line in open(f"{out}/results.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) != 5:
            continue
        blk, model, cfg, metric, value = f
        for lst, x in ((blocks, blk), (models, model), (configs, cfg)):
            if x not in lst:
                lst.append(x)
        if metric in ("greedy_sha", "exact", "error", "served_range"):
            text[(blk, model, cfg, metric)] = value
        else:
            vals[(blk, model, cfg, metric)].append(float(value))

    def get(blk, model, cfg, metric):
        v = vals.get((blk, model, cfg, metric))
        return statistics.median(v) if v else None

    pipe = meta.get("pipe_args", "--pipe-depth 3")
    gen = next((m[2:] for (_, _, _, m) in vals if m.startswith("tg")), "128")
    pp = next((m[2:] for (_, _, _, m) in vals if m.startswith("pp")), "512")
    cards = meta.get("cards", "9")
    head = "| Engine | " + " | ".join(f"{m} generation | {m} served" for m in models) + " |"
    rule = "|---|" + "---|---|" * len(models)
    offsets = meta.get("tuned_offsets", "")
    uniq = sorted(set(offsets.split(","))) if offsets else []
    tuned_desc = (f"memory +{uniq[0]} MT/s on every card" if len(uniq) == 1 else f"memory offsets {offsets} MT/s"
                  if offsets else "memory offset")

    print(f"<!-- bench/release_matrix, {out.rstrip('/').split('/')[-1]}: paste into the README's hardware-tuning "
          f"section and the model pages -->")
    for blk in [b for b in ("untuned", "tuned") if b in blocks]:
        title = "Stock clocks" if blk == "untuned" else f"Tuned: {tuned_desc}"
        print(f"\n**{title}**, {cards} cards, tokens per second:\n")
        print(head)
        print(rule)
        for cfg in configs:
            cells = []
            for m in models:
                cells += [fmt(get(blk, m, cfg, f"tg{gen}")), fmt(get(blk, m, cfg, "served"))]
            if any(c != "—" for c in cells):
                print(f"| {engine_name(cfg, pipe)} | " + " | ".join(cells) + " |")
    if "untuned" in blocks and "tuned" in blocks:
        print(f"\n**What the tuning adds** ({tuned_desc}, each engine against itself at stock clocks):\n")
        print(head)
        print(rule)
        for cfg in configs:
            cells = []
            for m in models:
                for metric in (f"tg{gen}", "served"):
                    a, b = get("untuned", m, cfg, metric), get("tuned", m, cfg, metric)
                    cells.append(f"{100 * (b / a - 1):+.1f} %" if a and b else "—")
            if any(c != "—" for c in cells):
                print(f"| {engine_name(cfg, pipe)} | " + " | ".join(cells) + " |")
        same = [k for k, v in text.items() if k[3] == "exact" and v == "same"]
        diff = [k for k, v in text.items() if k[3] == "exact" and v != "same"]
        print(f"\nGreedy output under the offset identical to stock clocks: {len(same)} of {len(same) + len(diff)} "
              "served configurations" + (": differs for " + ", ".join(f"{k[2]} on {k[1]}" for k in diff) if diff else "")
              + ".")
    print(f"\nPrefill (reading the prompt), {pp} tokens, tokens per second:\n")
    print("| Engine | " + " | ".join(f"{m}, {b}" for b in blocks for m in models) + " |")
    print("|---|" + "---|" * (len(blocks) * len(models)))
    for cfg in configs:
        cells = [fmt(get(b, m, cfg, f"pp{pp}")) for b in blocks for m in models]
        if any(c != "—" for c in cells):
            print(f"| {engine_name(cfg, pipe)} | " + " | ".join(cells) + " |")
    print("\nMethod:")
    print(f"- Generation: `llama-bench {meta.get('bench_args', '')}`: {gen} tokens right after a {pp}-token prompt in "
          "the same process, so the cards are at their working clocks.")
    print(f"- Served: llama-server `{meta.get('server_args', '')}`, the median of requests of up to "
          f"{meta.get('max_tokens', '?')} tokens ({meta.get('prompts', '').strip()} × seeds {meta.get('seeds', '')}) with "
          "the makers' sampling, after one warm-up request; reasoning tokens count.")
    builds = ", ".join(f"{ENGINE_NAMES.get(k[6:], k[6:])} `{v}`" for k, v in meta.items() if k.startswith("build_"))
    print(f"- Builds: {builds}. " + " ".join(f"{ENGINE_NAMES.get(k[4:], k[4:])} runs with `{v}`." for k, v in meta.items()
                                            if k.startswith("env_") and v))
    clocks = [f"{meta[f'mem_clock_max_{b}']} MHz {b}" for b in blocks if f"mem_clock_max_{b}" in meta]
    xids = [f"{meta[f'xid_{b}']} {b}" for b in blocks if f"xid_{b}" in meta]
    if clocks:
        print(f"- Highest memory clock under load: {', '.join(clocks)}; kernel Xid lines: {', '.join(xids)}.")
    errors = [k for k in text if k[3] == "error"]
    if errors:
        print("- Failed runs: " + ", ".join(f"{k[0]} {k[2]} {k[1]} ({text[k]})" for k in errors) + ".")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("bench", "sha", "served", "tables"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "bench":
        cmd_bench(arg)
    elif cmd == "sha":
        print(json.load(open(arg))["sha256"])
    elif cmd == "served":
        cmd_served(arg)
    else:
        cmd_tables(arg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
