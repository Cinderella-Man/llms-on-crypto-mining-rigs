"""The data folder's tables from the raw outputs (no numpy):
  guess_cost.tsv      every DRAFTER_BENCH line of profile.sh / variants.sh logs (build tag, option set, ring cells,
                      tokens in flight, median / p90 / mean / launch microseconds)
  served_guesses.tsv  llama-server's per-request guess counts summed per arm (agent check server logs)
  break_even.tsv      cost_sim.py's JSON outputs, one row per configuration

usage: python curate.py OUT_DIR --bench TAG=bench.log ... --served ARM=server.log ... --sim cost.json ...
"""
import argparse
import json
import os
import re


def bench_rows(tag, path):
    """Rows of a bench log, labelled by the log's headers. The first runs logged some without: profile.sh's in its
    fixed order (per ring size plain with 300 guesses, prof with 100, host with 300, then window with one token in
    flight), variants.sh's per-kernel traces (its only 100-guess rows) in its loop's order."""
    traces = [("trace-q8", "LLAMA_DRAFT_HEAD=q8_0"),
              ("trace-q8mmq", "LLAMA_DRAFT_HEAD=q8_0 GGML_CUDA_MMVQ_MAX_COLS=4"),
              ("trace-q8split", "LLAMA_DRAFT_HEAD=q8_0 LLAMA_DRAFT_SPLIT=4")]
    rows, opts, seen, in_traces, sweep = [], "", set(), False, False
    for line in open(path):
        if line.startswith("=== cost of one guess"):
            sweep = True
        m = re.match(r"--- (\S+) \((.*)\) ctx (\S+)", line)
        if m:
            opts = m.group(1) + "\t" + (m.group(2) or "-")
            continue
        m = re.match(r"=== ctx (\d+) (\w+)", line)   # profile.sh: plain, prof (per-kernel events), host, window (1)
        if m:
            opts = f"{m.group(2)}\t-"
            continue
        if line.startswith("=== traces"):
            in_traces, opts = True, ""
            continue
        if "DRAFTER_BENCH" not in line:
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
        ctx = int(kv["ctx"])
        cells = ((max(4096, ctx + 64) + 32 + 16 + 255) // 256) * 256   # the drafter's ring: n_ctx + window + scratch
        if sweep and kv["n"] == "100" and not opts.startswith("trace-"):
            in_traces, opts = True, ""
        o = opts
        if not o and in_traces and traces:
            o = "\t".join(traces.pop(0))
        elif not o:
            mode = "prof" if kv["n"] == "100" else "window" if kv["window"] == "1" else \
                   "host" if ("plain", cells) in seen else "plain"
            seen.add((mode, cells))
            o = f"{mode}\t-"
        if in_traces:
            opts = ""   # every trace has its own header (or its place in the order)
        rows.append([tag, o, str(cells), kv["window"], kv["commits"], kv["n"], kv["median_us"], kv["p90_us"],
                     kv["mean_us"], kv.get("launch_us", "")])
    return rows


def served_rows(arm, path):
    r = w = dg = dr = n_req = 0
    for line in open(path):
        if "pipelined = " not in line:
            continue
        f = line.split("pipelined =", 1)[1].split()
        n_req += 1
        r += int(f[2])
        w += int(f[4])
        dg += int(f[11])
        dr += int(f[13])
    return [arm, str(n_req), str(r + w), str(r), f"{r / max(r + w, 1):.3f}", str(dg), str(dr), f"{dr / max(dg, 1):.3f}",
            str(r + w - dg), str(r - dr), f"{(r - dr) / max(r + w - dg, 1):.3f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--served", nargs="*", default=[])
    ap.add_argument("--sim", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.bench:
        with open(os.path.join(a.out, "guess_cost.tsv"), "w") as f:
            f.write("build\toptions\tenvironment\tcells\tin_flight\tcommits\tguesses\tmedian_us\tp90_us\tmean_us\tlaunch_us\n")
            for spec in a.bench:
                tag, path = spec.split("=", 1)
                for row in bench_rows(tag, path):
                    f.write("\t".join(row) + "\n")
    if a.served:
        with open(os.path.join(a.out, "served_guesses.tsv"), "w") as f:
            f.write("arm\trequests\tchecked\tright\tright_share\thead_checked\thead_right\thead_share\t"
                    "lookup_checked\tlookup_right\tlookup_share\n")
            for spec in a.served:
                arm, path = spec.split("=", 1)
                f.write("\t".join(served_rows(arm, path)) + "\n")
    if a.sim:
        with open(os.path.join(a.out, "break_even.tsv"), "w") as f:
            keys = None
            for path in a.sim:
                d = json.load(open(path))
                for r in d["rows"]:
                    st = r["stats"]
                    row = {"order": r["order"], "head_card": r["head_card"], "depth": r["depth"],
                           "head_ms": r["head_ms"], "head_right": r["accept"], "tok_s": f"{r['tok_s']:.2f}"}
                    for k in ("hit_copy", "miss_copy", "hit_cont1", "miss_cont1", "hit_cont2", "miss_cont2", "hit_head",
                              "miss_head", "head_launches"):
                        row[k] = st.get(k, 0)
                    if keys is None:
                        keys = list(row)
                        f.write("\t".join(keys) + "\n")
                    f.write("\t".join(str(row[k]) for k in keys) + "\n")


main()
