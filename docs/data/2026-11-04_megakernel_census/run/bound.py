#!/usr/bin/env python3
# bound.py CENSUS_DIR Q88_CSV KPASS_FIT_CSV: the bound on what one persistent kernel per card can save per token, from
# census.py's tables of one trace (ops.csv, cards.csv, tokens.csv in CENSUS_DIR), the P106's cost of an empty CUDA-graph
# node and of the small kernels of one MoE layer inside a CUDA graph (Q88's rig run, q88_s4l0.csv: section "reference"
# key graph_node_us, section "multi_launch_kernel" rows under 30 us), and kpass.py's line through the k-token passes.
#
# Prints and writes CENSUS_DIR/bound.csv:
#   - the launch groups and est. kernels per token, the groups doing near-zero work (no weights, not one of the ops
#     that move a state, a cache or the attention) and their profiled time;
#   - the profiler's own share: (profiled duration of a near-zero group) - (the same kind of kernel in a graph);
#   - the ceiling of a per-card persistent kernel: every est. kernel's launch cost removed, no barrier paid;
#   - the net saving for a grid-barrier cost b per stage and a share of stages that need no barrier.
# Standard library only.
import csv, statistics, sys

WORK_OPS = {"GATED_DELTA_NET", "FLASH_ATTN_EXT", "GET_ROWS", "CONT", "CONCAT", "SSM_CONV", "CPY", "SET_ROWS"}
BARRIER_US = [0.5, 1.0, 1.5, 2.0]
NO_BARRIER_SHARE = [0.0, 0.3]


def main():
    d, q88, fit = sys.argv[1], sys.argv[2], sys.argv[3]
    ops = list(csv.DictReader(open(f"{d}/ops.csv")))
    toks = [r for r in csv.DictReader(open(f"{d}/tokens.csv")) if r["consecutive"] == "1"]
    node_us, tiny = None, []
    for r in csv.reader(open(q88)):
        if len(r) < 8:
            continue
        if r[4] == "reference" and r[5] == "graph_node_us":
            node_us = float(r[6])
        if r[4] == "multi_launch_kernel" and float(r[6]) < 30.0:
            tiny.append((r[5], float(r[6])))
    small = [us for name, us in tiny if us < 5.0]  # the norm, quantizations, sums and adds of the layer
    tiny_us = statistics.median(small)
    wl_fit = {(r["quantity"], r["k_range"]): r for r in csv.DictReader(open(fit))}

    groups = sum(float(r["groups_per_token"]) for r in ops)
    kernels = sum(float(r["kernels_est_per_token"]) for r in ops)
    wl = [r for r in ops if r["weights"] == "-"]
    near0 = [r for r in wl if r["class"] not in WORK_OPS]
    work = [r for r in wl if r["class"] in WORK_OPS]
    g_near0 = sum(float(r["groups_per_token"]) for r in near0)
    ms_near0 = sum(float(r["ms_per_token"]) for r in near0)
    g_work = sum(float(r["groups_per_token"]) for r in work)
    ms_work = sum(float(r["ms_per_token"]) for r in work)
    ms_wl = sum(float(r["ms_per_token"]) for r in wl)
    gpu = statistics.median(float(r["gpu_ms"]) for r in toks)
    wall = statistics.median(float(r["wall_ms"]) for r in toks)
    hops = statistics.median(float(r["hops_ms"]) for r in toks)
    pre = statistics.median(float(r["pre_ms"]) for r in toks)
    prof_near0_us = 1000.0 * ms_near0 / g_near0
    event_us = prof_near0_us - tiny_us

    rows = [
        ("launch groups per token", groups, ""),
        ("est. kernels per token", kernels, "a product with quantized weights adds its activation quantization"),
        ("groups without weights", g_near0 + g_work, ""),
        ("  of them near-zero work (elementwise, norms, routing, hyper-connection mixing)", g_near0, ""),
        ("  of them moving a state, a cache or attention (GDN, FA, GET_ROWS, CONT, CONCAT, SSM_CONV, CPY, SET_ROWS)", g_work, ""),
        ("ms per token, groups without weights (profiled)", ms_wl, ""),
        ("  near-zero work (profiled)", ms_near0, ""),
        ("  moving a state, a cache or attention (profiled)", ms_work, ""),
        ("us per near-zero group (profiled mean)", prof_near0_us, ""),
        ("us per small kernel in a CUDA graph (Q88 rig, median of the layer's kernels under 5 us)", tiny_us,
         " ".join(f"{n}={u:.2f}" for n, u in tiny if u < 5.0)),
        ("us per empty CUDA-graph node (Q88 rig)", node_us, ""),
        ("us the profiler adds per group (est.: profiled near-zero mean minus the graph kernel)", event_us, ""),
        ("ms per token the profiler adds over all groups (est.)", groups * event_us / 1000.0, ""),
        ("ms per token of near-zero groups without the profiler (est.)", g_near0 * tiny_us / 1000.0, ""),
        ("profiled ms per token: GPU, hops between cards, host before card 0, wall (medians)", gpu,
         f"hops {hops:.2f}, host {pre:.2f}, wall {wall:.2f}"),
        ("ceiling: every est. kernel's launch cost removed, barriers free (ms per token)", kernels * tiny_us / 1000.0,
         f"node-only cost: {kernels * node_us / 1000.0:.2f}"),
    ]
    for b in BARRIER_US:
        for share in NO_BARRIER_SHARE:
            net = kernels * tiny_us - kernels * (1.0 - share) * b
            rows.append((f"net saving with a {b:g} us barrier, {int(100 * share)} % of stages without one (ms per token)",
                         net / 1000.0, ""))
    for (q, kr), r in sorted(wl_fit.items()):
        rows.append((f"k-token passes (Q583 llama-bench, k {kr}): {q} = a + b k: a (per pass), b (per token)",
                     float(r["per_pass_ms"]), f"b = {float(r['per_token_ms']):.2f}"))
    with open(f"{d}/bound.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "note"])
        for q, v, n in rows:
            w.writerow([q, f"{v:.2f}", n])
    for q, v, n in rows:
        print(f"{v:9.2f}  {q}" + (f"  [{n}]" if n else ""))


if __name__ == "__main__":
    main()
