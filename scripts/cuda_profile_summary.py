#!/usr/bin/env python3
"""Summarizes a GGML_CUDA_PROFILE trace: GPU time per token by launch group.

The fork's CUDA backend (branch pascal-p106) writes one CSV line per launch group of every graph evaluation when
GGML_CUDA_PROFILE=<file> is set: device, eval, host_ms, group, label, us. NVIDIA's profilers refuse the CMP mining
GPUs (P106-090 and others), this trace works on any GPU. Run a generation benchmark with it, e.g.

    GGML_CUDA_PROFILE=/tmp/prof.csv llama-bench -m model.gguf -ngl 99 -fa on -p 0 -n 32 -r 1

then summarize the steady-state tokens (the last --tokens evaluations of every device):

    scripts/cuda_profile_summary.py /tmp/prof.csv --tokens 24

A GGML_CUDA_PROFILE_HOST=<file> trace (one line per graph evaluation and device: host time at entry to and exit from
the backend's graph_compute, GPU time of the evaluation) is recognized by its header and summarized as a timeline:
wall time per token, the GPU chain (the devices run one after the other, so their GPU times add up) and the host time
spent inside graph_compute. If the wall time is well above the GPU chain, the GPUs wait for the host or for the
copies between them.

Labels look like "MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_up_exps:q5_0:2048x512 ffn_gate_exps:q5_0:2048x512": the ops of the
group, then each matrix product's weight (layer number removed), type and shape (K x N). The time of a matrix product
includes the quantization of its activations, which runs in the same group. Every group boundary is a CUDA event, which
adds a little GPU time (compare the tokens per second with an unprofiled run).
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--tokens", type=int, default=24, help="steady-state evaluations per device to average (default 24)")
    ap.add_argument("--by", choices=["label", "op"], default="label",
                    help="group by full label, or by op only (matrix products keep their weight names)")
    ap.add_argument("--top", type=int, default=60, help="rows to print (default 60)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if rows and "host_enter_ms" in rows[0]:
        return host_summary(rows, args.tokens)
    evals = defaultdict(set)
    for r in rows:
        evals[int(r["device"])].add(int(r["eval"]))
    keep = {d: set(sorted(e)[-args.tokens:]) for d, e in evals.items()}
    n_tok = min(len(k) for k in keep.values())

    per_label = defaultdict(lambda: [0.0, 0])        # total us, calls
    per_dev = defaultdict(float)
    host = defaultdict(dict)                          # device -> eval -> host_ms
    for r in rows:
        d, e = int(r["device"]), int(r["eval"])
        if e not in keep[d]:
            continue
        label = r["label"]
        if args.by == "op" and " " in label:
            ops = label.split(" ", 1)[0]
            if "MUL_MAT" not in ops:
                label = ops
        us = float(r["us"])
        per_label[label][0] += us
        per_label[label][1] += 1
        per_dev[d] += us
        host[d][e] = float(r["host_ms"])

    total = sum(v[0] for v in per_label.values())
    print(f"devices: {len(evals)}, steady-state tokens: {n_tok} (last {args.tokens} evaluations per device)")
    print(f"GPU time per token, sum over devices: {total / n_tok / 1000:.2f} ms")
    for d in sorted(per_dev):
        print(f"  device {d}: {per_dev[d] / n_tok / 1000:.2f} ms")
    # wall time per token from the host timestamps of the first device's evaluations
    d0 = min(host)
    ts = [host[d0][e] for e in sorted(host[d0])]
    if len(ts) > 1:
        print(f"wall time per token (device {d0} evaluation to evaluation): {(ts[-1] - ts[0]) / (len(ts) - 1):.2f} ms")
    print()
    print(f"{'ms/token':>9} {'%':>6} {'calls/tok':>9} {'us/call':>9}  label")
    for label, (us, n) in sorted(per_label.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f"{us / n_tok / 1000:9.3f} {100 * us / total:6.2f} {n / n_tok:9.1f} {us / n:9.2f}  {label}")


def host_summary(rows, n_tokens):
    by_dev = defaultdict(dict)
    for r in rows:
        by_dev[int(r["device"])][int(r["eval"])] = (float(r["host_enter_ms"]), float(r["host_exit_ms"]), float(r["gpu_us"]))
    devs = sorted(by_dev)
    n_eval = min(len(v) for v in by_dev.values())
    evals = list(range(n_eval - n_tokens, n_eval))
    d0 = devs[0]
    wall = (by_dev[d0][evals[-1]][0] - by_dev[d0][evals[0]][0]) / (len(evals) - 1)
    gpu = {d: sum(by_dev[d][e][2] for e in evals) / len(evals) / 1000 for d in devs}
    host_in = {d: sum(by_dev[d][e][1] - by_dev[d][e][0] for e in evals) / len(evals) for d in devs}
    # when each device's evaluation is entered, relative to the first device's, and how long the host takes from the
    # first device's entry to the last device's exit
    lead = {d: sum(by_dev[d][e][0] - by_dev[d0][e][0] for e in evals) / len(evals) for d in devs}
    span = sum(by_dev[devs[-1]][e][1] - by_dev[d0][e][0] for e in evals) / len(evals)
    chain = sum(gpu.values())
    print(f"devices: {len(devs)}, steady-state tokens: {len(evals)}")
    print(f"wall time per token:                         {wall:7.2f} ms")
    print(f"GPU chain per token (sum over devices):      {chain:7.2f} ms  ({100 * chain / wall:.0f} % of the wall time)")
    print(f"host time inside graph_compute, all devices: {sum(host_in.values()):7.2f} ms")
    print(f"host: first device entered to last exited:   {span:7.2f} ms")
    print(f"{'device':>6} {'GPU ms':>8} {'host in compute ms':>18} {'entered after device ' + str(d0) + ' (ms)':>28}")
    for d in devs:
        print(f"{d:6d} {gpu[d]:8.2f} {host_in[d]:18.3f} {lead[d]:28.2f}")


if __name__ == "__main__":
    main()
