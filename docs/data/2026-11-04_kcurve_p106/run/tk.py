#!/usr/bin/env python3
"""tk.py: T(k), the wall time of a k-token pass of the whole model, from llama-bench JSON (verify_cost.sh's wall step,
llama-bench -p 1,2,...,8 -n 0 -o json). Prints a TSV per arm (k, mean and median ms over the repetitions, sd, ms per
token, T(k)/T(1), the step T(k) - T(k-1)) and writes, with --cost DIR, one LLAMA_SPEC_ADAPT_COST file per arm: one
T(k) in ms per line for k = 1, 2, ... (llama-server reads past the last entry as the last slope).
  python3 -I tk.py [--cost DIR] [--verify-extra k=ms,...] ARM=bench-wall.json ...
--verify-extra adds per k the output layer's cost for the k - 1 rows a verify also needs (llama-bench's passes ask
for the last row's logits only), e.g. from test-backend-ops perf of the output layer at n = k minus n = 1."""
import json, statistics, sys, os

def load(path):
    rows = json.load(open(path))
    out = {}
    for r in rows:
        if r.get("n_gen", 0) != 0:
            continue
        k = r["n_prompt"]
        s = [x / 1e6 for x in r.get("samples_ns", [])] or [r["avg_ns"] / 1e6]
        out[k] = s
    return out

def main(argv):
    cost_dir, extra, arms = None, {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--cost":
            cost_dir = argv[i + 1]; i += 2; continue
        if a == "--verify-extra":
            for kv in argv[i + 1].split(","):
                k, v = kv.split("="); extra[int(k)] = float(v)
            i += 2; continue
        name, path = a.split("=", 1)
        arms.append((name, path)); i += 1
    print("arm\tk\tmean_ms\tmedian_ms\tsd_ms\tms_per_token\tratio_to_k1\tstep_ms\tverify_ms")
    for name, path in arms:
        d = load(path)
        t1 = statistics.median(d[min(d)])
        prev, costs = None, []
        for k in sorted(d):
            s = d[k]
            med = statistics.median(s)
            sd = statistics.stdev(s) if len(s) > 1 else 0.0
            ver = med + extra.get(k, 0.0)
            costs.append(ver)
            step = "" if prev is None else f"{med - prev:.1f}"
            print(f"{name}\t{k}\t{statistics.mean(s):.1f}\t{med:.1f}\t{sd:.1f}\t{med / k:.1f}\t{med / t1:.2f}\t{step}\t{ver:.1f}")
            prev = med
        if cost_dir:
            os.makedirs(cost_dir, exist_ok=True)
            with open(os.path.join(cost_dir, f"t_k_{name}.txt"), "w") as f:
                f.write(f"# T(k) in ms, k = 1..{max(d)}: whole-model k-token passes on one rig ({name}); "
                        f"median of {len(d[min(d)])} repetitions{', plus the verify extra' if extra else ''}\n")
                for c in costs:
                    f.write(f"{c:.1f}\n")

if __name__ == "__main__":
    main(sys.argv[1:])
