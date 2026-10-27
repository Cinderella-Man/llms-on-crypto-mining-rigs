#!/usr/bin/env python3
# prof_summary.py CSV [MAX_EVAL_MS]: a GGML_CUDA_PROFILE trace (device,eval,host_ms,group,label,us) summarized per
# decoded token. An eval on one device whose groups sum to less than MAX_EVAL_MS (default 60) counts as a one-token
# decode step, so a server trace with prompts mixed in needs no token count; per launch group, per first op and per
# weight type the time per decoded token summed over the cards, its share, and per card the time per token.
# scripts/cuda_profile_summary.py summarizes the last N evaluations of a llama-bench trace instead. Standard library only.
import collections, csv, sys

path = sys.argv[1]
max_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
evals = collections.defaultdict(list)  # (device, eval) -> [(label, us)]
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        evals[(row["device"], int(row["eval"]))].append((row["label"], float(row["us"])))
dec = {k: v for k, v in evals.items() if sum(us for _, us in v) < max_ms * 1000}
pre = {k: v for k, v in evals.items() if k not in dec}
devs = sorted({d for d, _ in evals})
n_tok = max(1, len(dec) // max(1, len(devs)))
print(f"{len(evals)} device evals: {len(dec)} decode-sized (< {max_ms} ms), {len(pre)} larger; ~{n_tok} decode steps")
per_dev = collections.Counter()
per_label = collections.Counter()
for (d, _), groups in dec.items():
    for label, us in groups:
        per_dev[d] += us
        per_label[label] += us
total = sum(per_dev.values())
print(f"decode: {total / n_tok / 1000:.2f} ms of GPU time per token summed over cards "
      f"(≈ {1e6 * n_tok / total:.2f} tok/s if the cards ran back to back with no gaps)")
print("per card, ms per token:", " ".join(f"{d}:{per_dev[d] / n_tok / 1000:.2f}" for d in devs))
by_op = collections.Counter()
by_type = collections.Counter()
for label, us in per_label.items():
    ops, _, weights = label.partition(" ")
    by_op[ops.split("+")[0]] += us
    types = sorted({w.split(":")[1] for w in weights.split() if w.count(":") == 2})
    by_type["+".join(types) or "(no weights)"] += us
print("\nby first op of the launch group (ms/token, share):")
for k, us in by_op.most_common(15):
    print(f"  {us / n_tok / 1000:7.2f} {100 * us / total:5.1f} %  {k}")
print("\nby weight type (ms/token, share):")
for k, us in by_type.most_common(15):
    print(f"  {us / n_tok / 1000:7.2f} {100 * us / total:5.1f} %  {k}")
print("\ntop launch groups (ms/token, share, label):")
for k, us in per_label.most_common(30):
    print(f"  {us / n_tok / 1000:7.2f} {100 * us / total:5.1f} %  {k[:150]}")
