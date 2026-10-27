# Q46: Pascal kernels for Q5_K and IQ4_XS (2026-10-23)

Registry row Q46 ([quants/experiments.md](../../../quants/experiments.md)) asked whether one-token kernels for Q5_K and
IQ4_XS, the formats of Unsloth's UD-Q4_K_XL routed experts, could reach the bandwidth of the tuned Q4_1/Q8_0 kernels.
Both gates passed in the microbenchmark (Q5_K ≥ 110, IQ4_XS ≥ 90 GB/s), and in Miner Llama UD-Q4_K_XL on 6 cards went
from 24.49 to 34.06 tok/s with one binary, kernels off and on. Design and checks:
[pascal_mmvq_integration.md](../../research/notes/pascal_mmvq_integration.md) and
[bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md). This folder holds the microbenchmark run, the
GPU profiles of UD-Q4_K_XL with the kernels off and on, and the commands.

| File | What |
|---|---|
| `kq.csv` | `bench/mmvq_pascal` on one P106-090 for Q5_K and IQ4_XS: every shape × variant (llama.cpp's kernel, PR #25479's shape, `widek`, one or two rows per lane group with direct loads `p_r*c1` or staged loads `st_r*c1`) plus per-token totals; 111 rows |
| `profile_by_label.csv` | GPU time per token of UD-Q4_K_XL on 6 cards by launch group, `ud-kernels-off` and `ud-kernels-on` (fork build `eeaa59660`) |
| `profile_classes.csv` | the same grouped by tensor class and type (the table below) |
| `profile_totals.csv` | per run: GPU ms per token summed over the cards, per card, and the profiled wall time per token |
| `run/run7.sh`, `run/run8.sh` | the rig runs: tests, tg128 off/on, profiles and KLD (7); the noise floor against the standard build (8) |

GPU time per token of UD-Q4_K_XL on 6 cards (ms, the last 24 tokens of a 32-token run, summed over the cards; the
MoE down-projection rescale is on, as UD-Q4_K_XL needs it on this build):

| Tensors | Type | Launches per token | Kernels off | Kernels on |
|---|---|---|---|---|
| attention q | Q8_0 / Q5_K | 42 / 7 | 4.62 / 0.80 | 3.33 / 0.46 |
| attention k, v | Q8_0 / Q5_K | 84 / 14 | 2.26 / 0.35 | 1.81 / 0.28 |
| attention output | Q8_0 / Q5_K | 42 / 7 | 4.77 / 0.76 | 3.39 / 0.52 |
| routed experts gate/up | IQ4_XS / Q5_K | 25 / 18 | 3.00 / 2.89 | 2.39 / 2.05 |
| routed experts gate/up | Q4_K / IQ3_S (no Pascal kernel) | 3 / 1 | 0.44 / 0.21 | 0.44 / 0.21 |
| routed experts down | Q5_K / IQ4_XS | 29 / 16 | 4.89 / 4.00 | 1.80 / 0.85 |
| routed experts down | Q6_K (no Pascal kernel) / Q8_0 | 1 / 1 | 0.19 / 0.11 | 0.19 / 0.07 |
| shared expert gate, up | Q8_0 / Q5_K / IQ4_XS | 68 / 24 / 2 | 2.53 / 0.88 / 0.06 | 2.00 / 0.64 / 0.05 |
| shared expert down | Q8_0 / Q5_K / IQ4_XS | 34 / 12 / 1 | 1.91 / 1.19 / 0.05 | 1.26 / 0.45 / 0.04 |
| dense FFN, layers 0–1 | Q8_0 | 5 | 0.99 | 0.62 |
| output head | Q8_0 | 1 | 3.34 | 2.16 |
| router | BF16 | 47 | 1.23 | 1.23 |
| everything else (incl. the rescale) | | 1,006 | 14.31 | 14.04 |
| **total** | | | **55.79** | **40.28** |

By format: Q5_K 11.8 → 6.2 ms, IQ4_XS 7.1 → 3.3 ms, Q8_0 20.5 → 14.6 ms per token. IQ4_XS's fused expert gate/up
gains least (3.00 → 2.39 ms): it runs one row per lane group, because two rows need 255 registers.

Made on 2026-10-23. The microbenchmark ran on one rig card with the benchmark as committed in `6e695b9`
(see [bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md), "Running on the rig"):

```bash
RIG=${RIG:-rig}
bench/mmvq_pascal/build.sh
scp bench/mmvq_pascal/build/mmvq_pascal "$RIG":/tmp/
ssh "$RIG" 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --types q5_K,iq4_xs --csv /tmp/kq.csv'
```

The engine runs used the fork's `pascal-p106` branch at `eeaa59660` and, for the noise floor, the standard build
(upstream `d77256999` with CUDA graphs on compute 6.1), both under `/opt/llama-builds/` on the rig:

```bash
ssh "$RIG" 'mkdir -p /tmp/q46'
scp scripts/bench.sh docs/data/2026-10-23_q46/run/run7.sh docs/data/2026-10-23_q46/run/run8.sh "$RIG":/tmp/q46/
ssh "$RIG" 'cd /tmp/q46 && bash run7.sh > run7.log 2>&1 && bash run8.sh > run8.log 2>&1'
```

The profile CSVs were made with [prof_to_csv.py](../2026-10-23_q38/run/prof_to_csv.py) and
[profile_classes.py](../2026-10-23_q38/run/profile_classes.py) (`--tokens 24`, runs `ud-kernels-off=prof-ud-mode0.csv`
and `ud-kernels-on=prof-ud-mode1.csv`). The raw traces (2.5 MB each), the test and KLD logs and the llama-bench
outputs are not published; their results are in the registry row.
