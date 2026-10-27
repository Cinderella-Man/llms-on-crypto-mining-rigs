# Q38: the Pascal matrix-vector kernels, from microbenchmark to the fork (2026-10-23)

Registry row Q38 ([quants/experiments.md](../../../quants/experiments.md)) ported the tuned one-token kernels of
[bench/mmvq_pascal](../../../bench/mmvq_pascal/) into Miner Llama for Q4_0, Q4_1, Q5_0 and Q8_0: Miner S4-L0 on 6
cards went from 29.20 to 37.37 tok/s with one binary, kernels off and on. The design and the checks are in
[pascal_mmvq_integration.md](../../research/notes/pascal_mmvq_integration.md). This folder holds the microbenchmark
runs behind the kernel choices, the fork's GPU and host profiles of S4-L0, the S4-L0 operation lists used for the
replay test, and the commands.

| File | What |
|---|---|
| `mmvq_step1.csv` | `bench/mmvq_pascal` on one P106-090, Q4_1, Q4_1_64, Q4_0, Q8_0, Q5_0, Q5_K and IQ4_XS: every shape × type × variant (llama.cpp's kernel, its small-K shape `llama_sk`, PR #25479's launch shape, the wide-rows shape `widek`, the tuned variants `p_*`, tensor-wide planes `soa_*`, row-local planes `rl_*`) plus per-token totals; 516 rows |
| `mmvq_staged.csv` | the run that added staged loads (`st_*`) for Q8_0, Q5_0, Q4_0 and Q4_1; 477 rows |
| `profile_by_label.csv` | GPU time per token of S4-L0 on 6 cards by launch group, three runs: `s4l0-kernels-off` and `s4l0-kernels-on` (fork build `f254fc7f6`, Q4_0/Q4_1/Q8_0 kernels), `s4l0-kernels-on-q5_0` (`112cdc597`, with the Q5_0 kernel) |
| `profile_classes.csv` | the same grouped by tensor class and type (the table below) |
| `profile_totals.csv` | per run: GPU ms per token summed over the cards, per card, and the profiled wall time per token |
| `phost-s4l0-mode0.csv`, `phost-s4l0-mode1.csv` | the fork's host timeline (`GGML_CUDA_PROFILE_HOST`), kernels off (0) and on (1): one line per card and graph evaluation with host entry and exit time and the evaluation's GPU time (registry row Q49) |
| `s4l0-ops.txt` | every distinct operation of S4-L0's one-token and 512-token graphs, from `test-export-graph-ops` (82 lines) |
| `s4l0-matmul-ops.txt` | its 24 matrix products (`MUL_MAT`, `MUL_MAT_ID`), the replay set for `test-backend-ops --test-file` |
| `run/run2.sh` … `run/run6.sh`, `run/prof.sh` | the rig runs, in order: tests and profiles (2), the Q5_0 build and KLD (3), four more files off/on (4), host timeline (5), the noise floor (6) |
| `run/prof_to_csv.py`, `run/profile_classes.py` | turn `GGML_CUDA_PROFILE` traces into the profile CSVs |

GPU time per token of S4-L0 on 6 cards (ms, the last 24 of 32 generated tokens, summed over the cards):

| Tensors | Type | Launches per token | Kernels off | Kernels on | With the Q5_0 kernel |
|---|---|---|---|---|---|
| attention q | Q4_1 | 49 | 3.71 | 2.71 | 2.68 |
| attention k, v | Q8_0 | 98 | 2.52 | 2.07 | 2.09 |
| attention output, layers 0–32 | Q8_0 | 33 | 3.78 | 2.69 | 2.69 |
| attention output, layers 33–48 | Q4_1 | 16 | 1.15 | 0.95 | 0.93 |
| routed experts gate/up | Q5_0 | 47 | 8.20 | 8.21 | 5.01 |
| routed experts down | Q8_0 | 47 | 5.21 | 2.97 | 2.96 |
| shared expert gate, up | Q8_0 | 94 | 3.44 | 2.65 | 2.66 |
| shared expert down | Q8_0 | 47 | 2.61 | 1.71 | 1.70 |
| dense FFN, layers 0–1 | Q8_0 | 5 | 1.00 | 0.63 | 0.62 |
| output head | Q8_0 | 1 | 3.29 | 2.26 | 2.20 |
| router | F16 | 47 | 1.21 | 1.19 | 1.18 |
| everything else | | 536 | 8.05 | 7.97 | 7.96 |
| **total** | | | **44.17** | **36.01** | **32.70** |

Each launch-group boundary is a CUDA event, which slows these cards (registry row Q49), so the profiles compare kernels
but don't give absolute token times. The host timeline (`scripts/cuda_profile_summary.py phost-s4l0-mode1.csv
--tokens 48`) shows the GPU chain at 37.91 ms (kernels off) and 26.58 ms (on) per token, with the host entering all
six cards' graphs within 1.9 ms of a token's start.

Made on 2026-10-23. The microbenchmark ran on one rig card from the repository root, as in
[bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md) ("Running on the rig"), with the benchmark as
committed in `52a8f4d`:

```bash
RIG=${RIG:-rig}
bench/mmvq_pascal/build.sh
scp bench/mmvq_pascal/build/mmvq_pascal "$RIG":/tmp/
ssh "$RIG" 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --csv /tmp/mmvq_step1.csv'
ssh "$RIG" 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --types q8_0,q5_0,q4_0,q4_1 --csv /tmp/mmvq_staged.csv'
```

The engine runs used the fork builds named in each script (branch `pascal-p106` at the commits shown) under
`/opt/llama-builds/` on the rig, copied there with a copy of [scripts/bench.sh](../../../scripts/bench.sh):

```bash
ssh "$RIG" 'mkdir -p /tmp/q38'
scp scripts/bench.sh docs/data/2026-10-23_q38/run/*.sh "$RIG":/tmp/q38/
ssh "$RIG" 'cd /tmp/q38 && for i in 2 3 4 5 6; do nohup bash run$i.sh > run$i.log 2>&1 & done'
```

Each later script waits for the `DONE` line of the one before. The profile CSVs were made with
`python3 run/prof_to_csv.py --tokens 24 profile s4l0-kernels-off=prof-s4l0-mode0.csv s4l0-kernels-on=prof-s4l0-mode1.csv
s4l0-kernels-on-q5_0=prof-s4l0-q50.csv` and `python3 run/profile_classes.py profile_by_label.csv > profile_classes.csv`;
the raw traces (1.9 MB each), the test-backend-ops and KLD logs and the llama-bench outputs are not published (their
results are in the registry row). `prof.sh`'s wide-rows profile was not kept. The operation lists were exported from a
sparse copy of the S4-L0 file (its first 64 MB and full length are enough, since only the metadata is read):

```bash
test-export-graph-ops -m flint-S4L0.gguf -o s4l0-ops.txt
grep -E '^(29|30) ' s4l0-ops.txt > s4l0-matmul-ops.txt     # GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID
```
