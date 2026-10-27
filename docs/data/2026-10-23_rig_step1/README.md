# Q34 step 1: the first matrix-vector microbenchmark and the 2-warp launch shape on the rig (2026-10-23)

The first run of [bench/mmvq_pascal](../../../bench/mmvq_pascal/) on a P106-090, and registry row Q34's step 1
([quants/experiments.md](../../../quants/experiments.md)): llama.cpp builds with 2 warps per block at one token
([patch](../../../patches/mmvq-pascal-2warps.diff)) and with 2 warps only for rows ≥ 2048 wide
([patch](../../../patches/mmvq-pascal-2warps-widek.diff)), against the same build without them, on five Flint files.
The microbenchmark's per-type totals and the Q4_1 per-shape table are in
[bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md) ("Results on the P106-090"); the speeds are in the
Q34 row.

| File | What |
|---|---|
| `mmvq_pascal.csv` | the microbenchmark on one card: Q4_1, Q4_1_64, Q4_0 and Q8_0 at Flint's eight matrix shapes, variants `llama`, `pr25479`, `p_r1c2`, `p_r2c2`, `p_r4c2`, `p_r2c4`, `soa_r2`, `soa_r4`: median/min/max µs per call, GB/s, registers, occupancy, grid and the correctness check, plus per-token totals and the graph-node and streaming-read baselines; 261 rows |
| `step1_tg128.csv` | runs A–M: build, build directory, file, rescale on/off, cards, llama-bench pp512 and tg128 (`-fa on`) ± standard deviation and the three tg128 repetitions |
| `run/step1_tg128.py` | builds `step1_tg128.csv` from the runs' llama-bench JSONL |

The builds (all upstream `d77256999`): **graphs61** is the standard build of that day (CUDA graphs on compute 6.1, the
MoE down-projection rescale switch `LLAMA_NO_MOE_DOWN_RESCALE`, the `Q4_1_64` type); **+ 2 warps** adds the 2-warp
launch shape; **wide-K** applies it only to rows ≥ 2048 wide; **stock** has no patches.

| Run | File | Cards | Build | tg128 (tok/s) |
|---|---|---|---|---|
| A → B | makers' Q4_1 | 5 | graphs61 → + 2 warps | 28.07 → 31.82 |
| C → D | makers-fast (rescale off) | 5 | graphs61 → + 2 warps | 30.91 → 35.51 |
| K, E → F | Unsloth UD-Q4_K_XL | 6 | stock, graphs61 → + 2 warps | 17.56, 19.53 → 24.80 |
| J | Unsloth UD-Q4_K_XL | 6 | wide-K | 24.58 |
| G → H → I | Miner S4-L0 (rescale off) | 6 | graphs61 → + 2 warps → wide-K | 23.09 → 26.29 → 28.02 |
| L, M | Unsloth Q8_0, rescale on / off | 9 | wide-K | 25.15, 26.78 |

The same day the wide-K build served real requests at up to 262,144 tokens of context with
[scripts/server_bench.sh](../../../scripts/server_bench.sh) (four one-shot prompts, 256 tokens each, all nine cards):

| File | Context | Generation (tok/s) | Busiest card (of 6,068 MiB) |
|---|---|---|---|
| Miner S4-L0 | 262,144 | 24.72 | 5,253 MiB |
| Unsloth UD-Q4_K_XL | 262,144 | 20.36 | 4,363 MiB |
| Unsloth Q8_0, rescale off | 131,072 | 22.86 | 5,785 MiB |
| Unsloth Q8_0, rescale off, `-ts 6,6,5,5,6,5,5,6,6` | 262,144 | 22.56 | 5,915 MiB |

Q8_0 at 262,144 tokens with the default split ran out of memory on one card while allocating, with the 16-bit KV cache
and with an 8-bit one.

Made on 2026-10-23 on the rig. The microbenchmark with the benchmark as committed in `d906e21`, as in
[bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md) ("Running on the rig"). The llama-bench runs with
[scripts/bench.sh](../../../scripts/bench.sh), one per run letter, e.g. run I:

```bash
RIG=${RIG:-rig}
ssh "$RIG" "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=/opt/llama-builds/llama.cpp-20261023-d77256999-graphs61-moe-down-rescale-env-flint-prism-hadamard-q4_1_64-type-mmvq-pascal-2warps-widek/bin MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s1-I GPU_SETS=0,1,2,3,4,5 BENCH_ARGS='-ngl 99 -fa on -p 512 -n 128 -r 3' bash -s" < scripts/bench.sh
```

Runs with the rescale on drop `LLAMA_NO_MOE_DOWN_RESCALE=1`; `step1_tg128.csv` lists each run's build directory, file
and cards. The table was made with `python3 run/step1_tg128.py <dir with s1-A … s1-M>`. The server runs used
`PROMPTS` left at its default and the context sizes above. The `nvidia-smi` traces, llama-bench logs and server logs
are not published.
