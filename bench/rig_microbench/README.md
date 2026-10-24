# rig_microbench: the one-card microbenchmarks in one run

`run_block8.sh` runs every microbenchmark that needs one P106-090 and no model, one after the other, on one card. It
logs everything to one folder and writes a summary of the numbers that decide each question:

| Step | Question | Program | Decides | Time (est.) |
|---|---|---|---|---|
| 1 | Q101: is the Pascal Q6_K kernel fast enough to make Q6_K the 6-bit rung? | [mmvq_pascal](../mmvq_pascal/README.md#q6_k-q101) `--types q6_K,q5_K,q8_0` | ≥ 130 GB/s on the dense K = 2048 shapes (the summary marks the gate met or not) | 2 min |
| 2 | Q101 in llama.cpp: the same products in the fork's `pascal-p106-v2-q6k` build, 1–4 tokens per step | [multitoken_perf.sh](../mmvq_pascal/multitoken_perf.sh) with `TYPES='(q6_K|q5_K|q8_0)' RUNS="off on"` | the Pascal kernels against llama.cpp's inside the fork | 10–15 min |
| 3 | Q99: how many 2 MiB pages does the GPU's TLB cover, and what does a page walk cost? | `mmvq_pascal --tlb-probe` | the knee of the step time against footprint | 5–10 min |
| 4 | Q99: do the kernels slow down when their weights spread over gigabytes? | `mmvq_pascal --spread-gb 0.5…5` | MoE calls < 10 µs slower at 5 GB than at 0.5 GB kills the page-walk theory | 8 min |
| 5 | Q103: does the card throttle an instruction class? | [insn_probe](../insn_probe/README.md) | each class's rate per SM clock against its nominal rate | 1 min |
| 6 | Q88: is one persistent kernel per layer faster than the fork's 13 launches? | [persistent_pascal](../persistent_pascal/README.md) `--check-only`, then S4-L0 and G3 timings and `--grid 5` | the speed-up per layer and per token | 6 min |

A step that fails is logged, and the next step runs. The whole block takes 35–60 minutes.

## Running it

The bundle holds the three binaries (`mmvq_pascal`, `insn_probe`, `persistent_pascal`, all built for sm_61 without
AVX), the scripts and `MD5SUMS`; the script checks the sums before it starts. Step 2 uses the fork's q6k build on the rig
(`Q6K_BIN`, default `/opt/llama-builds/llama.cpp-20261024-03b92cf10-pascal-p106-v2-q6k/bin`).

```bash
ssh rig "mkdir -p /tmp/b8 && tar xzf /tmp/rig-block8-20261024.tar.gz -C /tmp/b8"
ssh rig "cd /tmp/b8/rig-block8-20261024 && setsid nohup ./run_block8.sh > console.log 2>&1 < /dev/null &"
ssh rig "tail -f /tmp/b8/rig-block8-20261024/out/block8.log"
ssh rig "cat /tmp/b8/rig-block8-20261024/out/summary.txt"
```

Env: `GPU` (default 0), `STEPS` (e.g. `"1 5"`), `OUT`, `Q6K_BIN`, `FORCE=1` (start even with a compute process on the
card). `out/` holds a log and CSV per step, `smi.csv` (the card's clocks, power, temperature and clock event reasons
every 500 ms) and `summary.txt`. `test/dryrun.sh` runs the script against stub programs on any Linux machine.
