# Q88: the persistent MoE-layer kernel against the fork's 13 launches on one P106 (2026-10-24)

The microbenchmark that decided to build the persistent MoE kernel into the fork: `bench/persistent_pascal` on one
P106-090 (card 0, nothing else on it), step 6 of [bench/rig_microbench/run_block8.sh](../../../bench/rig_microbench/README.md).
One MoE layer of Flint per call, the fork's 13 launches against one persistent kernel (`r1`: one routed gate/up row per
warp pass, the fork's launch shape; `r2`: two), for the S4-L0 and G3 tensor types, plus a run with one block per SM.
Result: registry row Q88 in [quants/experiments.md](../../../quants/experiments.md) and
[pascal_mmvq_integration.md](../../research/notes/pascal_mmvq_integration.md) section 17; the fork's version is measured
in [2026-10-24_q88_persistent](../2026-10-24_q88_persistent/README.md). Steps 1–5 of the block (Q101, Q99, Q103) were not
run (`STEPS=6`), so their sections in `summary.txt` are empty.

| File | What |
|---|---|
| `q88_s4l0.csv`, `q88_g3.csv` | per tensor mix (`format`: S4-L0 Q5_0/Q8_0 experts, G3), one row per measurement: occupancy (registers, shared memory per block, blocks per SM, grid) of `r1` and `r2`, the CPU reference check, µs per layer of the 13-launch path, `r1` and `r2` (`timing`), each of the 13 kernels alone (`multi_launch_kernel`), and the in-kernel trace of one `r1` call (start and end of each stage in µs, wait polls) |
| `summary.txt` | the block's summary: weights read per layer and token, the 13-launch path and both persistent variants in µs per layer with GB/s and the speed-up, for S4-L0, G3 and S4-L0 with one block per SM (`--grid 5`) |
| `smi.csv` | nvidia-smi of card 0 every 500 ms during the step: timestamp, P-state, SM and memory clock (MHz), power (W), temperature (°C), clock event reasons |

## What it shows

| Mix | 13 launches | persistent `r1` | persistent `r2` |
|---|---|---|---|
| S4-L0, 10 blocks | 198.5 µs per layer, 109.6 GB/s | 173.2 µs, 1.146× | 171.3 µs, 1.159× (−1.28 ms per token over 47 layers) |
| G3, 10 blocks | 203.1 µs, 113.6 GB/s | 176.0 µs, 1.154× | 173.8 µs, 1.168× |
| S4-L0, 5 blocks (one per SM) | 203.6 µs | 231.4 µs, 0.880× | 224.0 µs, 0.909× |

The kernel's outputs equal the 13-launch graph's in every timed run, and the CPU reference self-test passes (worst
2.6e-9 of the sum of |terms|). Two blocks per SM are needed: with one, the persistent kernel is slower than the 13
launches. In the fork the gain per token was +5.6–8.8 % (Q88 row).

## How it was made

On 2026-10-24 14:39–14:40, from the bundle of [bench/rig_microbench](../../../bench/rig_microbench/README.md) unpacked on
the rig (the `persistent_pascal` binary built for sm_61 from [bench/persistent_pascal](../../../bench/persistent_pascal/)):

```bash
ssh "$RIG" 'cd /tmp/b8/rig-block8-20261024 && GPU=0 STEPS=6 OUT=/tmp/b8/rig-block8-20261024/out ./run_block8.sh'
```

Step 6 runs `persistent_pascal --check-only --sets 2 --trials 2`, then `--no-check --format s4l0 --csv q88_s4l0.csv`,
`--no-check --format g3 --csv q88_g3.csv` and `--no-check --no-trace --grid 5`. The weights are random data in Flint's
layer shapes; nothing model-derived is in these files.
