# Q566: DFlash drafters on Qwen3.8-Flash-Next in Miner Llama v3 (2026-11-03)

A DFlash drafter reads the target model's hidden state at a few of its layers (Inco AI's DFlash2 drafter for
Flash-Next reads layers 5, 15, 25, 35 and 45) as `[n_embd, tokens]` rows. Flash-Next (`qwen4exp`) keeps four parallel
residual streams (hyper-connections), and before upstream llama.cpp #30927 it exposed the four-stream residual
`[n_embd, 4, tokens]` as the layer input. The first DFlash request then aborted in `llama_context::extract_layer_inputs`
at `GGML_ASSERT(row_floats == n_embd)`. The fix gives the drafter the plain mean of the four streams, built only for the
layers a drafter asks for, as deepseek4 does for its own hyper-connections. The registry row is Q566 in
[quants/experiments.md](../../../quants/experiments.md); why DFlash2 is the first drafter to try on the rig is in
[flash_next_drafters.md](../../research/notes/flash_next_drafters.md) (Q561).

## The branch

`pascal-p106-v3-dflash` on Miner Llama's fork, off `pascal-p106-v3` (`37e717ff6`):

| Commit | What |
|---|---|
| `dd693b72f` | upstream `d9ee007a0` (#30927, `src/models/qwen4exp.cpp`, +19/−1), cherry-picked with its author; it applies without changes |
| `2deed575e` | `test-backend-ops`: `HC_STREAM_MEAN`, the fix's graph as a test case |

The rest of the DFlash path was already in the v3 base: DFlash (#22105), DFlash2 (#27816), rollback of `qwen4exp`'s
recurrent state (#28123), sharing the target's output head (#30111) and `--spec-draft-n-ubatch` (#30459). With the
fix, `src/models/qwen4exp.cpp`, `src/models/dflash.cpp` and `common/speculative.cpp` are the same as upstream's.
Miner Llama's changes to `src/llama-context.cpp` leave the layer-input code alone, and in `src/llama-graph.cpp` they add
one more reader of the layer inputs: the staircase drafter's taps (Flint's pipelined speculation), kept as outputs
only while that drafter is loaded. The checks below cover the ops the fix adds, on the CUDA backend Miner Llama
ships.

The same two commits apply without conflicts to the pipelined-speculation branch `pascal-p106-v3-hybrid` (Q550) and the
BF16 kernel branch `pascal-p106-v3-bf16mv` (Q546). Only `pascal-p106-v3-dflash` was built and checked.

## What the fix computes

For each tapped layer, from the residual `x` of shape `[n_embd, hc, T]`:

```
acc = view(x, stream 0) + view(x, stream 1) + ... + view(x, stream hc-1)    # strided [n_embd, T] row views
layer_inp = acc * (1 / hc)
```

Each view has row stride `x->nb[2]` and offset `s * x->nb[1]`, so the first ADD has a non-contiguous first source,
and the ADDs form a chain. llama.cpp's CUDA backend fuses chains of ADDs and also matches views of one tensor followed
by a chain of ADDs (the MoE expert sum, #25952), so the test case builds exactly these nodes and runs them as one
graph.

## Result: correct on the PTX the P106 runs

All runs used `test-backend-ops` from the rig tarball on an RTX 3090, through the build's compute_61 PTX (the device
code the P106's sm_61 machine code is compiled from), comparing CUDA with the CPU backend at the suite's tolerances.
No timings were taken. The ADD and SCALE kernels have no compute-capability branches, so the 3090 runs the code a
P106 runs.

| Run | Cases | Result |
|---|---|---|
| `HC_STREAM_MEAN`: n_embd 2,560 and 4 streams at 1, 4, 64 and 512 tokens; 64 × 2 streams × 7; 1,025 × 3 × 5 | 6 | 6/6 |
| the same with the CUDA fusions off (`GGML_CUDA_DISABLE_FUSION=1`) | 6 | 6/6 |
| every `ADD` case (broadcasts, permuted and overlapping sources, fused chains) | 167 | 167/167 |
| `ADD_ADD` (two fused ADDs, a non-contiguous first source among them) | 12 | 12/12 |
| every `SCALE` case | 7 | 7/7 |

No case was unsupported on CUDA, so on the rig these ops stay on the card that holds the layer.

**Not covered here:** a request with the real model and drafter. That is the rig's smoke test (Q561), which also
compares greedy output with the same build run without the drafter. The two agree, or split late at a near-tie: the
verify step runs several tokens through multi-token kernels, which sum in a different order than one-token decoding.

## Running it on the rig

The flags exist under these names in this build (`common/arg.cpp`). The line keeps the settings of Q215's v3 line and
the split of Q226's draft arm, which leaves the last card room for a draft:

```bash
llama-server -m <Swift Q2_0 shard 1> -ngl 99 -fa on -c 65536 -np 1 -b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio \
  -md Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-n-ubatch 64 \
  -devd CUDA8 -ngld 99 ...
```

- `-devd CUDA8 -ngld 99`: the output head goes to the last card in the split (`llama_model::dev_output`), and a
  DFlash2 drafter uses the target's head and token embeddings (`src/models/dflash.cpp`), so its layers belong on that
  card. `-ts …,3` leaves that card room for the drafter's ~0.6 GB of weights and its buffers.
- `--spec-draft-n-max 3` drafts 3 tokens per round (at most block size − 1 = 7 for this drafter); each drafted token
  keeps one more recurrent-state copy for rollback, ~113 MiB over all cards.
- `-b 512`: the layer inputs are copied to a host buffer of 5 taps × 2,560 × `-b` floats, 25 MiB at `-b 512` against
  100 MiB at `-b 2048`.

## Files

| File | What |
|---|---|
| `tbo.csv` | the five runs above: filter, environment, cases, passed, failed, unsupported, exit status |
| `run/check.sh` | the runs, one container each, on one GPU |

## How it was made

On 2026-11-03, on a 7-GPU machine (one RTX 3090 used), in `nvidia/cuda:12.9.2-devel-ubuntu24.04`. From the repository
root, with `WORK` a work folder on the build machine:

```bash
# the rig build (sm_61 and compute_61 PTX, no AVX), with test-backend-ops
WORK=$WORK TESTS=1 JOBS=8 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-dflash
# the five runs on GPU n
BIN=$WORK/llama.cpp-20261103-2deed575e-pascal-p106-v3-dflash WORK=$WORK GPU=n docs/data/2026-11-03_dflash_fix/run/check.sh
```

The build is `llama.cpp-20261103-2deed575e-pascal-p106-v3-dflash` (293,466,637 bytes, md5
`c509a5d406d0ef536591d4f96c54c1e7`; 0 AVX instructions, highest glibc symbol 2.38). The test inputs are random; no
model file or text is involved.
