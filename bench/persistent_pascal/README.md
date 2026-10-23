# persistent_pascal: one Flint MoE layer as one persistent kernel on the P106-090

When an LLM generates a token on these cards, most of the time goes into streaming weights, but not all of it: in a
profile of one [Flint](https://huggingface.co/RecursiveSI/flint) token on six P106-090 cards with this project's
llama.cpp fork, ~9 ms of the 26.8 ms are kernel boundaries, ramp-ups and tails of small kernels, and the non-matrix
work between them ([breakthrough candidates](../../docs/research/notes/breakthrough_candidates.md), section 2). Flint's
makers run a whole token as one persistent kernel on Apple GPUs ([rec](../../docs/research/notes/rec_engine.md)). This
benchmark is the first step towards that on the P106: **one MoE layer of Flint at one token as a single kernel launch**,
next to the same layer as the 13 launches the fork makes, in one binary, so a single card gives a direct A/B.

It runs on one card (compute capability 6.1, 5 SMs, ~179 GB/s) and reuses the tuned Pascal dot products of
[mmvq_pascal](../mmvq_pascal/) in the form the fork runs them (branch `pascal-p106-v2`: staged loads, FP32 activation
sums).

## One MoE layer today: 13 launches

Flint has 49 layers; 47 of them end in a mixture-of-experts block: 256 routed experts (intermediate 512) of which a
sigmoid router picks 6, plus a shared expert (intermediate 1,024), on a hidden size of 2,048. For one token the fork
(with the shared-expert fusion of registry Q50) launches:

| # | Kernel | Weights read (S4-L0) | Real token, rig profile |
|---|---|---|---|
| 1 | RMS norm × weight | | 10 µs |
| 2 | router, 2,048 × 256 F16 (`mul_mat_vec_f`, FP32 accumulation on compute 6.1) | 1.05 MB | 26 µs |
| 3 | sigmoid, top-6, normalisation (`topk_moe`) | | 14 µs |
| 4–5 | Q8_1 quantization of the normed input; routed gate/up + SwiGLU, Q5_0 (`MUL_MAT_ID`) | 8.65 MB | 109 µs |
| 6–7 | quantization of the 6 × 512 SwiGLU outputs; routed down, Q8_0 | 6.68 MB | 63 µs |
| 8 | weighted sum of the six experts | | 10 µs |
| 9–10 | quantization of the normed input again; shared gate/up + SwiGLU, Q8_0 | 4.46 MB | 65 µs (unfused: gate 28, up 28, SwiGLU 9) |
| 11–12 | quantization of the shared SwiGLU output; shared down, Q8_0, plus the routed sum | 2.23 MB | 35 µs |
| 13 | residual add | | 10 µs |
| | **total** | **23.07 MB** | **~340 µs** |

The last column is the fork's event profile of S4-L0 on six cards (`GGML_CUDA_PROFILE`, before Q50; median over
1,363 layer evaluations): each launch group carries ~5–8 µs of event overhead, so the layer takes roughly 270 µs
(~245 µs with Q50) where streaming its 23.07 MB at the 150 GB/s the Pascal kernels reach would take 154 µs. G3 (layers
9–45) reads its routed down in Q5_0 and, as its recipe leaves it, an F32 router: 21.76 MB per layer.

## The persistent kernel

One launch of `k_moe_layer` ([src/persistent.cuh](src/persistent.cuh)) does the whole layer for one token: RMS norm,
router, sigmoid and top-6 with renormalisation, one quantization of each activation vector, the six routed experts'
gate/up with SwiGLU and their down projections, the shared expert, the weighted sum and the residual.

**Work list.** The layer is cut into 2,944 warp-sized work items, numbered so that every dependency points to a smaller
number:

| Items | Count | Each | Waits for |
|---|---|---|---|
| router | 128 | 2 rows | – |
| shared gate/up | 256 | 4 rows (gate and up) | – |
| routed gate/up | 768 | 4 rows of one expert, experts in turn | the top-6 |
| shared down | 256 | 8 rows | the shared activations |
| routed down | 1,536 | 8 rows of one expert, by row group: g0 e0..e5, g1 e0..e5, … | that expert's activations only |

A warp takes the next item from a counter in global memory, and claims its following item before it works on the
current one, so the atomic's latency hides behind the work. Every block first computes the RMS norm of the layer
input and its Q8_1 copy into its own shared memory (the norm needs the whole vector, and doing it per block costs one
8 KB read instead of a dependency).

**Dependencies are counters, not barriers.** There is no grid-wide barrier:
- the warp that finishes the last router item computes the top-6 and publishes it; until then the shared expert's
  gate/up keeps every warp busy, since it does not depend on the routing;
- the warp that finishes the last shared gate/up item quantizes the 1,024 shared activations once and publishes them;
  the same per routed expert for its 512 activations, so each expert is a team of its own: an expert's down
  projections start as soon as its own gate/up is done;
- each 8-row output group counts its seven contributions (six routed experts and the shared expert); the warp that adds
  the seventh computes those rows' weighted sum, adds the shared expert and the residual, and writes the layer output.
  Nobody waits for it.

Producers store their results, `__threadfence()`, then count; consumers poll the flag, fence, and read the data with
`ld.global.cg` (L2), since the read-only path is not coherent with writes made during the same launch.

**It cannot deadlock, at any grid size.** A work item only ever waits for items with smaller numbers. A warp takes items
in increasing order and holds at most two (the one it works on and the one it claimed next). So the smallest unfinished
item always belongs to a warp that is working on it, and all of its inputs are complete: it finishes, and by induction
every item does. This does not assume that all blocks are resident at once: a block that starts late finds the list
empty and leaves. The kernel therefore needs neither a cooperative launch nor an occupancy guarantee, and it runs
unchanged on 5 SMs or 82. A wait longer than `spin_limit` polls (a bug, or a fault injected on purpose) sets an error
flag, every warp leaves, and the host runs the 13-launch path instead. The last warp to leave resets the counters for
the next launch.

**Same arithmetic, bit for bit.** Every product, reduction, quantization and sum is the same device function the
13-launch path calls ([src/fork_kernels.cuh](src/fork_kernels.cuh), copied from the fork), fed with the same values:
- the norm runs 256 threads standing in for the norm kernel's 512, with the same warp and block reductions; the normed
  values are not stored: the quantization and the router recompute each one from the input with the same two
  multiplies;
- a router row is one warp standing in for `mul_mat_vec_f`'s 256-thread block: eight partial sums per lane, then the
  same two butterfly reductions;
- each expert row is computed by the fork's lane split (lanes per row, blocks per lane, staged loads, reduction order);
- the top-6 is the fork's warp code; the weighted sum, the fused add of the shared expert and the residual add keep
  their order.

So the persistent kernel's output equals the 13-launch path's exactly, and a model run with it would give the same
logits.

**Two variants.** `r1` uses the fork's launch shape everywhere. `r2` lets a warp pass cover two routed gate/up rows
instead of one: the rows share the activation load, each row's sum is unchanged, and a warp has twice the bytes in
flight (2.8 → 5.6 KB). Both are checked; the rig decides.

**Resources** (sm_61, 256 threads per block): 128 registers, no spills; 37 KB (`r1`) or 47 KB (`r2`) of shared
memory per block, mostly each warp's staging buffer for the weight rows. Two blocks fit per SM (64 K registers; 96 KB
of shared memory), i.e. 16 warps per SM and a grid of 10 blocks on a P106. The fork's product kernels run 20 warps per
SM. On an RTX 3090 (82 SMs) the grid is 164 blocks.

### What adding attention would take

The same scheme extends to the rest of a layer, at the cost of three more layer-wide dependencies:
- Q/K/V products as row items; the per-head RMS norms and RoPE by the warp that completes a head (a team counter per
  head, as per expert); the KV-cache write by the same warp;
- attention as items per KV head and context partition (rec's layout: a lane owns 4 of the 128 dimensions for all 8
  query heads of a KV head), partial softmax states merged by the warp that completes a head;
- the output projection waits for all 32 heads (a full dependency, like the top-6), and the FFN's norm waits for all
  2,048 rows of the residual. With per-block prologues no longer possible mid-kernel, the warp that completes the last
  row would compute the norm and quantization once and publish them, as the top-6 is published here.
- Each full dependency stalls the card unless independent work fills it. Within a layer there is little; across
  layers, the next phase's weight rows can be prefetched into L2 (`prefetch.global.L2`) while a warp waits.

A whole token per card then needs one such kernel per card per token, with the hop to the next card through mapped
host memory or a host launch, as today.

## Building

Needs Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image (CUDA 13 can no longer build for Pascal):

```
bench/persistent_pascal/build.sh
```

This writes `bench/persistent_pascal/build/persistent_pascal` with sm_61 machine code for the P106, compute_61 PTX,
and sm_86 and sm_120 machine code for RTX 30- and 50-series GPUs (correctness runs only). Host code has no AVX and the
CUDA runtime is linked statically, as for `mmvq_pascal`. `build/ptxas.log` and `build/res-usage.txt` list registers
and shared memory per kernel; `build/sm61.sass` is the sm_61 disassembly. The build takes about two minutes.

## Checking correctness

```
bench/persistent_pascal/build/persistent_pascal --check-only --sets 2 --trials 3                        # native code
CUDA_FORCE_PTX_JIT=1 bench/persistent_pascal/build/persistent_pascal --check-only --sets 2 --trials 3   # the Pascal code path, JIT-compiled from compute_61 PTX
bench/persistent_pascal/build/persistent_pascal --no-check --smoke --sets 2 --chain 8                   # the timing code path, once, untimed
```

Random weights in the layer's formats, random inputs; every check prints its own line and the program exits with
status 2 if any fails:
- **Against the 13-launch path, bit for bit**, for both variants: the normed input and its Q8_1 copy, the 256 router
  logits, the six expert ids and weights, the routed and shared SwiGLU outputs and their Q8_1 copies, the six routed
  down outputs and the layer output.
- **Against double precision, stage by stage**, for both paths, each stage computed on the CPU from the path's own
  inputs to that stage (worst error over tolerance, must be ≤ 1): norm (4·10⁻⁶ relative), each Q8_1 quantization
  (scale within FP16 rounding, every code equal to round(x/d) except at a rounding boundary), router logits and every
  product (10⁻⁵ of the sum of absolute terms), the top-6 set and order (near-ties within 10⁻⁶ allowed) and the
  normalised weights, the weighted sum and the output.
- **The whole layer in double precision without quantizing any activation**: the relative L2 error of the layer's update
  (output minus input), which is the error of 8-bit activations, and how often the exact expert set differs.
- **Grid sizes**: 1, 2, one block per SM, the default, 4 × the default + 3, and 64 blocks per SM (far more than can be
  resident): the output must equal the 13-launch path's bit for bit each time, and the counters must be back at zero.
- **Fault injection**: the top-6 is never published; every wait must time out, the host must detect it and fall back to
  the 13-launch path with the correct output, and the next launch must work again.
- **Chains** of 16 layers, each fed the previous output: every layer's output identical between the paths.
- **The CPU reference itself**: its dequantization of each weight type the layer uses against `mmvq_pascal`'s
  `ref_dot` on random rows.
- **The timing code path** (`--smoke`, and on every timing run): the 47-layer CUDA graphs of the persistent kernel must
  leave every layer's output identical to the 13-launch graph's, and the traced build must count every work item once.
  `--smoke` captures and launches each graph once and times nothing.

**Results** on an RTX 3090, natively (sm_86) and through the compute_61 PTX (the P106's code path), for both formats,
with two weight sets and three random inputs each:
- Both variants are identical to the 13-launch path bit for bit in all 6 evaluations of every run, in every intermediate
  and the output; the 16-layer chains are identical at every layer, and so are the CUDA graphs the timing captures
  (`--smoke`, 8 layers).
- Every grid size, from 1 block (8 warps) to 5,248 blocks (64 per SM, 41,984 warps), gives the same output, and the
  counters are back at zero after each run.
- Fault injection: the timeout is detected, the fallback output is correct, and the next launch is correct again.
- Stage checks, worst error over tolerance (the same for both paths): router, top-6 and every product ≤ 0.004, norm
  0.04, weighted sum and output ≤ 0.25, the Q8_1 quantizations ≤ 0.95 (the FP16 rounding of each block's scale).
- The CPU reference's dequantization agrees with `mmvq_pascal`'s `ref_dot` to 2.6·10⁻⁹ of the sum of |terms|.
- Against a double-precision layer with unquantized activations, the layer's update (output minus input) is off by
  1.1–1.2 % (relative L2), the cost of 8-bit activations in both paths; the expert sets agree in every evaluation.
- On the RTX 3090 the persistent kernel gets 2 blocks per SM (16 warps), a grid of 164 blocks for the 2,944 items.

## Running on the rig

From the repository root, with nothing else running on card 0 (the binary needs ~2.7 GB of its memory for four
layers' weights):

```
bench/persistent_pascal/build.sh
scp bench/persistent_pascal/build/persistent_pascal rig:/tmp/
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/persistent_pascal --check-only --sets 2 --trials 2' | tee persistent_check_rig.txt
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/persistent_pascal --no-check --format s4l0 --csv /tmp/persistent_s4l0.csv' | tee persistent_s4l0_rig.txt
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/persistent_pascal --no-check --format g3 --csv /tmp/persistent_g3.csv' | tee persistent_g3_rig.txt
ssh rig 'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/persistent_pascal --no-check --no-trace --grid 5' | tee persistent_s4l0_grid5_rig.txt
scp 'rig:/tmp/persistent_*.csv' .
ssh rig 'rm -f /tmp/persistent_pascal /tmp/persistent_*.csv'
```

The check computes every stage in double precision on the rig's Celeron (est. one to two minutes); each timing run
takes about a minute (est.). The last run puts one block on each SM (8 warps per SM) instead of two, to see whether
the persistent kernel is limited by warps in flight.

**What a timing run does.** It streams memory for ~2 s to reach the sustained clock, measures an empty CUDA-graph node
and the streaming-read bandwidth, then captures 47 layers (Flint's MoE layers per token; four weight sets in turn,
each layer fed the previous layer's output) as one CUDA graph per path and times 30 launches of it. It prints µs per
layer for the 13-launch path and for each persistent variant, the speed-up, and the saving per token over 47 layers;
then each of the 13 kernels alone (47 launches of it per graph), and a traced run of each variant (`%globaltimer`
timestamps): when the prologue, the router, the top-6, each item class, the activations and the output groups finish,
and how many polls the waits took.

**Expected** (S4-L0, est.):
- 13-launch path: ~190 µs per layer from the microbenchmark's per-shape speeds (gate/up 63, routed down 46, shared
  gate/up 30, shared down 16, router ~10–15 µs) plus ~25–35 µs for the eight small kernels. If it comes out near the
  real token's ~245 µs instead, the bench reproduces the slowdown the fork's routed calls show in real tokens.
- Persistent: the 23.07 MB at 140–155 GB/s plus a few µs of prologue and tail: **150–170 µs per layer**.
- So a saving of 20–40 µs per layer against the bench's own baseline (1.12–1.27×) and of 75–95 µs against the real
  token's layer, i.e. **0.9–4.5 ms per token** over 47 layers: 26.8 → 22.3–25.9 ms on six cards (+4 to +20 %).

**How it scales to a token.** Layer split runs the cards one after another, so a token's time is the sum over all its
layers: the per-layer saving × 47 is the saving per token whatever the card count (six or nine cards only change how
the layers are distributed). The 2 dense layers, attention and the output head are not included; attention is the next
step (above).

**Decision.** The registry's gate for this idea (Q88) is a fused layer at ≥ 1.3× the sum of the fork's kernels for that
layer. Read it against both baselines: the bench's 13-launch path (same binary, same data) and the real token's
~245 µs. If the persistent kernel streams below ~130 GB/s, the trace shows where (waits at the top-6 or the
activations, a slow prologue, a long tail) and `--grid 5` against the default shows whether warps in flight are the
limit.

## Options

| Option | Effect |
|---|---|
| `--format s4l0\|g3` | S4-L0 (routed gate/up Q5_0, routed down Q8_0, shared expert Q8_0, router F16) or G3's layers 9–45 (routed down Q5_0, router F32) |
| `--router f16\|f32` | override the format's router type |
| `--sets N` | distinct layers in GPU memory, used in turn (default 4; 0.66 GB each for S4-L0, 0.56 GB for G3) |
| `--chain N` | layers per timed CUDA graph (default 47) |
| `--reps N` | timed graph launches, median reported (default 30) |
| `--trials N` | random inputs per weight set in the checks (default 4) |
| `--grid N` | blocks of the persistent kernel (default: resident blocks per SM × SMs) |
| `--variant r1\|r2` | time one persistent variant only (default both) |
| `--check-only`, `--no-check`, `--no-trace` | skip timing, the checks, or the traced runs |
| `--smoke` | run the timing code path once without timing anything |
| `--csv FILE` | results as CSV (`device,code_arch,format,router,section,name,value,unit`) |

## Files

| File | Contents |
|---|---|
| `src/fork_kernels.cuh` | the fork's kernels for one MoE layer as device functions (norm, router, top-6, quantization, Pascal products, weighted sum, add) and the 13-launch path's kernels |
| `src/persistent.cuh` | the persistent kernel: work list, counters, prologue, trace |
| `src/ref.cuh` | CPU reference in double precision and the stage checks |
| `src/main.cu` | formats, data, the 13-launch path, checks, timing, CSV |
| `build.sh` | Docker build of `build/persistent_pascal` |

Block layouts and the random weights and inputs come from [mmvq_pascal](../mmvq_pascal/src/).
