# megakernel_p106: small dependent kernels against one persistent kernel on a P106-090

When a P106-090 decodes one token of Qwen3.8-Flash-Next, each of the nine cards runs ~250–340 launch groups, and most
of them do almost no work: elementwise ops, norms and short products on vectors of 2,560 or 10,240 values
([census](../../docs/data/2026-11-04_megakernel_census/README.md)). A persistent kernel (a "megakernel", as in Hazy
Research's Llama-1B work and Mirage's MPK) would run those stages inside one launch, with a barrier or a counter between
dependent stages instead of a kernel boundary. Whether that pays on a 5-SM Pascal card depends on two numbers nobody
has measured on one: what a small kernel costs as a CUDA-graph node, and what a grid-wide barrier costs. This benchmark
measures both, on the stage sizes of the census, in one binary. Design note:
[megakernel.md](../../docs/research/notes/megakernel.md); registry row Q670 in
[quants/experiments.md](../../quants/experiments.md).

## What it runs

A **chain** of N dependent stages. Each stage reads the vector the stage before it wrote (two ping-pong buffers of
12,288 floats) and writes its own:

| Stage | Work | Virtual blocks (256 threads) | Stands for |
|---|---|---|---|
| `ew<n>` | `y = a x + b` on n values | n / 256 | the scales, gates, activations and adds (2,560: the hidden vector; 10,240: the four hyper-connection streams) |
| `norm<n>` | `y = a x / rms(x)`, one block for the whole vector | 1 | `RMS_NORM` (one block per row, as in ggml), the hyper-connection mixing |
| `mv<K>x<rows>` | `y = a W x + b`, FP16 weights, one block per row | rows | the short products: injects 10,240 × 4 (82 KB), recurrent alpha/beta 2,560 × 48 (245 KB) |
| `empty` | nothing | 1 | the launch or barrier floor |

Every MV stage has its own weights, so no stage finds another's in L2. The **mixes** run by default:

| Mix | Stages |
|---|---|
| `empty`, `ew2560`, `ew10240`, `norm2560` | 256 of one kind |
| `mv10240x4`, `mv2560x48` | 64 of one kind |
| `census1`, `census6` | Flash-Next's small launch groups for 1 and 6 layers (38 a layer: 6 norms, 28 elementwise, 4 short products), from the R13 trace: about one card's token of small stages at `census6` |

Each chain runs as **variants** that share the same device function per stage and the same virtual blocks, so only the
boundaries differ:

| Variant | Boundaries |
|---|---|
| `stream` | N kernel launches on a stream (the host launches each) |
| `stream+events` | the same with a CUDA event before the first stage and after each, as the fork's `GGML_CUDA_PROFILE` records them |
| `graph` | N kernels captured once as a CUDA graph and replayed (how the fork decodes on compute 6.1) |
| `graph+events` | the graph with an event-record node between stages (`GGML_CUDA_PROFILE` under graph capture) |
| `coop` | one cooperative launch (`cudaLaunchCooperativeKernel`), `grid.sync()` between stages |
| `coop-graph` | the cooperative kernel as one CUDA-graph node (CUDA 11+ supports the capture; the binary checks it on the card) |
| `barrier` | one plain launch, a grid barrier from one atomic counter between stages (needs every block resident; the grid is capped at the occupancy) |
| `barrier-graph` | `barrier` as one CUDA-graph node: the counter is monotonic, so the node replays without a reset |
| `barrier-local` | `barrier` with the barrier skipped before an elementwise stage that follows an elementwise stage of the same size, where each thread reads only what it wrote: what fusing a chain costs with no barrier at all |

The persistent variants run at grids of 5, 10 and 20 blocks and the largest resident grid (`--grids`). For the two
`+events` variants the binary also prints the time between the events around each stage, by stage kind: what
`GGML_CUDA_PROFILE` would report for that stage, next to its real cost in the plain `graph`. That calibrates the
profiler: R13's trace shows ~9–10 µs per elementwise group, against 2.1–2.4 µs for such kernels in a graph in Q88's
bench.

`--hop A,B` times one card-to-card hop of a residual vector (10 KB, 40 KB (Flash-Next's 4 × 2,560 FP32 residual),
160 KB) three ways, with the host clock: two copies through a pinned host buffer with the host waiting on each; one
`cudaMemcpyPeerAsync` (staged through the host by the driver when there is no peer access) with card B's stream
waiting on an event, as ggml's asynchronous copy between CUDA backends does; and a mailbox in mapped pinned memory
that a kernel on card B polls while card A's kernel writes it.

**Checks.** For every mix, the stream launches are compared with a double-precision CPU reference of the whole chain
(max error < 1e-3 of the largest value), and every other variant and grid with the stream launches (< 1e-4; reported
when not bit-identical). The run fails (exit code 2) if any check fails.

## Building

On a machine with Docker, from the repository root:

```bash
DOCKER_ARGS='--cpu-shares=256' bench/megakernel_p106/build.sh
```

The binary holds sm_61 code (the P106), compute_61 PTX (runs the Pascal path on newer cards through
`CUDA_FORCE_PTX_JIT=1`) and sm_86 code; host code without AVX for the rig's Celeron, the CUDA runtime linked
statically. `build/BUILDINFO.txt` records the toolkit and the md5; `build/ptxas.log` the registers (the persistent
kernels: 61 registers, no spills on sm_61, so 4 blocks of 256 threads per SM).

## Running on the rig

With nothing else on the cards (the hop uses cards 0 and 1, the stage timings card 0), ~10 minutes:

```bash
RIG=${RIG:-rig}
ssh "$RIG" 'mkdir -p /tmp/mk'
scp bench/megakernel_p106/build/megakernel_p106 bench/megakernel_p106/run_rig.sh "$RIG":/tmp/mk/
ssh "$RIG" 'cd /tmp/mk && setsid nohup ./run_rig.sh > console.log 2>&1 < /dev/null &'
ssh "$RIG" 'test -e /tmp/mk/out/DONE && cat /tmp/mk/out/run.log'
scp -r "$RIG":/tmp/mk/out ./megakernel_rig_out
ssh "$RIG" 'rm -rf /tmp/mk'
```

`run_rig.sh` refuses to start while a compute process is on the cards (`FORCE=1` overrides), logs `nvidia-smi` every
500 ms into `smi.csv`, and runs the smoke pass (every timing path once), the checks, the timings (`timing.csv`) and the
hop (`hop.csv`), in that order. Env: `GPU` (default 0), `HOP` (default `0,1`; empty skips it), `OUT`.

## Options

| Option | Meaning |
|---|---|
| `--mixes LIST` | mixes, comma-separated; a bracketed pattern such as `[norm2560,ew2560*3,mv2560x48]*4` is a mix too |
| `--stages N` | stages of the one-kind mixes |
| `--grids LIST` | grids of the persistent kernels (`max`: resident blocks per SM × SMs) |
| `--reps N`, `--trials N` | chains per measurement (default: ~20 ms) and measurements per variant (default 7, median reported) |
| `--check-only`, `--no-check`, `--smoke` | checks only; no checks; every timing path once |
| `--hop A,B`, `--hop-sizes LIST`, `--hop-reps N`, `--hop-only` | the card-to-card hop |
| `--csv FILE` | `section,mix,stages,variant,grid,key,value,unit` |

## What decides what

- **`graph` µs per stage** on `census6` is what a launch costs the fork today; with `empty` it splits into the node
  floor (Q88's bench measured 1.84 µs for an empty graph node) and the small kernel's own run.
- **`barrier`, `coop` µs per stage** is what a persistent kernel pays instead. A P100 measured 1.8–2.3 µs per
  `grid.sync()` at one block per SM; a 5-SM card should be cheaper, but nobody has measured it.
- **The gate for building stage 1** (Q670): the persistent kernel at ≤ 50 % of the graph's time per stage on `census6`,
  i.e. a saving of ≥ ~1.2 µs per stage, ≥ ~3.5 ms per token over the ~2,900 kernels of a Flash-Next token.
- **`graph+events` against `graph`** gives the profiler's cost per group, the correction every `GGML_CUDA_PROFILE`
  number of a small op needs (Q681).
- **The hop** says whether the eight hops of a token are bound by the bytes over PCIe 1.0 x1 or by the host's
  synchronize, copy and launch, and what a mailbox would save (Q682).

## Files

| File | What |
|---|---|
| `src/main.cu` | the benchmark |
| `build.sh` | the Docker build |
| `run_rig.sh` | the rig run: quiet check, `nvidia-smi` log, smoke, checks, timings, hop |
