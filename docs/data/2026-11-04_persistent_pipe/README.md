# Q684: what a token-persistent pipeline would save on the rig, the prototype (2026-11-04)

Registry row Q684 in [quants/experiments.md](../../../quants/experiments.md) (the rig run is C417). On a layer split
over the nine P106 cards every token crosses eight card boundaries and comes back to card 0 as a token id, and the host
(a 2-core Celeron) drives every launch, copy and synchronize of that loop. A token-persistent pipeline keeps one kernel
resident on each card for the whole generation: it polls its input mailbox in mapped host memory, runs its layers,
writes the residual into the next card's mailbox, and the last card mails the token id back to card 0. The host only
starts the kernels and reads the results.

[bench/persistent_pipe](../../../bench/persistent_pipe/README.md) measures the upper bound of that saving before
anything is built into llama.cpp: a stand-in share of a token on every card (four FP16 matrix-vector products over a
dummy 10,240 × 10,240 weight matrix, est. ~5.6 ms a card on a P106), the persistent pipeline (`persist`, and
`persist-direct`, which writes the residual while its last product runs) against the same work driven by the host the
three ways a layer split can hand the residual on (`sync`, `peer`, `staged`, the last being the release line's
`GGML_CUDA_HOP_STAGED=1`), with the residual in F32 (40,960 bytes a token) or Q8_0 (10,880 bytes, the release line's
`LLAMA_HOP_CAST=q8_0`). Every variant's token ids and final residual must be bit-identical.

## Status

| Step | Where | State |
|---|---|---|
| The benchmark, its build and rig scripts | this repository | written |
| The build: sm_61 code, compute_61 PTX, sm_86 code ([build](#the-build)) | a machine with Docker, the CUDA 12.9 container | done |
| The smoke on two RTX 3090s: the Pascal path through the PTX JIT, then sm_86, correctness only ([the smoke](#the-smoke)) | the quant machine | passed, 9 of 9 runs |
| The rig run: every card, F32 and Q8_0, 1,000 passes a variant, two cards, 4 tokens a pass, the poll spacing ([run/rig_run.sh](run/rig_run.sh), ~15 min) | one rig, idle | queued (C417) |

## What to expect (estimates, from C140's hop numbers)

[C140](../2026-11-04_megakernel_rig/README.md) measured one token's hop between two P106 cards at 455 µs as the
driver's peer copy, 357 µs through the fork's staged hop in two chunks and 127 µs in Q8_0 through it, bandwidth-bound
(~26 µs fixed plus ~10 µs per KB, both PCIe 1.0 x1 legs in series). A persistent hop that writes the residual while the
last product runs leaves exposed only the last chunk's second leg and a poll: est. ~20 µs in Q8_0, ~60 µs in F32. So
**est. −0.9 ms a token on the release line's Q8_0 hops** (eight boundaries at ~110 µs, plus the id's way back), and
**−2.5 ms with F32 hops**, against a ~120 ms Flash-Next token. Not in the benchmark: llama.cpp's own per-token host work
(graph update, sampling, the server), which a persistent pipeline also moves off the per-token path; C405 measures how
much of a speculative step that is.

## The build

`bench/persistent_pipe/build.sh` in `nvidia/cuda:12.9.2-devel-ubuntu24.04`, `--cpu-shares=256`:

| | |
|---|---|
| Toolkit | CUDA 12.9 (`cuda_12.9.r12.9/compiler.36037853_0`) |
| md5 | `2bfe2ab6d38b23257e0336349ad598f4` |
| Host code | no AVX instructions, glibc ≤ 2.38, the CUDA runtime static |
| sm_61 registers | the persistent kernel 64 at 1 token a pass (4 blocks of 256 threads an SM: a grid of 20 on a P106), 128, 184 and 214 at 2, 4 and 8; the kernels for 1 and 2 tokens spill 12 and 20 bytes (seven local loads and stores in the whole kernel); a product kernel 64, 80, 104 and 144, no spills. The rig run's per-card product time of the persistent kernel against the products alone shows what the spill costs |

## The smoke

[smoke.tsv](smoke.tsv): two RTX 3090s without peer access, the binary above, every variant of every run checked against
the first (`sync`); no timings (that machine's speed says nothing about a P106). Seven runs on the compute_61 PTX through
the driver's JIT, which is the Pascal code path, and two on the native sm_86 code: 2, 3 and 4 pipeline stages (3 and 4
by listing the two cards more than once, with persistent grids of 8 blocks), F32 and Q8_0, 1, 2, 4 and 8 tokens a pass,
3, 4 and 16 chunks a mailbox, 1, 3 and 4 products a card, n = 2,560 and 10,240, 22 and 205 passes. **All nine give the
same token ids and final residual in every variant**, with no mailbox timeout and no GPU error. The clock-offset
ping-pong put the two cards' timers 175.2–175.3 ms apart, moving by up to 3 µs during a run.

## Files

| File | What |
|---|---|
| [smoke.tsv](smoke.tsv) | the smoke's runs, options and verdicts |
| [run/smoke_gpus.sh](run/smoke_gpus.sh) | the correctness smoke on two newer GPUs: 7 runs through the PTX JIT (2, 3 and 4 pipeline stages, F32 and Q8_0, 1, 2, 4 and 8 tokens a pass, 1–16 chunks, odd product counts), 2 native |
| [run/rig_run.sh](run/rig_run.sh) | the rig run from another machine: the binary and `run_rig.sh` to the rig, the md5 checked, the run detached, its output back, the rig cleaned |

## Commands

```bash
# the build (a machine with Docker), from the repository root
DOCKER_ARGS='--cpu-shares=256' bench/persistent_pipe/build.sh
# the smoke on two newer GPUs (no timings)
BIN=bench/persistent_pipe/build/persistent_pipe GPUS=device=0,1 OUT=<dir> bash docs/data/2026-11-04_persistent_pipe/run/smoke_gpus.sh
# the rig run (the rig idle; FROM: ssh arguments of the machine that holds the binary, if not this one)
RIG=<rig alias> BIN=<binary> MD5_BIN=<md5> OUT=<dir> bash docs/data/2026-11-04_persistent_pipe/run/rig_run.sh
```

Provenance: the benchmark and scripts are this project's; the weights are hashes generated on the cards, so no model
or data was downloaded for this work.
