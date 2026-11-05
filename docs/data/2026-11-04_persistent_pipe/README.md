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
| The rig run: every card, F32 and Q8_0, 1,000 passes a variant, two cards, 4 tokens a pass, the poll spacing ([run/rig_run.sh](run/rig_run.sh), ~15 min) | one rig, idle | **done 2026-11-05** (C417), [the result](#the-rig-run-c417-2026-11-05) |

**Result:** the persistent hop is what the estimate said, −0.77 ms a token on Q8_0 hops (28 µs a hop exposed against
~113 µs through the staged hop) and −2.2 ms on F32 hops. That is 1.3 % of a Flash-Next token, so the persistent
pipeline is not worth building into the engine for its hops. The run measured −5.7 ms a pass (−10.6 %) and −30.8 ms at
4 tokens a pass (−23.8 %) because of the clock, not the hop. The host-driven nine-card split left every card at the
1,354 MHz base clock, while the resident kernels held every card at ~1,800–1,911 MHz. That held clock, without a
persistent pipeline, is Q1021.

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

## The rig run (C417, 2026-11-05)

The run used nine P106-090 cards of one rig with nothing else on them and no model loaded. It ran the binary above
(md5 checked on the rig) through [run/rig_run.sh](run/rig_run.sh) with `run_rig.sh`'s default steps, in 14 minutes.
Every step exited 0, and every variant gave the same token ids and final residual as `sync`. The tables are in
[rig/](rig/), one `<step>.txt` and `<step>.csv` a step. The cards' clocks, from `nvidia-smi` every 500 ms in 10-second
windows, are in [rig/clocks_10s.tsv](rig/clocks_10s.tsv).

### A pass on nine cards, the median of 1,000 (250 at 4 tokens)

| Variant | 1 token, Q8_0 hops (ms) | 1 token, F32 hops (ms) | 4 tokens, Q8_0 hops (ms) |
|---|---|---|---|
| `sync` | 53.941 | 56.269 | 129.910 |
| `peer` | 53.859 | 56.141 | 129.800 |
| `staged` (the release line's hop) | 53.688 | 55.381 | 128.974 |
| `persist` | 48.144 | 49.535 | 99.801 |
| `persist-direct` | 47.991 | 48.228 | 98.219 |
| **`persist-direct` against `staged`** | **−5.694 (−10.6 %)** | **−7.152 (−12.9 %)** | **−30.756 (−23.8 %)** |

p10 to p90 spans at most 0.14 ms in every variant. On two cards (`two`, F32) the persistent pipeline is slower:
`staged` 10.183 ms, `persist-direct` 10.658 ms (+0.474).

### Where the saving comes from: the clock, not the hop

In P0 the cards ran at one of two clocks, with the memory at 4,006 MHz throughout ([rig/clocks_10s.tsv](rig/clocks_10s.tsv)):

| Phase | Each card works | SM clock | Power a card |
|---|---|---|---|
| The host-driven variants on nine cards (`sync`, `peer`, `staged` in `f32`, `q8`, `k4`) | ~1/9 of a pass | 1,354 MHz (the base clock) in every sample | ~29 W |
| The persistent variants on nine cards | ~1/9 of a pass, and polls the rest | ~1,800–1,911 MHz, mostly 1,860–1,911 | ~40 W |
| Every variant on two cards (`two`) | ~1/2 of a pass | 1,860–1,885 MHz | |

A card that idles for most of every pass falls to its base clock. A resident kernel keeps it busy, so it stays at
boost (the same effect as registry row Q154).

The same products ran at both clocks within one measurement. The `f32` step timed its products alone while cards 0
and 1 were still at boost from the smoke and cards 2–8 were just out of P8 at 1,354–1,493 MHz. Cards 0 and 1 took
4.90 ms, cards 2–8 took 5.78–5.95 ms (mean 5.85, +19 %). At 4 tokens a pass (the `k4` step's line) the products took
10.18 ms at boost and 13.8–14.1 ms at base (+37 %). Using the base-clock times as the host-driven variants' compute, a
pass splits as follows:

| ms a pass (est. where marked) | 1 token, Q8_0 | 1 token, F32 | 4 tokens, Q8_0 |
|---|---|---|---|
| the products in their own kernel at boost | 44.16 | 44.16 | 91.66 |
| the same at the base clock (est., from the cold cards) | 52.67 | 52.67 | 125.96 |
| `staged`: the pass minus the products at base = hops and host (a hop) (est.) | 1.02 (113 µs) | 2.71 (301 µs) | 3.01 (335 µs) |
| `persist-direct`: the products inside the persistent kernel, at boost | 47.74 | 47.69 | 97.52 |
| `persist-direct`: hops (a hop) | 0.25 (28 µs) | 0.54 (60 µs) | 0.70 (78 µs) |
| **saved by the clock** (base to boost) | **8.51** | **8.51** | **34.30** |
| saved on the hops | 0.77 | 2.17 | 2.31 |
| lost to the persistent kernel's slower products | −3.58 | −3.53 | −5.86 |
| = `persist-direct` against `staged` | 5.69 | 7.15 | 30.76 |

The derived `staged` hops cross-check this split. At 113 µs (Q8_0) and 301 µs (F32), averaged over the eight residual
hops and the ids' way back, they sit within ~15 % of [C140](../2026-11-04_megakernel_rig/README.md)'s two-card
measurements of the same staged hop (127 and 357 µs).

- **The hop alone matches the estimate.** `persist-direct` leaves 25–30 µs of a Q8_0 hop exposed and 61–67 µs of an
  F32 hop. Against the staged hop that saves 0.77 ms a token on Q8_0 hops (est. 0.9) and 2.2 ms on F32 hops (est. 2.5).
- **The persistent kernel's products run 6–8 % slower** than the standalone product kernel at the same clock (5.21–5.34
  against 4.90–4.92 ms a card). Its grid is 20 blocks capped at 64 registers with a 12-byte spill, behind a grid
  barrier. `persist-direct` spends ~90 µs more a card because its last product writes into the next mailbox. On two
  cards, where the host-driven variants stay at boost, this alone makes the persistent pipeline the slower one.
- **The clock is the large part.** On these memory-bound FP16 products, the boost clock against the base clock is worth
  19 % a card at 1 token and 37 % at 4 tokens. That is more than the hops and the host's work together.

### The other readings

- **Per boundary:** a hop between two cards behind the same switch takes 5–10 % longer than a hop into or out of card 4,
  which has a root port of its own. In `persist` that is 136–140 against 124–127 µs in Q8_0, and 316–320 against
  290–297 µs in F32: both legs of a hop inside one switch share its uplink. No boundary is slower because it crosses
  from one switch to the other. In `persist-direct` the spread is 25–30 µs. The ids' way back to card 0 takes
  22–25 µs.
- **Polls:** polling every 0.5, 2 or 8 µs after the sender's hint moves an F32 hop by at most 4 µs (`persist-direct`:
  63, 64 and 67 µs) and a pass by at most 0.04 ms. Faster polling does not slow the active hop on the shared uplinks,
  and the default of 2 µs stays.
- **Timers:** the nine cards' `%globaltimer`s were up to 19 ms apart and drifted by up to 0.55 ms during a run. With
  the ping-pong offsets interpolated, the per-boundary medians add up to the overhead taken from the pass time
  (`persist-direct`, Q8_0: 28.0 against 28.1 µs a hop), so the alignment holds.
- **Power:** the persistent variants draw ~11 W more a card than the host-driven ones (~40 against ~29 W). That is the
  boost clock plus the polling.

### What it decides

**The token-persistent pipeline is not worth building into the engine for Flash-Next's nine-card split.** Its own gain,
the hop, is 0.77 ms a token on the release line's Q8_0 hops. That is 1.3 % of the release line's plain-decode token of
58.6 ms (17.06 tok/s with `LLAMA_HOP_CAST=q8_0` and the staged hop, registry Q750 and Q963). Its kernels' products also
run 6–8 % slower than standalone kernels, and a per-card megakernel (Q670) would have to win that back first. Q684 is
parked with this bound. The hop work stays with three rows:
- Q682, the mailbox hop inside the host-driven engine. It covers llama.cpp's own per-hop host work, which this
  benchmark's ideal host loop of four launches a card does not have.
- Q963, fewer bytes a hop.
- Q964, fewer boundaries.

**What the run found instead is the clock.** A host-driven nine-card split leaves every card at the 1,354 MHz base
clock, and any resident kernel holds it at ~1,800–1,911 MHz. On these products that is 19 % a card at 1 token and 37 %
at 4 tokens. Two questions remain: whether Flash-Next's cards also sit at base while the engine decodes, and what a
light keep-alive kernel on each card costs and saves. Both belong to registry row Q1021.

## Files

| File | What |
|---|---|
| [smoke.tsv](smoke.tsv) | the smoke's runs, options and verdicts |
| [run/smoke_gpus.sh](run/smoke_gpus.sh) | the correctness smoke on two newer GPUs: 7 runs through the PTX JIT (2, 3 and 4 pipeline stages, F32 and Q8_0, 1, 2, 4 and 8 tokens a pass, 1–16 chunks, odd product counts), 2 native |
| [run/rig_run.sh](run/rig_run.sh) | the rig run from another machine: the binary and `run_rig.sh` to the rig, the md5 checked, the run detached, its output back, the rig cleaned |
| [run/clock_table.sh](run/clock_table.sh) | `run_rig.sh`'s `smi.csv` and `run.log` as 10-second windows: the step, the card samples in P8, at base, between and at boost, the mean power of a card |
| [rig/smoke.txt](rig/smoke.txt), [.csv](rig/smoke.csv) | the rig run's smoke: every variant once on two cards, 20 passes |
| [rig/f32.txt](rig/f32.txt), [.csv](rig/f32.csv) | nine cards, F32 hops, every variant, 1,000 passes |
| [rig/q8.txt](rig/q8.txt), [.csv](rig/q8.csv) | nine cards, Q8_0 hops, every variant, 1,000 passes: the headline |
| [rig/two.txt](rig/two.txt), [.csv](rig/two.csv) | two cards, F32 hops, every variant, 1,000 passes |
| [rig/k4.txt](rig/k4.txt), [.csv](rig/k4.csv) | nine cards, Q8_0 hops, 4 tokens a pass, every variant, 250 passes |
| [rig/poll0.5.txt](rig/poll0.5.txt), [.csv](rig/poll0.5.csv), [rig/poll8.txt](rig/poll8.txt), [.csv](rig/poll8.csv) | nine cards, F32 hops, the persistent variants with polls every 0.5 and 8 µs, 300 passes |
| [rig/clocks_10s.tsv](rig/clocks_10s.tsv) | the cards' clock states and power through the run, 10-second windows (`run/clock_table.sh`) |

## Commands

```bash
# the build (a machine with Docker), from the repository root
DOCKER_ARGS='--cpu-shares=256' bench/persistent_pipe/build.sh
# the smoke on two newer GPUs (no timings)
BIN=bench/persistent_pipe/build/persistent_pipe GPUS=device=0,1 OUT=<dir> bash docs/data/2026-11-04_persistent_pipe/run/smoke_gpus.sh
# the rig run (the rig idle; FROM: ssh arguments of the machine that holds the binary, if not this one)
RIG=<rig alias> BIN=<binary> MD5_BIN=<md5> OUT=<dir> bash docs/data/2026-11-04_persistent_pipe/run/rig_run.sh
# the clock table from its output
bash docs/data/2026-11-04_persistent_pipe/run/clock_table.sh <dir>/out/smi.csv <dir>/out/run.log > clocks_10s.tsv
```

Provenance: the benchmark and scripts are this project's; the weights are hashes generated on the cards, so no model
or data was downloaded for this work.
