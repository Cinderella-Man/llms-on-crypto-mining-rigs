# The pipe step on the cards (C1009, Q1742; 2026-11-24)

What this is: a script that checks Miner Llama's pipelined decoding on the cards, for the GPU check of every branch and
merge. Pipelined decoding guesses tokens ahead and runs them through the cards as a pipeline; it must give exactly
plain decoding's output. Since C935 ([2026-11-24_c935_pipelined_identity](../2026-11-24_c935_pipelined_identity/README.md))
every CPU test run checks that on the processor. A fault that lives only in a CUDA path passes there, and part of
C932's fault ([2026-11-21_c932_pipe_logits](../2026-11-21_c932_pipe_logits/README.md)) was in the CUDA `gdn_conv` cache
fusion. The pipe also has paths that run only on the cards: the hops between cards, a sampler on the device and the
next token fed on the devices. The new script, [bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh),
runs the CPU step's random models on a CUDA build with every layer on the cards. Registry row Q1742 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The GPU check now decodes pipelined on the cards, and it catches C932's fault there.** On `pascal-p106-v5` at
  `8a9629942`, which still carries the fault (the fix is not merged yet), the Flash-Next-type model differs from plain
  decoding in every arm and layout: 13 of 13 pipelined runs with host sampling, 13 of 13 with greedy sampling on the
  device, and the rollback check. The other four models pass.
- **With C932's fix on the same commit, every check passes:** 0 of 13 runs differ for each of the five models in every
  arm and layout, and every rollback check passes.
- **The fork's Pascal kernels ran where forced, and only on quantized weights.** With them forced on in a P106's launch
  shapes, the Q8_0 copies' plain logits come out with other bits than with llama.cpp's own kernels, for all five
  models, so the fork's kernels computed those products. The F32 models give the same bits either way: none of the
  fork's kernels takes F32 weights. Forcing the fork's other compute-6.1-only paths as well changed no bit.
- **Every layout computes the same plain run:** one card, three cards and three virtual devices on one card give one
  plain logits hash per model and arm, on both trees.
- **About a minute and a half a build** with the step's two default arms on three cards: 90 seconds for each tree once
  the cards had compiled the build's code, 151 seconds for the first tree in the pass where the cards first compiled the
  forced kernels.

## The step

`gpu_pipe_check.sh` takes one or more builds (a rig tarball's `bin/`, `scripts/build_llamacpp_docker.sh` with `TESTS=1`,
which carries `llama-pipe-recurrent-check` and `test-llama-archs` since C984). For each build it makes the random models
with the build's own `test-llama-archs` (`-s 1`, a 4,096-token vocabulary, the MoE model where the architecture has one:
the CPU step's models, the same files) and a Q8_0 copy of each, then runs each arm and layout in its own container with
`-ngl 99 -fa on -c 2048 -fit off`, greedy, 64 tokens of [prompts/rewrite.txt](../../../bench/pipespec/prompts/rewrite.txt):

| Check | What it runs | Passes when |
|---|---|---|
| `host` | [check_identity.sh](../../../bench/pipespec/check_identity.sh): plain decoding, depth 0, and at depths 1 and 3 an oracle with 0 % and 50 % of its guesses wrong plus the four guessers; host sampling | every pipelined run's tokens, per-token logits, KV state and probe hashes are plain's |
| `fed` | the same with greedy sampling on the device in every run (`--pipe-backend-greedy`), the token embedding on the first card (`LLAMA_INPUT_ON_GPU=1`) and `FEED=1`, which adds depth 0 with each next token fed on the devices from the last card's sample | the same; the logits and probe hashes then hash token ids, so the KV state hash carries the check |
| `recurrent` | `llama-pipe-recurrent-check`: a token sent to the slot of the token before it, wrong guesses dropped, the token in flight dropped and sent again | the same four hashes (exit status 0) |
| `kernels` | one token of `llama-completion -lv 4`: the fork's features ggml lists and each device's model buffer | recorded, not judged |

| Layout | Cards | What it adds |
|---|---|---|
| `one` | the first card | every CUDA kernel of the model |
| `split` | the three cards, `-ts 1,1,1` | the hops: each model's two layers and its output on three cards, the embedding on the first, the device sampler on the last |
| `virt` | three virtual devices on the first card (`GGML_CUDA_DEVICES=3`, `-ts 1,1,1`) | the same hops on a machine with one card |

| Arm | Models | Kernel switches |
|---|---|---|
| `f32-stock` (default) | as `test-llama-archs` writes them | none: on a card that is not compute 6.1, llama.cpp's own kernels |
| `q8-p106` (default) | the Q8_0 copies | the fork's Pascal matrix-vector kernels and sparse attention forced on in a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as a P106 runs them |
| `f32-p106`, `q8-stock` | | the other two pairings, to show which kernels a setting reaches |
| `q8-p106all` | the Q8_0 copies | `q8-p106` plus the fork's other compute-6.1-only paths forced on: the attention-block fusion, the persistent MoE kernel, the Pascal flash-attention tile kernel, the F32 multi-token kernel |

The defaults are layouts `one split` (with two or more cards; `one virt` with one) and arms `f32-stock q8-p106`.
Verdicts go to `summary.tsv`: OK, DIFFERS (a pipelined run gives other hashes), FAIL (a tool stopped before comparing)
or SKIP (the build lacks a tool). `llama_pipe_submit_fed` refuses, by design, a model whose KV cells keep their token
(M-RoPE in `qwen35` and `qwen35moe`, the per-layer embeddings in `qwen4exp`); there the fed run is left out and the
detail says so, and the other 13 runs are judged. The rig tarballs carry no `llama-quantize`, so `QUANT` names one.

## The run

On a build machine with three RTX 3090s, which run the rig build's compute 6.1 code through its PTX, every container
labelled, at a low CPU share and without network; the cards checked with a CUDA preflight before each round. Nothing
was timed but the rounds themselves.

- **tip**: `pascal-p106-v5` at `8a9629942`, as it is: C984's `TESTS=1` tarball (md5 `59ca8336b87367a6522e63c6fcc1381b`,
  1,252,901,378 bytes).
- **fix**: the same commit with C932's fix `80d83fa1a` (fork branch `pascal-p106-v5-pipe-logits`) as a patch, built
  here by `scripts/build_llamacpp_docker.sh` with `TESTS=1` in 75 s from C984's compiler cache (tarball md5
  `5bb62ea924a347906de1bc5685efdeb3`, 1,252,900,232 bytes).
- **The quantizer**: a static CPU build of `llama-quantize` from the fork's C983 branch (`91eead67a`, md5
  `d66f6eaa258775134cc8b600ba581e6e`) with `--tensor-type hc_=f32 --tensor-type ple_norm=f32`; Q8_0 rounding is the
  same in every build, and both trees got the same ten files ([models.tsv](models.tsv)).

Three rounds of the script over both trees, 08:46:20–08:54:49: `step` (its defaults, as a GPU check runs it), `reach`
(arms `f32-p106 q8-stock q8-p106all`, one card and split) and `virt` (the default arms on virtual devices). Two earlier
passes, 08:22:32–08:32:08 and 08:34:31–08:43:03, with the script before its last changes (the kernels line without the
placement, then comments), gave the same verdicts and plain hashes in all 360 checks.

### The verdicts

The same in every arm and layout of a tree, so one column per tree ([pipe.tsv](pipe.tsv) has all 384 lines):

| Model | tip: host | tip: fed | tip: recurrent | fix: host | fix: fed | fix: recurrent |
|---|---|---|---|---|---|---|
| `qwen4exp` | **DIFFERS, 13 of 13** | **DIFFERS, 13 of 13** ¹ | **DIFFERS** | OK, 0 of 13 | OK, 0 of 13 ¹ | OK |
| `qwen35` | OK, 0 of 13 | OK, 0 of 13 ¹ | OK | OK, 0 of 13 | OK, 0 of 13 ¹ | OK |
| `qwen35moe` | OK, 0 of 13 | OK, 0 of 13 ¹ | OK | OK, 0 of 13 | OK, 0 of 13 ¹ | OK |
| `qwen3next` | OK, 0 of 13 | OK, 0 of 14 | OK | OK, 0 of 13 | OK, 0 of 14 | OK |
| `flint` | OK, 0 of 13 | OK, 0 of 14 | OK | OK, 0 of 13 | OK, 0 of 14 | OK |

¹ the fed run refused by design (M-RoPE or per-layer embeddings), the other 13 judged.

Each tree ran 12 arm-and-layout pairs (the five arms on one card and split, the two default arms on virtual devices):
180 verdicts a tree. On the tip, every pipelined `qwen4exp` run with host sampling gives plain decoding's tokens with
other logits, KV state and probe hashes, from depth 0 on, as on the processor (C935). With device sampling only the KV
state hash differs, in 13 of 13 runs of every pair: the tokens are the same, and there the logits and probe hashes hash
token ids. The rollback check gives plain's tokens with other logits from the second token (5 of the 12 pairs) or the
third (7), and other state and probe hashes.

### Which kernels ran

Plain decoding's logits hash with host sampling, on the same tree, models, layout and model ([hashes.tsv](hashes.tsv)):

| Models | `p106` against `stock` | `p106all` against `p106` |
|---|---|---|
| F32 | the same bits, all 5 models, both trees, one card and split | — |
| Q8_0 | **other bits**, all 5 models, both trees, one card and split | the same bits, all 5 models |

- Under `p106` and `p106all` ggml lists `MMVQ_PASCAL` and `MMVF_PASCAL` as on; under `stock` neither. On F32 weights
  they are listed and compute nothing: the matrix-vector kernels take quantized (and BF16) weights only.
- So in the default arms `f32-stock` runs llama.cpp's own CUDA kernels on F32 weights and `q8-p106` the fork's Pascal
  matrix-vector kernels on Q8_0 weights. The extra compute-6.1-only paths of `p106all` changed no bit on these models:
  either they did not take a product or they give the same bits.
- Each arm gives one plain logits hash per model over both trees and every layout (6 runs for `f32-stock` and
  `q8-p106`, 4 for the others): the fix leaves plain decoding as it was, and a split computes what one card computes.

Where the layers went (the kernels line, `qwen4exp`'s model): one card `CUDA0 21.95 MiB` (F32) and `CUDA0 12.13 MiB`
(Q8_0); split over three cards `CUDA0 5.53, CUDA1 12.36, CUDA2 4.07 MiB` (F32) and `CUDA0 1.62, CUDA1 9.38,
CUDA2 1.13 MiB` (Q8_0), the same on virtual devices.

### Time per tree ([times.tsv](times.tsv))

| Round | tip | fix |
|---|---|---|
| `step` (the defaults: 2 arms, one card and split) | 90 s | 90 s |
| `reach` (3 arms, one card and split) | 134 s | 134 s |
| `virt` (2 arms) | 28 s | 29 s |

The cards keep the compiled PTX in a cache shared by every run. In the first pass, where the cards first compiled the
forced kernels, `step` took 151 s for the tip and 91 s for the fix.

### What the random models show and what they do not

With random weights the greedy tokens depend mostly on the embedding and output weights, which the same seed makes
alike: `qwen35`, `qwen35moe` and `qwen3next` emit the same token list. A fault that changes only the layers' sums
therefore rarely changes a token here; the step rests on the bit-exact logits, KV state and probe hashes, as the CPU
step does. The Q8_0 copies reach the fork's Q8_0 kernels only, not the kernels for the release files' types (Q2_0,
Q4_K, Q5_K, Q6_K, IQ4_XS) or the Flash-Next preset's kernel switches (registry row Q1751).

## Files

| File | What |
|---|---|
| [pipe.tsv](pipe.tsv) | every verdict of the three rounds (384: 2 trees × 12 arm-and-layout pairs × 5 models × 3 checks, plus 24 kernels lines) with their details |
| [hashes.tsv](hashes.tsv) | each host-sampled identity check's plain tokens and logits hashes (120) |
| [models.tsv](models.tsv) | the random models and their Q8_0 copies, md5 in both trees |
| [binaries.tsv](binaries.tsv) | the md5 of the tools each tree ran |
| [times.tsv](times.tsv) | each tree's time in each round |
| [run/run.sh](run/run.sh) | the driver: the fix's build, the two tarballs unpacked, the three rounds of the script |
| [run/tables.py](run/tables.py) | the tables of this folder from the driver's `out/` |
| [../../../bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh) | the step |

## Commands

```bash
# the step as a GPU check runs it, on the branch's and the base's TESTS=1 tarballs (their bin/), on the build machine
WORK=<dir> BUILDS='base=<base>/bin head=<branch>/bin' GPUS=0,1,2 QUANT=<a static llama-quantize> \
  MOUNTS='<folder holding the builds>' PREFLIGHT=<cuda_preflight.sh> DOCKER_ARGS='<labels> --cpu-shares=256' \
  bash bench/pipespec/gpu_pipe_check.sh
# this run: REPO a copy of this repository (scripts/, bench/pipespec/), the fix as a patch file under WORK/build
WORK=<dir> REPO=<copy> TIP=8a9629942a4635ea0b2711b72e59556b96a63f86 TIP_TGZ=<the tip's TESTS=1 tarball> \
  FIX_PATCH=patches/c932-fix.diff LOCK=<build lock file> QUANT=<llama-quantize> PREFLIGHT=<cuda_preflight.sh> \
  DOCKER_ARGS='<labels> --cpu-shares=256' JOBS=8 bash run.sh      # STEPS='build unpack step reach virt'
# the patch: git format-patch -1 --stdout 80d83fa1a > <WORK>/build/patches/c932-fix.diff
# the tables, from the driver's out/ copied back
python3 -I run/tables.py <WORK>/out <this folder>
```

The image is `nvidia/cuda:12.9.2-devel-ubuntu24.04`; the containers mount the builds and this folder read-only and
share the PTX cache `WORK/jitcache`.

Provenance and licence: every table here was made by these scripts from their own run; the random models come from
llama.cpp's `test-llama-archs`; llama.cpp and its tests are MIT-licensed.
