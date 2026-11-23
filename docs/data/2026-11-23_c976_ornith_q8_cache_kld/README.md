# Q1707: does a Q8_0 cache keep Ornith-1.5-122B's quality? KLD at 8,192 to 49,152 tokens of agent text (2026-11-23)

Ornith's v5 preset keeps its attention cache in F16. A Q8_0 cache is half the size, and pull request 9 added a decode
kernel that reads it directly on a P106 ([Q1703](../../../quants/experiments.md)). Before a rig run times it
([Q1706](../../../quants/experiments.md)), this folder checks the cache's quality: Ornith-1.5-122B Pascal N2P GGUF
(`valid1`) with `-ctk q8_0 -ctv q8_0` against the same file with its F16 cache, on held-out agent text, through the rig
build's compute 6.1 PTX on three RTX 3090s. It is a correctness and quality check only: no time on these cards means
anything for a P106.

**Result: within noise of the F16 cache, with one caveat in perplexity beyond 8,192 tokens.**
- **Against Ornith's own Q8_0 model** (12 stretches of 8,192 tokens), the Q8_0 cache is as close as the F16 cache:
  KLD 0.052815 against 0.052227, same top token 95.028 % against 95.008 %, perplexity 0.08 % lower. Across the 12
  stretches the cache adds 0.000590 ± 0.000317 to the KLD, a fifth of the band fixed before the run.
- **Against the F16 cache**, at 8,192, 16,384, 32,768 and 49,152 tokens, the Q8_0 cache moves the output less than
  reading the same F16 cache in micro-batches of 128 instead of 512 does: KLD and same top token at every depth, and
  through the decode path (one token a pass), both with the base's kernels and with pull request 9's decode kernel.
- **Perplexity was the one measure that moved.** On single stretches the Q8_0 cache's perplexity is 1.73 % higher at
  16,384, 1.07 % at 32,768 and 0.60 % at 49,152 (−0.15 % at 8,192). The first two are outside the ±1 % band fixed
  before the run, so by that band (B) fails at 16,384 and 32,768. Repeats on more stretches of text put both failures
  on single stretches: on six stretches of 16,384 tokens the rise is that one stretch (+1.73 %, then −0.06 %, +0.29 %,
  −0.20 %, −0.04 %, −0.01 %; mean +0.28 % ± 0.29 %), the stretch where rounding alone also moves the output most.
  On four stretches of 32,768 tokens the same: +1.07 % on (B)'s window, then −0.11 %, +0.33 %, +0.11 %
  (mean +0.35 % ± 0.26 %). Both repeats keep the average rise below 1 % by more than two standard errors; a rise of a
  few tenths of a percent beyond 8,192 tokens is not ruled out.
- **What the check cannot see:** scaling the Pascal prompt kernel's output by 1.05 (its fault switch) moves the KLD
  from 0.009892 to 0.014446, which shows the kernel read the Q8_0 cache, but stays inside (B)'s KLD band: on this model
  rounding alone moves the output so far (KLD 0.010796 at 8,192) that an error of that size cannot be told from it by
  KLD. The validity rule fixed before the run asked for the fault to land outside the band, so it failed. The
  comparison against the Q8_0 model and the perplexity repeats carry the verdict.

For the rig run: Ornith can run its agent check with a Q8_0 cache. Its tasks will show whether a perplexity cost of a
few tenths of a percent beyond 8,192 tokens matters; this check can neither show nor rule out one that small.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`). Its GPU code is sm_61 machine code and compute 6.1 PTX only (`CUDA : ARCHS = 610` in every log), so a newer card runs the PTX |
| Decode kernel arm | `pascal-p106-v5` at `14d679604`, which holds pull request 9 with its mask fix (rig tarball `llama.cpp-20261123-14d679604-pascal-p106-v5-dec-mask.tar.gz`, md5 `d8dd415dea36f9d064ce67e65840c218`; `llama-perplexity` md5 `6e4655f59ab106f5cc119a7fe3c16d6e`) |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Reference for (A) | Ornith-1.5-122B Q8_0's log-probabilities on that text, 12 chunks of 8,192 tokens, from [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (its perplexity 1.5683 ± 0.00887) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight passed before every pass |
| Settings in every run | `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096` (the preset's batch); the v5 preset's engine variables `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes: `GGML_CUDA_FATTN_TILE_PASCAL=2` (the prompt attention kernel, which reads F16 and Q8_0 K/V directly; `1`, a P106's default, asks for compute 6.1 exactly), `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`; `GGML_SCHED_DEBUG_REALLOC=1` |
| Time | 2026-11-23 21:07–22:37, 30 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)); a first try of pass v with `-b 4096`
(eight 512-token chunks at once) stopped at the scheduler's reallocation guard during its warm-up and was run again with `-b 512` runs (one model load each) |

The arms (each one load of `llama-perplexity`; the first arm of a pass writes the reference unless the pass has one):

| Pass | Arm | Cache | `-ub` | Other | What it is |
|---|---|---|---:|---|---|
| a | `f16`, `q8` | F16, Q8_0 | 512 | | each scored against Ornith's Q8_0 model, 12 chunks of 8,192 |
| b, d | `f16` | F16 | 512 | | the reference: the v5 preset's cache and micro-batch |
| b, d | `q8` | Q8_0 | 512 | | the test arm |
| b, d | `f16u` | F16 | 128 | | the rounding yardstick: the same model and cache, its sums in another order |
| b | `f16b` | F16 | 512 | | `f16` loaded again: the floor |
| b2 | `q8f` | Q8_0 | 512 | `GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1` | the Pascal prompt kernel's output × 1.05 |
| c | `f16d` | F16 | 1 | | the decode path's reference: every token its own pass |
| c | `q8d` | Q8_0 | 1 | | the decode path through the base's kernels (llama.cpp's vector kernel reads the Q8_0 cache) |
| c | `f16p` | F16 | 512 | | the decode pass's yardstick: the prompt path |
| c | `q8k` | Q8_0 | 1 | pull request 9's build, `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` | the decode kernel the rig run uses |
| e | `f16`, `q8`, `f16u` | | 512, 512, 128 | | pass b's arms on six chunks of 16,384 |
| f | `f16`, `q8` | | 512 | | the reference and the test arm on four chunks of 32,768 |
| v | `f16v`, `q8kv` | | 1 | `-lv 4`, `-b 512` | one chunk of 512 at a higher log level: the logs name the cache type and the decode kernel |

## Pass band (fixed before the run)

Each band is wider than the measurement's own error, so noise alone cannot fail or pass it.

- **(A) Against Ornith-1.5-122B's own Q8_0 model** (its reference log-probabilities on the agent text, 12 chunks of
  8,192 tokens): the Q8_0 cache's mean KLD at most 0.003 above the F16 cache's, and its same top token at most 0.3
  points below. One file's standard error on this text is about 0.001 in KLD and 0.1 points in same top token, so the
  bands are three of them, and more than twice the error of a difference of two independent runs.
- **(B) The Q8_0 cache against the F16 cache**, at each depth (8,192, 16,384, 32,768, 49,152): mean KLD at most
  the larger of twice the rounding yardstick's (the F16 cache read in micro-batches of 128 instead of 512, the same sums
  in another order) and 0.003; the share of tokens whose top token differs at most the larger of twice the yardstick's
  and 0.3 points; the mean log-ratio of the perplexities within ±0.01 (1 %).
- **(C) The decode path at 8,192** (every token its own pass): the Q8_0 cache through the base's kernels and through
  pull request 9's decode kernel, each against the F16 cache in the same passes, by (B)'s rule, with F16 read in
  micro-batches of 512 as the yardstick.
- **Validity:** a second load of the F16 reference must repeat it (KLD ≤ 0.0001, same top token 100 %), and the Q8_0
  cache with the Pascal prompt kernel's output scaled by 1.05 (its fault switch) must land outside (B)'s band at 8,192.

The Q8_0 cache keeps Ornith's quality if (A) holds and (B) holds at every depth; (C) says whether the decode kernel
that the rig run uses keeps it too.

**How each came out** ([verdict.tsv](verdict.tsv)): (A) passed; (B) passed in KLD and same top token at every depth and
in perplexity at 8,192 and 49,152, and failed in perplexity at 16,384 (+0.017132) and 32,768 (+0.010673); (C) passed;
the floor repeated; the fault stayed inside the band. Passes e and f were added after the run to test the two
perplexity failures across more text; they are not part of the band.

The single-window perplexity rule was too narrow for what it judged. Its ±1 % was wider than the per-token error
`llama-perplexity` reports (±0.0044 at 16,384), but that error treats neighbouring tokens as independent; across
stretches the Q8_0 cache's ln ratio spreads by 0.0072 a stretch (pass e), so one window can leave ±1 % on the text alone.
The per-stretch repeats, whose error holds for correlated tokens, are the measure to judge a perplexity change by.

## The result

### (A) Against Ornith-1.5-122B's Q8_0 model, 12 chunks of 8,192 tokens

From [against_q8_model.tsv](against_q8_model.tsv) and, per chunk, [chunks.tsv](chunks.tsv):

| Cache | Mean KLD | Same top token | Perplexity (the Q8_0 model's 1.568219) |
|---|---|---|---|
| F16 | 0.052227 ± 0.001020 | 95.008 ± 0.098 % | 1.579125 |
| Q8_0 | 0.052815 ± 0.001079 | 95.028 ± 0.098 % | 1.577901 |
| Q8_0 − F16, mean of the 12 chunks' differences | +0.000590 ± 0.000317 | +0.020 points | ln ratio −0.000780 ± 0.000742 |

The per-chunk difference has the honest error: each chunk is its own stretch of text, so the spread of the twelve
differences holds even though neighbouring tokens are correlated.

### (B) The Q8_0 cache against the F16 cache, by depth

A depth D is one chunk of the text's first D tokens, scored on its second half. Mean KLD ± its standard error, same top
token (%) and the mean log-ratio of the perplexities, each arm against `f16` on the same tokens ([kld.tsv](kld.tsv)):

| Depth (scored positions) | `q8` KLD | `f16u` KLD (yardstick) | `q8` same top | `f16u` same top | `q8` ln PPL ratio | `f16u` ln PPL ratio |
|---|---|---|---|---|---|---|
| 8,192 (4,096–8,191) | **0.009892 ± 0.000784** | 0.010796 ± 0.000816 | **97.900 ± 0.224** | 97.534 ± 0.242 | **−0.001453 ± 0.002695** | +0.000440 ± 0.002709 |
| 16,384 (8,192–16,383) | **0.027474 ± 0.002250** | 0.028489 ± 0.001996 | **97.290 ± 0.179** | 97.058 ± 0.187 | **+0.017132 ± 0.004448** | +0.004396 ± 0.004215 |
| 32,768 (16,384–32,767) | **0.033803 ± 0.002224** | 0.035843 ± 0.002338 | **97.149 ± 0.130** | 97.162 ± 0.130 | **+0.010673 ± 0.003139** | −0.002536 ± 0.003252 |
| 49,152 (24,576–49,151) | **0.012461 ± 0.000954** | 0.012870 ± 0.000918 | **98.271 ± 0.083** | 98.059 ± 0.088 | **+0.005996 ± 0.001482** | +0.002238 ± 0.001315 |

The floor (`f16b`, the reference loaded again) at 8,192 and 49,152: KLD −0.000003 and −0.000002, largest single-token
KLD 0.000010 and 0.000022, same top 100 %. `llama-perplexity` stores the reference's log-probabilities at 16 bits, so
equal logits score just below zero; every difference in the table comes from the arm.

### Perplexity on more stretches of text (added after the run)

Each chunk is its own stretch of text with its own context; chunk 1 of pass e is (B)'s 16,384 window and chunk 1 of
pass f is (B)'s 32,768 window. Each chunk's ln PPL ratio against `f16` ([chunks.tsv](chunks.tsv),
[chunks_summary.tsv](chunks_summary.tsv)):

| Pass | Arm | Chunk 1 | 2 | 3 | 4 | 5 | 6 | Mean ± SE across chunks |
|---|---|---:|---:|---:|---:|---:|---:|---|
| e, 16,384 | `q8` | +0.01713 | −0.00063 | +0.00285 | −0.00199 | −0.00036 | −0.00008 | +0.002820 ± 0.002934 |
| e, 16,384 | `f16u` | +0.00440 | +0.00272 | −0.00037 | +0.00001 | −0.00081 | +0.00119 | +0.001190 ± 0.000826 |
| f, 32,768 | `q8` | +0.01067 | −0.00111 | +0.00328 | +0.00112 | | | +0.003490 ± 0.002556 |

Over the six chunks of pass e the Q8_0 cache's KLD is 0.009697 ± 0.000436 and its same top token 98.222 ± 0.060 %.
Over the four chunks of pass f: 0.012008 ± 0.000578 and 98.312 ± 0.050 %.
Chunk 1, where the perplexity rose, is also where any change moves the output most: its KLD is 0.02747 for `q8` and
0.02849 for `f16u`, against 0.00411 to 0.01184 for the other chunks.

### (C) The decode path at 8,192, one token a pass

Each arm against `f16d` ([decode.tsv](decode.tsv)):

| Arm | Mean KLD | Same top token | ln PPL ratio |
|---|---|---|---|
| `q8d`, the base's kernels | 0.012175 ± 0.000968 | 97.924 ± 0.223 % | −0.001764 ± 0.002762 |
| `q8k`, pull request 9's decode kernel | 0.011923 ± 0.000904 | 97.753 ± 0.232 % | −0.003354 ± 0.002720 |
| `f16p`, the prompt path (yardstick) | 0.012152 ± 0.000841 | 97.534 ± 0.242 % | −0.002727 ± 0.002829 |

Pass v shows the decode kernel at work. At `-lv 4` the `q8kv` log names the cache (`K (q8_0): 3.19 MiB, V (q8_0):
3.19 MiB` for 512 cells, against `K (f16): 6.00 MiB, V (f16): 6.00 MiB` in `f16v`) and the kernel's one-time line
(`compute-6.1 decode attention kernel: head 256, q8_0 K/V, up to 4 rows`). Its window, the text's first 512 tokens
scored on positions 256–511 (255 tokens at a perplexity of 12.34), is too short to say anything about quality: `q8kv`
0.177723 ± 0.035987, the base's decode path `q8vd` 0.179559 ± 0.029122, and the same F16 cache as one prompt batch
`f16vp` 0.177384 ± 0.026362 ([decode.tsv](decode.tsv)); any change moves this window that far.

### What the tables say

- **The cache's error is the size of rounding.** In every comparison against the F16 cache, at every depth, in prompt
  passes and in one-token passes, the Q8_0 cache moves the output by less than reading the same F16 cache in another
  micro-batch does, in KLD and in same top token.
- **Against the model it approximates, the Q8_0 cache is as close as the F16 cache.** (A) measures quality the way the
  quality bar does, and the difference is +0.000590 ± 0.000317 in KLD and −0.08 % in perplexity, both far inside the
  band.
- **The perplexity rises in (B) follow the text.** A single window's perplexity moves with a few positions where the
  top two tokens are near a tie; the six stretches of pass e put the 16,384 rise on one stretch, where every change
  moves the output most. The four stretches of pass f do the same at 32,768: (B)'s window is the one at +1.07 %. Averaged over
  stretches, the rise beyond 8,192 is +0.28 % ± 0.29 % (16,384) and +0.35 % ± 0.26 % (32,768): not separable from
  zero, and below 1 % by more than two standard errors. Rounding's own average is +0.12 % ± 0.08 % (pass e), and single
  stretches move further with the Q8_0 cache (−0.20 % to +1.73 %) than with rounding (−0.08 % to +0.44 %), so a small
  cost of a few tenths of a percent is not ruled out.
- **The fault arm shows the check's limit.** The Pascal prompt kernel read the Q8_0 cache (its fault moves `q8`'s KLD
  from 0.009892 to 0.014446, 3.8 standard errors), but an output error of 5 % in that kernel lands inside (B)'s KLD band
  (0.021592) and moves the perplexity by +0.001113 ± 0.003051. On this model, KLD against a reference cannot tell an
  error of that size from rounding; FN64 found the same for a 1 % error on Flash-Next
  ([2026-11-11_fn64_sparse_prompt_ptx](../2026-11-11_fn64_sparse_prompt_ptx/README.md)).

**Limits.** These cards run the fork's prompt attention and matrix-vector kernels through the compute 6.1 PTX, forced
on with a P106-090's launch shapes, and llama.cpp's own kernels elsewhere; a P106 runs the same code natively. One load
per arm and depth. The decode path was scored at 8,192 only. (A) reaches 8,192 tokens, the length of the Q8_0 model's
reference; beyond it the comparison is against the F16 cache.

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the band: the value, the band and pass or fail |
| [against_q8_model.tsv](against_q8_model.tsv) | (A): both caches against Ornith-1.5-122B Q8_0's reference, and their difference |
| [kld.tsv](kld.tsv) | (B): per arm and depth, scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th and 99th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [decode.tsv](decode.tsv) | (C): the same columns for the one-token passes |
| [chunks.tsv](chunks.tsv), [chunks_summary.tsv](chunks_summary.tsv) | each chunk's own ln PPL ratio, KLD and same top token in passes a, e and f, and their means with the spread across chunks |
| [runs.tsv](runs.tsv) | every run: pass, arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the passes in order, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | the driver: FN64's, with per-arm arguments and builds, a chunk count and an outside reference |
| [run/c976_table.py](run/c976_table.py) | the tables and the band from the logs |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~70 GB free: the references
take 2 to 12 GB a depth; the bin folders unpacked from the two rig tarballs):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin BIN_PR9=<14d679604 tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  AGENT_BASE=<Ornith-1.5-122B-Q8_0's agent reference, 2026-10-27_ornith_kld> \
  MOUNTS="<the folders of the builds, model, text and reference>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver-1.log 2>&1 < /dev/null                    # passes a b b2 c d
# once $W/DONE exists (run.sh removes it at its start and writes it at its end):
PASSES="e f v" setsid -f nohup bash run/run.sh > $W/driver-2.log 2>&1 < /dev/null
python3 -I run/c976_table.py $W .
```

Pass a's two runs took about 7 minutes each, a load at 49,152 tokens about 4. The JIT cache (`$W/jit`) keeps the
compiled PTX between loads.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT). The Q8_0 reference is the makers' Q8_0.
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0; `run/agent_text.py` there); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
