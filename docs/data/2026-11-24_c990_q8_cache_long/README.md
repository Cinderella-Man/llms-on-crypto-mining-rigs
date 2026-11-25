# Q1723: does a Q8_0 cache keep Ornith-1.5-122B's quality at 65,536 and 98,304 tokens? (2026-11-24)

A Q8_0 attention cache is half the size of Ornith's F16 cache, so it allows a longer context on the same cards. Up to
49,152 tokens its quality was within noise of the F16 cache
([2026-11-23_c976_ornith_q8_cache_kld](../2026-11-23_c976_ornith_q8_cache_kld/README.md), Q1707), with one caveat: past
8,192 tokens its perplexity ran a few tenths of a percent higher on average (+0.28 % on six stretches of 16,384 tokens,
+0.35 % on four of 32,768), which that check could neither show nor rule out. With pull request 9's decode kernel the
cache was a gain on agentic speed on the rig at 49,152 tokens ([2026-11-24_r232b_ornith_q8_agent](../2026-11-24_r232b_ornith_q8_agent/README.md)).
Before the rig tries a longer context ([Q1807](../../../quants/experiments.md)), this folder checks the cache at
65,536 and 98,304 tokens: does the perplexity rise stay where C976 put it, or grow with the context? It runs the same
comparison as C976 (Ornith-1.5-122B Pascal N2P GGUF with `-ctk q8_0 -ctv q8_0` against the same file with its F16
cache, on held-out agent text, through the rig build's compute 6.1 PTX on three RTX 3090s), through the Pascal prompt
kernel and through the decode kernel. It is a correctness and quality check only: no time on these cards means
anything for a P106.

**Result: the perplexity rise does not grow with the context, and it stays within 1 %.**
- **Through the prompt kernel** the Q8_0 cache's perplexity is +0.233 % ± 0.228 % above the F16 cache's at 65,536
  tokens and +0.236 % ± 0.216 % at 98,304 (three stretches each), the same size as C976's +0.282 % ± 0.293 % at
  16,384 and +0.349 % ± 0.256 % at 32,768. By the rule fixed before the run it is not measured growing (t −0.34 at
  both depths against 32,768) and within 1 % at both (mean plus three standard errors 0.916 % and 0.883 %).
- **The rise is one stretch at each depth:** +0.687 % at 65,536 and +0.664 % at 98,304 on the first stretch (the
  text's opening, where C976's single windows also rose most); +0.033 %, −0.021 %, −0.022 % and +0.066 % on the
  other four.
- **KLD and top-token disagreement beside it:** 0.007094 and 1.582 % of tokens at 65,536, 0.021781 and 2.127 % at
  98,304 (C976: 0.009700 and 1.778 % at 16,384, 0.012010 and 1.688 % at 32,768). They move with the text more than
  with the depth (per stretch 0.01313, 0.00239 and 0.00575 at 65,536; 0.03342, 0.00814 and 0.02378 at 98,304), and at both depths reading the F16 cache one token a pass moves the output
  further than the Q8_0 cache does: 0.008836 and 1.755 %, 0.027542 and 2.408 %.
- **Through the decode kernel** (one token a pass) the cache costs what it costs through the prompt kernel. Against the
  F16 cache read the same way its perplexity is +0.255 % ± 0.247 % at 65,536 and +0.431 % ± 0.394 % at 98,304:
  0.022 % ± 0.031 % (t 0.71) and 0.195 % ± 0.210 % (t 0.93) more than through the prompt kernel, so no added error is
  seen. On top of the F16 decode path it adds KLD 0.000690 ± 0.000613 and 0.000540 ± 0.000188 and top-token
  disagreement +0.028 and +0.061 points: about what a 1 % error in the attention output added in the same kind of
  comparison on the same day (KLD +0.000500 ± 0.000157,
  [2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)). Within 1 % at 65,536 (0.996 %); at
  98,304 three stretches cannot show it (1.613 %: its first stretch is +1.193 %).
- **The fit:** 98,304 tokens fit three 24 GB cards and the build machine's 121.5 GiB host. The Q8_0 cache is 1,224 MiB
  there against the F16 cache's 2,304; the cards held at most 17,213, 18,583 and 19,297 MiB (in the reference run,
  while another job's small checks shared them), and the host's available memory went down to 18.51 GiB in that run.
- **One part of the rule was flawed, and I say so plainly:** rule 2's KLD and top-token parts (see "How each came
  out"). Its perplexity part, which carries the decode verdict above, holds.

For the rig: Ornith's Q8_0 cache keeps the quality it has at 49,152 tokens at 65,536 and 98,304, so the longer context
can be tried on the rig ([Q1807](../../../quants/experiments.md)). As at 49,152, a perplexity cost of a few tenths of a
percent is not ruled out; the agent check's tasks are the place to see whether it matters.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `af00bb13b`, which holds pull request 9's decode kernel (rig tarball `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz`, md5 `b51ad2740d5822e7e14af656c413ec22`; its `llama-perplexity` md5 `1c967216e551e89cff64fb769bf1de83`). Its GPU code is sm_61 machine code and compute 6.1 PTX only, so a newer card runs the PTX |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`), as in C976 |
| Depths | 98,304 tokens (run first: its reference run is the fit check) and 65,536 tokens, three chunks each (`-c D --chunks 3`): each chunk is its own stretch of the text, scored on its second half (positions D/2 to D−1), so a depth is three independent repeats |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight passes before each depth |
| Settings in every run | C976's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -lv 4`; the v5 preset's engine variables `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes: `GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`; `GGML_SCHED_DEBUG_REALLOC=1` |

The arms (each one load of `llama-perplexity`; every arm is scored against `f16`'s log-probabilities on the same
tokens):

| Arm | Cache | `-ub` | Switches | What it is |
|---|---|---:|---|---|
| `f16` | F16 | 512 | | the reference: the v5 preset's cache and micro-batch |
| `q8` | Q8_0 | 512 | `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` | the cache as the rig's agent runs used it, through the Pascal prompt kernel |
| `q8k` | Q8_0 | 1 | the same | every token its own pass: the cache through pull request 9's decode kernel at every position up to the depth |
| `f16d` | F16 | 1 | | every token its own pass, as the preset decodes: the decode path's F16 control, so `q8k` − `f16d` is the cache's cost through the decode path |

`GGML_CUDA_FATTN_DEC_PASCAL=4` sends one to four query tokens with a Q8_0 cache at head 256 to the decode kernel,
which is reached only from `GGML_CUDA_FATTN_TILE_PASCAL_MIN` tokens, hence `_MIN=1`. With 512-token micro-batches both
switches leave the prompt kernel in charge; at one token a pass every attention call of `q8k` goes to the decode kernel.

## Rule (fixed before the run)

Per depth D and per stretch s (three each), each arm's ln(PPL(arm)/PPL(`f16`)), mean KLD and top-token disagreement
(100 − same top %), from `llama-perplexity`'s per-chunk lines. Across the three stretches: the mean, its standard
error (the spread of the three divided by √3) and t = mean / standard error. The spread across stretches is the error
that holds for neighbouring tokens being correlated (C976 found the per-token error too small for that).

1. **The answer (prompt kernel, `q8` against `f16`), per depth:**
   - **grows with the context** if its mean ln PPL ratio m exceeds C976's at 32,768 tokens (+0.003490 ± 0.002556 over
     four stretches) by t = (m − 0.003490) / √(SE² + 0.002556²) ≥ 3;
   - **not measured growing** if that t is below 3;
   - **within 1 %** if m + 3 SE ≤ 0.01 (C976 judged its repeats by the same 1 %, at two standard errors).
   KLD and top-token disagreement are reported beside it, against C976's per-stretch means (16,384: 0.009697 and
   1.778 points over six stretches; 32,768: 0.012008 and 1.688 over four).
2. **The decode kernel, per depth:** per stretch, Δ = (`q8k` − `f16d`) − (`q8` − `f16`), in ln PPL ratio, KLD and
   top-token disagreement (`f16` against itself is zero). **The decode kernel adds an error at this length** if Δ's
   t ≥ 3 in KLD or top-token disagreement (one-sided, as [2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)
   judges a fault) or |t| ≥ 3 in ln PPL ratio. The cache's cost through the decode path, `q8k` − `f16d` in ln PPL ratio,
   is **within 1 %** if its mean + 3 SE ≤ 0.01.
3. **Validity** (a depth's verdict holds only if all of these hold): every run exits 0 and logs `computing over 3
   chunks, n_ctx=D`; the logs of `f16` and `f16d` show an F16 cache and those of `q8` and `q8k` a Q8_0 cache; `q8`'s
   log names the prompt kernel at head 256 with Q8_0 K/V, `q8k`'s the decode kernel (`head 256, q8_0 K/V, up to 4
   rows`), and `f16d`'s no decode kernel.
4. **Fit:** the reference at 98,304 tokens finishing is the fit on three 24 GB cards and this host; the lowest host
   memory available during the run is reported. If it does not fit, the largest multiple of 16,384 tokens whose
   reference runs takes its place, and this page says so.

What size of attention error this kind of check sees was measured on Ornith the same day
([2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)): against an unfaulted run, an error of
1 % in the attention output by KLD and of 2 % by top-token disagreement, on twelve stretches of 8,192 tokens. Three
stretches give a larger standard error; the rule above takes it from the data.

## How each came out

From [verdict.tsv](verdict.tsv) and [validity.tsv](validity.tsv); t is the mean over its standard error across the
three stretches:

| Rule | 65,536 | 98,304 |
|---|---|---|
| 1, growth: `q8` ln PPL ratio against C976's 32,768 (+0.003490 ± 0.002556) | +0.002330 ± 0.002275, t −0.34: **not measured growing** | +0.002360 ± 0.002155, t −0.34: **not measured growing** |
| 1, bound: mean + 3 SE ≤ 0.01 | 0.009156: **within 1 %** | 0.008825: **within 1 %** |
| 2, ln PPL: (`q8k` − `f16d`) − `q8` | +0.000220 ± 0.000310, t 0.71: **no added error seen** | +0.001950 ± 0.002095, t 0.93: **no added error seen** |
| 2, KLD: (`q8k` − `f16d`) − `q8` (flawed, below) | −0.006400 ± 0.002575, t −2.49 | −0.021240 ± 0.007264, t −2.92 |
| 2, top-token disagreement: the same (flawed, below) | −1.554 ± 0.317 points, t −4.91 | −2.066 ± 0.180 points, t −11.46 |
| 2, bound: `q8k` − `f16d` ln PPL, mean + 3 SE ≤ 0.01 | +0.002550 ± 0.002471: 0.009962, **within 1 %** | +0.004310 ± 0.003942: 0.016135, **not shown within 1 %** |
| 3, validity | **holds** | **holds** |
| 4, fit | | **fits** (lowest host memory available 18.51 GiB) |

- **Validity:** all eight runs exit 0; every log reports three chunks at its depth (the reference's log words it
  `calculating perplexity over 3 chunks`, the scored arms' `computing over 3 chunks`); `f16` and `f16d` log an F16
  cache, `q8` and `q8k` a Q8_0 cache (816 MiB at 65,536 and 1,224 MiB at 98,304, against 1,536 and 2,304 MiB);
  `q8`'s log names the prompt kernel at head 256 with Q8_0 K/V, `q8k`'s the decode kernel (`head 256, q8_0 K/V, up to
  4 rows`) and no prompt kernel, `f16d`'s neither Pascal kernel. `q8`'s log names the decode kernel too: the load's
  warm-up pass is a few tokens long.
- **Rule 2's KLD and top-token parts were flawed.** They subtract `q8`'s divergence, which is measured from the F16
  reference itself, from `q8k` − `f16d`, which is measured on top of the decode path's own divergence. Divergences on
  the same tokens do not add: the F16 cache read one token a pass already moves the output as far as `q8k` does (KLD
  0.027542 against 0.028077 at 98,304; top token differs on 2.408 % against 2.469 % of tokens), so those differences
  came out strongly negative and judge nothing. In their place, `q8k` − `f16d` itself is set beside the same day's
  measured floor, which has the same form (an arm on top of an already moved run, both against F16): see the decode
  table. The ln PPL part is exact, since log ratios against one reference subtract, and it holds.

## The result

### The prompt kernel, by depth

Each stretch is its own piece of the text with its own context, scored on its second half; mean ± standard error across
stretches ([summary.tsv](summary.tsv), [chunks.tsv](chunks.tsv); C976's from its
[chunks.tsv](../2026-11-23_c976_ornith_q8_cache_kld/chunks.tsv)):

| Depth (scored positions) | Stretches | ln PPL ratio `q8` − `f16` | Per stretch | KLD | Top token differs (% of tokens) |
|---|---:|---|---|---|---|
| 16,384 (8,192–16,383), C976 | 6 | +0.002820 ± 0.002934 | +0.01713, −0.00063, +0.00285, −0.00199, −0.00036, −0.00008 | 0.009700 ± 0.003620 | 1.778 ± 0.246 |
| 32,768 (16,384–32,767), C976 | 4 | +0.003490 ± 0.002556 | +0.01067, −0.00111, +0.00328, +0.00112 | 0.012010 ± 0.007298 | 1.688 ± 0.412 |
| **65,536** (32,768–65,535) | 3 | **+0.002330 ± 0.002275** | +0.00687, +0.00033, −0.00021 | 0.007090 ± 0.003172 | 1.582 ± 0.331 |
| **98,304** (49,152–98,303) | 3 | **+0.002360 ± 0.002155** | +0.00664, −0.00022, +0.00066 | 0.021780 ± 0.007366 | 2.127 ± 0.215 |

C976's first stretch at each depth is the text's first D tokens, as here, so the four first stretches overlap: that
opening is where every arm moves most.

### The decode kernel

Each arm against `f16` on the same tokens, and the cache's cost inside the decode path (`q8k` − `f16d`), mean ±
standard error across the three stretches ([summary.tsv](summary.tsv)):

| Depth | Arm | ln PPL ratio | Per stretch | KLD | Top token differs (% or points) |
|---|---|---|---|---|---|
| 65,536 | `q8` (prompt kernel) | +0.002330 ± 0.002275 | +0.00687, +0.00033, −0.00021 | 0.007090 ± 0.003172 | 1.582 ± 0.331 |
| 65,536 | `q8k` (decode kernel) | +0.003810 ± 0.003242 | +0.01029, +0.00077, +0.00037 | 0.009530 ± 0.004416 | 1.783 ± 0.367 |
| 65,536 | `f16d` (F16, one token a pass) | +0.001260 ± 0.000860 | +0.00287, −0.00007, +0.00098 | 0.008840 ± 0.003816 | 1.755 ± 0.344 |
| 65,536 | **`q8k` − `f16d`** | **+0.002550 ± 0.002471** | +0.00742, +0.00084, −0.00061 | **+0.000690 ± 0.000613** (t 1.13) | **+0.028 ± 0.052** (t 0.54) |
| 98,304 | `q8` (prompt kernel) | +0.002360 ± 0.002155 | +0.00664, −0.00022, +0.00066 | 0.021780 ± 0.007366 | 2.127 ± 0.215 |
| 98,304 | `q8k` (decode kernel) | +0.003750 ± 0.002675 | +0.00910, +0.00100, +0.00115 | 0.028080 ± 0.009063 | 2.469 ± 0.256 |
| 98,304 | `f16d` (F16, one token a pass) | −0.000560 ± 0.001549 | −0.00283, −0.00125, +0.00240 | 0.027540 ± 0.008959 | 2.408 ± 0.218 |
| 98,304 | **`q8k` − `f16d`** | **+0.004310 ± 0.003942** | +0.01193, +0.00225, −0.00125 | **+0.000540 ± 0.000188** (t 2.87) | **+0.061 ± 0.040** (t 1.52) |

The measured floor for comparison ([2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)): the
Pascal prompt kernel's output scaled by 1.01 (a 1 % attention error), on top of the Q8_0 cache's prompt path and
against F16, added KLD +0.000500 ± 0.000157 (t 3.18) on twelve stretches of 8,192 tokens; scaled by 1.02, top-token
disagreement +0.147 ± 0.041 points (t 3.56). The Q8_0 cache's whole cost inside the decode path at 65,536 and 98,304
tokens is that size or smaller. Whole-run figures of each arm (per-token errors, percentiles of KLD, perplexities) are
in [kld.tsv](kld.tsv).

### The fit

From the memory log the run kept every 30 s ([fit.tsv](fit.tsv)): the reference run at 98,304 tokens is the largest
(it holds the chunk's log-probabilities for the whole chunk and the logits of its scored half); the host's available
memory went down to 18.51 GiB of 121.5 GiB there, to 41.98, 44.85 and 45.28 GiB in the scored arms at 98,304 and to
51.83 GiB or more at 65,536. The cards' highest use, 17,213, 18,583 and 19,297 MiB of 24,576, came in that reference run, while another
job's small checks were on the same cards; the scored arms at 98,304 held at most 15,775, 16,991 and 18,025 MiB. The
host's 8 GB swap file was near full during `f16d` at 98,304 (1.13 GiB free at the lowest) and full for one 30-second
sample during `q8` at 65,536, while at least 45 GiB of memory stayed available; it was empty again after the run.

At 131,072 tokens this `llama-perplexity` would not fit the same host: its reference run sizes the log-probability
buffer for the whole chunk though it writes only the scored half, est. 125 GiB in all with a vocabulary of 248,320
tokens ([Q1820](../../../quants/experiments.md) halves it).

## Files

| File | What it is |
|---|---|
| [verdict.tsv](verdict.tsv) | the rule applied: each check, its value, its line and its result (and the two rows added after the run) |
| [validity.tsv](validity.tsv) | each run's exit status, chunk count and depth, cache types and the attention kernels its log names |
| [summary.tsv](summary.tsv) | per depth: each arm's and each difference's mean, standard error across stretches and t, in ln PPL ratio, KLD and top-token disagreement |
| [chunks.tsv](chunks.tsv) | per stretch: each arm's ln PPL ratio, KLD and same top token against `f16` |
| [kld.tsv](kld.tsv) | each arm's whole-run figures from `llama-perplexity` (per-token errors, KLD percentiles, perplexities) |
| [fit.tsv](fit.tsv) | per run: the host's lowest available memory and swap and each card's highest use |
| [runs.tsv](runs.tsv) | the eight runs: arm, depth, micro-batch, exit status, start and end |
| [run/run.sh](run/run.sh) | the run: both depths, four arms each, the memory log |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c990_table.py](run/c990_table.py) | the tables above from the logs, and the rule |

## Commands

```bash
WORK=<work dir on a disk with ~130 GB free> BIN=<build>/bin MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> \
  TEXT=<agent-heldout.txt> GPUS=0,1,2 MOUNTS="<folders the container reads>" \
  DOCKER_ARGS="--cpu-shares=256" PREFLIGHT=<cuda check script> \
  setsid nohup bash run/run.sh > driver.log 2>&1 < /dev/null &
python3 -I run/c990_table.py <WORK> .
```

Ran 2026-11-24 20:47 to 2026-11-25 01:14 on a build machine's three RTX 3090s, eight runs (one model load each), every
one exit 0 ([runs.tsv](runs.tsv)).

Host memory: at 98,304 tokens the reference run holds the chunk's log-probabilities for the whole chunk (98,304 ×
248,324 × 2 bytes, 48.8 GB) and the logits of its scored half (49,152 × 248,320 × 4 bytes, 48.8 GB), about 95 GiB in
all; a scored arm about 72 GiB. Base files: 73.2 GB at 98,304 and 48.8 GB at 65,536.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0; `run/agent_text.py` there); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
