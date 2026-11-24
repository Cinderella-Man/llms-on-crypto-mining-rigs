# Q1722: how small an attention-kernel error does the per-stretch quality check see on Ornith-1.5-122B? (2026-11-24)

[C976](../2026-11-23_c976_ornith_q8_cache_kld/README.md) judged a Q8_0 attention cache on Ornith-1.5-122B Pascal N2P
GGUF (`valid1`) by KLD and perplexity. Two of its measures behaved very differently. A single window's KLD could not
tell a 5 % error in the Pascal prompt kernel's output from rounding, while the paired difference over 12 stretches of
8,192 tokens resolved 0.000590 ± 0.000317. This folder calibrates that per-stretch check with deliberate kernel errors,
so that later attention-kernel checks on Ornith know what size of error they can and cannot see
([Q1722](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

**Result: against the F16 cache, the per-stretch check sees a 5 % error in the attention output clearly, and est. down
to about 3.4 % in KLD. Against Ornith's own Q8_0 model it only just sees 5 %, and per-stretch perplexity does not see
5 % at all. A dropped key tile is seen by every measure.**
- **A 5 % error (`=1`), against the F16 cache:** on the 12 stretches it adds 0.004280 ± 0.000671 to the KLD
  (6.4 standard errors) and 0.403 ± 0.037 points to the share of tokens whose top token differs (11 standard errors).
  It stands out from rounding too: 0.003570 ± 0.000817 more KLD than reading the F16 cache in micro-batches of 128
  (4.4 standard errors).
- **The same error against Ornith-1.5-122B's Q8_0 model,** the quality bar's comparison and C976's (A): it adds
  0.001470 ± 0.000463 (3.2 standard errors, just over the line) and leaves the top token as it was (−0.025 ± 0.070
  points). Part of the change must point toward the Q8_0 model's output, so this comparison sees less of it.
- **Perplexity per stretch does not see a 5 % error** in either comparison: +0.00035 ± 0.00118 and +0.00036 ± 0.00119
  in ln PPL (0.3 standard errors). The stretches move both ways and cancel.
- **The smallest error the check sees, est.:** 3.43 % in KLD against the F16 cache, an upper estimate that keeps the
  spread across stretches as it was at 5 %. The spread follows how sensitive each stretch is (correlation 0.86 with
  rounding's KLD on the same stretch), so it shrinks with the error. Against the Q8_0 model: est. 4.86 %. A 6.58 %
  error (est.) moves the output as far as rounding does.
- **A dropped key tile (`=3`)** is a gross error: KLD 0.366182 against the Q8_0 cache's 0.006699, same top token 85.368 %
  against 98.175 %, perplexity +33.6 %. That is 12 to 14 standard errors in every measure and comparison, and plain on
  every single stretch (the smallest rise in one stretch's KLD is +0.1667).

**For later attention-kernel checks on Ornith:** compare against the F16 cache or the base kernel, per stretch, and judge
by KLD and top-token disagreement. With 12 stretches of 8,192 tokens that comparison sees a uniform error of 5 % and,
est., down to about 3.4 %. Smaller errors pass unseen. A structural bug, such as a missing tile or a wrong mask, shows
plainly. The comparison against the Q8_0 model is the quality bar's measure: it sees kernel errors only from about 5 %.
Perplexity is not a kernel-error test. More stretches help little: twice as many lower the KLD estimate by about a
sixth, est. A size knob for the fault switch would measure the floor instead of estimating it, and scoring each fault
against its own unfaulted run would remove the reference's own error from the check. Both are new rows
([Q1726, Q1727](../../../quants/experiments.md)).

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C976's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096`, `-ub 512` unless the arm says otherwise; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only), so each log names the kernel that took the prompt passes |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before every pass |
| Time | 2026-11-24 00:37–01:19, 8 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |
| Kernels | attention: the fork's compute 6.1 prompt kernel (`ggml_cuda_flash_attn_ext_tile_pascal`, head 256, plain scores) in every prompt pass of every arm, with F16 or Q8_0 K/V as the arm's cache; each log names it ([verdict.tsv](verdict.tsv)). Ornith attends in 12 of its 48 layers (32 query heads, 2 KV heads); everything else ran as in C976, the same build and switches |

### The faults

The Pascal prompt attention kernel (`fattn-tile-pascal.cu`) has a fault switch for testing the tests,
`GGML_CUDA_FATTN_TILE_PASCAL_FAULT`. Only that kernel reads it, so a fault that moves the output also shows that the
kernel took the call. On a 3090 the kernel runs only when forced (`GGML_CUDA_FATTN_TILE_PASCAL=2`; its default `1`
asks for compute 6.1 exactly). It takes FLASH_ATTN_EXT at head 128 or 256 with F16 or Q8_0 K/V and at least 3 query
tokens with Q8_0 (9 with F16), so it takes every prompt pass here: Ornith's attention is head 256, and every
micro-batch is 512 tokens (128 in `f16u`).

- **`=1`, output × 1.05:** every output of the kernel scaled by 1.05, a uniform 5 % error in every attention layer's
  output on every token.
- **`=3`, last key tile dropped:** each KV split of a block skips its last key tile (128 keys at head 256). In a
  prompt pass of 512 tokens the kernel walks the whole cache written so far, so the dropped tiles are the newest keys,
  the micro-batch's own. A token near the end of a micro-batch loses its most recent keys, itself included; a token
  earlier in it loses nothing, since those keys were masked anyway. The number of splits comes from the card's
  occupancy and SM count, so how many keys a 3090 drops differs from a P106. A large error on a few tokens.

### The arms

| Pass | Arm | Cache | `-ub` | Fault | Scored against |
|---|---|---|---:|---|---|
| g | `f16` | F16 | 512 | | writes the reference: the v5 preset's cache and micro-batch |
| g | `q8` | Q8_0 | 512 | | the F16 cache |
| g | `q8f1` | Q8_0 | 512 | `=1` | the F16 cache |
| g | `q8f3` | Q8_0 | 512 | `=3` | the F16 cache |
| g | `f16u` | F16 | 128 | | the F16 cache: the rounding yardstick, the same sums in another order |
| h | `q8`, `q8f1`, `q8f3` | Q8_0 | 512 | none, `=1`, `=3` | Ornith-1.5-122B Q8_0's own log-probabilities on the same 12 stretches (C976's pass a); C976's `f16` there is the F16 cache's run |

## The check and its rule (fixed before the run)

**The paired per-stretch check.** For each stretch i (one chunk), take an arm's own KLD, top-token disagreement
(100 − same top %) and ln PPL ratio, and subtract the paired arm's on the same stretch: d_i. The check reports the mean
d̄ over the 12 stretches and its standard error SE = sd(d_i)/√12, and t = d̄/SE. Each stretch is its own stretch of
text with its own context, so the spread across stretches holds even though neighbouring tokens are correlated.
`llama-perplexity` prints running means over the chunks at five decimals; a chunk's own value is k·M_k − (k−1)·M_{k−1},
which carries up to about 0.00006 of printing error a stretch at chunk 12. That error is part of the spread, so the SE
includes it.

**"Seen" means d̄ at least 3 SE above zero** in KLD (a fault on top of an arm can only add KLD on average), and
|d̄| ≥ 3 SE in ln PPL ratio. With 11 degrees of freedom, noise alone gets 3 SE above zero 0.6 % of the time.

The pairs:
- **The fault against its own cache:** `q8f1 − q8` and `q8f3 − q8`, in pass g (against the F16 cache) and pass h
  (against the Q8_0 model). This is what the check sees of the fault alone.
- **The fault against rounding:** `q8f1 − f16u` and `q8f3 − f16u` in pass g; a later kernel check compares a new
  kernel with rounding, since any kernel that sums in a new order moves the output by rounding.
- **The fault the way C976 compared caches:** `q8f1 − f16` and `q8f3 − f16` against the Q8_0 model, with C976's
  `f16` run of pass a (the same build, settings, stretches and reference).

**The smallest scale error seen, est.:** a small error in the kernel's output changes the logits in proportion to its
size, and KLD is second order in a small logit change, so the fault's KLD grows with the square of the error. If
`q8f1 − q8` is seen with mean d̄ and error SE, the smallest output scale error the check sees is
est. 0.05 × √(3 SE / d̄). It is an estimate from one size, and it is marked so.

**Validity:**
- **The repeats:** in pass h, `q8` repeats C976's pass a `q8` on every stretch, and in pass g its first stretch repeats
  C976's pass b `q8` at 8,192 (0.009892), the same tokens with the same settings; otherwise C976's runs are not paired
  with this run's.
- **The kernel:** every arm's log names the compute 6.1 prompt attention kernel at head 256 with the arm's cache type.

**How it came out** ([verdict.tsv](verdict.tsv)): both repeats held (pass g's first stretch 0.00989, pass h's `q8`
equal to C976's on every stretch, KLD 0.052815 ± 0.001079 again), and every arm's log names the prompt kernel with its
cache type. Both faults are seen in KLD in every pair; `=1` is seen in no perplexity pair, `=3` in all of them.

## The result

### Each arm over the 12 stretches

Mean KLD ± its standard error, same top token and the mean log-ratio of the perplexities, 49,140 scored tokens
([kld.tsv](kld.tsv)):

| Pass | Arm | Against | Mean KLD | Same top token | ln PPL ratio | 99th percentile KLD | Median KLD |
|---|---|---|---|---|---|---|---|
| g | `q8` | F16 cache | 0.006699 ± 0.000258 | 98.175 ± 0.060 % | −0.000728 ± 0.000651 | 0.082733 | 0.000162 |
| g | `q8f1` | F16 cache | 0.010980 ± 0.000311 | 97.772 ± 0.067 % | −0.000377 ± 0.000798 | 0.143445 | 0.000270 |
| g | `q8f3` | F16 cache | 0.366182 ± 0.004520 | 85.368 ± 0.159 % | +0.289517 ± 0.005354 | 5.234957 | 0.014694 |
| g | `f16u` | F16 cache | 0.007409 ± 0.000190 | 97.996 ± 0.063 % | +0.001331 ± 0.000671 | 0.095878 | 0.000179 |
| h | `q8` | Q8_0 model | 0.052815 ± 0.001079 | 95.028 ± 0.098 % | +0.006155 ± 0.001750 | 0.791006 | 0.000883 |
| h | `q8f1` | Q8_0 model | 0.054291 ± 0.001129 | 95.053 ± 0.098 % | +0.006505 ± 0.001748 | 0.779864 | 0.000912 |
| h | `q8f3` | Q8_0 model | 0.382672 ± 0.004592 | 84.733 ± 0.162 % | +0.296400 ± 0.005410 | 5.314685 | 0.015914 |

The errors in this table treat neighbouring tokens as independent; the paired table below has the errors that hold.
C976's `f16` against the Q8_0 model, from its pass a: 0.052227 ± 0.001020, 95.008 ± 0.098 %.

### The paired per-stretch differences

Mean over the 12 stretches of (arm − paired arm) on the same stretch, ± the standard error across stretches, and t =
mean / SE; "seen" is t ≥ 3 in KLD and |t| ≥ 3 in ln PPL ratio ([paired.tsv](paired.tsv)):

| Against | Pair | KLD | t | Top token differs (points) | t | ln PPL ratio | t |
|---|---|---|---:|---|---:|---|---:|
| F16 cache | `q8f1 − q8`: 5 % error, alone | **+0.004280 ± 0.000671** | **6.38** | +0.403 ± 0.037 | 10.95 | +0.00035 ± 0.00118 | 0.30 |
| F16 cache | `q8f1 − f16u`: 5 % error, against rounding | **+0.003570 ± 0.000817** | **4.37** | +0.224 ± 0.054 | 4.11 | −0.00171 ± 0.00162 | −1.05 |
| F16 cache | `q8f3 − q8`: dropped tile, alone | **+0.359480 ± 0.028256** | **12.72** | +12.807 ± 0.929 | 13.79 | **+0.29025 ± 0.02347** | **12.37** |
| F16 cache | `q8f3 − f16u`: dropped tile, against rounding | **+0.358770 ± 0.028410** | **12.63** | +12.628 ± 0.926 | 13.64 | **+0.28819 ± 0.02336** | **12.34** |
| F16 cache | `q8 − f16u`: the Q8_0 cache against rounding | −0.000710 ± 0.000231 | −3.07 | −0.179 ± 0.058 | −3.07 | −0.00206 ± 0.00090 | −2.28 |
| Q8_0 model | `q8f1 − q8`: 5 % error, alone | **+0.001470 ± 0.000463** | **3.17** | −0.025 ± 0.070 | −0.36 | +0.00036 ± 0.00119 | 0.30 |
| Q8_0 model | `q8f1 − f16` (C976's): 5 % error and the Q8_0 cache | **+0.002060 ± 0.000423** | **4.87** | −0.045 ± 0.061 | −0.73 | −0.00042 ± 0.00170 | −0.25 |
| Q8_0 model | `q8f3 − q8`: dropped tile, alone | **+0.329850 ± 0.024567** | **13.43** | +10.295 ± 0.722 | 14.25 | **+0.29025 ± 0.02347** | **12.36** |
| Q8_0 model | `q8f3 − f16` (C976's) | **+0.330440 ± 0.024530** | **13.47** | +10.275 ± 0.695 | 14.77 | **+0.28947 ± 0.02354** | **12.30** |
| Q8_0 model | `q8 − f16` (C976's): C976's (A) again | +0.000590 ± 0.000317 | 1.86 | −0.020 ± 0.068 | −0.29 | −0.00078 ± 0.00074 | −1.05 |

Bold: seen by the rule in KLD or ln PPL ratio. Top-token disagreement was not part of the rule; its t is given beside.

### The 5 % error stretch by stretch, against the F16 cache

Each stretch's own values ([chunks.tsv](chunks.tsv)): the Q8_0 cache's KLD, rounding's KLD, and what each fault adds
to the Q8_0 cache's KLD, top-token disagreement and ln PPL ratio on that stretch:

| Stretch | `q8` KLD | `f16u` KLD | `q8f1 − q8` KLD | `q8f1 − q8` top differs (points) | `q8f1 − q8` ln PPL | `q8f3 − q8` KLD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00989 | 0.01080 | +0.00456 | +0.562 | +0.00256 | +0.3541 |
| 2 | 0.00293 | 0.00460 | +0.00250 | +0.292 | +0.00108 | +0.1667 |
| 3 | 0.00794 | 0.00932 | +0.00386 | +0.538 | +0.00212 | +0.3262 |
| 4 | 0.01992 | 0.01896 | +0.01092 | +0.588 | −0.00952 | +0.5168 |
| 5 | 0.00517 | 0.00577 | +0.00336 | +0.290 | −0.00064 | +0.3281 |
| 6 | 0.00473 | 0.00581 | +0.00300 | +0.490 | −0.00124 | +0.2869 |
| 7 | 0.00661 | 0.00865 | +0.00498 | +0.481 | +0.00263 | +0.3377 |
| 8 | 0.00201 | 0.00249 | +0.00258 | +0.399 | −0.00163 | +0.3571 |
| 9 | 0.00434 | 0.00452 | +0.00483 | +0.410 | −0.00274 | +0.4588 |
| 10 | 0.00446 | 0.00528 | +0.00491 | +0.300 | +0.00568 | +0.4502 |
| 11 | 0.00174 | 0.00190 | +0.00202 | +0.259 | +0.00038 | +0.2722 |
| 12 | 0.01066 | 0.01082 | +0.00384 | +0.227 | +0.00552 | +0.4588 |

The 5 % error raises the KLD and the top-token disagreement on all 12 stretches, and moves the perplexity up on seven and
down on five. What it adds to the KLD follows the stretch: correlation 0.86 with rounding's KLD on the same stretch
(stretch 4, where every change moves the output most, gets +0.01092; stretch 11 gets +0.00202).

### What the tables say

- **Against the F16 cache the check sees a 5 % error with room to spare.** In KLD it is 6.4 standard errors from zero,
  and 4.4 above rounding, which any new kernel brings with it. The top-token disagreement sees it more sharply still
  (11 standard errors): a uniform error tips the tokens whose top two candidates are nearly tied.
- **How small an error it sees (est.):** the KLD of a small output error grows with the square of its size. With the
  spread across stretches held at its value for 5 %, the check reaches 3 standard errors at a 3.43 % error. The spread
  is mostly the stretches' own sensitivity, which shrinks with the error, so 3.43 % is the upper estimate. For the
  top-token disagreement, which grows about in proportion to the error, the same reasoning gives est. 1.37 %; that
  estimate was not part of the rule fixed before the run. Reading the cache in micro-batches of 128 moves the output as
  far as a 6.58 % error would (est.).
- **Against the Q8_0 model the check is weaker.** The same 5 % error adds a third as much KLD there (0.001470 against
  0.004280) and no top-token disagreement, so part of its change points toward the larger model's output. The quality
  bar's comparison resolves cache and file quality to about 0.0003 in KLD (C976), but a kernel error only from about
  5 % (est. 4.86 %).
- **Perplexity cannot see a uniform 5 % error.** Its per-stretch differences go both ways and average to +0.00035
  ± 0.00118. It does see the dropped tile (+0.29 in ln PPL, +33.6 %).
- **A structural bug is not subtle.** Dropping the newest key tile of each split corrupts the tokens at the end of each
  micro-batch, and through their keys and values in the cache every later token: the median token's KLD goes from
  0.000162 to 0.014694, the smallest single stretch's rise is +0.1667, and every measure sees it on any one stretch.
- **The Q8_0 cache moves the output less than rounding does** (KLD 0.000710 ± 0.000231 below `f16u`, 3.1 standard
  errors), as C976 found at every depth.

**Limits.** Two fault sizes, so the smallest error seen is an estimate from one size, not a measurement. One kind of
error (a uniform scale); an error that hits some heads, layers or positions would spread differently. The faults hit
the prompt kernel only; the decode path was not faulted. These cards run the rig build's compute 6.1 PTX with the
fork's kernels forced on; how many keys `=3` drops depends on the card's occupancy, so a P106 drops a different count.
One load per arm. The per-stretch values come from `llama-perplexity`'s five-decimal running means, which adds up to
0.00006 a stretch to the spread.

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the rule: the value, the rule and seen or not; the estimates; the validity checks |
| [paired.tsv](paired.tsv) | every pair: the mean per-stretch difference in KLD, top-token disagreement and ln PPL ratio, its standard error across stretches and t |
| [kld.tsv](kld.tsv) | per arm: scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th, 99th and 90th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [chunks.tsv](chunks.tsv) | each stretch's own ln PPL ratio, KLD and same top token, per pass and arm |
| [runs.tsv](runs.tsv) | every run: pass, arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the two passes, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c989_table.py](run/c989_table.py) | the tables and the rule from the logs and C976's tables |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for pass g's
reference; the bin folder unpacked from the rig tarball):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  AGENT_BASE=<Ornith-1.5-122B-Q8_0's agent reference, 2026-10-27_ornith_kld> \
  MOUNTS="<the folders of the build, model, text and reference>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # passes g and h; $W/DONE at the end
# once $W/DONE exists, from this folder:
python3 -I run/c989_table.py $W ../2026-11-23_c976_ornith_q8_cache_kld .
```

Each run took about 5 minutes (a model load and 12 chunks); `f16u`, in micro-batches of 128, took 7.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT). The Q8_0 reference is the makers' Q8_0.
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
