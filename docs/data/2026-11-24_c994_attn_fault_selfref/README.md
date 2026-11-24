# Q1727: attention faults on Ornith-1.5-122B scored against their own unfaulted run, the form a new-kernel check takes (2026-11-24)

[C989](../2026-11-24_c989_attn_error_floor/README.md) calibrated the per-stretch quality check on Ornith-1.5-122B
Pascal N2P GGUF (`valid1`) with deliberate errors in the Pascal prompt attention kernel, scored against references that
have errors of their own: the F16 cache, and Ornith's own Q8_0 model. This folder scores the same faults against the
unfaulted run itself, the Q8_0 cache with the base kernel, so the reference adds no error of its own. That is the form a
check of a new attention kernel takes: the new kernel's output against the old kernel's, on the same cache, with the
old kernel in micro-batches of 128 as the yardstick for what reordering the sums alone does
([Q1727](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

**Result: against its own unfaulted run, a 5 % attention error stands well clear of rounding in KLD (6.54 standard
errors, 6.73 with each stretch divided by its rounding), but this form does not see smaller errors than C989 did. The
limit is rounding itself: a 5 % error moves the output only 1.39 times as far as reading the cache in micro-batches of
128 does, so errors below est. about 4.4 % move it less than that and pass unseen.**
- **The fault alone** (`=1`, output × 1.05) has a KLD of 0.011030 ± 0.001957 across the stretches (t 5.63) and makes
  the top token differ on 2.249 ± 0.263 % of tokens (t 8.56), beside C989's t 6.38 and 10.95. In this form those t
  values say nothing about size: any change shows, and reordering the sums alone has t 4.47 and 7.66.
- **Against rounding, the new-kernel check:** the fault adds 0.003070 ± 0.000469 KLD per stretch over the Q8_0 cache
  read in micro-batches of 128 (t 6.54), and more on every one of the 12 stretches. Divided by each stretch's rounding
  KLD: +0.504 ± 0.075 (t 6.73). In top-token disagreement it adds 0.188 ± 0.076 points (t 2.49, not seen), and divided
  +0.121 ± 0.038 (t 3.21, just seen). Per-stretch perplexity does not see it (t −0.02). C989's comparison against
  rounding, which still carried the F16 cache's own error, had t 4.37 in KLD and 4.11 in top-token disagreement.
- **The smallest error seen, est.:** 4.58 % in KLD above rounding, 4.43 % divided by rounding; 5.09 % and 4.97 % in
  top-token disagreement. An error of est. 4.25 % has rounding's KLD. C989's est. 3.43 % assumed a new kernel brings no
  rounding of its own; without rounding to beat, this form sees an error of any size, since the fault alone's t does
  not depend on its size.
- **The F16 reference hid most of the fault in C989.** The fault's own KLD is 2.58 times what C989 read as its added
  KLD against the F16 cache (0.011030 against 0.004280): stretch by stretch, C989's difference is 0.006750 ± 0.001396
  smaller (t −4.83), and 1.846 ± 0.245 points smaller in top-token disagreement. Scored against the F16 cache, part of
  the fault cancels against the Q8_0 cache's own change.
- **Dividing by rounding removes the stretches' sensitivity from the fault alone** (correlation 0.97 between the fault's
  and rounding's KLD on the same stretch, C989's was 0.86 against the F16 cache; the spread falls from 0.615 to 0.172 of
  the mean), but the difference from rounding already removes most of it, so t rises only from 6.54 to 6.73.
- **The dropped tile (`=3`)** is as gross as in C989: KLD 0.365240, same top token 85.336 %, perplexity +33.7 %: 12 to 14
  standard errors alone and against rounding, 5.59 and 8.84 divided by rounding.
- **The reference has no error of its own:** `q8` run again gives the same top token on every token and KLD 0.00000 on
  every stretch. Over all tokens, KLD −0.000003 and ln PPL ratio 0.000026 ± 0.000030 are the reference file's 16-bit
  storage of the log-probabilities.

**For later attention-kernel checks on Ornith:** score the new kernel against the old one on the same cache, stretch
by stretch, beside the old kernel in micro-batches of 128, and judge by the KLD above that yardstick, as a difference or
divided per stretch. That sees structural bugs at once and uniform errors from est. about 4.4 %. Smaller errors move
the output less than reordering the whole model's sums does. A new attention kernel reorders only the attention sums,
so a yardstick that changes only the attention kernel may be much smaller and would let the check see smaller errors
([Q1765](../../../quants/experiments.md)). Making the check a ready tool is
[Q1766](../../../quants/experiments.md).

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's and C989's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C976's and C989's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -ctk q8_0 -ctv q8_0`, `-ub 512` unless the arm says otherwise; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only), so each log names the kernel that took the prompt passes |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before the pass |
| Time | 2026-11-24 14:03–14:37, 5 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |

The faults are C989's ([its README](../2026-11-24_c989_attn_error_floor/README.md#the-faults)): the Pascal prompt
kernel's switch `GGML_CUDA_FATTN_TILE_PASCAL_FAULT`, `=1` scaling every output of the kernel by 1.05 (a uniform 5 %
error in every attention layer's output on every token) and `=3` dropping each KV split's last key tile.

### The arms (one pass, s)

| Arm | `-ub` | Fault | What it is |
|---|---:|---|---|
| `q8` | 512 | | writes the reference: the Q8_0 cache with the base kernel, C989's `q8` |
| `q8f1` | 512 | `=1` | a 5 % error in the attention output |
| `q8f3` | 512 | `=3` | the newest key tile of each split dropped |
| `q8u` | 128 | | the rounding yardstick: the same cache and kernel, the sums in another order |
| `q8r` | 512 | | `q8` again: whether the reference run reproduces itself |

## The check and its rule (fixed before the run)

For each stretch i, an arm's KLD against `q8`, its top-token disagreement (100 − same top %) and its ln PPL ratio:
F_i, T_i, L_i for `q8f1`; U_i, V_i for `q8u`. As in C989, the check reports the mean over the 12 stretches, its
standard error SE = sd/√12 and t = mean/SE; a stretch's own value comes from `llama-perplexity`'s running means
(k·M_k − (k−1)·M_{k−1}), with up to about 0.00006 of printing error a stretch at chunk 12, which is part of the
spread.

**The forms:**
- **The fault alone:** F_i and T_i, whose t is set beside C989's `q8f1 − q8` against the F16 cache (6.38 in KLD,
  10.95 in top-token disagreement). Against its own unfaulted run a fault's KLD is above zero on every stretch by
  construction, and if it grows with the square of the error, its t does not depend on the error's size. So this form
  says only that something changed, which any kernel that sums in a new order also does.
- **The fault against rounding, per stretch:** F_i − U_i (KLD) and T_i − V_i (top-token disagreement), and L_i − the
  yardstick's ln PPL ratio. This is the new-kernel check: is the new kernel further from the old one than reordering
  the sums is?
- **The same, divided by each stretch's rounding:** (F_i − U_i)/U_i and (T_i − V_i)/V_i. C989 found the fault's added
  KLD follows each stretch's sensitivity (correlation 0.86 with rounding's KLD), so dividing by the stretch's rounding
  may remove most of the spread.

**"Seen" means a mean at least 3 SE above zero** in KLD and top-token disagreement, in both forms against rounding
(a fault can only add on average), and |t| ≥ 3 in ln PPL ratio. With 11 degrees of freedom noise alone gets 3 SE
above zero 0.6 % of the time.

**The smallest error seen, est.:** a uniform output error of ε moves the logits in proportion to ε, so its KLD grows as
ε² and its top-token disagreement about as ε. Replacing F_i by (ε/0.05)²·F_i and T_i by (ε/0.05)·T_i in each form
against rounding, the smallest ε at which t reaches 3 is that form's estimate. Also the ε whose KLD equals rounding's,
est. 0.05·√(mean U / mean F) (C989: 6.58 %). These are estimates from one size and are marked so.

**Validity:**
- **The repeat:** `q8r` against `q8`, every stretch's KLD 0.00000 and same top 100.000 %; otherwise its values are the
  floor of this form and are reported.
- **The reference repeats C989:** `q8`'s perplexity over the 12 stretches equals C989's pass g `q8` (the same tokens,
  build and settings).
- **The kernel:** every arm's log names the compute 6.1 prompt attention kernel at head 256 with Q8_0 K/V.

**How it came out** ([verdict.tsv](verdict.tsv)): every validity check held. `q8r` matched `q8` on every stretch (KLD
0.00000, same top 100.000 %), `q8`'s run has C989's pass g `q8` perplexity exactly (1.577901), and every arm's log names
the prompt kernel at head 256 with Q8_0 K/V. `=1` is seen above rounding in KLD in both forms and in top-token
disagreement only divided per stretch; it is seen in no perplexity form. `=3` is seen in every form.

## The result

### Each arm over the 12 stretches

Every arm against the Q8_0 cache's own run (`q8`), 49,140 scored tokens ([kld.tsv](kld.tsv)). The ± in this table treats
neighbouring tokens as independent; the paired table below has the errors that hold.

| Arm | Mean KLD | Same top token | ln PPL ratio | 99th percentile KLD | Median KLD |
|---|---|---|---|---|---|
| `q8f1`: 5 % error | 0.011030 ± 0.000373 | 97.751 ± 0.067 % | +0.000377 ± 0.000827 | 0.139895 | 0.000268 |
| `q8f3`: dropped tile | 0.365240 ± 0.004507 | 85.336 ± 0.160 % | +0.290272 ± 0.005356 | 5.241600 | 0.014716 |
| `q8u`: rounding (`-ub 128`) | 0.007963 ± 0.000276 | 97.939 ± 0.064 % | +0.000428 ± 0.000703 | 0.094858 | 0.000185 |
| `q8r`: the repeat | −0.000003 | 100.000 % | +0.000026 ± 0.000030 | 0.000001 | 0.000000 |

### The forms, per stretch

Mean over the 12 stretches ± the standard error across stretches, and t = mean / SE; "seen" is t ≥ 3 in KLD and
top-token disagreement, |t| ≥ 3 in ln PPL ratio ([paired.tsv](paired.tsv)):

| Form | Arm or pair | KLD | t | Top token differs (points) | t | ln PPL ratio | t |
|---|---|---|---:|---|---:|---|---:|
| alone | `q8f1` | 0.011030 ± 0.001957 | 5.63 | 2.249 ± 0.263 | 8.56 | +0.00038 ± 0.00119 | 0.32 |
| alone | `q8u` | 0.007960 ± 0.001781 | 4.47 | 2.061 ± 0.269 | 7.66 | +0.00043 ± 0.00132 | 0.33 |
| against rounding | `q8f1 − q8u` | **+0.003070 ± 0.000469** | **6.54** | +0.188 ± 0.076 | 2.49 | −0.00005 ± 0.00216 | −0.02 |
| divided by rounding | `(q8f1 − q8u) / q8u` | **+0.504 ± 0.075** | **6.73** | **+0.121 ± 0.038** | **3.21** | | |
| alone | `q8f3` | 0.365240 ± 0.029057 | 12.57 | 14.664 ± 1.089 | 13.47 | +0.29027 ± 0.02346 | 12.37 |
| against rounding | `q8f3 − q8u` | **+0.357280 ± 0.027976** | **12.77** | **+12.603 ± 0.901** | **13.99** | **+0.28984 ± 0.02378** | **12.19** |
| divided by rounding | `(q8f3 − q8u) / q8u` | **+63.2 ± 11.3** | **5.59** | **+7.02 ± 0.79** | **8.84** | | |

Bold: seen by the rule. "Alone" was not judged: against its own run every change is above zero. Beside them, C989
against the F16 cache: `q8f1 − q8` t 6.38 (KLD) and 10.95 (top token); `q8f1 − f16u` (against rounding) t 4.37 and
4.11.

**The smallest error seen, est.** ([verdict.tsv](verdict.tsv)): with `q8f1`'s KLD scaled by (ε/0.05)² and its top-token
disagreement by ε/0.05, the smallest ε at which t reaches 3:

| Form | KLD | Top token differs |
|---|---|---|
| against rounding | 4.58 % | 5.09 % |
| divided by rounding | 4.43 % | 4.97 % |
| the error as large as rounding | 4.25 % | 4.58 % |

### The 5 % error stretch by stretch

Each stretch's own values ([chunks.tsv](chunks.tsv)): the fault's KLD, rounding's, the difference and the difference
divided by rounding; the same for top-token disagreement; and C989's added KLD of the same fault against the F16 cache
on the same stretch:

| Stretch | `q8f1` KLD | `q8u` KLD | difference | ÷ rounding | `q8f1` top differs | `q8u` top differs | difference | C989's `q8f1 − q8` KLD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01534 | 0.01267 | +0.00267 | +0.211 | 2.393 | 2.686 | −0.293 | +0.00456 |
| 2 | 0.00538 | 0.00359 | +0.00179 | +0.499 | 1.343 | 1.100 | +0.243 | +0.00250 |
| 3 | 0.01144 | 0.00879 | +0.00265 | +0.301 | 2.834 | 2.637 | +0.197 | +0.00386 |
| 4 | 0.02820 | 0.02495 | +0.00325 | +0.130 | 3.834 | 3.933 | −0.099 | +0.01092 |
| 5 | 0.00834 | 0.00605 | +0.00229 | +0.379 | 1.976 | 2.149 | −0.173 | +0.00336 |
| 6 | 0.00798 | 0.00557 | +0.00241 | +0.433 | 2.368 | 1.979 | +0.389 | +0.00300 |
| 7 | 0.01110 | 0.00810 | +0.00300 | +0.370 | 3.151 | 2.708 | +0.443 | +0.00498 |
| 8 | 0.00502 | 0.00244 | +0.00258 | +1.057 | 1.269 | 0.976 | +0.293 | +0.00258 |
| 9 | 0.00926 | 0.00569 | +0.00357 | +0.627 | 2.099 | 1.686 | +0.413 | +0.00483 |
| 10 | 0.00924 | 0.00575 | +0.00349 | +0.607 | 1.563 | 1.366 | +0.197 | +0.00491 |
| 11 | 0.00343 | 0.00209 | +0.00134 | +0.641 | 0.853 | 0.758 | +0.095 | +0.00202 |
| 12 | 0.01763 | 0.00983 | +0.00780 | +0.793 | 3.305 | 2.754 | +0.551 | +0.00384 |

The fault's KLD is above rounding's on all 12 stretches; its top-token disagreement is above on 9 and below on 3. The
fault's KLD and rounding's follow each other closely (correlation 0.97), so the stretches that move most under one move
most under the other.

### What the tables say

- **Against its own run, a fault's size is the KLD itself, with nothing taken away.** C989's comparison against the F16
  cache read the 5 % error as 0.004280; on its own it is 0.011030. Part of the fault's change cancels against the Q8_0
  cache's own change when both are scored against the F16 cache, so a reference with its own error can make a kernel
  error look smaller than it is.
- **The yardstick sets the limit.** In this form there is no reference error left, and the 12 stretches resolve the
  fault's excess over rounding to ± 0.000469. What stops smaller errors from being seen is that rounding is itself as
  large as a 4.25 % error (est.): reading the cache in micro-batches of 128 changes the order of every sum in the model,
  not only the attention's.
- **Top-token disagreement is the weaker measure here,** the opposite of C989: rounding tips nearly as many near-tied
  tokens as the 5 % error does (2.061 against 2.249 %).
- **Dividing by rounding helps little once rounding is subtracted:** t 6.54 to 6.73 in KLD, 2.49 to 3.21 in top-token
  disagreement.
- **Perplexity is still not a kernel-error test:** the 5 % error moves it by +0.00038 ± 0.00119, rounding by +0.00043
  ± 0.00132.

**Limits.** One fault size, so the smallest error seen is an estimate from the square law, not a measurement (Q1726's
size knob measures it). One kind of error (a uniform scale), and the prompt kernel only. One rounding yardstick, which
reorders the whole model; a new attention kernel reorders less, and two correct kernels may differ by less than this
yardstick. One load per arm. Per-stretch values come from `llama-perplexity`'s five-decimal running means, up to
0.00006 a stretch. These cards run the rig build's compute 6.1 PTX with the fork's kernels forced on.

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the rule: the value, the rule and seen or not; the estimates; the correlations and spreads; the cross term against C989; the validity checks |
| [paired.tsv](paired.tsv) | every form: the mean per stretch in KLD, top-token disagreement and ln PPL ratio, its standard error across stretches and t |
| [kld.tsv](kld.tsv) | per arm: scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th, 99th and 90th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [chunks.tsv](chunks.tsv) | each stretch's own ln PPL ratio, KLD and same top token, per arm |
| [runs.tsv](runs.tsv) | every run: arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the pass, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c994_table.py](run/c994_table.py) | the tables and the rule from the logs and C989's tables |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for the reference;
the bin folder unpacked from the rig tarball):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  MOUNTS="<the folders of the build, model and text>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # pass s; $W/DONE at the end
# once $W/DONE exists, from this folder:
python3 -I run/c994_table.py $W ../2026-11-24_c989_attn_error_floor .
```

The run took 34 minutes, 2026-11-24 14:03–14:37: five model loads, every one exit 0 ([runs.tsv](runs.tsv)); each
arm read the 24 GB reference from a hard disk.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
