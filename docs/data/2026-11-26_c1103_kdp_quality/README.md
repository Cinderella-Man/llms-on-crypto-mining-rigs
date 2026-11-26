# Q1836: does `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` keep Ornith-1.5-122B's quality? (2026-11-26)

**Result: `_KDP=1` keeps Ornith's quality. Rounding the query to 8 bits in the prompt attention kernel moves the
output no further than summing in another order does, over 12 stretches of 8,192 tokens and at 49,152 tokens;
perplexity moves by +0.06 % over the stretches and +0.20 % at 49,152.** Whether the preset takes the switch is the owner's call; the speed side
is the two rig rows linked below.
- **Against the preset, over the 12 stretches:** KLD 0.005951, same top token on 98.382 % of tokens, perplexity
  +0.060 % ± 0.061 % (mean of the per-stretch ln ratios). The two yardsticks measured in the same run: the attention-only
  one (the build's generic attention kernel) 0.006160 and 98.462 %; the whole model's sums
  reordered (micro-batches of 128) 0.005794 and 98.466 %.
- **Within both yardsticks, stretch by stretch:** `_KDP` minus the attention-only yardstick −0.000210 ± 0.000239 KLD
  (t −0.88), minus the reordering one +0.000160 ± 0.000238 (t 0.67); in top-token disagreement +0.080 ± 0.046 points
  (t 1.73) and +0.084 ± 0.070 (t 1.20). Its KLD follows the attention-only yardstick's stretch by stretch (correlation
  0.99): it is the size of what any change of the attention's arithmetic does on this model.
- **At 49,152 tokens:** KLD 0.011747 against 0.010911 and 0.010408 for the two yardsticks at the same window (1.08 and
  1.13 times them, inside C976's band of twice the yardstick), same top token 98.344 % against 98.368 % and 98.417 %,
  perplexity +0.20 % ± 0.13 %.
- **The check could see an error:** the kernel's output scaled by 1.05 stands above the attention-only yardstick at
  t 4.96 (divided per stretch, t 7.72), so uniform attention errors from est. 3.8 % to 4.1 % would show. `_KDP`'s own
  change is as large as a uniform error of est. 3.64 %, the same size as the yardsticks' (3.59 % and 3.70 %), and
  0.53 of the 5 % error's KLD (per stretch −0.005290 ± 0.001224).
- **TF32 matters for the yardsticks, not for the comparison, and the yardsticks were measured again.** Both arms of
  the check ran the same cuBLAS settings. The yardsticks did move: with cuBLAS computing as on a P106 (C1108) and on
  today's engine, the whole model's reordering is 0.728 of C994's (0.005794 against 0.007963, measured with TF32 on and
  on `3dfad876f`), while the attention-only yardstick barely moved (0.975 of C1032's 0.006319). Their order is now
  reversed: the reordering is 0.94 of the attention-only one, where C1032 found 1.26. On the same engine, switching
  TF32 and F16 cuBLAS back on alone moves Ornith's output by KLD 0.007139, more than either yardstick (per stretch t 4.81
  and 4.75 above them) and more than `_KDP` does.
- **One validity check failed by one token:** the preset loaded again repeated the reference with KLD 0.00000 on every
  stretch, but not bit for bit: from the third stretch on single tokens differ slightly (largest 0.000059), and one top
  token of 49,140 differs, where the rule fixed before the run asked for 100.000 %. C994's repeat on an older build
  with TF32 on was identical. This floor is four orders of magnitude below every difference above, so it does not move
  the verdict; which setting makes the run vary is [Q1902](../../../quants/experiments.md).

## What the switch changes

Ornith's v5 preset keeps its attention cache in Q8_0 and reads it on a P106 with the fork's compute 6.1 prompt
attention kernel (`flash_attn_tile_pascal`). That kernel decodes each K block to floats and scores it against the
query in FP32. With `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` (pull request 9, a contributor's) it keeps K as int8 in shared
memory and scores it with `dp4a` against the query rounded to int8 in blocks of 32 values, one scale a block, in every
prompt-kernel call of 5 query tokens or more at head 256. The query's rounding is new arithmetic, not only a new order
of sums, so it can change the output. The same rounding is what llama.cpp's CPU reference and its own vector kernel do
for a Q8_0 cache, and what the preset's decode kernel (`GGML_CUDA_FATTN_DEC_PASCAL=4`, 1 to 4 tokens) already does;
that kernel passed C976's decode check ([2026-11-23_c976_ornith_q8_cache_kld](../2026-11-23_c976_ornith_q8_cache_kld/README.md), (C)).

On a P106 the switch made the preset's prompt attention 39.67 % faster at 8,192 tokens
([2026-11-25_c1037_pr9_lds_kdp](../2026-11-25_c1037_pr9_lds_kdp/README.md)) and Ornith's coding-agent tasks 1.9 %
shorter at equal work ([2026-11-25_c1102_kdp_agent](../2026-11-25_c1102_kdp_agent/README.md)). This folder is the
quality check that comes before the preset takes it ([Q1836](../../../quants/experiments.md)). It is a correctness and
quality check on three RTX 3090s through the rig build's compute 6.1 PTX; no time on these cards means anything for a
P106.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `2d839da1d` (rig tarball `llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3.tar.gz`, md5 `7d774f05c29460a6993b19d38796047c`; its `llama-perplexity` md5 `8ebeab4e6ab839cfff0454625fb6db0b`). Its GPU code is sm_61 machine code and compute 6.1 PTX only, so a newer card runs the PTX |
| Why this build | it is the engine the preset runs on today (`pascal-p106-v5`). The speed rows ran the b12674 candidate `eee615cdb`, its parent; between the two the Pascal attention file changed only so that the fault switch's factor can be set (default 1.05, unchanged), and the other changes are off by default or touch no computation (a 64-row MMQ tile behind its own switch, logging, a probe), so both builds compute the same here |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191); and one window of 49,152 (`-c 49152 --chunks 1`, scored on positions 24,576–49,151) |
| The preset | [release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env): its `PRESET_ENV` (the Q8_0 cache's two attention switches `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` and `GGML_CUDA_FATTN_DEC_PASCAL=4` among them) and `-ctk q8_0 -ctv q8_0 -b 4096 -ub 512`; context and split are this check's (`-ngl 99 -fa on -fit off --no-host -ts 1,1,1`). `llama-perplexity`'s scored passes are 512 tokens, so the decode switches reach only the load's unscored warm-up call (its log names the decode kernel) |
| As a P106 runs it | `$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")` from [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh): the fork's compute 6.1 paths at `2` (any card) with a P106-090's launch shapes, `LLAMA_NO_MOE_DOWN_RESCALE=tg`, and cuBLAS computed as on a P106, in F32 without TF32 (`NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, C1108). Every arm logs at `-lv 4` (log level only) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before each pass |
| Time | 2026-11-26 07:44–08:23, 11 runs (one model load each), every one exit 0 ([runs.tsv](runs.tsv)) |

### The arms

Every arm is scored against the preset's own run (`q8`) on the same tokens; each is one load of `llama-perplexity`
with its whole variable list.

| Pass | Arm | What changes from `q8` | What it is |
|---|---|---|---|
| s, w | `q8` | | the preset as a P106 runs it; writes the reference |
| s, w | `kdp` | `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` | the arm under test |
| s, w | `q8g` | `GGML_CUDA_FATTN_TILE_PASCAL=0` | the attention-only yardstick of [C1032](../2026-11-25_c1032_attn_yardstick/README.md) (point 26): the build's generic tile kernel takes the prompt passes, on F16 copies of K/V |
| s, w | `q8u` | `-ub 128` | the rounding yardstick of [C994](../2026-11-24_c994_attn_fault_selfref/README.md) and C976: every sum in the model in another order |
| s | `q8f1` | `GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1` | the prompt kernel's output × 1.05: the error the check must see |
| s2 | `q8t` | cuBLAS as ggml sets it on a newer card (no `NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`) | what C1108's setting changes: the yardsticks above were first measured this way |
| s2 | `q8r` | | `q8` loaded again: the floor |

Pass s and s2 are the 12 stretches of 8,192 (s2 reads pass s's reference), pass w the window of 49,152.

**TF32 and the yardsticks.** C994's and C1032's yardsticks were measured before C1108, with cuBLAS allowed to round
F32 products to TF32 and to compute F16 and quantized sources in F16 on these cards. Both arms of this check run the
same products, so TF32 adds no difference between `q8` and `kdp` by itself. A yardstick, though, is the size of the
change that summing in another order makes, and that size depends on the arithmetic of everything after the change:
`q8u` reorders cuBLAS's own sums, and an attention change grows through the layers after it. So the yardsticks are
measured again here in the same settings as the check, and `q8t` measures how far C1108's setting moves the output.

## The check and its rule (fixed before the run)

For each stretch i, an arm's KLD against `q8`, its top-token disagreement (100 − same top %) and its ln PPL ratio, read
from `llama-perplexity`'s running means as in C976, C994 and C1032 (k·M_k − (k−1)·M_{k−1}, up to about 0.00006 of
printing error a stretch): K_i for `kdp`, G_i for `q8g`, U_i for `q8u`, F_i for `q8f1`. Mean over the 12 stretches,
standard error SE = sd/√12, t = mean/SE.

- **(1) KLD within each yardstick:** `kdp` passes against a yardstick Y when the mean of K_i − Y_i is not 3 SE above
  zero (t < 3): its change cannot be told from what summing in another order does. Both yardsticks, `q8g` and `q8u`,
  must hold. Top-token disagreement is reported by the same rule; it does not decide.
- **(2) Perplexity within ±1 %:** the mean of `kdp`'s per-stretch ln PPL ratio within ±0.01, and the window's ln PPL
  ratio within ±0.01.
- **(3) The window of 49,152** (one stretch, so no honest paired error; C976's band B): `kdp`'s mean KLD at most the
  larger of twice each yardstick's and 0.003, its top-token disagreement at most the larger of twice each yardstick's
  and 0.3 points.
- **Validity:** the 5 % error is seen above the attention-only yardstick (mean F_i − G_i at least 3 SE above zero), so
  the check can see an error of that size in these settings; `q8r` repeats `q8` (KLD 0.00000 and same top 100.000 % on
  every stretch); every arm of a pass reads the same reference perplexity; the logs: `kdp`'s names the line
  "q8_0 K scored in int8 with dp4a", no other arm's does; every arm except `q8g` names the compute 6.1 prompt kernel at
  head 256 with Q8_0 K/V, `q8g` none; `q8f1`'s names its factor 1.05.
- **Sizes, est.:** each arm's KLD as a uniform attention output error, 0.05 × √(mean KLD / mean F); and the smallest
  uniform error seen against each yardstick (F_i scaled by (ε/0.05)², the smallest ε with t ≥ 3), as in C1032.

`_KDP=1` keeps Ornith's quality if (1), (2) and (3) hold and the run is valid.

**How each came out** ([verdict.tsv](verdict.tsv)): (1) passed against both yardsticks; (2) passed over the
stretches and at the window; (3) passed in KLD and top-token disagreement against both; the 5 % error was seen; every
arm of a pass read the same reference; every log named the kernels the rule asked for (`kdp`'s the int8 line, no other
arm's; `q8g`'s no prompt kernel line; `q8f1`'s the factor 1.04999995, the float nearest 1.05). The repeat failed its
top-token part by one token, as above.

## The result

### Each arm over the 12 stretches

Every arm against the preset's own run (`q8`), 49,140 scored tokens ([kld.tsv](kld.tsv)). The ± in this table treats
neighbouring tokens as independent; the paired table below has the errors that hold.

| Arm | Mean KLD | Same top token | ln PPL ratio | 99th percentile KLD | Median KLD |
|---|---|---|---|---|---|
| `kdp`: the switch | **0.005951 ± 0.000202** | **98.382 ± 0.057 %** | **+0.000599 ± 0.000600** | 0.079529 | 0.000149 |
| `q8g`: the attention-only yardstick | 0.006160 ± 0.000265 | 98.462 ± 0.056 % | −0.000469 ± 0.000646 | 0.080081 | 0.000145 |
| `q8u`: the reordering yardstick (`-ub 128`) | 0.005794 ± 0.000203 | 98.466 ± 0.055 % | −0.000941 ± 0.000593 | 0.076426 | 0.000147 |
| `q8f1`: 5 % error | 0.011238 ± 0.000406 | 97.855 ± 0.065 % | +0.000768 ± 0.000830 | 0.139197 | 0.000266 |
| `q8t`: TF32 on, cuBLAS in F16 | 0.007139 ± 0.000208 | 98.215 ± 0.060 % | +0.000597 ± 0.000676 | 0.093083 | 0.000173 |
| `q8r`: the repeat | 0.000000 | 99.998 ± 0.002 % | +0.000040 ± 0.000030 | 0.000032 | 0.000000 |

The reference's perplexity over the 12 stretches is 1.579161 (C994's, with TF32 on and `3dfad876f`: 1.577859).

### The forms, per stretch

Mean over the 12 stretches ± the standard error across stretches, and t = mean / SE ([paired.tsv](paired.tsv)):

| Pair | KLD | t | Top token differs (points) | t | ln PPL ratio | t |
|---|---|---:|---|---:|---|---:|
| `kdp − q8g` | **−0.000210 ± 0.000239** | −0.88 | +0.080 ± 0.046 | 1.73 | +0.00107 ± 0.00060 | 1.78 |
| `kdp − q8u` | **+0.000160 ± 0.000238** | 0.67 | +0.084 ± 0.070 | 1.20 | +0.00154 ± 0.00055 | 2.79 |
| `(kdp − q8g) / q8g` | −0.002 ± 0.031 | −0.07 | +0.088 ± 0.054 | 1.63 | | |
| `(kdp − q8u) / q8u` | +0.009 ± 0.034 | 0.26 | +0.121 ± 0.056 | 2.18 | | |
| `q8f1 − q8g` (validity) | **+0.005080 ± 0.001023** | **4.96** | +0.607 ± 0.098 | 6.18 | +0.00124 ± 0.00114 | 1.09 |
| `(q8f1 − q8g) / q8g` | +0.962 ± 0.125 | 7.72 | +0.448 ± 0.081 | 5.55 | | |
| `q8f1 − q8u` | +0.005450 ± 0.001150 | 4.74 | +0.611 ± 0.054 | 11.33 | +0.00171 ± 0.00110 | 1.56 |
| `kdp − q8f1` | −0.005290 ± 0.001224 | −4.32 | −0.527 ± 0.095 | −5.54 | −0.00017 ± 0.00141 | −0.12 |
| `q8t − q8g` | +0.000980 ± 0.000204 | 4.81 | +0.247 ± 0.077 | 3.23 | +0.00107 ± 0.00114 | 0.94 |
| `q8t − q8u` | +0.001350 ± 0.000284 | 4.75 | +0.251 ± 0.043 | 5.86 | +0.00154 ± 0.00086 | 1.79 |

Bold: the rule's checks. `kdp`'s ln PPL ratio alone is +0.00060 ± 0.00061 (t 0.98); the reordering yardstick's alone
is −0.00094 ± 0.00041 (t −2.29), which is most of the `kdp − q8u` perplexity difference.

**Sizes, est.** ([verdict.tsv](verdict.tsv)): an arm's KLD as a uniform attention output error,
0.05 × √(mean KLD / mean `q8f1`): `kdp` 3.64 %, `q8g` 3.70 %, `q8u` 3.59 %, `q8t` 3.99 %. The smallest uniform error
the check sees, from `q8f1` scaled by (ε/0.05)²: against `q8g` 4.05 % (KLD), 3.97 % (KLD divided per stretch), 4.11 %
(top-token disagreement); against `q8u` 3.91 %, 3.80 % and 3.94 %. Both are estimates from one size of error.

### Stretch by stretch

Each stretch's own values ([chunks.tsv](chunks.tsv)); top token differs in %:

| Stretch | `kdp` KLD | `q8g` KLD | `q8u` KLD | `kdp − q8g` | `kdp − q8u` | `kdp` top differs | `q8g` top differs | `q8u` top differs | `kdp` ln PPL ratio | `q8f1` KLD | `q8t` KLD | `q8r` KLD / top differs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01207 | 0.01169 | 0.00965 | +0.00038 | +0.00242 | 2.125 | 2.295 | 1.978 | +0.00361 | 0.01545 | 0.01175 | 0.00000 / 0.000 |
| 2 | 0.00329 | 0.00311 | 0.00303 | +0.00018 | +0.00026 | 1.025 | 0.855 | 0.854 | −0.00111 | 0.00559 | 0.00385 | 0.00000 / 0.000 |
| 3 | 0.00729 | 0.00680 | 0.00673 | +0.00049 | +0.00056 | 2.052 | 1.929 | 2.151 | −0.00217 | 0.01223 | 0.00900 | 0.00000 / 0.000 |
| 4 | 0.01559 | 0.01780 | 0.01631 | −0.00221 | −0.00072 | 2.782 | 2.761 | 3.173 | +0.00439 | 0.03217 | 0.01888 | 0.00000 / 0.024 |
| 5 | 0.00451 | 0.00515 | 0.00453 | −0.00064 | −0.00002 | 1.856 | 1.585 | 1.244 | +0.00293 | 0.00786 | 0.00532 | 0.00000 / 0.001 |
| 6 | 0.00405 | 0.00423 | 0.00385 | −0.00018 | +0.00020 | 1.488 | 1.465 | 1.490 | −0.00051 | 0.00698 | 0.00502 | 0.00000 / −0.001 |
| 7 | 0.00605 | 0.00540 | 0.00539 | +0.00065 | +0.00066 | 2.105 | 2.200 | 2.200 | +0.00112 | 0.01002 | 0.00736 | 0.00000 / −0.003 |
| 8 | 0.00251 | 0.00230 | 0.00299 | +0.00021 | −0.00048 | 0.927 | 0.582 | 0.638 | +0.00094 | 0.00546 | 0.00250 | 0.00000 / 0.003 |
| 9 | 0.00386 | 0.00445 | 0.00413 | −0.00059 | −0.00027 | 1.021 | 0.980 | 0.924 | +0.00043 | 0.00909 | 0.00571 | 0.00000 / 0.003 |
| 10 | 0.00408 | 0.00387 | 0.00409 | +0.00021 | −0.00001 | 1.299 | 1.198 | 1.148 | −0.00193 | 0.00835 | 0.00481 | 0.00000 / −0.007 |
| 11 | 0.00160 | 0.00142 | 0.00189 | +0.00018 | −0.00029 | 0.711 | 0.826 | 0.590 | −0.00077 | 0.00417 | 0.00203 | 0.00000 / 0.002 |
| 12 | 0.00650 | 0.00770 | 0.00689 | −0.00120 | −0.00039 | 2.025 | 1.780 | 2.018 | +0.00027 | 0.01751 | 0.00945 | 0.00000 / 0.002 |

`kdp` is above the attention-only yardstick on 7 stretches and below on 5. The repeat's small negative values are the
five- and three-decimal printing of `llama-perplexity`'s running means; stretch 4's 0.024 % is one token of 4,095.

### The window of 49,152

One stretch, scored on positions 24,576–49,151 (24,575 tokens), each arm against `q8` ([window.tsv](window.tsv)):

| Arm | Mean KLD | Top token differs | ln PPL ratio | Perplexity (reference 1.431276) |
|---|---|---|---|---|
| `kdp` | **0.011747 ± 0.000924** | **1.656 %** | **+0.002030 ± 0.001333** | 1.434184 |
| `q8g` | 0.010911 ± 0.000839 | 1.632 % | +0.002215 ± 0.001387 | 1.434450 |
| `q8u` | 0.010408 ± 0.000780 | 1.583 % | +0.000600 ± 0.001295 | 1.432135 |

The band: KLD at most 0.021822 and 0.020816 (twice each yardstick), top-token disagreement at most 3.264 and
3.166 points, ln PPL ratio within ±0.01. The ± here treats neighbouring tokens as independent, so the window says only
that `kdp` is the yardsticks' size at this depth, not finer.

### What the tables say

- **The int8 query costs no measurable quality on Ornith.** Its change to the output is the size of a yardstick's,
  stretch by stretch and at 49,152 tokens, and follows the same stretches. This is the rounding llama.cpp's CPU
  reference and its vector kernel apply to a Q8_0 cache, and the preset's decode kernel already applies it to every
  generated token.
- **The yardsticks depend on the cards' arithmetic.** C994's reordering yardstick was 1.37 times today's. It was
  measured with TF32 on (and on an older engine), where reordering the model's sums also reorders products rounded to
  TF32's 10-bit inputs and F16 sums inside cuBLAS, which a P106 never computes; TF32 alone moves the output further
  than either yardstick here. Measured as a P106 computes, the two yardsticks are about the same size (0.005794 and
  0.006160), and either can be the one a later attention check is read against, measured in the same run.
- **The check's reach is unchanged:** with both yardsticks re-measured it sees uniform attention errors from est.
  3.8 % to 4.1 %, as C1032 did with TF32 on (4.0 % and 4.4 %).

**Limits.** These cards run the fork's attention and matrix-vector kernels through the compute 6.1 PTX, forced on with a
P106-090's launch shapes, and cuBLAS as on a P106; one choice is still not a P106's: a dense quantized product of 64 or
more tokens goes to cuBLAS on F16 copies of the weights here and to MMQ on a P106
([Q1901](../../../quants/experiments.md)). Both arms share it. One load per arm and depth. The sizes are estimates
from one fault size (5 %, a uniform scale). The window is one stretch. The repeat is not bit for bit (above).

## Files

| File | What |
|---|---|
| [verdict.tsv](verdict.tsv) | every check of the rule: the value, the rule and the result; the sizes and the smallest error seen, est.; the TF32 comparisons; the validity checks |
| [paired.tsv](paired.tsv) | every form: the mean per stretch in KLD, top-token disagreement and ln PPL ratio, its standard error across stretches and t |
| [kld.tsv](kld.tsv) | per pass and arm: scored tokens, mean KLD ± SE, same top token ± SE, the KLD's maximum, 99.9th, 99th and 90th percentiles and median, RMS of the probability change, both perplexities and their mean log-ratio ± SE |
| [chunks.tsv](chunks.tsv) | each stretch's own ln PPL ratio, KLD and same top token, per arm of pass s and s2 |
| [window.tsv](window.tsv) | the window of 49,152: each arm's KLD, top-token disagreement, ln PPL ratio and perplexities |
| [runs.tsv](runs.tsv) | every run: pass, arm, depth, `-ub`, exit status, start and end |
| [run/run.sh](run/run.sh) | the passes in order, with each arm's settings built from the preset file |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c1103_table.py](run/c1103_table.py) | the tables and the rule from the logs |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~40 GB free: the references are
~24 GB for the 12 stretches and ~12 GB for the window; the bin folder unpacked from the rig tarball), from this folder,
with [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh) and the preset file copied beside `run.sh` when
`run/` is copied elsewhere:

```bash
export WORK=$W BIN=<2d839da1d tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  PRESET_FILE=<release/ornith-v5/settings.env> \
  MOUNTS="<the folders of the build, model and text>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
PASSES= bash run/run.sh                                                  # prints each arm's settings, runs nothing
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null         # passes s, w, s2; $W/DONE at the end
# once $W/DONE exists:
python3 -I run/c1103_table.py $W .
```

The run took 39 minutes, 2026-11-26 07:44–08:23: the first load 9 minutes (the model from a hard disk, the JIT
compile), later ones 2 to 6 minutes ([runs.tsv](runs.tsv)). The references were removed after the tables were written.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
