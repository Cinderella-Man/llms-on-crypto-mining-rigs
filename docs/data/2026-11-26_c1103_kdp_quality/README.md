# Q1836: does `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` keep Ornith-1.5-122B's quality? (2026-11-26)

RESULT-PLACEHOLDER

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
| The preset | [release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env): its `PRESET_ENV` (the Q8_0 cache's two attention switches `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` and `GGML_CUDA_FATTN_DEC_PASCAL=4` among them) and `-ctk q8_0 -ctv q8_0 -b 4096 -ub 512`; context and split are this check's (`-ngl 99 -fa on -fit off --no-host -ts 1,1,1`). `llama-perplexity`'s passes are 512 tokens, so the decode switches are set but never reached |
| As a P106 runs it | `$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")` from [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh): the fork's compute 6.1 paths at `2` (any card) with a P106-090's launch shapes, `LLAMA_NO_MOE_DOWN_RESCALE=tg`, and cuBLAS computed as on a P106, in F32 without TF32 (`NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, C1108). Every arm logs at `-lv 4` (log level only) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before each pass |
| Time | TIME-PLACEHOLDER |

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

RESULTS-PLACEHOLDER
