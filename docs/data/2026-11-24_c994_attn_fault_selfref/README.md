# Q1727: attention faults on Ornith-1.5-122B scored against their own unfaulted run, the form a new-kernel check takes (2026-11-24)

[C989](../2026-11-24_c989_attn_error_floor/README.md) calibrated the per-stretch quality check on Ornith-1.5-122B
Pascal N2P GGUF (`valid1`) with deliberate errors in the Pascal prompt attention kernel, scored against references that
have errors of their own: the F16 cache, and Ornith's own Q8_0 model. This folder scores the same faults against the
unfaulted run itself, the Q8_0 cache with the base kernel, so the reference adds no error of its own. That is the form a
check of a new attention kernel takes: the new kernel's output against the old kernel's, on the same cache, with the
old kernel in micro-batches of 128 as the yardstick for what reordering the sums alone does
([Q1727](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's and C989's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C976's and C989's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -ctk q8_0 -ctv q8_0`, `-ub 512` unless the arm says otherwise; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only), so each log names the kernel that took the prompt passes |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before the pass |

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
