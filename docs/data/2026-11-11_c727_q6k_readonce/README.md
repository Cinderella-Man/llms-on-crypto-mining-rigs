# Q6_K's products of 2–4 tokens, read once and exact (C727, Q1286, 2026-11-11)

What this is: a design note and the checks behind a new kernel path, not yet a measurement on a P106. It adds Q6_K to
C722's read-once kernel ([2026-11-11_c722_short_batch_products](../2026-11-11_c722_short_batch_products/README.md):
IQ4_XS's and Q4_K's products of 2–4 tokens with every weight read and decoded once, each token's sums the one-token
kernel's). Q6_K is the type of some of Qwen3.8-Flash-Next's attention output and `ssm_out` projections, whose 2–4-token
products the k-column kernel takes today, with sums other than one-token decoding's. The rig run that checks and times
it natively is the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c727` (on the v4 day merge `pascal-p106-v4-b12246-day`,
`18d0fa309`), behind its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_Q6K_MT`
([docs/switches.md](../../switches.md)). Registry row [Q1286](../../../quants/experiments.md).

## The products

- **Flash-Next's Q6_K products in a verify pass:** `attn_output` and `ssm_out`, 2,560 rows of 6,144 weights, take 1.06
  and 0.53 ms of a four-token pass's GPU time summed over the nine cards (`attn_output` 0.40 at one token;
  [2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). Under the Flash-Next preset
  (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, `_KCOL_TABLE=0`) the k-column kernel takes them at 2–8 tokens, and the
  one-token Pascal kernel (`mul_mat_vec_q_pascal`) at one token, so a token's sums in a verify are not the ones it gets
  in plain decoding.
- **Not taken here:** the shared expert's two Q6_K layers (2,560 → 640) and the MTP draft's Q6_K output layer
  (2,560 → 248,320) have 2,560-wide rows: two lanes per row, whose 420-byte segments are not whole 16-byte units, and
  rows that start 4 bytes off a 16-byte boundary. They stay on the k-column kernel (a new row, C782/Q1371).

## The design

`mul_mat_vec_q_pascal_dmt<GGML_TYPE_Q6_K, NT, R, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`: C722's kernel (its
lanes, order, persistent grid, activation staging, register prefetch) with Q6_K's super-block step. No new hook: the
two tests that send IQ4_XS's and Q4_K's products there (`mmvq-pascal-kcol.cu`, `mmvq.cu`) ask the same function, which
now takes Q6_K when `GGML_CUDA_MMVQ_PASCAL_Q6K_MT` is set.

- **The one-token kernel's order:** L lanes per row (the largest power of two, at most 32, that divides the row's
  super-blocks: 8 at K = 6,144), lane `sl` takes super-blocks `sl`, `sl + L`, … in that order, the lanes' sums added
  with the same xor shuffles. Per super-block, `mmvqp_dot_q6_K`'s step, for sub-block s (32 weights) and token:
  `sumf = fmaf(d8, (float) (sc0 · (isum0 − t0) + sc1 · (isum1 − t1)), sumf)`, then `acc = fmaf(d, sumf, acc)`; isum0
  and isum1 are the dp4a sums of the 6-bit codes (0–63) and the int8 activations of the sub-block's two 16-weight
  halves, t0 and t1 32 times the halves' int8 sums (`mmvqp_load_act`), sc0 and sc1 the halves' int8 scales.
- **Each 32-weight sub-block of a row decoded once** into eight words of codes (the nibbles of `ql` OR-ed with a 2-bit
  field of `qh`, as the one-token kernel decodes them) and multiplied with dp4a by every token's activations.
- **The integer expression made exact in FP32, without I2F or integer multiplies:** each half's dp4a chain starts at
  0x4B400000 − t (staged per token beside d8), so its result read as a float is 12,582,912 + (isum − t), with
  |isum − t| ≤ 16 · 32 · 128 = 2¹⁶ < 2²². Then `fmaf(12582912 + (isum − t), sc, −12582912 · sc)` is sc · (isum − t)
  rounded once, which is exact (|·| ≤ 2²³), and the two halves' sum is the integer expression exactly (|·| ≤ 2²⁴). The
  int8 scales come out of a `__byte_perm` of (scale XOR 0x80) into 12,582,912's low mantissa bits, less 12,583,040:
  (float) scale exactly. So every FP32 operation of the one-token kernel (the FFMA with d8, the FFMA with d, the
  shuffles' FADD) sees the same operands, and every token's result is the one-token kernel's bit for bit.
- **The activations once per block:** every token's int8 words (natural order), d8 and the two dp4a starts are staged
  once into shared memory per super-block position; the positions an odd number of bank groups apart, so that the L
  lanes of a group read distinct banks (two 128-bit reads of the words, one 32-bit of d8, one 64-bit of the starts a
  sub-block and token); every lane group of a warp reads the same L records, a broadcast.
- **The weights, 2-byte aligned:** Q6_K's super-blocks are 210 bytes. A row segment (L super-blocks) is staged in
  16-byte units into the warp's buffer, so rows must be 16-byte aligned and L a multiple of 8 (rows of a multiple of 8
  super-blocks: K = 2,048, 4,096, 6,144, …). Each lane reads its super-block from the staging with 32-bit loads of the
  aligned words around it, shifted by 2 bytes with a `__byte_perm` where it starts at an odd half-word, as the one-token
  kernel reads its staging (`mmvqp_lds_h`); one half of the super-block at a time (its `ql`, `qh` and scale words), to
  keep the registers down. The lane groups' segments are padded (a host search, `dmt_gstride`, now also for 32-bit
  reads) so that the 32 lanes' reads hit distinct banks: at L = 8 a 16-byte pad does it; at L = 16 and 32 no pad does,
  and the reads meet 2 words a bank.
- **One row per lane group by default** (`_DMT_R=2` shares the activation loads between two rows), the register
  prefetch of each warp's next segments at R = 1 (`GGML_CUDA_MMVQ_PASCAL_Q6K_MT=2` runs without it; at R = 2 the two
  rows' prefetched segments took 255 registers and 24 bytes of stack at 4 tokens, so R = 2 never prefetches), and a product whose
  activations leave room for one warp's staging only is left to the k-column kernel (C722's kernel lost 46–47 % to it
  at one warp a block, [R79](../2026-11-11_r79_c722_p106/README.md)), unless `_DMT_NW=1`.
- **`_Q6K_MT_FAULT=1`** (checks only): every token stages the first token's activations (a column stride of 0, set on
  the host); the checks must fail.
- **IQ4_XS and Q4_K unchanged:** the type branches are compile-time, the kernel's arguments and staging code are
  C722's, and the host rules for those types are as they were; the SASS check below compares their code.

The shapes at Flash-Next's width (the host code's rules; R = 1 unless noted):

| K (L) | Tokens | R | Activations | Weight staging a warp | Warps a block | Shared memory | Blocks a SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6,144 (8) | 2 | 1 | 17,472 B | 6,784 B | 4 | 44,624 B | 2 |
| 6,144 (8) | 3 | 1 | 25,920 B | 6,784 B | 3 | 46,288 B | 2 |
| 6,144 (8) | 4 | 1 | 34,368 B | 6,784 B | 2 | 47,952 B | 2 |
| 6,144 (8) | 2 | 2 | 17,472 B | 13,440 B | 2 | 44,368 B | 2 |
| 6,144 (8) | 3–4 | 2 | 25,920–34,368 B | 13,440 B | 1: left to the k-column kernel | | |

At four tokens a SM holds 4 warps (2 blocks of 2), the K = 6,144 case of Q1287 (C728), which tiles K for this kernel
family.

{{SASS_COUNTS}}

## What was checked without a GPU

{{CHECKS}}

## The rig run (R117)

`run/rig_c727.sh` on one P106 of an idle rig, detached (C726's `rig_c726.sh` with Q6_K's arms). Its steps:

- **replay:** the recorded set `q6kmt` through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`, the
  fault run's with its 2–4-token cases failing.
- **bitid:** `multitoken_bitid q6k` at `Q6K_MT=1`, `=2`, `_DMT_R=2`, `_DMT_NW=1`: `0 with differences` each. Then the
  controls, which must report differences: the switch off (the k-column kernel at 2–4 tokens) and `_FAULT=1`.
- **perf:** `test-backend-ops perf` at Flash-Next's Q6_K shape (2,560 × 6,144), 1–5 tokens, arms `off` (the preset),
  `on`, `nopf`, `r2`, `nw1`, `nw1nopf`, two passes alternated; `perf/us_run.tsv`. The 1-token cases run the one-token
  kernel and the 5-token ones the k-column kernel in every arm, so they give the spread.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft, the switch off and on in the order
  off, on, on, off; `on` takes the perf step's fastest arm at 4 tokens. The tables: R49's `verify_parts.py` on each
  profiled load, then R79's `run/served_table.py --types q6_K`
  ([2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md)).

Decides: exact or not (any difference is a fault to find first); gain, negligible or loss at 2–4 tokens against `off`,
each pass by value; the Q6_K products' time per verify pass.

## Files

{{FILES}}

## Commands

{{COMMANDS}}

## Provenance and licence

Everything here was produced by the scripts in `run/` from the fork's code (MIT, as llama.cpp) and random weights and
activations made by `test-backend-ops`, `multitoken_bitid` and `run/q6k_emul.c` from fixed seeds; no model file was
read. The tables are this repository's (MIT).
