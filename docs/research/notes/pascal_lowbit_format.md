# A 2–3-bit format for Pascal: what Ornith-1.5-122B needs to fit nine P106 cards and decode fast (2026-10-27)

Ornith-1.5-122B (`qwen35moe`: 48 layers, 36 Gated DeltaNet and 12 attention, hidden size 3072, 256 routed experts of
width 1024 with 8 used per token, a shared expert, ~10 B active parameters) fits the rig's nine 6 GB cards only at
about 2.7–2.9 bits per routed-expert weight. The files that fit today store the experts in IQ2_S and IQ3_XXS, and on
Miner Llama v2.0 those experts take 43 % of the decode time while reading weights at 19–30 GB/s, a sixth of the card's
bandwidth ([Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)). This note asks whether a new, simple 2- or
3-bit block format, built for dp4a on compute 6.1, would decode faster than those formats at the same size and
quality, and what it would take to build one.

Conventions, as in the other notes here:
- **(measured)**: measured on the rig or on the quant machine and stated with its source.
- **(my calc)**: arithmetic in this note. **(est.)**: an estimate from instruction counts, to be measured.
- Costs are counted in slot-equivalents as in [Pascal forks and quants](pascal_forks_and_quants.md), section 5: one
  issue slot for a logic op, add, dp4a, byte permute or shared-memory load, two for a shift or an integer multiply, four
  for an int↔float conversion; staging costs three per 16 bytes, and the activation work is shared by two rows.
- Speed is given as time per weight (ps), because the number of weights a token reads is fixed by the model; GB/s
  follows from it and the bits per weight.

## Summary

1. **Today's 2–3-bit experts are slow because of how llama.cpp's generic kernel reads codebooks, not because of the
   bit width (measured, my calc).** IQ2_S gate/up run at 10.8 ps per weight and IQ3_XXS down-projections at
   20.5 ps, against 4.2 ps for Q4_0 in the fork's Pascal kernel. Both IQ types do their grid lookups at the same rate,
   ~1.4 per SM per clock (one warp-wide gather from a table in global memory every ~23 clocks), although IQ2_S
   decodes twice as many weights per lookup: the gathers set the speed.
2. **Below 4 bits the P106 is limited by instructions per weight, not by bytes (my calc).** The fork's measured
   kernels fit `time per weight = max(bytes per weight / 150 GB/s, slot-equivalents per weight / 0.56 T per s)`. At
   2.5–3.25 bits per weight the bandwidth term is 2.1–2.7 ps, so every candidate is issue-bound; a 2–3-bit format can
   at best match Q4_0's time per weight, and by the byte-based definition in the glossary no 2–3-bit format can be a
   fast format on this card. What matters is ps per weight.
3. **Candidate formats** (section 3), all 256-weight super-blocks with an FP16 scale and dp4a against Q8_1 activations:
   - **N2+**, 2.6875 bits: four non-uniform levels (ik_llama.cpp's IQ2_K table) held in one register and decoded with
     one byte permute per 4 weights; per 8 weights a 4-bit scale and a bit that picks the table's shifted copy.
     Est. 3.7 ps per weight.
   - **N3+**, 3.25 bits: eight levels (IQ3_K's table) and the shifted copy in four registers; codes in a 2-bit and a
     1-bit plane assembled into byte-permute selectors; per 32 weights a 5-bit scale and the shift bit; 104-byte,
     8-byte-aligned blocks. Est. 4.0 ps per weight.
   - Without the shift bit (N2, N3) they cost about the same, and so do uniform-level variants (U2, U3).
   - Q2_K and Q3_K are already uniform-level formats: Pascal kernels in the Q4_K style would put them at est. 3.4 and
     4.9 ps per weight with no new type.
   - The IQ2_S/IQ3_XXS Pascal kernels with tables in shared memory (fork branch `pascal-p106-v3-iqmmvq`) measure
     9.9–12.9 ps per weight on the rig (28–36 GB/s), 1.2–1.7× llama.cpp's kernels: their shared-memory gathers stall on
     bank conflicts, twice what the slot count alone predicted (5–7 ps).
4. **Quality proxy (measured, section 4):** importance-weighted reconstruction error of 84 real expert matrices of
   Ornith-1.5-122B, CPU only.
   - Register-table levels match mainline's codebooks at equal size: N2 against IQ2_S at 2.5625 bits, N3-16 against
     IQ3_S at 3.4375.
   - ik_llama.cpp's shift bit puts the 3-bit formats ~9–12 % ahead per bit.
   - At ~2.875 bits, the expert mix N2+ (gate/up) / N3+ (down) has 6 % less error than the best IQ mix of the same size
     and 18 % less than AD-IQ3_XXS-IQ2_S's mix at 2.729 bits.
   - Uniform levels and Q2_K/Q3_K lose 11–24 %; Hadamard rotation loses 9 %; mainline's Q2_0 is unusable for
     post-training quantization.
5. **Recommendation (section 5):**
   - First, without new code: measure the IQ Pascal kernels on the rig (done since: 9.9–12.9 ps per weight, section 3),
     and move Ornith's dense tensors off Q8_0. Est. −10 to −12 ms of a 118-ms token.
   - Then two cheap gates before building anything: N2+/N3+ prototype kernels in `bench/mmvq_pascal` against the IQ
     kernels on one P106, and a simulated-quantization KLD test on Ornith-1.5-35B.
   - Build N2+ and N3+ (~7–8 days) only if they decode ≥ 1.3× faster per weight than the IQ kernels and their KLD is
     at least as good. Expected then: ~12.4 tok/s with Q8_0 dense tensors and ~14.5 with Q5_0 ones, against today's
     8.5, 9.4 for the IQ kernels (their measured speed) and ~10.4 for the IQ kernels with Q5_0 dense tensors.

## 1. Why today's 2–3-bit formats are slow on the P106

### What the rig measured

Miner Llama v2.0, Ornith-1.5-122B AD-IQ3_XXS-IQ2_S (routed experts: gate and up IQ2_S, down IQ3_XXS; everything else
Q8_0), one token per step, GPU time per token summed over the nine cards (profile from the
[Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)):

| Part | ms per token | Weights per token | ps per weight | GB/s |
|---|---|---|---|---|
| Routed gate + up, IQ2_S (fused launch) | 26.0 | 2.42 G | 10.8 | 29.7 |
| Routed down, IQ3_XXS | 24.8 | 1.21 G | 20.5 | 18.7 |
| Dense Q8_0 (attention, DeltaNet projections, shared expert, output head) | 38.7 | 5.35 G | 7.2 | 147 |
| Router (F32) | 2.1 | 0.04 G | | |
| Operations without weights (attention, DeltaNet, norms, copies) | 26.1 | | | |
| **Total** | **117.7** (8.5 tok/s) | | | |

The fork's Pascal matrix-vector kernels cover Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS, so the IQ2_S and
IQ3_XXS experts run llama.cpp's generic `mul_mat_vec_q`. Their per-weight times are 2.6–4.9 times Q4_0's 4.2 ps in the
fork's kernel ([mmvq_pascal benchmark](../../../bench/mmvq_pascal/README.md)), and 1.5–2.8 times Q8_0's.

### What the generic kernel does per 32 weights

`mul_mat_vec_q` gives each thread one 32-weight sub-block of a 256-weight super-block per iteration (`vdr` = 2,
`qi` = 16), so eight threads share a super-block; at one token a block has 2 warps and 1 row for the 3072-wide gate/up
rows, and 4 warps with one row each for the 1024-wide down rows. Per thread and 32 weights:

- **IQ2_S** (82-byte super-blocks: FP16 scale, 32 index bytes, 32 sign bytes, 8 bytes of index high bits, 4-bit scales
  per 16 weights):
  - four 16-bit loads for the index and sign words (the super-block is only 2-byte aligned), one byte each of index
    high bits and scales;
  - four lookups of 8 bytes in `iq2s_grid`, a 1024-entry table (8 KB) in global memory, at an index assembled from
    the index byte and two high bits;
  - per 8 weights two sign masks, built from the sign byte with shifts and ORs and turned into byte masks with
    `__vcmpne4`, then `__vsub4(grid ^ mask, mask)`: compute 6.1 has no SIMD-video hardware (removed with Maxwell), so
    each of these intrinsics is a sequence of several integer instructions;
  - 8 dp4a, and the two 4-bit scales applied in integer arithmetic.
- **IQ3_XXS** (98-byte super-blocks: FP16 scale, 64 index bytes, then per 32 weights a word of four 7-bit sign patterns
  and a 4-bit scale): eight lookups of 4 bytes in `iq3xxs_grid` (256 entries, 1 KB), the eighth sign of every group
  recovered as a parity with `__popc`, the same `__vcmpne4`/`__vsub4` sign handling, 8 dp4a.
- **Both** read the activations per row and per thread (8 words and a scale per 32 weights), with no sharing across
  rows.

**The lookups set the speed (my calc).** A token's gate/up products decode 2.42 G weights in 26.0 ms, 8 weights per
grid lookup: 11.6 G lookups per second. The down-projections decode 1.21 G weights in 24.8 ms, 4 weights per lookup:
12.2 G lookups per second. Over 5 SMs at ~1.73 GHz that is 1.34 and 1.41 lookups per SM per clock, one warp-wide gather
every ~23 clocks, for two formats whose other work differs by 2× per weight. A warp's lookup is a gather of up to 32
different 32-byte sectors of the table, which the L1/texture path serves a few sectors at a time.

**Q2_K and Q3_K have no tables, but the generic kernel spends extra instructions on them.** Q2_K (84 bytes per 256
weights: a 4-bit scale and a 4-bit minimum per 16 weights, FP16 scale and minimum) runs a second dp4a with
`0x01010101` per 4 weights to get the activation sum its minimum needs. Q3_K (110 bytes: 2-bit plane, 1-bit plane,
6-bit scales per 16 weights packed across 12 bytes) shifts both planes per 4 weights, runs a second dp4a with
`0xFCFCFCFC` for its −4 offset, and unpacks its scales from three words per thread. Both super-blocks are only 2-byte
aligned, and both are re-derived per 32 weights per thread. These are exactly the costs the fork's Q4_K, Q5_K and Q6_K
kernels remove: one super-block per lane, staged 128-bit loads, codes masked in place, activation sums computed once and
shared by all rows ([K-quant decode](kquant_decode_pascal.md)).

### Why fewer bits do not mean less time on this card

The fork's measured kernels fit one model (my calc, from the slot-equivalent counts of
[Pascal forks and quants](pascal_forks_and_quants.md), section 5): Q5_0 at 4.1 slot-equivalents per byte runs 138 GB/s,
IQ4_XS at 5.3 runs 104 GB/s, both ~0.56 T slot-equivalents per second; Q4_1 (2.35) and Q8_0 (1.1) stay at the
~145–150 GB/s the memory gives a matrix-vector kernel. So

> time per weight = max(bytes per weight / 150 GB/s, slot-equivalents per weight / 0.56 T per second)

| Format | Bits per weight | Bandwidth term | Issue term | Time per weight | Source |
|---|---|---|---|---|---|
| Q8_0 | 8.5 | 7.1 ps | 2.1 ps | 7.1 ps | measured 150 GB/s |
| Q5_0 | 5.5 | 4.6 ps | 5.0 ps | 5.0 ps | measured 138 GB/s |
| Q4_1 | 5.0 | 4.2 ps | 2.6 ps | 4.2 ps | measured 145 GB/s |
| Q4_0 | 4.5 | 3.8 ps | ~2.6 ps | 4.2 ps | measured 134 GB/s |
| IQ4_XS | 4.25 | 3.5 ps | 5.0 ps | 5.0 ps | measured 104 GB/s |
| any 2.5-bit format | 2.5 | 2.1 ps | ≥ ~2.9 ps | issue-bound | |
| any 3.25-bit format | 3.25 | 2.7 ps | ≥ ~3.3 ps | issue-bound | |

A dp4a takes 4 weights, and each 2- or 3-bit field needs at least one more instruction to isolate it, so the issue
term cannot drop much below Q4_0's ~1.5 slot-equivalents per weight. Below about 3.5 bits the time per weight is set by
the decode work, and bits only buy memory. The design goal for a 2–3-bit format on the P106 is therefore **Q4_0's time
per weight at 2.5–3.25 bits**, which is 2.5–5 times faster than today's experts.

## 2. Prior art

Sources were read from GitHub and the papers on 2026-10-27; layouts come from each project's `ggml-common.h` or source.

| Format | Engine | Bits/weight | Levels | Scales | Decode on CUDA | Fit for sm_61 |
|---|---|---|---|---|---|---|
| Q2_K, Q3_K | llama.cpp ([#1684](https://github.com/ggml-org/llama.cpp/pull/1684)) | 2.625, 3.4375 | uniform (Q2_K with a minimum) | 4- or 6-bit per 16, FP16 per 256 | masks, dp4a, second dp4a for offsets | yes; kernel-limited today |
| IQ2_XXS / IQ2_XS / IQ2_S | llama.cpp ([#4773](https://github.com/ggml-org/llama.cpp/pull/4773), #4897, #5721) | 2.06 / 2.31 / 2.56 | 256/512/1024-entry 8-weight grids × signs | 4-bit per 32 or 16 | grid gather from global memory every 8 weights | gather-bound |
| IQ3_XXS / IQ3_S | llama.cpp ([#5196](https://github.com/ggml-org/llama.cpp/pull/5196), [#5676](https://github.com/ggml-org/llama.cpp/pull/5676)) | 3.06 / 3.44 | 256/512-entry 4-weight grids × signs | 4-bit per 32 | gather every 4 weights | gather-bound |
| TQ1_0 / TQ2_0 | llama.cpp ([#8151](https://github.com/ggml-org/llama.cpp/pull/8151); CUDA [#11183](https://github.com/ggml-org/llama.cpp/pull/11183) open) | 1.69 / 2.06 | ternary | FP16 per 256 | shift/mask, dp4a | ternary-trained models only |
| Q1_0 / Q2_0 | llama.cpp ([#21273](https://github.com/ggml-org/llama.cpp/pull/21273), [#24448](https://github.com/ggml-org/llama.cpp/pull/24448), CUDA [#25707](https://github.com/ggml-org/llama.cpp/pull/25707), [#25603](https://github.com/ggml-org/llama.cpp/pull/25603)) | 1.125 / 2.25 | ±d; {−1, 0, 1, 2}·d | FP16 per 128 / 64 (absmax) | `__byte_perm` into int8, dp4a | cheap, but made for PrismML's natively 1-bit and ternary Bonsai models; quantizer ignores the importance matrix |
| IQ2_NL / IQ3_NL | llama.cpp draft [#27322](https://github.com/ggml-org/llama.cpp/pull/27322) | 2.5 / 3.5 | non-uniform per 32 | FP16 per 32 | CUDA in sibling #27325 | stalled draft |
| IQ2_K / IQ2_KS | ik_llama.cpp ([#7](https://github.com/ikawrakow/ik_llama.cpp/pull/7), [#85](https://github.com/ikawrakow/ik_llama.cpp/pull/85)) | 2.375 / 2.1875 | 4 int8 levels {−31, −13, 1, 17} and a copy shifted +5, one bit per sub-block picks the copy | 4-bit signed per 16 / 5-bit per 32 + FP16 per row | 8-entry table in two registers, 4 byte permutes per 8 weights, dp4a | yes |
| IQ3_K / IQ3_KS | ik_llama.cpp (#7, [#566](https://github.com/ikawrakow/ik_llama.cpp/pull/566)) | 3.4375 / 3.1875 | 8 int8 levels {−63, −40, −23, −10, 1, 13, 28, 47} and a copy shifted +4 | 5-bit signed per 16 / per 32 + per row | 16-entry register table (`get_int_from_table_16`), dp4a | yes |
| IQ2_KL | ik_llama.cpp ([#602](https://github.com/ikawrakow/ik_llama.cpp/pull/602)) | 2.6875 | 32 pairs of IQ3_K levels | 6-bit per 32 + per row | pair gather | gather per 2 weights |
| IQ1_KT … IQ4_KT | ik_llama.cpp ([#529](https://github.com/ikawrakow/ik_llama.cpp/pull/529)) | 1.75–4.0 | integer trellis (a 32-bit multiply per weight) | 4-bit per 32 + per row | multiply, mask, two dp4a per 4 weights | issue-bound: compute 6.1 emulates the 32-bit multiply with 3 XMADs |
| EXL2 / EXL3 | exllamav2 / v3 | 2–8 / 1–8 | GPTQ groups / QTIP trellis | per group | FP16 dequantize / tensor-core MMA (sm_75+) | no (FP16 runs at 1/64 rate on GP106) |
| GPTQ, AWQ, HQQ 2–3-bit | AutoGPTQ/GPTQModel, AutoAWQ, hqq | ~2.1–3.2 | uniform, asymmetric | FP16 scale and zero per 64–128 | FP16 dequantize or Marlin/AWQ MMA (sm_75/80+) | only slow FP32/FP16 fallback kernels |
| NF4 | bitsandbytes | 4.5 | 16-level table | absmax per 64 | shared-memory table, float math | runs (cc ≥ 6.0), no 2–3-bit |
| q3f16_1 | MLC-LLM | 3.6 | uniform, 10 codes per 32-bit word | FP16 per 40 | FP16 | no |
| SqueezeLLM, any4/any3, FLUTE | — | 3–4 | per-row or per-channel tables | — | shared-memory FP32 tables; tensor cores (any4, FLUTE) | no |
| QuIP#, AQLM, QTIP | — | ~2 | E8 lattice, additive codebooks, trellis | — | large codebooks or FP16 trellis, Ampere+ | no |

What this means for a Pascal format:

- **Tables in memory are what slows Pascal down; tables in registers fix it.** ik_llama.cpp
  [#2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509) (merged 2026-09-23) replaced IQ4_K's per-weight byte load
  from a table in global memory with `__byte_perm` lookups in registers: Qwen3.5-4B decode went from 31.8 to 67.9 tok/s
  on a P40 (sm_61) and from 34.5 to 60.3 on a P100, with identical perplexity. Mainline
  [#15451](https://github.com/ggml-org/llama.cpp/pull/15451) did the same for MXFP4: P40 at microbatch 1 from 63.6 to
  82.0 tok/s. Mainline's IQ2/IQ3 grids (1–8 KB, 256–1024 entries) cannot fit in registers.
- **ik_llama.cpp's IQ2_K/IQ3_K family is the closest existing design**: a few non-uniform int8 levels in registers,
  small integer sub-scales, dp4a against Q8_1. Its one extra bit per sub-block, which picks a copy of the table shifted
  by a few int8 steps, costs no decode work beyond choosing a register. ikawrakow reports IQ2_K and IQ3_K as "basically
  the same" quality as IQ2_XS and IQ3_S ([discussion #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)), and
  on Llama-3.1-8B IQ3_K scores perplexity 7.91 against IQ3_XXS 8.46 and Q3_K 8.15 (Q8_0 7.33; #529). ik_llama.cpp runs
  on sm_61 ([docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md)); its types
  are not in mainline (the port, [#19726](https://github.com/ggml-org/llama.cpp/pull/19726), was closed unmerged).
- **No importance-matrix-aware 2- or 3-bit "_0" format exists in mainline.** Q2_0's quantizer maps |w| ≤ absmax to
  {−1, 0, 1}, so as post-training quantization it is ternary; section 4 measures it at 17× IQ3_XXS's error. The only
  open proposal, IQ2_NL/IQ3_NL (#27322), is a stalled draft, and a mainline maintainer is "very much against adding
  more data types": a new type will stay fork-only.
- **No published Pascal measurement exists for any 2–3-bit IQ type, ik type or trellis type.** The only 2-bit K-quant
  report is a 2024 user comment on [#5676](https://github.com/ggml-org/llama.cpp/pull/5676): a P40 ran a 13B model at
  26.8 tok/s in Q2_K_S against 19.7 in Q4_K_M, while another P40 owner in the same thread saw Q2_K halve the speed. The
  P40 has 30 SMs for 346 GB/s, about three times the P106's issue slots per byte of bandwidth.
- **Nothing outside the llama.cpp family runs 2–3-bit weights well on sm_61**: the other engines compute in FP16 or need
  tensor cores.

## 3. Candidate designs

### Layouts and sizes

Every candidate uses 256-weight super-blocks (rows of Ornith's experts are 3072 and 1024 wide) and one FP16 scale per
super-block; sub-blocks carry small integer scales. "Levels" are the values a code stands for, before scaling.

| Name | Bits/weight | Bytes per 256 | Alignment | Levels | Scales | Proxy name |
|---|---|---|---|---|---|---|
| **N2** | 2.5625 | 82 (64 codes + 16 scales + 2) | 2 B | 4 non-uniform, one int8 table | 4-bit signed per 8 weights | `nl2_s4x8` |
| N2-16 | 2.3125 | 74 (64 + 8 + 2) | 2 B | same | 4-bit signed per 16 | `nl2_s4x16` |
| U2 | 2.3125 | 74 | 2 B | uniform −2..1 | 4-bit signed per 16 | `u2a_s4x16` |
| U2-8 | 2.5625 | 82 | 2 B | uniform −2..1 | 4-bit signed per 8 | `u2a_s4x8` |
| **N2+** | 2.6875 | 86 (64 + 20 + 2) | 2 B | IQ2_K's 4 levels and their +5 copy | 4-bit signed + shift bit per 8 | `ik2x_s4x8` |
| N2-16+ (IQ2_K's layout) | 2.375 | 76 (64 + 10 + 2) | 4 B | same | 4-bit signed + shift bit per 16 | `ik2x_s4x16` |
| N3 | 3.25 | 104 (64 + 32 + 6 + 2) | 8 B | 8 non-uniform, two int8 words | 6-bit signed per 32 | `nl3_s6x32` |
| N3-4 | 3.1875 | 102 (64 + 32 + 4 + 2) | 2 B | same | 4-bit signed per 32 | `nl3_s4x32` |
| N3-16 | 3.4375 | 110 (64 + 32 + 12 + 2) | 2 B | same | 6-bit signed per 16 | `nl3_s6x16` |
| U3 | 3.25 | 104 | 8 B | uniform −4..3 | 6-bit signed per 32 | `u3a_s6x32` |
| **N3+** | 3.25 | 104 (64 + 32 + 6 + 2) | 8 B | IQ3_K's 8 levels and their +4 copy | 5-bit signed + shift bit per 32 | `ik3x_s5x32` |
| N3-16+ (IQ3_K's layout) | 3.4375 | 110 (64 + 32 + 12 + 2) | 2 B | same | 5-bit signed + shift bit per 16 | `ik3x_s5x16` |
| Q2_K (existing) | 2.625 | 84 | 2 B | uniform 0..3 with a minimum | 4-bit scale + 4-bit min per 16 | `Q2_K` |
| Q3_K (existing) | 3.4375 | 110 | 2 B | uniform −4..3 | 6-bit signed per 16 | `Q3_K` |
| IQ2_S (existing) | 2.5625 | 82 | 2 B | 1024-entry 8-weight grid × signs | 4-bit per 16 | `IQ2_S` |
| IQ3_XXS (existing) | 3.0625 | 98 | 2 B | 256-entry 4-weight grid × signs | 4-bit per 32 | `IQ3_XXS` |

The N2 and N3 tables were fitted once, by weighted Lloyd iterations on two experts of layer 6 (held out from the test
sample), and rounded to int8: 2-bit `[-127, -43, 23, 96]`, 3-bit `[-127, -84, -50, -19, 9, 39, 73, 121]`. Both are
asymmetric, like IQ4_NL's table; with a signed sub-scale a block can use either orientation. The "+" formats use
ik_llama.cpp's tables instead, `iq2nl_values` {−31, −13, 1, 17} × 4 and `iq3nl_values` {−63, −40, −23, −10, 1, 13, 28,
47} × 2 in int8, each with a copy shifted up by 5 × 4 and 4 × 2, and one bit per sub-block that picks the copy (the
"extra" bit of IQ2_K and IQ3_K). N3+ pays for its shift bit with one bit less of sub-scale than N3, so both are 104
bytes.

### Decode kernels

All candidates fit the fork's Pascal kernel template (`mmvq-pascal-impl.cuh`): one super-block per lane, the lane
group's segment staged through shared memory with 128-bit loads, activations as Q8_1 blocks of 32 int8 values held in
registers and shared by the R rows of a lane group (2R for the fused gate/up), integer sums per sub-block, one FP32
multiply per super-block.

**N2: codes as byte-permute selectors.** `__byte_perm(x, y, s)` picks four bytes out of eight using the low 3 bits of
each nibble of `s` (the fork's IQ4_XS kernel relies on bit 3 being ignored). With the table in one register `T` and
`__byte_perm(T, T, ·)`, bit 2 of a nibble picks the second copy, so only bits 0–1, the code, matter. Codes are packed
so that a 32-bit word `w` holds 16 codes in four groups:

```
level bytes of weights  0.. 3 = __byte_perm(T, T, w)         // codes in bits 0-1 of nibbles 0-3
                         4.. 7 = __byte_perm(T, T, w >> 2)    // bits 2-3 of nibbles 0-3
                         8..11 = __byte_perm(T, T, w >> 16)
                        12..15 = __byte_perm(T, T, w >> 18)
isum_sub += dp4a(levels, activation word)                      // 4 per word, natural weight order
```

Per 32 weights: 8 byte permutes, 6 shift slot-equivalents, 8 dp4a; four 4-bit sub-scales (extract and multiply the
8-weight integer sum: ~5 each); one conversion and FMA per 32; staging ~2; activations ~5 (no activation sums needed,
the levels carry their own offset). About 63 slot-equivalents per 32 weights, 2.0 per weight: **est. 3.5 ps per
weight** (286 G weights/s, 92 GB/s). N2-16 halves the scale work: est. 3.0 ps.

**N3: two planes assembled into selectors.** The 2-bit plane is packed as N2's codes; the 1-bit plane holds a word per
32 weights arranged so that `(h >> s) & 0x4444` puts the third bits of four consecutive weights into bit 2 of the
selector nibbles, for shifts s ∈ {−2, −1, 0, 1} and the same +16. One `LOP3` merges both:

```
sel = LOP3(lo >> t, h >> s, 0x3333)        // (lo & 0x3333) | (h & 0xCCCC); bit 3 is ignored by the permute
q   = __byte_perm(T0, T1, sel)             // 8 int8 levels in two registers
isum += dp4a(q, activation word)
```

Per 32 weights: 13 shifts (26), 8 `LOP3`, 8 byte permutes, 8 dp4a, a 6-bit scale (~6), final ~8, staging ~2.4,
activations ~5: about 71 slot-equivalents, 2.2 per weight: **est. 4.0 ps per weight** (250 G weights/s, 102 GB/s).
Uniform U3 with in-place masks (`lo & 0x03030303 << 2f`, the third bit through Q5_0's transposed-activation trick) costs
the same within a few slots, so the non-uniform table is free in time.

**The shift bit is free.** ik_llama.cpp's IQ2_K and IQ3_K give each sub-block one more bit that picks a copy of the
table shifted up by a few int8 steps. ik_llama.cpp folds that bit into every weight's index and reads an 8- or
16-entry table (4 byte permutes per 8 weights for IQ2_K, `get_int_from_table_16` for IQ3_K). Since the bit is constant
over a sub-block, a Pascal kernel can instead select the table registers once per sub-block (one or two `SEL`) and keep
the N2/N3 inner loops unchanged. N3+ decodes at N3's estimate (two `SEL` per 32 weights, four table registers);
N2+ selects per 8 weights and extracts a 5-bit field per 8: est. 3.7 ps per weight.

**U2 and Q2_K: masks in place.** Four 2-bit fields of a word are masked without shifting (`q & 0x03030303`,
`0x0C0C…`, `0x3030…`, `0xC0C0…`, the last with an unsigned-by-signed dp4a) and their sums combined with the powers of
4 once per sub-block, as the fork's Q4_K kernel does with its high nibbles. Q2_K's layout already puts four consecutive
weights in each 2-bit field of a word. U2: ~53 slot-equivalents per 32 weights, **est. 3.0 ps**; Q2_K with its minimum
(the activation half-block sums, shared by all rows): ~61, **est. 3.4 ps**.

**Q3_K:** both planes masked in place (the 1-bit plane with `0x01010101 << m`), combined per 16-weight sub-block with
shifts, plus its 12-byte scale packing: ~88 slot-equivalents, **est. 4.9 ps**.

**IQ2_S and IQ3_XXS with shared-memory tables** (the approach of fork branch `pascal-p106-v3-iqmmvq`, commit
`6b880e344`: grid and sign-mask tables copied into each block's shared memory, signs applied as
`sum(g·u) − 2·sum((g & neg)·u)`): per 32 weights IQ3_XXS needs 8 grid loads and 4 mask loads with their index and
address arithmetic, 16 dp4a and 8 ANDs, about 93 slot-equivalents (est. 5.2 ps); IQ2_S about 85 (est. 4.7 ps). Random
gathers from a table in shared memory also conflict on banks (32 banks of 4 bytes; 8-byte entries for IQ2_S), which
stalls without costing issue slots. **Measured on the rig** (UD-Q2_K_XL, 9 × P106, GGML_CUDA_PROFILE, the same binary
with the kernels on and off): IQ3_XXS down-projections 12.9 ps per weight (llama.cpp's kernel 22.5), IQ2_S fused
gate/up 9.9 ps (11.6), IQ3_XXS gate/up 10.6 ps, IQ2_S down 11.4 ps; 28–36 GB/s. The stalls double the slot estimate.
N2P and N3P read no table from memory, so their estimates carry no such term; the rig measures them next.

### What it means for an Ornith token

A token reads 3.62 G routed-expert weights (2.42 G gate and up, 1.21 G down) and 5.35 G dense weights. Routed-expert
time per token at the estimates above (my calc):

| Experts | ps per weight (gate+up / down) | ms per token | Token time (rest as measured) | tok/s |
|---|---|---|---|---|
| Today: IQ2_S / IQ3_XXS, generic kernel (measured) | 10.8 / 20.5 | 50.8 | 117.7 | 8.5 |
| IQ2_S / IQ3_XXS, shared-memory Pascal kernels (measured per weight) | 9.9 / 12.9 | 39.6 | 106.5 | 9.4 |
| Q2_K / Q3_K, Pascal kernels | 3.4 / 4.9 | 14.1 | 81.0 | 12.3 |
| N2 / N3 | 3.5 / 4.0 | 13.3 | 80.2 | 12.5 |
| N2+ / N3+ | 3.7 / 4.0 | 13.8 | 80.7 | 12.4 |

The dense weights are the larger lever: 5.35 G weights per token at Q8_0 take 38.7 ms. At Q5_0 (5.0 ps) they would
take ~27 ms and at Q6_K (est. 5.5–6 ps) ~30–32 ms, independent of the expert format, and the bytes saved go to the
experts (next section). Unsloth's UD-Q2_K_XL already keeps most of these tensors at Q4_K/Q5_K.

### The bit budget

Ornith-122B's file without the token embedding (kept on the host) and without the MTP block (loaded only for MTP
decoding) puts 45.40 GB on the cards as AD-IQ3_XXS-IQ2_S; that size fits 122,880 tokens of context at `-ub 128`
(measured). Holding the cards at 45.4 GB:

| Dense tensors (5.35 G weights) | Dense GB | Router (F32) | Left for 115.96 G expert weights | Expert bits per weight |
|---|---|---|---|---|
| Q8_0 (AD file) | 5.68 | 0.16 | 39.56 GB | 2.729 |
| Q6_K | 4.39 | 0.16 | 40.85 GB | 2.818 |
| Q5_0, output head and attention k/v Q8_0 | 3.98 | 0.16 | 41.26 GB | 2.847 |
| Q5_0 | 3.68 | 0.16 | 41.56 GB | 2.867 |

Expert mixes inside 2.73–2.87 bits per weight (gate and up / down):

| Mix | Bits per weight |
|---|---|
| IQ2_S / IQ3_XXS (AD) | 2.729 |
| N2-16+ / N3-16+ (IQ2_K / IQ3_K layouts) | 2.729 |
| N2 / N3 | 2.792 |
| IQ2_S / IQ3_S | 2.854 |
| N2 / N3-16 | 2.854 |
| N2+ / N3+ | 2.875 (dense all Q5_0: 45.5 GB, 0.1 GB over AD) |
| Q2_K / Q3_K | 2.896 (needs dense at ~5 bits) |

## 4. Quality proxy

### Method

[`scripts/lowbit_proxy.py`](../../../scripts/lowbit_proxy.py) quantizes real routed-expert matrices of
Ornith-1.5-122B on the CPU and measures each format's importance-weighted relative reconstruction error

> E = Σ over rows and columns of imp_j · (w_ij − q_ij)² / Σ imp_j · w_ij²

where imp_j is the mean squared activation of input column j for that expert, from Unsloth's importance matrix for the
122B (694 chunks of 512 tokens). If a matrix's inputs were uncorrelated, E would be the expected squared error of its
output relative to the output's energy. It ranks formats on the same weights; it is not a substitute for KLD.

- **Sample:** layers 2, 10, 18, 26, 34, 42 and 47; in each, the two experts the calibration text routed to most and two
  random ones; their gate, up and down matrices (84 matrices, 264 M weights, read from the BF16 safetensors one expert
  slice at a time). The non-uniform tables were fitted on layer 6.
- **Reference formats** use llama.cpp's own quantizers with the importance matrix (`ggml_quantize_chunk` from a CPU
  build of `libggml-base` at d77256999) and its dequantizers.
- **Candidates** use the same weighting as llama.cpp's quantizers (imp_j · sqrt(σ² + w_j²), σ² the row's mean square),
  a grid of 14 scales per block refined by weighted least squares, FP16 rounding of the stored scales, and for
  super-blocks a search over 7–14 super-scales with the integer sub-scales rounded under each. As a check, the same
  code with Q4_0's layout lands within 1 % of llama.cpp's Q4_0. The "+" formats choose per sub-block between the table
  and its shifted copy by the lower error.
- **Rotation:** variants marked `+r256` quantize the matrix after a random-sign 256-point Hadamard rotation of its
  input dimension (undone at run time by rotating the activations) and measure the error after rotating back.

### Results

Geometric means over the sampled matrices (56 gate/up, 28 down); "vs IQ3_XXS" is the paired ratio on the same 84
matrices. Lower is better.

| Format | Bits/weight | Gate/up | Down | All | vs IQ3_XXS |
|---|---|---|---|---|---|
| Q4_0 (llama.cpp) | 4.5 | 0.0070 | 0.0040 | 0.0058 | 0.22 |
| Q4_0 layout, this script's quantizer (check) | 4.5 | 0.0069 | 0.0039 | 0.0057 | 0.22 |
| IQ4_XS | 4.25 | 0.0058 | 0.0038 | 0.0051 | 0.19 |
| IQ3_S | 3.4375 | 0.0186 | 0.0118 | 0.0160 | 0.60 |
| Q3_K | 3.4375 | 0.0214 | 0.0116 | 0.0174 | 0.65 |
| N3-16 (`nl3_s6x16`) | 3.4375 | 0.0185 | 0.0099 | 0.0151 | 0.56 |
| U3 per 32 (`u3a_32`, FP16 per 32) | 3.5 | 0.0278 | 0.0163 | 0.0233 | 0.87 |
| non-uniform per 32 (`nl3_32`, FP16 per 32, like IQ3_NL) | 3.5 | 0.0243 | 0.0140 | 0.0202 | 0.76 |
| **N3** (`nl3_s6x32`) | 3.25 | 0.0244 | 0.0142 | 0.0203 | 0.76 |
| N3, rotated (`+r256`) | 3.25 | 0.0262 | 0.0160 | 0.0222 | 0.83 |
| U3 (`u3a_s6x32`) | 3.25 | 0.0279 | 0.0165 | 0.0234 | 0.88 |
| U3 without a zero level (`u3s_s6x32`) | 3.25 | 0.0312 | 0.0187 | 0.0263 | 0.99 |
| N3-4 (`nl3_s4x32`) | 3.1875 | 0.0253 | 0.0158 | 0.0217 | 0.81 |
| IQ3_XXS | 3.0625 | 0.0324 | 0.0182 | 0.0267 | 1.00 |
| Q2_K | 2.625 | 0.0708 | 0.0328 | 0.0548 | 2.05 |
| IQ2_S | 2.5625 | 0.0622 | 0.0411 | 0.0541 | 2.03 |
| **N2** (`nl2_s4x8`) | 2.5625 | 0.0637 | 0.0351 | 0.0522 | 1.95 |
| U2-8 (`u2a_s4x8`) | 2.5625 | 0.0709 | 0.0409 | 0.0590 | 2.21 |
| IQ2_XS | 2.3125 | 0.0888 | 0.0547 | 0.0756 | 2.83 |
| N2-16 (`nl2_s4x16`) | 2.3125 | 0.0855 | 0.0498 | 0.0714 | 2.67 |
| U2 (`u2a_s4x16`) | 2.3125 | 0.0952 | 0.0571 | 0.0803 | 3.00 |
| Q2_0 (llama.cpp, no importance matrix) | 2.25 | 0.525 | 0.371 | 0.467 | 17.5 |
| IQ2_XXS | 2.0625 | 0.1167 | 0.0728 | 0.0997 | 3.73 |

The shifted-table formats, over the same 84 matrices (a second run with the same experts):

| Format | Bits/weight | Gate/up | Down | All | vs IQ3_XXS |
|---|---|---|---|---|---|
| N3-16+ (`ik3x_s5x16`, IQ3_K's layout) | 3.4375 | 0.0167 | 0.0093 | 0.0138 | 0.52 |
| N3+, 6-bit scales (`ik3x_s6x32`) | 3.28 | 0.0218 | 0.0125 | 0.0181 | 0.68 |
| **N3+** (`ik3x_s5x32`) | 3.25 | 0.0220 | 0.0129 | 0.0184 | 0.69 |
| IQ3_K's table without the shift bit (`ik3_s6x32`) | 3.25 | 0.0239 | 0.0145 | 0.0202 | 0.76 |
| N3+, 4-bit scales (`ik3x_s4x32`) | 3.22 | 0.0227 | 0.0143 | 0.0195 | 0.73 |
| **N2+** (`ik2x_s4x8`) | 2.6875 | 0.0521 | 0.0288 | 0.0428 | 1.60 |
| IQ2_K's table without the shift bit (`ik2_s4x8`) | 2.5625 | 0.0653 | 0.0376 | 0.0544 | 2.04 |
| N2+, 6-bit scales per 16 (`ik2x_s6x16`) | 2.5 | 0.0732 | 0.0370 | 0.0583 | 2.18 |
| N2+, 5-bit scales per 16 (`ik2x_s5x16`) | 2.4375 | 0.0736 | 0.0383 | 0.0592 | 2.22 |
| N2-16+ (`ik2x_s4x16`, IQ2_K's layout) | 2.375 | 0.0751 | 0.0421 | 0.0619 | 2.32 |

What the proxy shows:

1. **Non-uniform levels in registers match mainline's codebooks at equal size.** N2 against IQ2_S at 2.5625 bits:
   1–4 % more error on gate/up in every sampled layer, 15 % less on the down-projections (most of it in layers 2 and
   10), 3.6 % less over all matrices. N3-16 against IQ3_S at 3.4375 bits: equal on gate/up in every layer, 16 % less on
   down. N3 at 3.25 bits has 22–24 % less error than IQ3_XXS at 3.0625, about what 0.19 more bits buy.
2. **Uniform levels cost 11–23 % more error at equal size, in every layer** (U2-8 against N2, U3 against N3), and
   uniform levels without zero more again. Q2_K and Q3_K, uniform with finer scales, sit with them: Q2_K's gate/up error
   is 12–15 % above IQ2_S's in every layer.
3. **Finer sub-scales help some down-projections a lot.** N3-16 (a scale per 16 weights) against IQ3_S (a scale per 32)
   at the same 3.4375 bits: 23 % and 65 % less down-projection error in layers 2 and 10, where the importance matrix
   concentrates on few columns, and equal in the other five layers.
4. **Rotation does not help** once the importance matrix weights the quantizer: the rotated N3 has 9 % more error, and
   all six rotated variants of the first run were worse than unrotated. Flint's rotation experiment (Q13) found the
   same for 4-bit quants with an importance matrix.
5. **llama.cpp's Q2_0 is unusable for post-training quantization** (17× IQ3_XXS's error at 2.25 bits): its quantizer
   is ternary absmax, made for models trained in that format.
6. **ik_llama.cpp's shift bit moves a format ahead of the codebooks.** At the same 104 bytes, N3+ (5-bit scale and
   shift bit) has 9.5 % less error than N3 (6-bit scale), 10 % less on gate/up in every layer; IQ3_K's table without
   the bit is no better than the fitted one. Per bit, taking ~6 dB per bit to compare sizes, N3+ and N3-16+ are ~9–12 %
   better on gate/up than IQ3_S and IQ3_XXS; N3-16+ against IQ3_S at the same 3.4375 bits: 10 % less gate/up error in
   every layer, 3–61 % less down. At 2 bits the gain is smaller: N2+ (2.6875 bits) has 16 % less gate/up error than
   IQ2_S, what its 0.125 extra bits buy on IQ2_S's own curve, and IQ2_K's layout (2.375 bits) has 15 % less than IQ2_XS
   at 2.3125, ~7 % ahead of IQ2_S's curve.

Expert mixes at the sizes the budget allows, summing the three matrices' errors (gate/up dominate the sum; this weighs
the three equally, which a KLD test has to confirm):

| Mix (gate and up / down) | Bits/weight | Gate/up | Down | Sum of three |
|---|---|---|---|---|
| IQ2_S / IQ3_XXS (AD file) | 2.729 | 0.0622 | 0.0182 | 0.143 |
| N2-16+ / N3-16+ (IQ2_K / IQ3_K layouts) | 2.729 | 0.0751 | 0.0093 | 0.159 |
| N2 / N3 | 2.792 | 0.0637 | 0.0142 | 0.142 |
| IQ2_S / IQ3_S | 2.854 | 0.0622 | 0.0118 | 0.136 |
| IQ2_S / Q3_K | 2.854 | 0.0622 | 0.0116 | 0.136 |
| N2 / N3-16 | 2.854 | 0.0637 | 0.0099 | 0.137 |
| IQ layer mix: gate/up 75 % IQ2_S + 25 % IQ3_XXS, down half IQ3_XXS + half IQ3_S (my calc) | 2.875 | 0.0548 | 0.0150 | 0.125 |
| **N2+ / N3+** | 2.875 | 0.0521 | 0.0129 | **0.117** |
| Q2_K / Q3_K | 2.896 | 0.0708 | 0.0116 | 0.153 |

At ~2.875 bits, N2+/N3+ has 6 % less summed error than the best IQ layer mix of the same size and 14 % less than
IQ2_S/IQ3_S at 2.854; Q2_K/Q3_K has 12 % more than IQ2_S/IQ3_S while using more bits. The gate/up matrices carry most
of the error, so a mix gains most from giving them more bits.

## 5. Recommendation

### What a new format buys and what it does not

- **Quality: a little, and only with ik_llama.cpp's shift bit.** Non-uniform levels in registers land on the same
  error-per-bit curve as mainline's codebooks (section 4), and better than uniform levels or the K-quants. The shift
  bit moves the 3-bit formats ~9–12 % ahead per bit, and the N2+/N3+ mix has 6 % less error than the best IQ mix of
  the same size: worth having, but not a reason on its own to build new types.
- **Speed: est. ~3× per expert weight over the IQ2_S/IQ3_XXS Pascal kernels with shared-memory tables (measured
  9.9–12.9 ps per weight), and 3–5× over today.** That is ~26 ms of a ~107-ms token (est. +30 % tok/s), if N2P and
  N3P reach their estimates on the rig.
- **Bits are free in time, bytes are not.** Expert decode in any of these formats is issue-bound (est. 3.5–4 ps per
  weight whether 2.4 or 3.3 bits), so expert bits should be as many as memory allows. The dense tensors are the
  opposite: Q8_0 is bandwidth-bound (7.2 ps per weight in the profile), and moving the 5.35 G dense weights to Q5_0
  (5.0 ps) saves ~10–12 ms per token and 1.7–2.0 GB, as much time as any expert format and with no new code. Dense
  tensors around 5 bits are what Unsloth's UD-Q2_K_XL already uses.

Projected decode (my calc; today's other work unchanged, MTP off):

| Route | Experts | Dense | ms per token | tok/s |
|---|---|---|---|---|
| Today (measured) | IQ2_S/IQ3_XXS, generic kernel | Q8_0 | 117.7 | 8.5 |
| IQ Pascal kernels | IQ2_S/IQ3_XXS, 9.9/12.9 ps (measured) | Q8_0 | 106.5 | 9.4 |
| IQ Pascal kernels + dense Q5_0 | IQ mix at ~2.85 bits, ~10–13 ps | Q5_0, head and k/v Q8_0 | ~96 | ~10.4 |
| New register-table formats | N2+/N3+, 3.7/4.0 ps | Q8_0 | 81 | 12.4 |
| New formats + dense Q5_0 | N2+/N3+ at 2.875 bits | Q5_0 | 69 | 14.5 |
| Q2_K/Q3_K Pascal kernels + dense Q5_0 | Q2_K/Q3_K, 3.4/4.9 ps | Q5_0, head and k/v Q8_0 | 71 | 14.1 (with ~12 % more expert error at equal size) |

The agent check gains less than decode does, because prompt processing (8.7 s of first-token time per turn on v2.0)
is not changed by any of this.

### What to do, in order

1. **No new code: measure the IQ Pascal kernels on the rig** (the other branch's work) and **move Ornith's dense tensors
   off Q8_0** in the next quant (KLD on the 35B and, once its reference exists, the 122B). These decide most of the
   speed.
2. **Two cheap gates before any new type (both ~1–2 days, independent):**
   - **Speed:** prototype N2+ and N3+ kernels in `bench/mmvq_pascal` at Ornith's expert shapes and time them on one
     P106 next to the IQ2_S/IQ3_XXS kernels of `6b880e344` and Pascal Q2_K/Q3_K kernels. Go on only if N2+/N3+ decode
     ≥ 1.3× faster per weight than the IQ kernels.
   - **Quality:** simulated-quantization KLD on Ornith-1.5-35B (test plan, step 2): the N2+/N3+ mix against the IQ mix
     and the Q2_K/Q3_K mix at matched bytes. Go on only if the N2+/N3+ mix is at least within noise of the IQ mix.
3. **If both gates pass: build two types, N2+ for the expert gate/up and N3+ for the down-projections** (mix at
   2.875 bits with the dense tensors at Q5_0, ~45.5 GB on the cards), as Miner Llama's own types with ids outside
   mainline's and ik_llama.cpp's ranges. Their level tables are ik_llama.cpp's (MIT-licensed; credit it in the source
   and the model cards). ~7–8 working days. The alternative of porting ik_llama.cpp's IQ2_K and IQ3_K exactly (ids 137
   and 138, quantizers that exist, files that ik_llama.cpp also loads) fits this budget worse: IQ2_K's 16-weight
   sub-blocks give gate/up 44 % more error than N2+, and a layer mix of IQ2_K and IQ3_K at 2.875 bits comes out ~12 %
   above N2+/N3+ (my calc from section 4).
4. **If the speed gate fails** (IQ kernels within ~1.3× of the prototypes): keep IQ2_S/IQ3_XXS with their Pascal
   kernels and spend the effort on the dense tensors and the mix. Pascal Q3_K kernels (~2 days) are still worth it for
   the down-projections if the IQ3_XXS kernel is the slow one.

### Risks

- **The proxy is not KLD.** It ignores how errors in gate/up and down propagate and assumes uncorrelated inputs; it ranks
  formats on the same matrices, nothing more. Step 2's simulated KLD on the 35B settles quality before code.
- **The speed figures are estimates** from slot counts calibrated on the fork's measured kernels; shared-memory bank
  conflicts (IQ) and register pressure (N3 holds four table registers with the shift bit) can move them by tens of
  percent. Step 2's microbenchmark settles speed before code.
- **The microbenchmark-to-model gap** (~1.3–2 µs per 2 MiB page a call touches,
  [Pascal forks and quants](pascal_forks_and_quants.md), section 3) adds the same time to every format.
- **Files that only Miner Llama loads.** A model published in the new types runs nowhere else (or only in ik_llama.cpp,
  for a port), and every new model needs quantizing by this project from its BF16 weights and an importance matrix.
- **Prompt processing** needs MMQ tile loaders for the new types; without them prefill falls back to dequantize and an
  FP32 GEMM and the agent check gets slower, not faster.
- **Multi-token steps** (MTP verify with 2–4 tokens) amortize the decode work over tokens, which narrows the gap between
  the IQ kernels and new formats.

### What building a new format takes

For each new type (N2+ and N3+ count as two), in the fork (`pascal-p106-v3` and a feature branch off it):

| Piece | Where | Effort (est.) |
|---|---|---|
| Type ids, block structs, type traits (block size, bytes, to/from float) | `ggml.h`, `ggml-common.h`, `ggml.c` | 0.5 day |
| CPU reference quantize and dequantize, and the importance-matrix quantizer (the search in `scripts/lowbit_proxy.py`: per-block scale grid with least-squares refit, then the super-block's sub-scale search) | `ggml-quants.c` | 1 day |
| CPU dot product against Q8_K or Q8_0, scalar (needed by `test-backend-ops`, which compares against the CPU) | `ggml-cpu/quants.c` | 0.5 day |
| `llama-quantize` accepts the type names for `--tensor-type`; gguf-py knows the type ids and block sizes (for `gguf_mix.py`, `token_cost.py`, header summaries) | `tools/quantize`, `gguf-py` | 0.5 day |
| CUDA dequantize (to F16/F32) and `get_rows` | `convert.cu`, `getrows.cu` | 0.5 day |
| CUDA generic MMVQ dot product, so other cards and fallbacks run | `vecdotq.cuh`, `mmvq.cu` | 0.5 day |
| Pascal MMVQ: block traits, activation loader, dot product, launch shape; 1–4 tokens; `MUL_MAT_ID` and the fused gate/up | `mmvq-pascal-impl.cuh`, `mmvq-pascal.cu` | 2 days |
| Prompt processing: MMQ tile loader (`load_tiles_*`, decoded to int8 tiles with a scale per 32 like IQ4_XS's), else prefill falls back to dequantize + FP32 GEMM | `mmq.cuh` | 1–1.5 days |
| Tests: `test-backend-ops` cases (MUL_MAT, MUL_MAT_ID at Ornith's shapes, GET_ROWS, 1–4 tokens), `test-quantize-fns` | `tests/` | 0.5 day |

About 7–8 working days for both types, plus the quantization and scoring runs. Compatibility: GGUF files that use the new
type ids load only in Miner Llama; stock llama.cpp refuses them as an unknown type. Mainline numbers its types in
sequence (`Q2_0` took 42 in July 2026) and ik_llama.cpp uses 133–161 for its own types and 202–399 for repacked ones
([ggml.h](https://github.com/ikawrakow/ik_llama.cpp/blob/main/ggml/include/ggml.h)), so Miner Llama's ids belong in a
range neither uses, to keep another engine's later type from reading these files as something else.

Pascal kernels for Q2_K and Q3_K need only the Pascal MMVQ row of this table (about 2 days); the IQ2_S/IQ3_XXS kernels
are already written on `pascal-p106-v3-iqmmvq`.

### Test plan

1. **Speed first, in the microbenchmark (rig, ~1 h; building ~1–2 days).** Add Ornith's expert shapes (3072 → 1024 and
   1024 → 3072, 8 of 256 experts, gate/up fused) and prototype kernels for N2+ and N3+ (with and without the shift
   bit), Q2_K and Q3_K to
   [`bench/mmvq_pascal`](../../../bench/mmvq_pascal/README.md), with random weights and the CPU reference check, plus
   the IQ2_S and IQ3_XXS kernels of `6b880e344`. Report ps per weight and GB/s for 1 and 4 tokens.
2. **Quality by simulation (quant machine, after the MTP head training).** Quantize Ornith-1.5-35B's routed experts with
   the candidate quantizers in numpy (`lowbit_proxy.py`'s code), write the dequantized values back as BF16, convert to
   GGUF and store the experts as Q8_0 (its error is ~100× below a 3-bit format's), keep every other tensor as in the
   comparison files, and score KLD against the 35B's Q8_0 file with `scripts/quality_kld.sh` (80 × 512 tokens), next
   to the same file with IQ2_S/IQ3_XXS and with Q2_K/Q3_K experts from `llama-quantize`. The 35B needs its own importance
   matrix first (`llama-imatrix`, two V100s). This scores the formats exactly, before any of them exists in ggml.
3. **Correctness** once a type is built: `test-backend-ops -o MUL_MAT,MUL_MAT_ID,GET_ROWS` on the quant machine for the
   new types against the CPU reference, at Ornith's shapes and 1–4 tokens; `test-quantize-fns`.
4. **Kernel speed on the rig** with the built types: `test-backend-ops perf` and `bench/mmvq_pascal` at the expert
   shapes, against the microbenchmark's prototypes.
5. **KLD on the 35B** with real files of the new types (must match step 2 within noise), then on the 122B once its Q8_0
   reference logits exist (the reference run waits for the MTP head training).
6. **Agent check on the rig**: the 122B in the chosen mix on v3 against AD-IQ3_XXS-IQ2_S on v3 with the IQ Pascal
   kernels, same context and split, paired by task.
