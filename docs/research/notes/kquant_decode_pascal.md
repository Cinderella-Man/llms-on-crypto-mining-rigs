# K-quant decode on the P106-090: how much faster could Unsloth's UD-Q4_K_XL of Flint run? (as of 2026-10-23)

Most people who run [Flint](https://huggingface.co/RecursiveSI/flint) (39.8 B-parameter MoE, ~2.5 B per token) download
Unsloth's `UD-Q4_K_XL` GGUF, and that file is this project's quality bar (KLD 0.120, 87.7 % same top token against
BF16, [quants/research.md](../../../quants/research.md)). On the rig (nine P106-090 cards, compute 6.1, 5 SMs each,
[hardware](../../mining_rig.md)) it is also one of the slowest Flint files, because its K-quant and I-quant tensors decode
expensively on 5-SM Pascal cards. This note asks how much faster llama.cpp itself could decode that unchanged file, so
that its users gain without switching to a custom quant, and compares that with the project's own route: simple-format
files such as S4-L0 (Q4_1/Q5_0/Q8_0 at the same quality).

Scope: single-token decode (MMVQ, llama.cpp's `mul_mat_vec_q`), llama.cpp `d77256999` with CUDA graphs on Pascal.
Speeds are `llama-bench` tg128 on 6 cards unless stated otherwise.

Conventions, as in the other notes here:
- **(measured)**: measured on the rig, from [quants/experiments.md](../../../quants/experiments.md) (Q25, Q34, Q36) and
  the [mmvq_pascal microbenchmark](../../../bench/mmvq_pascal/README.md).
- **(my measurement)**: static analysis of the binaries the rig runs: the sm_61 code of `llama-bench` from the builds
  `llama.cpp-20261022-d77256999-graphs61` (4 warps per block at one token, llama.cpp's default on Pascal) and the same
  tree with [patches/mmvq-pascal-2warps.diff](../../../patches/mmvq-pascal-2warps.diff) (2 warps), disassembled with
  `cuobjdump -sass` from the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image. Instructions were counted per basic block
  and summed along the path each warp takes at Flint's shapes ([appendix](#appendix-path-counts)).
- **(my calc)**: arithmetic in this note. **claim to test**: not measured on the rig.

## 1. Summary

1. **Two thirds of UD-Q4_K_XL's bytes are already Q8_0 (my measurement).** A token reads 2.21 GB: 1.48 GB Q8_0
   (attention q and output, the output head, the shared expert, attention k/v, the dense layers), 0.68 GB of K- and
   I-quants (Q5_K 0.42, IQ4_XS 0.22, Q4_K, Q6_K and IQ3_S 0.03), all in the routed experts plus seven attention layers
   and thirteen shared-expert layers, and 0.05 GB of BF16 router. The K/I-quants are 31 % of the bytes but about 46 %
   of the matrix-vector time in the 2-warp build.
2. **llama.cpp spends 12–50 instructions per weight byte on these tensors, against ~9–15 for Q4_1 (my measurement).**
   The card can issue only ~6 per byte of its memory stream. Three things make K/I-quants worse than the legacy formats
   at Flint's shapes:
   - **One loop iteration per thread, every 16 or 32 weights a full re-derivation** of 176- or 136-byte superblock
     addresses, the 6-bit scales and the activation-block addresses: integer and address arithmetic is 38 % of the
     Q5_K loop, scales, minimums and the recomputed activation sums another ~23 %.
   - **Idle threads.** IQ4_XS gives each thread 32 weights, so a 2048-wide row needs only 64 threads: with llama.cpp's
     4 warps per block, half of every gate/up block idles. In the 512-wide expert down-projection only 16 of 128 threads
     work, on a single row per block (50 instructions per byte), because llama.cpp excludes IQ4_XS from its "small-K"
     rule on Pascal, which would give each block 4 rows. The Q5_K down-projection takes the small-K path but still
     leaves 3 of 4 warps idle.
   - **The IQ4_XS lookup table:** 32 byte-permute instructions per 32 weights, a third of its loop.
3. **The 2-warp launch shape (+27 % on this file, measured) came mostly from its Q8_0 tensors (my calc):** attention q
   and the output head go from 63 to 99–103 GB/s, and the IQ4_XS gate/up blocks stop idling. The K/I-quant part
   improves only ~10 %.
4. **A Pascal kernel for Q5_K and IQ4_XS could roughly double their decode speed, from 51 to 105–130 GB/s (my calc,
   claim to test).** Rows sized to K, several rows per warp sharing the activations, wide loads and loop-invariant
   addressing bring Q5_K to ~1.8 instructions per byte in the loop (the tuned Q4_1 kernel's loop: 1.6) and IQ4_XS to
   ~3.5 (2.4 of it the table lookup, which sm_61 cannot avoid). Both are below the card's issue limit, so they should
   become bandwidth-bound like the tuned Q4_1/Q8_0 kernels (measured 85–159 GB/s).
5. **End to end (my calc, calibrated on the measured 2-warp build):** UD-Q4_K_XL goes from 24.8 tok/s to 26.2 with a
   Pascal Q8_0 kernel alone, to 30–31 with Pascal K/I-quant kernels alone, to **31–33 with both**, and to **33–37** once
   Flint's MoE rescale is gone. That needs FP32 activation sums in the engine: switching the rescale off as it stands
   is exact for this file's decode, but not for its prompt processing, where the Q5_K down-projection of layer 46
   reads the FP16 sums. A one-line fix of the small-K rule for IQ4_XS on Pascal gives ~+2 % on its own.
6. **The simple-format route ends at about the same place:** S4-L0 runs 28.0 tok/s today (2 warps for wide rows,
   measured) and an estimated 33–34 tok/s with tuned Q4_1, Q8_0 and Q5_0 kernels, at the same quality and already
   rescale-free.
7. **Better investment: the Pascal Q8_0 kernel first (shared by both routes, measured 1.52×), then Q5_K and IQ4_XS
   kernels rather than a Q5_0 kernel.** With the kernels in place both routes reach ~33–35 tok/s, so the deciding
   factor is reach: K/I-quant kernels serve everyone who downloads Unsloth's or bartowski's files of any model (the
   rig's Ornith and Qwen3.8-distill files are Q4_K_M, and its Qwen3.6 Q4_0 file keeps a Q6_K head) and make the custom
   32.5 GB file unnecessary,
   while S4-L0 only helps people who find and download it. S4-L0 stays the faster option until the kernels exist
   (28.0 against 24.8 tok/s). The step that decides it is cheap: add Q5_K and IQ4_XS to
   [bench/mmvq_pascal](../../../bench/mmvq_pascal/) (1–2 days, one card, no model).

## 2. What a UD-Q4_K_XL token reads

From the GGUF header ([gguf_mix.py](../../../scripts/gguf_mix.py), [token_cost.py](../../../scripts/token_cost.py);
experts count 6 of 256 per token; the embedding row is left out):

| Tensor class | Q8_0 | K/I-quants | MB per token |
|---|---|---|---|
| Attention q (2048 → 4096) | 42 layers | Q5_K in 7 layers (19, 26, 35, 39, 43, 45, 46) | 374 + 40 |
| Attention output (4096 → 2048) | 42 layers | Q5_K in the same 7 | 374 + 40 |
| Output head (2048 → 150,144) | all | | 327 |
| Shared expert gate, up, down | 34 layers | Q5_K in 12 layers, IQ4_XS in 1 | 227 + 52 + 3 |
| Attention k, v (2048 → 512) | 42 layers | Q5_K in the same 7 | 94 + 10 |
| Dense FFN, layers 0–1 | all | | 80 |
| Routed expert gate and up (2048 → 512) | | IQ4_XS in 25 layers (mostly 19–48), Q5_K in 18 (mostly 3–18), Q4_K in 21–23, IQ3_S in 2 | 167 + 156 + 21 + 5 |
| Routed expert down (512 → 2048) | 1 layer | Q5_K in 29 layers, IQ4_XS in 16, Q6_K in 1 | 7 + 125 + 54 + 5 |
| Router (2048 → 256) | | BF16 | 49 |
| **Total** | **1,483 (67 %)** | **680 (31 %)** | **2,213** |

The two block formats that matter:

- **Q5_K** (5.5 bits per weight): superblocks of 256 weights in 176 bytes: an FP16 scale and an FP16 minimum, 12 bytes
  holding eight 6-bit sub-block scales and eight 6-bit minimums for sub-blocks of 32, 32 bytes of fifth bits, 128 bytes
  of low nibbles. Weight = d · scale_j · q − dmin · min_j.
- **IQ4_XS** (4.25 bits per weight): superblocks of 256 weights in 136 bytes: an FP16 scale, 2 + 4 bytes holding eight
  6-bit sub-block scales, 128 bytes of 4-bit indices into a fixed table of 16 non-uniform values (−127 … 113). Weight =
  d · (scale_j − 32) · table[q].

Both are decoded against llama.cpp's 8-bit activation blocks (Q8_1: 32 int8 values with an FP16 scale and an FP16 sum).

## 3. How llama.cpp decodes these types at Flint's shapes

### Thread mapping

At one token llama.cpp's MMVQ runs one output row per block of 4 warps on Pascal (the generic parameter table,
[mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/mmvq.cu)). Each thread takes
`vdr` 32-bit words of one superblock per call ([vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/vecdotq.cuh)):

| Type | Threads per superblock | Weights per call | Superblocks per 4-warp iteration | Flint gate/up (K = 2048) | Flint down (K = 512) |
|---|---|---|---|---|---|
| Q4_1 (for comparison) | 2 per 32-block | 16 | 64 blocks = 2,048 weights | 1 iteration, all threads | small-K: 4 rows, 1 of 4 warps works |
| Q5_K, Q4_K, Q6_K | 16 | 16 | 8 = 2,048 weights | 1 iteration, all threads | small-K: 4 rows, 1 of 4 warps works |
| IQ4_XS | 8 | 32 | 16 = 4,096 weights | **2 of 4 warps idle** | **16 of 128 threads work, 1 row** |
| IQ3_S | 8 | 32 | 16 | 2 of 4 warps idle | (not used for down here) |

The "small-K" rule (PR [#20635](https://github.com/ggml-org/llama.cpp/pull/20635), extended in
[#20905](https://github.com/ggml-org/llama.cpp/pull/20905)) gives a block as many rows as it has warps when one
iteration covers the whole row. On Pascal it is switched off for IQ4_XS, IQ3_S, Q2_K, Q3_K and the 1–3-bit IQ types
(IQ4_NL keeps it). With the 2-warp patch, IQ4_XS gate/up uses all 64 threads, Q5_K takes 2
iterations per thread, and the down-projections keep the same idle pattern with 2 rows instead of 4.

### Instructions per weight byte, along the path each warp takes

Lane-instructions issued per byte of weights (warp-instructions × 32 ÷ bytes), whole kernel including set-up, idle
warps and the reduction (my measurement; path counts in the [appendix](#appendix-path-counts)):

| Kernel at Flint's shape | 4 warps | 2 warps | Model GB/s, 4 w / 2 w |
|---|---|---|---|
| Q4_1 expert gate/up, fused (calibration) | 11.9 | 8.5 | 84 / 103 (measured) |
| Q4_1 expert down, K = 512, small-K | 13.1 | 14.2 | 57 / 52 (measured) |
| Q4_1 attention q, K = 2048 | 15.2 | 10.6 | 68 / 97 (measured) |
| **Q5_K expert gate/up, fused** | 16.2 | 12.9 | 61 / 68 |
| **IQ4_XS expert gate/up, fused** | 14.2 | 11.8 | 70 / 74 |
| Q4_K expert gate/up, fused | 18.4 | 14.2 | 54 / 62 |
| IQ3_S expert gate/up, fused | 24.7 | 21.8 | 40 / 40 |
| **Q5_K expert down, K = 512, small-K** | 18.5 | 18.9 | 40 / 39 |
| **IQ4_XS expert down, K = 512** | **50.0** | **38.7** | **15 / 19** |
| IQ4_XS expert down with small-K (compiled, never launched on Pascal) | 23.2 | 25.9 | 32 / 29 |
| Q5_K attention, K = 2048 / 4096 | 20.2 / 15.4 | 15.6 / 13.0 | 51–67 / 65–78 |
| Q5_K shared-expert down, K = 1024 | 15.0 | 20.8 | 49 / 49 |

GB/s for the K/I-quants comes from dividing the measured issue rate of llama.cpp's own Q4_1 kernels of the same kind
by these counts. That rate is remarkably constant on the rig (my calc from the measured Q4_1 GB/s): 1.02–1.04 T
lane-instructions per second for the plain kernels, 0.99 (4 warps) and 0.88 (2 warps) for the fused gate/up kernel,
0.73–0.74 for the small-K kernels, where only one warp of each block works. The card's peak is 1.11 T (5 SMs × 128
lanes × ~1.73 GHz). llama.cpp's Q8_0 kernels run at only ~0.67–0.83 T on the same kind of count (16-bit loads, 64
registers in the plain kernel), so for the Q8_0 tensors this note uses the measured GB/s directly.

### Where the instructions go

Loop bodies of the plain kernels (my measurement):

| Per call | Q4_1 (32 weights, 20 B) | Q5_K (16 weights, 11 B) | IQ4_XS (32 weights, 17 B) |
|---|---|---|---|
| Instructions | 102 | 115 | 133 |
| Per weight byte | 5.1 | 10.5 | 7.8 |
| Integer and address arithmetic (XMAD chains for ×176 / ×136 / ×36 strides, 64-bit carries) | 42 | 44 (38 %) | 35 (26 %) |
| Loads (+ waits) | 16 (+5) | 14 (+4): 9 × 32-bit, 5 × 16-bit | 16 (+2) |
| Weight unpack | 12 (nibble masks) | ~16: nibble masks plus fifth bits (shift, mask, merge) | **~44: 32 PRMT for the table, ~12 for indices** |
| Scale handling | 8 F2F | ~14 for the packed 6-bit scales and minimums (3 × 16-bit loads, branchless decode from [#26705](https://github.com/ggml-org/llama.cpp/pull/26705)), 8 F2F/I2F | ~6 for the 6-bit scale, 3 conversions |
| dp4a | 8 | 8, of which **4 recompute the activation sum** | 8 |
| Float | 8 | 7 | 2 |

Points worth knowing before writing a faster kernel:

- **Q4_K and Q5_K recompute each activation block's sum with `dp4a(0x01010101, u)` in int32** instead of reading the
  FP16 sum the activation block carries. That costs half their dp4a work, but it is why K-quant decode never overflows
  on Flint, whose down-projection inputs reach channel RMS 48,400 in layer 48. The compiler already shares these sums
  between gate and up (12 dp4a instead of 16 in the fused kernel) and across rows (20 instead of 32 in the 4-row
  small-K kernel). A new kernel should keep an integer or FP32 sum, computed once per activation chunk.
- **The 16-entry IQ4_XS table costs no memory loads**: the compiler keeps its four words in immediates and the
  constant bank, and `get_int_from_table_16` selects bytes with `__byte_perm`, 8 PRMT per 32-bit word of indices.
  sm_61 has no instruction that selects bytes by 4-bit index, so about 1.1–1.5 instructions per weight stay in any
  IQ4_XS kernel.
- **The superblock layout costs addressing, not bandwidth.** 176- and 136-byte superblocks are 16- and 8-byte aligned
  at Flint's row widths (rows of 1,408 and 1,088 bytes are multiples of 16), so a kernel that hands each lane whole
  superblocks can use 128-bit loads (64-bit for odd IQ4_XS superblocks) without any repacking.

## 4. Why the 2-warp patch helped this file most

Measured on 6 cards: UD-Q4_K_XL 17.56 tok/s on stock llama.cpp, 19.53 with CUDA graphs, **24.80 with 2 warps per block
at one token (+27 %)**, and 24.58 with 2 warps only for rows ≥ 2048 wide. For comparison the makers' Q4_1 gained
+13.4 % and S4-L0 +13.9 % (28.02 with the wide-rows variant) ([Q34](../../../quants/experiments.md)).

The model of section 3 splits UD-Q4_K_XL's gain (my calc):

| | 4 warps | 2 warps |
|---|---|---|
| Q8_0 matrix-vector products (measured GB/s per shape × bytes) | 21.2 ms | 15.8 ms |
| K/I-quant matrix-vector products (counted instructions ÷ calibrated issue rate) | 14.9 ms | 13.3 ms |
| Everything else (measured total minus the two lines above) | 15.1 ms | 11.2 ms |
| Measured per token | 51.2 ms | 40.3 ms |

Most of the gain is Q8_0: at K = 2048 a 4-warp block gives each thread only 2 Q8_0 calls, which the compiler runs
through its unrolled-loop remainder path; with 2 warps it gets 4 and the main loop. UD-Q4_K_XL has ~1.0 GB of Q8_0 at
K = 2048 (attention q/k/v, the head, shared-expert and dense gate/up), more than any other Flint file, and almost no Q8_0
down-projections, where 2 warps are slower (81 → 43 GB/s). S4-L0 has 314 MB of those, which is why it gains less.

The "everything else" line should not change between the builds, since the patch touches only MMVQ. It falls by
3.9 ms, and the same check on S4-L0 gives 13.0 → 9.8 ms. So in a real token the 4-warp kernels run ~10 % slower than
the microbenchmark shows, by about the same amount for both files; with 2 warps model and measurement agree for both
files (rest 11.2 ms with the rescale on, 9.8–10.1 ms for S4-L0 with it off). The projections below are therefore
calibrated on the 2-warp build.

## 5. What a Pascal kernel for K- and I-quants would cost

The tuned kernels in [bench/mmvq_pascal](../../../bench/mmvq_pascal/) (Q4_1, Q4_0, Q8_0, Q4_1_64) split a warp into
lane groups sized to K, give each group several rows that share one activation load, load 64–128 bits at a time,
compute row pointers once and reduce with shuffles. Measured on one card at Flint's shapes: Q4_1 1.75× (1.89× with
separate planes), Q8_0 1.52×, up to 159 GB/s against a 179 GB/s streaming ceiling
([Q25](../../../quants/experiments.md)). The same design applies to superblocks, with each lane taking one
superblock per row (8 lanes per row at K = 2048, 2 at K = 512).

Instruction budget per 256-weight superblock and row (my calc):

| | Q5_K (176 B) | IQ4_XS (136 B) |
|---|---|---|
| Weight loads | 11 × 128-bit (qs 8, fifth bits 2, header 1) | 9–17 (128-bit or 64-bit by alignment) |
| Unpack | 160: per 32-bit word, low nibbles + shifted fifth bit merged with one LOP3 (2), high nibbles (3) | 320: 10 per word with activations pre-permuted so the final reorder PRMTs go away (12 without) |
| dp4a | 64 | 64 |
| Scales and minimums | 55: all 16 six-bit values decoded 4 per word in 7 ops, then ~6 per sub-block | 42 |
| Activations (shared by 2 rows) | 17 | 33 (includes the permutation) |
| Addressing | 4 | 4 |
| **Per weight byte** | **~1.8** (tuned Q4_1: 1.6) | **~3.5** |

Both are well under the ~6.4 instructions per byte at which the card runs out of issue slots, so the kernels should be
limited by memory and latency, like the tuned Q4_1 variants at 2.6–5.8 instructions per byte (112–159 GB/s). The P100
fork that rewrote K-quant decode for its own hardware found the same ordering: in situ Q4_K 517, Q5_K 427 and IQ4_XS
346 GB/s ([qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp)). That fork's FP16 path can't be used here: the
P106's FP16 throughput is 1/64 of FP32.

Expected speed per type (my calc, claim to test):

| Tensors in UD-Q4_K_XL | MB per token | llama.cpp, 4 w / 2 w | Pascal kernel | Over the 2-warp build |
|---|---|---|---|---|
| Q5_K expert and shared-expert gate/up | 190 | 61 / 68 GB/s | 115–140 | 1.7–2.1× |
| IQ4_XS expert gate/up | 171 | 70 / 74 | 95–120 | 1.3–1.6× |
| Q5_K expert down (K = 512) | 125 | 40 / 39 | 110–135 | 2.8–3.5× |
| IQ4_XS expert down (K = 512) | 54 | 15 / 19 | 90–115 | 4.7–6× |
| Q5_K attention (7 layers) | 91 | 51–67 / 65–78 | 120–140 | 1.5–2.2× |
| Q5_K shared-expert down (K = 1024) | 17 | 49 / 49 | 100–115 | 2.0–2.3× |
| Q4_K, Q6_K, IQ3_S | 32 | 40–54 / 40–62 | 60–145 | ~2× |
| **All K/I-quants** | **680** | **46 / 51** | **106–131** | **2.1–2.5×** (13.3 → 5.2–6.4 ms) |
| Q8_0, for comparison (measured per shape) | 1,483 | 70 / 94 | 109 | 1.16× (15.8 → 13.6 ms) |

Q5_K and IQ4_XS together hold 95 % of the K/I-quant bytes; the last 32 MB can stay on llama.cpp's kernel for ~0.6 ms.

## 6. What upstream and other engines have done

Cited findings (P = primary source):

- **Pascal launch shape.** PR [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) (open since 2026-07-09, no
  review): 2 warps at one token for compute 6.1/6.2. GTX 1060 6GB, Llama-3.2-3B: Q4_K_M +5.4 %, Q5_K_M +3.2 %, Q6_K
  −0.3 %, Q4_0 +4.6 %, Q8_0 +2.5 %. On the rig the same change gives +13–27 % (measured), because the P106-090 has half
  the 1060's SMs and Flint's rows are short.
- **The small-K rule and IQ4_XS on Pascal.** In [#20905](https://github.com/ggml-org/llama.cpp/pull/20905) the
  maintainer specified the rule as "On NVIDIA Pascal or older disable for iq3_s, q2_k, and q3_k"
  ([comment](https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4129210860), 2026-03-25), and his own P40
  run of the original small-K commit showed IQ4_XS +3 % at one token on Qwen3-30B-A3B (IQ4_NL +6 %, Q4_K_S +7 %, Q5_K_S
  +5 %, IQ3_S −19 %; [comment](https://github.com/ggml-org/llama.cpp/pull/20635#issuecomment-4104332920), 2026-03-21).
  The merged condition applies the list meant for "all other GPUs" to Pascal as well, so IQ4_XS (and IQ1/IQ2/IQ3_XXS)
  lose small-K at one token (P, source at `d77256999`). Enabling it is expected to save ~0.9 ms per UD-Q4_K_XL token
  in the 2-warp build (+2.3 %) and ~1.9 ms in the 4-warp build (my calc).
- **Small-K everywhere on a small GPU.** [#30858](https://github.com/ggml-org/llama.cpp/pull/30858) (open, 2026-10-20)
  uses small-K for every K at one token on Jetson Orin: tg128 Q4_K_M +15 %, Q4_K_S +18 %, Q8_0 +23 %, Q4_0 +38 % on an
  Orin Nano. More rows per block means more loads in flight per thread, the same effect the rig shows.
- **Narrow matrices.** [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) (open, limited to RDNA3/4 after
  review): nwarps sized to K for MoE experts, Qwen3.5-35B-A3B +9–10 % on RDNA4. [#30788](https://github.com/ggml-org/llama.cpp/pull/30788)
  (open, 2026-10-19): power-of-two lane groups and several rows per warp for small-K NVFP4, the structure section 5
  proposes for K-quants.
- **Activation layout for K-quants.** HIP PRs [#22821](https://github.com/ggml-org/llama.cpp/pull/22821) (closed) and
  [#23685](https://github.com/ggml-org/llama.cpp/pull/23685) (open) store four Q8_1 blocks together so K-quant MMVQ
  reads activations with coalesced wide loads: Qwen3-30B-A3B Q4_K_M tg128 +36 %, Llama-3.2-3B Q5_K_M +21 % on a Radeon
  Pro W7800.
- **Reordered layouts.** The SYCL backend repacks K-quants at load time for its MMVQ (Q5_K reorder layout and fused GLU,
  [#29375](https://github.com/ggml-org/llama.cpp/pull/29375), merged 2026-10-09; Q4_K row pairing,
  [#30226](https://github.com/ggml-org/llama.cpp/pull/30226)).
- **Other K-quant MMVQ tuning.** [#26705](https://github.com/ggml-org/llama.cpp/pull/26705) (merged 2026-09-07)
  made the Q4_K/Q5_K scale decode branchless (in the rig's build); [#30517](https://github.com/ggml-org/llama.cpp/pull/30517)
  (merged 2026-10-17) gives 1–3-bit types two rows per block (Q2_K/Q3_K 1.6–1.7× on an RTX 3090), not Q4_K/Q5_K or
  IQ4_XS; [#24386](https://github.com/ggml-org/llama.cpp/pull/24386) tunes RDNA4 warps for Q4_K/Q6_K.
- **P100 fork** ([qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp), updated 2026-10-16; S/P): 4 rows per
  block for K-quants, +33 % tg on Llama-3.1-8B Q4_K_M; a full FP16 rewrite of batch-1 decode, 2.03×. sm_60 has no dp4a
  and fast FP16, the opposite of sm_61, so only the row-sharing part transfers.
- **Vulkan backend on the same chip.** [#19817](https://github.com/ggml-org/llama.cpp/issues/19817) (GTX 1060, b8130):
  Vulkan tg128 beat CUDA on both K-quant files tested, Granite-4.0-H-Tiny Q4_K_M 90.6 against 61.7 tok/s (+47 %) and
  Falcon-H1-7B Q4_K_S 28.1 against 25.4 (+11 %). The rig's Vulkan builds exist ([guide](../../vulkan_on_the_rig.md))
  but have not been run on UD-Q4_K_XL.
- **Activation sums.** [#22571](https://github.com/ggml-org/llama.cpp/pull/22571) (open since 2026-05-01) stores the
  Q8_1 sums as BF16 so they can't overflow, which would make Flint's rescale unnecessary for every format.
  [#31019](https://github.com/ggml-org/llama.cpp/pull/31019) (open, 2026-10-23) fixes a different FP16 overflow, in the
  `m · s` product of Q4_1/Q5_1 on GPUs with fast FP16; compute 6.1 already multiplies in FP32.
- **Engines outside llama.cpp.** None decodes GGUF K-quants faster on sm_61: ik_llama.cpp's own types are lookup or
  trellis types and it needs AVX2 on the host; vLLM's GGUF kernels (which its Pascal forks run) are an older copy of
  llama.cpp's MMVQ with the same costs; Marlin-family and tensor-core kernels don't run on Pascal ([quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md),
  [engines_other.md](engines_other.md)).

## 7. End-to-end estimates

All figures are llama-bench tg128 on 6 cards (my calc unless marked measured). The non-matrix-vector time is taken from
the 2-warp build: 11.2 ms for UD-Q4_K_XL (rescale on), 10.1 ms for S4-L0 (rescale off). Q4_1 and Q8_0 kernel speeds are
the measured per-shape bests of the microbenchmark; K/I-quant and Q5_0 kernel speeds are the estimates above.

**UD-Q4_K_XL (Unsloth's file, unchanged):**

| Engine | ms per token | tok/s |
|---|---|---|
| Stock llama.cpp (measured) | 56.9 | 17.6 |
| + CUDA graphs on Pascal (measured) | 51.2 | 19.5 |
| + 2 warps per block at one token (measured) | 40.3 | 24.8 |
| + small-K for IQ4_XS | 39.4 | 25.4 |
| + Pascal Q8_0 kernel only | 38.1 | 26.2 |
| + Pascal Q5_K/IQ4_XS kernels only | 32.2–33.4 | 30–31 |
| + both | 30.0–31.2 (30.0–32.7 if the K/I-quants already run 1.5 ms faster than modelled) | **31–33** |
| + rescale removed by FP32/BF16 activation sums (−2.4 to −2.8 ms) | 27.2–30.3 | **33–37** |

**S4-L0 (this project's file at the same quality, 32.5 GB):**

| Engine | ms per token | tok/s |
|---|---|---|
| CUDA graphs, rescale off (measured) | 43.3 | 23.1 |
| + 2 warps (measured) | 38.0 | 26.3 |
| + 2 warps only for rows ≥ 2048 (measured) | 35.7 | 28.0 |
| + Pascal Q4_1 and Q8_0 kernels (measured per shape) and a Pascal Q5_0 kernel (100–125 GB/s, estimate) | 29.3–30.1 | **33–34** |

Uncertainties:
- The K/I-quant time in the 2-warp build may be ~1.5 ms lower than modelled: the two files' residuals differ by 1.1 ms
  where the rescale alone should account for 2.4–2.8. That narrows UD-Q4_K_XL's kernel gain to the lower end.
- In a real token the 4-warp kernels ran ~10 % slower than the microbenchmark predicts (section 4). If the tuned
  kernels lose the same in a real token, every "tuned" figure above drops by up to ~2 tok/s, for both files alike.
- Quality is unchanged by any of these kernels; only the summation order changes.

## 8. Which route, and what to run

**The better investment is the engine, in this order:**

1. **Port the Pascal Q8_0 kernel** (measured 1.52×). Q8_0 is 67 % of UD-Q4_K_XL's bytes and 64 % of S4-L0's, so both
   routes need it first.
2. **Fix the small-K rule for IQ4_XS on Pascal** (one line in `should_use_small_k`; ~+2 % on UD-Q4_K_XL, claim to
   test). Upstreamable as is: it brings the code in line with the maintainer's stated rule and his P40 numbers.
3. **Pascal kernels for Q5_K and IQ4_XS**, in the bench first. Gate: tuned Q5_K ≥ 110 GB/s and IQ4_XS ≥ 90 GB/s at
   Flint's gate/up and down shapes on one card. If they pass, UD-Q4_K_XL reaches S4-L0's speed at Unsloth's quality in
   a 25 GB file, for everyone who already has it.
4. **FP32 activation sums** (or BF16 as in #22571), which remove Flint's rescale for every file: +8–10 % on
   UD-Q4_K_XL once the kernels are in. Switching the rescale off without them is exact for UD-Q4_K_XL's decode (none
   of its decode kernels reads the FP16 sum) but not for its prompt processing, where the Q5_K down-projection of
   layer 46 does.

Why not the Q5_0 kernel for S4-L0 instead of step 3: both end at ~33–35 tok/s, and S4-L0's remaining kernel (Q5_0,
406 MB per token) is as unmeasured as Q5_K and IQ4_XS. K/I-quant kernels reach every user of the standard files,
including Unsloth's Qwen3.6 files and the rig's Ornith and Qwen3.8-distill Q4_K_M files (Q4_K and Q6_K follow the
same design as Q5_K), while a custom file reaches only people who find it. S4-L0 remains the fastest option at the quality bar until step 3 is done, and the
fallback if the bench gate fails.

Cheap measurements that sharpen these numbers, all on the rig with existing builds:
- **Vulkan A/B on UD-Q4_K_XL** (an hour; the builds exist): #19817 shows Vulkan ahead on K-quant files on the same chip.
- **nvprof of one UD-Q4_K_XL token per kernel** (1–2 h, [quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md)
  E1): measured time per K/I-quant kernel replaces the issue-rate model of section 3, and shows where the 4-warp
  build's extra ~4 ms go.
- **`test-backend-ops -o MUL_MAT,MUL_MAT_ID perf` for q5_K and iq4_xs** at K = 512 and 2048 (minutes): per-kernel
  GB/s for the 4- and 2-warp builds.

Qwen3.6-35B-A3B has the same 2048/512 shapes, so every kernel number here carries over to its K-quant files.

## Appendix: path counts

Warp-instructions per thread block along the path taken at Flint's shape (my measurement from the sm_61 SASS of the
rig's `llama-bench`; `mul_mat_vec_q<type, 1 column, fusion, small_k>`). P = prologue up to the loop, H = loop set-up,
B = loop body per iteration (gate and up together in fused kernels), E0 / E1 = epilogue of warp 0 / of the other
warps (shared-memory store, barrier; warp 0 also reduces, applies SwiGLU and writes). Idle warps run P + E1.

| Kernel | Registers | P | H | B | E0 | E1 | Path per block |
|---|---|---|---|---|---|---|---|
| Q5_K fused gate/up, 4 w | 61 | 56 | 64 | 192 | 137 | 15 | 4 warps × 1 iteration = 1,430 |
| Q5_K fused gate/up, 2 w | 61 | 56 | 64 | 192 | 111 | 15 | 2 warps × 2 iterations = 1,134 |
| IQ4_XS fused gate/up, 4 w | 56 | 56 | 56 | 231 | 117 | 17 | 2 working + 2 idle warps = 966 |
| IQ4_XS fused gate/up, 2 w | 56 | 56 | 56 | 231 | 102 | 17 | 2 warps × 1 iteration = 805 |
| Q5_K down small-K, 4 w (4 rows) | 78 | 45 | 58 | 450 | 101 | 9 | 1 working + 3 idle = 816 |
| Q5_K down small-K, 2 w (2 rows) | 64 | 44 | 80 | 191 | 50 | 7 | 1 working + 1 idle = 416 |
| IQ4_XS down, 4 w (1 row, 16 lanes work) | 47 | 41 | 77 | 133 | 36 | 5 | 1 working + 3 idle = 425 |
| IQ4_XS down, 2 w | 47 | 41 | 77 | 133 | 32 | 5 | 1 working + 1 idle = 329 |
| IQ4_XS down small-K, 2 w (2 rows) | 69 | 44 | 72 | 223 | 50 | 7 | 1 working + 1 idle = 440 |
| Q5_K plain, 4 w | 41 | 41 | 54 | 115 | 36 | 5 | K = 2048: 891; K = 4096: 1,351 |
| Q5_K plain, 2 w | 41 | 41 | 54 | 115 | 33 | 5 | K = 2048: 688; K = 4096: 1,148 |
| Q4_1 fused gate/up, 4 w / 2 w | 32 | 53 | 60 | 84 | 118 / 104 | 15 | 951 / 681 |
| Q4_1 down small-K, 4 w / 2 w | 56 / 40 | 45 / 43 | 44 / 64 | 182 / 83 | 91 / 43 | 9 / 7 | 524 / 283 |
| Q5_0 fused gate/up, 4 w / 2 w | 48 | 49 | 60 / 48 | 163 | 118 / 104 | 15 | 1,251 / 965 |

Lane-instructions per byte = path per block × 32 ÷ weight bytes per block. Block bytes: Q5_K gate/up 2 × 2048 × 176/256;
IQ4_XS gate/up 2 × 2048 × 136/256; down-projections rows × 512 × bytes per weight.

## Sources

- llama.cpp at the rig's commit `d77256999` (2026-10-22): [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/mmvq.cu),
  [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/vecdotq.cuh),
  [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/mmq.cuh) (Q4_K/Q5_K use the
  `DS4` activation layout with FP16 sums in prompt processing; Q8_0, Q6_K and IQ4_XS use `D4`),
  [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-common.h) (block layouts).
- llama.cpp PRs and issues: [#25479](https://github.com/ggml-org/llama.cpp/pull/25479),
  [#20635](https://github.com/ggml-org/llama.cpp/pull/20635), [#20905](https://github.com/ggml-org/llama.cpp/pull/20905),
  [#20831](https://github.com/ggml-org/llama.cpp/pull/20831), [#30858](https://github.com/ggml-org/llama.cpp/pull/30858),
  [#30788](https://github.com/ggml-org/llama.cpp/pull/30788), [#22821](https://github.com/ggml-org/llama.cpp/pull/22821),
  [#23685](https://github.com/ggml-org/llama.cpp/pull/23685), [#29375](https://github.com/ggml-org/llama.cpp/pull/29375),
  [#30226](https://github.com/ggml-org/llama.cpp/pull/30226), [#26705](https://github.com/ggml-org/llama.cpp/pull/26705),
  [#30517](https://github.com/ggml-org/llama.cpp/pull/30517), [#24386](https://github.com/ggml-org/llama.cpp/pull/24386),
  [#22571](https://github.com/ggml-org/llama.cpp/pull/22571), [#31019](https://github.com/ggml-org/llama.cpp/pull/31019),
  [#19817](https://github.com/ggml-org/llama.cpp/issues/19817).
- [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) README (P100 K-quant decode, per-type in-situ GB/s).
- This repository: [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) (tuned kernels and their rig results),
  [quants/experiments.md](../../../quants/experiments.md) (Q25 microbenchmark, Q34 launch shapes, Q36 S4-L0),
  [quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) (static SASS counts per format, issue-bound
  model), [flint_speedups.md](flint_speedups.md) (the MoE down-projection rescale),
  [scripts/gguf_mix.py](../../../scripts/gguf_mix.py), [scripts/token_cost.py](../../../scripts/token_cost.py).
