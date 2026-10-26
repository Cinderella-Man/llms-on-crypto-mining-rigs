# Survey B: new weight formats, memory layouts and CUDA kernels for single-token decode on a 5-SM Pascal card (as of 2026-10-23)

Scope: class B levers, meaning weight formats or memory layouts that need new or changed llama.cpp CUDA kernels, for single-token decode (MMVQ, including `MUL_MAT_ID` for 6 of 256 experts) on the P106-090 (sm_61, 5 SMs, ~173 GB/s). Kernel-only changes for the existing formats are included, because they set the baseline any new format has to beat.

Conventions:
- **P** = primary source (PR, issue, source code, paper, model card); **S** = secondary (blog, forum, Reddit).
- **(my calc)** marks this note's arithmetic. **claim to test** marks anything not measured on the rig.
- **(my measurement)** marks a static analysis of the rig's own binary. Method:
  - The sm_61 cubins were extracted from the llama-bench binary that runs on the rig (`llama.cpp-20261022-d77256999-graphs61`).
  - They were disassembled with `cuobjdump -sass` and `cuobjdump -res-usage` from the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image.
  - Instructions were counted per kernel and per loop body.
  - These are static counts. Dynamic per-thread counts are estimated by following the code path for a given K (section 1).
- Source links point to the rig's commit `d77256999` (2026-10-22) unless another date is given.

Starting point (brief and [flint_speedups.md](flint_speedups.md)):
- The makers' Q4_1 file runs at 27.9 tok/s in llama-bench tg128 (35.8 ms per token on 5 cards with CUDA graphs) and 24.3 tok/s in llama-server with `-bs`.
- Per token it reads 1.66 GB. That splits into ~23–26 ms of weight reading and ~10–13 ms of fixed cost: ~1,900 launches, of which ~470 come from the MoE down-projection rescale.
- Single-card efficiency on Llama-2-7B, measured on build `85d5703a3`: Q4_0 42 %, Q4_K_M 35 %, Q8_0 54 %. The Q8_0 run used 2 cards, which relieves the 75 W power cap, so the cores ran at ~1830 MHz against 1710–1760 MHz on one card.

---

## 1. Executive summary

1. **On this card, decode cost follows instructions, not bytes (my measurement).** The MMVQ kernels in the rig's binary issue 3.2–8.4 SASS instructions per weight inside their main loop. A thread working on Flint's 2048-wide rows makes only one loop iteration, so it needs ~15 instructions per weight byte (Q4_1).
   - Five SMs at ~1.73 GHz can issue 6.4 lane-instructions per byte of a 173 GB/s stream. Flint's Q4_1 matvecs need ~20.7 G lane-instructions per token, which takes 18.7 ms at 100 % issue efficiency and 21–25 ms at 76–90 %.
   - That matches the ~23–26 ms the earlier note attributes to "weight reading". The card is issue-bound; neither dp4a nor DRAM is the limit.
2. **Address arithmetic dominates the loop, not unpacking.** dp4a is 5–8 % of the loop, and nibble unpacking 0–12 % for Q4_0/Q4_1/Q8_0.
   - Integer and address instructions are 41–59 %: Pascal has no full-rate 32-bit IMAD, so the 18/20/34/36-byte block strides become chains of 16-bit XMAD multiplies with 64-bit carries.
   - Loads are another 16–19 %. Every MMVQ kernel on sm_61 uses only 8-, 16- or 32-bit loads, with no 64- or 128-bit vector loads.
   - The activation (Q8_1) block is re-read for every output row.
3. **Instruction counts reproduce the measured format ranking (my measurement).** Loop instructions per byte: Q8_0 4.85 < Q4_1 5.1 < Q4_0 6.2 < Q6_K 7.2 < IQ4_XS 7.8 < Q5_0 8.7 < IQ4_NL 9.0 < Q4_K 10.6 < MXFP4 11.3 < NVFP4 14.9.
   - The Q4_0/Q8_0 ratio (1.28) matches the measured efficiency ratio (54 %/42 % = 1.29; about 1.22 after the Q8_0 run's clock advantage).
   - The 10-SM P106-100 runs the same Q4_0 file 1.56–1.57× faster on the same memory. That is direct evidence that issue capacity, not bandwidth, sets the speed.
4. **The cheapest lever is the launch shape (no format change).** Pascal uses the generic MMVQ table: 4 warps per block and 1 row per block at one token.
   - With Flint's K = 2048, each thread does one iteration, so the prologue and epilogue cost about as much as the dot product.
   - For the expert down-projection (K = 512), 3 of 4 warps in each block idle.
   - Open PR #25479 (2 warps on Pascal, +2.5–5.4 % on a GTX 1060) plus a Pascal version of PR #20831's narrow-matrix nwarps should help Flint more than the GTX 1060's model: **+5–20 % (29–33 tok/s), claim to test, about an hour of work.**
5. **A Pascal-specific MMVQ kernel for the existing Q4_0/Q4_1/Q5_x/Q8_0 blocks is the main class-B lever.** It would put several rows in each warp so they share activations, give each lane several iterations, strength-reduce the addressing, and skip idle warps.
   - Expected: **+20–45 % (34–40 tok/s)**, anchored on the P106-100's 1.56× from twice the SMs; up to +55 % if it becomes bandwidth-bound.
   - It is exact, and it helps the makers' Q4_1, UD-mirror v1 and Qwen3.6 alike.
   - Pascal evidence is indirect: a P100 profile found MMVQ instruction-issue-bound and 4 rows per block gave +33 % there; 2 rows were neutral on a P40, a card with 3× more SM per byte.
6. **Load-time repacking to a structure-of-arrays (SoA) layout is the format-level step after that.** Nibbles go in a 16-byte-aligned plane and scales in a separate plane, read with 128-bit loads and power-of-two strides.
   - Precedent: llama.cpp's SYCL backend reorders Q4_0, Q8_0 and K-quants exactly this way on Intel GPUs (Llama-2-7B Q4_0 42 → 55 tok/s on an Arc 770; Q8_0 3.1× on an Arc Pro B70).
   - CUDA attempts exist only for big GPUs (NVFP4, MXFP4), where TG did not improve. A maintainer asked for a non-repacked baseline first.
   - On top of item 5: **+5–15 %, so 38–49 tok/s combined (claim to test).**
7. **The makers' Q4_1 file is exactly a group-64 file (verified).** Every linear layer in RecursiveSI/flint-int4 is MLX affine INT4 with group 64 and BF16 scales and biases. The router is BF16 and the embedding 8-bit.
   - In the GGUF, each group-64 (scale, bias) is stored as two consecutive Q4_1 blocks with identical (d, m), equal to the MLX values. The BF16→FP16 conversion was lossless on 128 groups each of attn_q, ffn_gate_exps, ffn_down_exps and output.
   - A group-64 layout at 4.5 bits per weight (bpw) therefore holds the makers' codes losslessly with 10 % fewer bytes (1.66 → ~1.50 GB per token). It can be done at load time (repack plus pair-compaction) or as a new GGUF type.
   - It only pays once the kernel is no longer issue-bound: +2–4 % with today's kernel style, ~+10 % on top of items 5–6.
8. **FP32 sums in the activation blocks are a small kernel change that removes the reason for the Flint rescale.** With them, the rescale (~470 launches, +7–15 % per the earlier note) can be dropped for every weight format, exactly. Reusing one Q8_1 quantization for Q/K/V and for routed plus shared gate/up removes ~145 more launches (+2–3 %). No upstream PR does either for CUDA yet; ik_llama.cpp reuses the quantization.
9. **The makers' NVFP4 file is predicted to be slower than their Q4_1 on this rig: 17–20 tok/s against 27.9 (claim to test).** NVFP4's dp4a path needs 8.4 instructions per weight against Q4_1's 3.2, because of the table lookup and a software FP8-scale conversion. It reads 13 % fewer bytes, which doesn't help an issue-bound kernel.
   - MXFP4 is similar: byte loads plus lookup, 6.0 instructions per weight.
   - A speed test of the NVFP4 file is worth running only as a cheap test that discriminates between the "issue-bound" and "bandwidth-bound" models.
10. **Fusing more of the MoE chain gives small returns per step.** Gate+up+SwiGLU is already one kernel, and the weighted expert sum is fused (#25952). What remains:
    - folding the down projection into the expert sum;
    - letting the producer op emit Q8_1 (closed PR #22710 claimed +3–5 % TG on an RTX 5090);
    - a full MOE_FFN op (#26322, open, prefill gains only).
    - Expected +2–5 %. A persistent per-layer decode kernel, as in vLLM's 7 kernels per MoE layer, could reach +25–40 %, but that is weeks of work with no Pascal precedent.
11. **ik_llama.cpp's types don't help here.** Its row-interleaved `_R4`/`_R8` types have no CUDA kernels; ik's own CUDA proof of concept for Q4_0_R4 matched Q4_0's TG speed. IQ4_KS/KSS are lookup-table types like IQ4_XS. The trellis types (KT) are ALU-heavy, which is the wrong direction for an issue-bound card. ik also needs AVX2 on the host.
12. **Outside llama.cpp, nothing is directly usable on sm_61.**
    - Marlin is sm_80+ (sm_75 since vLLM #29901), Machete sm_90. FLUTE and BitBLAS need tensor cores, or have sm_70 as their lowest tested target. T-MAC is CPU-only.
    - No measured Pascal numbers were found for MLC-LLM. ExLlamaV2 computes in FP16 (~1.2 tok/s for a 34B on a P40). The vLLM Pascal forks refuse Marlin and run GGUF kernels.
    - Ideas that transfer: Marlin's offline shuffle for 128-bit loads with static offsets; Microsoft BitNet's dp4a W2A8 GEMV (one LOP3 per 4 weights, 128-bit activation loads); ExLlamaV3's dp4a int8 GEMV.
13. **A new GGML type is a ~600-line job, with low odds upstream.** Q2_0 took 262 lines for CPU and 326 for CUDA.
    - CONTRIBUTING.md requires a small HF model, PPL/KLD against BF16, and pure-CPU llama-bench numbers.
    - A maintainer opposes new formats unless the author is an established expert, and asked for a non-repacked baseline before any repacking.
    - A private fork is realistic; upstreaming is better aimed at the kernel retunes (items 4, 5, 8).

**What a Q8_0-level format would buy.** At 54 % efficiency (93 GB/s) and 4.5 bpw, Flint's matvecs would take 1.50 GB / 93 GB/s = 16.1 ms instead of 23–26 ms.
- Total: 26–29 ms, so **34–38 tok/s** (+23–37 %).
- With the rescale gone as well (fixed cost 5.3–10.5 ms): 21–27 ms, so **38–47 tok/s**.
- At 75 % (130 GB/s, the bandwidth-bound regime items 5–6 aim for): 11.5 ms + 10–13 ms = 22–25 ms, so 40–46 tok/s; with the rescale gone, 17–22 ms, so 45–60 tok/s (my calc; nothing measured).

---

## 2. Lever table

T = F + M, where F ≈ 10–13 ms is the fixed per-token cost (rescale 2.5–4.7 ms of it) and M ≈ 23–26 ms is the matvec time. Today's 1,900 launches and 20.7 G lane-instructions are my estimates (section 3). Baseline is 27.9 tok/s in llama-bench tg128 on 5 cards. Served speed is about bench speed minus 5.4 ms per token.

| Lever | Class | Mechanism | Expected effect on single-stream decode (arithmetic) | Quality effect | Effort | Evidence strength | Carries over to Qwen3.6? |
|---|---|---|---|---|---|---|---|
| L1. Pascal launch shape: 2 warps at 1 token (#25479) and no idle warps for small K | B, kernel only | Gives each thread 2 iterations at K = 2048; removes 3 idle warps per block in the expert down-projection | Static instructions −28 % (2 warps −4.2 G; small-K −1.6 G of 20.7 G). M −2.5 to −7 ms → **29–33 tok/s (+5–20 %)** | Exact (summation order changes) | 1–2 h with existing PRs | GTX 1060 measured +2.5–5.4 % (#25479); RDNA +5–10 % for MoE (#20831); rig untested | Yes: same K = 2048/512 shapes |
| L2. Pascal MMVQ kernel for the existing block formats | B, kernel only | 2–4 rows per warp sharing activation loads, several iterations per lane, strength-reduced 32-bit addressing, mask-only nibble unpack | Instructions 2–3× fewer → M ≈ 13–17 ms → **34–40 tok/s (+20–45 %)**; up to 43 if bandwidth-bound at 75 % | Exact | 3–7 days (fork) | P100: issue-bound, +33 % from 4 rows (qr0n); P40 neutral for 2 rows (ik #2701); P106-100/P106-090 = 1.56× | Yes |
| L3. Load-time SoA repack (+ group-64 compaction of the makers' Q4_1) | B, layout at load; GGUF unchanged | Quants plane plus scales plane, 16-byte aligned, 128-bit loads, power-of-two strides; merges the identical Q4_1 block pairs | On top of L2: M ≈ 10–13 ms (1.50 GB at 65–85 %) → **38–49 tok/s**; +5–15 % over L2 | Exact (makers' file); exact repack for others | +3–5 days; MMQ (prefill) must read the layout too | SYCL reorder on Intel: Q4_0 +31–87 %, Q8_0 3.1×; CUDA PoCs on Blackwell: TG flat | Yes (no compaction for Qwen3.6 Q4_0) |
| L4. New GGUF type "Q4_1 group 64" (4.5 bpw, affine) | B, new type | 36-byte block or SoA; holds Flint's QAT codes exactly | Alone with today's kernel style +2–4 % (fewer scale loads); with L2/L3 same as L3's compaction, plus file 26.0 → 23.4 GB | Exact for the makers' Q4_1; for BF16-derived weights between Q4_0 and Q4_1 (claim to test) | 1–2 weeks (CPU + CUDA + converter) | No precedent; the closest (HQQ Q4_H) was closed as not worth it | No (no group-64 QAT) |
| L5. Q8_1 activations with FP32 sums; quantize reuse across Q/K/V and routed/shared gate/up | B, kernel only | Sum stored as FP32 (or ×1/32), so the overflow is gone; one quantize kernel per shared input | Rescale removable for all types: F −2.5–4.7 ms (+7–15 %); reuse −145 launches ≈ −0.7–1.0 ms (+2–3 %) → **31–33 tok/s** alone | Exact | 1–2 days | Overflow mechanism measured (#30704); BF16-sum PR #22571 open; ik reuses Q8_1 | Reuse yes; the rescale isn't used for Qwen3.6 |
| L6. Cheaper 5-bit bit order and unpack tricks | B, new layout or kernel | Q5_K-style high-bit order for Q5_0 (unpack 56 → ~20 per 32 weights); dp4a on `x & 0xF0F0F0F0` as unsigned bytes | Q5_0 loop −17 %; Q4 loops −3 %. Matters for UD-mirror v1's Q5_0 tensors (+3–8 % on a Q5_0-heavy mix, claim) | Exact | 2–4 days | P40: Q5_0 is the slowest legacy type per byte (#20831 comment) | Partly |
| L7. Fuse more of the MoE chain | B, kernel only | down + weighted sum; Q8_1 emitted by norm/SwiGLU/attention; MOE_FFN op | −140 to −235 launches → +2–5 %; a persistent per-layer kernel could reach +25–40 % | Exact | Days; weeks for the persistent kernel | #22710 claimed +3–5 % TG on an RTX 5090; #26322 prefill only; vLLM 7 kernels per layer on Blackwell | Yes |
| L8. Makers' NVFP4 file / MXFP4 | A (file); B to fix kernels | dp4a + `__byte_perm` lookup; NVFP4 FP8 scales converted in software | Predicted **17–20 tok/s (−30–40 %)** for NVFP4; MXFP4 1.3–1.9× Q4_0's loop cost | NVFP4 is QAT ("within 0.7 points" of 16-bit) | Test: copy 22.9 GB + 10 min | Static analysis only; no Pascal NVFP4 numbers anywhere | NVFP4 file is Flint-only |
| L9. ik_llama.cpp types (IQ4_KS/KSS/KT, _R4/_R8) | B (other engine) | Lookup or trellis decode; row interleave | No gain expected; CPU host lacks AVX2 | — | — | ik's Q4_0_R4 CUDA PoC = Q4_0 speed (RTX 4080) | No |
| L10. External kernels (Marlin, Machete, FLUTE, LUT-GEMM, BitBLAS, MLC, ExLlama, vLLM-Pascal) | — | Tensor-core or FP16 designs | Not runnable or not faster on sm_61; ideas feed L2/L3 | — | — | None on sm_61 | — |

---

## 3. Why fast formats are fast here: the decode cost model

### Takeaway
- **Where the instructions go.** On the P106-090 the MMVQ kernels are bound by instruction issue.
  - Inside the loop, Q4_1 needs 102 instructions per 32 weights and Q8_0 165.
  - Only 8 are dp4a. Address arithmetic and loads make up 57–79 %.
  - With Flint's 2048-wide rows, each thread makes one iteration, so per-block setup and the reduction roughly double the cost per byte.
- **Fast versus slow formats.** "Fast formats" are fast because they have the fewest instructions per weight, mostly through cheaper loads and addressing, not cheaper unpacking.
  - Q4_1's 4-byte-aligned 20-byte blocks beat Q4_0's 2-byte-aligned 18-byte blocks per weight.
  - Q8_0 wins per byte because it has twice the bytes per instruction.
- **Memory-level parallelism (MLP) is marginal for the same reason.** Each thread has 10–20 bytes of weight loads outstanding and spends most of its short life issuing instructions.

### Cited findings

**How MMVQ runs on sm_61** (source at `d77256999`, P):
- [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/mmvq.cu): Pascal falls through to `MMVQ_PARAMETERS_GENERIC`.
  - At one token: `calc_nwarps` returns 4 warps (128 threads) and `calc_rows_per_block` returns 1 row.
  - The "small K" variant sets rows per block equal to nwarps (4). Its trigger is `blocks_per_row_x < nwarps × blocks_per_iter_1warp`.
  - Two rows per block at one token is allowed only for Q1_0, Q2_K, Q3_K, IQ1_S and IQ1_M. The code comment says a second row "shares the q8_1 loads and adds memory parallelism".
  - `MUL_MAT_ID` at one token uses the same kernel with grid.y = the 6 expert slots; each block first reads `ids[]`, then computes its weight addresses.
  - From 2 to 8 tokens a dedicated `mul_mat_vec_q_moe` kernel runs, with one warp per token and 2 rows per block. MMVQ is used for `MUL_MAT_ID` up to 6 tokens for Q4_x/Q5_x/IQ4_NL and 4 for Q8_0/MXFP4/NVFP4 (`get_mmvq_mmid_max_batch_pascal_older`).
- Loop shape:
  - Thread `tid` starts at block `tid/(qi/vdr)` and strides by `vdr·nwarps·32/qi` blocks.
  - For Q4_0/Q4_1 (qi = 4, vdr = 2), 128 threads cover 64 blocks = 2,048 weights per iteration. **A K = 2048 row is exactly one iteration per thread.**
  - For K = 512, threads 32–127 get no work but still run the prologue, the shared-memory store and `__syncthreads`.
- [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/vecdotq.cuh) and [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-common.h). Per format:
  - **Q4_0**: 18-byte block (fp16 d + 16 B), 2-byte aligned.
    - `get_int_b2` assembles each 32-bit word from two 16-bit loads.
    - A call (2 threads per block, 16 weights) unpacks with `x & 0x0F0F0F0F` and `(x >> 4) & 0x0F0F0F0F` and does 4 dp4a.
    - The −8 offset is applied through the Q8_1 block sum: `d4·(sumi·d8 − 8·s8)`.
  - **Q4_1**: 20-byte block (half2 d, m + 16 B), 4-byte aligned; 32-bit loads; same unpack. The min term uses the Q8_1 sum.
  - **Q5_0 / Q5_1**: the 5th bits go in with 4 × (shift, AND, OR) per 32-bit word. Both use the sum.
  - **Q8_0**: 34-byte block, 2-byte aligned, 16-bit loads; 2 dp4a per call (8 weights); scale only, no sum.
  - **IQ4_NL / IQ4_XS**: 16-entry table through `__byte_perm` (8 PRMT per word in `get_int_from_table_16`). IQ4_XS adds a 6-bit scale per 32. No sum.
  - **Q4_K / Q5_K**: per 16 weights, 4 dp4a for the dot product and 4 dp4a with 0x01010101 that recompute the activation sum. Packed 6-bit scales and mins (made branchless in [#26705](https://github.com/ggml-org/llama.cpp/pull/26705), merged 2026-09-07, after the calibration build).
  - **MXFP4**: 17-byte block, 1-byte aligned. `get_int_b1` builds each word from 4 byte loads, then the table lookup; E8M0 scale. No sum.
  - **NVFP4**: 36 bytes per 64 weights (4 UE4M3 scales + 32 B). VDR 4 (32 weights per call). Table lookup; `ggml_cuda_ue4m3_to_fp32` goes through `__nv_fp8_e4m3` ([common.cuh](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/common.cuh)), which sm_61 lacks in hardware. No sum.
- [quantize.cu](https://github.com/ggml-org/llama.cpp/blob/d77256999/ggml/src/ggml-cuda/quantize.cu) `quantize_q8_1`:
  - One thread per value; warp reductions of amax and sum; `ds = make_half2(d, sum)`, where the sum is the FP32 sum of the unquantized values, stored as FP16.
  - Each `ggml_cuda_mul_mat_vec_q` call quantizes its own src1 with a separate launch. There is no reuse across calls.
- `FAST_FP16_AVAILABLE` is undefined for `__CUDA_ARCH__ == 610`, so scale products run in FP32 after `F2F` conversions.

**Static SASS of the rig's binary** (my measurement; `mul_mat_vec_q<type, 1 column, no fusion>`; counts per 32 weights in the main loop body; "int/addr" = XMAD/IADD/ISCADD/SHL/ISETP/MOV; "bit" = LOP/SHR/PRMT):

| Type | Bytes / 32 w | Instr / 32 w | Instr / byte | Instr / weight | Loads (widths) | dp4a | bit ops | int/addr | float + convert | Regs → occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4_0 | 18 | 112 | 6.2 | 3.5 | 20 (half 16-bit) | 8 | 12 | 53 (47 %) | 14 (8 F2F/I2F) | 48 → 62.5 % |
| Q4_1 | 20 | 102 | 5.1 | 3.2 | 16 (32-bit) | 8 | 12 | 42 (41 %) | 18 | 40 → 75 % |
| Q5_0 | 22 | 192 | 8.7 | 6.0 | 24 | 8 | 56 | 82 | 14 | 38 → 75 % |
| Q5_1 | 24 | 182 | 7.6 | 5.7 | 18 | 8 | 56 | 72 | 18 | — |
| Q8_0 | 34 | 165 | 4.85 | 5.2 | 32 (24 × 16-bit) | 8 | 0 | 98 (59 %) | 20 | 64 → 50 % |
| IQ4_NL | 18 | 162 | 9.0 | 5.1 | 20 | 8 | 52 (32 PRMT) | 62 | 10 | 32 → 100 % |
| IQ4_XS | 17 | 133 | 7.8 | 4.2 | 16 | 8 | 60 | 41 | 5 | 47 → 62.5 % |
| Q4_K | 18 | 190 | 10.6 | 5.9 | 24 | 16 | 34 | 79 | 30 | 56 → 56 % |
| Q5_K | 22 | 230 | 10.5 | 7.2 | 28 | 16 | 46 | 100 | 30 | 41 → 62.5 % |
| Q6_K | 26.25 | 188 | 7.2 | 5.9 | 34 | 16 | 28 | 86 | 16 | 46 → 62.5 % |
| MXFP4 | 17 | 192 | 11.3 | 6.0 | 28 (18 × 8-bit) | 8 | 70 (44 PRMT) | 64 | 10 | 32 → 100 % |
| NVFP4 | 18 | 269 | 14.9 | 8.4 | 16 | 8 | 91 (64 PRMT) | 103 | 13 + 36 control | 39 → 75 % |

Notes on the table:
- No kernel uses 64- or 128-bit loads.
- On Pascal a 32-bit multiply is 3 XMAD (16-bit) instructions. The Q4_0 loop strength-reduces the weight pointer, but recomputes the 36-byte activation-block addresses with XMAD chains and 64-bit carries on every iteration.
- The fused gate+up kernels (`<type,1,true>`) cost per byte: Q4_0 5.0, Q4_1 4.2, Q8_0 3.65, IQ4_NL 7.3, NVFP4 13.7.

**Dynamic path at Flint's shapes, Q4_1 (my measurement plus path reading):**
- One iteration (K = 2048): prologue ~54 + peeled iteration ~80 + epilogue ~43 for warp 0 (~10 for warps 1–3, which exit after the shared-memory store). That averages ~153 lane-instructions per 16 weights, or **~15 per byte**, three times the loop's 5.1.
- Two iterations (K = 4096, attn_output): ~205 per 32 weights, ~10 per byte.
- Fused gate+up at K = 2048: ~11.8 per byte (estimated at ~0.7× of its 408 static instructions).
- Small-K expert down-projection (K = 512):
  - Warp 0 runs ~377 instructions for 4 rows; the 3 idle warps ~98 each.
  - That is **~16.8 per byte**, and 44 % of the block's instructions come from idle warps.
  - The kernel uses 56 registers, so 9 blocks fit per SM, which means **9 working warps per SM**.

**Flint per-token instruction budget, makers' Q4_1 file (my calc):**

| Matvec group | MB per token | Kernel path | ≈ Instr per byte | G lane-instr |
|---|---|---|---|---|
| attn q, k, v + output head (K = 2048) | 513 | 1 iteration | 15.3 | 7.9 |
| attn output (K = 4096) | 257 | 2 iterations | 10.3 | 2.6 |
| routed + shared gate/up, fused (K = 2048) | 493 | 1 iteration | 11.8 | 5.8 |
| routed down (K = 512) | 185 | small K, 1 of 4 warps working | 16.8 | 3.1 |
| shared down (K = 1024) | 62 | small K, 2 of 4 warps working | 11.9 | 0.7 |
| dense layers 0–1 | 47 | mixed | ~10 | 0.5 |
| **Total** (+99 MB F32 router on the float path) | 1,557 | | 13.3 average | **20.7** |

Issue capacity is 5 SMs × 128 lanes × 1.73 GHz = 1.107 T lane-instructions/s. That gives 18.7 ms at 100 % issue efficiency and 21–25 ms at 76–90 %. The earlier note's ~23–26 ms of "weight reading" is the same quantity: **the per-token matvec time is the instruction time, not the byte time.** At 173 GB/s the bytes would take 9.6 ms.

**Other cards and other measurements:**
- **P106-100 versus P106-090** (P; [calibration](../../benchmarks/2026-10-22-single-card-calibration.md) and [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)): the same GP106 chip and 192-bit GDDR5 runs Llama-2-7B Q4_0 at ~30.6 tok/s with 10 SMs against 19.45 with 5. That is 1.57× from doubling issue capacity at equal bandwidth.
- **Tesla P100, sm_60** ([qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) README, 2026-09-07 to 2026-10-16; S/P, a community fork with measurements):
  - `mul_mat_vec_q` is 92 % of decode time and "instruction-issue bound": warp issue at 76–99 % of peak, 27 % of the measured 607 GB/s, 9.25 SASS instructions per weight.
  - P100 has no dp4a, so its per-weight cost is higher than sm_61's.
  - Giving K-quants 4 rows per block (P100-only patch) took Llama-3.1-8B Q4_K_M from 39.3 to 52.2 tok/s.
  - The author found that launch-count reductions pay only if they add no work to the hot loop.
- **P40 (sm_61, 30 SMs, 346 GB/s)**, one-token llama-bench by format ([#20831 comment](https://github.com/ggml-org/llama.cpp/pull/20831#issuecomment-4189227947), JohannesGaessler, 2026-04-05; P). Llama 8B tok/s:
  - Q4_0 58.8, Q4_1 61.1, Q5_0 43.8, Q5_1 52.1, Q8_0 34.7, IQ4_NL 57.5, IQ4_XS 60.1, Q4_K_S 58.9, Q6_K 39.3.
  - Effective bandwidth (my calc, standard file sizes minus embeddings): **Q4_1 ≈ 293 GB/s and Q5_1 ≈ 296 GB/s (≈ 85 %)**, Q8_0 ≈ 277, Q4_0 ≈ 256, IQ4_NL/XS ≈ 250, Q6_K ≈ 242, **Q5_0 ≈ 230 (66 %)**.
  - The 4-byte-aligned types lead even on a card with 3× more SM per byte than the rig.
  - The same comment's granitemoe 3B MoE runs: Q4_1 160.9, Q4_0 157.5, IQ4_NL 154.1, Q5_0 135.7, Q8_0 128.6 tok/s.
- **Memory latency and caches:**
  - Jia et al., "Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking" ([arXiv 1903.07486](https://arxiv.org/abs/1903.07486), 2019-03-18; P), Tables 3.1–3.4, Tesla P4 (GP104, GDDR5):
    - L1 hit 82 cycles, L2 hit ~216 cycles, 16 load/store units per SM.
    - **L1 load throughput measured at 15.7 B/clock/SM** (64 theoretical).
    - L2 throughput 979 GB/s; DRAM 162 of 192 GiB/s.
  - DRAM latency on Maxwell GM204: 368–385 cycles unloaded at ~1.27 GHz, about 290–300 ns.
    - Volkov's thesis ([UCB/EECS-2016-143](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf), 2016; P): latency rises with load (on Kepler from 303 to 501 cycles at 94 % of peak).
    - Mei & Chu ([arXiv 1509.02308](https://arxiv.org/abs/1509.02308); P).
  - NVIDIA's [Pascal tuning guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/) (P) says Pascal DRAM latency is similar to Maxwell's.

### Inferences
- **Issue-bound arithmetic (my calc).**
  - The weight stream needs 173 GB/s ÷ 5 SMs = 34.6 GB/s per SM, which is 20 B per core clock at 1.73 GHz.
  - An SM issues at most 4 warp-instructions (128 lane-instructions) per clock, so full bandwidth allows **6.4 lane-instructions per byte** at perfect issue efficiency.
  - Q4_1's loop already needs 5.1, and Flint's one-iteration path ~13 on average.
  - A P40 has 11.5 GB/s per SM, which allows ~17 instructions per byte. That is why tuning that is neutral on a P40 (ik #2701) can still matter here.
- **Ratios against the calibration.**
  - Instructions per byte: Q4_0/Q8_0 = 6.2/4.85 = 1.28, against measured efficiency 54/42 = 1.29 (about 1.22 after the Q8_0 run's clock advantage).
  - Q4_K/Q4_0 = 1.7 against a measured 1.2. The K-quant gap is overstated by the loop count: a Q4_K_M file mixes Q6_K, its per-row overhead weighs less, and the calibration build predates #26705.
  - The format ordering Q8_0 > Q4_0 > Q4_K_M per byte is reproduced.
- **Why Q4_1 is a fast format on Pascal.** 32-bit loads (16 against Q4_0's 20 per 32 weights) and cheaper addressing outweigh 11 % more bytes. Expect a Q4_1 file to run at least as fast as Q4_0 per weight on this rig (claim to test; the P40 data agree).
- **Little's law (my calc).**
  - At ~290 ns unloaded DRAM latency, or 500 core cycles at 1.73 GHz, rising to 650–850 cycles near peak, an SM must keep **~10–17 KB in flight** to sustain 20 B/clock.
  - Q4_1 at one token: 40 registers allow 48 warps (1,536 threads) per SM. Each thread has ~10 B of weight loads outstanding, at most ~15 KB if every thread waited at once.
  - A thread actually spends ~150 instructions issuing: 48 warps sharing 4 issue slots per clock means roughly 1,800 clocks of life, of which 500–850 are spent waiting. So effective in-flight bytes are ~4.5–7 KB, short of the 10–17 KB needed.
  - The expert down-projection has only 9 working warps per SM, at most ~11.5 KB.
  - Q8_0 keeps ~40 B per thread in flight (4 calls unrolled) across 32 warps, one more reason it is the most efficient per byte.
  - Low MLP is a symptom of the same cause: too few bytes per thread-instruction.
- **The L1 co-limit (claim to test).**
  - Every output row re-reads its Q8_1 activation block through the L1/texture path: 36 B per 32 weights for 20 B of Q4_1 weights.
  - At full bandwidth that is ~36 B/clock/SM of cache reads, while the P4 measured 15.7 B/clock/SM for L1 hits and its L2 delivers ~32 B/clock/SM.
  - If the texture path behaves like the measured L1 path, activation re-reads alone cap weight streaming at well under half of bandwidth. More rows per block, with the activations shared in registers, fixes this as well.
- **Answer to "ALU, issue slots or loads".** Issue slots, with the load/store units and cache path as co-limits.
  - dp4a is ≤ 8 % of instructions.
  - The F2F/I2F conversions run on a quarter-rate pipe (8–12 per 32 weights). They are not binding alone but add issue pressure.
  - Too few outstanding loads is real, but it is a consequence of the per-thread work shape.

### Gaps
- Static counts only. No dynamic profile (instructions executed, issue-slot use, stall reasons) of MMVQ on the rig. Dual issue of loads alongside ALU work is not modelled.
- The calibration (`85d5703a3`) and the SASS (`d77256999`) come from different builds. Q4_0/Q4_1/Q8_0 MMVQ code did not change between them; Q4_K did (#26705).
- No measurement of the texture/L1 throughput of `LDG.E.CI` on GP106, and no tabulated GDDR5 Pascal DRAM latency (GM204 used as a proxy).

### Experiment ideas
- **E1. Profile MMVQ on one card with nvprof (1–2 h).** Nsight Compute does not support Pascal; nvprof ships with CUDA 12.9.
  ```
  GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 /usr/local/cuda-12.9/bin/nvprof --kernels "regex:mul_mat_vec_q" \
    --metrics inst_executed,inst_issued,issue_slot_utilization,ipc,achieved_occupancy,dram_read_throughput,\
  tex_cache_throughput,l2_read_throughput,ldst_fu_utilization,stall_exec_dependency,stall_memory_dependency,\
  stall_pipe_busy,stall_sync,stall_not_selected \
    llama-bench -m <hidden-2048 model, e.g. Qwen3-1.7B or Llama-3.2-1B, pure Q4_1> -p 0 -n 8 -r 1
  ```
  - The issue-bound model predicts issue_slot_utilization ≥ 70 %, dram_read_throughput ≤ 80 GB/s and inst_executed per weight close to the table above.
  - Run it on the rig's own build; it needs no rebuild.
- **E2. Per-format calibration at Flint's width (1–2 h).**
  - `llama-quantize --pure` a hidden-2048 dense model to Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS, MXFP4 and NVFP4 (with `--output-tensor-type` set to the same type).
  - Then `llama-bench -p 0 -n 128 -r 5 -fa 1` on one card.
  - The prediction is the per-weight instruction ranking: Q4_1 ≥ Q4_0 > IQ4_XS > IQ4_NL ≈ Q8_0 (per weight) > Q5_1 ≈ Q5_0 ≈ MXFP4 > NVFP4.
  - This also fills the brief's gap of unmeasured Q4_1, Q5_0, IQ4_NL, MXFP4 and NVFP4 efficiency.

---

## 4. L1: Pascal launch shape (2 warps per row at one token; no idle warps for small K)

### Takeaway
Two small kernel-parameter changes target exactly the waste found in section 3, need no format change, and exist as open PRs or close relatives:
- At one token, 2 warps (or 1) per block, so a K = 2048 row gives each thread 2–4 iterations instead of 1.
- For the K = 512 / 1024 down-projections, nwarps sized to K (1–2), so no warps idle.

Expected +5–20 % for Flint. This is the cheapest lever in this survey.

### Cited findings
- [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479) "ggml-cuda : optimize mmvq nwarps for Pascal DP4A" (2026-07-09, open, P).
  - Adds `MMVQ_PARAMETERS_PASCAL_DP4A` for compute capability 6.1/6.2 with 2 warps at one token.
  - GTX 1060 6GB, Llama-3.2-3B tg128, paired runs: Q4_0 +4.6 %, Q4_K_M +5.4 %, Q5_K_M +3.2 %, Q8_0 +2.5 %, Q6_K −0.3 %; prefill unchanged.
  - JohannesGaessler asked why it was still a draft; there has been no review since.
- [PR #20831](https://github.com/ggml-org/llama.cpp/pull/20831) "cuda : dynamic MMVQ nwarps for narrow matrices" (2026-03-21, open, P).
  - Its body says MoE expert layers are narrow (512–2048 columns), so most warps have no work but still pay `__syncthreads()` and the shared-memory reduction.
  - RDNA4: Qwen3.5-35B-A3B +9–10 %. After review it was limited to RDNA3/RDNA4. The maintainer asked for the name `calc_max_nwarps` and wanted to check for regressions.
- [Issue #30334](https://github.com/ggml-org/llama.cpp/issues/30334) (2026-10-11, open, P): turning MMVQ fusion off on a P40 gives ±5 % depending on quant.
- ik_llama.cpp: [#2701](https://github.com/ikawrakow/ik_llama.cpp/pull/2701) (2 rows per block at one column: P40 neutral, P100 +12 %); #2721 (2026-10-21): per-row sums kept in registers in fused MMVQ, P100 +5.8 %, P40 0 % (P, reported by the GitHub survey).

### Inferences
- **Instruction savings at Flint's shapes, makers' Q4_1 (my calc, static path counts from section 3):**
  - 2 warps: the K = 2048 matvecs go from ~15.3 to ~10.7 instructions per byte (the prologue and epilogue are shared by 2 iterations). Gate/up drops ~25 %, attn_output ~20 %.
    - −4.2 G of 20.7 G.
  - The expert down-projection with 1 warp and 4 rows (instead of 4 warps with 3 idle) removes ~44 % of its instructions and raises working warps per SM from 9 to ~32.
    - −1.4 G; the shared down −0.2 G.
  - Total −5.8 G (−28 %). If M scales with instructions: M 23–26 → 16.5–19 ms, T 26.5–32 ms, 31–38 tok/s.
  - Fewer threads per row halve the bytes in flight per row, so I would expect only part of that: **+5–20 % (29–33 tok/s), claim to test.**
- **Why Flint should gain more than the GTX 1060's model.** Flint's rows are 2048 wide (one iteration per thread at 4 warps), against Llama-3.2-3B's 3072 (1.5 iterations). Its experts are 512 wide (the small-K path). The rig has half the 1060's SMs for the same bandwidth.
- **Qwen3.6-35B-A3B** has the same hidden size 2048 and 512-wide experts, so the same relative gain on its matvec share.
- **Risk.** Launch-shape changes are the kind of Pascal-only table edits upstream already carries (the GB10, Turing and RDNA tables). #25479 has sat unreviewed for 3.5 months.

### Gaps
- Not measured on a 5-SM card. #25479's GTX 1060 run has no MoE model.
- How the small-K and 2-warp settings interact with the fused gate/up kernel's register use (46 → ? registers) is unknown.

### Experiment ideas
- **E3 (≈1–2 h).**
  - Apply #25479 to `d77256999` plus the graphs patch (#27721). Add a Pascal branch that uses 1 warp and 4 rows when `blocks_per_row_x ≤ blocks_per_iter_1warp` (that is, #20831's `calc_max_nwarps` without the RDNA restriction).
  - Build with the usual docker recipe and run Flint with `llama-bench -p 0 -n 128 -r 5 -fa 1` on 5 cards. Also run Qwen3.6-35B-A3B Q4_0.
  - Expected 29–33 tok/s for Flint. Check that `test-backend-ops -o MUL_MAT,MUL_MAT_ID` passes, which needs a build with tests.

---

## 5. L2: a Pascal-specific MMVQ kernel for the existing block formats

### Takeaway
Rewrite the one-token matvec for sm_61 around the instruction budget:
- each warp computes 2–4 rows, so one activation load serves several rows;
- each lane runs several iterations (K = 2048 → 64 weights per lane per row);
- block and row base pointers are computed once, and the loop advances them by constant 32-bit offsets with immediate-offset loads instead of re-deriving ×18/×36 addresses with XMAD chains;
- a warp-shuffle reduction per row, with no shared memory or `__syncthreads`;
- nwarps sized to K (section 4).

This keeps the GGUF formats, so it serves the makers' Q4_1, UD-mirror v1 (Q4_1/Q5_0/Q8_0) and Qwen3.6 alike. Expected +20–45 %.

### Cited findings
- [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) (2026-09/10; S/P):
  - Profiled MMVQ as instruction-issue bound on P100 and called instruction count the lever, with "cuts convert to speedup nearly 1:1".
  - 4 rows per block for K-quants: +33 % on P100 (Llama-3.1-8B Q4_K_M).
  - Skipping the per-row recomputation of the activation sum (`dp4a(0x01010101, u)` → stored `ds.y`): 148 → 138 instructions per 16 weights, +4 %, architecture-neutral.
  - Its sm_61 builds were checked for byte-identical SASS, not for speed.
- mmvq.cu (P): the comment on `mmvq_two_rows_ncols1` states the mechanism (a second row shares the Q8_1 loads and adds MLP), but enables it only for 1–3-bit types.
- ik_llama.cpp #2701 (P): 2 rows at one column, P40 +0.3 % on Llama and −4.5 % on DeepSeek, P100 +12 %.
- ik_llama.cpp reuses one Q8_1 quantization across consecutive matmuls with the same input (`ggml-cuda.cu`; P, per the GitHub survey).
- Microsoft BitNet GPU kernel ([microsoft/BitNet/gpu](https://github.com/microsoft/BitNet/tree/main/gpu), 2025-05; P), a dp4a W2A8 GEMV generated with Ladder:
  - Weights are permuted offline; activations come in as 128-bit int8 loads.
  - One `lop3` per 4 weights, `__vsubss4`, then `__dp4a`, and a shuffle reduction.
  - Every instruction exists on sm_61, though it is compiled for sm_80.
- ExLlamaV3 `exl3_gemv_int8_kernel.cuh` (by 2026-07; P, per the external survey): an int8-activation dp4a GEMV with per-warp 256 B contiguous loads and two-row prefetch. [Issue #512](https://github.com/turboderp-org/exllamav3/issues/512) (2026-10-14) proposes an sm_61 lane around it.

### Inferences
**Instruction estimate for a rows-per-warp Q4_1 kernel at K = 2048 (my calc):**
- Per 32 weights per row:
  - 4 32-bit weight loads + 1 `dm` load;
  - 9 activation loads amortized over 4 rows (~2.3);
  - unpack 8–12, dp4a 8;
  - float ~5 (I2F, FMUL, FFMA, a shared F2F per 32);
  - addressing ~2.
- That is **~32 instructions per 20 bytes = 1.6 per byte in the loop.**
- Per-row overhead with 4 rows per warp and 2 lane-iterations per row: prologue ~40 shared by 4 rows, shuffle reduction ~10 per row, store.
- Overall **~2.5–3.5 instructions per byte**, against today's ~13.3 average. The issue time for 1.66 GB falls to 4–6 ms, below the 9.6 ms the bytes need at full bandwidth, so the kernel becomes bandwidth- or latency-bound.
- **Memory parallelism:** 4 rows × 2 iterations × 16 B = 128 B in flight per lane. Sixteen warps per SM then hold ~64 KB, well above the 10–17 KB needed.

**Expected speed (my calc):**
- Anchor: doubling SMs (P106-100) gave 1.56× on Q4_0. Halving instructions per byte should do about the same to M: 23–26 → 15–17 ms, T = 25–30 ms, **33–40 tok/s**.
- If the kernel reaches a bandwidth-bound 60–75 % of 173 GB/s: M = 1.66 GB / 104–130 GB/s = 12.8–16 ms, T = 23–29 ms, 34–43 tok/s.
- I quote **+20–45 %**.
- Q8_0 benefits less in relative terms (it is already the least issue-bound). Q5_0 benefits more.
- UD-mirror v1 (~2.2–2.4 GB per token, my estimate from UD-Q4_K_XL's 2.21 GB) would gain in proportion to its matvec share.

**Applies to MUL_MAT_ID unchanged.** Gate/up is 512 rows × 6 experts and is already fused; 4 rows per warp gives 768 warps. The ids lookup stays one dependent load per block.

**Quality:** exact, except for float summation order.

### Gaps
- No sm_61 measurement of rows-per-warp at one token on any card with this little SM per byte. The P40's neutral result shows the gain depends on the SM-to-bandwidth ratio.
- Register pressure with 4 rows plus fusion: the current fused small-K kernels already use 70–80 registers, which would cap occupancy at 25–37 %.

### Experiment ideas
- **E4. Standalone microbenchmark first (2–3 days).**
  - Write `mmvq_pascal.cu` with Flint's shapes: [2048→512] × 6 gathered experts with fused gate/up, [512→2048] × 6, [2048→4096], [4096→2048] and [2048→150144].
  - Variants: (a) llama.cpp's `mul_mat_vec_q<Q4_1>` copied verbatim; (b) #25479 shape; (c) 1 warp × 4 rows with strength-reduced addressing; (d) the SoA layout of section 6.
  - Build in the CUDA 12.9 container (`nvcc -arch=sm_61 -O3`), copy to `/tmp` on the rig, run on one card, report µs and GB/s per shape, and delete it afterwards.
  - This decides L2/L3 before touching llama.cpp.
- **E5.** If (c) beats (a) by more than 1.3× on the gate/up and down shapes, port it as a Pascal path in `mmvq.cu` for Q4_0/Q4_1/Q8_0 (then Q5_0), including the fused GLU epilogue. Run `test-backend-ops` and Flint tg128 (1 week).

---

## 6. L3: load-time SoA repack (and exact group-64 compaction of the makers' Q4_1)

### Takeaway
Once the kernel is fixed (L2), the format is what remains:
- 18/20/34-byte blocks forbid 128-bit loads and need non-power-of-two strides;
- scales are interleaved with quants.

A load-time "reorder" turns each quantized weight into a quants plane (16 B per 32 weights, 16-byte aligned rows) and a scales plane. This is what llama.cpp's SYCL backend does for Intel GPUs.

For the makers' Q4_1 the repack can also merge each pair of identical (d, m) blocks into one group-64 entry: an exact 10 % byte cut, from 1.66 to ~1.50 GB per token. No new GGUF type is needed.

Expected +5–15 % on top of L2.

### Cited findings
- **SYCL precedent** (P):
  - [#12035](https://github.com/ggml-org/llama.cpp/pull/12035) "[SYCL] Optimize mul_mat for Q4_0 on Intel GPU" (2025-02-23, merged 2025-02-24): separates quants from scales once, in the graph. Llama-2-7B Q4_0: PVC 1550 39 → 73, Flex 170 39 → 50, Arc 770 42 → 55, MTL 13 → 16 tok/s.
  - [#21527](https://github.com/ggml-org/llama.cpp/pull/21527) (merged 2026-04-07): Q8_0 reorder on an Arc Pro B70, 4.88 → 15.24 tok/s, bandwidth use 21 % → 66 %. It blames Q8_0's non-power-of-two 34-byte blocks for poor coalescing.
  - Since extended to Q4_K/Q5_K/Q6_K, MoE, IQ3 and MXFP4 (#24452, #29337, #22152, #29107, #29809; per the GitHub survey).
- **CUDA attempts** (P):
  - [#20563](https://github.com/ggml-org/llama.cpp/pull/20563) "CUDA: WIP: Repack NVFP4 blocks into new SoA layout" (2026-03-14, closed 2026-03-31). JohannesGaessler said no data should be repacked before a baseline without repacking exists. ORippler said every contiguous dimension would need padding to 16-byte alignment through `get_alloc_size`.
  - #22112 "CUDA: PoC for repacking mxfp4" (2026-04-19, open): RTX 5090, prefill +8–9 %, TG slightly down, "underwhelming".
  - #26311 NVFP4 linear repack + MMVQ (2026-07-30, open): B6000 unchanged at ubatch 1.
  - Vulkan #21024 repack PoC: RTX 3090 Q4_0 TG −0.6 %.
- **ik_llama.cpp** [#127](https://github.com/ikawrakow/ik_llama.cpp/pull/127) "Q4_0_R4 on CUDA" (2024-12-08, closed, P): 4-row interleaving reached the same TG as Q4_0, which ik called disappointing.
- **CPU repack history in mainline** (P, per the GitHub survey):
  - Q4_0_4_4 / 4_8 (#5780, 2024-07) and Q4_0_8_8 (#8713, #9532) were removed from GGUF in #10446 (2024-12-07) in favour of runtime repacking in the CPU "extra" buffer type; AMX and KleidiAI followed.
  - Mainline has no CUDA extra or repack buffer type.
- **The makers' file is pair-structured** (my measurement; [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) config and safetensors headers, 2026-10-21; [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf), 2026-10-21; P):
  - The config has `"quantization": {"bits": 4, "group_size": 64}`; `flint-int4.json` declares `"weight_format": "mlx-affine-int4"`.
  - Every q/k/v/o, dense, routed and shared expert, and `lm_head` weight is U8-packed with BF16 `scales` and `biases` of shape [rows, K/64]. The router `mlp.gate` is BF16 unquantized, and `embed_tokens` is 8-bit.
  - Range reads of the Q4_1 GGUF (first 256 blocks of blk.12 attn_q, ffn_gate_exps and ffn_down_exps, and of output) show consecutive Q4_1 block pairs with identical (d, m), equal to the MLX scale and bias converted BF16→FP16 without loss.
  - Some scales sit near FP16's smallest normal value (attn_q d ≈ 7.1e-5), so a full-tensor check is needed before relying on it everywhere.

### Inferences
- **Why big-GPU repack PoCs don't transfer, and why this rig differs.** On a 5090 or B6000 MMVQ is not issue-bound, so cheaper loads and addressing change nothing. On a 5-SM Pascal they attack the 41–59 % of loop instructions spent on address arithmetic and 16/32-bit loads.
- **The SYCL Arc 770 case resembles ours:** few execution units per byte, with the Q4_0 TG gain 31 %.
- **Loop cost after repack (my calc).** One `LDG.128` brings 32 weights (in place of 4–8 narrow loads), and the strides become powers of two. Per 32 weights per row: ~1 weight load, ~0.75 activation load with 4 rows and SoA activations, 8–12 unpack, 8 dp4a, ~6 float/scale, ~1–2 address. That is **~26–30 instructions**, against ~32 for the L2 kernel on 20-byte blocks and 102 today.
- **Speed (my calc).**
  - With compaction: 1.50 GB at 65–85 % of 173 GB/s gives M = 10.2–13.3 ms and T = 20–26 ms, **38–49 tok/s**.
  - Without compaction (Qwen3.6, UD-mirror): the same efficiency on unchanged bytes.
  - The increment over L2 is +5–15 %.
- **Implementation constraints.**
  - Prefill (MMQ, at ≥ 7 tokens for Q4_x experts) must read the same layout. Two copies would double weight VRAM, impossible with 26 GB on 5–9 × 6 GB. So an MMQ tile loader for the SoA layout is also needed (~1–2 days).
  - The tensor must report a padded `get_alloc_size`.
  - `-sm layer` keeps tensors whole per card, so splitting is not an issue.
- **Activations.** The activation side also wants SoA (int8 plane plus FP32 scale and sum plane) for 128-bit loads. This is the natural place for L5's FP32 sums.

### Gaps
- No CUDA repack measurement on any pre-Volta card; no full-tensor check of the pair structure; no MMQ cost estimate for an SoA tile loader.

### Experiment ideas
- Variant (d) of E4: an SoA group-64 Q4_1 kernel with `LDG.128`, run on the same shapes.
- **E6 (30 min).** Script the full pair check: range-read every Q4_1 tensor's (d, m) and compare with MLX scales/biases (`check_q41_pairs.py`-style, about 100 MB of range reads). That establishes whether load-time compaction is exact for the whole file.

---

## 7. L4: a new GGUF type, "Q4_1 group 64" (and other group-size and scale variants)

### Takeaway
- **What the type would be.** A 4-bit affine type with one (d, m) per 64 weights is 36 bytes per 64 weights (4.5 bpw). It holds Flint's quantization-recovered codes exactly at 10 % fewer bytes than the makers' Q4_1. MLX checkpoints and HQQ/GPTQ group-64 imports could use it losslessly as well.
- **What it buys with today's kernel style.** Little (+2–4 %), because the cost is per weight, not per byte.
- **What it adds over L3.** A smaller file (26.0 → 23.4 GB) and VRAM, both of which L3's load-time compaction already gives in VRAM.
- **Other variants.** Group 128, int8 scales and shared scales cut 2–6 % more bytes, but lose exactness for Flint and save almost no instructions.
- **Verdict.** Worth doing only in a fork, after L2/L3, or if an MLX/GPTQ importer is wanted anyway.

### Cited findings
- **Mainline group-64 precedent.** Q2_0 (group 64, fp16 scale) for PrismML's Ternary Bonsai models: CPU in [#24448](https://github.com/ggml-org/llama.cpp/pull/24448) (merged 2026-07-07), CUDA in [#25707](https://github.com/ggml-org/llama.cpp/pull/25707) (merged 2026-07-30). Ggerganov argued for group 64 over 128 in discussion #22019 (2026-04-21) (P, per the GitHub survey).
- **Closest 4-bit attempt.** [#29368](https://github.com/ggml-org/llama.cpp/pull/29368) HQQ "Q4_H" on Q4_1's 32-weight layout (2026-09-24, closed by its author as not worth it; P, per the GitHub survey). No 4-bit group-64/128 affine type exists in mainline or in ik_llama.cpp.
- **Importers.** MLX quantized tensors are unsupported: CISC on #22431 (2026-04-27) said to dequantize first. GPTQ import (#14810, merged 2025-10-23) dequantizes and requantizes (P, per the GitHub survey).
- **Flint's export.** Group 64 affine with BF16 scales and biases; the router is BF16 (section 6).

### Inferences
- **Bytes (my calc).**
  - Q4_1-G64 for all of the makers' linear weights: (1.66 − 0.099) GB × 0.9 + 0.099 GB (F32 router) = 1.50 GB per token. With a BF16 router, as in the MLX source: 1.455 GB.
  - Group 128 with fp16 d, m would be 4.25 bpw, another −5.6 % of bytes. But Flint's codes are group-64, so it would need requantization or QAT (lossy).
  - Int8 group scales with a per-row FP32 super-scale (IQ4_KS style) are 4.25 bpw affine (two bytes per 64) and also lose exactness, since the BF16 scales are arbitrary.
- **Speed (my calc).**
  - With today's kernel structure a group-64 block saves one `dm` load and two F2F per 64 weights, about −5–7 % of loop instructions, so +2–4 % in tok/s.
  - With L2/L3 the byte cut is worth its full ~10 % of M (bandwidth-bound).
- **Quality for BF16-derived quants.** Group 64 affine sits between Q4_0 (G32 symmetric) and Q4_1 (G32 affine). It would need its own KLD check against UD-mirror v1 (claim to test). For the makers' QAT file it is exact.
- **Conversion path.** A GGUF→GGUF repacker that verifies each Q4_1 pair's (d, m) are equal and emits one header per 64 (~100 lines of Python with gguf-py), or a direct MLX→GGUF importer.

### Gaps
- Upstream appetite for a 4-bit group-64 type is untested; the precedents cut both ways (Q2_0 accepted, Q4_H abandoned).

### Experiment ideas
- None before L2/L3 show the bytes matter. Then: implement it in the fork as a GGUF type carrying the SoA layout (avoids load-time repacking code) and compare KLD for a BF16-derived Flint quant at G64 against Q4_1 (80 × 512 wikitext on the quant machine, ~1 h).

---

## 8. L5: Q8_1 activations with FP32 sums, and one quantization per shared input

### Takeaway
- **FP32 sums.** The Flint rescale exists only because CUDA's Q8_1 activation block stores its sum in FP16. Storing the sum as FP32, in a separate SoA plane or as a 40-byte block, or pre-scaled by 1/32 as the earlier note proposed, makes Q4_0/Q4_1/Q5_x down-projections overflow-safe. Then `LLM_ARCH_FLINT` can come out of the rescale condition with no numerical risk. That is the earlier note's +7–15 % (2.5–4.7 ms per token), obtained exactly.
- **Quantization reuse.** Reusing one Q8_1 quantization across Q/K/V, and across the routed and shared gate/up inputs, removes ~145 more launches.
- **Prior art.** Neither change exists for CUDA in mainline. ik_llama.cpp does the reuse.

### Cited findings
- PRs #30861 and #30928, issue #30704 and the mechanism: covered in flint_speedups.md section 2 (P).
- Comments on #30861 (2026-10-21; P, per the GitHub survey):
  - ggerganov: the F16 block sums can overflow even when the values don't.
  - jeffbolznv suggested BF16 for the Q8_1 scale; ggerganov: "I'll look into that."
  - The #30928 review calls the rescale "a workaround until we have a better fix".
- [#22571](https://github.com/ggml-org/llama.cpp/pull/22571) "Swap out F16 for BF16 in Q8_1 activations" (open since 2026-05-01, last updated 2026-05-07; P).
- No PR opened 2026-10-20 to 22 adds FP32/BF16/int32 sums, a CUDA skip for formats that don't use the sum, an env flag, or a parallel `POOL_1D` (GitHub search, per the survey).
- ik_llama.cpp quantizes src1 once for consecutive matmuls with the same input, and uses a "Q8_2" with BF16 scale and sum on CPU because DeepSeek produced NaNs (#292, 2025-03-26) (P, per the GitHub survey).
- Mainline's #25441 dedups gate/up activation quantization for MoE (merged 2026-07-16; perf-neutral for Q8_1). #22710 "Fuse rms_norm, mul, quantize_q8_1" (closed 2026-08-10) claimed +3–5 % TG on an RTX 5090 (P, per the GitHub survey).

### Inferences
- **Launch count (my calc).** Per token, every MMVQ call launches its own `quantize_q8_1`:
  - per MoE layer: Q, K, V, O, routed gate/up, routed down, shared gate/up, shared down = 8 launches;
  - dense layers 6 each, plus the head.
  - That is ~389 quantize launches per token, about 20 % of all launches.
- **Reuse savings (my calc).** Q/K/V reuse saves 98 and routed/shared gate-up reuse 47: −145 launches, ≈ −0.7–1.0 ms (+2–3 %).
- **Combined.** FP32 sums plus dropping the Flint rescale: F −2.5–4.7 ms. Alone, about 31–33 tok/s; stacked on L2/L3, see section 15 (my calc, claim to test).
- **Cost of FP32 sums.** One more 4-byte value per 32 activations (a 2 KB vector grows by 256 B). Every vec_dot that uses `ds.y` changes, in both MMVQ and MMQ (DS4 layout). In a fork that is mechanical work, 1–2 days. Upstream might accept it given #30861's open "better fix" TODO, but would want all backends consistent.
- **Qwen3.6.** It doesn't use the rescale; the quantization reuse applies.

### Gaps
- No measurement of quantize-launch cost on the rig. #22710's 5090 gain may not transfer; graph launches already remove most CPU cost.

### Experiment ideas
- **E7 (1–2 days).**
  - Patch `quantize_q8_1` to write `s` as FP32 into a side array, or as FP16 of `sum/32` with ×32 in the readers.
  - Update `vec_dot_q4_0/q4_1/q5_0/q5_1` and the MMQ DS4 loaders, and drop `LLM_ARCH_FLINT` from the rescale guard.
  - Run Flint Q4_1 perplexity (80 × 512 wikitext, no NaN) and tg128.
  - Speed A/B is possible right away with the existing rescale-off patch built for this project (`...-moe-down-rescale-env`).

---

## 9. L6: 5-bit bit order and unpack tricks

### Takeaway
- **Q5_0.** Its 5th-bit insertion costs 56 of 192 loop instructions per 32 weights. Q5_K's bit order needs ~2–3 instructions per 4 weights for the same job, against ~8–12 for Q5_0. A Q5_0 layout with Q5_K-style high bits would cut the Q5_0 loop by ~17 %. That matters because UD-mirror v1 uses Q5_0, the slowest legacy type per byte on Pascal.
- **4-bit types.** The "one LOP3 per eight weights" idea mostly exists already: Q4_0/Q4_1's nibble order gives 8 dp4a operands with 3 ops per 32-bit word. Running dp4a on unsigned high-nibble bytes (`x & 0xF0F0F0F0`, then dividing the partial sum by 16) cuts that to 2.
- **Size of the gains.** Both are small next to L2/L3.

### Cited findings
- vecdotq.cuh (P):
  - `vec_dot_q5_0_q8_1_impl` inserts each high bit with a separate shift, AND and OR.
  - `vec_dot_q5_K_q8_1_impl_vmmq` uses `((vh >> i) << 4) & 0x10101010` for 4 weights at once.
- P40 per-format data (section 3): Q5_0 ≈ 66 % of bandwidth against Q5_1 ≈ 85 % on Llama 8B. Q5_0's 2-byte alignment adds 16-bit loads on top of the unpack.

### Inferences
- **Q5 SoA layout (my calc).** 16 B nibble plane + 4 B high-bit plane in Q5_K order + scale plane. Unpack ~20 per 32 weights (against 56), loads 2–3 wide loads per 32 weights, loop ~150 → **~−20–25 % per Q5 weight**.
- **Class-A spin-off for the file-only survey.** Q5_1 in place of Q5_0 is 5 % cheaper per weight in the loop on this rig's binary (182 against 192) at 9 % more bytes, and has the better affine fit. On an issue-bound card that is roughly speed-neutral for a quality gain (claim to test). Both formats use the FP16 sum, so neither is rescale-safe without L5.
- **Mask-only dp4a.** `dp4a.u32.s32` on masked high nibbles (values 16–240) times int8 activations gives at most 4 × 240 × 127 = 121,920 per dp4a, so 32-bit accumulation is safe. It saves one SHR per 32-bit word: 4 of ~102 instructions per 32 weights.

### Gaps
- UD-mirror v1's byte share in Q5_0 isn't in the brief, so its gain can't be computed here.

### Experiment ideas
- Fold into E2: the per-format bench gives Q5_0 against Q5_1 directly. If Q5_0 tensors are a large share of UD-mirror v1's bytes, try a Q5_1 variant of UD-mirror (class A, ~2 h on the quant machine).

---

## 10. L7: fusing the MoE expert chain and other launch cuts

### Takeaway
On sm_61 the chain is already mostly fused:
- top-k;
- gate+up+SwiGLU in one MMVQ;
- the weighted expert sum in one kernel ([#25952](https://github.com/ggml-org/llama.cpp/pull/25952), merged 2026-09-01);
- add chains.

What remains for decode:
- folding the down-projection into the weighted sum;
- having the producers (RMSNorm, SwiGLU, attention output) emit Q8_1;
- a MOE_FFN op;
- ultimately a persistent per-layer kernel.

The first three give +2–5 % together. The persistent kernel could be worth +25–40 % but is weeks of work with no Pascal precedent.

### Cited findings
All P, per the GitHub survey unless noted.
- [#25952](https://github.com/ggml-org/llama.cpp/pull/25952) fuses the router-weight multiply and the expert sum (k = 2–15); prefill-only numbers.
- [#29184](https://github.com/ggml-org/llama.cpp/pull/29184) (merged 2026-10-02) fuses the shared expert into MMVQ only when its width equals a routed expert's. Flint's 1024 against 512 doesn't qualify.
- [#26322](https://github.com/ggml-org/llama.cpp/pull/26322) "CUDA: add MOE_FFN" (2026-07-30, open):
  - Router through to reduction as one op. On the decode path the steps still run as separate MMVQ calls inside the op.
  - Only prefill +6–11 % was reported, and it broke partial offload in one report.
- [#22710](https://github.com/ggml-org/llama.cpp/pull/22710) (closed 2026-08-10): rms_norm + mul + quantize fusion, +3–5 % TG claimed on a 5090. am17an found fusing all the way into the matmul complex and disappointing.
- ik_llama.cpp: `-fmoe` (#229, 2025-02-23, RTX 4080 TG +7 %), a fused down path in `ggml_cuda_moe_up_gate_unary`, and `-mqkv`. No Pascal measurements of these.
- vLLM's Flint low-latency path runs each MoE layer as seven kernels on Blackwell (flint_speedups.md section 4; S).

### Inferences
- **Remaining FFN launches per MoE layer once the rescale is gone (my calc).** ~14: norm, quantize, router, top-k, gate/up, quantize, down, sum, plus four for the shared expert, plus adds.
  - Producer-emitted Q8_1 and a down+sum fusion remove ~3–5 per layer: 140–235 launches, ≈ 0.7–1.6 ms, **+2–5 %**.
- **Persistent per-layer kernel (my calc).** One kernel per layer with grid-wide barriers between router, gate/up and down (cooperative groups grid sync exists on sm_60+) would cut ~40 launches per layer to ~3–8. That is −1,400 launches, ≈ −7–10 ms, +25–40 %.
  - It also needs L2's kernel inside, plus attention.
  - This is the vLLM design, and the largest and riskiest item here.
- **The carry-over caveat.** qr0n's P100 found CUDA graphs and MMVQ fusion slightly negative there, because fusion added work to an issue-bound loop. On this rig graphs measured +14 %. Each fusion must avoid adding instructions to the MMVQ loop.

### Gaps
- No per-kernel timing on the rig (which of the ~1,900 launches cost how much).

### Experiment ideas
- After E1: run nvprof `--print-gpu-trace` on one Flint token (5 cards, graphs off) to get the per-kernel time distribution, so fusion targets come from data (1 h). Host RAM is the constraint: keep `-n` small.

---

## 11. L8: MXFP4 and NVFP4 on sm_61 (and whether to test the makers' NVFP4 file)

### Takeaway
- **Both have full paths in mainline on sm_61.** MMVQ and MMQ use dp4a plus a 16-entry `__byte_perm` lookup.
- **Both are among the most expensive formats per weight in the rig's binary.**
  - MXFP4: 6.0 instructions per weight. Its blocks are 17 bytes, 1-byte aligned, so it does byte loads.
  - NVFP4: 8.4 per weight. Its UE4M3 scales are converted in software on sm_61, and its K = 2048 rows leave 2 of 4 warps idle (64 weights per block, VDR 4).
- **The makers' NVFP4 file is therefore predicted to be slower than their Q4_1 on this rig, despite 13 % fewer bytes: 17–20 tok/s.**
- **MXFP4 is not useful.** It isn't Flint's QAT target and is costlier than Q4_0.

### Cited findings
- vecdotq.cuh and common.cuh (P):
  - `vec_dot_mxfp4_q8_1` uses `get_int_b1` (4 byte loads per word).
  - `vec_dot_nvfp4_q8_1` converts a UE4M3 scale per 16 weights through `__nv_fp8_e4m3`.
  - mmvq.cu gives NVFP4 a fused per-tensor scale multiply (#24481).
- [#15451](https://github.com/ggml-org/llama.cpp/pull/15451) (merged 2025-08-25, P): the `__byte_perm` lookup took gpt-oss-20B MXFP4 on a P40 from 63.6 to 82.0 tok/s at physical batch 1 (+29 %). Lookup formats had been instruction-bound on Pascal.
- [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf) (2026-10-21, P): 22.9 GB, tested only on Blackwell. The blog says NVFP4 is "within 0.7 points of the 16-bit model".
- No sm_61 NVFP4 decode number was found anywhere. Mainline's NVFP4 work since #20644 targets Blackwell.

### Inferences
- **NVFP4 against Q4_1 (my calc).**
  - Loop instructions per weight: 8.4 against 3.2 (2.6×).
  - About half of today's Q4_1 matvec instructions are per-block overhead, which is similar across types. So NVFP4 needs ~1.8× Q4_1's matvec instructions: M ≈ 41–47 ms against 23–26 ms, T ≈ 51–60 ms, **17–20 tok/s**.
  - A bandwidth-bound model predicts the opposite, ~30+ tok/s, from 1.45 GB against 1.66 GB.
  - The rescale stays for NVFP4 (it is keyed to the architecture, not the format), but could be dropped exactly, since NVFP4 doesn't use the sum.
- **Worth a speed test?** Only as a discriminating experiment for the cost model. E1/E2 answer the same question without moving 22.9 GB on the pendrive. If tested: `llama-bench -p 0 -n 128 -r 3 -fa 1` on 5 cards, ~10 min after the copy (~1 h at USB speed).
- **Fixing NVFP4 for Pascal** would need a hardware-free UE4M3→FP32 table (a 256-entry FP32 table in constant memory, or integer bit tricks) and a non-idle launch shape. It would still pay ~2× Q4_1's lookup cost. Not worth it while Q4_1 holds the same QAT quality.

### Gaps
- No measured MXFP4 or NVFP4 decode efficiency on any 5–10-SM Pascal card.

### Experiment ideas
- Covered by E2 (pure MXFP4 and NVFP4 quants of a small model). The Flint NVFP4 copy is optional.

---

## 12. L9: ik_llama.cpp's types (IQ4_KS, IQ4_KSS, IQ*_KT, row-interleaved _R4/_R8)

### Takeaway
- **Nothing here suits a 5-SM Pascal decoding through CUDA.**
  - The row-interleaved types are a CPU GEMM optimization with no CUDA kernels, except the IQ*_K_R4 family added for convenience. ik's own Q4_0_R4 CUDA test matched Q4_0.
  - IQ4_KS/KSS are lookup types like IQ4_XS.
  - The trellis KT types spend ALU per weight, the wrong trade on an issue-bound card.
- **ik_llama.cpp itself won't run here.** It needs AVX2 on the host.

### Cited findings
All P, per the GitHub survey, unless noted.
- **Layouts:**
  - IQ4_KS (#83, 2024-10-09): 4.25 bpw plus a per-row float.
  - IQ4_KSS (#89, 2024-10-16): 4.0 bpw.
  - IQ2/3/4_KT trellis (#441, 2025-05-20; integer trellis #529, 2025-06-14): 2.125 / 3.125 / 4.0 bpw plus a row float.
  - Row-interleaved Q4_0_R8, Q8_0_R8, IQ4_XS_R8 and IQ4_KS_R4/R16 (#118–#178, 2024-12 to 2025-01; IQ4_KS_R16 #2480, 2026-09-18).
- **CUDA support:**
  - MMVQ, MMQ and dequantize exist for IQ4_K/KS/KSS, the KT types, the IQ*_K_R4 family and IQ*_KS_R16.
  - Q4_0_R8, Q8_0_R8, IQ4_XS_R8, IQ4_NL_R4 and Q4_K_R4 have no CUDA kernels.
  - [#127](https://github.com/ikawrakow/ik_llama.cpp/pull/127) (2024-12-08): Q4_0_R4 on CUDA "achieves the same TG speed as Q4_0" (verified, P).
  - #2586 (2026-10-02): IQ4_KS_R16 on CUDA reached ~80–93 % of IQ4_KS's TG speed.
- **Pascal numbers.** Only for IQ4_K: [#2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509), P40 31.75 → 67.88 tok/s after moving to `__byte_perm`, reaching Q4_K speed. Q4_K_M on a P40 measured 47.77 tok/s on Llama-3.1-8B (#2721). None for KS, KSS, KT or R-types.
- **Host requirement.** flint_speedups.md section 7: ik needs AVX2; the rig's Celeron has no AVX. Below Ampere it does not use CUDA graphs.

### Inferences
- **Lookup cost.** IQ4_KS's per-32 decode is IQ4_XS-like (7.8 instructions per byte in mainline's loop), worse than Q4_1's 5.1.
- **Trellis cost.** The KT decode (an integer multiply hash per weight before dp4a) adds at least 2–3 instructions per weight. On Pascal a 32-bit multiply is 3 XMAD.
- **Usefulness.** The only ik idea worth porting is CUDA-side quantization reuse (L5).

### Gaps
- No Pascal decode numbers for any ik type except IQ4_K.

### Experiment ideas
- None.

---

## 13. L10: prior art outside llama.cpp, and what transfers to sm_61

### Takeaway
- **Nothing runs better on sm_61.** No external low-bit decode kernel runs on sm_61 with a measured advantage over llama.cpp's dp4a MMVQ. Each surveyed design needs one of: tensor cores (Marlin, Machete, FLUTE, BitBLAS's main paths), FP16 math (ExLlamaV2, LUT-GEMM, FastGEMV), or Volta+ toolchains (vLLM mainline, GPTQModel, torch cu128).
- **What transfers:**
  - Marlin's offline shuffle for one 128-bit load per thread at static offsets;
  - interleaved bit orders that a single LOP3 can split;
  - the BitNet and ExLlamaV3 dp4a GEMVs as code templates.
- **Where those ideas land.** All of them are already folded into L2/L3.

### Cited findings
All per the external survey, P unless marked.
- **Marlin** ([IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), 2024):
  - Needs compute capability ≥ 8.0, group −1 or 128.
  - Weights are permuted so each 32-bit word becomes an MMA fragment, with nibble order `[0,2,4,6,1,3,5,7]`.
  - Dequant is the LOP3 + FP16 magic-number trick; loads are 16-byte `cp.async` with an evict-first policy; work is split in stripes.
  - [vLLM #29901](https://github.com/vllm-project/vllm/pull/29901) (merged 2025-12-16) added an sm_75 path with synchronous copies and 2 × m16n8k8. There is no sm_6x path.
- **Machete** ([Red Hat post](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel), 2024-10-14): Hopper only (wgmma, TMA).
- **LUT kernels:**
  - LUT-GEMM ([arXiv 2206.09557](https://arxiv.org/abs/2206.09557), ICLR 2024): CUDA cores with FP16 lookup tables in shared memory; built for sm_80.
  - FLUTE ([arXiv 2407.10960](https://arxiv.org/abs/2407.10960), 2024): SM80 MMA atoms.
  - T-MAC (CPU-only) and LUT Tensor Core (hardware proposal).
  - CodeGEMM ([arXiv 2512.17970](https://arxiv.org/abs/2512.17970), 2025-12): CUDA cores, A100.
  - None targets Pascal. On an issue-bound card a table lookup is extra instructions per weight (IQ4_NL already shows this: 9.0 against 6.2 instructions per byte for Q4_0).
- **BitBLAS and TileLang:**
  - BitBLAS lists V100 (sm_70) as its lowest tested GPU.
  - BitBLAS [issue #251](https://github.com/microsoft/BitBLAS/issues/251) (2024-11-27): the GEMV path already uses dp4a, and a SIMT dp4a GEMM landed via TileLang (#255). No Pascal benchmark.
  - TileLang inherits sm_60/61 targets and has a `T.dp4a` intrinsic (2026-07). Untested on Pascal.
- **MLC-LLM.** Issues [#802](https://github.com/mlc-ai/mlc-llm/issues/802) (2023, GTX 1080 Ti "sm61 not supported yet") and #2100 (2024, P40: should work without FlashInfer and CUTLASS). No measured Pascal decode speed against llama.cpp; the "fast on old cards" claim found no support.
- **ExLlamaV2 / ExLlamaV3:**
  - V2 [#40](https://github.com/turboderp-org/exllamav2/issues/40): a P40 ran a 34B 4.0bpw at ~1.19 tok/s because all math is FP16; an FP32 path was declined as a "tall order" (#185).
  - V3 has no Pascal support; #512 (2026-10-14) proposes an sm_61 dp4a lane around its int8 GEMV.
- **vLLM / AutoGPTQ / Aphrodite on Pascal:**
  - [sasha0552/pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci) (2024-07 to 2025-08) builds vLLM 0.5.5–0.10 and Triton for Pascal. `gptq_marlin` is refused below capability 8.0; GGUF (llama.cpp-derived dp4a kernels) runs. No decode numbers.
  - AutoGPTQ [#232](https://github.com/AutoGPTQ/AutoGPTQ/issues/232) (2023, user reports, S-grade): its ExLlama kernel dropped a P40 to 1–2 tok/s; the old CUDA kernels gave ~8–15 tok/s.
  - Aphrodite (now "Sonar") listed Pascal support (GPTQ, GGUF; AWQ needs 7.5). No P40 numbers found.
- **Communities.** Searches of Chinese and Russian mining-card communities (P106/P104 terms) found nothing beyond generic llama.cpp pages.
  - The ExLlamaV3 #509 anecdote (2026-10-13, S): a P106 owner got ~30–45 tok/s on llama.cpp with "qwen 3.5 9b q4_k_s". Probably a P106-100.

### Inferences
- **The Marlin-to-sm_61 translation:**
  - pre-shuffle so each lane's 128-bit load holds 32 consecutive dp4a-ready nibbles (Q4_0/Q4_1 order already is);
  - unroll so offsets are immediates;
  - mask with LOP3 and feed dp4a (BitNet's kernel does exactly this);
  - share activations across rows in registers.
  - That is L2 + L3.
- **MLC/TVM.** They could in principle generate such kernels for sm_61, but nothing shows they do, and building them on this rig is a project in itself.

### Gaps
- No Pascal measurement exists for MLC-LLM's q4f32_1, TileLang dp4a GEMV, or BitNet's kernel compiled for sm_61.

### Experiment ideas
- Optional: compile Microsoft BitNet's `ladder_int8xint2_kernel` for sm_61 as a third variant in E4. A 2-bit kernel isn't usable for Flint, but its GB/s on [2048×4096] would show what a shuffle-plus-dp4a design reaches on this card (half a day).

---

## 14. Effort, risk and upstream acceptance

### Takeaway
- **Size of a new GGML type.** A full type touches ~35–40 files and ~600–700 lines (CPU reference plus CUDA), not counting converter and Metal. Upstream wants a model, PPL/KLD against BF16 and pure-CPU benchmarks, and maintainers are reluctant.
- **Kernel retunes (L1, L2, L5)** change no file format. They are smaller, testable with `test-backend-ops`, and have Pascal-specific precedents (#25479, the GB10 and Turing tables).
- **Recommendation.** Do everything in a private fork first. Offer upstream the launch-shape and FP32-sum pieces with rig numbers. Treat a new type as fork-only unless the data are striking.

### Cited findings
- **Diffstats of recent new types** (my measurement from git history at `d77256999`):
  - Q1_0: CPU 285 lines / 21 files ([#21273](https://github.com/ggml-org/llama.cpp/pull/21273), 2026-04-06), CUDA 204 lines / 11 files (#21629, 2026-04-15).
  - Q2_0: CPU 262 / 17 (#24448, 2026-07-07), CUDA 326 / 20 (#25707, 2026-07-30), plus a `__byte_perm` follow-up (#25603).
  - NVFP4: core 710 lines / 31 files including converter and graph (#19769, 2026-03-11), CUDA dp4a 120 lines / 8 files (#20644, 2026-03-26), MMQ later (#21074).
- **CONTRIBUTING.md at `d77256999`** (P): "adding new data types (extension of the `ggml_type` enum) carries with it a disproportionate maintenance burden". The minimum is a small HF model, perplexity and KL divergence against FP16/BF16 and same-size types, and pure-CPU llama-bench numbers. Features should land CPU-first.
- **Maintainer statements** (P, per the GitHub survey):
  - JohannesGaessler on #19762 (2026-02-20): more formats only from known experts, given the coordinated multi-backend work.
  - JohannesGaessler on #20563 (verified): repacking only after a baseline.
  - Ggerganov on #21273: asked for QX_0/QX_1/QX_K naming.
  - The ik IQ*_K port (#19726) was closed over licensing; a "clean-room" version (#30716–#30718, 2026-10-17) is open.
  - Q3_HIFI, SQ4, TurboQuant Q4_T and HQQ Q4_H were closed.
- **The rig's build script omits tests** (flint_speedups.md). `test-backend-ops` needs a build with `-DLLAMA_BUILD_TESTS=ON`.

### Inferences
**Effort in a fork (my estimate):**

| Item | Effort | Main risk |
|---|---|---|
| L1 | 1–2 h | Interaction with fused kernels' register use |
| L2 for Q4_1/Q4_0/Q8_0 incl. MUL_MAT_ID and the fused GLU epilogue | 3–7 days | Register pressure, Pascal-only code paths to maintain across rebases |
| L3 repack + MMQ loader | 3–5 days more | Alloc padding, every op that reads the tensor |
| L4 new type | 1–2 weeks | GGUF incompatibility with mainline and other tools |
| L5 FP32 sums | 1–2 days | Touches every Q8_1 consumer, MMQ included |

- **Rebase burden.** `mmvq.cu` changed often in 2026: #30517, #26705, #29184, #25441 and #27621 are all within three months. A fork kernel in a separate file, dispatched only for `cc == 610`, minimizes conflicts.
- **Upstream odds:** L1 good, if the rig data are clean. L5 fair (it fixes a known TODO). L2 fair as a Pascal table plus kernel variant. L3/L4 poor.

### Gaps
- None beyond those listed per lever.

---

## 15. Top candidates for a deep dive (ranked)

1. **Measure the cost model first: E1 (nvprof issue and stall profile) + E2 (per-format single-card bench at hidden 2048), ~3 h.**
   - Every class-B lever's size depends on whether the rig is issue-bound (this note's claim, from static SASS and the P106-100 comparison) or latency/bandwidth-bound.
   - E2 also gives the missing Q4_1/Q5_0/IQ4_NL/MXFP4/NVFP4 efficiencies for the class-A surveys.
2. **L1: #25479 plus no idle warps for small K, ~1–2 h, +5–20 %, exact.**
   - The cheapest test; it also checks the model (it predicts a larger gain on Flint than #25479 saw on a GTX 1060).
3. **L2: a Pascal rows-per-warp MMVQ kernel for Q4_1/Q4_0/Q8_0 (then Q5_0), +20–45 %, exact, applies to every file and to Qwen3.6.**
   - Start with the standalone microbenchmark E4 (2–3 days), so the decision rests on GB/s per Flint shape before any llama.cpp integration.
4. **L5: FP32 Q8_1 sums plus quantization reuse, +9–18 % on top of anything, exact.**
   - Removes the Flint rescale for all formats and ~145 quantize launches. Small and independent.
5. **L3: load-time SoA repack with exact group-64 compaction of the makers' Q4_1, +5–15 % on top of L2.**
   - Do it only if E4 variant (d) beats (c). Run E6 first to confirm the pair structure file-wide.
6. **L7 producer-emitted Q8_1 and down+sum fusion (+2–5 %).** After the above, guided by the per-kernel trace.

Not recommended:
- a new GGUF type (L4) before L3;
- MXFP4;
- ik types;
- LUT/trellis designs;
- the NVFP4 file test, except as a model check.

**Combined estimate (my calc, claim to test).**
- Assume the fixed cost falls from 10–13 ms to 6–10 ms (rescale gone, ~145 fewer quantize launches).
- L1 + L2 + L5 on the makers' Q4_1: T ≈ (6–10 ms fixed) + (13–17 ms matvec) ≈ 19–27 ms, so ~37–53 tok/s in llama-bench (~30–41 served at +5.4 ms per token).
- Adding L3's compaction brings the matvec part to 10–13 ms: T ≈ 16–23 ms, ~43–62 tok/s.
- None of these numbers has Pascal measurements behind it beyond the cited analogues.
