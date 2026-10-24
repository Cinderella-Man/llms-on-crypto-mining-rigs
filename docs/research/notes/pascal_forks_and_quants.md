# Pascal forks, other engines and quant formats: what others did on old cards, and what it means for the rig (2026-10-24)

Scope: who else has optimized LLM decode for Pascal or similar old and low-end NVIDIA cards (llama.cpp forks, other
engines, mining-card projects, papers), what each changed and measured, and whether this project's llama.cpp fork
already has it; why the fork's matrix-vector calls run slower inside a real token than in the microbenchmark; Samsung's
sub-1-bit methods and other 2025–2026 formats for low-end hardware; and what a quant format built for sm_61 decode
would look like. Ideas already in the [registry](../../../quants/experiments.md) (Q1–Q91),
[breakthrough candidates](breakthrough_candidates.md) and [unconventional ideas](unconventional_ideas.md) are not
proposed again; "Already covered here" maps them. Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "The fork" is this project's llama.cpp fork, branch `pascal-p106-v2` at `032b130f4`, unless a branch is named.
- Slot-equivalents: on compute 6.1 an SM retires 128 results per clock for FP32 arithmetic, integer add and logic
  operations, 64 for shifts, bit-field extract and compares, and 32 for type conversions, warp shuffles and population
  count (CUDA Programming Guide, arithmetic throughput table). A half-rate instruction is counted as 2 slot-equivalents
  and a quarter-rate one as 4. dp4a is not in that table; it is counted as full rate.

Starting point, measured ([registry](../../../quants/experiments.md), [kernel note](pascal_mmvq_integration.md)):
- S4-L0 on six P106-090 cards, fork build: 37.4 tok/s in llama-bench tg128, 26.8 ms per token, 2.24 GB of weights per
  token. G3 (KLD 0.101 against BF16) reads 2.33 GB.
- Profiled GPU time per token: 24.7 ms of matrix products, 8.0 ms of everything else; each launch carries 5–8 µs of
  event overhead in that profile. The host launches all six cards' graphs within 1.9 ms (Q49).
- In the [microbenchmark](../../../bench/mmvq_pascal/README.md) the fork's kernels stream 138–165 GB/s on large shapes
  against a 179 GB/s streaming read. In a real token the same calls run longer: attention q by ~13 µs, the routed
  down-projection by ~25 µs, the routed gate/up by ~33 µs, the output head by ~11 %.
- Decode cost follows bytes for Q4_0/Q4_1/Q5_0/Q8_0 on the fork's kernels; Q5_K and IQ4_XS are ported (Q46).

## Verdict

- **No fork or engine decodes faster on sm_61 than the fork's Pascal kernels, and none was tuned on a card like the
  P106-090.** Every Pascal-tuned project measured cards with 2–3× more issue slots per byte of bandwidth (GTX 1080,
  1080 Ti, P40, P100), so their tuning conclusions transfer only after re-costing in instructions per byte. The P100
  forks rewrite decode in FP16, which runs at 1/64 rate on GP106.
- **The largest new finding is the microbenchmark-versus-model gap, and its most likely cause is GPU address
  translation.** The benchmark is TLB-warm by construction: each MoE shape reads one 256-expert tensor with 64 fixed
  expert sets replayed 30 times, and every footprint stays under ~1.2 GB. Pascal's L2 TLB covers ~2 GiB in 32 MiB
  entries (measured on GP100 and GP104), while each card sweeps ~5.4 GB of weights per token. After subtracting the
  profiler's event overhead, the per-call excess is ~1.3–2.0 µs per 2 MiB page the call touches, in all four measured
  calls over a 45× range of sizes. Est. 2.7–4.1 ms per token if every page walk is exposed; placement and translation
  prefetch could recover est. +5–12 %. A pointer-chase probe and a spread-footprint benchmark mode settle it in an
  afternoon on one card.
- **Three ideas from other Pascal forks are worth porting:**
  - **The norm writes the Q8_1 activations directly, with the residual add folded in** (Mrinohk's P100 patches,
    shinbunbun/sbendhiba's Q8_1 cache, ArtyomITA's GTX 1080 fusions). Est. +4–6 %. It extends Q38 step 9. Built as
    Q100 (fork branch `pascal-p106-v2-normq8`): 287 fewer launches per S4-L0 token on six cards, logits identical;
    rig speed pending ([kernel note](pascal_mmvq_integration.md), section 8).
  - **Persistent layer kernels.** This is Q88; Mrinohk measured +36 % decode on 2 × P100 at Qwen3.6's MoE shape, which
    is nearly Flint's.
  - **Upstream PR #28874.** It is exact, and it matters whenever the scheduler runs one pipeline copy: Q89, or any `-ot`.
- **Samsung's sub-1-bit methods (NanoQuant, ICML 2026; LittleBit, NeurIPS 2025; LittleBit-2, ICML 2026) are not
  usable on this rig.**
  - **The format would decode well on sm_61.** A low-rank binary factorization keeps its scales on the matrix borders,
    so its matrix-vector product is pure mask plus dp4a, ~4–4.5 instructions per byte (est.).
  - **Quality fails, not decode.** At 1 bit per weight Llama-2-7B goes from perplexity 5.47 to 8.27–10.34, the best
    case being LittleBit-2.
  - **The rank caps quality on Flint's experts.** For 512-wide experts the full-rank factorization already costs
    ~1.25 bits per weight, so the method cannot trade up to the 4–5 bits the quality bar needs.
  - **The published kernel does not run here usefully.** It accumulates in FP16 half2 (1/64 rate on GP106).
  - **As a self-drafter it loses to a trained head.** A 49-layer self-drafter on a spare card pays the full model's
    per-layer launch costs.
- **A Pascal-native format is mostly a kernel question, and it is already answered for 4–8 bits.**
  - **The budget.** The card can issue ~5.3 slot-equivalents per byte at full bandwidth.
  - **K-quant layouts already fit it.** They decode in ~2.5–2.7 per byte with a Pascal kernel (est.), so a new GGUF type
    would save ≤ 3 % against them.
  - **What to build instead:**
    - a Pascal Q6_K kernel as the 6-bit rung (Q90 found Q6_K gentle), rather than a new Q6_0 type;
    - optionally a Q4_K kernel as a 4.5-bit rung;
    - shift- and conversion-free inner loops for the kernels that are still issue-limited.
  - **A 3-bit expert format fails twice.** It misses the quality bar by arithmetic (+0.07–0.12 KLD from the experts
    alone), and its bit-plane decode (~5.7 slot-equivalents per byte) leaves it issue-bound: per weight at best ~20 %
    faster than Q4_1 on 5 SMs, and in the realistic case no faster.
- **Nothing found in other engines beats the fork.**
  - ExLlama v1/v2, calm and the P100 forks compute in FP16.
  - vLLM-Pascal's W4A16 GEMV and PXA's codebook formats cost 2–3× the fork's instructions per byte.
  - NVFP4 emulation (nvfp4effma) is a clean design but has more decode work than Q4_x.
  - LUT GEMV (Kindi, T-MAC style) loses to dp4a at ≥ 2 bits on 5 SMs.
  - CMP-card "unlocks" address issue-rate throttles of Turing and Ampere mining cards that have no reported Pascal
    counterpart. A one-hour throughput calibration on a P106-090 should still confirm that, because single-token
    decode exercises dp4a at only ~3 % of its rate and could not reveal a moderate throttle (est.).

## Ranked ideas

Gains are for single-stream decode of S4-L0/G3 on six cards against the fork build, unless the row says otherwise.

| # | Idea | Source | Gain on the rig (est.) | Effort | First experiment | Kill criterion |
|---|---|---|---|---|---|---|
| 1 | Hide or avoid page walks: translation prefetch (touch every 2 MiB page of a call's weights early and in parallel; for experts right after the top-k), and TLB-aware placement (each card's dense tensors and head contiguous, experts as expert-contiguous records, hot experts first) | Jia et al. Table 3.1 (Pascal TLB reach); section 3 fit of the fork's profile against the benchmark | 1.4–3.3 ms per token, +5–12 % | Benchmark 1–2 days; fork 3–5 days | E1 TLB probe and E2 `--spread` mode on one rig card (2–3 h) | MoE calls with a ~5 GB footprint are not ≥ 10 µs slower than with ~0.5 GB |
| 2 | The norm writes Q8_1 (FP16 d; the fork's kernels compute block sums from int8) and absorbs the preceding residual add; Q/K/V and routed + shared gate/up skip their own quantization | Mrinohk patches 0003/0004 (P100); sbendhiba #10 (+1.17 % MoE), #19 (+0.70 %); ArtyomITA F1 (part of +8.0 % on a GTX 1080) | 240–340 fewer launches, 1.0–1.5 ms, +4–6 % | 2–4 days | Count launches per op in one profiled token; then attn_norm → Q8_1 with Q/K/V reading it | < 150 launches removed, or tg128 < +1.5 % |
| 3 | Persistent MoE layer (Q88), with a barrier that survives graph replay | Mrinohk 0018 (P100: tg128 85.6 → 116.7, KLD 0.0043 against a 0.0039 noise floor); killy-netsphere's persistent mat-vec (1080 Ti) | +10–25 % (graphs already remove part of what P100 gained) | Weeks | The existing [persistent_pascal](../../../bench/persistent_pascal/) prototype on the rig | Q88's own gate |
| 4 | Pascal Q6_K kernel as the 6-bit rung, instead of a new Q6_0 type | Q90 (G3 with non-head Q8_0 → Q6_K: KLD 0.118 against 0.104, −0.26 GB); section 5 | −1.3 ms of matrix time, +4–5 % at +0.006–0.014 KLD | 2–3 days (Q5_K kernel as template) | Add `q6_K` to `bench/mmvq_pascal` at Flint's shapes | < 120 GB/s on the dense K = 2048 shapes |
| 5 | Upstream PR #28874: copy user inputs before cross-device inputs, so the host does not wait for the previous card at every split | PR #28874 (CUPTI trace of a 3-GPU layer split) | Exact; matters only with one pipeline copy (Q89's one-copy build, any `-ot`); size unmeasured | 1 h | One-copy build ± the PR, tg128 and a host timeline on 6 cards | < 1 % difference |
| 6 | Instruction-throughput calibration of the P106-090 (FFMA, IADD, LOP3, SHF, BFE, PRMT, dp4a, I2F, F2F, LDS, SHFL) | xrip/cmp50hx-unlock issue-rate bench (method); CMP 90HX dp4a measured 29× throttled | None directly; decides how far 7, Q78 and Q80 can go | 1 h | One kernel per class at full occupancy on one card | — |
| 7 | Conversion- and shift-free inner loops: dp4a accumulation from the bits of 1.5·2²³ instead of I2F; powers of two folded into scales instead of SHR on bit planes | srose69/nvfp4effma (`gemv_nvfp4effma.cuh:83`); section 5 | +1–3 % on files with IQ4_XS/Q5_0/Q5_K; +5–15 % on 3–4-token steps where Q5_0/Q5_K/IQ4_XS are issue-limited | 1–2 days | SASS counts, then the bench at 1 and 4 tokens | < 3 % on the 4-token Q5_0 gate/up shape |
| 8 | K-quant rungs in the allocation search: experts in Q5_K (same bytes as Q5_0, lower error) or Q4_K (4.5 bits), priced at the fork's measured GB/s | llama.cpp perplexity table (Llama-3-8B: Q5_K_S 0.0166 against Q5_0 0.0222 KLD at 5.5 bits); kquant note | 0–3 % | Quant machine 1 day; Q4_K kernel 2 days | G3 with experts in Q5_K, and in Q4_K, KLD against BF16 | Q4_K experts cost > 0.02 KLD per ms saved; Q5_K gains < 0.003 |

For prompt processing (Q80) two sm_61 MMQ retunes exist:
- cycle-five/pascal-mmq-autorunner: on a P40, Q4_0 MMQ `I` 64 → 128 gave pp512 1009.5 → 1174.2 tok/s (+16.3 %) with
  identical perplexity.
- madsteph74/pascallama.cpp: `mmq_y` = 96 in every row, published with no A/B.

## 1. Pascal-focused llama.cpp forks and projects

Every project below was read at its source (patch files, commits, READMEs with raw numbers). None of them measured
a card with fewer than 10 SMs, and only the Bonsai and P40 projects run compute 6.1. (llama.cpp's scoreboard has a
5-SM Quadro P1000 and a 6-SM GTX 1050 Ti, but both have half the P106-090's bandwidth per SM; quant_speed_d section 3.)

| Project | Card | What it changed | Measured | In the fork |
|---|---|---|---|---|
| qr0n/p100-llama-cpp (18 patches) | 2 × P100 (sm_60) | 4 rows per block for K-quants in a GP100 MMVQ table; K-quant mins from the stored FP16 sum instead of `dp4a(0x01010101, u)` per row; an FP16 HFMA2 mat-vec path; AllReduce tuning for `-sm tensor` | Llama-3.1-8B Q4_K_M 39.28 → 52.22 tok/s (+32.9 %) from rows + sums; FP16 path 2.03×; CUDA graphs −1.4 % with tensor split; forced MMVQ fusion −2.1 % | Rows: yes (R rows per lane group). The FP16 sum would bring back Flint's overflow; the fork computes FP32 sums once per activation block. FP16 path: not applicable (1/64 rate) |
| ArtyomITA/bonsai2-ternary-pascal (one patch, +987 lines) | GTX 1080 (sm_61) | CUDA graphs on Pascal for ≤ 8 columns only; an ordered PQ2_0 mat-vec; top-k in two launches; FWHT fused with Q8_1 quantization; residual add + RMSNorm + mul extended to Pascal with an alias check that allows only same-address in-place outputs; RMSNorm + mul + SwiGLU; F32 copies with fastdiv | Ternary-Bonsai-2-9B 55.9 → 84.4 tok/s (+51 %), bit-identical; fusions 976 → 755 launches (+8.0 % in ablation); FWHT + Q8_1 +4.5 %; staged or 2-row variants of its mat-vec neutral or −6 % in the model | Graphs: yes, at every width. Add + norm fusion, fastdiv copies, FWHT + Q8_1: no (FWHT only matters for a rotated file) |
| killy-netsphere/bonsai2-1080ti (31 files, +4,190 lines) | GTX 1080 Ti, memory overclocked | Row-lane PQ2_0 mat-vec with Q8_1 activation tiles staged in shared memory and the next tile prefetched into registers; a persistent grid-stride variant that stages the whole activation vector once per block; graph key hashed from node count and first/last shapes; batched scheduler input copies; fused SwiGLU/add-RMSNorm + FWHT writing a Q8_1 side copy | Mat-vec 300–305 GB/s tiled, 330–370 persistent (their bench; their streaming ceiling is ~335, so part of it is cache); 42.6 tok/s plain decode, 86.4 with an MTP drafter | Weight staging: yes; activation staging: opt-in on the multi-token branch; persistent mat-vec: no; graph key: no (Q52 covers it) |
| Mrinohk/qwen36-p100-patches (35 patches) | 2 × P100 (sm_60), layer split | Persistent cooperative kernels per layer half (attention/GDN half and MoE half: norm, router, top-8, shared and routed experts, weighted sum, residual), software grid barriers between phases, F32 activations; 2–6-token verify versions; add + RMSNorm + mul (0003); the fused norm writes Q8_1 into a cache the mat-vec reads (0004); staged Q8_0 loads (0010) | tg128 68.0 → 125.7 tok/s overall; the persistent kernels alone 85.6 → 116.7 (+36 %); MTP 93.4 → 183.1; KLD 0.0043 against a 0.0039 noise floor. A 17 KB FP16 kernel slowed the other kernels by 0.43 ms per token, attributed to the instruction cache | Staging: yes. Persistent kernels: prototype only ([persistent_pascal](../../../bench/persistent_pascal/)). Norm → Q8_1, add + norm: no |
| sbendhiba/llama-cpp-pascal-patches (41 patches, from shinbunbun's set) | 2–3 × P100 | Q8_1 activation cache (#10), residual add fused before RMSNorm (#19), add → unary → mul (#20), sibling-node fusion (#18), lazy scheduler reset (#21), copy with fastdiv (#16), graphs on sm_60 (#50) | #10 +1.17 % MoE / +0.88 % dense; #19 +0.70 % (off by default: it skips the fusion memory check and gave non-deterministic MoE output on one version); #18 +1.06 %; #21 +0.94 %; copies 11.26 → 4.96 µs; graphs on sm_60: MoE layer split −3.45 % | None of these, except the norm's register caching |
| Frostbite1536/P100 | none yet | sm_60 half2 mat-vec, FA FP16 accumulator flush | Static SASS ALU instructions per weight on sm_61: Q4_0 1.3, Q8_0 1.1, Q4_K 2.7 | Not applicable |
| madsteph74/pascallama.cpp | P40 + P6000 | MMQ `mmq_y` = 96 for every row; nine `#pragma unroll` re-enabled | Prefill only, no A/B | No |
| cycle-five/pascal-mmq-autorunner | P40 | Search-based MMQ autotuner; Q4_0 row `I` 64 → 128 | pp512 +16.31 %, tg128 unchanged | No (Q80) |
| ik_llama.cpp | P40, P100 | #2701 (2 rows per block: P40 +0.3 %, DeepSeek −4.5 %), #2721 (row sums in registers: P100 +5.8 %, P40 SASS identical), #2740 (2 warps for Q2_K/Q3_K: P40 +7.9 %), #2509 (`__byte_perm` table for IQ4_K: P40 +113 %); `docs/pascal-volta.md` (no CUDA graphs below Ampere) | as listed | The fork's kernels cover these, except that the stock `mul_mat_vec_q_moe` (used for 2–8 tokens on `pascal-p106-v2`, 5–8 on the multi-token branch) reads `tmp[threadIdx.x]` (`mmvq.cu:989`), the runtime-indexed local array that #2721 and PXA found spilling to local memory |
| llama.cpp PRs not in the earlier notes | — | #28874 (input copy order in `ggml_backend_sched_compute_splits`), #31047 (linearized MoE layout for MMQ/MMF, prefill; P40 1.40× at ubatch 64), #26487 (blocking sync), #29393 (RMSNorm + scale, in the fork) | as listed | #28874 and #31047: no |
| Other repos | — | Kmic-68/llama.cpp: activation staging called "the largest single win" on P100; boetti-dev/p40-kernel: `ld.global.nc` + `prefetch.global.L2` + explicit dp4a 1.92× slower (64 registers, 50 % occupancy); Niko1221/Strata: P40 rows-per-warp table, +6–10 % | as listed | Activation reuse across rows: yes; no cache-hint gains anywhere on sm_61 |

What carries over from these projects:
- **Fusions that remove the activation quantization launches.** Every fast P100 or GTX 1080 fork ends with the norm
  producing the mat-vec's input format, and with the residual add inside the norm kernel. Both forks that did it safely
  allow an output to alias an input only at exactly the same address.
- **Persistent kernels.** Mrinohk's layer-half kernels match Flint's MoE geometry except for 8 instead of 6 experts.
  Their barrier keeps a host-side epoch, which a captured CUDA graph would freeze (est.), so a port needs a barrier
  whose epoch lives on the device. The fork's prototype avoids grid barriers altogether (dependency counters).
- **Graph handling.** Graphs only up to 8 columns (ArtyomITA: graphs at every width cost −4.9 % prefill on a GTX 1080),
  and a graph key that includes shapes when several widths alternate (killy, and upstream #31004 in Q52).
- **Negative results that save time.** Cache hints and prefetch instructions did not help on sm_61 (boetti-dev; Q48
  shows the fork's loads already go through the read-only path). Staged variants of an already-coalesced mat-vec gained
  nothing inside a model (ArtyomITA). An aligned weight copy bought +2.4 % for 901 MiB of VRAM.

**Mining-card projects.** The 2026 CMP activity is about Turing and Ampere mining cards:
- **What is throttled on CMP cards.** The SM issue rate is clamped per instruction class through issue-rate selector
  registers, opened by patched open kernel modules (Wh1stle05/cmp90hx: FP32 0.72 → 18.24 TFLOPS on a CMP 90HX).
  Rhonstin/beellama-cmp90hx measured DP4A 29× throttled and FFMA 14–15× on a CMP 90HX, and replaced dp4a with four
  IMADs: decode +47.6 %. Cyridd/cmpunlocker: a CMP 40HX has no FP32 lock, but dp4a and WMMA are clamped, and the link
  can be retrained from Gen1 to Gen2.
- **Pascal mining cards.**
  - No measurement of instruction throttling on the P106-090, P106-100, P104-100 or P102-100 was found, and the unlock
    mechanisms need GSP-era secure boot and the open kernel modules, which do not support Pascal.
  - No PCIe-generation unlock for P104/P106 exists.
  - Prefill-side projects for P104 rigs (sergey-automation/TurboPrefill: 10 × P104-100 on Gen1 x1, GPT-OSS-120B prefill
    77 → 345 tok/s) leave decode unchanged.

## 2. Other engines and ideas on old NVIDIA cards

| Engine or project | Pascal status and technique | Measured (hardware) | Useful on a 5-SM sm_61 card |
|---|---|---|---|
| ExLlama v1/v2, TabbyAPI | All math in FP16; an FP32 path was declined (ExLlamaV2 #185) | 34B 4.0 bpw on a P40: 1.19 tok/s (#40) | No; no P40 discussion with numbers in TabbyAPI |
| ExLlamaV3 | Issue #512 (open, no maintainer reply) proposes an sm_61 lane around `exl3_gemv_int8_kernel.cuh`: the codebook value is affine in a byte-sum of `idx * 0x83DCD12D`, so one dp4a per weight after a window extract and a 32-bit multiply | none | No: ~5–7 instructions per weight, with the multiply emulated on sm_61 (est.) |
| vLLM-Pascal (Mikec78660) | W4A16 split-K GEMV, per-nibble FP32 dequant and FMA into ≤ 4 rows | P40: 1.69 → 0.24 ms per GEMM at M = 1 ("213 GB/s"); 2 × P40 Qwen3.8-27B AWQ with MTP 13.2 tok/s against llama.cpp Q4_0 13.6–13.7 | No: ~12 instructions per byte (est.); Python and torch on a 4 GB, no-AVX host |
| Aphrodite (now Sonar), MLC-LLM/TVM, candle, mistral.rs, PowerInfer, gpt-fast | Pascal tolerated or unsupported; candle's quantized CUDA kernels are an early-2024 llama.cpp MMVQ port; MLC runs sm_61 without its CUTLASS parts and its maintainers will not optimize old cards (#2100); gpt-fast needs Triton (sm_80+) | no Pascal decode numbers | No |
| fastllm | FP32 path for GPUs without tensor cores; its GGUF kernels use dp4a and `__ldcs` (evict-first) weight loads | P40: ChatGLM2 int4 15.0 → 27.6 tok/s (PR #449); Qwen2.5-7B AWQ 20.29 → 35.3 (PR #537) | The evict-first hint is worth one benchmark variant (E3); the engine is not |
| zeux/calm | gf4: eight 3-bit values and an FP8 e5m2 scale per 32 bits; e5m2 → half by one shift; half arithmetic | RTX 4090 only | No (FP16 math) |
| PXA (poisonxa16/pxa) and pxq_llama | PXQ: 64-row panels, 16-entry non-uniform books, 4-bit sub-scale indices per 16 values into a fixed table; PXQ2/3 books as PRMT constants; PXQN (rotated, new books) ships closed; a multi-column verify GEMV with its local-memory pitfall fixed | P100: "uniform PXQ2 moves 39 % fewer bytes than uniform PXQ4 and decodes at the same speed (18.50 vs 18.01 t/s)"; 1080 Ti same-quant engine control 53.9 against ik's 52.2 tok/s; int8 dp4a prefill tile 251 → 709 tok/s | No: decode is IQ4_XS-class (~5 instructions per byte for PXQ4, 8–9 for PXQ2/3, est.); PXQ2 = PXQ4 speed on P100 shows the same issue limit as here |
| ahtx/kindi | Ternary LUT GEMV: 5 trits per byte, 24 tables of 243 entries per 128 columns in shared memory, one LDS + FADD per byte; FWHT fused into the norm | GTX 1080, Ternary-Bonsai-2-27B: 27.9 against 19.4 tok/s (Prism llama.cpp) | No for Flint (ternary needs retraining); on 5 SMs random-index LDS with bank conflicts caps it near ~70 GB/s (est.) |
| srose69/nvfp4effma | NVFP4 on sm_50–sm_90: four nibbles → int8 with two PRMT tables, a sign-replicate PRMT and one LOP3 (6 instructions per 4 values); E4M3 scale by a shift and one exact FMUL; dp4a started from the bits of 1.5·2²³ so no I2F is needed | GTX 1080: 8192² GEMV 0.139 ms ("271 GB/s", 85 %); 8 columns cost 1.14× one | As a storage format no (~4.3–4.7 instructions per byte, est.); the I2F-free epilogue yes (idea 7) |
| microsoft/BitNet GPU kernel | W2A8: 16 two-bit values interleaved so one `lop3` yields four int8, `__vsubss4` then dp4a; built for compute_80 | A100 only | Its interleave is what Q2_K already does; `__vsubss4` is several instructions on sm_61 |
| T-MAC, LUT-GEMM, CodeGEMM, any-precision LLM | Table lookup or bit-plane GEMV; T-MAC is CPU/NPU, LUT-GEMM builds for sm_80, CodeGEMM measured on A100, any-precision stores bit planes for nested 3–8-bit models | none on Pascal | No (section 4.2) |

Engines worth nothing here are still informative about this card's limits:
- PXA's P100 result shows that halving the bytes does not speed up a lookup format.
- legacy-gpu-llm-notes measured the same on a GTX 1070: PQ2_0 (7.2 GB) at 14 tok/s against PTQ1_0 (5.9 GB) at 8.1.

Both are the instruction-bound regime that quant_speed_b predicted for this card, seen on cards with more issue
headroom than ours.

## 3. The microbenchmark-versus-model gap

### What the benchmark does not reproduce

[bench/mmvq_pascal](../../../bench/mmvq_pascal/) (`src/main.cu`):
- **MoE shapes.** Each MoE shape gets one weight tensor per matrix holding all 256 experts (`ncopies = 1`), and 64
  random sets of 6 experts. The 64 calls are captured in one CUDA graph, which is launched 3 times to warm up and then
  30 times for the timing.
- **Dense shapes.** These cycle over copies totalling ≥ 64 MB and at most 1.2 GB.
- **Footprint.** Every timed working set therefore stays under ~1.2 GB and is touched again and again.
- **The real model.** A card holds ~8 MoE layers of S4-L0, about 5.4 GB:
  - `ffn_gate_exps` and `ffn_up_exps` of 176 MiB each in Q5_0, `ffn_down_exps` of 272 MiB in Q8_0;
  - plus ~23 MiB of dense tensors per layer;
  - plus the 312 MiB output head on the last card.
- **Per token.** A card reads ~370 MB of this, scattered over the whole 5.4 GB, and touches each expert page only when
  that expert is chosen again.

### Pascal's TLBs (primary data)

- Jia et al. measured with pointer chases (Table 3.1 of the Volta report; repeated in the Turing report):
  - **Kepler (K80) and Maxwell (M60):** three TLB levels: L1 ~2 MiB in 128 KiB entries, L2 ~128 MiB, L3 ~2 GiB in
    2 MiB entries.
  - **Pascal P100 (GP100) and P4 (GP104):** two levels. L1 TLB ~32 MiB of coverage in 2 MiB entries; L2 TLB
    ~2,048 MiB in 32 MiB entries.
  - **Volta and Turing:** the L2 TLB grew to ~8,192 MiB.
  - The L2 data cache is physically indexed, so every L2 access goes through the TLBs.
- **Cost of a miss.** On a V100 the first access to a page costs 1,029 cycles (TLB miss and L2 miss) against 375 for an
  L2 miss with a TLB hit; on a T4 616 against 296.
- **GP106.** Not measured. It is assumed here to match GP104 (est.).
- **Page sizes.** `cudaMalloc` uses 2 MiB pages (Mei and Chu). With unified memory the pages are 64 KiB, and a GTX 1080
  Ti then shows an L1 TLB reach of ~1 MB and three levels (Nayak et al., ASIACCS 2021, as summarized by IISc; the paper
  was not reachable). `GGML_CUDA_ENABLE_UNIFIED_MEMORY` must therefore stay off on the rig.
- **No larger pages.** Pascal's MMU has no page larger than 2 MiB; `cudaMallocAsync` and the VMM API allocate at that
  granularity. Bigger pages are not available, so only the number of pages touched and the timing of the walks can
  change.

What this means for one card and one token (est.):
- **Pages touched per MoE layer.**
  - A routed expert's gate or up slice is 0.69 MiB and its down slice 1.06 MiB, so at random alignment they touch 1.34
    and 1.53 pages of 2 MiB.
  - Six experts touch ~25 expert pages (16.1 for gate/up, 9.2 for down).
  - The dense tensors of the layer add ~13.
- **Per token.** ~39 pages per MoE layer, ~310 per card, plus 157 for the head card; ~2,050 over all cards.
- **32 MiB regions.** A card touches ~107 regions per token against 64 L2 TLB entries. Even the dense tensors, read on
  every token, are evicted before they come round again, which the head's gap confirms (below).

### The gap is proportional to pages

Subtracting the profiler's own overhead per launch (5–8 µs, taken as 6.5) from the measured excess over the benchmark:

| Call (S4-L0) | 2 MiB pages touched (est.) | Excess over the benchmark | Minus event overhead | Per page |
|---|---|---|---|---|
| Attention q, Q4_1, 5.0 MiB | 3.5 | ~13 µs | ~6.5 µs | ~1.9 µs |
| Routed down, Q8_0, 6 experts | 9.2 | ~25 µs | ~18.5 µs | ~2.0 µs |
| Routed gate/up, Q5_0, 6 experts × 2 matrices | 16.1 | ~33 µs | ~26.5 µs | ~1.6 µs |
| Output head, Q8_0, 312 MiB | 157 | ~210–240 µs (~11 % of 2.2 ms) | ~205–235 µs | ~1.3–1.5 µs |

A constant ~1.3–2.0 µs per page across a 45× range of call sizes fits address translation: one exposed walk per page.
Under this card's full streaming load, a walk of two or three dependent page-table reads at the 0.3–0.5 µs DRAM latency
the walk competes with plausibly takes that long (est.). Unloaded, Volta and Turing need 0.2–0.5 µs per walk.

The other suspects:
- **Clocks.** A card working 4–8 ms of every 26.8 ms runs cooler and boosts higher than the benchmark's continuous
  stream, which predicts the opposite sign.
- **DRAM row and bank effects.** Expert slices are 0.7–1.1 MiB contiguous runs read by neighbouring warps, as in the
  benchmark, so they predict no difference.
- **L2 data cache.** Weights stream through it in both cases, and the activation vector is L2-resident in both.
- **Instruction cache.** Each real call follows a different kernel, while the benchmark repeats one. On P100 a large
  kernel measurably slowed its neighbours (Mrinohk 0010). This predicts a constant per call, not one proportional to
  pages; it may sit inside the per-launch term above.
- **The event profiler.** Its 5–8 µs per launch is the constant term. It cannot produce the head's ~200 µs.

What it costs per token (est.):
- **Every page walk exposed:** ~2,050 pages × 1.3–2.0 µs = 2.7–4.1 ms of a 26.8 ms token.
- **Only expert pages missing:** ~1,190 pages, 1.5–2.4 ms.
- **Against the unexplained time:** both fit inside the ~6–8 ms by which the token exceeds the benchmark-based matrix
  time (14.9 ms) plus the non-matrix work and the hops.

### Fixes

1. **Translation prefetch.**
   - **Dense calls.** In the prologue of block (0, 0), lanes issue one load per 2 MiB page of the call's weights. A
     128-thread block covers the head's 157 pages in two rounds.
   - **Expert calls.** The fused top-k kernel, which writes the expert ids, issues the loads for the selected experts'
     gate, up and down pages: ~25 loads per layer, issued before the gate/up call starts, so the walks overlap each
     other instead of queueing one expert after another, and the down-projection's walks finish during gate/up.
   - **Mechanics.** The loaded values are consumed only at the end of the kernel, in a branch that is never taken, so
     the warp keeps issuing while the walks run. A real load is used rather than `prefetch.global.L2`, which may be
     dropped on a TLB miss.
   - **What it buys.** If the GPU's page walker serves several walks at once, all walks of a call overlap instead of
     queueing behind the stream. If it does not, only placement helps.
2. **Fewer pages per call.** Store each expert's gate, up and down rows as one record of 2.44 MiB in S4-L0, so 6 experts
   touch ~13.3 pages instead of ~25 (breakthrough candidate 9's layout). That saves ~12 walks per layer, est. 0.7–1.1 ms
   per token.
3. **TLB-aware placement.**
   - **Layout per card.** All dense tensors of the card's layers first, then the head, then the expert records with each
     layer's most-used experts first (Q51: a layer's top 25 % of experts take 44 % of reads).
   - **Effect.** The card then touches ~6 dense regions, ~10 head regions and ~40 expert regions per token, fewer than
     the L2 TLB's 64 entries, so dense tensors and the head stay translated from token to token (est.).
   - **How.** This needs a loader change: a second weight buffer per card, or a creation order by class. An `-ot`
     override would also switch off pipeline parallelism.
4. **More cards.** Nine cards hold ~3.6 GB each instead of 5.4. That helps only if the TLB's real organization is closer
   to 1,024 entries of 2 MiB than to 64 of 32 MiB. E1 tells which.

### Experiments

All run on one rig card, with no model and no system change. E1 and E2 decide whether the rest is worth doing.

- **E1. TLB probe (new `bench/tlb_probe`, 30–60 min).**
  - **Method.** A Jia/Mei-style pointer chase over 16 MiB to 5 GiB at 2 MiB and 32 MiB strides. Then a chase through N
    random 2 MiB pages spread over 5 GiB (N = 16 … 2,048).
  - **Load.** Each runs alone, and again while four SMs stream reads.
  - **Output.** The L1/L2 TLB reach of GP106, the walk latency unloaded and loaded, and whether capacity is limited by
    pages or by 32 MiB regions.
- **E2. `--spread N` in `bench/mmvq_pascal` (1–2 h).**
  - **Method.** N separate weight sets per shape (N = 1, 2, 4, 8, 16; up to ~5 GB), cycled call by call, with fresh
    random expert ids per call instead of 64 replayed sets.
  - **Prediction.** Under the TLB model, per-call time is flat while the footprint stays under the L2 TLB reach and
    rises by ~1.3–2.0 µs per page beyond it. That is +20–30 µs on the MoE shapes, +10 % on the head.
  - **Kill.** Under 10 µs on the MoE shapes at 5 GB.
- **E3. `--interleave`.** The graph runs one MoE layer's sequence of shapes in Flint's order (q, k, v, o, router-like,
  gate/up, down, shared expert) at a TLB-warm footprint. This measures the per-call cost of switching kernels
  (instruction cache, ramp-up). A variant with evict-first weight loads (`ld.global.cs.v4`, as fastllm uses) shows
  whether keeping code and activations in L2 changes it.
- **E4. Translation prefetch under `--spread`.** Three variants: (a) the block-(0, 0) touch loop; (b) a touch kernel one
  graph node ahead; (c) the expert pages touched by the previous call. Gate: recovers ≥ 2/3 of E2's excess.
- **E5. Expert records and hot-first order under `--spread`.**

If E2 and E4 pass, the fork gets (a)/(b) in `mmvq-pascal.cu` and the top-k kernel, then the loader change. tg128 then
measures it on 6 and 9 cards.

## 4. Quantization research

### 4.1 Samsung's sub-1-bit methods: NanoQuant, LittleBit, LittleBit-2

**NanoQuant** (Chong, Kim, Kim, Choi; arXiv 2602.06694, v3 2026-06-14, ICML 2026; code SamsungLabs/NanoQuant, Apache-2.0,
three commits, kernels added 2026-05-28):
- **Representation (Eq. 1).** W ≈ s₁ ⊙ (U V^T) ⊙ s₂^T with U ∈ {±1}^(d_out × r) and V ∈ {±1}^(d_in × r).
  - s₁ is a per-output-channel scale, s₂ a per-input-channel scale; the code adds an optional per-rank scale between
    the two products.
  - Storage is r(n + m) + 16(n + m) bits per matrix (Appendix F.5).
- **Fitting.** Latent binary matrices and scales are initialized by an ADMM solver, then refined by block-wise and
  model-level reconstruction on calibration data. It is post-training: Llama-2-70B takes 13 h on one H100.
- **Decode** (`src/nanoquant/kernel/cuda/gemv_kernel.cu`): two GEMV stages.
  - Stage 1 multiplies the scaled input by V: each sign bit flips the sign bit of an FP16 activation, followed by
    `__hfma2` with the input scale.
  - Stage 2 multiplies the rank-r intermediate by U with `__hadd2`.
  - Everything is half or bfloat16 ("Support only Half or BFloat16"), two rows per warp, pipelined 128-bit loads. The
    build compiles for the local GPU's architecture.
- **Quality** (WikiText-2, Table 2):
  - Llama-2-7B: FP16 5.47; 10.34 at 1.00 bit, 12.20 at 0.80, 16.66 at 0.55.
  - Llama-2-70B: 3.32 → 6.52 at 1.00 bit.
  - Qwen3-8B: 12.47 at 1 bit.
  - No MoE model is evaluated.
- **Speed.**
  - RTX 3050: Llama-2-7B at 0.55 bit 134.1 tok/s (sequence 32); Llama-2-70B fits 8 GB at 20.11 tok/s.
  - Jetson TX2 (Pascal GP10B, sm_62, which has fast FP16, unlike GP106): "up to a 12.2× speedup" over PyTorch FP16.

**LittleBit** (Lee, Kim, You, Kim; arXiv 2506.13771, NeurIPS 2025) and **LittleBit-2** (Lee, Kim; arXiv 2603.00042,
ICML 2026); code SamsungLabs/LittleBit, CC BY-NC 4.0:
- **Representation.** W ≈ diag(h) U_sign diag(ℓ) V_sign^T diag(g), plus a second, residual factorization of the same
  form.
- **Training.** Quantization-aware: 5 epochs of KL plus intermediate-MSE distillation against the FP16 model.
  LittleBit-2 changes only the initialization: a rotation of the SVD factors by joint iterative quantization toward
  the binary hypercube, "zero inference overhead".
- **Quality** (WikiText-2):
  - Llama-2-7B, LittleBit: 9.08 at 1.0 bit, 10.47 at 0.55, 15.92 at 0.1.
  - Llama-2-7B, LittleBit-2: 8.27, 9.65 and 14.70.
  - Llama-3-8B: 6.10 → 11.53 at 1.0 bit (LittleBit-2).
  - For comparison, EfficientQAT at 2 bits: 7.17 (Llama-2-7B).
- **Code.** The repository holds PyTorch modules and a bit packer, no CUDA kernel.
- **Speed.** The paper reports a single Llama-2-70B MLP layer 11.6× faster than cuBLAS FP16 at 0.1 bit (A100, custom
  1-bit GEMV), and Llama-2-7B at 0.1 bit 82.6 → 203.2 tok/s end to end.

On this rig (est.):
- **Decode would fit sm_61.** Because both scales sit on the matrix borders, the per-input scale can be applied to the
  activations once and the result quantized to int8 once per matrix, shared by all rows. A row's dot product is then
  Σ sᵢaᵢ = 2·Σ_{bᵢ=1} aᵢ − Σaᵢ.
  - **Per 32 sign bits.** `dp4a.u32.s32(w & (0x01010101 << j), t_j)` for j = 0…7 over activations transposed once per
    block, with one accumulator per j combined by eight shifts at the end of the row: 8 LOP + 8 dp4a.
  - **Per byte.** That is 0.5 instructions per stored bit, ~4–4.5 per byte with the activation work, all full rate:
    under the ~5.3 budget, so the decode is roughly bandwidth-bound.
  - **For Flint.** At 1 bit per weight Flint's ~2.5 B active parameters would read ~0.31 GB per token, against G3's
    2.33 GB.
  - **The published kernel is the problem.** FP16 runs at 2 results per SM per clock on GP106, ~0.2 s per token for a
    7B model at 1 bit.
- **Quality is out of reach.** The best 1-bit result raises Llama-2-7B's perplexity by 51 %; the bar allows ~4 %.
  Nothing in either paper measures a model with an outlier-heavy, routing-sensitive MoE like Flint.
- **The rank caps Flint's experts.** For a 512 × 2048 expert the full-rank factorization (r = 512) already costs
  512 × 2560 / (512 × 2048) = 1.25 bits per weight. The method cannot trade up to the 4–5 bits the bar needs, and at
  ≤ 1 bit it works at r ≈ 400 of 512, where the low-rank prior helps little.
- **The drafter idea does not survive either.**
  - A 0.55–1 bit copy of Flint fits one spare card (2.8–5 GB) and could guess for pipelined speculation (Q87).
  - Its 49 layers still pay ~1,300 launches per token, est. 6–9 ms on one card, above the six-card cycle of ~5 ms.
  - Its acceptance is unknown.
  - A trained two-layer pipeline drafter (breakthrough candidate 1) costs ~1.5–2 ms per guess.
- **Cost on the quant machine.** NanoQuant would need est. days for Flint's ~36,000 expert matrices.

**Verdict.** Not for this rig: the format's decode would fit sm_61, but its quality is an order of magnitude outside
the bar, its rank cannot rise to where the bar is, and its kernel needs fast FP16.

### 4.2 Other 2025–2026 formats for low-end hardware

Per format family, the decode on sm_61 and its cost, against the fork's Q4_1 (~2.35 slot-equivalents per byte with
staging) and the card's ~5.3 slot-equivalents per byte at full bandwidth:

| Family | Examples | Decode on sm_61 | Slot-equivalents per byte (est.) | Verdict |
|---|---|---|---|---|
| Binary or ternary with scales | BitNet b1.58, PrismML Bonsai (PQ2_0, PTQ1_0), llama.cpp TQ1_0/TQ2_0 | 2-bit fields by mask and dp4a; ternary via base-3 bytes needs lookups | 4–6 | Needs a model trained for it; GTX 1070 data show fewer bytes running slower |
| Low-rank binary factorization | NanoQuant, LittleBit(-2) | Mask + dp4a, scales only on matrix borders | ~4–4.5 per stored byte | Quality (4.1) |
| Vector codebooks | AQLM, QuIP#, VPTQ, PXQ/PXQN | Shared-memory or PRMT lookups per weight or pair | 5–9 | Lookups make the card issue-bound; PXQ2 = PXQ4 speed on P100 |
| Trellis | QTIP, EXL3 (ExLlamaV3 #512's sm_61 proposal) | Hash multiply per weight (3 XMAD on sm_61), then dp4a | 13–28 at 2–3 bits | Rejected (breakthrough candidates); #512 adds nothing that changes it |
| Learned non-uniform 4-bit | IQ4_NL/IQ4_XS, any4, PX16 books | 16-entry `__byte_perm` table | ~5.3 (IQ4_XS in the fork: 104 GB/s) | Slower per weight than Q4_1 (5.1 against 4.3 ps) |
| Microscaling FP4 emulated with integers | MXFP4, NVFP4 (llama.cpp; nvfp4effma) | Two-table PRMT + sign select, shift-decoded scales | 4.3–4.7 (nvfp4effma), 11–15 (llama.cpp's loop) | No gain over Q4_x per weight; the makers' NVFP4 file stays a test of the cost model only |
| LUT GEMV, bit-serial | T-MAC, LUT-GEMM, CodeGEMM, Kindi, any-precision LLM | Tables of activation partial sums in shared memory; one LDS (32 per SM per clock, random banks) per 1–4 weights | LDS-bound: ≤ ~136 GB/s-equivalent at 4 bits before bank conflicts, ~45–90 after | Loses to dp4a at ≥ 2 bits on 5 SMs; wins only for ternary or ≤ 2-bit models |
| Coarser scale groups | Q8_0 per 128, Q5_0 per 64 | Fewer scale loads and conversions | slightly lower than today | Covered by unconventional ideas, section 5 |

### 4.3 What the bits-versus-error ladder says

llama.cpp's Llama-3-8B table (mean KLD against FP16, RTX 4090; file sizes as published; no importance matrix unless
marked):

| Type | File (GiB) | KLD |
|---|---|---|
| Q8_0 | 7.96 | 0.0014 |
| Q6_K | 6.14 | 0.0055 |
| Q5_K_S | 5.21 | 0.0166 |
| Q5_0 | 5.21 | 0.0222 |
| Q4_K_S, with importance matrix | 4.37 | 0.0320 |
| IQ4_XS, with importance matrix | 4.14 | 0.0363 |
| Q4_1 (simple min/max rounding without an importance matrix) | 4.78 | 0.0717 |
| Q3_K_M, with importance matrix | 3.74 | 0.0844 |
| IQ3_S, with importance matrix | 3.42 | 0.1113 |

Two things follow for Flint:
- **A 6-bit rung costs about a quarter of what Q5_0 costs.** Q90's Q6_K proxy agrees.
- **K-quant super-blocks beat the legacy blocks at equal bits by ~25 %.** Q5_K_S against Q5_0, files of the same size.
  - On Flint the measured difference is smaller: UD-Q4_K_XL scores 0.120 against UD-mirror v1's 0.124–0.129 at the
    same budget (Q7, Q12).
  - With the fork's kernels, Q5_K streams at 128 GB/s against Q5_0's 138, so the two routes end at the same speed (the
    [K-quant note](kquant_decode_pascal.md)).

## 5. A Pascal-native format: design sketch

### The budget

- **Issue capacity.** 5 SMs × 128 lanes × ~1.73 GHz = 1.107 T lane-instructions per second; ~83 % usable gives ~0.92 T.
- **Per byte.** At the 179 GB/s streaming ceiling that is ~5.1–5.3 slot-equivalents per weight byte.
- **Comfortable target.** A kernel at ≤ ~3 per byte keeps enough warps issuing loads to stay bandwidth-bound.
- **Consistency with quant_speed_b.** That note's model gives 6.4 instructions per byte at 173 GB/s and 100 %
  efficiency; the figures here use its 83 % and the 179 GB/s ceiling the benchmark measured.

Per 32 weights per row, with two rows sharing the activation work (R = 2) and staged loads costing three instructions
per 16 bytes (global load, shared store, shared load), from the fork's code:

| Kernel | Bytes / 32 w | Weight decode | dp4a | Shifts (×2) | Conversions (×4) | Activation share | Slot-eq. per byte (est.) | Measured GB/s |
|---|---|---|---|---|---|---|---|---|
| Q8_0 (fork) | 34 | ~4 PRMT for odd blocks | 8 | 0 | F2F d, I2F sum: 8 | ~6 | ~1.1 | 150–165 |
| Q4_1 (fork) | 20 | 8 LOP | 8 | 1 SHL: 2 | 2 F2F, I2F: 12 | ~9.5 | ~2.35 | 145 |
| Q5_0 (fork) | 22 | 8 LOP + 8 LOP + ~2.5 PRMT | 16 | 8 SHR + 1: 18 | F2F, I2F: 8 | ~19 (transposes) | ~4.1 | 138 |
| Q6_K (proposed, Q5_K's template) | 26.25 | 8 LOP (low nibbles) + 8 LOP (2-bit plane by masks) | 16 | ~6 combines: 12 | magic-number scale and sum: ~4 | ~10 | ~2.7 | est. 130–150 |
| Q4_K (proposed) | 18 | 8 LOP + ~7 for 6-bit scales | 8 | ~2 | ~6 | ~12 | ~2.6 | est. 125–140 |
| "P3", 3.5 bits (2-bit + 1-bit planes) | 14 | 8 LOP + 8 LOP | 16 | 2-bit combines 6, 1-bit plane 16 | 8 | ~15 | ~5.7 | est. 90–125 (issue-bound) |

What this shows:
- **Layout is not the limit for 4–8 bits.** The K-quant super-blocks (144 B for Q4_K, 176 B for Q5_K, 210 B for Q6_K)
  are 16- or 2-byte aligned and decode with masks, so a Pascal kernel keeps them at ≤ ~2.7 per byte.
- **The most a new type could buy is ~0.3–0.5 slot-equivalents per byte.** That comes from a from-scratch layout with u8
  sub-scales, power-of-two super-blocks and no staging, and is invisible once a kernel is bandwidth-bound. Est. ≤ 3 %
  on small shapes, 0 % on large ones, for a type that no other tool reads.
- **Fewer bits do not mean less time on this card.** Bit-plane extraction costs per weight, not per bit. A 3.5-bit
  format lands at ~5.7 per byte, issue-bound: 0.44 B per weight at 90–125 GB/s is 3.5–4.9 ps per weight, against Q4_1's
  4.3 and Q5_0's 5.0. The same effect shows in PXA's P100 result and in the GTX 1070's PQ2_0/PTQ1_0.
- **Where shifts and conversions matter.** Shifts and conversions are ~30 % of Q5_0's and Q4_1's slot-equivalents.
  They matter only where a kernel is issue-limited: IQ4_XS, and Q5_0/Q5_K/IQ4_XS at 3–4 tokens per
  step (8.9 instructions per byte for Q5_0 at 4 tokens; [kernel note](pascal_mmvq_integration.md) section 14).
  - The dp4a magic start (accumulate from the bits of 1.5·2²³, read the result back as a float, subtract 1.5·2²³) turns
    each I2F (4 slots) into one FADD. It is exact while the block sum stays under 2²² in magnitude: Q8_0's worst case
    is 516,128, and Q4_1/Q5_0 in the ×16 domain stay under ~2.1 M.
  - Powers of two folded into per-plane accumulators remove the SHR from bit-plane decode when a plane spans several
    blocks of one scale.

### Rules for a Pascal-native layout (for any future type)

1. Super-blocks of 256 weights whose fields start on 16-byte boundaries, so lanes can use 128-bit loads without staging.
2. 4-bit codes in Q4_0 order (byte i holds weights i and i + 16), decoded with two masks and an unsigned-by-signed dp4a,
   with the factor 16 folded into the scale: no shifts.
3. Extra bit planes stored so that one mask per dp4a group selects them, with per-plane integer accumulators combined
   once per scale group.
4. Integer sub-scales (u8 or 6-bit) converted by the magic-number trick, and at most one FP16 field per super-block.
5. No lookup tables: on 5 SMs a 16-entry table costs ~1.1–1.5 instructions per weight, more than the codes save.
6. Scales that can live on matrix borders (per row or per input channel) are free. They fold into the activations or the
   output, as in NanoQuant's s₁/s₂ or AWQ-style scaling, and are the only quality lever with zero decode cost.

Q4_K, Q5_K and Q6_K already meet rules 1–5 closely enough (Q6_K's 210-byte super-block is only 2-byte aligned and
relies on the fork's staged loads; its 6-bit and 8-bit sub-scales need a few instructions per 32 weights). Rule 6 is a
file-side transformation, not a type.

### Bytes per token for G3-like mixes

Matrix time at the benchmark's speeds:
- Q8_0 150 GB/s; the Q8_0 head 165.
- Q5_0 138; Q5_K 128.
- Q6_K and Q4_K at the estimates above (140 and 130).
- The router about 0.9 ms (F32) or 0.6 ms (F16).

| Mix | Q8_0 (head) | Q8_0 (other) | 6-bit | Experts | Router | GB per token | Matrix ms (est.) | KLD against BF16 |
|---|---|---|---|---|---|---|---|---|
| G3 (measured file) | 0.327 | 1.134 | — | 0.766 Q5_0 | F32 0.099 | 2.327 | ~16.0 | 0.101–0.104 |
| G3, F16 router | 0.327 | 1.134 | — | 0.766 Q5_0 | F16 0.050 | 2.277 | ~15.7 | same (Q35) |
| G3-6: every non-head Q8_0 in Q6_K | 0.327 | — | 0.875 | 0.766 Q5_0 | F16 | 2.018 | ~14.4 | 0.118 (Q90, Q6_K measured) |
| G3-6K: G3-6 with experts in Q5_K | 0.327 | — | 0.875 | 0.766 Q5_K | F16 | 2.018 | ~14.8 | 0.110–0.115 (est.) |
| G3-6 with experts in Q4_K | 0.327 | — | 0.875 | 0.627 Q4_K | F16 | 1.879 | ~13.7 | 0.13–0.14 (est., over the bar) |
| G3-6 with experts in "P3" | 0.327 | — | 0.875 | 0.488 | F16 | 1.740 | ~12.7–14.2 | ≥ 0.18 (est., fails) |

How the 3-bit row is estimated:
- Imatrix Q4_1 experts cost ~0.05 KLD over Q8_0 experts (Q18: 0.031 for gate/up, 0.019 for down).
- Q5_0 experts therefore cost ~0.015.
- Each bit removed multiplies the rounding error energy by ~4, so 3.5 bits add 0.07–0.12 to G3's 0.101.
- GPTQ-style rounding would recover a third at most (YAQA, breakthrough candidate 6). A 3-bit expert format does not
  fit the bar without quantization-aware training (Q40).

The 6-bit rung prices well against the frontier:
- It saves ~1.3 ms of matrix time for +0.014 KLD (1.6 ms counting the F16 router), ~0.01 KLD per ms.
- The file-side frontier near the bar trades 0.015–0.025 per ms ([unconventional ideas](unconventional_ideas.md),
  "The frame").
- So the Q6_K kernel is the format work with the best return. A GPTQ-rounded Q5_0 head (Q56: −0.116 GB for +0.002)
  stacks with it.

**Verdict.** Do not create a new GGUF type for decode speed. Build Pascal kernels for Q6_K first (idea 4) and Q4_K
second (idea 8), and apply rules 2–4 to the kernels that are still issue-limited (idea 7). The format lever left is
quality per byte (rounding, Q40's recovery, Q90's allocation), not decode.

## 6. Anything else that would materially beat what we have

- **Address translation (section 3)** is the one new item that could be worth as much as a format change, and it costs
  an afternoon to confirm or kill.
- **Q8_1 from the norm (idea 2) and #28874 (idea 5)** are the two exact, cheap engine changes other Pascal forks have
  measured and the fork lacks.
- **A throughput calibration (idea 6) is needed before the multi-token kernels (Q78) and prefill (Q80) are tuned further.**
  - Single-token decode uses ~4 dp4a per SM per clock of a nominal 128 (est., from Q8_0 at 150 GB/s), and prefill ~10 %
    of the dp4a peak.
  - So neither would reveal a 2–4× throttle on a mining SKU. The CMP precedent (Rhonstin, xrip) shows NVIDIA does
    throttle per instruction class on mining cards, though no such report exists for Pascal.
- **Do not chase:**
  - FP16 decode paths (P100 forks, ExLlama, calm, NanoQuant's kernel);
  - CMP-style IMAD/HFMA2 substitution;
  - cache-hint or prefetch-instruction tuning of the mat-vec itself;
  - Vulkan for decode (covered in quant_speed_d; the fork's CUDA kernels now run at 2× stock);
  - any 2–3-bit format for Flint's experts.

## Already covered here

| Finding in this research | Where it is |
|---|---|
| Rows per warp sharing activations, wide/staged loads, hoisted addressing (qr0n, Kmic-68, killy) | Q25, Q38 (the fork's `mmvq-pascal.cu`) |
| 2 warps per block, small-K rule, Q2_K/Q3_K warps (ik #2740) | Q34, Q47, upstream note |
| FP32 activation sums; the overflow qr0n's FP16-sum shortcut reintroduces | Q76 |
| One quantization per shared input (ik, sbendhiba #10) | Q38 step 9; idea 2 extends it to the norm |
| Shared-expert gate/up fusion | Q50 |
| `__ldg`/read-only path, `-dlcm=ca` | Q48 |
| CUDA graphs on cc 6.1 and graph keys per width (killy, ArtyomITA) | Q38 build, Q52 (#31004) |
| Multi-token kernels that decode each weight once for 2–8 columns (PXA, nvfp4effma, Mrinohk 0019) | Q78 (`pascal-p106-v2-multicol`) |
| Persistent layer kernels (Mrinohk, killy) | Q88, breakthrough candidate 2, `bench/persistent_pascal` |
| Expert-contiguous layout | Breakthrough candidate 9; section 3 adds the TLB data, the fit and prefetch |
| 6-bit rung | Q90, breakthrough candidate 4; idea 4 picks Q6_K over Q6_0 |
| Group-64 Q4_1, coarser scale groups, F16 router, router kernel | Q16/Q28, unconventional ideas section 5, Q23/Q35 |
| Hadamard rotations, FWHT fusion (ArtyomITA, Kindi, PXQN) | Q13/Q15 (rotation did not help Flint's quants) |
| Trellis, vector codebooks, learned 4-bit lookups | Breakthrough candidates, rejected list |
| NVFP4/MXFP4, ik types, external kernels (Marlin, BitBLAS, FLUTE, LUT-GEMM, ExLlama, vLLM-Pascal) | quant_speed_b L8–L10, engines_other |
| Vulkan on Pascal, Bonsai sm_61 fusions, `GGML_CUDA_GRAPH_OPT`, MTP depth on P40 | quant_speed_d |
| A card driving a display loses ~13 % (killy) | quant_speed_d K2, Q83 |
| Pipelined speculation, drafters | Q87, breakthrough candidate 1, pipelined_speculation |
| Prefill MMQ tiles (cycle-five, madsteph74, #30235, #31047) | Q80 |

## Not verified

- GP106's TLB sizes, organization and walk latency, and whether its page walker overlaps several walks; the figures
  used are GP100/GP104's (Jia et al.), and Volta/Turing's walk latencies.
- The page-proportional fit rests on four calls measured with the event profiler; the per-launch overhead subtracted
  (6.5 µs) is a midpoint of 5–8 µs.
- That a load issued for translation prefetch makes the walk start early without stalling the issuing warp, and that
  the walk's result stays in the L2 TLB until the stream reaches the page.
- Placement: llama.cpp's weight buffer per card and tensor order were taken from the loader's design, not dumped from
  the rig.
- Mrinohk's baseline ran without CUDA graphs (inferred from the upstream gate at its base commit and the separate sm_60
  graphs patch in sbendhiba's set), so how much of its +36 % the fork's graphs already capture is open.
- killy's 330–370 GB/s exceeds its own streaming ceiling; part of it is cache.
- Instruction throughput on the P106-090 for dp4a, PRMT and conversions; the slot-equivalent counts assume the CUDA
  guide's cc 6.1 rates and full-rate dp4a and PRMT.
- Q6_K and Q4_K Pascal kernel speeds (130–150 GB/s) are estimates from the Q5_K kernel; Q5_K and Q4_K expert quality on
  Flint is extrapolated from Llama-3-8B and the UD-Q4_K_XL/UD-mirror comparison.
- NanoQuant's 2-bit comparison table (Table 8) and LittleBit's training cost were not read in full; no sub-1-bit method
  has been evaluated on an MoE model.
- Nayak et al.'s GTX 1080 Ti TLB numbers come from a secondary summary.
- PXA's closed PXQN decode is known only from its published geometry.

## Sources

Address translation and Pascal microarchitecture
- Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU Architecture via
  Microbenchmarking", [arXiv 1804.06826](https://arxiv.org/abs/1804.06826), Table 3.1, Figures 3.2 and 3.12, section 3.8 (P)
- Z. Jia, M. Maggioni, J. Smith, D. P. Scarpazza, "Dissecting the NVidia Turing T4 GPU via Microbenchmarking",
  [arXiv 1903.07486](https://arxiv.org/abs/1903.07486), Table 3.1, Figure 3.5, section 3.8 (P)
- X. Mei, X. Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking",
  [arXiv 1509.02308](https://arxiv.org/abs/1509.02308) (P)
- A. A. Nayak, B. Pratheek, V. Ganapathy, A. Basu, "(Mis)managed: A Novel TLB-based Covert Channel on GPUs", AsiaCCS
  2021, [IISc page](https://www.csa.iisc.ac.in/~vg/papers/asiaccs2021) (S: summary only)
- R. Ausavarungnirun et al., "Mosaic: A GPU Memory Manager with Application-Transparent Support for Multiple Page
  Sizes", MICRO 2017, [PDF](https://ghose.cs.illinois.edu/papers/17micro_mosaic.pdf) (P; simulated, for the mechanism)
- NVIDIA, CUDA C++ Programming Guide 12.8, "Arithmetic Instructions" throughput table (P)

Samsung sub-1-bit quantization
- H. Chong, D. Kim, C. Kim, M. Choi, "NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models", ICML 2026,
  [arXiv 2602.06694](https://arxiv.org/abs/2602.06694) (v3), Eq. 1, Table 2, Table 12, Appendices E.2, F.1, F.5;
  [SamsungLabs/NanoQuant](https://github.com/SamsungLabs/NanoQuant) at `a9e0a4308`, `src/nanoquant/kernel/cuda/gemv_kernel.cu`,
  `src/nanoquant/modules/linear.py` (P)
- B. Lee, D. Kim, Y. You, Y. Kim, "LittleBit: Ultra Low-Bit Quantization via Latent Factorization", NeurIPS 2025,
  [arXiv 2506.13771](https://arxiv.org/abs/2506.13771) (v5), Eq. 4–5, 9–10, Table 1, section 5 (P)
- B. Lee, Y. Kim, "LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment",
  ICML 2026, [arXiv 2603.00042](https://arxiv.org/abs/2603.00042) (v2), Table 1 (P);
  [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) at `42d658b0c`, `quantization/modules/littlebit.py` (P)

Pascal forks and projects
- [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) patches 0001–0018 (P)
- [ArtyomITA/bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal) (P)
- [killy-netsphere/bonsai2-1080ti](https://github.com/killy-netsphere/bonsai2-1080ti), `bonsai2-pascal.patch` (P)
- [Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches), README, `results/stock-vs-patched-2026-10-12.md`,
  patches 0003, 0004, 0010, 0018, 0019 (P)
- [sbendhiba/llama-cpp-pascal-patches](https://github.com/sbendhiba/llama-cpp-pascal-patches), `docs/patches.md` (P);
  [shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches) (P)
- [Frostbite1536/P100](https://github.com/Frostbite1536/P100), [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp),
  [cycle-five/pascal-mmq-autorunner](https://github.com/cycle-five/pascal-mmq-autorunner) (P)
- ik_llama.cpp [#2701](https://github.com/ikawrakow/ik_llama.cpp/pull/2701), [#2721](https://github.com/ikawrakow/ik_llama.cpp/pull/2721),
  [#2740](https://github.com/ikawrakow/ik_llama.cpp/pull/2740), [#2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509),
  [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md) (P)
- llama.cpp [#28874](https://github.com/ggml-org/llama.cpp/pull/28874), [#31047](https://github.com/ggml-org/llama.cpp/pull/31047),
  [#26487](https://github.com/ggml-org/llama.cpp/pull/26487), [#29393](https://github.com/ggml-org/llama.cpp/pull/29393) (P)
- [boetti-dev/p40-kernel](https://github.com/boetti-dev/p40-kernel), [Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp),
  [Niko1221/Strata](https://github.com/Niko1221/Strata) (P)
- [sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill) (P)

Mining cards
- [Rhonstin/beellama-cmp90hx](https://github.com/Rhonstin/beellama-cmp90hx), [Wh1stle05/cmp90hx](https://github.com/Wh1stle05/cmp90hx),
  [Cyridd/cmpunlocker](https://github.com/Cyridd/cmpunlocker), [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock),
  [WifiLast/torch-cmp-patch](https://github.com/WifiLast/torch-cmp-patch) (P)
- [dartraiden/NVIDIA-patcher #241](https://github.com/dartraiden/NVIDIA-patcher/issues/241) (S)

Other engines and kernels
- [poisonxa16/pxa](https://github.com/poisonxa16/pxa) (README, `docs/LEVERS.md`, `docs/KNOWN-ISSUES.md`, `pxq-mmvq.cuh`,
  `pxa-enhance.cuh`), [wepromen/pxq_llama](https://github.com/wepromen/pxq_llama) (P)
- [ahtx/kindi](https://github.com/ahtx/kindi) (`docs/HOW-IT-WORKS.md`, `docs/BENCHMARKS.md`) (P)
- [srose69/nvfp4effma](https://github.com/srose69/nvfp4effma) (`src/gemv_nvfp4effma.cuh`, `src/nvfp4effma.cuh`,
  `docs/design.md`, `benches/1080.md`) (P)
- [Mikec78660/vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal), [zxmaster/vllm-pascal-sm61](https://github.com/zxmaster/vllm-pascal-sm61),
  [sasha0552/pascal-pkgs-ci](https://github.com/sasha0552/pascal-pkgs-ci) (P)
- ExLlamaV2 [#40](https://github.com/turboderp-org/exllamav2/issues/40), [#185](https://github.com/turboderp-org/exllamav2/issues/185);
  ExLlamaV3 [#512](https://github.com/turboderp-org/exllamav3/issues/512), [#509](https://github.com/turboderp-org/exllamav3/issues/509) (P)
- fastllm [#449](https://github.com/ztxz16/fastllm/pull/449), [#537](https://github.com/ztxz16/fastllm/pull/537) (P);
  [zeux/calm](https://github.com/zeux/calm) (P); [microsoft/BitNet](https://github.com/microsoft/BitNet) `gpu/` (P);
  MLC-LLM [#802](https://github.com/mlc-ai/mlc-llm/issues/802), [#2100](https://github.com/mlc-ai/mlc-llm/issues/2100) (P)
- [jdkruzr/p40bench](https://github.com/jdkruzr/p40bench), [sarge18/p40-llm-engine-bakeoff](https://github.com/sarge18/p40-llm-engine-bakeoff),
  [christopherrobertbrooks-tech/legacy-gpu-llm-notes](https://github.com/christopherrobertbrooks-tech/legacy-gpu-llm-notes) (P)
- Y. Park et al., "Any-Precision LLM", ICML 2024, [arXiv 2402.10517](https://arxiv.org/abs/2402.10517),
  [code](https://github.com/SNU-ARC/any-precision-llm) (P); T-MAC [arXiv 2407.00088](https://arxiv.org/abs/2407.00088) (P);
  CodeGEMM [arXiv 2512.17970](https://arxiv.org/abs/2512.17970) (P)
- llama.cpp [perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md), Llama-3-8B
  table (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q7, Q12, Q16, Q18, Q21, Q25, Q34, Q35, Q38, Q46, Q48, Q49, Q51, Q52,
  Q56, Q76, Q78, Q80, Q87, Q88, Q89, Q90), [kernel integration note](pascal_mmvq_integration.md) (sections 1, 4, 10, 14),
  [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) and its `src/main.cu`, [K-quant note](kquant_decode_pascal.md),
  [formats and kernels survey](quant_speed_b_formats_kernels.md), [community reports](quant_speed_d_community.md),
  [breakthrough candidates](breakthrough_candidates.md), [unconventional ideas](unconventional_ideas.md)
- Fork source, branch `pascal-p106-v2` (`032b130f4`): `ggml/src/ggml-cuda/mmvq-pascal.cu` (dot products, `__half2float`
  scales, `(qh >> j)` fifth bits, `(float) isum` conversions), `ggml/src/ggml-cuda/ggml-cuda.cu` (`GGML_CUDA_PROFILE`:
  events as graph nodes, stream synchronized after each evaluation; fusions RMS_NORM → MUL (→ ADD/ROPE) only)
