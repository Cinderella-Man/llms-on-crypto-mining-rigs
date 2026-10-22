# Survey D: community claims log for old NVIDIA cards (Pascal, mining cards), Flint and Qwen3.6, as of 2026-10-23

Scope: what owners of Pascal cards (P40, P100, P102/P104/P106 mining cards, GTX 10-series, Quadro P) have found and
claimed about decode speed, quant formats, flags, forks and mining-rig tricks, plus everything the community has said
about Flint and about Qwen3.6-35B-A3B on Pascal. Papers, kernels and retraining are covered by surveys A–C.

Credibility labels used throughout:
- **M** = measured, with numbers and a stated setup. **M-same** = A/B on one machine and one build (the strongest kind).
- **A** = anecdote (a number without a setup, or an impression).
- **S** = speculation or advice without measurement (often LLM-written).
- Source type as in the brief: **P** = primary (PR, issue, source code, model card, repo README with raw numbers),
  **S** = secondary (Reddit, blog, forum, news).

How the trawl was done, and what could not be reached:
- Reddit: the site blocks scripted access, so posts and comment trees came from the Arctic Shift archive API. Post
  titles and bodies of r/LocalLLaMA from 2026-05-01 to 2026-10-22 (15,954 posts) were scanned for Pascal terms plus
  quant/speed terms; r/LocalLLM, r/LocalAIServers and r/gpumining were scanned the same way (r/homelab timed out).
  The archive's full-text **comment** search timed out every time, so offhand comments were found only inside threads
  that were opened. Older threads (2023–2025) were opened from title searches.
- GitHub: issue/PR/discussion search on ggml-org/llama.cpp and ikawrakow/ik_llama.cpp; the CUDA scoreboard #15013 and
  the Vulkan scoreboard #10879 were downloaded in full. Repository search for Pascal forks.
- Hugging Face: discussions on all Flint repos, model list for "flint".
- Chinese: CSDN search works; two CSDN articles were read in a browser, then CSDN put up a CAPTCHA (not bypassed).
  Bilibili's search API refused the request; Zhihu and Tieba were not reachable as text. No Chinese source beyond CSDN.
- Russian: Habr and overclockers.ru searches returned nothing P106-specific; the only Russian technical claim found
  is a comment in the Reddit thread of the $35 nine-P106 rig. Level1Techs: nothing relevant found.

---

## 1. Executive summary

1. **Our card is the only Pascal SKU that is clearly instruction-bound, so other Pascal owners' quant rankings
   understate the format penalty we see.** Every other Pascal card on the CUDA scoreboard turns 49–66 % of its rated
   bandwidth into Llama-2-7B Q4_0 decode; the P106-090 turns 39–43 %. It has ~35 GB/s of bandwidth per SM against
   ≤ 19 GB/s on every other Pascal SKU, and its ~8.6 bytes per SM per cycle sits at the ceiling the other ≤ 6-SM cards
   also hit (my calc, §3). Consequence: "Q4_K_M ≈ Q4_0" on a P40 (measured, equal per byte), 6 % apart on a GTX 1060,
   22 % apart on our card.
2. **Vulkan beat CUDA on decode on several Pascal cards, including the same GP106 chip, in one same-machine A/B** —
   GTX 1060, same build b8130 on Linux: +47 % on a small hybrid MoE, +11 % on a dense 7B (issue #19817, Feb 2026,
   before CUDA graphs were enabled on Pascal). Scoreboard pairs (different machines) give +7 to +17 % on 1080 Ti, P40,
   P100, 1070 Ti, but ±0 on a P106-100 and −15 % on a Tesla P4. Flint already runs on llama.cpp's Vulkan backend
   (RX 7800 XT report). Nobody has tried Vulkan on a 5-SM card. **Cheapest big unknown on the list.**
3. **Launch count is the other half of our token time, and Pascal forks show where launches can go.** A sm_61 patch
   for a ternary Bonsai model on a GTX 1080 got +51 % decode bit-identically; fusing add+norm, norm+mul+SwiGLU and
   copies alone took launches from 976 to 755 and was worth +8 %. Mainline has an env switch for Q/K/V stream
   concurrency (`GGML_CUDA_GRAPH_OPT=1`, +1–9 % decode on Ada/Blackwell) that nobody has measured on Pascal.
4. **The MMVQ fusion gate is a free A/B**: on a P40, `GGML_CUDA_DISABLE_FUSION=1` gave Q4_0 +4.7 % but Q4_K_M −1.2 %
   (#30334, already in our notes). It has not been measured on a Flint/Q4_1 file or a 5-SM card.
5. **FP4 formats lose on Pascal in every community measurement found**: NVFP4 −13 % (Qwen3.6) and −31 % (Gemma 4 MoE)
   against K-quant files of similar size on a 1080 Ti + 2 × P102-100 (Vulkan); MXFP4_MOE −11 % against UD-Q4_K_S on
   3 × GTX 1070 (Vulkan). This supports not using Recursive's NVFP4 file or MXFP4 experts on the rig.
6. **I-quants: the "IQ is slow on P40" folklore is about prompt processing.** For decode, a P40 owner saw no difference
   moving Qwen3.6 from Q4 to IQ4 (MoE); a 2 × P102-100 owner says Q4_0 is 20 % faster than IQ4 on dense Qwen3.8-27B
   (anecdote); IQ4_NL was 8 % slower than UD-Q4_K_S on 3 × GTX 1070 (Vulkan). Consistent with "IQ4_NL ≤ Q4_K on
   Pascal decode", so IQ4_NL down-projections (rescale-safe) are not free.
7. **Flint: no Pascal report exists anywhere, and no community report of the Q4_0/Q4_1 NaN or the rescale.** Speed
   reports are from an RX 7800 XT (49 t/s, Vulkan, UD-Q4_K_XL with experts partly on CPU), an RTX PRO 6000 (vLLM
   NVFP4, 110–180 t/s), Macs (125–128 t/s) and Unsloth's 150 t/s on an RTX Spark. The community's quality impressions
   are mixed to negative (loops, overthinking; "broken in llama.cpp and SGLang" is an unverified claim); a serving
   report shows tool calls need a grammar outside Recursive's `rec`. No MTP or draft model; a request on the model page
   is unanswered.
8. **Qwen3.6-35B-A3B MTP on Pascal: two drafted tokens is the optimum, six is worse than none.** Single P40, same
   setup: 37.9 t/s without MTP, 51.6 with `--spec-draft-n-max 2`, 48.3 with 3, 34.5 with 6 (discussion #25978). Two
   other P40 owners and a 3090 owner independently settled on 2. GPU-side draft sampling added ~4 % on a P40
   (PR #25532, merged in b10355).
9. **Other engines do not beat llama.cpp on this rig.** A vLLM fork for P40/P100 with a custom W4A16 GEMV "meets"
   llama.cpp Q4_0+MTP on 2 × P40 (13.2 vs 13.6–13.7 t/s) on Gen1 x1 links; ik_llama.cpp's multi-P40 doubling relies on
   P2P (we have none) and needs AVX2 (we have no AVX); PXA (llama.cpp-derived Pascal/Volta engine, v3.1) ships its
   fast PXQN kernels closed and supports sm_61 only as "one card, not measured"; Strata is model-specific (no Flint).
10. **Mining-rig experience agrees with our layer-split findings**: x1 risers cost load time and prefill, not
    layer-split decode (a 3090+5060 Ti owner measured peak PCIe traffic under 400 MB/s during decode); tensor split on
    Gen1 x1 collapses (12 × P104-100: 1.2 t/s prefill with `-sm tensor`). The P104/P106 PCIe-1.0 lock is reported as a
    VBIOS limit; the only "Gen2 unlock" found is for Ampere CMP cards.
11. **The $35 nine-P106 rig on Reddit (2,127 upvotes, 2026-10-06) is the same Manli barebones as ours** (identified
    by a commenter who holds pallets of them) and its owner reports "30+ tok/s" on Qwen3.6-35B-A3B in Ollama. If those
    cards were P106-090s this would contradict our 18.5–19.6 t/s with Q4_0; the 10-SM P106-100 explains it (the
    scoreboard ratio between the two cards is 1.56×). Variant unconfirmed; claim to check.
12. **Folklore check** (§11): "compile with FORCE_MMQ for P40" was true in 2023 (≈2× on a GTX 1080) and is a no-op
    for decode now; `LLAMA_CUDA_FORCE_DMMV`, `LLAMA_CUDA_F16` and `GGML_CUDA_F16` no longer exist in current builds but
    still appear in 2026 P40 scripts; "-ub 1 makes it faster" was not found anywhere; "P40 hates K-quants" is false on
    a P40 and true on our card.
13. **Core clock should matter on an instruction-bound card.** A P102-100 lost 11 % decode at 150 W vs 250 W, a
    3 × GTX 1070 owner ~5 % at 120 W. Our cards run 1,700–1,760 MHz at 75 W; if decode is issue-bound it scales with
    core clock, not memory clock (claim to test; changing clocks or power limits needs the owner's go-ahead).

---

## 2. Lever table

Budget used for arithmetic (from the brief): Flint Q4_1 token = 35.8 ms on 5 cards = ~23–26 ms reading 1.66 GB of
weights + ~10–13 ms fixed cost from ~1,900 launches (~5–7 µs each).

| Lever | Class | Mechanism | Expected effect on this rig (arithmetic) | Quality effect | Effort | Evidence strength | Carries over to Qwen3.6? |
|---|---|---|---|---|---|---|---|
| Vulkan backend instead of CUDA | B (engine only, same file) | Different mat-vec shaders (int-dot or dequant-to-FP32), command-buffer submission instead of CUDA-graph replay | Unknown sign. If the fixed 10–13 ms shrank by 30 % and weight time stayed equal: 35.8 → ~32 ms (+12 %); if Vulkan mat-vecs are 15 % slower per byte: → ~39 ms (−9 %) (my calc). The 2-core Celeron recording command buffers is a risk | None (same weights; different rounding) | 0.5–1 day incl. build | M-same on GTX 1060 (+11 %/+47 %, pre-graphs), M on several other Pascal cards, none on ≤ 6 SMs | Yes (GDN runs on Vulkan; 1080 Ti/P102 and 3 × 1070 reports) |
| `GGML_CUDA_GRAPH_OPT=1` (Q/K/V on concurrent streams) | B (env var) | Overlaps the three small attention-input mat-vecs and their norms | 49 layers × 1–2 hidden kernel tails of 5–10 µs ≈ 0.25–1 ms → +1–3 % (my calc) | None | Minutes | M on RTX 4090/5090 only (+1–9 % tg); untested multi-GPU and Pascal | Only its 10 attention layers |
| MMVQ fusion off (`GGML_CUDA_DISABLE_FUSION=1`) | B (env var) | Skips fused gate/up(+bias) MMVQ, which #30334 found slower for Q4_0 on sm_61 | P40: Q4_0 +4.7 %, Q4_K_M −1.2 %. On Flint's Q4_1/Q5_0 experts: −2 to +5 % (claim to test); it also turns off every other fusion | None | Minutes | M (P40, one model) | Yes |
| Pascal decode fusions (port of the Bonsai2 sm_61 patch ideas) | B (kernel) | add+RMSNorm, RMSNorm+mul+SwiGLU, batched strided copies, fused small ops | Bonsai: 976 → 755 launches (−23 %) = +8 %. Flint: if 100–250 of ~1,900 launches go at 5–7 µs → 0.5–1.75 ms → +1.5–5 % (my calc) | Bit-identical in the source patch | 1–3 days | M on sm_61 (GTX 1080, other model) | Yes (the patch already fuses GDN pieces) |
| Pick formats by weights per SM-cycle (legacy over K/IQ/FP4) | A (file) | 5 SMs at ~1.73 GHz decode ~15 Q4_0 weights, ~11 Q4_K weights, ~10 Q8_0 weights per SM-cycle (my calc from our calibration) | Already the basis of UD-mirror; community data add that NVFP4/MXFP4 and IQ4_NL are slower than K-quants on Pascal, so they are the last choice per byte | Depends on mix | Done / ongoing | M (rig) + M (community) | Yes |
| Q8_0 output head (pxq_llama recipe) | A (file) | Cheaper unpack per byte | For Flint the 4-bit head is ~0.2 GB/token; Q8_0 doubles it: ~2.9 ms → ~4.3 ms, i.e. −4 % (my calc). A quality lever, not a speed lever here | Better | Trivial | M on P100 (combined with a DeltaNet fusion) | Quality only |
| Codebook / ternary formats for Pascal (PXQ, PXQN, PQ2_0) | B / C | Fewer bytes per weight with custom kernels | PQ2_0 on a GTX 1080 still ran at only ~145 GB/s (45 %), instruction-bound; PXQN kernels are closed; ternary needs retraining (C) | PXQN2 ≈ "3-bit class" | Weeks | M (vendor-measured, P100/V100/1080) | Format exists for Qwen3.x-class models only |
| MTP draft depth 2 + GPU draft sampling | Engine (Qwen3.6 only) | Fewer rejected drafts on a compute-poor card | +36 % on a P40 at depth 2 vs none; depth 6 −9 % vs none | Exact | Minutes | M-same (P40) ×3 sources | Qwen3.6 only; Flint has no MTP |
| Core clock / power limit | Rig setting | If issue-bound, decode ∝ core clock | 1.73 → 1.90 GHz would be +10 % if the limit allows (my calc) | None | Minutes, needs user OK | A/M on P102 (−11 % at 150 W) | Yes |
| Other engines (vLLM-Pascal, ik_llama.cpp, PXA, Strata, FreeToken-Pascal) | Engine | — | None better for this rig (no P2P, no AVX, Flint unsupported or sm_61 unmeasured) | — | — | M (vLLM fork vs llama.cpp on 2 × P40) | Strata/PXA support Qwen-family only |
| Prefill-only tricks (TurboPrefill, device order, `-ts`, `-ub`) | Engine | Pipeline scheduling across cards | 0 on decode | — | — | M on P104 rigs | Prefill only |

---

## 3. Which formats decode fastest on Pascal, by SM count

### Takeaway
Community numbers are almost all from bandwidth-bound Pascal cards (P40, 1080 Ti, P102, GTX 1060). On those, Q4_0
and Q4_K_M are nearly equal per byte and FP4 formats are clearly slower. On the 5-SM P106-090 the per-SM issue rate
is the limit, so the cheaper-to-unpack format wins by a much larger margin. No community measurement exists for
Q4_1, Q5_0 or IQ4_NL decode on a ≤ 6-SM card.

### Cited findings

**Scoreboard rows: Llama-2-7B Q4_0, CUDA, tg128 without FA** ([discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), 2025-08 to 2026-07; P). Effective bandwidth = tg × 3.82 GB; clocks are nominal boost; all arithmetic mine.

| Card | SMs | Rated GB/s | tg128 | Effective GB/s | % of rated | Bytes per SM per cycle | Rated GB/s per SM |
|---|---|---|---|---|---|---|---|
| Quadro P1000 | 5 | 82 | 13.99 | 53 | 65 % | 7.2 | 16.4 |
| GTX 1050 Ti (2026-06-24) | 6 | 112 | 19.1 | 73 | 65 % | 8.7 | 18.7 |
| **P106-090 (our rig, [calibration](../../../llms-on-crypto-mining-rigs/docs/benchmarks/2026-10-22-single-card-calibration.md))** | **5** | **192 (173 measured)** | **19.45** | **74** | **39 % (43 %)** | **8.6** | **38.4** |
| Quadro P2000 | 8 | 140 | 23.63 | 90 | 64 % | 7.6 | 17.5 |
| GTX 1060 6GB | 10 | 192 | 27.79 | 106 | 55 % | 6.2 | 19.2 |
| P106-100 | 10 | 192 | 30.40 | 116 | 61 % | 6.8 | 19.2 |
| Tesla P4 | 20 | 192 | 33.29 | 127 | 66 % | 6.0 | 9.6 |
| Tesla P40 | 30 | 347 | 54.74 | 209 | 60 % | 4.6 | 11.6 |
| GTX 1080 Ti | 28 | 484 | 62.49 | 239 | 49 % | 5.4 | 17.3 |

**Same model, two formats, one card:**

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| F1 | Q4_0 and Q4_K_M decode at the same bytes/s on a P40 | [#30334](https://github.com/ggml-org/llama.cpp/issues/30334) (P) | 2026-10-11 | Tesla P40 | Llama-3.1-8B: Q4_0 52.28, Q4_K_M 48.94 t/s; with files of 4.66 / 4.92 GB that is ~244 vs ~241 GB/s (my calc) | M-same |
| F2 | On a GTX 1060 Q4_0 leads Q4_K_M by ~6 % per byte | [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) (P) | 2026-07-09 | GTX 1060 6GB (GP106, 10 SMs) | Llama-3.2-3B tg128: Q4_0 49.41, Q4_K_M 43.49, Q5_K_M 39.92, Q6_K 34.10, Q8_0 32.62; ≈ 99 / 93 / 96 / 90 / 114 GB/s ([dense notes §3](../../../llms-on-crypto-mining-rigs/docs/research/notes/dense_27b_speedups.md)) | M-same |
| F3 | On our 5-SM card Q4_0 leads Q4_K_M by 22 % per byte, Q8_0 leads both | rig calibration (P) | 2026-10-22 | P106-090 | 73 / 60 / 93 GB/s | M-same |
| F4 | NVFP4 is the slowest 4-bit file on Pascal; Q6_K matches Q4_K_M per token | [Reddit 1v6llio](https://www.reddit.com/r/LocalLLaMA/comments/1v6llio/) (S) | 2026-07-25 | 1080 Ti + 2 × P102-100, Vulkan b10107, 150 W each | Gemma-4-26B-A4B: Q4_K_M 48.14, Q6_K 49.67, NVFP4 33.28; Qwen3.6-35B-A3B: UD-Q4_K_S 50.48, NVFP4 43.50 t/s | M-same |
| F5 | MXFP4_MOE and IQ4_NL are slower than UD-Q4_K_S on Pascal | [Reddit 1u7tarc](https://www.reddit.com/r/LocalLLaMA/comments/1u7tarc/) (S) | 2026-06-16 | 3 × GTX 1070, Vulkan b9204, 120 W | Qwen3.6-35B-A3B tg128: MXFP4_MOE 40.24, UD-Q4_K_S 45.30, UD-IQ4_NL 41.70 (files 16.4–20.2 GiB) | M-same |
| F6 | Q4_0 is 20 % faster than IQ4 on dense Qwen3.8-27B | [Reddit 1wsobzz comment](https://www.reddit.com/r/LocalLLaMA/comments/1wsobzz/comment/pcnwxsi/) (S) | 2026-09-28 | 2 × P102-100 | "20 % faster", no table | A |
| F7 | Moving Qwen3.6 from Q4 to IQ4 did not feel slower on a P40 | [Reddit 1wsobzz](https://www.reddit.com/r/LocalLLaMA/comments/1wsobzz/) (S) | 2026-09-28 | P40, MTP on | 37–83 t/s range | A |
| F8 | 2024 consensus "IQ quants are slow on P40" was about prompt processing ("cuts the processing time in half" with K or 0 quants) | [Reddit 1dmhpud](https://www.reddit.com/r/LocalLLaMA/comments/1dmhpud/) (S) | 2024-06-23 | P40 | no tg numbers | A |
| F9 | A vLLM fork's authors used llama.cpp **Q4_0** as the P40 reference to beat | [Mikec78660/vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal) (P) | 2026-10-16 | 2 × P40 | see §9 | M |
| F10 | "Stay on Q4_K/Q8 with the MMQ int8 path, avoid FP16" on P106s | [Reddit 1wz6ist comment](https://www.reddit.com/r/LocalLLaMA/comments/1wz6ist/comment/pe9ytpy/) (S) | 2026-10-06 | P106 (advice) | none | S (half wrong: decode uses MMVQ, and Q4_K is the slow one on our card) |
| F11 | ternary PQ2_0 mat-vec on a GTX 1080 runs at ~145 GB/s, "bound by instructions and occupancy, not memory" | [ArtyomITA/bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal) (P) | 2026-10-17 | GTX 1080 (20 SMs, 320 GB/s) | ~65 % of decode time in that mat-vec | M |
| F12 | K-quants spill registers at 2 blocks/SM in MoE MMQ tiles; legacy types gain 4–18 % per op | [poisonxa16/pxa docs/LEVERS.md](https://github.com/poisonxa16/pxa/blob/main/docs/LEVERS.md) (P) | 2026-10 | V100 (prefill path) | Q4_0/Q4_1/Q5_1/Q8_0 1.04–1.18×, K-quants 0.3–0.76× | M (not Pascal, not decode) |

### Inferences
- **The P106-090 is the odd card out (my calc).** Every Pascal card with ≤ 8 SMs lands at 7–9 bytes per SM-cycle for
  Q4_0 mat-vecs; cards with more SMs are memory-limited at 49–66 % of rated bandwidth. Ours reaches 8.6 bytes per
  SM-cycle but only 43 % of bandwidth because it has twice the bandwidth per SM of any other Pascal part. So the
  P1000/1050 Ti, despite also having 5–6 SMs, are not good stand-ins: they are bandwidth-bound too.
- **Per-weight cost on our card (my calc from the calibration):** Q4_0 ≈ 15 weights per SM-cycle (0.5625 bytes/weight),
  Q4_K_M ≈ 11, Q8_0 ≈ 10. A format's speed on this rig is set by its instructions per weight; bytes matter only through
  the file size. This is why community "smaller is faster" advice (true on bandwidth-bound cards) misleads here.
- **FP4 and IQ4_NL on Pascal:** F4 and F5 are Vulkan numbers, but the CUDA kernels also decode FP4/IQ4_NL through a
  16-entry table (dense notes §3), so the ordering is expected to hold on CUDA. Claim to test on a single P106-090.
- **"Q6_K ≈ Q4_K_M per token" on Vulkan (F4)** is surprising and probably a Vulkan-specific kernel effect; it is
  worth knowing if the Vulkan experiment (§4) succeeds, because it would change which mixes are cheap there.

### Gaps
- No community measurement of Q4_1, Q5_0, Q5_1 or IQ4_NL decode on any 5–6 SM card, CUDA or Vulkan.
- No Pascal measurement of MoE expert mat-vecs (MUL_MAT_ID) by format; all format comparisons above are whole-model.

### Experiment ideas
- Single P106-090, `llama-bench -n 128 -r 5` on Llama-2-7B in Q4_0, Q4_1, Q5_0, Q8_0, IQ4_NL, Q4_K_M, MXFP4 (all
  quantized from one F16 on the quant machine): fills the per-format weights-per-SM-cycle table. ~1 hour of rig time
  plus 30 min of quantizing. Probably overlaps with survey B; coordinate.

---

## 4. Vulkan instead of CUDA

### Takeaway
On Pascal, llama.cpp's Vulkan backend has repeatedly matched or beaten CUDA on decode while losing badly on prefill.
The strongest evidence is a same-machine, same-build A/B on a GTX 1060 (the same GP106 chip as ours) — but it predates
CUDA graphs on Pascal, which closed much of the launch-overhead gap for MoE. Flint is known to run on Vulkan. Nobody
has run Vulkan on a 5-SM card or on a 5–9-card Pascal layer split.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| V1 | Vulkan decodes much faster than CUDA on a GTX 1060 (same build b8130, Linux; CUDA build with arch 61) | [#19817](https://github.com/ggml-org/llama.cpp/issues/19817) (P) | 2026-02-22 | GTX 1060 6GB | Granite-4.0-H-Tiny Q4_K_M (hybrid MoE, ~1B active): tg128 Vulkan 90.63 vs CUDA 61.65 (+47 %); Falcon-H1R-7B Q4_K_S: 28.14 vs 25.35 (+11 %). Prefill: CUDA 2–5× faster | M-same |
| V2 | Same on a 4-SM Quadro P620, Windows | [#16230](https://github.com/ggml-org/llama.cpp/issues/16230) (P) | 2025-09-24 | Quadro P620 (GP107, 4 SMs) | Qwen2.5-1.5B Q4_K_M, f32 KV: Vulkan 34.19 vs CUDA 17.96 t/s. A maintainer replied that his P40 on Linux decodes the same on both backends and suspected Windows | M-same (Windows) |
| V3 | Scoreboard pairs, Llama-2-7B Q4_0 tg128, CUDA → Vulkan (different machines and commits) | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013), [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) (P) | 2024-12 to 2026-07 | various | 1080 Ti 62.49 → 67.75 (FA on: 61.38 → 71.58); P40 54.74 → 59.36; P100 58.35 → 63.16; 1070 Ti 37.82 → 42.86; P2000 23.63 → 23.77; 1050 Ti 19.1 → 20.08; **P106-100 30.40 → 29.77 (Vulkan reported "int dot: 0")**; Tesla P4 33.29 → 28.37 | M (cross-machine) |
| V4 | GTX 1060 on Vulkan 32.00 vs the CUDA scoreboard's 27.79 (same owner, CUDA run on Windows) | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) (P) | 2025-08-11 | GTX 1060 6GB (OCuLink x2) | 32.00 vs 27.79 | M (different OS) |
| V5 | Flint runs on llama.cpp Vulkan | [unsloth/flint-GGUF discussion #1](https://huggingface.co/unsloth/flint-GGUF/discussions/1) (S) | 2026-10-21 | RX 7800 XT 16 GB, 24 expert layers on CPU | UD-Q4_K_XL: pp2048 1886, tg128 49.0 t/s | M (AMD) |
| V6 | Vulkan was faster than CUDA for a CPU-offloaded MoE on a 1080 Ti | [Reddit 1v6llio comment](https://www.reddit.com/r/LocalLLaMA/comments/1v6llio/comment/ozuhhhz/) (S) | 2026-07-26 | GTX 1080 Ti, `n-cpu-moe 19` | 9.34 vs 7.81 t/s | A (offload case) |
| V7 | Vulkan integer-dot mat-vec exists for legacy quants | [PR #14903](https://github.com/ggml-org/llama.cpp/pull/14903) (P), merged 2025-09-01 | 2025-07-27 | — | title only read; needs the driver to expose integer dot product | P |
| V8 | Vulkan's MoE mat-vec row count is tuned separately for pre-Turing NVIDIA (kept at the old setting) | [PR #29274](https://github.com/ggml-org/llama.cpp/pull/29274) (P) | 2026-09-22 | RTX 5070 Ti data only | — | P |
| V9 | CUDA graphs on Pascal gave +40 % on MoE, +7 % dense (the PR our build carries) | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) (P) | 2026-08-25 | GTX 1080, P102-100 | e.g. granite-3.1-3b-a800m 81.2 → 124.5 | M-same |

### Inferences
- V1's +47 % on a small MoE is almost the same size as V9's +40 % from CUDA graphs on MoE, so most of V1's MoE gap was
  probably launch overhead that our graphs build no longer pays (my inference). V1's +11 % on a dense 7B and the
  +7–17 % scoreboard pairs suggest Vulkan's mat-vec itself can be at least as efficient as CUDA's MMVQ on Pascal.
- On our card the mat-vec is instruction-bound, so the question is instructions per weight. Vulkan's legacy-quant
  int-dot shader (V7) and its FP32-dequant fallback (used when the driver reports no integer dot product, as on the
  P106-100 in V3) have different instruction mixes from CUDA's MMVQ; neither has been profiled on a 5-SM card.
- Risks specific to this rig: the 2-core Celeron must record Vulkan command buffers for ~1,900+ nodes per token
  (CPU cost unknown; claim to test); inter-card copies go through the host either way; the brief's Flint rescale is a
  graph-level op, so Vulkan pays it too ([Flint notes §2](../../../llms-on-crypto-mining-rigs/docs/research/notes/flint_speedups.md)).

### Gaps
- No Vulkan number on any ≤ 5-SM Pascal card on Linux; no multi-GPU Vulkan Pascal layer-split decode number for a MoE.
- Whether the rig's driver 580 exposes `VK_KHR_shader_integer_dot_product` on the P106-090.

### Experiment ideas
- Build a Vulkan llama.cpp on the workstation with `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF` and AVX/AVX2/FMA off (the
  Celeron has no AVX), copy to `/opt/llama-builds/` (installing `libvulkan`/`glslc` on the rig itself would need the
  user's go-ahead). Then:
  1. Single card: `llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -n 128 -r 5` CUDA vs Vulkan; repeat for Q8_0 and
     Q4_K_M. Note `int dot:` in the Vulkan device line. ~30 min.
  2. Flint Q4_1 and UD-mirror v1 on 5 cards, `llama-bench -n 128 -p 0 -r 3`, plus `top` on the host during decode.
     ~1 hour. Decision rule: adopt only if ≥ 5 % faster on Flint.

---

## 5. Flags, env vars and builds

### Takeaway
Most "Pascal flags" in circulation are obsolete. What still exists and can matter for decode: CUDA graphs (already
on), the fusion switch, the Q/K/V stream-concurrency switch, FA on, and possibly blocking sync for a 2-core host.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| B1 | `GGML_CUDA_FORCE_MMQ=1` changes nothing for decode | [#27980](https://github.com/ggml-org/llama.cpp/issues/27980) (P) | 2026-08-29 | 4 × P40 | qwen4exp decode 18.0 / 18.1 with and without | M-same |
| B2 | Compiling with FORCE_MMQ made a GTX 1080 "almost twice as fast" | [Reddit 18helbs comment](https://www.reddit.com/r/LocalLLaMA/comments/18helbs/comment/kedszk5/) (S) | 2023-12-21 | GTX 1080 | no table | A (historic; MMQ is now the default on Pascal) |
| B3 | Current llama.cpp build options are FORCE_MMQ, FORCE_CUBLAS, NO_PEER_COPY, NO_VMM, FA_ALL_QUANTS, GRAPHS; runtime env vars include DISABLE_FUSION, DISABLE_GRAPHS, GRAPH_OPT, MMQ_PREC, PDL, NO_PINNED | local llama.cpp checkout at 4413a1924 (P, `ggml/CMakeLists.txt`, `ggml-cuda/*.cu`) | 2026-10-12 | — | `GGML_CUDA_F16`, `*_DMMV_*`, `LLAMA_CUDA_*` are gone | P |
| B4 | P40 owners still export `LLAMA_CUDA_FORCE_DMMV=1` / `LLAMA_CUDA_F16=1` or build with `-DGGML_CUDA_F16=OFF` | [discussion #19248](https://github.com/ggml-org/llama.cpp/discussions/19248), [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978), [Reddit 1tgsesb](https://www.reddit.com/r/LocalLLaMA/comments/1tgsesb/) (S) | 2026-02 to 2026-07 | P40 | — | no-ops (B3) |
| B5 | `GGML_CUDA_GRAPH_OPT=1` runs the Q, K, V branches on concurrent streams inside the CUDA graph; +1–9 % tg | [PR #16991](https://github.com/ggml-org/llama.cpp/pull/16991) (P), merged as c7af376c2 | 2025-11-30 | RTX 5090/4090 | qwen3moe 30B-A3B Q4_K_M 306.8 → 334.2 (+9 %), Llama-8B Q8_0 +5 % | M (modern GPUs) |
| B6 | Turning MMVQ fusion off speeds Q4_0 on a P40 | [#30334](https://github.com/ggml-org/llama.cpp/issues/30334) (P) | 2026-10-11 | P40 | Q4_0 +4.7 %, Q3_K_M +5.3 %, Q4_K_M −1.2 %, Q2_K −5.5 % | M-same |
| B7 | FA on is faster for decode on a P40 behind a x1 link | [Reddit 1o1wb1p comment](https://www.reddit.com/r/LocalLLaMA/comments/1o1wb1p/comment/nijypke/) (S) | 2025-10-09 | P40, PCIe x1 | Qwen3-30B-A3B UD-Q4_K_XL: tg 43 vs 37 at ctx 512; 36 vs 21 at 4k | M-same |
| B8 | For MLA models (DeepSeek-style) FA on is 75–326 % slower decode on sm_61 | [pxa docs/KNOWN-ISSUES.md](https://github.com/poisonxa16/pxa/blob/main/docs/KNOWN-ISSUES.md) (P, citing community P40 runs) | 2026-10 | P40 | — | M (second-hand) |
| B9 | `GGML_CUDA_BLOCKING_SYNC` turns the host's busy-wait in `cudaStreamSynchronize` into a sleep; throughput "essentially the same" | [PR #26487](https://github.com/ggml-org/llama.cpp/pull/26487) (P, open) | 2026-08-03 | Quadro P5000 | CPU use only | M (no speed data) |
| B10 | A "custom fork binary with forced CUDA graphs" gave +4 % on Qwen3.6 with MTP | [Reddit 1uxln6i comment](https://www.reddit.com/r/LocalLLaMA/comments/1uxln6i/comment/oy4pvtk/) (S) | 2026-07-17 | 24 GB card (model unstated) | 49.3 → 51.2 t/s | A |
| B11 | Graphs on Kepler gave ~25–30 % | [PR #30727](https://github.com/ggml-org/llama.cpp/pull/30727) (P, open) | 2026-10-17 | 4 × K40c | Qwen3.6 Q8_0 ~45 → 55+ t/s | A (pending data) |
| B12 | ik_llama.cpp's graph-UID change skips CUDA-graph compatibility checks; only matters above ~100 t/s | [ik PR #2316](https://github.com/ikawrakow/ik_llama.cpp/pull/2316) (P) | 2026-08-14 | — | — | P |
| B13 | P102-100 loses 11 % decode at 150 W vs 250 W; 3 × GTX 1070 lose ~5 % at 120 W | [#15013 (karambaso)](https://github.com/ggml-org/llama.cpp/discussions/15013) (P), [Reddit 1u7tarc](https://www.reddit.com/r/LocalLLaMA/comments/1u7tarc/) (S) | 2025-09-14; 2026-06-16 | P102-100; GTX 1070 | Llama-3.1-8B Q4_0 55.76 → 49.61 | M-same; A |

### Inferences
- **GRAPH_OPT on Flint (my calc):** 49 layers have the Q/K/V fan-out from `attn_norm` (the code only targets that
  node). K and V are 2048 → 512 mat-vecs; with one row per block they occupy 512 blocks each, enough to fill 5 SMs, so
  the gain comes only from overlapping kernel tails and the per-node gaps: maybe 5–10 µs per layer, 0.25–0.5 ms per
  token, +1–2 %. The PR notes it breaks `-ot`; whether it works with multi-GPU layer split is untested.
- **Blocking sync** frees one of the Celeron's two cores during decode, which could help `llama-server`'s HTTP and
  sampling threads, but it adds wake-up latency at each of the ~5 per-token synchronizations. Net sign unknown.
- **Clock:** if our decode is instruction-bound (§3), the 75 W limit matters through the core clock it allows. B13 is
  on bandwidth-bound cards, where clock matters less; our card should be more sensitive (claim to test).

### Experiment ideas
- On Flint Q4_1 (5 cards) and UD-mirror v1: `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_DISABLE_FUSION=1`, both, neither;
  `llama-bench -p 0 -n 128 -r 5`. ~20 min. Check output equality with a fixed seed.
- With the owner's go-ahead only: lock clocks (`nvidia-smi -lgc`) at 1,500 / 1,650 / 1,750 MHz on one card and measure
  Llama-2-7B Q4_0 tg; the slope tells whether a higher power limit is worth asking for. ~20 min.

---

## 6. Launch-count reduction: what Pascal forks did

### Takeaway
Two independent sm_61/sm_60 projects measured double-digit decode gains from fusing small ops and keeping CUDA graphs
on — the same fixed-cost half of our token time. None of their patches targets Flint, but the fusion patterns
(add+RMSNorm, RMSNorm+mul+SwiGLU, strided copies, top-k) are generic.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| K1 | Five sm_61 changes to PrismML's llama.cpp fork give +51 % decode, bit-identical logits | [ArtyomITA/bonsai2-ternary-pascal](https://github.com/ArtyomITA/bonsai2-ternary-pascal) (P) | 2026-10-17 | GTX 1080 | 55.9 → 84.4 t/s (Ternary-Bonsai-2-9B, PQ2_0 + MTP). Worth inside the final build: graphs for ≤ 8 columns +17 %, PQ2 mat-vec +37 %, small-kernel rewrites +3 %, fused Hadamard+q8_1 +4.5 %, identical-output fusions +8 % (launches 976 → 755) | M-same |
| K2 | Same family of patches on a 1080 Ti: 86 t/s on code with an MTP drafter, 42.6 t/s plain decode | [killy-netsphere/bonsai2-1080ti](https://github.com/killy-netsphere/bonsai2-1080ti) (P) | 2026-09-25 | GTX 1080 Ti, memory +1000 MHz | Ternary-Bonsai-2-27B 7.2 GB; "a card driving a monitor loses ~13 %" | M (overclocked) |
| K3 | Removing a 32-byte local-memory stack frame from fused MMVQ with 2 rows per block: +5–7 % on P100; on sm_61 (1 row per block) SASS unchanged | [ik PR #2721](https://github.com/ikawrakow/ik_llama.cpp/pull/2721) (P) | 2026-10-21 | P100, P40, M10, RTX 4000 | P100 Llama-3.1-8B Q4_K_M 47.49 → 50.23; P40 47.77 → 47.75 | M-same |
| K4 | pxq_llama found a DMMV path on Pascal a measured loss and a CUDA-graph replay rewrite neutral-to-negative; recommends a q8_0 output head plus bit-exact DeltaNet decode fusion (+9 % together on P100) | [wepromen/pxq_llama README](https://github.com/wepromen/pxq_llama) (P) | 2026-07-19 | P100, V100, 1080 Ti | PXQU-16 decode 57.2 → 62.4 (P100) | M (vendor) |
| K5 | Qwen3.6-specific P100 patches (persistent kernels, reduced draft vocab) reached ~125 t/s without MTP on 2 × P100, but the author later saw quality drift | [Reddit 1x4cwiy](https://www.reddit.com/r/LocalLLaMA/comments/1x4cwiy/), [Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches) (S) | 2026-10-12 | 2 × P100 (sm_60) | KLD shown only as images | A (sm_60, quality unclear) |

### Inferences
- K1's identical-output fusions removed 221 of 976 launches for +8 %, i.e. ~0.36 ms per 100 launches on a GTX 1080
  (my calc). Our launches cost 5–7 µs each, so 100–250 removed launches would be 0.5–1.75 ms of 35.8 ms (+1.5–5 %).
  Which of K1's fusions already exist upstream for Flint's graph (mainline has some RMSNorm and top-k-MoE fusions) needs
  a node-by-node check with `GGML_SCHED_DEBUG` or nsys.
- K1's rule "keep a kernel only if bit-identical" is a good gate for our own kernel work: quality questions vanish.
- K3 matters if we adopt 2-row MMVQ on sm_61 (#25479 / ik #2701): check fused kernels for local-memory spills.

### Experiment ideas
- Dump Flint's decode graph (`GGML_SCHED_DEBUG=2`, one token) and count consecutive add→rms_norm→mul and
  norm→mul→swiglu chains that are not fused today. 1 hour, no rig time beyond one run.

---

## 7. Custom quant recipes for speed on old hardware

### Takeaway
Nobody in the community has published a "legacy-only" recipe aimed at Pascal decode speed for an MoE; the closest are
(a) terelabs' Flint recipe (everything but routed experts at Q8_0, for quality), (b) ByteShape's hardware-aware
per-tensor datatype search (for modern GPUs, where it picks IQ+K mixes), and (c) the PXA ecosystem's codebook formats
(closed kernels). The repeated practical choice of Pascal owners is simply Q4_0 or a K-quant MoE file.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| R1 | ShapeLearn "learns the optimal datatype per tensor"; ships separate CPU (K-quant) and GPU (IQ+K hybrid) files chosen by tokens/s on each device class | [byteshape/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-GGUF) (P), [blog](https://byteshape.com/blogs/Qwen3.6-35B-A3B/) (S) | 2026-05-19 | RTX 4080/4090/5060 Ti/5090/PRO 6000, CPUs, Pi 5 | RTX 4090 GPU-5 (4.15 bpw) 214.5 t/s; no Pascal device; method not described | M (vendor) |
| R2 | A 3090 owner found the ByteShape IQ4_XS file the best for hybrid offload of Qwen3.6 | [Reddit 1uxln6i comment](https://www.reddit.com/r/LocalLLaMA/comments/1uxln6i/comment/oxzl5uk/) (S) | 2026-07-16 | RTX 3090 | 96.1 t/s without MTP, 123.0 with n-max 2 | M |
| R3 | PXQ: learned codebook for expert tensors with per-row FP16 anchors and 4-bit sub-scales per 16 values, backbone kept MXFP4, fused MoE kernels for Pascal/Volta | [wepromen/pxq_llama](https://github.com/wepromen/pxq_llama) (P) | 2026-07-20 | P100, V100, 1080 Ti | PXQ4 4.27 bpw, PXQ3 3.27, PXQ2 2.27; vs upstream ik on a 1080 Ti: decode +25 %, prefill −10 % | M (vendor) |
| R4 | PXQN tiers "match classic quants at fewer bits" (KLD on Qwen3.8-27B); kernels ship compiled only, encoder via a licence server | [poisonxa16/pxa](https://github.com/poisonxa16/pxa) (P) | 2026-10-22 (v3.1) | P100, V100 | PXQN2 9.6 GB KLD 0.161; PXQN4 15.7 GB KLD 0.0079; plain decode +5–35 % vs their previous release; 1080 Ti and P40 "not measured" for v3 | M (vendor) |
| R5 | terelabs' Flint recipe keeps attention, dense FFN, shared expert, embedding and head at Q8_0 and compresses only routed experts | [terelabs/flint-GGUF](https://huggingface.co/terelabs/flint-GGUF) (S, already in [Flint notes §5](../../../llms-on-crypto-mining-rigs/docs/research/notes/flint_speedups.md)) | 2026-10-21 | — | KLD/BFCL table in notes | M |
| R6 | Gemma 4 QAT `q4_0` files are a common choice on old cards | [Reddit 1v6llio comment](https://www.reddit.com/r/LocalLLaMA/comments/1v6llio/comment/ozuhhhz/) and others (S) | 2026-07 | 1080 Ti | — | A |
| R7 | Unsloth's UD-Q4_K_XL for Flint "already has a lot of IQ4_XS in the later layers" | [unsloth/flint-GGUF discussion #6](https://huggingface.co/unsloth/flint-GGUF/discussions/6) (S) | 2026-10-22 | — | — | A (checkable from the header) |

### Inferences
- R1's method (search per-tensor datatypes against measured tokens/s on the target device) is exactly the shape of a
  rig-specific UD-mirror search; ByteShape's GPU files would be a poor fit here because IQ formats are the slow ones on
  our card. A per-tensor speed table measured on one P106-090 would let us run the same search ourselves.
- R7 matters for interpreting our UD-mirror: if UD-Q4_K_XL's later layers are IQ4_XS, the mirror replaces them with
  Q4_1/Q5_0, which on our card should be strictly faster per byte (§3).
- PXQ/PXQN and ternary formats are not usable for Flint without either closed tooling or retraining (class C).

---

## 8. Mining rigs, x1 risers, multi-card tricks

### Takeaway
Community experience matches the rig's own measurements: layer split over x1 costs load time and prefill, not decode;
tensor split over Gen1 x1 is unusable; there is no software PCIe-generation unlock for P104/P106. Useful folklore for
prefill exists (device order, TurboPrefill) but nothing new for decode.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| X1 | The $35 Russian rig is nine P106 6 GB cards; "30+ tokens a second" for Qwen3.6-35B-A3B in Ollama, default quant | [Reddit 1wz6ist](https://www.reddit.com/r/LocalLLaMA/comments/1wz6ist/comment/pe8v4u2/) (S); [Guru3D](https://www.guru3d.com/story/retired-ninegpu-mining-rig-bought-for-provides-54gb-vram-for-local-llm-inference/) (S) | 2026-10-06; 2026-10-10 | 9 × P106 (variant not stated), PCIe x1 | "30+" | A |
| X2 | That rig is a Manli barebones; a commenter's company holds pallets of them new in box | [r/LocalAIServers 1wz6jn2](https://www.reddit.com/r/LocalAIServers/comments/1wz6jn2/) (S) | 2026-10-06 | — | — | A |
| X3 | A 2024 eBay listing of the same 9 × P106-100 rig; Gigabyte's G431-MM0 mining server gives PCIe 1.0 x1 per slot | [Reddit 1houz68](https://www.reddit.com/r/LocalLLaMA/comments/1houz68/) (S) | 2024-12-29 | — | — | A |
| X4 | x1 mining boards made a 30B-A3B "not reasonably faster than on my CPU"; moving the same P106s to x4 helped "a little" | [Reddit 1wz6ist comment](https://www.reddit.com/r/LocalLLaMA/comments/1wz6ist/comment/pe9buwz/) (S) | 2026-10-06 | P106s | no numbers | A |
| X5 | Peak PCIe traffic during layer-split decode stayed under 400 MB/s | [Reddit 1xagsfh comment](https://www.reddit.com/r/LocalLLaMA/comments/1xagsfh/comment/phtxbq3/) (S) | 2026-10-20 | 3090 + 5060 Ti | Gemma 4 decode 40 → 30 t/s when split (slower card) | M |
| X6 | Tensor split collapses on Gen1 x1; TurboPrefill pipelines ubatches across cards for prefill | [sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill) (P); [#22926 comment](https://github.com/ggml-org/llama.cpp/issues/22926) | 2026-06 to 2026-10 | 10–12 × P104-100, Gen1 x1 | 12 × P104 Llama-3-70B prefill: tensor 1.18, layer 37, TurboPrefill 199 t/s; 10 × P104 GPT-OSS-120B 77 → 345 | M (prefill only) |
| X7 | Keeping the input embedding on the CPU adds a graph split in multi-GPU layer split; moving it to GPU 0 gave 1.5–3× prefill on 2 × 3090 | [#22926](https://github.com/ggml-org/llama.cpp/issues/22926) (P) | 2026-05-10 | 2 × 3090 | prefill only; closed as stale | M-same (prefill) |
| X8 | P104-100's VBIOS locks the link to PCIe 1.0 (LnkCap 2.5 GT/s); BIOS settings fixed a x1 downgrade to x4 | [CSDN laot007](https://blog.csdn.net/laot007/article/details/161087262) (S) | 2026-05-14 | P104-100, X99 | prefill 9.8 → 51 t/s with CPU-offloaded MoE | M |
| X9 | Driver patches can "software-enable PCIe v2" on mining cards (Russian comment) | [Reddit 1wz6ist comment](https://www.reddit.com/r/LocalLLaMA/comments/1wz6ist/comment/pedch5h/) (S) | 2026-10-07 | — | — | S; the only patch found ([cmpunlocker](https://next.tangled.org/clee.sh/cmpunlocker/blob/main/docs/ARCHITECTURE.md)) targets Ampere CMP device IDs |
| X10 | "Pascals must stay under 9.7 GB VRAM, else PCIe micro-paging tanks speed" | [Reddit 1wzkywp](https://www.reddit.com/r/LocalLLaMA/comments/1wzkywp/) (S) | 2026-10-07 | 3 × P102-100 + 3060, PCIe 1.0 | — | A |
| X11 | Mainline beat ik_llama.cpp on that hybrid rig: tg 10.86 vs 8.43 (ik used no CUDA graphs) | same thread (S) | 2026-10-07 | same | as stated | M-same |
| X12 | ik_llama.cpp "can double P40 performance" with several cards, using P2P at ~5 GB/s | [Reddit 1tgsesb comment](https://www.reddit.com/r/LocalLLaMA/comments/1tgsesb/comment/omj0idk/) (S) | 2026-05-18 | multi-P40 | Mistral 123B Q8: <7 → ~14 t/s | A |
| X13 | Card order via `CUDA_VISIBLE_DEVICES` and `-ts` let a x16 GTX 960 feed a x4 P104 | [CSDN laot007](https://blog.csdn.net/laot007/article/details/162419501) (S) | 2026-06-29 | P104 + GTX 960 | prefill 55 → 211 t/s; decode unchanged (30.4) | M (offload case) |
| X14 | P40s idle in P8 and need warm-up runs; `nvidia-pstated`/gppm lower idle power to ~10 W | [vLLM-Pascal README](https://github.com/Mikec78660/vLLM-Pascal); [Reddit 1x5blg4 comment](https://www.reddit.com/r/LocalLLaMA/comments/1x5blg4/comment/ph9nl24/) (S) | 2026-10 | P40 | — | A |

### Inferences
- **X1 vs our rig (my calc):** our bartowski Q4_0 Qwen3.6 runs 18.46 t/s on 4 cards
  ([baseline](../../../llms-on-crypto-mining-rigs/docs/benchmarks/2026-10-22-qwen-baselines.md)). Ollama's default
  is a K-quant, slower per byte on our card. A P106-090 rig doing "30+" would be 1.6× ours; a P106-100 rig doing it
  matches the 1.56× scoreboard ratio. Most likely the poster's cards are P106-100. Claim to check (asking the poster
  for `nvidia-smi -L` would settle it).
- X11/X12: ik's multi-GPU advantage depends on P2P and tensor-style splitting, both unavailable here.

---

## 9. Other engines and forks with Pascal paths

| # | Engine | Claim | Source | Date | Numbers | Cred. | Relevance here |
|---|---|---|---|---|---|---|---|
| E1 | vLLM-Pascal (Mikec78660) | Custom W4A16 split-K GEMV (dequantize each nibble once, FP32 FMAs) makes AWQ usable on P40/P100; on 2 × P40 with MTP it "meets" llama.cpp Q4_0+MTP; machine has PCIe Gen1 x1 links | [repo](https://github.com/Mikec78660/vLLM-Pascal) (P) | 2026-10-16 | Qwen3.8-27B AWQ TP=2: 4.08 → 13.2 t/s vs llama.cpp 13.6–13.7 | M-same | No gain; needs Python/torch on a 4 GB, no-AVX host |
| E2 | vllm-pascal-sm61 | Patches vLLM 0.31 for sm_61 with GGUF loading; tuned for a 4 GB GTX 1050 Mobile | [zxmaster/vllm-pascal-sm61](https://github.com/zxmaster/vllm-pascal-sm61) (P) | 2026-10-18 | none comparative | A | No |
| E3 | PXA v3.1 | llama.cpp-derived server for P100/V100/10-series; picks settings per card; layer split only for links < x4 or > 4 cards; ships a no-AVX2 CPU library | [poisonxa16/pxa](https://github.com/poisonxa16/pxa) (P) | 2026-10-22 | P100 Qwen3.8-27B PXQN2 plain decode 27.4 → 33.3 vs their previous release | M (vendor) | No Flint, sm_61 unmeasured, fast kernels closed |
| E4 | pxq_llama (ik fork) | "faster than mainline ik in every config measured" on Pascal/Volta MoE | [repo](https://github.com/wepromen/pxq_llama) (P) | 2026-07-20 | 1080 Ti: decode +25 %, prefill −10 % vs ik | M (vendor) | ik lineage needs AVX2 |
| E5 | Strata | MoE engine with an "experimental Pascal pipeline"; model-specific (Qwen3.8-Flash-Next, Swift); a 1050 Ti Mobile port claims 3× llama.cpp decode with experts offloaded | [Reddit 1xa0vfz](https://www.reddit.com/r/LocalLLaMA/comments/1xa0vfz/), [r/LocalLLM 1x1qh9n](https://www.reddit.com/r/LocalLLM/comments/1x1qh9n/) (S) | 2026-10-09/19 | 1050 Ti Mobile 8K: 7.88 → 24.53 t/s (offload) | M (offload) | No Flint |
| E6 | FreeToken-Pascal, pascallama.cpp | sm_61 ports; pascallama.cpp changes only MMQ tiles (prefill) | [Reddit 1wg4sns](https://www.reddit.com/r/LocalLLaMA/comments/1wg4sns/) (S); [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp) | 2026-09-14 | — | A | No |
| E7 | ExLlama / MLC-LLM on Pascal | No 2025–2026 Pascal numbers found; 2024 posts say EXL2 on P40 is FP16-starved and only P100 gains | [Reddit 1f6hjwf](https://www.reddit.com/r/LocalLLaMA/comments/1f6hjwf/) (S) | 2024-09-01 | P100 EXL2 36.96 vs GGUF Q8_0 21.48 t/s (sm_60) | M (P100 only) | No |
| E8 | ik_llama.cpp | Mainline beat it on a hybrid P102 rig (X11); ik's CUDA graphs are off below Ampere | Reddit; [ik docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md) | 2026-10 | — | M | No (no AVX, no P2P) |

---

## 10. Flint: everything the community has said (to 2026-10-23)

### Takeaway
Three days after release there is no Pascal, Maxwell or mining-card Flint report, no community legacy-only quant, and
no community mention of the MoE rescale, the Q4_0/Q4_1 NaN or the Q8_1 FP16 sums (those come only from the llama.cpp
PRs in our notes). Speed reports are from modern GPUs and Macs. The community's quality verdict so far is lukewarm.

### Cited findings

| # | Claim | Source | Date | Hardware | Numbers | Cred. |
|---|---|---|---|---|---|---|
| L1 | Flint on Vulkan with experts partly on CPU | [unsloth/flint-GGUF #1](https://huggingface.co/unsloth/flint-GGUF/discussions/1) (S) | 2026-10-21 | RX 7800 XT 16 GB, `-ncmoe 24` | UD-Q4_K_XL: tg128 49.0 (48.3 at 16k depth), pp2048 1886 with `-ub 2048` vs 845 with 512 | M |
| L2 | Unsloth: "125 t/s on Mac, 150 on RTX Spark" | [unsloth/flint-GGUF #3](https://huggingface.co/unsloth/flint-GGUF/discussions/3) (S) | 2026-10-21 | — | — | A |
| L3 | vLLM NVFP4 single-stream 110–180 t/s; owner reran 45 LiveCodeBench and 50 SWE-bench Verified tasks above the published scores | [Reddit 1xc6j8t](https://www.reddit.com/r/LocalLLaMA/comments/1xc6j8t/) (S) | 2026-10-22 | RTX PRO 6000 | LCB 82.2, SWE 80 (small samples) | A |
| L4 | 128 t/s on an M4 Pro with the Kapitsa engine | [Reddit 1xcgzp4](https://www.reddit.com/r/LocalLLaMA/comments/1xcgzp4/) (S) | 2026-10-22 | M4 Pro 48 GB | — | A |
| L5 | Quality complaints: looping, overthinking, broken code at Q4 and Q8; one user says reports show it "broken in both llama.cpp and SGLang" from Q4 to FP8 | [Reddit 1xc23iq](https://www.reddit.com/r/LocalLLaMA/comments/1xc23iq/), [unsloth #3](https://huggingface.co/unsloth/flint-GGUF/discussions/3), [unsloth #1](https://huggingface.co/unsloth/flint-GGUF/discussions/1) (S) | 2026-10-21/22 | various | — | A (could be sampling/template) |
| L6 | NVFP4 follows instructions less strictly than 16-bit ("chattier") | [Reddit 1xc6j8t comment](https://www.reddit.com/r/LocalLLaMA/comments/1xc6j8t/comment/19rgs8mjq7grk/) (S) | 2026-10-22 | RTX PRO 6000 | — | A |
| L7 | Without a tool-call grammar Flint invents argument names: 41 of 85 tool calls failed in OpenCode; a schema grammar plus all four stop tokens (incl. `<\|observation\|>`) brought it to 0 of 50 | [RecursiveSI/flint #11](https://huggingface.co/RecursiveSI/flint/discussions/11) (S) | 2026-10-22 | mlx-vlm, 3-bit build | as stated | M |
| L8 | Speculative drafts (MTP/DFlash/EAGLE) requested; no answer | [RecursiveSI/flint #2](https://huggingface.co/RecursiveSI/flint/discussions/2), [unsloth #4](https://huggingface.co/unsloth/flint-GGUF/discussions/4) | 2026-10-21/22 | — | — | — |
| L9 | Users ask whether Recursive's Q4_1 is QAT/QAD and how it compares to UD; unanswered | [flint-int4-gguf #1](https://huggingface.co/RecursiveSI/flint-int4-gguf/discussions/1), [unsloth #5](https://huggingface.co/unsloth/flint-GGUF/discussions/5) | 2026-10-22 | — | — | — |
| L10 | KV is small: ~1 GB at 32k, ~7.5 GB at 262k (FP16), because only 14 layers are full attention | [RecursiveSI/flint #8](https://huggingface.co/RecursiveSI/flint/discussions/8) (S) | 2026-10-22 | — | estimate | S |
| L11 | Community files beyond those in our notes: Abiray Q4_K_M, lmstudio-community (no imatrix), prithivMLmods MXFP4_MOE, FutureMLS-Lab "nvfp4-coldgptq", HawkBearPig FP8/NVFP4, lilcheaty abliterated GGUF, OsaurusAI JANG (MLX), EigenLabs int4 | [HF model search "flint"](https://huggingface.co/models?search=flint) | 2026-10-20 to 22 | — | — | P |

### Inferences
- Nothing in the community data changes the plan in the Flint notes. L1 shows the Vulkan path works for Flint's
  sinks/SWA/per-head norms, which removes one risk from §4.
- L5/L6 are worth a guard: before attributing any quality loss to a quant, run the same prompts on the makers' Q4_1.
- No legacy-format Flint file other than bartowski's Q4_0/Q4_1 exists publicly; no Pascal speed report exists.

---

## 11. Qwen3.6-35B-A3B on Pascal

| # | Hardware | File | Engine / flags | Decode t/s | Source | Date | Cred. |
|---|---|---|---|---|---|---|---|
| Q1 | 1 × P40 | UD-Q3_K_XL | llama.cpp, `--spec-type draft-mtp` | 37.9 none; n-max 2: 51.6; 3: 48.3; 4: 41.9; 6: 34.5; 50.5 at 131k ctx (q8 KV) | [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978) (P) | 2026-07-21 | M-same |
| Q2 | 1 × P40 | UD-IQ4_NL | MTP, CPU vs GPU draft sampling | 59.6–75.9 → 61.6–79.4 (+~4 %) per task | [Reddit 1vf8obs](https://www.reddit.com/r/LocalLLaMA/comments/1vf8obs/), [PR #25532](https://github.com/ggml-org/llama.cpp/pull/25532) | 2026-08-04 | M-same |
| Q3 | 1 × P40 | UD-Q4_K_M | TurboQuant fork, MTP | avg 62 (53–69), burst 80 | [Reddit 1uxln6i](https://www.reddit.com/r/LocalLLaMA/comments/1uxln6i/) | 2026-07-15 | M |
| Q4 | 2 × P40 | UD-Q4_K_XL | MTP n-max 4 | 50–70 | [Reddit 1w54s3q](https://www.reddit.com/r/LocalLLaMA/comments/1w54s3q/) | 2026-09-02 | A |
| Q5 | 1080 Ti + 2 × P102-100 | UD-Q4_K_S / NVFP4 | Vulkan b10107 | 50.48 / 43.50 | [Reddit 1v6llio](https://www.reddit.com/r/LocalLLaMA/comments/1v6llio/) | 2026-07-25 | M-same |
| Q6 | 3 × GTX 1070 | MXFP4_MOE / UD-Q4_K_S / UD-IQ4_NL | Vulkan b9204, 120 W | 40.24 / 45.30 / 41.70 | [Reddit 1u7tarc](https://www.reddit.com/r/LocalLLaMA/comments/1u7tarc/) | 2026-06-16 | M-same |
| Q7 | 9 × P106 (variant?) | Ollama default | Ollama | "30+" | X1 | 2026-10-06 | A |
| Q8 | 4 × P106-090 (our rig) | bartowski Q4_0 | llama.cpp | 18.46 | [baseline](../../../llms-on-crypto-mining-rigs/docs/benchmarks/2026-10-22-qwen-baselines.md) | 2026-10-22 | M |
| Q9 | 4 × K40c (Kepler) | Q4_K_XL | MTP, BF16 cuBLAS fix (PR closed unmerged) | 17.5 none → 25 with 2 drafts | [Reddit 1uwjd45](https://www.reddit.com/r/LocalLLaMA/comments/1uwjd45/), [PR #25680](https://github.com/ggml-org/llama.cpp/pull/25680) | 2026-07-14 | A |
| Q10 | 1 × P102-100 | — | — | ~40 | [Reddit 1uvcjd0 comment](https://www.reddit.com/r/LocalLLaMA/comments/1uvcjd0/comment/oxa1twa/) | 2026-07-13 | A |
| Q11 | RTX 3090 (reference for MTP depth) | ByteShape IQ4_XS | MTP sweep | 96.1 none; n=2 p_min 0: 123.0; n=4/6 at p_min 0.75: 101.7/103.9 | R2 | 2026-07-16 | M-same |

Also: Gated DeltaNet ops work on sm_61 (Q1); an out-of-bounds read in `gated_delta_net_cuda` on SM61 was reported and
closed ([#26546](https://github.com/ggml-org/llama.cpp/issues/26546), 2026-08-04); a recurrent model decoded 39 %
slower than a bigger-active attention MoE on 4 × P40 with the per-token SSM step as the suspect
([#27980](https://github.com/ggml-org/llama.cpp/issues/27980)), so Qwen3.6's 30 GDN layers likely carry a similar
fixed cost on our cards.

**Inference:** for Qwen3.6 on the rig, `--spec-draft-n-max 2` (not the default 3) and Q4_0/legacy experts are the
community-supported settings; depth ≥ 4 is likely a loss on 5-SM cards, which are even more compute-poor than a P40.

---

## 12. Folklore that hints at something real

| Saying | Where | Verdict |
|---|---|---|
| "Compile with FORCE_MMQ for P40/1080" | Reddit 2023–2024 (B2) | True then (≈2×), obsolete: MMQ is default on Pascal and decode uses MMVQ (B1) |
| `LLAMA_CUDA_FORCE_DMMV=1`, `LLAMA_CUDA_F16=1`, `-DGGML_CUDA_F16=OFF` | 2026 P40 scripts (B4) | No-ops in current builds (B3); a 2026 fork measured a DMMV path on Pascal as a loss (K4) |
| "P40 hates K-quants" | Reddit 2024 | False on a P40 for decode (F1, equal per byte); true on few-SM, high-bandwidth cards like ours (F3, −22 %) |
| "IQ quants are slow on Pascal" | Reddit 2024 (F8), 2026 (F6, F7) | Mostly a prompt-processing effect; for decode, IQ4 is a few to ~20 % slower than Q4_0/Q4_K (F5, F6) |
| "Repack/requant to Q4_0 for P40" | implied by F9, R6 | Consistent with data on our card; the one fork that benchmarked against llama.cpp used Q4_0 as the reference |
| "-ub 1 makes it faster" | not found anywhere | No evidence; `-ub 1` appears only as a way to exercise the single-token kernels in KLD tests (K1, K3). Decode always uses MMVQ for ≤ 8 columns |
| "x1 risers kill multi-GPU" | many threads | False for layer-split decode (X5), true for tensor split (X6) and load time |
| "Vulkan is slower than CUDA on NVIDIA" | general belief | For prefill, yes (2–5×); for Pascal decode, often the opposite (V1–V4) |
| "More MTP drafts = faster" | MTP guides | On Pascal the optimum is 2; 6 is slower than none (Q1) |
| "FA is bad on Pascal" | P40 threads | For GQA models FA on is faster for decode (B7); for MLA models on sm_61 it is much slower (B8) |
| "FP16 KV/tensors are slow on Pascal" | many | llama.cpp upcasts to FP32 on Pascal; a P40 owner says FP16 KV is fine ([1tgsesb](https://www.reddit.com/r/LocalLLaMA/comments/1tgsesb/comment/omjl2ji/)); another found F16 KV gave better MTP acceptance than Q8 ([1w8sqp3](https://www.reddit.com/r/LocalLLaMA/comments/1w8sqp3/)) |
| "A card driving a monitor loses ~13 %" | K2 | Plausible for our rig, where Xorg holds a context on all nine cards; untested here |
| "Mining cards are locked to PCIe 1.1" | HN 2023 ([38187178](https://news.ycombinator.com/item?id=38187178)), CSDN (X8) | True (VBIOS); irrelevant to layer-split decode |

---

## 13. Top candidates for a deep dive, ranked

1. **Vulkan vs CUDA on the P106-090** (§4). Largest uncertain upside for the least work, no file change, carries to
   Qwen3.6. Strongest community evidence is on the same chip family. Its result also decides which formats to tune
   for, since Vulkan's per-format costs differ (F4).
2. **Launch reduction by fusion, using the Bonsai sm_61 patch as a template** (§6), plus the free switches
   `GGML_CUDA_GRAPH_OPT` and `GGML_CUDA_DISABLE_FUSION` (§5). Attacks the 10–13 ms fixed part of Flint's token;
   bit-exact; complements the rescale work in the Flint notes.
3. **A measured per-format, per-SM-cycle table on one P106-090** (§3), including Q4_1, Q5_0, IQ4_NL and MXFP4, then a
   ShapeLearn-style per-tensor search for Flint against that table (§7). Turns community ordering into rig numbers.
4. **Core-clock sensitivity** (§5): a 20-minute test that tells whether asking for a higher power limit is worth it.
5. **Qwen3.6 MTP at depth 2 with GPU draft sampling** (§11): secondary model only, already supported upstream.

---

## Claims worth testing on our rig, ranked

Ranked by expected value per hour; all speed tests are rig-only, quality checks use the quant machine.

1. **"Vulkan decodes faster than CUDA on GP106"** (V1, V3). Test: single-card Llama-2-7B Q4_0/Q8_0/Q4_K_M, then Flint
   Q4_1 and UD-mirror v1 on 5 cards; record `int dot` and host CPU load. ~2 h plus a cross-build. Adopt at ≥ +5 % on
   Flint.
2. **"`GGML_CUDA_GRAPH_OPT=1` gives +1–9 % decode"** (B5). Env var on the current build, Flint and Qwen3.6. 15 min.
   Check it does not break layer split or change outputs.
3. **"MMVQ fusion off speeds legacy quants on sm_61"** (B6). `GGML_CUDA_DISABLE_FUSION=1` on Flint Q4_1, UD-mirror v1,
   Qwen3.6 Q4_0. 15 min.
4. **"MTP depth 2 is the Pascal optimum; 6 is worse than none"** (Q1, Q2, Q11). Qwen3.6 Q4_0 with n-max 1/2/3/4,
   GPU draft sampling on. 30 min.
5. **"Decode on an issue-bound card scales with core clock"** (§3, B13). Needs the owner's go-ahead for `nvidia-smi
   -lgc`; 20 min. If the slope is near 1, a power-limit or clock change is a real lever.
6. **"Q4_0 ≫ IQ4_NL ≥ MXFP4/NVFP4 per byte on Pascal"** (F4–F6). Single-card per-format table (§3 experiment); 1.5 h.
   Directly prices the rescale-safe IQ4_NL down-projection in Flint layout A.
7. **"Fusing add+norm and norm+mul+SwiGLU removes ~20 % of decode launches for +8 %"** (K1). Count fusable chains in
   Flint's graph first (1 h); prototype only if ≥ 100 launches per token are fusable.
8. **"A display context costs ~13 %"** (K2). Compare a card with and without Xorg's context only if the owner wants to
   change the desktop setup (it is a system change); low priority.
9. **"The $35 nine-P106 rig does 30+ t/s on Qwen3.6"** (X1). Not a rig test: confirm the card variant from the poster
   before treating it as a contradiction of our 18.5–19.6 t/s.
10. **"Blocking sync frees a host core at no speed cost"** (B9). Only if the host CPU turns out to be a bottleneck in
    the Vulkan or server tests; needs a patched build.
