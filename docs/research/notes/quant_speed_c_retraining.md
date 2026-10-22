# Survey C: trained changes that make Flint decode faster on the P106 rig

*Class C: changes that need training, sized for the quant machine (3 × RTX 3090 24 GB, 2 × V100 32 GB without BF16,
2 × RTX 3080 10/12 GB, 94 GB RAM, slow PCIe, no P2P, no paid cloud). Researched 2026-10-22/23.*

Conventions:
- "(my calc)" marks this note's own arithmetic. "Claim to test" marks anything unmeasured on the rig or unconfirmed by a primary source.
- P = primary (paper, PR, issue, source code, model card), S = secondary (blog, forum, aggregator).
- "Local tree" means the llama.cpp checkout at a local llama.cpp checkout, commit `d77256999` (2026-10-22), the rig's build.

**Rig time model used throughout (my calc, from the brief and `flint_speedups.md`):**
- Plain token time T1 is 35.8 ms in llama-bench and 41.2 ms served with `-bs` (24.3 tok/s).
- Token time = fixed per-kernel cost + bytes ÷ η.
  - Fixed cost: 10–13 ms from ~1,900 launches; the rescale accounts for 2.5–4.7 ms of it.
  - η: Q4_x 64–73 GB/s, Q8_0 85–93 GB/s.
- A verify step of b tokens costs r(b)·T1:
  - r(2) = 1.03–1.13, measured as two slots.
  - r(3) = 1.1–1.25 and r(4) = 1.2–1.4 are claims to test.
  - The MUL_MAT_ID matvec path covers up to 6 tokens for Q4_x experts on sm_61.
- A drafter round with k drafts yields τ tokens (accepted drafts plus the target's own token) and costs k·d + d_enc + r(k+1)·T1.

---

## 1. Executive summary

1. **An EAGLE-3 head is the largest trained lever and needs no llama.cpp code.**
   - Mainline `draft-eagle3` is target-agnostic. It reads the target's layer inputs, which Flint's graph already records (`res->t_layer_inp[il]` in `src/models/flint.cpp`).
   - It can borrow Flint's token embedding and output head, or carry its own reduced-vocabulary head through a `d2t` table.
   - It converts with `convert_hf_to_gguf.py <head> --target-model-dir <flint HF dir>`.
   - Predicted served speed is **1.3–2.0× (mid ≈ 1.6×, 24 → ~39 tok/s)** (my calc), because Flint's verify step is nearly free and the draft step costs ~2.5–4 ms.
   - Training cost: ~2–4 days wall-clock, ~60–150 GPU-hours, mostly for generating Flint's own responses.
2. **Measure before training.** Both measurements fit in about a day, without training, and give the go/no-go:
   - `--spec-synth-len` (synthetic acceptance) on the rig prices the verify side.
   - A random-weight EAGLE-3 GGUF with Flint's shapes prices the draft step.
   - The risk they rule out: MoE targets have run *slower* with EAGLE-3 elsewhere. gpt-oss-120b on a DGX Spark ran at 0.46–0.71× because verification reads more experts. The rig's two-slot data says Flint's expert matvecs are latency-bound, so this should not apply here, but it must be checked.
3. **A 32k draft vocabulary is worth +8–12 % on top of any drafter** (my calc). The full 150k head costs ~2.7 ms per draft step on the rig; a 32k Q8_0 head costs ~0.8 ms. EAGLE-3, DFlash and DSpark GGUFs all support `d2t` in the local tree.
4. **Block drafters (DFlash, DSpark, LiLiCorr) are also supported for any target that records layer inputs.**
   - They draft a whole block in one pass. On Qwen3-8B the paper's average τ is 6.5, against 3.0 for EAGLE-3.
   - On this rig, verify widths above 6 tokens leave the cheap matvec path, so the blocks must be capped at about 5 drafts.
   - Training is ~3–5× EAGLE-3.
   - The concept can be checked now with z-lab's existing Qwen3.6-35B-A3B DFlash drafter.
5. **The other drafters are poor fits.**
   - An MTP head for Flint needs a per-architecture MTP graph in llama.cpp. Medusa is not supported.
   - A transplanted small draft model gives 1.1–1.35× (my calc) at a higher training cost.
   - Training-free layer-skip self-speculation (SWIFT, Draft&Verify, CLaSp, LayerSkip without its training) gives 0.9–1.1×, because a half-model draft costs ~0.5·T1.
   - Lookahead/Jacobi decoding needs 50–75-token verify steps and loses here.
6. **Most of the quantization error sits in the tensors that are read every token.**
   - Imatrix Q4_1 everywhere scores 0.280 KLD. Moving only the expert down-projections to Q8_0 gives 0.257.
   - UD-mirror v1, which keeps attention, shared expert and head mostly at Q8_0, scores 0.129.
   - Attention, shared expert and head are therefore where 4-bit hurts, and they are also where the bytes per token are.
7. **Cheapest trained step: block-wise QAT of only attention, shared expert, dense FFN and head to Q4_0, with experts left as in UD-mirror.**
   - About 1.5 B trainable parameters; hours on one 3090.
   - Bytes per token fall from ~2.26 GB (UD-mirror v1, my calc) to ~1.54 GB.
   - Predicted +10–20 % over UD-mirror v1 at the same quality, if QAT closes the gap (claim to test).
8. **Full block-wise QAT to Q4_0 everywhere plus IQ4_NL expert down-projections** reads 1.45 GB per token and makes removing the rescale exact.
   - Predicted 30–37 tok/s in llama-bench (+10–31 % over the makers' Q4_1 at 27.9; +20–45 % over UD-mirror v1's predicted 23–27) (my calc).
   - One Flint layer (831 M parameters including all 256 experts) fits one 3090 for EfficientQAT-style Block-AP: ~17–20 GB with FP32 master weights and Adam.
   - All 49 layers take ~12–40 GPU-hours sequentially (my calc).
   - V100s can only help in FP32.
9. **Recursive already proved that 4-bit Flint can be recovered by training.**
   - They quantized everything, including the head, to 4 bits and recovered it with offline distillation: the quantized model trained on its own pre-sampled responses with the BF16 model as teacher.
   - Their objective was task quality, not KLD to BF16, which is why their Q4_1 scores 0.316 KLD yet equals BF16 on BFCL. A KD-on-logits objective targets our KLD bar directly.
10. **Structural pruning is a poor trade here.**
    - Dropping 8 of 49 layers would give +17–19 % (my calc), but Qwen-family models already collapse after ~20 % layer removal even with healing.
    - Query-head pruning or low-rank q/o projections give +9–12 % at high risk. The healing budgets that work (Minitron: ~94 B tokens) are out of reach.
    - Expert pruning (REAP) shrinks the file but not the bytes read per token, so it gives no speed.
11. **Smaller levers.**
    - A trimmed or factorized output head gives +4–7 % (my calc).
    - Activation sparsity is unusable without a column-blocked weight format and a gather kernel, because GGUF blocks run along the input dimension.
    - Long context: converting half of the 14 full-attention layers to sliding-window would save ~0.47 GB per token at 32k (+~19 % there) but needs long-sequence healing.
12. **Compute feasibility is better than expected.**
    - transformers has native Flint since 2026-10-21 (PR #49824), with grouped-GEMM experts on sm80+. SDPA cannot apply attention sinks, so use eager or flex attention.
    - llama.cpp can dump any layer's input from *any* GGUF through `llama_set_embeddings_layer_inp` (`src/llama-ext.h`). Drafter features can therefore come from the exact quant the rig serves.
    - vLLM and SGLang Flint support targets Hopper/Blackwell. llama.cpp's own finetune is FP32-only. Unsloth has no Flint training support.
13. **Qwen3.6 carry-over.**
    - Drafting is already covered by its MTP layer (76 % acceptance at k = 3 on the rig).
    - What carries: the MTP draft-vocabulary trim (open PR #29143; +9–13 % by my calc, training-free) and z-lab's DFlash drafter.
    - The QAT pipeline carries over directly (same 256 × 512 × 2048 expert shapes), but Qwen3.6 already runs Q4_0, so QAT buys quality, not speed.

---

## 2. Lever table

Speed figures are single-stream decode on the rig. The served baseline is 24.3 tok/s (T1 = 41.2 ms); llama-bench is 27.9 tok/s (35.8 ms). All are claims to test unless marked measured.

| Lever | Class | Mechanism | Expected effect on rig decode (arithmetic) | Quality effect | Effort on our machines | Evidence | Carries over to Qwen3.6? |
|---|---|---|---|---|---|---|---|
| **EAGLE-3 head, 32k draft vocab** | C | 1-layer drafter fed target layers 2/24/46; chain drafts k = 2–3 | Round = 3·(2.5–4) + 1 + (1.2–1.4)·41.2 = 58–71 ms; τ = 2.3–2.9 → 32–50 tok/s served, **1.3–2.0×** (mid 1.6×) | Lossless (verified by target) | 2–4 days wall, 60–150 GPU-h (mostly response generation) | Medium: code read; τ from MoE EAGLE-3 cards; verify cost from 2-slot data | Not needed (has MTP) |
| Draft-vocab reduction 150k → 32k (`d2t`) | C (fixed at training) | Smaller drafter LM head | 192 MB → 70 MB per draft step, −2 ms × 3 drafts = −6 ms per ~64 ms round → **+8–10 %** on top of a drafter | ~none (coverage ~97–99 %) | Part of the head training | Medium (FR-Spec, VocabTrim, PR #29143) | **Yes**: MTP head trim, +9–13 % (PR #29143 open) |
| DFlash / DSpark / LiLiCorr block drafter | C | ~5-layer drafter drafts a block in one pass | d_block 5–8 ms; verify ≤ 6 tokens r ≈ 1.4–1.7 → round 63–78 ms; τ 3–4 → **1.6–2.6×** (wide uncertainty) | Lossless | 4–8 days, 3–5× EAGLE-3 GPU-h | Low–medium | Drafter exists (z-lab), test now |
| MTP head (DeepSeek-style) | C | NextN block | Same as EAGLE-3 class, but needs a Flint MTP graph in llama.cpp | Lossless | + 1–2 weeks of engine code | — | Already has one |
| Medusa / Hydra | C | Parallel heads | Not in llama.cpp | — | — | — | — |
| Small same-tokenizer draft (vocab-transplanted 0.15–0.6 B) | C | `draft-simple` | d = 5–10 ms (0.12–0.25·T1), a = 0.5–0.65 → **1.1–1.35×** | Lossless | 0.3–2 B tokens of fine-tuning: 1–3 days on 3 × 3090 | Low | Qwen3.5-0.8B exists; hybrid-state checkpoints make it lose |
| Training-free layer-skip self-speculation | A/B | Draft = target with 40–60 % of layers skipped | d ≈ 0.45–0.6·T1 → **0.9–1.1×** | Lossless | Engine work, days | Medium (papers); none in llama.cpp | Same |
| Lookahead / Jacobi | A | Parallel n-gram pool | 50–75-token step ≈ 10·T1 → **loss** | Lossless | — | llama.cpp example, no gain even on a 3090 Ti | Same |
| **Partial QAT: attention + shared + dense + head → Q4_0**, experts as UD-mirror | C | Block-wise reconstruction on ~1.5 B parameters | 2.26 → 1.54 GB per token; 27–32 tok/s bench vs predicted 23–27 for UD-mirror v1 → **+10–20 %** | Aim: within noise of UD-mirror v1 | Hours to 1 day, 3–8 GPU-h | Medium (error concentrated there, per our KLD table) | Yes (quality of Q4_0) |
| **Full block-wise QAT → Q4_0 + IQ4_NL expert down** (+ rescale removed) | C (+ one-line patch) | Train weights and scales per block against BF16 outputs, then end-to-end scales | 1.66 → 1.45 GB; fixed 7.5–10 ms → 27–33 ms → 30–37 tok/s bench: **+10–31 % vs makers' Q4_1, +20–45 % vs UD-mirror v1** | Target KLD ≤ 0.14 from PTQ's 0.28: unproven | 2–4 days, 20–60 GPU-h | Medium (EfficientQAT, Gemma QAT, makers' own QAT) | Yes (quality only) |
| KD scale-tuning of the makers' Q4_1 | C | Train each 32-group's FP16 d/m against BF16 logits | 0 % speed vs makers' file | Unknown; may undo their task tuning | 1–2 days | Low | n/a |
| 4-bit type with 64-value groups (exact re-expression of makers' INT4) | B (C origin) | Q4_1 at 4.5 instead of 5.0 bpw | Their codes at −10 % bytes: 1.66 → ~1.51 GB → **+6–8 %** | Exact | New type + kernel, 2–4 days | — | No |
| Layer dropping (8 of 49) + healing | C | Remove most-similar layers, LoRA or KD heal | −237 MB and −320 launches → −5.1–5.7 ms → **+17–19 %** | Large; Qwen family collapses near 20 % removal | Healing on 164–328 M tokens: 1–3 days | Medium (Gromov et al.) | Same idea |
| Query-head pruning 32 → 16 / low-rank q,o | C | Fewer q heads or rank-512–683 factors | −411 M params = −257 MB (Q4_1) → **+11–12 %**; low-rank adds 98 launches → +9–12 % | High risk | Weeks; proper healing ~100 B tokens | Low | Less (10 attention layers) |
| Expert pruning (REAP 25–50 %) | C-lite | Drop experts, calibration only | **0 %** (still 6 experts per token) | 1–3 points on benchmarks for huge MoEs; KLD up | Hours | Medium | Same, no speed |
| Trimmed or factorized output head (+ KD) | C/B | Head 150k → 64k/32k rows, or rank-512 | −110 to −151 MB → **+4.5–6.5 %** | Trimmed head can't emit cut tokens | 1 day + engine change | Low | **Yes**, larger (248k head, 0.51 B params) |
| Activation sparsity (TEAL/CATS/ReLU-fication) | B + C | Skip weight columns of zero inputs | Ceiling 40 % of 1.32 GB = −0.53 GB; realistic 0–15 % with a new column-blocked format + gather kernel | TEAL: small loss; ReLU-fication needs ~100 B+ tokens | Weeks | Low | Partial |
| Long context: 7 of 14 full layers → SWA, or cross-layer KV sharing, + healing | C | Halve full-attention KV reads | At 32k: −0.47 GB → 51.3 → ~43 ms (**+19 %**); at 8k +5 % | Long-context retrieval risk | Long-sequence healing, 2–5 days | Low–medium | Less (10 full + GDN) |

---

## 3. Levers

### 3.1 EAGLE-3 drafter for Flint

#### Takeaway
EAGLE-3 is the one trained drafter that llama.cpp runs for an arbitrary target today. It is the best ratio of rig speed-up to training effort.
- Speed: a 1-layer head with a 32k head, used at k = 2–3 drafts, should give 1.3–2.0× served speed (my calc).
- Cost: training on the quant machine is feasible. The expensive part is regenerating ~100k responses with Flint, not training the head.
- Before training, measure the verify cost and the draft-step cost on the rig (experiments C1 and C2).

#### Cited findings
**llama.cpp support (local tree, P).**
- `--spec-type draft-eagle3` with `-md <head.gguf>`. Flags: `--spec-draft-n-max`, `--spec-draft-p-min`, and `--spec-draft-device`/`-devd` to place the head on one card.
- `--spec-draft-backend-sampling` is on by default and samples drafts with top-k 10 on the GPU.
- `--spec-draft-sampling probabilistic` enables rejection sampling.
- `common_speculative_types_from_gguf` auto-detects only MTP, DFlash and DSpark, so EAGLE-3 must be named explicitly (`common/speculative.cpp`).

**EAGLE-3 merged into mainline** ([PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039), 2026-06-12, P).
- Design: an encoder FC compresses three target layers' features; a single-layer decoder drafts; a `d2t` tensor maps draft rows to target ids.
- Follow-ups in the local tree: long-prompt segfault fix (#24707, 2026-06-17); "eagle3-v3" for gpt-oss with norm-before-fc (#25794, 2026-07-28).

**How the drafter attaches to the target (local tree, P).**
- `src/models/eagle3.cpp`:
  - Uses the draft's own `token_embd` if present, otherwise the target's (`ctx_other`).
  - Uses its own `output` if present, otherwise the target's head.
  - With `d2t`, scatters the draft logits into a full-vocabulary −inf tensor.
  - The drafter RoPE-encodes its own attention.
- `common/speculative.cpp` (`common_speculative_impl_draft_eagle3`) turns on extraction of three target layer inputs and aborts if a layer id exceeds the target's layer count.
- Extraction works for any architecture whose graph sets `res->t_layer_inp[il]`. `src/models/flint.cpp` does this for every layer (line 97), as do 27 other model files.

**Conversion (local tree `conversion/llama.py`, P).**
- An HF checkpoint is treated as EAGLE-3 when its config has `draft_vocab_size` and `num_hidden_layers == 1`.
- `--target-model-dir` supplies the tokenizer and `num_hidden_layers`.
- Target layers default to [2, L/2, L−3], i.e. **[2, 24, 46] for Flint**, unless `eagle_aux_hidden_state_layer_ids` is set.
- `t2d`/`d2t` are written as absolute target ids.
- `docs/speculative.md` names SpecForge `LlamaForCausalLMEagle3` and vLLM/AngelSlim `Eagle3LlamaForCausalLM` as supported formats.
- The tokenizer comes from the target dir via the GPT-2 BPE path. Flint's pre-tokenizer "flint" was added to the converter by PR #30928.

**MoE targets can lose.**
- On a DGX Spark, gpt-oss-120b with its EAGLE-3 head ran at 0.46–0.71× of baseline in both llama.cpp and SGLang at 3 or 8 drafts (single GPU, chain).
- The PR author attributes this to verification activating extra experts in a bandwidth-bound regime ([comment](https://github.com/ggml-org/llama.cpp/pull/18039#issuecomment-3755925892), 2026-01-15, P).
- The PR lists gpt-oss-20b and gpt-oss-120b heads "with performance issue due to its MoE arch".

**Acceptance of EAGLE-3 heads on MoE targets** (accepted length per round including the bonus token, at k = 1…5).
- [RedHatAI/gpt-oss-20b-speculator.eagle3](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3) (P; 0.9 B parameters; trained on gpt-oss-120b hidden states over Magpie + UltraChat):
  - coding 1.65 / 2.06 / 2.29 / 2.42 / 2.52;
  - QA 1.50–1.58 / 1.74–1.90 / 1.88–2.07;
  - math 2.43 at k = 3.
- [RedHatAI/Qwen3-30B-A3B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.eagle3) (P; 0.5 B):
  - HumanEval 1.81 / 2.44 / 2.90 / 3.21 / 3.44;
  - QA 2.44 at k = 3;
  - math 3.02 at k = 3.
- From `dense_27b_speedups.md`: a post-hoc EAGLE-3 head on Qwen3.6-27B reached acceptance 0.50 at n-max 3, against 0.72–0.77 for the model's own MTP on the same DGX Spark. Native MTP beats post-hoc heads.

**The original recipe** ([EAGLE-3, arXiv 2503.01840](https://arxiv.org/abs/2503.01840), 2025-03, P).
- Data: ShareGPT ~68k plus UltraChat-200K ~464k entries, with responses regenerated by the target.
- Optimizer: AdamW (0.9, 0.95), lr 5e-5, gradient clipping 0.5.
- "Training-time test": the head is trained on its own multi-step outputs.
- τ at temperature 0 with trees: 6.1–6.7 for LLaMA-3.1-8B. Chain drafts were used for SGLang (3) and vLLM (2).
- H100 SGLang throughput, LLaMA-3.1-8B MT-bench: 158 → 373 tok/s.
- No training hours are given.

**SpecForge** ([arXiv 2603.18567](https://arxiv.org/abs/2603.18567), 2026-03-19, P; [repo](https://github.com/sgl-project/SpecForge), v0.3.0 2026-08, P).
- Methods: EAGLE-3, P-EAGLE, EAGLE3.1, DFlash/DFlash2, Domino, DSpark.
- Modes: online and offline.
- Target backends: SGLang, HF, and a "custom" backend for models missing from HF/SGLang.
- Recipe: 1.4M Open-PerfectBlend conversations regenerated at temperature 0.8, 2 epochs, lr 1e-4.
- Training throughput on 8 × H200 at 4,096 context: Qwen3-30B-A3B 126k tok/s (TP 4); Llama-3.1-8B 127k tok/s.
- Training-time-test length: gains are steep from 1 to 3; the best length is ~3 for chat and ~13 for math and code. Use 3–5 under tight memory.
- Data mix moved Llama-3.1 Math500 τ from 2.82 to 3.48.

**vLLM speculators** (P docs: [offline](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_offline/), [online](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/); S: [v0.3.0 blog](https://vllm.ai/blog/2025-12-13-speculators-v030), 2025-12-13).
- Offline training caches 3 verifier layers plus the loss mask per sample. The tutorial estimates ~1.6 TB for 50k Llama-3.1-8B samples.
- `--draft-vocab-size 32000`, with d2t/t2d built from a token-frequency file.
- Hidden states are generated by a vLLM worker extension. Training supports MoE verifiers.

**No Flint drafter exists.**
- Discussion "MTP/Dflash2/Eagle4 coming?" ([#2](https://huggingface.co/RecursiveSI/flint/discussions/2), 2026-10-21/22, P) has no staff reply.
- A Hub search for "flint" (2026-10-23) lists only quants, conversions and fine-tunes.

#### Inferences
**Drafter cost on the rig (my calc).**
- Head shape: hidden 2048, 32 q heads × 128, 8 KV heads, FFN 6144.
  - Decoder layer: q 16.8 M (input 2 × 2048), k + v 8.4 M, o 8.4 M, FFN 37.7 M ≈ 71 M parameters.
  - 32k head: 65.5 M.
  - Encoder FC: 6144 → 2048 = 12.6 M.
- Per draft step at Q8_0: (71 + 65.5) M × 1.0625 B ≈ 145 MB → 1.6–1.7 ms at 85–93 GB/s.
- Plus ~25 graph nodes (~0.2–0.3 ms) and the per-call host overhead of a llama_decode on the 2-core Celeron (~0.5–1.5 ms, claim to test). That makes **d ≈ 2.5–4 ms (0.06–0.1·T1)**.
- With the target's 150k Q4_1 head instead: +2.7 ms, so d ≈ 5–7 ms.
- Encoder pass d_enc (FC on k+1 rows + call overhead) ≈ 1 ms.
- Feature transfer: 3 × 2048 FP32 per verified token = 24 KB, ~0.13 ms per token over x1. Each card ships only its own layers.

**Speed** (served T1 = 41.2 ms; r(3) = 1.1–1.25, r(4) = 1.2–1.4; τ from the MoE cards above, trimmed 0–20 % for a smaller home-made data set):

| k | Round time | τ | Served tok/s | vs 24.3 |
|---|---|---|---|---|
| 2 | 2·(2.5–4) + 1 + (1.1–1.25)·41.2 = 51–61 ms | 2.0–2.45 | 33–48 | 1.36–1.97× |
| 3 | 3·(2.5–4) + 1 + (1.2–1.4)·41.2 = 58–71 ms | 2.3–2.9 | 32–50 | 1.33–2.06× |

**Why Flint should not repeat gpt-oss's slowdown (inference, claim to test).**
- A 2-token step would read 1.33× the bytes of a 1-token step if experts were bandwidth-bound: 1.10 GB fixed + 0.55 GB × ~2 experts' worth.
- The rig measured only 1.03–1.13×, so the 5-SM card is latency-bound on the 6-expert matvecs and has spare bandwidth.
- On the DGX Spark, by contrast, gpt-oss verification is bandwidth-bound.
- Experiment C1 measures r(b) directly.

**Temperature.**
- The figures above are greedy. Flint's recommended sampling is temperature 1.0, top-p 0.95, top-k 20.
- With `--spec-draft-sampling probabilistic`, expect τ 10–25 % lower (claim to test).

**Tool calls.**
- A request carrying tools disables `-bs` for the whole request (lazy grammar; `flint_speedups.md` §6).
- Each verify step then copies k+1 logit rows of 600 KB over the last card's x1 link (~3.2 ms per row) and samples them on the Celeron (~3 ms per row).
- That adds ~20–25 ms per k = 3 round, cutting the gain to roughly 1.2–1.4× for agent requests with tools (my calc).
- Keeping backend sampling until the grammar triggers would be an engine change (class B).

**Chaining with n-gram drafts.** `--spec-type ngram-mod,draft-eagle3` keeps n-gram drafts for file rewrites. Priority is fixed with n-gram types first; the first type that returns a draft wins.

**Training recipe sized for the quant machine (my calc).**

| Step | Work | Time |
|---|---|---|
| Prompts | 60–100k: chat (UltraChat/Magpie), code (Magicoder/CodeAlpaca), tool calls (xLAM-60k / Hermes function calling, rendered with Flint's `<tool_call><arg_key>` template), reasoning | — |
| Responses | Regenerate with Flint at temperature 0.8–1.0, keeping `<think>` blocks. ~100k × ~1.5k tokens = 150 M tokens. Four llama-server copies (each V100 32 GB holds the 26 GB Q4_1 alone; 3090 pairs hold another), `-np 32`, aggregate 2–4k tok/s (claim to test) | 10–21 h |
| Features | **Online** preferred. Offline storage at 16 KB per token (4 × 2048 BF16: three aux layers + final hidden) would be 2.4 TB for 150 M tokens. Option A: SpecForge HF backend with transformers' native Flint in BF16 across 3 × 3090 + 1 V100 (claim to test). Option B: a small in-tree llama.cpp tool using `llama_set_embeddings_layer_inp` to stream features from the *deployment GGUF* to the trainer. B also gives features exactly as the rig sees them | — |
| Training | 1 layer + 32k head, training-time test 5–7: ≈ 6 GFLOP per token × 150 M tokens × 2 epochs = 1.8e18 FLOP. At ~14 TFLOPS sustained per 3090 that is ~36 GPU-h, ~12 h on 3 × 3090 with data parallelism. A 12 GB 3080 also fits: ~3 GB of parameters and optimizer state plus ~2 GB of 32k-vocab logits at 2k context and training-time test 7 | ~12–36 h |

- Cross-check: SpecForge's 126k tok/s on 8 × H200 for Qwen3-30B-A3B is ~16k tok/s per H200. Scaled by ~30× less compute, that is ~500 tok/s per 3090 for target + drafter, i.e. 300 M token-passes in ~2 days on one card. That is consistent with the table.
- Total: ~2–4 days wall-clock, 60–150 GPU-hours. A pilot with 20k prompts takes ~1 day.

**Drafter quantization.** Q8_0 is the safe choice; ~150 MB fits any card. Q4_0 saves ~0.5 ms per draft step at some acceptance risk.

#### Gaps
- No r(b) curve for Flint on the rig, and no draft-step timing for any EAGLE-3 head on Pascal. Users reported EAGLE-3 slowdowns on V100 (15–17 against a 27 tok/s baseline, in the #24593 thread, per `dense_27b_speedups.md`), with no root cause.
- Whether SpecForge's HF backend runs transformers' `FlintForCausalLM` across GPUs without P2P, and captures aux hidden states, is untested.
- τ at temperature 1.0 and on agent traces is not published for any comparable model.

#### Experiment ideas
- **C1 – verify cost and synthetic speed on the rig (1–2 h).**
  - `llama-bench -m flint-int4-q4_1.gguf -ngl 99 -fa on -p 1,2,3,4,5,6,7,8 -n 0 -r 3` gives r(b).
  - Then `llama-server … -bs --spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m {2,3} --spec-synth-len {2.0,2.5,3.0}` gives served tok/s for a zero-cost drafter at the τ a head should reach.
  - Subtract k·d + d_enc from C2. **Go if the predicted gain is ≥ 1.4×.**
- **C2 – draft-step cost with a random-weight head (3–4 h).**
  - Write an HF-format `LlamaForCausalLMEagle3` checkpoint: config hidden 2048, 1 layer, 32 heads, 8 KV heads, FFN 6144, `draft_vocab_size` 32000, `target_hidden_size` 2048, `eagle_aux_hidden_state_layer_ids` [2, 24, 46]. Use random weights and a d2t made from a token-frequency count of Flint outputs.
  - `python convert_hf_to_gguf.py ./flint-eagle3-rand --target-model-dir <dir with Flint config.json + tokenizer files only> --outtype q8_0`.
  - On the rig: `llama-server -m flint.gguf -md flint-eagle3-rand.gguf --spec-type draft-eagle3 --spec-draft-n-max 3 -devd CUDA4 -bs …`. Read the draft timing in the log; acceptance will be ~0.
  - This checks loading, feature extraction across 5 cards and the per-draft cost d.
- **C3 – pilot head (~1 day).**
  - 20k prompts, Flint responses, SpecForge (HF backend) or a speculators-format offline set built from llama.cpp features of ~30 M tokens at ~0.5 TB. 1–2 epochs, training-time test 5.
  - Measure τ at k = 1/2/3, temperature 0 and 1, on the four server_bench prompts plus a 20-turn agent replay.
- **C4 – full head** with 100k prompts, only if C3 reaches τ(k=3) ≥ 2.3.

**Carries over to Qwen3.6?** Not needed. Its built-in MTP already reaches 76 % at k = 3 on the rig. The same pipeline would apply to any other target without MTP.

---

### 3.2 Draft vocabulary reduction

#### Takeaway
- The draft head's vocabulary is chosen at training time.
- 32k rows instead of 150,144 cuts the per-draft-step head read from ~192 MB (the target's Q4_1 head) to ~70 MB (Q8_0). That is worth ~+8–10 % on top of the drafter (my calc).
- For Qwen3.6 the same trim applied to the MTP head is training-free and worth more (248k head).

#### Cited findings
- **EAGLE-3, DFlash and DSpark** GGUFs load an optional `d2t` and scatter draft logits into the full vocabulary (`src/models/eagle3.cpp`, `src/models/dflash.cpp`, local tree, P).
- **[FR-Spec](https://arxiv.org/abs/2502.14856)** (2025-02, ACL 2025, P): drafts from a frequency-ranked subset; cuts the LM-head computation by 75 %; 1.12× over EAGLE-2.
- **[VocabTrim](https://arxiv.org/abs/2506.22694)** (2025-06, P): training-free trim of the drafter head to the most-sampled target tokens; +16 % memory-bound speed-up for Llama-3.2-3B. Acceptance drops slightly.
- **A later trade-off paper** ([arXiv 2603.05210](https://arxiv.org/html/2603.05210), 2026-03, P) notes that any token outside the fixed subset forces a rejection.
- **Coverage example:** a Kimi EAGLE-3 card reports 97.5 % token coverage at 32k ([cm00cm/Kimi-K2.7-Code-EAGLE3](https://huggingface.co/cm00cm/Kimi-K2.7-Code-EAGLE3), S).
- **FastMTP** ([arXiv 2509.18362](https://arxiv.org/abs/2509.18362), 2025-09, P) adds language-aware vocabulary compression to a fine-tuned MTP head: 1.81× → 2.03×.
- **[llama.cpp PR #29143](https://github.com/ggml-org/llama.cpp/pull/29143)** (open since 2026-09-19, P): d2t support for qwen35/qwen35moe MTP.
  - The draft head becomes 32,768 rows (87 % smaller).
  - Qwen3.8-27B Q8 on an RTX PRO 6000: 43 → 97.9 tok/s at N = 3 (author's figures; the comparison baseline is plain decoding).

#### Inferences
- **Flint (my calc):** saving 2 ms per draft step × k = 3 is 6 ms per ~64 ms round, about +9 %.
- **Qwen3.6 (my calc).** The MTP draft reads a 248,320 × 2048 head (508 M parameters): ~286 MB at Q4_0, ~417 MB at Q6_K, so 4–6 ms per draft step.
  - Trimming to 32k saves 3.5–5 ms per draft step, ~10–15 ms per k = 3 round.
  - The measured round is ~118 ms (3.28 tokens at 27.7 tok/s), so the gain is **+9–13 %**.
  - Building the trimmed sidecar GGUF needs only a script: copy the NextN block, select head rows by token frequency, add d2t. Load it with `-md` once #29143 lands.

#### Gaps
- No token-frequency coverage numbers for Flint or for the owner's own workload.

#### Experiment ideas
- **Coverage (1 h).** Tokenize ~5 M tokens of Flint outputs from experiment C3's generation step. Report the share of tokens inside the top 16k, 32k and 64k.
- **Qwen3.6 MTP trim (half a day).** Apply #29143 to the rig build, build a 32k MTP sidecar, and compare served tok/s and acceptance with the built-in MTP at k = 2/3.

---

### 3.3 Block drafters: DFlash, DFlash2, DSpark, LiLiCorr

#### Takeaway
- These drafters produce a whole block per pass and reach much higher τ than EAGLE-3 in papers.
- llama.cpp supports all of them for any target with layer-input extraction.
- On this rig two things limit them:
  - Verification must stay within ~6 tokens to keep the expert matmuls on the matvec path.
  - The drafter has ~5 layers, so a draft pass costs 5–8 ms.
- Expected gain is 1.6–2.6× with wide uncertainty, at 3–5× EAGLE-3's training cost. Prove EAGLE-3 first; probe the concept with Qwen3.6's existing DFlash drafter.

#### Cited findings
- **llama.cpp** (local tree, P).
  - Support: DFlash ([#22105](https://github.com/ggml-org/llama.cpp/pull/22105), 2026-06-28), DSpark (#25173, 2026-07-28), LiLiCorr ([#30506](https://github.com/ggml-org/llama.cpp/pull/30506), 2026-10-15).
  - `src/models/dflash.cpp` shares the target's `tok_embd` and LM head through `ctx_other` unless the draft has its own reduced-vocabulary head plus d2t.
  - The DFlash converter (`conversion/qwen.py`, `DFlashModel.set_vocab`) calls the *target architecture's* own vocabulary handler, so a Flint target works.
  - Usage: `--spec-type draft-dflash|draft-dspark`. DFlash, DSpark and LiLiCorr are auto-detected from GGUF metadata.
- **[DFlash paper](https://arxiv.org/abs/2602.06036)** (2026-02-05, ICML 2026, P).
  - Recipe: ~800K Nemotron-V2 + CodeAlpaca samples with target-regenerated responses; 6 epochs; lr 6e-4; up to 3,072 tokens per sample with 512 anchors.
  - Drafter: 5 layers (8 for Qwen3-Coder-30B-A3B), block 16, 5 target layers.
  - Qwen3-8B at temperature 0: average τ 6.49 (HumanEval 6.50, MT-Bench 4.24), against 2.96 for EAGLE-3 at 16 drafts.
  - SGLang concurrency 1: 4.2–5.1×. No training hours are given.
- **LiLiCorr** ([arXiv 2608.20530](https://arxiv.org/abs/2608.20530), 2026-08, P): adds candidate correlation on top of DFlash; +9–19 % acceptance length.
- **In llama.cpp vs vLLM:** on 2 × R9700, llama.cpp accepted ~3 tokens per block against ~5.5 in vLLM ([z-lab/dflash #170](https://github.com/z-lab/dflash/issues/170), S, via `dense_27b_speedups.md`).
- **A Qwen3.6 drafter exists:** [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) (P), trained at 40k context with sliding-window attention.

#### Inferences
**Flint drafter (my calc).**
- 5 layers × ~56 M + FC 21 M ≈ 300 M parameters ≈ 320 MB at Q8_0 → 3.6 ms, plus ~75 nodes ≈ 0.5 ms.
- Plus the target head on the block positions (+2.7 ms), or a 32k d2t head (+0.8 ms). d_block ≈ 5–8 ms.

**Verify.**
- Cap `--spec-draft-n-max` at 5 so the batch stays ≤ 6 tokens, inside the MUL_MAT_ID matvec limit for Q4_x experts on sm_61.
- r(6) ≈ 1.4–1.7 (claim to test).
- Round ≈ 6.5 + (1.4–1.7) × 41.2 = 64–77 ms. With τ = 3–4 that is 39–62 tok/s, **1.6–2.6×**.

**Training (my calc).**
- Per sample: 512 blocks × 16 positions × 300 M parameters × 6 ≈ 1.5e13 FLOP.
- 100k samples × 2 epochs ≈ 3e18 FLOP → ~60 GPU-h on 3090s. Target features for 5 layers on ~300 M tokens add ~20 h.
- Total ~4–8 days wall-clock.
- SpecForge has DFlash online and offline training; HF-backend support is unconfirmed.

#### Gaps
- No Pascal measurement of any DFlash drafter. Qwen3.6's GDN layers make block rejections costlier (state rewrite), so a Qwen3.6 run is a pessimistic proxy for Flint.

#### Experiment ideas
- **D1 (2–3 h).** Convert z-lab's Qwen3.6 DFlash drafter: `convert_hf_to_gguf.py z-lab/Qwen3.6-35B-A3B-DFlash --target-model-dir <Qwen3.6 HF config+tokenizer> --outtype q8_0`.
  - Run on the rig with `--spec-type draft-dflash --spec-draft-n-max {3,5,7}` against `draft-mtp` (k = 3).
  - Record tok/s and accepted tokens per block.
  - If DFlash at n-max 5 beats MTP there, a Flint DFlash drafter becomes a serious second candidate.

**Carries over to Qwen3.6?** Yes, directly: the drafter already exists.

---

### 3.4 MTP-style head, Medusa, HASS (why not, for Flint)

#### Takeaway
- **MTP:** llama.cpp's `draft-mtp` is implemented per architecture. Flint has no NextN tensors, no `graph_mtp` and no converter support. EAGLE-3 gives the same class of drafter with existing code.
- **Medusa:** not implemented in llama.cpp.
- **HASS:** a training-recipe improvement, applicable inside EAGLE training.

#### Cited findings
- **NextN (MTP) blocks exist in 23 model files in the local tree** (grep for `nextn.eh_proj` in `src/models/*.cpp`; Flint is not among them).
  - Example: `qwen35moe.cpp` creates `nextn.eh_proj/enorm/hnorm/shared_head_*` and a `LLM_GRAPH_TYPE_DECODER_MTP` path.
  - Type detection looks for `blk.<last>.nextn.eh_proj.weight` (`common/speculative.cpp`, P).
- **FastMTP** ([arXiv 2509.18362](https://arxiv.org/abs/2509.18362), 2025-09, P): fine-tuning one shared MTP head on self-distilled data lifts acceptance to ~80 / 56 / 36 % at draft positions 1 / 2 / 3, against ~70 / 10 / ~0 % for vanilla reuse. Average 2.03× on 7B models.
- **MTP-D** ([arXiv 2603.23911](https://arxiv.org/abs/2603.23911), 2026-03, P): +7.5 % MTP-head acceptance via self-distillation.
- **HASS** ([arXiv 2408.15766](https://arxiv.org/abs/2408.15766), 2024-08, P): 8–15 % faster than EAGLE-2 by aligning training with decoding contexts and objectives.
- **Medusa** ([repo](https://github.com/FasterDecoding/Medusa), P): 2.2–3.6× with tree attention. Not among llama.cpp's speculative types (`common/common.h` enum, local tree).

#### Inferences
- **An MTP layer built as a Flint MoE block** (attention + 6 experts + shared expert) reads ~44 M parameters plus a head per draft step. That is the same order as the EAGLE-3 layer, so there is no speed advantage to justify the engine work.
- **For Qwen3.6,** FastMTP-style fine-tuning of the existing MTP layer could raise acceptance at k = 2–3, but 76 % aggregate at k = 3 already leaves little room.

#### Gaps
None that matter for a decision.

#### Experiment ideas
None. Use EAGLE-3.

**Carries over to Qwen3.6?** Its MTP exists; a FastMTP-style fine-tune is a low-priority option.

---

### 3.5 Small same-tokenizer draft model (100–500 M dense)

#### Takeaway
Feasible on the quant machine:
1. Transplant the vocabulary of an existing ~0.5 B model to Flint's tokenizer.
2. Fine-tune it on 0.3–2 B tokens.
3. Align it on Flint outputs.

It is still inferior to EAGLE-3 on this rig: 1.1–1.35×, because the draft reads its own 150k-row head and 12–28 layers on every draft token, without the target's features. Only worth doing if EAGLE-3 fails for an engine reason.

#### Cited findings
- **Compatibility check:** llama.cpp requires the draft and target to have the same vocabulary type, BOS/EOS, sizes within 128 tokens, and identical token text from id 5 on (`common_speculative_are_compatible`, `SPEC_VOCAB_MAX_SIZE_DIFFERENCE 128`, local tree, P).
- **Precedent:** [jukofyork/GLM-4.5-DRAFT-0.6B-v3.0](https://huggingface.co/jukofyork/GLM-4.5-DRAFT-0.6B-v3.0) (P, transplant log dated 2025-08-07).
  - Qwen2.5-0.5B-Instruct was moved to GLM-4.5's tokenizer with `transplant-vocab`. Head rows: 81 % copied, 19 % mean-initialized.
  - Then a full fine-tune on ~2.3 B tokens (common crawl, The Stack, instruct outputs) on 6 × A6000.
  - No acceptance figures.
- **Cross-vocabulary drafting without a transplant** (OmniDraft, [arXiv 2507.02659](https://arxiv.org/abs/2507.02659), 2025-07, P) is not available in llama.cpp.
- **Vocabulary overlap:** Flint's tokenizer shares 68.5 % of its token strings with Qwen3.6's (`flint_speedups.md` §0), which favours a Qwen3-family donor.

#### Inferences
**Draft cost on the rig (my calc).**
- Qwen3-0.6B-like (28 layers, hidden 1024, ~155 M-parameter tied head) at Q8_0 ≈ 0.6 GB → 7 ms of reads + ~400 launches ≈ 2 ms → ~10 ms per token (0.25·T1).
- A 12-layer × 768 model with a 115 M-parameter head ≈ 0.28 GB → ~5 ms (0.12·T1).
- Small independent drafts typically get per-token acceptance a = 0.5–0.65 (claim to test). At k = 3, τ = 1 + a + a² + a³ = 1.9–2.4.
- Round = 3d + 1.3 × 41.2 = 68–84 ms → 23–35 tok/s, **0.95–1.4×**, centre ~1.2×.

**Training cost (my calc).**
- 0.6 B model full fine-tune: 6 × 0.6e9 = 3.6 GFLOP per token. At ~14 TFLOPS per 3090 that is ~4k tok/s per card, ~12k tok/s on 3 × 3090.
- 1 B tokens takes ~23 h; 0.3 B tokens takes ~7 h.
- Plus 20–50 M tokens of Flint outputs for alignment (~0.5 day of generation).
- Training from scratch instead would need ~5 B tokens for 265 M parameters (Chinchilla-style): ~2–3 days on 3 × 3090. Teacher logits from Flint for that volume are out of reach (~14 days of Flint prefill at ~4k tok/s).

#### Gaps
- No published acceptance for transplanted drafts.

#### Experiment ideas
Only if C2 shows EAGLE-3 cannot be made to work:
- Transplant Qwen3-0.6B with `transplant-vocab` to Flint's tokenizer.
- Fine-tune 300 M tokens.
- `llama-server -md flint-draft.gguf --spec-type draft-simple --spec-draft-n-max 2`.

**Carries over to Qwen3.6?** Qwen3.5-0.8B shares its vocabulary already. On a hybrid (GDN) target every round checkpoints ~150 MB of recurrent state to host RAM (`dense_27b_speedups.md`), so expect a loss.

---

### 3.6 Training-free self-speculation, for comparison

#### Takeaway
- Layer-skipping self-drafters (SWIFT, Draft&Verify / Self-SD, CLaSp, LayerSkip used without its training) and Kangaroo (shallow exit + trained adapter) report 1.3–1.7× on A100-class GPUs, where verification is free.
- On this rig a half-model draft costs ~0.5·T1 in reads *and* launches, giving 0.9–1.1×.
- llama.cpp has none of these.
- Lookahead/Jacobi decoding exists only as an example program and loses on wide verify steps.

#### Cited findings
- **[CLaSp](https://arxiv.org/abs/2505.24196)** (2025-05, ACL 2025, P): 1.3–1.7× on LLaMA-3. On LLaMA-3-70B (greedy) 1.67×, against 1.47× for Self-SD and 1.28× for SWIFT.
- **[Kangaroo](https://arxiv.org/abs/2404.18911)** (2024-04-29, P): up to 1.68× on Spec-Bench with a 67 M-parameter adapter (Medusa-1: 591 M).
- **[LayerSkip](https://arxiv.org/abs/2404.16710)** (2024-04, P): needs training with layer dropout and an early-exit loss so that early layers can draft.
- **Lookahead decoding** in llama.cpp ([#4207](https://github.com/ggml-org/llama.cpp/pull/4207), merged 2023-11-26, P):
  - No speedup on Apple Silicon; ~10 % slower than plain decoding on a 3090 Ti.
  - The author notes it needs large batches and fast batched kernels for quantized models.
  - It lives in `examples/lookahead`, not in llama-server (local tree).
- **No layer-skip, early-exit or self-speculative-by-layers code** in `common/` or `src/` (local tree grep, P). A web search found no llama.cpp PR.

#### Inferences
**Layer-skip draft (my calc).**
- d ≈ 0.5·T1 when skipping 50 % of layers; the head is still read.
- At k = 2 and per-token acceptance a = 0.7–0.8: τ = 1 + a + a² = 2.19–2.44.
- Round = 2 × 0.5 + 1.15 = 2.15·T1 → **1.02–1.13×**.
- Skipping 65 %: d ≈ 0.38·T1, a drops to ~0.6 → τ = 1.96 against a 1.91·T1 round → **1.03×**.

**Lookahead (my calc).** A 64-token verify step runs at the measured 161–171 tok/s prompt speed, ~0.38–0.5 s (~10·T1). It would need >10 accepted tokens per step.

#### Gaps
None that would change the conclusion.

#### Experiment ideas
None. The n-gram drafters (`flint_speedups.md` experiment 2) are the training-free option that fits this rig.

**Carries over to Qwen3.6?** Same conclusion.

---

### 3.7 Full block-wise QAT into fast formats (Q4_0 / Q4_1 / IQ4_NL)

#### Takeaway
- **Goal:** train Flint's weights and quantization scales block by block against the BF16 model, then export bit-exact GGUF Q4_0 or Q4_1 blocks.
- **Target layout:** Q4_0 everywhere, IQ4_NL for the routed down-projections, BF16 router. That reads 1.45 GB per token and makes removing the rescale exact.
- **Predicted speed:** 30–37 tok/s llama-bench, against 27.9 for the makers' Q4_1 and a predicted 23–27 for UD-mirror v1 (my calc).
- **Cost:** one Flint layer, including all 256 experts, fits one 24 GB card for full Block-AP. All 49 layers take ~12–40 GPU-hours sequentially, or ~4–8 h wall-clock with parallel per-layer jobs.
- **The open question is quality:** can KD-QAT bring 4-bit attention, shared expert and head down to KLD ≤ 0.14? Section 3.8 is the cheap test.

#### Cited findings
**EfficientQAT** ([arXiv 2407.11062](https://arxiv.org/abs/2407.11062), v1 2024-07-10, v3 2025-05-19, ACL 2025, P; numbers from Table 7).
- Block-AP trains weights, scale s and zero point z per group, block by block. Then E2E-QP trains only the step sizes end to end.
- The quantizer is asymmetric uniform per group, which maps exactly onto Q4_1 (d = s, m = −z·s).
- Block-AP data: 4,096 RedPajama samples × 2,048 tokens, 2 epochs. E2E-QP: 4,096 × 4,096, 1 epoch.
- Time and memory on one A100-80GB:

  | Model | Block-AP | E2E-QP | Total |
  |---|---|---|---|
  | Llama-2-7B | 3.3 h, 8.5 GB | 1.5 h | 4.8 h |
  | Llama-2-70B | 26.6 h, 29.9 GB | 14.3 h, 48.4 GB at 4 bit | 40.9 h |

- Llama-2-7B w3g128: average accuracy 64.85 → 64.02 (wikitext PPL 5.47 → 5.81).

**Gemma 3 QAT** ([Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/), 2025-04, S, maker's own): ~5,000 QAT steps with the unquantized checkpoint's probabilities as targets. This cut the llama.cpp perplexity drop at **Q4_0** by 54 %. It is the closest precedent for our target format.

**torchao QAT** ([PyTorch blog](https://pytorch.org/blog/quantization-aware-training/), 2024-07, S).
- Llama3-8B with int4 grouped weights and int8 dynamic activations.
- Recovered up to 96 % of the HellaSwag degradation and 68 % of the wikitext perplexity degradation compared with PTQ.
- Its prepare/convert API swaps `nn.Linear` modules. Flint's experts are 3-D `nn.Parameter`s, so they need custom fake-quant.
- Unsloth integrated torchao QAT (torchao news feed) but lists no Flint support.

**AutoRound** ([intel/auto-round](https://github.com/intel/auto-round), v0.16.1, P).
- Sign-gradient tuning of rounding and clipping per block (200 iterations, 128 × 2,048 samples by default).
- Exports GGUF Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0, plus K-quants.
- "MoE tuning >3× faster" (2026-09). About 10 minutes per 7B on one GPU.
- A cheaper rung than Block-AP. Whether it handles Flint's fused 3-D experts is a claim to test.

**Methods that target lower bit widths:**
- PV-Tuning ([arXiv 2405.14852](https://arxiv.org/abs/2405.14852), 2024-05, P) targets 1–2-bit vector quantization (AQLM, QuIP#).
- BitDistiller ([arXiv 2402.10631](https://arxiv.org/abs/2402.10631), 2024-02, P) combines asymmetric quantization, clipping and confidence-aware KL self-distillation for 2–3 bits.
- ParetoQ ([arXiv 2502.02631](https://arxiv.org/abs/2502.02631), 2025-02, P) finds that at ≥ 3 bits fine-tuned models stay close to their pre-trained distribution, so 4-bit QAT is a gentle correction.
- LLM-QAT ([arXiv 2305.17888](https://arxiv.org/abs/2305.17888), 2023-05, P) distils on data generated by the full-precision model itself. That is the same idea as Recursive's coreset.

**MoE calibration imbalance:** [MoEQuant](https://arxiv.org/abs/2505.03804) (2025-05, ICML 2025, P) builds an expert-balanced calibration set by self-sampling. It reports +10 HumanEval points for DeepSeekMoE-16B at 4 bits.

**Recursive's own recovery** ([blog](https://www.recursive.com/updates/introducing-recursive-flint), 2026-10-21, S, maker's own; [flint-int4](https://huggingface.co/RecursiveSI/flint-int4), [flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf), P).
- The head was quantized to 4 bits, against common practice.
- Recovery was an offline distillation stage: the quantized model trained on its own pre-sampled responses from a self-discovered coreset, with the 16-bit model as teacher. It needed about half the steps of their QAT baseline.
- NVFP4 is within 0.7 points of 16-bit.
- The INT4 export is group-64 affine with an MSE scale rule. The GGUF stores "trained INT4 codes" as group-32 Q4_1 with FP16 scales and offsets.
- No data volume or GPU hours are published.

**Our measured PTQ data** (quants README, 80 × 512 wikitext tokens against Q8_0):

| File | Top-1 | KLD |
|---|---|---|
| Q4_1 imatrix everywhere | 80.9 % | 0.280 |
| Same, expert down-projections Q8_0 | 81.5 % | 0.257 |
| UD-mirror v1 | 87.4 % | 0.129 |
| Makers' Q4_1 | 78.6 % | 0.316 |

**Framework support** (P).
- transformers PR #49824 "Add Flint model support", merged 2026-10-21 (`0ea108daf`).
- `modeling_flint.py`:
  - Experts are fused 3-D parameters (`gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`) under `@use_experts_implementation`.
  - The router `gate.weight` and the `sinks` are kept FP32.
  - `_supports_sdpa = False` because SDPA cannot apply sinks; flex and flash variants are supported (the listed flash kernels are FA3/FA4).
- `integrations/moe.py`: grouped GEMM needs SM80+ with torch ≥ 2.9, otherwise a per-expert fallback.
- The HF repo's `config.json` names `transformers_version` 5.12.1 and has no `auto_map` (no remote code).

#### Inferences
**Bytes per token (my calc; parameters from the config).**
- Q4_0 = 0.5625 B per parameter, Q4_1 = 0.625, Q5_0 = 0.6875, Q8_0 = 1.0625.
- UD-mirror v1 recipe:
  - attention 42 layers Q8_0 + 7 Q5_0 = 933 MB;
  - experts Q4_1/Q5_0 (2 down layers Q8_0) = 585 MB;
  - shared expert mostly Q8_0 = 283 MB;
  - dense FFN 80 MB, head Q8_0 327 MB, BF16 router 49 MB.
  - Total **2.26 GB**: 36 % more than the makers' 1.66 GB.
- Predicted UD-mirror v1 bench speed: 1.49 GB Q8_0 at 85–93 GB/s (16–17.5 ms) + 0.72 GB Q4/Q5 at ~55–70 GB/s (10.3–13.1 ms) + router 0.7 ms + fixed 10–13 ms = 37–44 ms → **23–27 tok/s** (claim to test; UD-mirror's speed is pending).
- QAT target layout, all Q4_0 with IQ4_NL expert down (same bytes as Q4_0) and BF16 router:
  - attention 520 MB, experts 499, shared 167, dense 42, head 173, router 49 → **1.45 GB**.
  - 1.40 GB at 64–73 GB/s = 19.2–21.9 ms + router 0.7 ms.
  - Fixed cost with the rescale removed: 7.5–10 ms. Removal is exact because IQ4_NL ignores the Q8_1 sum.
  - Total 27–33 ms → **30–37 tok/s bench**.
  - Without removing the rescale (Q4_0 down-projections): 27–33 tok/s.
- With Q4_1 instead of Q4_0 everywhere: 1.66 GB, the makers' speed (27.9).
- A 64-value-group variant (section 3.9) would save a further 5–10 % of bytes.

**Memory per layer** (Block-AP, all 831 M parameters of a MoE layer trainable; my calc):

| Item | Size |
|---|---|
| BF16 working weights | 1.66 GB |
| FP32 master weights | 3.3 GB |
| Adam m and v | 6.6 GB |
| FP32 gradients | 3.3 GB |
| Dequantized temporaries | ~1.7 GB |
| Activations for 2 × 2,048 tokens with recompute | 1–3 GB |
| **Total** | **≈ 17–20 GB** |

- This fits a 3090.
- A 12 GB 3080 needs BF16 master weights and 8-bit Adam (~7–9 GB).
- A V100 fits it in pure FP32 (~22 GB). That is the only safe mode there: no BF16, and FP16 would overflow on Flint's large down-projection activations and residuals.
- Exposing those large activations is the reason llama.cpp needs the rescale for Q4_x down-projections at all.

**Time** (my calc, EfficientQAT budget of 8.4 M tokens × 2 epochs per layer).
- Active parameters per MoE layer are 44.6 M, so fwd + bwd ≈ 268 MFLOP per token, plus attention scores ≈ 22–50 MFLOP. Total ≈ 5e15 FLOP per layer, 2.5e17 for 49 layers.
- Each expert sees only ~100 tokens per 4,096-token step, so the GEMMs are tiny. Assume 5–10 TFLOPS sustained with grouped GEMM on a 3090: 7–14 h of compute.
- Adam and fake-quant over 831 M parameters × 4,096 steps (~20–25 ms each) add ~90 s per layer, ~1.2 h for all 49 layers.
- Without grouped GEMM (the per-expert Python loop, which is what V100s get) launch overhead can add several hours.
- **Total ≈ 12–40 GPU-hours** if layers run sequentially on quantized inputs.
- Running each layer independently on the teacher's FP inputs uses all 5 large GPUs: ~10 rounds of 49 layers, ~4–8 h wall-clock. An E2E-QP pass then fixes the accumulated drift.
- An AutoRound-sized budget (128 × 2,048 × 200 iterations) is ~3–8 GPU-hours.

**Calibration storage.**
- The inputs and targets of one layer for 8.4 M tokens × 2,048 × BF16 are 34 GB each, 69 GB together. That does not fit next to the working set in 94 GB of RAM.
- Use 2,048 samples (34 GB total) or stream from NVMe.
- Data: Flint's own sampled responses (Recursive's coreset idea, LLM-QAT), mixed with general text and code. Enforce a minimum token count per expert (MoEQuant). Never use the wikitext test split.

**E2E-QP (scales only).**
- Trainable FP16 d/m: 2 per 32 weights → 2.5 B parameters (~30 GB with Adam).
- Frozen 4-bit codes: ~20 GB packed. Dequantize per layer on the fly.
- Spread over 3 × 3090 + 2 × V100 in a pipeline. At ~1–3k tok/s, 16 M tokens take 1.5–4.5 h. This needs custom code.

**Export.**
- Q4_1 block = FP16 d, FP16 m and 32 × 4-bit codes; value = d·q + m. EfficientQAT's (q − z)·s maps exactly with d = s and m = −z·s.
- Q4_0 is d·(q − 8); train it with z fixed at 8.
- IQ4_NL: fake-quant to its fixed 16-entry codebook with one FP16 scale per 32 values.
- Groups run along the input dimension, which matches HF's `[out, in]` layout and the fused expert tensors.
- Write the GGUF with gguf-py by packing the blocks directly: ~a day of scripting, verified by KLD against the fake-quant model.

**Where the error is.**
- Q4_1 everywhere → Q8_0 expert down only moved KLD 0.280 → 0.257.
- UD-mirror v1, with attention, shared expert and head mostly Q8_0, reaches 0.129.
- So most of the 4-bit loss lives in the tensors read every token. QAT effort should start there (section 3.8).

**Quality expectation (claim to test).**
- Published QAT recovers 54–70 % of PTQ's perplexity gap (Gemma, torchao).
- Applied to KLD 0.28, that would land at ~0.08–0.13. KLD and perplexity gaps do not scale identically, so this is indicative only.
- The makers' own QAT shows that 4-bit everything can reach task parity, but their objective left KLD at 0.316. Use a KL-to-BF16 loss.

#### Gaps
- No public Block-AP or QAT code for fused 3-D MoE experts. EfficientQAT is Llama-centric; AutoRound's support for transformers-v5 fused experts on Flint is unknown.
- Flint's activation range for FP16 on V100s is unmeasured (same gap as experiment 6 in `flint_speedups.md`).
- The IQ4_NL decode rate on 5-SM cards is unmeasured.

#### Experiment ideas
- **Q2 – full block-wise run after Q1 (3.8) succeeds (2–4 days).**
  - Fork EfficientQAT's Block-AP loop. Use transformers' `FlintDecoderLayer` with fake-quant on all `nn.Parameter` weights (Q4_0 for 2-D and gate_up, IQ4_NL codebook for down).
  - Inputs: 2,048 × 2,048 samples of Flint responses mixed with RedPajama and code.
  - Run per-layer jobs on 3 × 3090 + 2 × V100 (FP32).
  - Optional: E2E-QP on scales, 16 M tokens.
  - Export the GGUF.
  - Measure KLD with `quality_kld.sh` against Q8_0, the task check, and rig tg128 on stock and rescale-removed builds.
- **Q3 – AutoRound shortcut (half a day).**
  - `auto-round --model <flint BF16 dir> --scheme W4A16 --group_size 32 --format gguf:q4_0` (or `gguf:q4_1`). Check that experts are covered.
  - Compare KLD with imatrix Q4_0/Q4_1.
  - This is the cheapest "trained rounding" baseline before writing Block-AP code.

**Carries over to Qwen3.6?** Yes. Its experts have the same shapes (256 × 512 × 2,048), and transformers support with GDN kernels exists. The rig already runs Q4_0 for Qwen3.6, so QAT improves Q4_0 quality, not speed.

---

### 3.8 Partial QAT: attention, shared expert, dense FFN and head → Q4_0; experts unchanged

#### Takeaway
- These are the cheapest trained bytes and the most likely source of the quality gap.
- Retrain only the always-read tensors (~1.5 B parameters) to Q4_0, keeping UD-mirror v1's expert formats.
- Bytes per token fall from ~2.26 to ~1.54 GB. Predicted 27–32 tok/s llama-bench, +10–20 % over UD-mirror v1 (my calc).
- Hours of compute on one 3090. This is the first experiment for the QAT branch.

#### Cited findings
- Our KLD table (section 3.7). The "attention at Q5_0" variant is already a planned next step in `quants/README.md`.
- EfficientQAT memory scaling (section 3.7).
- Recursive quantized attention and head to 4 bits and recovered them by distillation (blog, S).

#### Inferences
**Bytes (my calc):**

| Part | Format | Bytes per token |
|---|---|---|
| Attention (925 M) | Q4_0 | 520 MB |
| Shared expert (296 M) | Q4_0 | 167 MB |
| Dense FFN (75 M) | Q4_0 | 42 MB |
| Head (308 M) | Q4_0 | 173 MB |
| Experts | as UD-mirror | 585 MB |
| Router | BF16 | 49 MB |
| **Total** | | **1.54 GB** |

- Time: 1.49 GB at 64–73 GB/s = 20.4–23.2 ms + router + fixed 10–13 ms (rescale kept, since the expert down-projections are Q4_1/Q5_0) = 31–37 ms → **27–32 tok/s**.

**Training (my calc).**
- Per layer only ~25 M trainable parameters: 0.5 GB of state. Experts stay frozen at their PTQ quant.
- 49 layers × (4–8 M tokens forward through the full layer + backward through attention and shared expert) ≈ 2–6 GPU-hours on a 3090.
- Head-only QAT is separate and trivial: freeze the body; train the head's Q4_0 codes and scales against BF16 logits from stored final hidden states. 2,048 floats per token; 50 M tokens ≈ 0.4 TB as BF16, or stream it. 1–2 h.

**Quality risk.** If 4-bit attention cannot get within noise even with QAT, fall back to Q5_0 attention with QAT: 0.6875 B per parameter, about +0.12 GB per token.

#### Gaps
- No per-tensor-class KLD breakdown for Flint. Quantizing only attention to Q4_0 with imatrix and measuring KLD would locate the loss before any training.

#### Experiment ideas
- **Q0 – locate the loss, no training (2 h on the PC).** Three files from Unsloth's Q8_0, each changing one class: attention → Q4_0 imatrix; shared expert → Q4_0; head → Q4_0. Run `quality_kld.sh` on each to see which class carries the 0.129 → 0.28 gap.
- **Q1 – partial Block-AP (1 day).**
  - Train attention + shared (+ dense FFN) per layer with experts frozen in their UD-mirror formats, against BF16 block outputs.
  - Then head-only QAT.
  - Export: UD-mirror recipe with the trained Q4_0 tensors swapped in.
  - Measure KLD (target ≤ 0.14, top-1 ≥ 87 %) and rig speed (predicted +10–20 % over UD-mirror v1).

**Carries over to Qwen3.6?** Yes. Its always-on share is even larger (1.94 B of 2.95 B per token, including a 0.51 B head), so per-token bytes and quality both move more.

---

### 3.9 Re-tuning the makers' INT4: KD on scales, and a 64-value-group type

#### Takeaway
- The makers' Q4_1 already carries QAT-trained codes. Two cheap ideas start from it.
- **(a) Scale-only KD toward BF16 logits.** Their group-64 scales became pairs of identical group-32 scales, so each half can now be tuned separately. This might lower KLD at zero byte cost, but could undo their task tuning.
- **(b) A "Q4_1 with 64-value groups" GGUF type (class B).** It stores their codes exactly at 4.5 instead of 5.0 bits per weight: 1.66 → ~1.51 GB, +6–8 % (my calc).

#### Cited findings
- Group-64 affine MLX export re-expressed as group-32 Q4_1 (brief; flint-int4-gguf card, P).
- The makers' file differs from BF16 deliberately and matches BF16 on 300 BFCL tasks (terelabs comparison, S, via `flint_speedups.md`).
- E2E-QP trains only step sizes (EfficientQAT, P).

#### Inferences
**(a) Cost (my calc).** 2.5 B trainable FP16 scale/offset parameters (two per 32 weights) plus frozen codes. Same multi-GPU pipeline as E2E-QP in 3.7: ~1–2 days including code.

**(b) Bytes (my calc).**
- Q4_1 at 64-value groups: 4 bits + 2 × 16 / 64 = 4.5 bpw = 0.5625 B per parameter, same as Q4_0.
- Applied to all 4-bit tensors: 1.57 GB of Q4_1 → 1.41 GB, plus router 99 MB F32 (or 49 MB BF16) → ~1.46–1.51 GB.
- The MMVQ kernel is a copy of Q4_1's with a different block size. The Q8_1 activation blocks are still 32 values, so a 64-value weight block spans two of them.

#### Gaps
- Whether the makers' codes were trained per 64-group with scales that are bit-identical across the two 32-halves (expected from the re-expression; verify by reading the GGUF).

#### Experiment ideas
- Read `flint-int4-q4_1.gguf` with gguf-py and confirm that d/m repeat in pairs of blocks. If so, (b) is exact.
- Implement the type and kernel only if section 3.7's QAT route fails. Otherwise QAT plus Q4_0 reaches the same bytes with better KLD.

**Carries over to Qwen3.6?** No. There are no QAT'd Qwen3.6 codes to re-express.

---

### 3.10 Layer dropping with healing

#### Takeaway
- Removing 8 of 49 layers saves 14 % of bytes and 17 % of launches: +17–19 % (my calc).
- Published layer-pruning evidence says modern (Qwen-family) models collapse after ~20 % removal even with healing, and next-token prediction degrades before QA scores do.
- The KLD bar would be missed by far. Not recommended.

#### Cited findings
- Gromov et al., [arXiv 2403.17887](https://arxiv.org/abs/2403.17887) (2024-03, P).
  - With QLoRA healing, MMLU holds until ~45–55 % of layers are removed for Llama-2, ~35 % for Mistral-7B, ~25 % for Phi-2 and **~20 % for Qwen**.
  - Healing budget: 5,000 steps × batch 16 (164–328 M tokens) on one A100, LoRA on FFNs only.
- Minitron ([arXiv 2408.11796](https://arxiv.org/abs/2408.11796), 2024-08, P): width pruning beat depth pruning after ~94 B tokens of distillation.

#### Inferences
**Speed (my calc).** A MoE layer reads 29.6 MB in the Q4_1 file and launches ~40 kernels. 8 layers ≈ 237 MB (3.5 ms) + 320 launches (1.6–2.2 ms) = −5.1–5.7 ms of 35.8 → **+17–19 %**.

**Healing on our machines (my calc).** LoRA through transformers' Flint in BF16 across all GPUs, at ~1–2k tok/s for training, takes 1–3 days for 164–328 M tokens.

**Quality.** Flint is a 2.5 B-active model trained with RL. Expect KLD well above the makers' Q4_1 (0.316) after removing 16 %.

#### Gaps
- No Flint layer-similarity data.

#### Experiment ideas
- **P1 – layer-importance scan (2 h on a 3090, no training).** With transformers, hook-skip one layer at a time and measure KLD on 20 × 512 wikitext tokens against the full model. Proceed only if several layers each cost < 0.02 KLD.

**Carries over to Qwen3.6?** The same scan applies. Its 30 GDN + 10 attention layers are fewer and less redundant.

---

### 3.11 Attention head pruning, fewer query heads, low-rank q/o

#### Takeaway
- q and o projections are a third of Flint's active parameters.
- Halving the query heads (32 → 16, keeping 4 KV heads) or factoring q/o to rank ~512–683 saves ~411–514 M parameters per token: +9–12 % at Q4_1 (my calc). Low-rank gains less because it adds 2 launches per layer.
- Recovering quality needs distillation budgets of tens of billions of tokens. Not feasible on our hardware; not recommended.

#### Cited findings
- Minitron's width pruning (heads, MLP and embedding channels) needed ~94 B tokens with a corrected teacher ([arXiv 2408.11796](https://arxiv.org/abs/2408.11796), P).
- No post-hoc query-head-reduction result for a sliding-window/sink model was found.

#### Inferences
**My calc:**
- 32 → 16 heads: q and o shrink to 2048 × 2048 → −4.19 M × 2 × 49 = −411 M parameters → −257 MB at Q4_1 → −3.8 ms → **+11–12 %**.
- With Q8_0 attention (UD-mirror): −436 MB → +15–17 %.
- Rank 683: same parameter saving, but +98 launches (+0.6 ms) → **+9–10 %**.
- Rank 512: −514 M parameters → −321 MB → +12 %.
- KV cache is unchanged (still 4 KV heads), so there is no long-context gain.

#### Gaps
- No published post-hoc result for this kind of model.

#### Experiment ideas
None recommended. If wanted: rank-sweep the SVD energy of `q_proj`/`o_proj` per layer (minutes on the PC). If 683 singular values hold < 95 % of the energy, drop the idea.

**Carries over to Qwen3.6?** Less to gain: only 10 attention layers; the GDN projections dominate.

---

### 3.12 Expert pruning (REAP and similar)

#### Takeaway
- Pruning 25–50 % of experts shrinks the file. It does not change bytes per token: still 6 experts per token, and card count does not change speed on this rig.
- So: no decode speed-up.
- Use it only to fit Flint on fewer cards or to free VRAM for KV and slots. It has a measurable quality cost.

#### Cited findings
- REAP ([arXiv 2510.13999](https://arxiv.org/abs/2510.13999), ICLR 2026, P; [repo](https://github.com/CerebrasResearch/reap); [Cerebras blog](https://www.cerebras.ai/blog/reap), S).
  - Router-weighted expert-activation pruning, one-shot, no healing.
  - 50 % pruning of Qwen3-Coder-480B kept 97.6 % of non-agentic coding and 96.7 % of SWE-Bench.
  - [cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) (20 % pruned) shows mixed ±2-point changes.
- Expert merging and pruning are covered in `models_large_moe_and_compression.md`.

#### Inferences
- **Speed: 0 %** (my calc).
- **Memory:** 25 % pruning turns a 26 GB file into ~20 GB, fitting 4 cards with more KV room. 5 vs 9 cards measured the same speed.

#### Gaps
- REAP has no Flint adapter. A calibration pass needs transformers' Flint (feasible).

#### Experiment ideas
None for speed.

**Carries over to Qwen3.6?** Same: no speed gain.

---

### 3.13 Output head: vocabulary trimming or a factorized head

#### Takeaway
The head is 12 % of each token's bytes in the makers' file (192 MB) and 14 % in UD-mirror v1 (327 MB at Q8_0).

| Option | Saving | Speed (my calc) | Requirements |
|---|---|---|---|
| Trim to the top 64k / 32k tokens | 110–151 MB | +4.5–6.5 % | llama.cpp scatter into −inf (as in eagle3.cpp) for the *target*; trimmed tokens can never be generated |
| Rank-512 factorized head trained by KD | ~143 MB | +6 % | KD on stored final hidden states; one extra launch |
| QAT of the head to Q4_0 (section 3.8) | 154 MB on UD-mirror | +4–6 % | Simplest; no engine change |

#### Cited findings
- `src/models/eagle3.cpp` shows the d2t scatter pattern. Flint's head is `build_lora_mm(model.output, …)` with no such hook (`src/models/flint.cpp`, local tree, P).
- Recursive put the head at 4 bits and recovered it (blog, S).

#### Inferences
**My calc:**
- Trim to 64k rows at Q4_1: 192 → 82 MB, −1.6 ms → +4.5 %.
- Rank-512 factorization: 2048 × 512 + 512 × 150,144 = 78 M parameters (49 MB at Q4_1) → −2.1 ms + 1 launch → +6 %.
- Training the factorized head: final hidden states (2,048 BF16 per token) with BF16-head teacher logits computed on the fly. 50–100 M tokens on one 3090 in a few hours.

#### Gaps
- Coverage of a trimmed vocabulary on the owner's real workload. CJK, rare symbols and tool tokens must stay in.

#### Experiment ideas
Do head QAT first (3.8). Trim only if every other lever is exhausted.

**Carries over to Qwen3.6?** Yes, with a bigger effect: the 248k head is 0.51 B parameters, ~286 MB at Q4_0 per token.

---

### 3.14 Activation sparsity (TEAL, CATS, ReLU-fication)

#### Takeaway
- Skipping weight columns for zero activations could, in principle, save up to ~40 % of the dense and expert bytes.
- **The rig cannot exploit it.** GGUF blocks run along the input dimension, so a block can be skipped only if all 32 consecutive inputs are zero (≈ never at 40–50 % sparsity).
- It needs a column-blocked quant format and a gather matvec kernel (class B). The quality-preserving high-sparsity variants (ReLU-fication) need ~100 B+ tokens of continued training. Not recommended.

#### Cited findings
- TEAL ([arXiv 2408.14690](https://arxiv.org/abs/2408.14690), ICLR 2025, P): training-free magnitude sparsity, 40–50 % model-wide; 1.53× / 1.8× decode at 40 / 50 % on A100/A6000 with GPT-Fast and custom kernels. Its repo supports FP16 Llama only.
- TurboSparse ([arXiv 2406.05955](https://arxiv.org/abs/2406.05955), P): dReLU sparsification of Mistral/Mixtral with long continued training; 2–5× decode.
- PowerInfer ([arXiv 2312.12456](https://arxiv.org/abs/2312.12456), P) is a llama.cpp fork with neuron-level sparse operators for ReLU models. No primary confirmation of Q4_0 sparse kernels was found.
- `dense_27b_speedups.md` §7 states the GGUF block-layout obstacle.

#### Inferences
**Ceiling (my calc).** 40 % of the attention + shared + expert bytes (1.32 GB in Q4_1) = −0.53 GB → −7.7 ms if a sparse kernel ran at MMVQ efficiency (+27 %).
- On 5 SMs a gather kernel is likely less efficient.
- Threshold or predictor kernels add ~150 launches.
- Realistic: **0–15 %**, with an unmeasured quality cost on Flint.

#### Gaps
- No Flint activation-sparsity profile. No column-blocked quantized format in any llama.cpp-family engine.

#### Experiment ideas
None now.

**Carries over to Qwen3.6?** Same obstacles.

---

### 3.15 Long context: full-attention → sliding-window conversion, or cross-layer KV sharing

#### Takeaway
- At 32k context, Flint's 14 full-attention layers read ~0.94 GB of KV per token, more than half the weight bytes.
- Converting 7 of them to the existing 513-token sliding window, or sharing KV between pairs, would cut that in half: ~+19 % at 32k, ~+5 % at 8k (my calc).
- It needs healing on long sequences and risks long-context retrieval. Worth considering only if agent sessions routinely run at 30k+ tokens.

#### Cited findings
- `flint_speedups.md`: 28 KiB of KV per token on the 14 full layers; 38.9 → 51.3 ms from 8k to 32k context.
- Cross-layer sharing after training:
  - PoD ([arXiv 2412.02252](https://arxiv.org/abs/2412.02252), P): ~35 % KV saving with continued training on LLaMA3-8B-32K.
  - xKV ([arXiv 2503.18893](https://arxiv.org/abs/2503.18893), ICML 2026, P): up to 8× compression via a shared low-rank subspace.
- A Cerebras blog (S) fine-tunes Llama-3-8B with most layers switched to sparse or sliding attention with sinks; KV memory nearly halved.
- "Sliding-window beats linear attention" ([arXiv 2608.28444](https://arxiv.org/abs/2608.28444), 2026-08, P): sliding windows with sinks recover much of full attention on knowledge tasks without training.

#### Inferences
**My calc.**
- At 32k: −0.47 GB per token at ~57 GB/s → −8.2 ms → 51.3 → ~43 ms (**+19 %**).
- At 8k: −0.115 GB → −2 ms (+5 %).
- Flint already uses SWA plus sinks, so converted layers need no new kernels: flip `layer_types` and the GGUF's `sliding_window_pattern`.
- Healing needs 16–64k-token sequences. With transformers' Flint in BF16 across all GPUs, ~1–2k tok/s, 100–300 M tokens take 1–3 days.

#### Gaps
- Which full-attention layers are least needed for long-range retrieval. Measure with a needle test after flipping one layer at a time, training-free first.

#### Experiment ideas
- **L1 (training-free, half a day).** Edit the GGUF's `flint.attention.sliding_window_pattern` to make one full-attention layer sliding at a time (this needs a rope_pattern check: full layers are NoPE). Run a 32k needle test plus KLD at 8k context on the PC. If 3–7 layers flip with little loss, consider healing.

**Carries over to Qwen3.6?** Less to gain: only 10 full-attention layers, and the GDN state is fixed-size.

---

### 3.16 Compute feasibility on the quant machine

#### Takeaway
Everything above can run on the quant machine:
- **Training:** transformers' native Flint (BF16 on 3090s, FP32 on V100s).
- **Features and generation:** llama.cpp, which runs any GGUF across mixed GPUs and can export any layer's input.
- **Not usable:**
  - vLLM and SGLang: Flint kernels target Hopper/Blackwell; PRs open.
  - llama.cpp's own finetune: FP32-only, no flash-attention backward.
  - Unsloth: no Flint.
- Disk is the binding constraint for offline features (16 KB per token), so prefer online feature generation.

#### Cited findings
- transformers PR #49824 (2026-10-21) and `modeling_flint.py` / `integrations/moe.py` details (section 3.7, P).
- The Flint HF repo has no modeling code (native class only) and an `sglang/` kit ([repo files](https://huggingface.co/RecursiveSI/flint/tree/main), P).
- `src/llama-ext.h` exposes `llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`. EAGLE-3 and DFlash use them (local tree, P).
- `examples/training/README.md` (local tree, P): finetuning is FP32-only and work in progress; flash attention is disabled in training because it has no backward pass.
- `examples/eval-callback` can print any tensor.
- vLLM [PR #62938](https://github.com/vllm-project/vllm/pull/62938) and SGLang [PR #45624](https://github.com/sgl-project/sglang/pull/45624) target Blackwell/Hopper (`flint_speedups.md` §7).

#### Inferences
**Throughput (claims to test).**

| Job | Estimate |
|---|---|
| llama.cpp Flint Q8_0 prefill on a 3090 (+ V100 or 3080 for the remainder) | 2–5k tok/s |
| Batched generation per server copy, `-np 32` | 0.5–1k tok/s |
| HF BF16 forward with grouped GEMM across 3 × 3090 + 1 V100 | 3–6k tok/s |
| LoRA or partial training through the whole model | 1–2k tok/s |

**Numerics.**
- Use llama.cpp at Q8_0 or the deployment quant for teacher features: activations go through Q8_1 blocks with FP16 scales but FP32 accumulation, and Q8_0 weights ignore the risky sums.
- Avoid FP16 compute on V100s for training.
- Keep the router and sinks in FP32, as transformers does.

**Storage (my calc).**

| Data | Size |
|---|---|
| EAGLE-3 features (3 aux + final, BF16) | 16 KB per token; 30 M tokens = 0.5 TB |
| Block-AP inputs and targets | 34 GB per layer at 2,048 × 2,048 |
| Head-KD final hidden states | 4 KB per token |

**The workstation is not the place:** its disk was 95 % full on 2026-10-22 (private CLAUDE.md).

#### Gaps
- None of the throughput numbers is measured.
- Free disk on the quant machine is not recorded here.

#### Experiment ideas
- **F1 (1 h).** On the quant machine:
  - `llama-bench -m flint-Q8_0.gguf -ngl 99 -fa on -p 512,2048 -n 0` across a 3090 + V100 split.
  - `llama-batched-bench` for generation at `-npl 1,8,32`.
  - A transformers forward of 8 × 2,048 tokens with `experts_implementation="grouped_mm"`, `attn_implementation="eager"` and `"flex_attention"`.
  - These size every plan above.

---

## 4. Top candidates for a deep dive (ranked)

1. **EAGLE-3 head with a 32k draft vocabulary for Flint.**
   - The largest lossless speed-up available (1.3–2.0×, my calc). It uses existing llama.cpp code paths (`draft-eagle3`, `d2t`, `--target-model-dir`) that this note checked in the rig's own source tree.
   - Gate it with C1 (verify cost, synthetic acceptance) and C2 (random-weight head timing on the rig). That is a day of work and settles the main risk: MoE verify cost, as with gpt-oss on the Spark.
   - Then train a pilot (C3) on ~20k Flint responses.
   - Publishing the first Flint drafter also serves the public repo's purpose.
2. **Partial QAT of attention, shared expert, dense FFN and head to Q4_0, on top of UD-mirror v1 (Q0 → Q1).**
   - Hours of compute, aimed exactly at the tensors that carry both the bytes per token and, by our KLD data, the quantization loss.
   - Predicted +10–20 % over UD-mirror v1 at its quality.
   - Q0 (no training) first shows which tensor class carries the gap.
3. **Full block-wise QAT to Q4_0 with IQ4_NL expert down-projections (Q2), plus the rescale patch.**
   - If (2) shows 4-bit always-read tensors can be recovered, this extends it to the experts and removes the rescale exactly.
   - Predicted 30–37 tok/s llama-bench at hopefully UD-mirror quality; 2–4 days.
   - AutoRound (Q3) is the cheap baseline to beat.
4. **DFlash-class drafter for Flint.**
   - Higher ceiling than EAGLE-3 (1.6–2.6×) but 3–5× the training and an untested verify-width trade-off on 5 SMs.
   - Test the concept first, free, with z-lab's Qwen3.6 DFlash drafter on the rig (D1).
5. **Qwen3.6 MTP draft-vocabulary trim (PR #29143).**
   - Training-free carry-over: +9–13 % on Qwen3.6's best configuration (my calc), half a day.
   - It also validates the head-read cost model used for the Flint drafter.

**Not recommended for speed on this rig:**

| Lever | Why |
|---|---|
| Small draft model | 1.1–1.35× at higher cost than EAGLE-3 |
| Training-free layer-skip self-speculation | ≈ 1.0× |
| Lookahead / Jacobi | Loss |
| Layer dropping | Quality collapse risk |
| Head pruning / low-rank q,o | Healing budget out of reach |
| Expert pruning | 0 % speed |
| Activation sparsity | No kernel or format; healing out of reach |
| MTP for Flint | Needs per-architecture engine code; EAGLE-3 covers it |
| Medusa | Unsupported in llama.cpp |
