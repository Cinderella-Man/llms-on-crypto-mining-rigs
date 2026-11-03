# What is new for Qwen3.8-Flash-Next on Pascal since 2026-10-27 (2026-11-03)

This note lists what upstream llama.cpp, Hugging Face and the other engines published after 2026-10-27 that could make
Qwen3.8-Flash-Next faster or better on the P106 rig (nine 6 GB cards, compute 6.1, PCIe 1.0 x1, 4 GB host), and ranks
what to do about it. It covers what the notes on the [upstream rebase](upstream_rebase.md),
[Flash-Next versions](flash_next_variants.md), [drafters](flash_next_drafters.md), [Strata](strata.md) and the engine
surveys ([llama.cpp family](engines_llamacpp_family.md), [others](engines_other.md),
[Pascal patches](upstream_pascal_patches.md)) do not. Sources were read on 2026-11-03: GitHub pull-request, issue and
discussion pages, the Hugging Face model listings and cards, and the engines' repositories. Nothing was run, built or
downloaded; numbers marked **est.** are arithmetic from the linked figures. The fork's base is upstream `20a51f24a`
(build 12092, 2026-11-03 16:26 UTC; upstream's b12076 tag is `36ee3caf7`, sixteen commits earlier).

## Summary

1. **Upstream has moved four commits past the fork's base, and none touches CUDA, `qwen4exp` or speculation** (two
   Hexagon ops, a Gemma 4 graph change, a UI test change). The news is in open pull requests and in merged work that the
   base already carries.
2. **Five upstream changes would move the rig's numbers:**
   - [#31156](https://github.com/ggml-org/llama.cpp/pull/31156) (draft): one CUDA graph and scheduler per verify width.
     Est. +2–7 % per draft step.
   - [#31352](https://github.com/ggml-org/llama.cpp/pull/31352) (open): a larger Pascal dp4a tile for 64-token
     batches. +13 % prompt speed for a Q2_0 MoE on a GTX 1080 Ti.
   - [#31525](https://github.com/ggml-org/llama.cpp/pull/31525) (open): vector flash attention pairs query heads that
     share a KV head. −37 % kernel time at 32k on a V100. Flash-Next's ratio is 12.
   - A maintainer's announced replay-buffer rollback for the Gated DeltaNet state (no PR yet): no state plane per
     drafted token.
   - [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) (open): reference-counted server checkpoints. Lower
     peak host memory.
3. **Nothing upstream is built for Pascal.** The new Gated DeltaNet prefill kernel, the sparse-attention gather and the
   few-row MoE kernel all need tensor cores (or AMD matrix cores). Pascal support is intact: a search of the issues found
   none that proposes dropping it, and maintainers are asking for Pascal numbers on the CUDA-graph PRs.
4. **No new GGUF file of at most 54 GB on the cards beats Swift-1.5 GSQ-RCO Q2_0 at equal bytes.** ISTA, ukisai, unsloth,
   bartowski and pfeifferj published no new Flash-Next quant since 2026-10-19. The week's ~100 new repositories are
   re-uploads, uncensored variants and formats Pascal cannot run. The one useful third-party evidence is a per-layer KL
   study that backs the rig's own dense-tensor plan.
5. **Drafters: nothing new loads in mainline llama.cpp.** The new Swift-1.5 MTP drafters need other forks. A DFlash2
   drafter retrained on its quantized target gained only +5 % in accepted length. DSpark now has acceptance numbers on
   agent sessions.
6. **No report of Flash-Next on a P106, P102 or any mining card exists.** The closest are a six-card mixed rig with three
   Tesla P100s (Strata), a GTX 1070 Ti and four Tesla P40s. Strata measured MTP 6–8 % faster than DFlash2 on the P100
   rig, so the rig's DFlash2 line keeps Q226's MTP line as its control.
7. **Other engines: nothing transfers wholesale.** ik_llama.cpp merged DFlash2 for `qwen4exp` and Gyro trellis CUDA
   kernels, with no Pascal figures; vLLM, SGLang and ExLlamaV3 do not run on compute 6.1.

## 1. Upstream llama.cpp

### 1.1 Merged work the fork's base carries, with what it changes here

| PR | State | What it does | Effect here |
|---|---|---|---|
| [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) | merged 2026-10-14 | Gated DeltaNet chunked prefill kernel (tensor-core MMA, `gated-delta-net-mma.cu`) | Enabled on sm_86 and sm_89 only, and skipped when a context verifies more than one token, so also on a target with a draft. Pascal runs the per-token kernel in all 36 DeltaNet layers; the registry's sm_61 variant (Q462) is the only route |
| [#28770](https://github.com/ggml-org/llama.cpp/pull/28770), [#31259](https://github.com/ggml-org/llama.cpp/pull/31259), [#31243](https://github.com/ggml-org/llama.cpp/pull/31243) | merged 09-20, 10-29, 10-27 | Sparse flash-attention gather for `qwen4exp` (MMA kernel, engages from ~32k tokens), then CDNA, then a selection built independently of the mask shape | The gather is gated to Turing tensor cores or AMD matrix cores, so Pascal keeps reading the whole cache. For scale, the author's DGX Spark decode gains +3 % at 10k and +18 % at 100k with the gather, so Q230's estimate (−34 ms at 32k, −145 ms at 128k) is an upper bound |
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) | merged 10-07 | `--moe-cache-mib`: a GPU cache for experts kept in host memory (batches up to 32 tokens) | Single device only: a multi-device run stops at start-up. Not usable on nine cards, and every expert is on a card anyway |
| [#31391](https://github.com/ggml-org/llama.cpp/pull/31391) | merged 11-03 | CUDA kernel that gathers fragmented state reads straight into mapped host memory | Meant for server checkpoint saves: 254 → 21 ms (Llama 3.1 8B, RTX 5090), 202 → 97 ms (Qwen3.8-27B). It pins the destination for each save (`GGML_CUDA_NO_PINNED=1` restores the old path). The rig's 113 MiB save takes 0.67 s at ~170 MiB/s, which is the x1 link, so est. no gain; the transient pin on a 4 GB host is the risk |
| [#31506](https://github.com/ggml-org/llama.cpp/pull/31506) | merged 11-03 | DFlash hidden features copied between contexts on the device (`llama_batch_ext_add_embd_row`) | DFlash prefill +22 % (RTX PRO 6000) to +98 % (RTX 4090), decode −0.4 % to +13 %. Card-to-card copies without peer access still stage through the host, so est. less than that here. The rig's DFlash2 measurements were taken on the older base |
| [#30878](https://github.com/ggml-org/llama.cpp/pull/30878), [#31443](https://github.com/ggml-org/llama.cpp/pull/31443) | merged 11-03 | Conv-state rollback windows written with one `set_rows`; `qwen4exp` writes the conv tail without an intermediate copy (72 → 36 device copies per token) | The authors claim 1–2 % generation; the next baseline includes it. The registry's fusions (Q437, Q438) go further |

Checked and not relevant: #30584 and #30704 make CUDA reject `MUL_MAT_ID` carrying FP32 source precision, which sends the
expert down projections to the CPU, but only Mellum and Mistral 4 set it; the fork's graph code has no such flag for
`qwen4exp`.

### 1.2 Open pull requests the base lacks

| PR | State | What it does | Numbers | What it would change here |
|---|---|---|---|---|
| [#31156](https://github.com/ggml-org/llama.cpp/pull/31156) | draft, 10-25 | A graph result per token count (≤ 8) and, with `LLAMA_SCHED_PER_SIZE=1`, a scheduler per size, so alternating verify widths stop rebuilding, re-splitting and re-capturing | GB300 DFlash2-7 / MTP-7 +7.2 % / +5.9 %; ~+400 MiB VRAM at `--spec-draft-n-max 5`. The narrower [#31004](https://github.com/ggml-org/llama.cpp/pull/31004) gave +3.7 to +5.2 % | Successor of the PR behind Q52. Verify widths 1–4 alternate on the rig, and a shape change costs est. 10–20 ms (Q52). A `--spec-draft-p-min` cut varies the widths further: with graphs off, the gain of [#31294](https://github.com/ggml-org/llama.cpp/pull/31294)'s first revision, which skipped the cut for sampled drafts, fell from +52 % to +21 % at temperature 1.0 |
| [#31274](https://github.com/ggml-org/llama.cpp/pull/31274) | open, 11-02 | Scheduler sizing and graph metadata allocated lazily (calloc) | The calloc commit alone: 50 MiB (DFlash2-7) and 169 MiB (MTP-7) of anonymous memory; with #31156, 0.5–2.9 GiB (GB300) | Est. −50 to −170 MiB on the 4 GB host, whose margin is ~100–200 MiB (Q609) |
| [#31352](https://github.com/ggml-org/llama.cpp/pull/31352) | open, 10-29 | Dp4a MMQ table: tile height 64 → 128 for the two J=64 rows of 11 types (Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS, IQ4_XS, IQ4_NL) in `mmq-config-pascal-dp4a.cuh` | GTX 1080 Ti pp512 +8.2 to +18.0 % per type, Qwen3.6-35B-A3B Q2_0 +13.0 %; tg and pp ≤ 48 flat, pp64 −3.1 %; K-quants regress 19–37 %, so they stay | The fork's table has these rows. Q108's tile choice sends few-token experts to small tiles, so the gain lands on the dense products and on long micro-batches: est. +3–8 % prompt speed on the 5-SM cards (Q285) |
| [#31525](https://github.com/ggml-org/llama.cpp/pull/31525) | open, 11-02 | Vector flash attention takes two query heads that share a KV head per block, loading V rows once | V100, D=256, Q4_0 K/V, 24 heads over 4: 453 → 285 µs at 32k, 3.85 → 2.46 ms at 262k. Gated to sm_70, Q4_0 and D=256 | Flash-Next is D=256 with 24 heads over 2 (ratio 12). If the profile shows the vector kernel at decode, un-gating it for sm_61 with F16 and q8_0 K/V is a long-context lever without the indexer (est. 35–50 % of Q230's figures) |
| [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) | open, 10-08 | Server checkpoints held as shared pointers instead of deep copies | Peak RSS 1,846 → 1,621 MiB (2 checkpoints), 4,210 → 3,198 MiB (8–9); steady RSS unchanged; saves only checkpoints that share a prefix. [#30683](https://github.com/ggml-org/llama.cpp/pull/30683) (open) also skips re-copying an existing checkpoint and fixes a `--cache-reuse` restore; the two conflict | Agent histories grow, so checkpoints share prefixes. The author's saving is about one checkpoint per checkpoint held: est. −225 to −450 MiB of peak (not steady) host memory with `--ctx-checkpoints 2–4` of 113 MiB (Q549) |
| announced in a comment on [#31584](https://github.com/ggml-org/llama.cpp/pull/31584) | no PR yet, 11-03 | Replay buffer for the Gated DeltaNet rollback: keep each sequence's few inputs and replay up to the accepted token, instead of a state per draft position; undoing steps fails numerically (> 30 % error) | Author's est.: ~3 GB less VRAM at `-np 4`; faster on low-bandwidth cards because the large state is not written. #31584 (open, one approval) fixes view bounds for the MTP rollback windows it builds on | The 113 MiB-per-drafted-token planes go away (−340 MiB at 3 drafted tokens, −790 MiB at 7). Writes cost est. ≤ 3 ms of a 200–230 ms verify step, so VRAM is the gain |
| [#31610](https://github.com/ggml-org/llama.cpp/pull/31610), [#31569](https://github.com/ggml-org/llama.cpp/pull/31569) | open, 11-03 | `--fit` leaves out drafts that need the target's context (DFlash, EAGLE-3, Gemma MTP) because their memory measurement fails; skipped NextN layers shift the layer count by one | 0.43 GiB of draft weights unaccounted in the reported case | The fit probe with a DFlash2 draft under-counts it. Measure real loads (as Q561 did) or take the PR |
| [#30657](https://github.com/ggml-org/llama.cpp/pull/30657) | open, 10-16 | Loader and graph for DSpark GGUFs that carry log-SNR conditioning tensors | Without it such a file fails the tensor count and speculation turns off silently | Needed only if the DSpark conversion carries them (Q568) |
| [#30592](https://github.com/ggml-org/llama.cpp/pull/30592) | open draft, 10-15 | `qwen4exp` loader fixes for MTP files: skipped scale tensors, head-mixer fallback, no pooled-key inputs on non-pooling MTP layers | Without it `draft-mtp` aborts at start-up for some files; one 8-card run loaded, a 6-card layout aborted | Check the Gyro and Unsloth heads load without it on nine cards |
| [#31533](https://github.com/ggml-org/llama.cpp/pull/31533) | open, 11-02 | `mul_mat_q_few`: each weight read once for 2–24 columns, int8 tensor-core MMA | Verify of 7 drafted tokens +11 % to +34 % per step (3090 Ti 1.34×) on Qwen3.8-27B | No dp4a path, and only K that is a multiple of 256 and ≥ 2,048, so the 640-wide expert-down and 320-wide hyper-connection rows fall outside. The idea is Q591's k-column kernels |
| [#31047](https://github.com/ggml-org/llama.cpp/pull/31047) | open, rebased 11-03 | Linearized MoE layout for MMQ/MMF | P40 pp8192 up to 1.40× at `-ub 64` | The fork carries it (Q108); a merge drops the carry |
| [#27721](https://github.com/ggml-org/llama.cpp/pull/27721), [#30727](https://github.com/ggml-org/llama.cpp/pull/30727) | open, no review | CUDA graphs from compute 6.1; graphs on all older GPUs | P102-100 tg128 70.8 → 88.4 t/s; four P100s with tensor split 39.6 → 70.6 t/s; a maintainer asks for Pascal verification | The fork already carries the gate; P106 numbers would unblock upstreaming |

Not applicable: [#31469](https://github.com/ggml-org/llama.cpp/pull/31469) (out-of-core expert streaming; its first table claims 16–22 tok/s,
later commits 2–7) and [#30966](https://github.com/ggml-org/llama.cpp/pull/30966) (missed experts on the CPU) serve models whose experts do not fit on the cards;
[#31419](https://github.com/ggml-org/llama.cpp/pull/31419) (skip uploads of unread inputs) gains ~1 % on one card and is untested on several;
[#31234](https://github.com/ggml-org/llama.cpp/pull/31234) (quantized K/V read directly in the tile kernel; 1.6–1.85× at 131k on an R9700) is not measured on Pascal.

### 1.3 Open issues that bear on the rig

| Issue | State | Finding | Effect here |
|---|---|---|---|
| [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) | open, 37 comments | With greedy sampling, MTP, DFlash and DSpark drafts change the output on quantized targets: the verify batch runs other kernels than one-token decode (MMVQ warp counts, flash-attention precision). Reports include an IQ2_XXS target with six drafted tokens that diverged on 5 of 6 prompts; a patch covers only batches up to 4 | The fork's one-token and k-column kernels sum in different orders (Q632). Run an identity check between 1- and k-token logits on the 2-bit file before reading acceptance as quality |
| [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) | open | On hybrid models the server's checkpoints, made only at message starts and near the prompt's end, are often all invalidated, so the prompt re-reads | Q548's spread option covers it |
| [#27980](https://github.com/ggml-org/llama.cpp/issues/27980) | closed stale | Four Tesla P40s: `qwen4exp` decodes 18.0 tok/s against 29.5 for a same-class `qwen3.5moe`, with the matrix path and the host table ruled out; the recurrent step was suspected and not profiled | Consistent with the weightless operations being a large share of a token (Q437, Q438) |
| [#30524](https://github.com/ggml-org/llama.cpp/issues/30524) | open | `MUL_MAT_ID` MMVQ launch limit uses the older PTX arch when a 61-virtual build runs on a newer GPU; one-line fix proposed | The fork fixed the same bug in the rebase |

## 2. New GGUF files and drafters

### 2.1 Files

The Hub's newest 100 repositories matching "Flash-Next" (2026-10-28 to 2026-11-03) include about 28 GGUF repos. Most are
uncensored or abliterated re-uploads of the GSQ-RCO files, mirrors, or EXL3, MLX, NVFP4 and W4A16 builds. None of these
have KLD or task numbers against the base model.

| Repository | On the cards | What it is | Evidence | Verdict |
|---|---|---|---|---|
| ISTA, ukisai, unsloth, bartowski, pfeifferj | unchanged | Last Flash-Next change: ISTA 09-29 and Coder 10-19, ukisai 10-07, unsloth 10-06, bartowski 10-13, pfeifferj 09-23 | ISTA's newest work is DeepSeek-V4.1 (11-02). New ISTA threads: Q2_0 emits Chinese tokens now and then in English output (RTX PRO 6000; not reproducible; ISTA suspects the base model), a LiveCodeBench comparison, requests for an IQ3_M | Swift-1.5 Q2_0 stays the first target |
| [pentacoxian-dev IQ3_XXS_M64](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS_M64-GGUF) and `_HC8` | 42.9 GiB of weights (ISTA's IQ3_XXS: 43.8) | ISTA's IQ3_XXS with the routed down experts of layers 29, 30, 31 and 35 moved from IQ4_NL to Q2_0 (taken from ISTA's Q2_0) and, in the `_HC8` file, the hyper-connection mixers from BF16 to Q8_0 (a further 0.55 GiB smaller) | KL against ISTA's IQ3_S (six 2,048-token texts, top-128 lower bound): IQ3_XXS 0.1834, M64 0.1835, M64 with Q8_0 mixers 0.1795; ISTA Q2_0 0.4380. Layer 47 alone adds 0.07 on Japanese text. Made for a Mac fork (ds4); no llama.cpp statement | Same quality for −0.9 GiB; a published A/B for Q218 and Q219 |
| [sirfyyn FLE](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF) | est. 42.8 (IQ2_M), 50.7 (IQ3_XXS), 60 GB (IQ3_M) | Whole expert layers spliced from Unsloth and ISTA donor files per layer; loads in upstream llama.cpp | KLD against BF16 on three texts: FLE-IQ2_M 0.160 / 0.163 / 0.339 at 71.6 GB in all, ISTA IQ2_XS 0.179 / 0.191 / 0.382 at 68.0, ISTA Q2_0 0.223 / 0.236 / 0.485 at 66.4; the maker says its files lie on ISTA's line | No gain at equal bytes |
| [zephel01 12G32G](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF) | est. 38.8 | 448 of 512 experts, gate/up IQ1_S, down IQ4_NL | KL 0.339 against the NVFP4 model, level with Unsloth's UD-IQ1_S (0.331); 3/5 coding tasks | IQ1 class; no gain |
| [WeeAIResearch 32B-A6B](https://huggingface.co/WeeAIResearch/Qwen3.8-Flash-Next-32B-A6B-Coder-Mixed) | ~15.5 GiB | Experts pruned to 85–177 per layer by C# usage, variable banks | No KLD; smoke tests only; needs a runtime patch on build b10662 | No |
| agentionai Gyro-Coder, [ik_llama.cpp #2812](https://github.com/ikawrakow/ik_llama.cpp/pull/2812) | 19.9 GiB | Trellis quants now load and run on CUDA in ik_llama.cpp (dequant, MMVQ and MMQ kernels), not in mainline | 2×3090: 54–56 tok/s decode on Gyro-M; no KLD in the PR; Pascal not mentioned | Q241's reopen needs a P106 timing (idea 9) |

### 2.2 Drafters

| Drafter | Size | What it is | Evidence | Verdict |
|---|---|---|---|---|
| [hotocooo DFlash2 v2](https://huggingface.co/hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash) | Q8_0 1.1 GB | Inco AI's drafter retrained 10,000 steps on 2.9 M rows captured from an IQ3_S target | Accepted length 2.82 → 2.98 of 7 on the training subset, 3.03 on 8,192 held-out anchors | A retrain on the target's own output gains ~+5 %: Q572 stays parked |
| [shadowpa0327 DSpark](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark) | 0.66B | 5 layers, block 7, trained on 95,596 coding-agent sessions after chat data | Accepted length on held-out agent sessions 3.70 (≤ 32K: 4.10, > 32K: 3.46), MTP with 7 steps 4.24; chat, math, code 4.55 against 4.67; H100 time per token 1.75 against 2.03 ms (0–8K), 3.87 against 3.44 ms (96–128K); benchmarks at step 1,600, weights at 1,800; block 7 needs a patched SGLang | Conversion to GGUF still untested (Q568) |
| [jakeatx Swift-1.5 MTP drafter](https://huggingface.co/jakeatx/Swift-1.5-Qwen3.8-Flash-Next-MTP-Drafter-GGUF) | 1.52 GB | MTP head with experts at 2 bits and shared embeddings and head | No acceptance figures | Loads only in llamAmpere (an sm_86 fork); the size is the point (idea 8) |
| [adriandj3 Swift MTP head](https://huggingface.co/adriandj3/Swift-1.5-Qwen3.8-Flash-Next-MTP-for-GGUF-and-NVFP4) | Q8_0 2.79 GB | A byte copy of Swift's own head, no training | 2.88 (BF16) and 2.63 (FP8) accepted tokens per step at 3 drafted (vLLM); 37–58 % in GGUF smoke tests | Needs a fork commit; no gain over the existing heads |

Searches for a Flash-Next EAGLE-3 drafter found nothing. The DFlash2 GGUFs are the ones the drafters note already lists
(Inco AI, ggml-org, nazeshinjite) plus one more repackaging (skeggsguy).

## 3. Reports from small, old and many-GPU setups

No report of Flash-Next on a P106, P102, CMP card or any 4 GB host exists.

| Setup | Source | Result | What it shows |
|---|---|---|---|
| RTX 2080 Ti, 2 × RTX 3060, 3 × Tesla P100 (PCIe 3.0 x8), 188 GiB RAM; ISTA IQ3_S | [Strata #5125](https://github.com/Niko1221/Strata/issues/5125), 11-03 | Decode 56.2 tok/s with MTP (`--spec 4 --spec-min-p 0.5`) against 52.2 with DFlash2 (`--spec 6 --draft-conf 0.35`); accepted 72.9 % against 61.6 %; prompts 391 / 1,010 / 1,490 tok/s at 4K / 32K / 128K | MTP beat DFlash2 by 6–8 % in every configuration, although Strata 0.2.10 makes DFlash2 its default (1–12 % faster on one card, 2.6 % slower on an Intel Arc A750). A P100 needed an inline-assembly fix in the emulated dp4a (~62,000 wrong sums per 65,536 without it); the fault is in the sm_60 path, and sm_61 has the hardware instruction |
| 2 × Tesla P100, 48 GB RAM, SATA SSD; IQ3_XXS | [Strata #2789](https://github.com/Niko1221/Strata/issues/2789) | 30–60 tok/s decode at ~230K context, prefill 220–401 tok/s | Benchmark record, no bottleneck named |
| GTX 1070 Ti 8 GB, 32 GB RAM; Gyro-S | [Strata #4517](https://github.com/Niko1221/Strata/issues/4517) | Decode 27–34 tok/s, prompts 89–211 tok/s; 12 agent tasks: Coder IQ1_M passed 12, Gyro-S 10 | Below ~512 MiB of free VRAM the driver pages the card and requests stall silently, with no error: keep a reserve |
| 4 × Tesla P40 | llama.cpp [#29174](https://github.com/ggml-org/llama.cpp/issues/29174), [#27980](https://github.com/ggml-org/llama.cpp/issues/27980) | 21.6 → 30.6 tok/s with an MTP draft (acceptance 0.45); `qwen4exp` 18.0 against 29.5 for a `qwen3.5moe` | Decode on Pascal is bound by something other than the matrix path |
| 6 × RTX 3090 on x8, x4 and x2 links, four behind USB4 docks, five 262k slots | [llama.cpp discussion #30071](https://github.com/ggml-org/llama.cpp/discussions/30071) | Decode at 250k: 10.2 → 33.7 tok/s (one session), 2.3 → 27.3 per session (five), with the patches of the [multi-GPU fork](https://github.com/lukolszewski/llama.cpp-multigpu); pipelined prefill 263 → 2,111 tok/s | The closest topology to the rig: a starved link. The fixes are per-sequence decode groups, prefill pipelined across cards, a top-k indexer with a persistent pooled-key cache and a compact q8_0 gather. MTP slowed it down; n-gram lookup helped (~2.5× on refactoring) with at most two active users. Only sm_86 is tested; sm_61 is PTX only |
| P102-100 10 GB | [llama.cpp #27721](https://github.com/ggml-org/llama.cpp/pull/27721) | tg128 70.8 → 88.4 t/s with CUDA graphs | A mining card; the gain is in the fork already |

Expert-cache setups (RTX 3060 12 GB, Radeon VII 16 GB at ~26 tok/s, RTX 5070 Ti at 33–62 tok/s) keep experts in host
memory and do not transfer to a rig whose experts all sit on cards.

## 4. Other engines

| Engine | Flash-Next | Compute 6.1 | What transfers |
|---|---|---|---|
| ik_llama.cpp | DFlash2 for `qwen4exp` ([#2761](https://github.com/ikawrakow/ik_llama.cpp/pull/2761), 10-27): at `n_max` 3, code 35.5 → 49.4 tok/s (88.6 % accepted), extraction 35.4 → 45.3 (73.1 %), story 35.4 → 33.0 (39.5 %), two GPUs with CPU experts. Fused indexer scoring and radix top-k ([#2784](https://github.com/ikawrakow/ik_llama.cpp/pull/2784), [#2767](https://github.com/ikawrakow/ik_llama.cpp/pull/2767), [#2809](https://github.com/ikawrakow/ik_llama.cpp/pull/2809)), no dense top-k mask ([#2801](https://github.com/ikawrakow/ik_llama.cpp/pull/2801)), fused sigmoid and scale in the DeltaNet path ([#2783](https://github.com/ikawrakow/ik_llama.cpp/pull/2783)), Gyro trellis kernels ([#2812](https://github.com/ikawrakow/ik_llama.cpp/pull/2812)) | No Pascal figure; its Pascal document is older | A draft that loses on prose wants a gate. The DMMV path for the 320 × 10,240 hyper-connection projections ([#2781](https://github.com/ikawrakow/ik_llama.cpp/pull/2781), +3 %) needs fast FP16, which a P106 lacks. Reordering `hc_inject` made no difference on CUDA |
| Strata 0.2.10 | DFlash2 default, MTP, 128K context on 8 GB cards | Runs on P100s and a 1070 Ti; needs 34–50 GB of host memory ([#4739](https://github.com/Niko1221/Strata/issues/4739)) | Section 3's MTP-against-DFlash2 result |
| ExLlamaV3 | Flash-Next with CPU experts and recurrent checkpoints (one 1.5 GiB host heap report, closed) | Does not compile for sm_61; the dp4a lane proposal ([#512](https://github.com/turboderp-org/exllamav3/issues/512), 10-14) has no maintainer reply | Nothing |
| vLLM, SGLang | DFlash2 and DSpark drafters, an XPU `qwen4exp` port ([#61607](https://github.com/vllm-project/vllm/pull/61607)), a fused DFlash2 scoring kernel ([#64916](https://github.com/vllm-project/vllm/pull/64916)) | Floor sm_75 (vLLM) and CUDA 13 (SGLang) | Nothing |
| ktransformers | Request open since 08-28 ([#2179](https://github.com/kvcache-ai/ktransformers/issues/2179)), no implementation | Its CPU kernels need AVX2; the rig's CPU has none | Nothing |
| llamAmpere 0.6, ds4 | llamAmpere is an sm_86 fork whose Swift MTP head has 2-bit experts; ds4 is a Mac fork | No | The 2-bit head (idea 8) |

## 5. Ideas for the rig

Ranked by est. effect on agentic speed per day of work. "No row" means the registry has no row yet.

1. **Take #31156's per-size graphs and #31274's calloc commit** on top of the Q52 graph-key layer, then count re-captures with `GGML_CUDA_GRAPH_DEBUG=1` on a DFlash2 agent check. Est. +2–7 % per draft step and −50 to −170 MiB of host memory; 1–2 days; +~400 MiB VRAM to weigh at the 54 GB edge. Q52 (layer 2), Q609.
2. **Apply #31352's 22 table rows**, then `llama-bench` prompts at `-ub` 64, 128 and 512 on one P106 for Q2_0, IQ4_NL and Q8_0. Est. +3–8 % prompt speed; 0.5 day. Q285.
3. **Re-time on the new base what upstream changed under the fork:** the 113 MiB checkpoint save and restore (#31391's mapped gather, with and without `GGML_CUDA_NO_PINNED=1`) and DFlash2's host buffers and prompt speed (#31506). No code; est. no change for the save, est. −25 to −100 MiB of pinned memory for the taps; 0.5 day on the rig. Q548, Q561, Q606, Q609.
4. **Un-gate #31525's head pairing for sm_61** with F16 and q8_0 K/V in the vector flash-attention kernel (24 heads over 2 KV heads: pair 2 or 4), after the profile shows that kernel at decode. Est. −35 to −50 % of the 12 attention layers' time at depth (est. 5–7 ms per token at 16k, 12–17 ms at 32k, 50–70 ms at 128k); 2–3 days, `test-backend-ops` cases for the new shapes. Partial alternative to Q230; no row.
5. **Take the replay-buffer rollback when its PR lands** (watch the #31584 thread), and rebase `-hybrid` on it. Est. −340 MiB of VRAM at 3 drafted tokens and ≤ 3 ms per step; 1–2 days after the PR; no row.
6. **Cherry-pick #30137** (reference-counted checkpoints) beside Q548's spread option. Est. −225 to −450 MiB of peak host memory at 2–4 checkpoints; 1 day, conflicts with #30683 and the fork's server branch. Q549, Q609, Q548.
7. **Convert the DSpark drafter to GGUF and run the agent check against DFlash2.** Accepted length 3.7 at six drafted tokens on agent sessions (SGLang, H100) against 2.6–3.1 for DFlash2 at two or three (Q561); it pays only once a drafted position costs under ~43 ms against Q605's 87 (est., from Q605's 261 ms step at 2.56 tokens), which Q591's kernels target; 1–2 days on the quant machine plus a rig run; take #30657 if the file carries log-SNR tensors. Q568.
8. **Requantize the MTP head's experts to Q2_0 or IQ4_NL** (Gyro's head 2.7 GB → est. 1.5 GB, as jakeatx's 2-bit and FLE's IQ4_NL heads) and measure acceptance on the agent check. Est. −1.2 GB on card or host; acceptance unmeasured anywhere; 1 day. Q558, Q226; no row.
9. **Time Gyro's TQ2_T kernels on one P106:** build ik_llama.cpp `main` for sm_61 on the quant machine and compare ps per weight with Q2_0's kernel. Settles Q241's reopen rule (< ~5 ps per weight; est. 9–15 now); 1 day, one card for an hour. Q241.
10. **Use the M64 card's layer study in the composed files:** Q8_0 hyper-connection mixers cost no KL (0.1834 → 0.1835 with four Q2_0 down layers; 0.1795 for the Q8_0-mixer file), layers 29, 30, 31, 35 take Q2_0 down within noise, layer 47 does not. Est. −0.9 to −1.4 GiB at equal KL for IQ3_XXS; no new run beyond Q217's KLD. Q218, Q219, Q221.
11. **Gate DFlash2 on content:** ik_llama.cpp measured +39 % on code and −7 % on prose at `n_max` 3, with acceptance 88.6 % against 39.5 %; the rig's break-even is ~1.6 tokens per step (Q572). Use the copy-gate's rule (Q53) or `--spec-draft-p-min`, which [#31294](https://github.com/ggml-org/llama.cpp/pull/31294) warns misjudges sampled drafts and varies the widths (idea 1). Est. removes the prose loss; 0.5 day. Q53, Q605.
12. **Fix the fit probe for drafters:** take #31610 and #31569, or keep measuring real loads, so Q213's "128k with the draft" counts the drafter. 0.5 day. Q213.
13. **Send P106 graph numbers to #27721 and #30727** and the older-PTX MMVQ limit to #30524. No speed; keeps the fork's gate upstreamable; 0.5 day. Q52.

Parked or dropped: #29887 and #31469 (single device, experts off the cards), #31533 (int8 MMA; the k-column kernels are the dp4a version), #31259, #29353 and #28770 (tensor-core gated), FLE, M64 and 12G32G files (no gain, or Mac-only, or IQ1 class), the Swift MTP drafters (other forks), DFlash2 retraining (+5 %, Q572 stays parked).

## Sources

- llama.cpp pull requests and issues linked in sections 1 and 3; commit list and tags read on 2026-11-03 (`7071ba3` newest).
- Hugging Face: [model listing for "Flash-Next"](https://huggingface.co/api/models?search=Flash-Next&sort=createdAt&direction=-1&limit=100), the repositories linked in section 2, the
  [ISTA GSQ-RCO discussions](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions).
- Engines: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) commits and pull requests (10-26 to 11-03), [Strata releases and issues](https://github.com/Niko1221/Strata/releases),
  [ExLlamaV3 #512](https://github.com/turboderp-org/exllamav3/issues/512), [vLLM](https://github.com/vllm-project/vllm/pulls) and [ktransformers](https://github.com/kvcache-ai/ktransformers/issues) searches,
  [llamAmpere](https://github.com/jakeatx/llamAmpere), the [multi-GPU fork](https://github.com/lukolszewski/llama.cpp-multigpu).
- Registry ids refer to [quants/experiments.md](../../../quants/experiments.md).
