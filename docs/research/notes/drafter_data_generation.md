# Generating Flint text for the drafter faster: engines, quants and GPU layout on the 7-GPU machine (2026-10-24)

Scope: the staircase drafter (Q20/D1, [pipeline drafter](pipeline_drafter.md),
[bench/staircase_drafter](../../../bench/staircase_drafter/README.md)) now gains more from training data than from
anything else: coupled acceptance on fresh text rose from 0.422 at 3.1 M tokens to 0.463 at 8.6 M, about +0.03 per
doubling, and the 0.55 gate projects to ~70 M tokens. Generation today is llama-server with the G3 mix on the 7-GPU
quant machine (3 × RTX 3090, 2 × V100 32 GB, 2 × RTX 3080): one copy on two 3090s at ~420–600 tok/s and one on the two
V100s at ~130–160 tok/s, ~0.75k tok/s or ~2.7 M tokens an hour. This note asks:
1. Which engines run Flint's architecture today, or with a small change?
2. Which formats fit which cards, given tensor or pipeline parallelism without peer-to-peer access?
3. What throughput they should reach at high batch, from published numbers for similar MoE models, and how much faster
   the whole machine could produce training text.
4. What to set up, what it risks, what needs the owner's approval, and a short trial.

A sibling note, [drafter acceptance](drafter_acceptance.md), covers what raises acceptance besides more data.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference, with the basis next to it.
- "P" and "S" in the source list mean primary (code, PRs, release notes, model cards, measurements in this repository)
  and secondary.
- Nothing was run for this note: no benchmark, no container and no file written on the quant machine. Read there:
  Flint's `config.json`, the makers' SGLang kit, `nvidia-smi` link and power fields, Docker's disk use, the generation
  logs. On a laptop: the kit's wheel unpacked and read, and length statistics of 7,109 generated answers.

## Summary

- **The generator only has to write Flint-like text; the dump labels it.** `train.py` takes the targets from the
  Q8_0 dump (`top_draft`, the top-20 log-probabilities of `llama-staircase-dump`) and the inputs from the same dump's
  hidden states; the generated tokens supply the context and a cross-entropy term at weight 0.2 next to a KL term at
  weight 1 (`drafter.kl_loss`). The run's manifest says the same: "features and targets still come from the Q8_0
  dump". So a different engine or quantization can write the text.
- **How much the writer matters, by the published evidence: little.** EAGLE trained on fixed ShareGPT answers, which
  another model wrote, and lost 3.5 % of acceptance length against answers its target wrote (τ 3.62 against 3.75).
  DistillSpec finds text written by the draft model "comparable or superior" to the target's own, as long as the target's
  distribution supervises. A 4-bit or FP8 copy of Flint is far closer to Flint than either case, so the cost should stay
  within ~1–3 % of acceptance (est.), against ~6 % gained per doubling of data. The dump measures it for free: the share
  of generated tokens outside Q8_0's kept top-k/top-p set, and their mean Q8_0 log-probability.
- **Engines:**
  - llama.cpp (Flint since PR #30928, 2026-10-21) and ik_llama.cpp (PR #2725, 2026-10-22) run Flint on every card in
    the machine.
  - vLLM merged Flint on 2026-10-22 (PR #62938, nightly only). Its fast kernels are for Blackwell; elsewhere it falls
    back to stock layers, but the attention class asserts FlashInfer, and vLLM's FlashInfer serves attention sinks only
    on compute 9.0, 10.x and 12.x. On RTX 3090s it needs a ~50-line patch to use the Triton backend, the only vLLM
    backend with sinks below compute 9.0. The V100s are out: vLLM needs compute ≥ 7.5, its images ship CUDA 13, and the
    model asserts BF16.
  - SGLang runs Flint through the makers' plugin kit (SGLang 0.5.16, CUDA 13). Its Triton attention serves sinks on
    Ampere, but no quantized format that fits three 3090s works there: BF16 is 79.6 GB, its FP8 MoE has no Marlin
    fallback, and it can't read the makers' NVFP4 GGUF.
  - TensorRT-LLM, ExLlamaV3, Aphrodite (now Sonar) and LMDeploy have no Flint code; each would need a new model
    implementation. ExLlamaV2 is archived.
- **Fit:**
  - BF16 fits nowhere.
  - The makers' NVFP4 GGUF runs in vLLM as W4A16 on Marlin at ~23.5 GB resident: pipeline-parallel over two 3090s, or
    over 3080 + 3080 + 3090. A single 24 GB card has no room left for KV.
  - vLLM's online FP8 (W8A16 on Marlin, from the local BF16, no download) is ~40 GB: three 3090s.
  - The V100s run only llama.cpp, one copy per card with a 21–23 GiB GGUF.
  - With PCIe 2.0 links (x8, x8, x4 and four x1) and no peer-to-peer, tensor parallelism pays ~98 host-staged
    all-reduces per step. Pipeline parallelism sends ~0.5–1 MB per micro-batch, so it is the only multi-GPU mode worth
    using (est.).
- **Throughput (est.):** vLLM with NVFP4, pipeline-parallel on the two 3090s, est. 1.5–2.5k tok/s, 3–5× the llama.cpp
  copy on the same cards. It gets this from both cards working at once, 3.5–7× larger batches out of paged KV, and fewer
  bytes than G3. Everything generating (vLLM on GPUs 5–6 and 0–2, single-card llama.cpp copies on the V100s): est.
  2.7–4.3k tok/s, **3.5–6× the machine's 0.75k**.
- **The dump then sets the pace.** The Q8_0 dump copies 150,144 logits per position to the host. Measured, it turns
  ~1.2k answer tokens/s with its last card on x8 and ~0.7k on GPUs 0–2. Split the seven GPUs optimally between
  generating and dumping and dumped training data comes est. 1.6–2.1× faster than today (~0.7k → 1.1–1.45k tok/s,
  ~4–5 M tokens an hour), not 3.5–6×. A top-20 on the card in the dump is the next lever.
- **Recommendation:**
  - vLLM nightly (≥ `899a1689`, CUDA 13.0 image) with the attention patch, the makers' NVFP4 GGUF, pipeline-parallel on
    GPUs 5–6 (`--attention-backend TRITON_ATTN`).
  - The V100s become two single-card llama.cpp copies (UD-Q4_K_XL or the Q4_0 finalist, q8_0 KV, 4,096 cells per slot).
  - GPUs 0–2 keep dumping.
  - The owner approves: the image pull (~10 GB compressed, onto the NVMe that must keep ≥ 50 GB free), the 22.9 GB
    NVFP4 download (or the FP8 route instead), a patched model file mounted into the container, and GPUs 5–6 moving to
    the trial.
  - Fallback with no approvals: llama.cpp tuning, est. 1.2–2×.
  - A three-step trial is in section 6.

## 1. What the generator has to produce

**The pipeline, from the code** ([README](../../../bench/staircase_drafter/README.md)):
- [generate.py](../../../bench/staircase_drafter/generate.py) asks llama-server for an answer at the makers' sampling
  (temperature 1.0, top-k 20, top-p 0.95, min-p 0, 2,048 tokens at most, seed 20000 + prompt number) and stores the
  exact prompt and answer token ids.
- `feats.py seqs` lists them and `llama-staircase-dump` replays each sequence through Flint Q8_0. It saves 17 layer
  inputs and the final hidden state in FP8, plus the top-20 next-token log-probabilities.
- [train.py](../../../bench/staircase_drafter/train.py) builds each batch from the dump record: `top_draft` from the
  dump's `top_ids`, the sampler distribution from its `top_lp`, and `sampled` from the next token of the sequence (lines
  47–54). [drafter.py](../../../bench/staircase_drafter/drafter.py) `kl_loss` returns
  `kl.mean() + 0.2 * ce.mean()`: forward KL to the Q8_0 distribution, plus cross-entropy on the generated token.

So the writer enters in two places only:
1. Which contexts the head sees.
2. The 0.2-weight cross-entropy target. A writer other than Flint Q8_0 shifts that term slightly towards its own choices.
   Setting the weight lower, or drawing that target from the dump's distribution, removes the shift if it ever matters.

The current set already mixes writers: 8.61 M tokens from Q8_0 and 5.33 M+ from G3, all dumped with Q8_0.

**How different the candidate writers are:**

| Writer | Divergence from BF16 | Source |
|---|---|---|
| G3 in llama.cpp (today's writer) | KLD 0.101–0.104 at 512 tokens | Q73, Q77 |
| UD-Q4_K_XL | 0.120 | Q73 |
| q8_0 KV instead of f16 | +0.003, inside the 0.005 noise | Q62 |
| The makers' Q4_1 (an exact repack of their group-64 INT4) | 0.316 | Q74 |
| The makers' NVFP4 | not measured; same family of round-to-nearest 4-bit as their INT4, so likely 0.2–0.35 (est.) | – |
| FP8 weights, per channel | not measured; typically ~0.01–0.03 (est.) | – |
| A second BF16 engine | top-1 agrees with FP32 on only 90–96 % of tokens; prompt KL against FP32 0.04–0.21 | vLLM PR #62938 |

Engine numerics alone already move Flint by a KL of the same order as a good 4-bit quant: "The model is ill-conditioned
in BF16" (vLLM PR #62938).

**The published evidence on how much the writer matters:**
- **EAGLE, §4.3.3, Table 6** (LLaMA2-Chat 7B, MT-bench): fixed ShareGPT answers give speed-up 2.78× and τ 3.62;
  answers from the target give 2.88× and 3.75. That is 3.5 % of acceptance length for text from a different model
  family, labelled with the target's own features. The authors call it "low sensitivity to training data".
- **DistillSpec, §5.1–5.2:** with white-box supervision, model-generated data beats a fixed dataset by ~20 % extra
  speed-up, and data from the draft model is "comparable or superior" to data from the target.
- **On Flint:** text from an NVFP4, FP8 or G3 copy of Flint is much closer to Flint's than either published case. The
  expected cost is within ~1–3 % of acceptance (est.). At +0.03 coupled per doubling (~6–7 % relative), a writer that
  is ≥ 1.5× faster wins even at the high end of that range (est.).

**A free fidelity check.** The dump already holds Q8_0's top-20 at every answer position, and the next token of the
sequence is what the writer sampled. Two numbers per writer, from the dump files with `feats.py` and no GPU time:
- the share of sampled tokens outside Q8_0's kept set (top-k 20, then top-p 0.95);
- the mean Q8_0 log-probability of the sampled token, counting tokens outside the top-20 at the 20th value as a bound.

Compare a new writer with G3 on the same prompts. If its out-of-set share is several times G3's, use the FP8 route
(section 5).

**The workload** (7,109 answers in two generation files, read on a laptop):

| | Mean | Median | p90 | p99 | Max |
|---|---|---|---|---|---|
| Prompt tokens | 411 | 251 | 1,059 | 2,992 | 4,825 |
| Answer tokens | 1,222 | 1,417 | 2,048 | – | 2,048 |

- 38 % of answers stop at the 2,048-token limit and 62 % at an end token.
- Prompts are ~25 % of all tokens, so the dump's answer-token rate is ~0.75 × its total rate (est.).
- KV per sequence: 28 KiB per token on the 14 full-attention layers plus at most 513 × 70 KiB ≈ 35 MiB on the 35
  sliding layers. That is ~63 MiB on average over a sequence's life and ~160 MiB at the longest (est.).

## 2. Engine support (question 1)

Flint's `config.json`:
- class `FlintForCausalLM`, `model_type: flint`, 49 layers, hidden 2,048, 32 query and 4 KV heads of 128;
- 256 experts × 512 with top-6 sigmoid routing and renormalized weights (`norm_topk_prob`), a 1,024-wide shared expert,
  dense layers 0–1;
- vocabulary 150,144, 262,144 positions;
- **hybrid attention:** `layer_types` puts 35 `sliding_attention` layers (window 513, current token included) between
  14 `full_attention` layers (0, 1, 5, 9, …, 45, 48);
- the full-attention layers carry no RoPE (`no_rope_layer_indices`);
- per-head QK RMSNorm and learned attention sinks (`softmax_type: learnable`).

Transformers does not support Flint yet. The makers' kit loads the config through its own `FlintConfig`.

| Engine | Flint today | RTX 3090 / 3080 (sm_86) | V100 (sm_70) | Change needed | Verdict |
|---|---|---|---|---|---|
| llama.cpp | merged in PR #30928 (2026-10-21); the generation tree is upstream `d77256999` plus the rescale switch | yes | yes (CUDA 12.9 build, sm 70 and 86) | none | the baseline; tuning in section 5 |
| ik_llama.cpp | merged in PR #2725 (2026-10-22): fused MoE, `-sm graph` tensor split, `--swa-compress`; the `-sm attn` fix PR #2744 is open | yes | yes (PR #2744 ran on P100s and P40s) | none (the host needs AVX2: the Ryzen 9 5900X has it) | possible; Flint's CUDA path is new (PR #2725 "CUDA was not exercised"), no batched-throughput evidence |
| vLLM | merged in PR #62938 (2026-10-22, nightly only; v0.32.0 of 2026-10-18 predates it). BF16 and NVFP4 GGUF (W4A16) | **not as merged**: `FlintAttention` asserts `FlashInferImpl`, and FlashInfer serves sinks only on compute 9.0/10.x/12.x | no: compute ≥ 7.5, CUDA 13.0.3 images (arch list 7.5–12.0), BF16 asserted | ~50 lines: skip the FlashInfer-only path below SM100 and call the standard `Attention` with `TRITON_ATTN` | **recommended for the 3090s and 3080s** |
| SGLang | the makers' kit: SGLang 0.5.16 (CUDA 13) + plugin `flint_sglang` 0.1.4; PR #45624 open | the plugin allows TP/PP with a warning and the Triton backend serves sinks, but no format fits: BF16 79.6 GB, FP8 MoE without a Marlin path, NVFP4 GGUF unsupported | no: CUDA 13 dropped Volta; FlashInfer's CUDA RMSNorm is required and starts at sm_75 | a W4A16 checkpoint SGLang loads, e.g. the makers' INT4 repacked, then validation | not now |
| TensorRT-LLM | no code, no issue | Ampere supported in general | no (Ampere is the floor) | a new model | no |
| ExLlamaV2 | archived | – | – | – | no |
| ExLlamaV3 | no code; has `gpt_oss.py` (sinks, sliding window) and `qwen3_moe.py` | yes | no (sm_75 instructions, exllamav3#512) | a new architecture file plus EXL3 quantization from BF16 (GPU-hours) | no: built for few streams, no batched-MoE evidence |
| Aphrodite (now Sonar) | no code | compute ≥ 8.0, CUDA 13 wheels | no | port vLLM's Flint model | no reason over vLLM |
| LMDeploy | no code; TurboMind has gpt-oss and Qwen3 MoE | yes | yes (Volta listed) | a C++ TurboMind port (NoPE layers, sigmoid routing, shared expert) and AWQ calibration without Hugging Face model code | no: weeks, not a small change |

**vLLM in detail** (main `e7473bf6`, 2026-10-24; the Flint model unchanged since its merge commit `899a1689`):
- `_flint_kernels()` returns False unless the device is in the SM100 family, TP = 1 and the format is BF16 or NVFP4.
  Then the MoE block takes the stock `FusedMoEFactory` with `scoring_func="sigmoid"` and the shared expert, which runs
  on any card.
- `FlintAttention.__init__` imports FlashInfer's metadata classes and asserts
  `isinstance(impl, FlashInferImpl), "Flint requires FlashInfer attention"`, BF16 activations and a BF16 KV cache, for
  every device. Its `forward` writes K and V into the FlashInfer cache layout with the fused Triton
  `qk_norm_rope_cache` before calling the backend.
- On sm_86, FlashInfer's `supports_sink()` returns False (only compute 9.0, 10.x and 12.x). The metadata builder then
  raises "FlashInfer backend currently does not support attention sinks, please use trtllm on blackwell or flash
  attention on earlier GPUs".
- vLLM's FlashAttention backend refuses sinks below compute 9.0 ("sink not supported on compute capability < 9.0").
  `TRITON_ATTN` reports `supports_sink() = True` and accepts every compute capability.
- **The patch** (untested sketch): set `self.flint_decode = _flint_kernels(quant_config)` first, and return from
  `__init__` before the FlashInfer imports and asserts when it is False. In `forward`, when it is False:
  1. split `qkv`;
  2. apply `q_norm` and `k_norm` per head;
  3. apply `rotary_emb` on the sliding layers only (it is `None` on the NoPE layers);
  4. return `o_proj(self.attn(q, k, v))`.

  The standard `Attention` layer writes its own cache. Serve with `--attention-backend TRITON_ATTN`.
- The PR's own validation used the stock layers as the reference ("prompt-logprob KL against an FP32 reference forward
  pass stays at the level of the stock-layer implementation"). So the fallback is the numerically checked path; it has
  only never run on Ampere.
- **NVFP4 on Ampere:** `_nvfp4_gguf_quant_config()` builds a `ModelOptNvFp4Config` with `quant_method="W4A16_NVFP4"`.
  That method's minimum capability is 7.5, and the NVFP4 MoE oracle has a `MARLIN` backend. Online FP8 MoE
  (`--quantization fp8`) also has a Marlin W8A16 backend.
- The model declares `SupportsPP` and builds its layers with `make_layers`, so `--pipeline-parallel-size` applies.
  `VLLM_PP_LAYER_PARTITION` sets uneven stage sizes.
- vLLM's custom all-reduce needs P2P (`_can_p2p`) and falls back to NCCL without it.
- **No torch.compile:** the model has no `@support_torch_compile`. vLLM falls back to decode-only CUDA graphs
  (FULL_DECODE_ONLY), so every step that carries a prefill chunk runs eagerly (vLLM issue #63020).

**SGLang in detail:**
- The kit (`sglang/README.md` in the BF16 repository) validates one H200 or B300 per replica. It lists "Tensor, expert
  or pipeline parallelism, DP attention, quantized weights or KV cache … not validated", and "80 GB GPUs are not
  supported".
- The plugin builds Flint on SGLang's Qwen3-MoE classes with a `RadixAttention(..., sinks=...)`. `_check_runtime` only
  warns for TP or PP > 1, but refuses MoE runners other than `auto`, `triton` and `flashinfer_trtllm_routed`.
- SGLang main's Triton attention backend passes `sinks` to its kernels, so attention is not the blocker on Ampere.
- The blocker is the format: BF16 does not fit three 3090s. SGLang's `Fp8MoEMethod` has no Marlin path (only its
  linear FP8 method does). SGLang has no loader for the NVFP4 GGUF.
- The kit's image is ~50 GB.

## 3. What fits where (question 2)

**The cards, read with `nvidia-smi`:**

| GPU | Card | Memory | PCIe link (current, max host gen 2) | Power limit |
|---|---|---|---|---|
| 0 | RTX 3080 | 10 GiB | x1 | 220 W |
| 1 | RTX 3080 | 12 GiB | x1 | 220 W |
| 2 | RTX 3090 | 24 GiB | x4 | 220 W |
| 3, 4 | V100-PCIE | 32 GiB each | x1 | 140 W |
| 5, 6 | RTX 3090 | 24 GiB each | x8 | 220 W |

- Every link runs at PCIe 2.0 (`pcie.link.gen.hostmax` 2): ~4 GB/s per direction at x8, ~2 at x4, ~0.5 at x1.
- GPUs 0–4 share one switch; 5 and 6 hang off the host bridge.
- There is no P2P. The power limits are the owner's presets and are not ours to change.

**Formats per engine and card set** (KV at ~63 MiB per sequence on average, section 1):

| Format | Size on the cards | Fits | KV left (est.) | Engine |
|---|---|---|---|---|
| BF16 safetensors (local) | 79.6 GB | no set (3 × 3090 = 72 GB; the V100s can't run BF16 engines) | – | – |
| FP8 W8A16, vLLM online from the local BF16 | ~40 GB (body FP8, embedding and head BF16) | PP = 3 on GPUs 2, 5, 6; PP = 4 with the 3080s | ~20 GB on 2, 5, 6 at 0.92 utilization: ~300 sequences | vLLM (Marlin) |
| NVFP4 GGUF, the makers' (22.9 GB download) | 23.5 GB resident (head dequantized to BF16) | PP = 2 on GPUs 5, 6; PP = 3 on 0, 1, 2 (uneven, ~9/11/29 layers); not on one 24 GB card (~1 GB left) | ~19 GB on 5, 6: ~300 sequences; ~15 GB on 0–2 | vLLM (Marlin) |
| G3 GGUF (today) | 27.6 GiB | 5 + 6, 3 + 4, 0 + 1 + 2 | 32–40 slots on 5 + 6 at 8,192 cells × f16 (~300 MiB per slot) | llama.cpp |
| UD-Q4_K_XL GGUF (local) | 23.3 GiB | one V100 (~6.5 GiB left); not one 3090 | ~60 slots at 4,096 cells × q8_0 (~100 MiB per slot) | llama.cpp |
| Q4_0 finalist `imatU-Q4_0-p4648b` (local) | 21.5 GiB | one V100 (~8.5 GiB left) | ~80 slots at 4,096 × q8_0 | llama.cpp |
| A ~3-bit mix (quantized locally from the BF16 GGUF with the importance matrix) | ~17 GiB (UD-Q3_K_XL is 17.2 GiB) | one 3090 (~5 GiB left) | ~45 slots at 4,096 × q8_0 | llama.cpp |

- **Tensor or pipeline parallel without P2P:**
  - Tensor parallel splits every layer, so each decode step needs two all-reduces per layer: 98 for Flint.
  - At 256 sequences each moves 256 × 2,048 × 2 B ≈ 1 MiB through host memory. At ~3 GB/s effective on PCIe 2.0 x8,
    plus ~50 µs of latency each, that is ~40–60 ms per step (est.), several times the compute.
  - Pipeline parallel sends one hidden state per sequence per stage boundary, ~1 MiB per micro-batch, ~0.3 ms even at
    x4 (est.).
  - So: pipeline parallel only, and the x1 cards can be stages.
- **Pipeline stages and memory:**
  - In vLLM's pipeline the slowest stage sets the rate.
  - Two 3090s balance.
  - 3080 (10 GiB) + 3080 (12 GiB) + 3090 must give the 3090 most of the layers to leave KV room on the 10 GiB card,
    so that pipeline runs at roughly 1.7 cards' worth (est.).
- **llama.cpp's slots are fixed in size.** With `-no-kvu` every slot reserves its share of `-c`, 8,192 cells today:
  ~224 MiB in f16 on the full-attention layers, ~77 MiB for its sliding window, ~300 MiB per slot (est.). The unified
  cache (`-kvu`) avoids the reservation, but every decode token then attends over all slots' cells: 64–72 slots
  "crawled" at ~300 tok/s together (Q20 log, 12:05).
  - 97–98 % of prompts are ≤ 2,048 tokens.
  - 4,096 cells per slot in q8_0 (`-ctk q8_0 -ctv q8_0`, no measurable KLD cost in Q62) is ~100 MiB per slot, about 3×
    the slots in the same memory (est.).
  - The few longer prompts go to a copy with longer slots (the V100s), or `--max-prompt 2048` skips them.

## 4. Expected throughput (question 3)

**Published numbers for Flint and similar MoE models:**

| Model, format | Hardware | Engine | Load | Output tokens/s | Source |
|---|---|---|---|---|---|
| Flint NVFP4 (W4A16) | 1 × GB200 (~8 TB/s) | vLLM with the Flint kernels | random 1,000 in / 1,000 out; TPOT 2.0 / 2.8 / 4.9 / 8.5 ms at concurrency 1 / 8 / 32 / 128 | ~500 / 2,900 / 6,500 / 15,000 (concurrency ÷ TPOT, est.) | vLLM recipe in PR #62938 (P) |
| Qwen3-30B-A3B Q4_K_M (17 GB; 128 experts × 768, top 8, 3.3B active) | 1 × RTX 5090 (1.79 TB/s) | llama.cpp `llama-batched-bench`, 64 + 64 tokens | 4 / 16 / 32 sequences | 814 / 1,581 / 2,434 | llama.cpp PR #26802 (P) |
| Qwen3-Coder-30B-A3B AWQ | 1 × RTX 5090 | vLLM | 4k in / 4k out, concurrency 8 → 24 | 869 → 1,186 (peak, KV-bound) | CloudRift (S) |
| Qwen3.8-27B dense, 4-bit | RTX 5090 / L40S / RTX 4090 | vLLM vs llama.cpp `-np 32` | 474 in / 256 out, concurrency 32 | 936 / 554 / 300 vs 369 / 241 / 306 | computingforgeeks, 2026-10 (S) |
| Llama 3.1 8B FP16 | 1 × RTX 3090 | vLLM | 100 concurrent users | 12.88 per user (~1.3k total, est.) | The Register / Backprop, 2024 (S) |

What these say:
- **vLLM beats llama.cpp at high concurrency only when it has KV room.** On single cards with a 4-bit model vLLM led
  2.3–2.5× at concurrency 32 (RTX 5090, L40S). On the 24 GB RTX 4090, with ~20 GB of weights, the two were level.
- **llama.cpp on one card is efficient.** In PR #26802's row at 32 sequences, the 5090 reads ~15 GB of expert and dense
  weights per step (87 % of experts touched) in 13.1 ms: ~64 % of its bandwidth (est.).
- **Flint's own number on GB200** at concurrency 128 is ~34 % of the weight-read bound (23.5 GB / 8 TB/s = 2.9 ms
  against 8.5 ms per step, est.).

**Measured in this project (Q20 logs), all llama.cpp with `-no-kvu -bs` unless noted:**

| Copy | Cards | Slots | tok/s |
|---|---|---|---|
| G3 | 3090 + 3090 | 32–40 | ~420–600 |
| G3 | 3080 + 3080 + 3090 | 28 | ~405–475 (40 slots ran out of memory on the 10 GB card) |
| G3 | V100 + V100 | 48–64 | ~130–160 |
| Q8_0 | 3090 + 3090 (`-kvu`) | 24 | ~310–370 |
| Q8_0 | 2 × 3080 + 2 × 3090 | 56 | ~470 |

- On GPUs 5–6 at ~36 slots a step takes ~72 ms (36 / 500).
- G3 then reads ~17 GB per step (57 % of experts touched at 36 sequences) at one card's bandwidth, since a layer split
  runs the cards in turn: ~19 ms of weight reads. So llama.cpp sits at ~25–30 % of that bound (est.).
- The rest is the layer split's idle card, the small batch and attention over 8,192-cell slots.

**Estimates for this machine (est.):**

| Setup | Cards | tok/s | Against today on the same cards | Basis |
|---|---|---|---|---|
| vLLM, NVFP4, PP = 2 | 5, 6 | 1.5–2.5k | 3–5× | Both cards busy instead of in turn (≤ 2×). 128–256 sequences instead of ~36: the per-token weight read falls from ~0.48 to ~0.1–0.17 GB (2–3× at equal efficiency). 23.5 instead of 29.6 GB. Less eager prefill steps, pipeline bubbles and lower Marlin efficiency at ~3–6 tokens per expert (×0.5–0.7). The GB200 number scaled by bandwidth (15k / 8.5 ≈ 1.8k per 3090-equivalent at concurrency 128) agrees |
| vLLM, online FP8, PP = 3 | 2, 5, 6 | 1.3–2.3k | – | 1.75× the weight bytes of NVFP4 on three cards instead of two |
| vLLM, NVFP4, PP = 3 (uneven) | 0, 1, 2 | 1.0–1.5k | 2–3× the G3 copy | ~1.7 cards' worth, ~200 sequences |
| llama.cpp, one copy per V100 (UD-Q4_K_XL or Q4_0 finalist, q8_0 KV, 4,096 cells, ~50–60 slots each) | 3, 4 | 0.2–0.3k | 1.3–2× | Each card runs its own batch instead of half the layers in turn. With h(B) the step time at batch B, the gain at equal total slots is h(2B) / h(B), ~1.3×; at equal slots per card it is 2× |
| llama.cpp, G3 with 4,096 × q8_0 slots, 96–128 slots | 5, 6 | 0.7–1.0k | 1.4–2× | Larger batches only: PR #26802 shows 1.54× from 16 to 32 sequences on one card |
| llama.cpp, one ~3-bit copy per 3090 | 5, 6 | 0.8–1.2k | 1.6–2.4× | Each card on its own batch with ~17 GiB of weights; lower text fidelity |

**The whole machine.** Generation alone, with vLLM on 5–6 and 0–2 and single-card llama.cpp copies on the V100s:
est. 2.7–4.3k tok/s, **3.5–6× today's 0.75k**, 10–15 M tokens an hour. With llama.cpp tuning only (V100 copies plus
larger batches or single-card copies on 5–6): est. 0.9–1.5k, 1.2–2×.

**The dump bounds the training data.** `llama-staircase-dump` copies 600 KB of logits per position to the host and
picks the top-20 there. Measured (Q20, total tokens including the prompt):
- ~1,650 tok/s with its last card on x8, about ~1.2k answer tokens/s (d1: 3.08 M tokens in 43 minutes on GPUs 0, 1, 5, 6);
- ~850 tok/s with the last card on x4 (GPUs 0–2: ~34 answers a minute, ~0.7k answer tokens/s);
- ~280 tok/s through the V100s' x1 links.

Today generation (0.65–0.75k) and the dump on GPUs 0–2 (~0.7k) are balanced: ~2.5 M dumped tokens an hour. With the
faster generator, the best split of the seven GPUs between the two jobs (a small linear program over the rates above)
gives est. 1.1–1.45k dumped tokens/s, ~4–5 M an hour: **1.6–2.1× end to end**. In practice that means generating on
the fast sets in long stretches and dumping on everything in between.

From there the dump is the lever:
- **A top-20 on the card.** ggml has `ggml_top_k`, and llama-server's `-bs` already samples top-k on the GPU over the
  150,144-token vocabulary. It would cut the host copy from 600 KB to ~160 B per position, which matters most on the
  x4 and x1 sets.
- **Dumping with the release file instead of Q8_0** fits one V100 per dump copy and matches the hidden states the head
  reads on the rig. Whether Q8_0 or the release file is the better teacher is Q20's call, not this note's.
- **Storage:** the dump writes ~37 KB per answer token (~12 KB for the six-card copy). 70 M tokens would fill ~2.6 TB,
  about all of the bulk HDD's 2.8 TB free; the six-card copy alone needs ~0.85 TB.

## 5. Recommendation (question 4)

**Setup:**
1. **GPUs 5–6: vLLM nightly, the makers' NVFP4 GGUF, pipeline-parallel over both cards.**
   - Image `vllm/vllm-openai:nightly-<sha>` from a commit at or after `899a1689` (2026-10-22). It ships CUDA 13.0.3,
     which driver 580 supports. Avoid the `cu134` tags, which need a newer driver than the machine's.
   - Pin the digest. Mount the patched `vllm/models/flint/model.py` read-only over the image's copy.
   - Untested sketch:

   ```bash
   docker run -d --label pascal-quants=1 --gpus '"device=5,6"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID --ipc=host \
     --network host -v <models>:<models>:ro -v <patched model.py>:<site-packages>/vllm/models/flint/model.py:ro \
     vllm/vllm-openai:nightly-<sha> \
     <models>/flint-nvfp4.gguf --hf-config-path <models>/flint --tokenizer <models>/flint \
     --served-model-name flint --pipeline-parallel-size 2 --attention-backend TRITON_ATTN \
     --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --port <port>
   ```

   - Pass `--hf-config-path` and `--tokenizer` as the BF16 directory: the GGUF carries neither config nor tokenizer.
   - Mount every drive the container reads, at the same path.
2. **The client:** a vLLM variant of `generate.py` that posts to `/v1/completions` and writes the same JSONL rows, so
   `feats.py seqs`, the continuous feed and the dump run unchanged. Each request carries:
   - `prompt` as the token ids llama.cpp's template produces today;
   - `max_tokens` 2048, `temperature` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0;
   - `seed`;
   - `stop_token_ids` [150001, 150003, 150005, 150018];
   - `return_token_ids` true.

   Check, on rows already generated:
   - that the Hugging Face template and tokenizer give the same prompt ids as llama.cpp (the rows store
     `prompt_tokens`);
   - that the final stop token is handled the way the existing rows handle it.

   Admit new requests in bursts, e.g. refill when 32 slots are free. Prefill steps then cluster, and the eager
   (uncompiled) mixed steps stay a small share.
3. **GPUs 3–4 (V100s): two single-card llama.cpp copies**, which no other engine can do there:
   - UD-Q4_K_XL or the Q4_0 finalist, the rescale left on (both have Q4-class down-projections, [V100 note](transfer_v100.md));
   - `-ctk q8_0 -ctv q8_0`, `-no-kvu -bs`, ~50–60 slots at 4,096 cells;
   - one copy with longer slots takes the prompts over 2,048 tokens.
4. **GPUs 0–2:** keep the continuous Q8_0 dump. When the generated stock runs ahead by more than a few hours of
   dumping, move 5–6 to the dump for a stretch.
5. **Later, if the trial holds:**
   - a second vLLM engine, NVFP4 PP = 3 over 0–2 (`VLLM_PP_LAYER_PARTITION`, e.g. 9,11,29), whenever those cards are
     not dumping;
   - the dump's top-20 on the card.

**Approvals the owner gives first:**
- **The image pull**, ~9.6 GB compressed for amd64 and more on disk. Docker keeps its data on the NVMe root, which had
  80 GB free when checked and must keep ≥ 50 GB. Docker lists ~27 GB of old images as reclaimable; pruning them is the
  owner's call.
- **The NVFP4 download**, `RecursiveSI/flint-nvfp4-gguf`, 22.9 GB, to the bulk HDD. The alternative needs no model
  download: online FP8 from the local BF16, PP = 3 on GPUs 2, 5, 6. It writes more faithful text, but takes GPU 2 from
  the dump set, and the V100s dump slowly.
- **Running our patched vLLM model file** inside the container. Nothing is installed on the host.
- **GPUs 5–6 moving from the llama.cpp copy to the trial.** Under the machine's no-speed-measurement rule the trial is
  a production generation run read from the client's own progress log. A concurrency sweep or any other benchmark
  needs the owner's separate OK.

**Risks:**
- **The patch and the path are new.** Flint has never run on vLLM's stock layers on Ampere: Marlin NVFP4 MoE with
  sigmoid routing and the shared expert, the Triton attention with sinks and the 513-token window, pipeline stages
  over the hybrid KV cache. Section 6's fidelity gate catches wrong output.
- **Eager mixed steps.** Without torch.compile only pure decode steps run as CUDA graphs. The low end of the estimate
  assumes they cost ~30–40 %.
- **NVFP4 text fidelity.** The makers' 4-bit Q4_1 scores KLD 0.316 against G3's 0.10. If the out-of-set share is high,
  switch to FP8 or keep G3 on llama.cpp.
- **Pipeline bubbles and host overhead** in vLLM's pipeline: gLLM reports 11–398 % more throughput than vLLM's pipeline
  and SGLang's tensor parallelism on PCIe boxes, so vLLM leaves some on the table.
- **The 220 W limits** on the 3090s cut compute at large batches, for every engine alike.
- **Our GPU containers block the machine's own service presets**: stop them when done. Disk headroom on the NVMe, as above.
- **The dump falls behind** (section 4); plan the generation stretches around it.

**Fallback with no approvals (llama.cpp only, est. 1.2–2× generation):**
- the V100 copies of item 3;
- on 5–6, G3 at 96–128 slots with 4,096 × q8_0 cells, or two single-card copies of a ~3-bit mix quantized locally
  from the BF16 GGUF (CPU lane), with the fidelity check deciding;
- keep `-no-kvu`, `-bs`, `-b 2048 -ub 512` (a decode step is at most one micro-batch), `--cache-ram 0` and
  `cache_prompt: false`;
- don't use `-kvu`, tensor splits over PCIe, `--swa-full` or q4_0 KV;
- ik_llama.cpp is a later option once someone measures it on Flint's CUDA path.

## 6. Trial plan

1. **Laptop (no compute): code.** Write the attention patch and the vLLM client in `bench/staircase_drafter/`, and
   the fidelity script on top of `feats.py`. Check the Hugging Face prompt ids against the stored llama.cpp ids on a
   few hundred rows.
2. **GPUs 5–6, ~1 h: correctness.** Start the engine and generate 100 unused prompts with fixed seeds.
   - Check: the answers stop at end tokens about as often as G3's (62 %), the length distribution looks like G3's, and
     the text is readable.
   - Dump those 100 and the same 100 prompts' G3 answers with Q8_0, and compute the fidelity numbers.
   - Gate: an out-of-set share at most ~2× G3's, and a mean Q8_0 log-probability within ~0.1 nats of G3's (est.
     thresholds; G3's own numbers set the scale). If the gate fails, the FP8 route repeats this step on 2, 5, 6.
3. **GPUs 5–6, ~2 h: production.** Run the generation loop on the unused prompts. Read tokens per hour from the
   client's log and compare with the G3 copy's logs on the same cards (~1.5–2.2 M an hour).
   - Keep it if it reaches ≥ 2.5× (≥ ~4.5 M an hour).
   - Then set up the V100 copies, then the 0–2 engine when the dump allows it.
   - Below 1.5×, go back to the llama.cpp fallback.

The heads keep training on whatever has been dumped. The text from every writer joins the same set, since the Q8_0
dump labels all of it.

## Not verified

- No throughput in this note was measured on the machine; every rate for a new setup is an estimate.
- The vLLM patch is a sketch. That vLLM's Flint stock path, Marlin NVFP4 MoE and Triton sinks run correctly on sm_86
  is inferred from the code, not observed. vLLM's pipeline parallelism with a hybrid sliding-window cache and sinks is
  likewise untested here.
- The NVFP4 file's KLD against BF16 is unknown; the 0.2–0.35 range is by analogy with the makers' INT4.
- The writer's effect on acceptance (~1–3 %) is extrapolated from EAGLE and DistillSpec, not measured on Flint. A
  head trained on equal amounts of G3 and NVFP4 text would measure it, at ~2 × 5 GPU-hours.
- Online MXFP4 in vLLM (a 4-bit route from the local BF16 with no download) exists in the code, but whether it runs as
  weight-only Marlin on sm_86 was not checked.
- ik_llama.cpp's batched throughput on Flint is unknown.

## Sources

Flint and its kits:
- Flint BF16 model card, `config.json`, `generation_config.json` and the SGLang kit (`sglang/README.md`, `Dockerfile`,
  `install.sh`, `serve_h200.sh`, wheel `flint_sglang-0.1.4`: `plugin.py`, `model.py`), revision of 2026-10-21,
  [huggingface.co/RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) (P)
- [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf): `flint-nvfp4.gguf`, 22,863,084,160
  bytes (P); [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) (P)

vLLM:
- [PR #62938](https://github.com/vllm-project/vllm/pull/62938), "[Model] Add Flint (RecursiveSI/flint)", merged
  2026-10-22 as `899a1689`: description, `vllm/models/flint/model.py`, `recipes/models/RecursiveSI/flint.yaml` (P)
- vLLM main `e7473bf6` (2026-10-24):
  - [`v1/attention/backends/flashinfer.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py)
    (`supports_sink`, the sinks error);
  - [`flash_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)
    (sinks below 9.0 refused);
  - [`triton_attn.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py)
    (`supports_sink` True);
  - [`quantization/modelopt.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py)
    (W4A16_NVFP4 → Marlin, minimum capability 75);
  - `fused_moe/oracle/{nvfp4,fp8}.py` (Marlin backends);
  - `distributed/device_communicators/custom_all_reduce.py` (`_can_p2p`);
  - `envs.py` (`VLLM_PP_LAYER_PARTITION`);
  - `entrypoints/openai/completion/protocol.py` (`return_token_ids`, `top_k`, `min_p`, `stop_token_ids`);
  - `docker/versions.json` and `docker/Dockerfile` (CUDA 13.0.3, arch list 7.5–12.0) (P)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/): "compute capability 7.5
  or higher" (P)
- [vLLM issue #63020](https://github.com/vllm-project/vllm/issues/63020): the `flint` image, decode-only CUDA graphs (P)
- Docker Hub `vllm/vllm-openai` tags (`nightly-*`, `flint`, `v0.32.0-cu129`), sizes as listed on 2026-10-24 (P)

SGLang:
- [PR #45624](https://github.com/sgl-project/sglang/pull/45624) (open) (P)
- SGLang main `773cb938`:
  [`triton_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_backend.py)
  (sinks), [`quantization/fp8.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8.py)
  (Marlin only in `Fp8LinearMethod`) (P)
- [SGLang attention backends](https://docs.sglang.io/advanced_features/attention_backend.html) (P)
- SGLang install docs and FlashInfer's sm_75 floor, as cited in [engines other](engines_other.md) (P)

llama.cpp family:
- [llama.cpp PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928) (Flint) (P)
- [llama.cpp PR #26802](https://github.com/ggml-org/llama.cpp/pull/26802) (CUDA graphs with `mul_mat_id`; batched-bench
  on Qwen3-30B-A3B, RTX 5090) (P)
- llama.cpp master `78a8ae14`, `ggml/src/ggml-cuda/ggml-cuda.cu` (graph compatibility check) (P)
- ik_llama.cpp [PR #2725](https://github.com/ikawrakow/ik_llama.cpp/pull/2725) (Flint) and
  [PR #2744](https://github.com/ikawrakow/ik_llama.cpp/pull/2744) (`-sm attn` fix, PPL on P100/P40) (P)

Other engines:
- ExLlamaV3 `da68d57d`, [`exllamav3/architecture/`](https://github.com/turboderp-org/exllamav3/tree/master/exllamav3/architecture);
  [exllamav3#512](https://github.com/turboderp-org/exllamav3/issues/512) (P)
- [LMDeploy supported models](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html) (P);
  TensorRT-LLM support matrix and Sonar install docs, as cited in [engines other](engines_other.md) (P)
- GitHub searches for "Flint" and `FlintForCausalLM` in TensorRT-LLM, ExLlamaV3, LMDeploy and dphnAI/sonar: no results,
  2026-10-24 (P)

CUDA:
- [CUDA Features Archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html): "Offline compilation and
  library support for these architectures [Maxwell, Pascal, Volta] have been removed in CUDA Toolkit 13.0" (P)
- [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html): CUDA 13.x runs on
  drivers ≥ 580 (P)

Training data:
- Y. Li et al., "EAGLE", [arXiv 2401.15077](https://arxiv.org/abs/2401.15077), §4.3.3 and Table 6 (P)
- Y. Zhou et al., "DistillSpec", [arXiv 2310.08461](https://arxiv.org/abs/2310.08461), §5.1–5.2 (P)

Throughput:
- Y. Guo et al., "gLLM", [arXiv 2504.14775](https://arxiv.org/abs/2504.14775) (P)
- [CloudRift: optimizing Qwen3-Coder for RTX 5090 and PRO 6000](https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000) (S)
- [computingforgeeks: llama.cpp vs vLLM vs Ollama, Qwen3.8-27B](https://computingforgeeks.com/?p=171658) (S)
- [The Register, 2024-08-23: an RTX 3090 serving Llama 3.1 8B](https://www.theregister.com/2024/08/23/3090_ai_benchmark/) (S)

This repository:
- [bench/staircase_drafter](../../../bench/staircase_drafter/README.md): `generate.py`, `train.py`, `drafter.py`,
  `feats.py` and the dump rates in its README (P)
- [quants/experiments.md](../../../quants/experiments.md): Q20, Q62, Q73, Q74, Q77 (P)
- [transfer to V100](transfer_v100.md), [transfer to RTX 3090](transfer_rtx3090.md), [idle cards](idle_cards.md),
  [engines other](engines_other.md), [rec engine](rec_engine.md) (P)
