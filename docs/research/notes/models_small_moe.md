# Small and medium MoE LLMs (up to ~40B total) for the 9 x P106-090 rig: catalogue, fit and expected speed (as of 2026-10-22)

Method note (applies to every section): architecture numbers come from each model's `config.json` and parameter counts and dates from the Hugging Face model API (`/api/models/<repo>`, field `createdAt` = repo creation date, which is usually the release date or a day or two before). GGUF file sizes come from the Hugging Face file-tree API (`/api/models/<repo>/tree/main?recursive=true`), with split files summed; sizes are decimal GB (1 GB = 10^9 bytes). All were fetched on 2026-10-22. Active-parameter, KV-cache and decode-ceiling numbers marked "computed" are my own arithmetic from the configs, not vendor figures; the formulas are given in the last two sections. Nothing was run on the rig and no model files were downloaded.

## Which MoE models of <= ~40B total parameters exist as of October 2026?

### Takeaway
There are about 30 open-weight MoE LLMs in or near the 6-49B range. The most relevant 2026 additions are Qwen3.5/3.6-35B-A3B (Gated DeltaNet hybrid), Gemma 4 26B-A4B, GLM-4.7-Flash, NVIDIA Nemotron 3.5 Lightning 30B-A3B (successor to Nemotron 3 Nano), Liquid LFM2.5-24B-A2B (released 2026-10-12), inclusionAI Ling-3.0-tiny and Cohere North-Mini-Code-1.0, plus fine-tunes built on the Qwen3.5/3.6 architecture (KAT-Coder-V2.5-Dev, MiniCPM-V-4.7-35B-A3B, Qwen-AgentWorld-35B-A3B). The newest Qwen generation, Qwen3.8, has no small MoE: its only MoE releases are Qwen3.8-Flash-Next (~180B stored parameters) and Qwen3.8-2.4T-A95B. I found no open Qwen3.7 weights from Qwen. Mistral and ByteDance have no open MoE at or under 40B, and Granite 4.1/4.2 30B turned out to be dense.

### Cited Findings
**Qwen family**
- Qwen3-30B-A3B-Instruct-2507 (repo created 2025-07-28), Qwen3-30B-A3B-Thinking-2507 (2025-07-29) and Qwen3-Coder-30B-A3B-Instruct (2025-07-31) each have 30.53B parameters (`qwen3_moe`) and are Apache-2.0 — [HF API: Instruct-2507](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-Instruct-2507), [Thinking-2507](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-Thinking-2507), [Coder-30B](https://huggingface.co/api/models/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- Qwen3-VL-30B-A3B-Instruct (2025-09-30, 31.07B, Apache-2.0) and Qwen3-Omni-30B-A3B-Instruct (2025-09-20, 35.26B including audio/vision towers, HF license tag "other") use the same 48-layer text backbone — [Qwen3-VL config](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/config.json), [Qwen3-Omni config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)
- Qwen3.5-35B-A3B (2026-02-24, 35.95B, Apache-2.0) and Qwen3.6-35B-A3B (2026-04-15, 35.95B, Apache-2.0) share the `qwen3_5_moe` architecture — [HF API Qwen3.5-35B-A3B](https://huggingface.co/api/models/Qwen/Qwen3.5-35B-A3B), [HF API Qwen3.6-35B-A3B](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B)
- Qwen-AgentWorld-35B-A3B (2026-06-22, 34.66B, Apache-2.0, `qwen3_5_moe`) is a "language world model": it simulates environment observations (MCP, search, terminal, SWE, Android, web, OS) and is not a general chat or agent model — [model card](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
- Qwen3.8 open releases: Qwen3.8-27B (2026-08-05, dense `qwen3_5`, 27.78B), Qwen3.8-Flash-Next (2026-08-24, `qwen4_exp`, 512 experts / top-10, 48 layers, 179.99B stored parameters per HF), and Qwen3.8-2.4T-A95B (2026-08). No small Qwen3.8 MoE exists — [Qwen HF listing](https://huggingface.co/Qwen), [Qwen3.8-Flash-Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
- A llama.cpp issue describes Qwen3.8-Flash-Next as "125B / 6B active". The HF safetensors total (180B) appears to include a 97.7 GiB per-layer n-gram embedding table described in the llama.cpp support PR. Either way it is far outside this rig's range — [llama.cpp #27980](https://github.com/ggml-org/llama.cpp/issues/27980), [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- Searching HF for "Qwen3.7" found only third-party repos that name a "Qwen3.7-Flash" teacher or a "Qwen3.7-plus" upload, and no Qwen-org weights — [HF search Qwen3.7](https://huggingface.co/models?search=Qwen3.7)
- Third-party models on the Qwen3.5/3.6 MoE architecture: Kwaipilot KAT-Coder-V2.5-Dev (2026-07-23, 34.66B, "35B total, 3B activated", Apache-2.0, coding) — [card](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev). openbmb MiniCPM-V-4.7-35B-A3B (2026-10-07, 35.21B, Apache-2.0, vision-language with a `qwen3_5_moe_text` backbone) — [config](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json)
- Alibaba-NLP Tongyi-DeepResearch-30B-A3B (2025-09-16, `qwen3_moe`, 30.53B, Apache-2.0, deep-research agent) — [HF API](https://huggingface.co/api/models/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- inclusionAI GroveMoE-Inst (2025-08-18, 32.95B, Apache-2.0, a Qwen3-30B-A3B-shaped backbone with extra "adjugate" experts) — [HF API](https://huggingface.co/api/models/inclusionAI/GroveMoE-Inst)

**OpenAI, Z.ai, NVIDIA**
- openai/gpt-oss-20b (2025-08-04, 20.91B, Apache-2.0; experts ship in MXFP4). gpt-oss-safeguard-20b (2025-09) is a safety-classifier variant — [HF API](https://huggingface.co/api/models/openai/gpt-oss-20b)
- zai-org GLM-4.7-Flash (2026-01-19, 31.22B, MIT, `glm4_moe_lite`) — [HF API](https://huggingface.co/api/models/zai-org/GLM-4.7-Flash). The newer GLM-5.3-Flash (2026-08-25) has 321B parameters (FP8), so "Flash" no longer means small — [HF API GLM-5.3-Flash](https://huggingface.co/api/models/zai-org/GLM-5.3-Flash)
- NVIDIA Nemotron 3 Nano 30B-A3B (2025-12-04, 31.58B, HF license tag "other") — [HF API](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- Nemotron 3 Nano successors with the identical `nemotron_h` 31.58B layout: Nemotron-Cascade-2-30B-A3B (2026-03-18, reasoning-focused), NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B (2026-04-01), Nemotron-3-Nano-Omni-30B-A3B-Reasoning (2026-04-20), and NVIDIA-Nemotron-3.5-Lightning-30B-A3B (HF repo 2026-08-01) — [NVIDIA HF listing](https://huggingface.co/nvidia), [Cascade-2 config](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B/blob/main/config.json), [Lightning config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json)
- Artificial Analysis dates Nemotron 3.5 Lightning to 2026-08-11 and describes it as 31.6B total / 3.6B active, a hybrid Mamba-Transformer like Nemotron 3 Nano, 1M context, text-only reasoning, licensed under OpenMDW-1.1 — [Artificial Analysis launch article](https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch)
- NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB (2026-10-07) is a Puzzle-compressed omni model (88 layers, 512 experts). It is stored in NVFP4 (20.9B U8 elements plus 1.44B BF16; I read this as roughly 43B parameters) and is above this range — [HF API](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)

**Other vendors**
- Baidu ERNIE-4.5-21B-A3B-PT (2025-06-28, 21.95B) and ERNIE-4.5-21B-A3B-Thinking (2025-09-08, 21.83B), both Apache-2.0. Baidu has no newer small text MoE on HF; ERNIE-4.5-VL-28B-A3B-Thinking (2025-11) is a VLM — [HF API PT](https://huggingface.co/api/models/baidu/ERNIE-4.5-21B-A3B-PT), [Baidu HF listing](https://huggingface.co/baidu)
- PowerInfer SmallThinker-21BA3B-Instruct (2025-07-26, 21.51B, Apache-2.0, "Activated Parameters 3B") — [card](https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct)
- LiquidAI: LFM2-8B-A1B (2025-10-07, 8.34B), LFM2-24B-A2B (2026-02-24, 23.84B), LFM2.5-8B-A1B (2026-05-28, 8.47B) and LFM2.5-24B-A2B (2026-10-12, 23.97B; "Active parameters: 2.45B", described as "the largest and final model in the LFM2.5 family"). HF license tag is "other" — [HF API LFM2.5-24B-A2B](https://huggingface.co/api/models/LiquidAI/LFM2.5-24B-A2B), [LFM2.5-24B-A2B card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- IBM: granite-4.0-h-small (2025-09-16, 32.21B) and granite-4.0-h-tiny (6.94B) are hybrid Mamba2/MoE, Apache-2.0. granite-4.1-30b (2026-04) and granite-4.2-30b (2026-08) are dense `GraniteForCausalLM` (64 layers, hidden 4096, no experts). The only newer Granite MoE is the research model granite-swash-3b-a600m (2026-07, 3.02B, 8K context) — [granite-4.0-h-small config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json), [granite-4.2-30b config](https://huggingface.co/ibm-granite/granite-4.2-30b/blob/main/config.json), [granite-swash config](https://huggingface.co/ibm-granite/granite-swash-3b-a600m/blob/main/config.json)
- inclusionAI: Ling-mini-2.0 and Ring-mini-2.0 (2025-09-08, 16.26B total, "1.43B activated", MIT) — [Ling-mini-2.0 card](https://huggingface.co/inclusionAI/Ling-mini-2.0). Ling-3.0-tiny (2026-08-10, "7.9B total parameters and only 1.3B activated", MIT, hybrid linear attention) — [card](https://huggingface.co/inclusionAI/Ling-3.0-tiny). LLaDA2.2-mini (2026-09-05, 16.26B, a diffusion LM on the same MoE shape) — [HF API](https://huggingface.co/api/models/inclusionAI/LLaDA2.2-mini). Ling-3.0-flash (127.5B) and Ling-3.1-flash (572B) are out of range — [inclusionAI listing](https://huggingface.co/inclusionAI)
- Moonshot Kimi-Linear-48B-A3B-Instruct (2025-10-30, 49.12B, MIT) and Moonlight-16B-A3B-Instruct (2025-02-22, 15.96B, MIT) — [HF API Kimi-Linear](https://huggingface.co/api/models/moonshotai/Kimi-Linear-48B-A3B-Instruct), [HF API Moonlight](https://huggingface.co/api/models/moonshotai/Moonlight-16B-A3B-Instruct)
- Arcee Trinity-Mini (2025-12-01, 26.12B, "26B, 3B active", HF license tag "other") and Trinity-Nano-Preview (2025-12-01, 6.12B). Trinity-Large is out of range — [Trinity-Mini card](https://huggingface.co/arcee-ai/Trinity-Mini)
- Google Gemma 4 26B-A4B-it (2026-03-11, 25.81B, HF license tag apache-2.0, multimodal), with QAT Q4_0 GGUF and unquantized-QAT variants (2026-04/05) and a separate "-assistant" draft model. diffusiongemma-26B-A4B-it (2026-06) is a diffusion variant — [HF API](https://huggingface.co/api/models/google/gemma-4-26B-A4B-it), [Google HF listing](https://huggingface.co/google)
- Microsoft Phi-mini-MoE-instruct (2025-06-23, 7.65B) and Phi-tiny-MoE-instruct (3.76B), both MIT with 4K context. Phi-3.5-MoE-instruct (2024-08, 41.87B) is older — [Phi-mini-MoE config](https://huggingface.co/microsoft/Phi-mini-MoE-instruct/blob/main/config.json)
- AllenAI OLMoE-1B-7B-0125-Instruct (2025-01-27, 6.92B, Apache-2.0). AllenAI's 2026 MoE uploads (StdMoE_1b14b, EMO) are research checkpoints — [HF API](https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0125-Instruct)
- DeepSeek-V2-Lite-Chat (2024-05, 15.71B). DeepSeek's 2026 "Flash" models are about 291B (V4-Flash) — [HF API V2-Lite](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V2-Lite-Chat), [HF API V4-Flash](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash)
- Tencent: Hy-MT2-30B-A3B (2026-05-11, translation) and Hy-Memory-30B-A3B (2026-10-15, a "long-term-memory writer model" for Mem0/MemOS-style frameworks), both 30.06B, `hy_v3`, Apache-2.0. Hunyuan-A13B (80B) and Hy3 (299B) are out of range — [Hy-Memory card](https://huggingface.co/tencent/Hy-Memory-30B-A3B), [HF API Hy-MT2](https://huggingface.co/api/models/tencent/Hy-MT2-30B-A3B)
- Kakao kanana-2-30b-a3b instruct/thinking (2025-12; "-2601" refresh 2026-01-14), 30.67B, DeepSeek-V3-style architecture, HF license tag "other", Korean-focused — [HF API](https://huggingface.co/api/models/kakaocorp/kanana-2-30b-a3b-instruct-2601)
- Cohere North-Mini-Code-1.0 (2026-06-05, 30.48B, "30B total; 3B active", Apache-2.0, coding and terminal agent) — [card](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)
- JD JoyAI-LLM-Flash (2026-02-14, 49.29B, DeepSeek-V3-style, no license tag on HF) is slightly above range — [HF API](https://huggingface.co/api/models/jdopensource/JoyAI-LLM-Flash)
- Infinigence Megrez2-3x7B-A3B (2025-09-12, 7.47B stored, with experts shared across layers; Apache-2.0) — [HF API](https://huggingface.co/api/models/Infinigence/Megrez2-3x7B-A3B)
- Mistral has no MoE at or under 40B: Mistral-Small-4-119B-2603 is a 119B MoE (128 experts, top-4), Leanstral-1.5 is 119B-A6B, and Ministral 3 (3/8/14B) is dense — [Mistral-Small-4 config](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603/blob/main/config.json), [Mistral HF listing](https://huggingface.co/mistralai)
- ByteDance-Seed released no small open MoE LLM (Seed-OSS-36B is dense). Meituan LongCat-Flash-Lite (69.1B), Xiaomi MiMo-V2.5/V2.6 (310.8B), StepFun Step-3.7-Flash (201.4B) and AI21 Jamba2-Mini (51.6B) are all out of range — [ByteDance-Seed listing](https://huggingface.co/ByteDance-Seed), [HF API LongCat-Flash-Lite](https://huggingface.co/api/models/meituan-longcat/LongCat-Flash-Lite), [HF API MiMo-V2.5](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.5), [HF API Step-3.7-Flash](https://huggingface.co/api/models/stepfun-ai/Step-3.7-Flash), [HF API Jamba2-Mini](https://huggingface.co/api/models/ai21labs/AI21-Jamba2-Mini)

### Inferences
- In practice the field has converged on a "30-36B total / ~3B active" class (Qwen3/3.5/3.6, Nemotron-H, GLM-4.7-Flash, North-Mini-Code, Trinity-Mini, Hy, kanana, Gemma 4 at 3.8B active). At Q4 these produce 14-25 GB files, which fit both the ~26 GB of free disk and the 54 GB VRAM with plenty of room for KV cache.
- A second cluster is "small active" models (Ling-3.0-tiny 1.3B, Ling-mini-2.0 1.4B, LFM2/2.5-8B-A1B, Trinity-Nano, LFM2.5-24B-A2B at 2.45B). These are mainly useful here as speed probes, or later as throughput workers, since several copies fit at once.
- Qwen3.8-27B, which has already run on these cards, is dense and its MoE siblings are 180B+, so the closest MoE to it is Qwen3.6-35B-A3B. It belongs to the same Gated DeltaNet family (Qwen3.8-27B is `qwen3_5`, Qwen3.6-35B-A3B is `qwen3_5_moe`).

### Gaps
- I could not confirm when "Qwen3.7" models were released or whether they exist as open weights; there is no Qwen-org repo. If Qwen3.7 small MoEs were API-only, that would explain the absence.
- Release dates are HF repo-creation dates; for most models I did not cross-check them against official blog dates (Nemotron 3.5 Lightning differs by about 10 days: repo 2026-08-01, AA launch 2026-08-11).
- I did not find a tech-report-confirmed active-parameter figure for Hy-MT2/Hy-Memory, kanana-2, North-Mini-Code or JoyAI beyond the "A3B" naming and card text.
- Exact license names for HF tag "other" (LFM, Nemotron 3 Nano, Trinity, kanana, DeepSeek-V2-Lite) were not checked.

## Architecture details: experts, layers, attention type, context, KV cache and MTP

### Takeaway
Attention design varies far more than MoE shape, and it determines KV-cache cost. The Qwen3.5/3.6 family uses Gated DeltaNet linear attention in 3 of every 4 layers, Nemotron-H and Granite-H use Mamba2, Kimi-Linear and Ling-3.0 use KDA (Kimi Delta Attention), LFM2 uses short convolutions, and Gemma 4, gpt-oss, Trinity and North use sliding-window attention. As a result, f16 KV cache per token ranges from about 6 KiB (Nemotron-H, Ling-3.0-tiny) to 96-128 KiB (Qwen3-30B-A3B, Hy, Phi, OLMoE). Context length is not the binding constraint on 54 GB for any 3B-active candidate. MTP heads ship with Qwen3.5/3.6, GLM-4.7-Flash, Nemotron 3.5 Lightning, ERNIE-4.5-PT and JoyAI.

### Cited Findings
Each row below is read directly from the linked config.json (fetched 2026-10-22). "L" = layers, "h" = hidden size, "E/k/shared" = routed experts / experts per token / shared experts, "moe_ffn" = per-expert intermediate size.

| Model | L | h | E / k / shared (moe_ffn) | Attention (q heads / KV heads / head dim) | Max context (config) | MTP layers | Source |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B family (2507, Coder, VL, Omni text, Tongyi-DR) | 48 | 2048 | 128 / 8 / 0 (768) | full GQA 32/4/128 | 262,144 (Omni thinker 65,536; Tongyi 131,072) | 0 | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json) |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B, AgentWorld, KAT-Coder-V2.5-Dev, MiniCPM-V-4.7 | 40 (30 Gated DeltaNet + 10 full, interval 4) | 2048 | 256 / 8 / 1 (512; shared 512) | full layers: gated GQA 16/2/256; GDN: 16 key heads x128, 32 value heads x128 | 262,144 | 1 (KAT-Coder: 0) | [Qwen3.6 config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json), [KAT config](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/blob/main/config.json) |
| gpt-oss-20b | 24 (12 sliding-128 + 12 full, alternating) | 2880 | 32 / 4 / 0 (2880) | GQA 64/8/64, attention sinks | 131,072 (YaRN x32 from 4,096) | 0 | [config](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) |
| GLM-4.7-Flash | 47 (first dense) | 2048 | 64 / 4 / 1 (1536) | MLA, 20 heads, q_lora 768, kv_lora 512, qk 192+64 rope, v 256 | 202,752 | 1 | [config](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json) |
| Nemotron 3 Nano / Cascade-2 / Elastic / 3.5 Lightning | 52 = 23 Mamba2 + 23 MoE + 6 attention | 2688 | 128 / 6 / 1 (1856, relu2 non-gated; shared 3712) | GQA 32/2/128 on 6 layers; Mamba2 64 heads x 64, d_state 128 | 262,144 (Lightning card: up to 1M) | 0 (Lightning: 1) | [Nano config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json), [Lightning config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json) |
| ERNIE-4.5-21B-A3B (PT/Thinking) | 28 | 2560 | 64 / 6 / 2 (1536) | GQA 20/4/128 | 131,072 | 1 (PT), 0 (Thinking) | [config](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/blob/main/config.json) |
| SmallThinker-21B-A3B | 52 | 2560 | 64 / 6 / 0 (768) | GQA 28/4/128, per-layer sliding window 4096 | 16,384 | 0 | [config](https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct/blob/main/config.json) |
| LFM2-8B-A1B / LFM2.5-8B-A1B | 24 = 18 conv + 6 attention | 2048 | 32 / 4 / 0 (1792), first 2 layers dense | GQA 32/8/64 | 128,000 | 0 | [config](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/main/config.json) |
| LFM2-24B-A2B / LFM2.5-24B-A2B | 40 = 30 conv + 10 attention | 2048 | 64 / 4 / 0 (1536), 2 dense layers | GQA 32/8/64 | 128,000 / 131,072 | 0 | [config](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json) |
| Granite-4.0-H-Small | 40 = 36 Mamba2 + 4 attention (NoPE) | 4096 | 72 / 10 / shared (768; shared 1536) | GQA 32/8/128; Mamba2 128 heads x 64, d_state 128 | 131,072 | 0 | [config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) |
| Granite-4.0-H-Tiny | 40 = 36 Mamba2 + 4 attention | 1536 | 64 / 6 (512) | GQA 12/4 | 131,072 | 0 | [config](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/main/config.json) |
| Ling-mini-2.0 / Ring-mini-2.0 | 20 (first dense) | 2048 | 256 / 8 / 1 (512) | GQA 16/4/128 | 32,768 | 0 | [config](https://huggingface.co/inclusionAI/Ling-mini-2.0/blob/main/config.json) |
| Ling-3.0-tiny | 24, `bailing_hybrid`, layer_group_size 4 (KDA linear attention + MLA) | 1536 | 128 / 8 / 1 (512) | MLA 16 heads, q_lora 256, kv_lora 512 | 131,072 | 0 | [config](https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/main/config.json) |
| Kimi-Linear-48B-A3B | 27 = 20 KDA + 7 MLA (full layers 4,8,...,24,27) | 2304 | 256 / 8 / 1 (1024) | MLA 32 heads, kv_lora 512; KDA 32 heads x128 | 1,048,576 (model_max_length) | 0 | [config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json) |
| Trinity-Mini | 32 (24 sliding-2048 + 8 global), 2 dense | 2048 | 128 / 8 / 1 (1024) | GQA 32/4/128 | 131,072 | 0 | [config](https://huggingface.co/arcee-ai/Trinity-Mini/blob/main/config.json) |
| Trinity-Nano-Preview | 56 (42 sliding + 14 global) | 1024 | 128 / 8 / 1 (256) | GQA 8/2/128 | 131,072 | 0 | [config](https://huggingface.co/arcee-ai/Trinity-Nano-Preview/blob/main/config.json) |
| Gemma 4 26B-A4B | 30 = 25 sliding-1024 + 5 global | 2816 | 128 / 8 + dense MLP (704; dense 2112) | sliding GQA 16/8/256; global 16 q / 2 KV / head dim 512 with K=V (`attention_k_eq_v`) | 262,144 | 0 (separate "-assistant" draft model) | [config](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json) |
| Phi-mini-MoE | 32 | 4096 | 16 / 2 (960) | GQA 32/8/128, sliding 2047 | 4,096 | 0 | [config](https://huggingface.co/microsoft/Phi-mini-MoE-instruct/blob/main/config.json) |
| OLMoE-1B-7B-0125 | 16 | 2048 | 64 / 8 (1024) | MHA 16/16 | 4,096 | 0 | [config](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/main/config.json) |
| Moonlight-16B-A3B | 27 (first dense) | 2048 | 64 / 6 / 2 (1408) | MLA 16 heads, kv_lora 512 | 8,192 | 0 | [config](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct/blob/main/config.json) |
| DeepSeek-V2-Lite | 27 (first dense) | 2048 | 64 / 6 / 2 (1408) | MLA 16 heads, kv_lora 512 | 163,840 (YaRN) | 0 | [config](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json) |
| Hy-MT2 / Hy-Memory 30B-A3B (`hy_v3`) | 48 (first dense) | 2048 | 128 / 8 / 1 (768 in Hy-MT2 config; Hy-Memory config says 1536 with the same total) | GQA 32/4/128 | 262,144 | 0 | [Hy-Memory config](https://huggingface.co/tencent/Hy-Memory-30B-A3B/blob/main/config.json) |
| kanana-2-30b-a3b | 48 (first dense) | 2048 | 128 / 6 / 2 (768) | MLA 32 heads, kv_lora 512 | 32,768 | 0 | [config](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601/blob/main/config.json) |
| North-Mini-Code-1.0 | 49 (13 global + 36 sliding-4096) | 2048 | 128 / 8 / 0 (768) | GQA 32/4/128 | 500,000 in config; card says 256K and 64K max output | 0 | [config](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/blob/main/config.json) |
| JoyAI-LLM-Flash | 40 (first dense) | 2048 | 256 / 8 / 1 (768) | MLA 32 heads, kv_lora 512 | 131,072 | 1 | [config](https://huggingface.co/jdopensource/JoyAI-LLM-Flash/blob/main/config.json) |
| Megrez2-3x7B-A3B | 31 | 2048 | 64 / 6 / 4 (1408) | GQA 16/4 | 163,840 | 0 | [config](https://huggingface.co/Infinigence/Megrez2-3x7B-A3B/blob/main/config.json) |
| Phi-3.5-MoE | 32 | 4096 | 16 / 2 (6400) | GQA 32/8/128 | 131,072 | 0 | [config](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/blob/main/config.json) |

- A llama.cpp contributor independently confirms the KV arithmetic for two of these: Gemma-4-26B-A4B and the Qwen 35B-A3B "both write an identical 20,480 bytes of KV per context token — Gemma across 5 full-attention layers of 30, Qwen across 10 of 40" — [llama.cpp PR #28450](https://github.com/ggml-org/llama.cpp/pull/28450)
- Nemotron 3.5 Lightning "includes Multi-Token Prediction (MTP) layers" and ships DSpark and DFlash external draft models — [Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- Vendor-stated active parameters: gpt-oss-20b about 3.6B (LFM2.5-8B card lists "gpt-oss-20b 21B/3.6B") — [LFM2.5-8B-A1B card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B). LFM2.5-24B-A2B 2.45B — [card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). Ling-3.0-tiny 1.3B — [card](https://huggingface.co/inclusionAI/Ling-3.0-tiny). Ling-mini-2.0 1.43B (789M non-embedding) — [card](https://huggingface.co/inclusionAI/Ling-mini-2.0). Nemotron 3.5 Lightning 3.6B — [AA](https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch). Trinity-Mini 3B with "128 total, 8 active, 1 shared" experts — [card](https://huggingface.co/arcee-ai/Trinity-Mini)

**Computed active parameters and KV cache** (my arithmetic from the configs above). "Always-on" covers attention, SSM or conv layers, shared experts, dense layers, routers and the LM head; it is read every token regardless of routing. KV assumes llama.cpp stores only the full-attention layers per token, caps sliding-window layers at their window, and stores MLA as the compressed latent (kv_lora_rank 512 + rope 64 = 576 values per layer). Recurrent and sliding-window state is a fixed per-sequence cost and is listed separately.

| Model | computed active (B) | of which routed experts | always-on (attn/SSM/shared/lm_head) | KV f16 KiB/token | KV q8_0 KiB/token | KV f16 @32K (GiB) | fixed state/SWA (MiB) | notes |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (2507/Coder/VL/Omni text, Tongyi-DR) | 3.04 | 1.81 | 1.23 | 96.0 | 51.0 | 3.00 | 0 |  |
| Qwen3.5/3.6-35B-A3B (+AgentWorld, KAT-Coder-V2.5-Dev, MiniCPM-V-4.7 text) | 2.95 | 1.01 | 1.94 | 20.0 | 10.6 | 0.62 | 60 |  |
| gpt-oss-20b | 3.61 | 2.39 | 1.22 | 24.0 | 12.8 | 0.75 | 3 |  |
| GLM-4.7-Flash | 3.58 | 1.74 | 1.84 | 52.9 | 28.1 | 1.65 | 0 |  |
| Nemotron-H 30B-A3B (3 Nano / Cascade-2 / 3.5 Lightning) | 3.23 | 1.38 | 1.85 | 6.0 | 3.2 | 0.19 | 46 | 23 Mamba2, 23 MoE, 6 attn layers |
| ERNIE-4.5-21B-A3B | 3.35 | 1.91 | 1.44 | 56.0 | 29.8 | 1.75 | 0 |  |
| SmallThinker-21B-A3B | 3.33 | 1.84 | 1.49 | 104.0 | 55.2 | 3.25 | 0 | some layers sliding-window 4096; 16K ctx |
| LFM2-8B-A1B | 1.56 | 0.97 | 0.59 | 12.0 | 6.4 | 0.38 | 0 |  |
| LFM2.5-8B-A1B | 1.69 | 0.97 | 0.72 | 12.0 | 6.4 | 0.38 | 0 |  |
| LFM2-24B-A2B | 2.33 | 1.43 | 0.89 | 20.0 | 10.6 | 0.62 | 0 |  |
| LFM2.5-24B-A2B | 2.45 | 1.43 | 1.02 | 20.0 | 10.6 | 0.62 | 0 |  |
| Granite-4.0-H-Small (32B-A9B) | 8.80 | 3.77 | 5.03 | 16.0 | 8.5 | 0.50 | 144 |  |
| Ling-mini-2.0 / Ring-mini-2.0 | 1.11 | 0.48 | 0.63 | 40.0 | 21.2 | 1.25 | 0 |  |
| Ling-3.0-tiny (7.9B-A1.3B) | 1.14 | 0.43 | 0.71 | 6.8 | 3.6 | 0.21 | 18 | KDA/MLA split 18/6 assumed from layer_group_size=4 |
| Kimi-Linear-48B-A3B | 3.11 | 1.47 | 1.63 | 7.9 | 4.2 | 0.25 | 40 |  |
| Trinity-Mini (26B-A3B) | 3.06 | 1.51 | 1.55 | 16.0 | 8.5 | 0.50 | 96 |  |
| Trinity-Nano-Preview (6B-A1B) | 0.82 | 0.34 | 0.48 | 14.0 | 7.4 | 0.44 | 84 |  |
| Gemma-4-26B-A4B | 3.82 | 1.43 | 2.39 | 20.0 | 10.6 | 0.62 | 200 | lm_head tied to 262k-vocab embedding |
| Phi-mini-MoE (7.6B-A2.4B) | 2.23 | 0.75 | 1.48 | 128.0 | 68.0 | 4.00 | 0 | 4K ctx |
| Phi-3.5-MoE (42B-A6.6B) | 6.51 | 5.03 | 1.48 | 128.0 | 68.0 | 4.00 | 0 |  |
| OLMoE-1B-7B-0125 | 1.18 | 0.81 | 0.37 | 128.0 | 68.0 | 4.00 | 0 | 4K ctx |
| Moonlight-16B-A3B | 2.58 | 1.35 | 1.23 | 30.4 | 16.1 | 0.95 | 0 | 8K ctx |
| DeepSeek-V2-Lite (16B-A2.4B) | 2.45 | 1.35 | 1.10 | 30.4 | 16.1 | 0.95 | 0 |  |
| kanana-2-30b-a3b | 3.35 | 1.33 | 2.02 | 54.0 | 28.7 | 1.69 | 0 | 32K ctx |
| Hy-MT2-30B-A3B / Hy-Memory-30B-A3B (hy_v3) | 3.20 | 1.77 | 1.43 | 96.0 | 51.0 | 3.00 | 0 | shared-expert size assumed 768 |
| North-Mini-Code-1.0 (30B-A3B) | 3.32 | 1.81 | 1.51 | 26.0 | 13.8 | 0.81 | 288 | first dense FFN size assumed 6144 |
| JoyAI-LLM-Flash (49B-A3B) | 3.04 | 1.47 | 1.57 | 45.0 | 23.9 | 1.41 | 0 |  |

Cross-check of the formulas against reported totals: Qwen3-30B-A3B computes to 30.5B total (HF: 30.53B), gpt-oss-20b to 3.61B active (vendor: 3.6B), LFM2.5-24B to 2.45B active (vendor: 2.45B), Trinity-Mini to 26.1B total (HF: 26.12B) and Granite-H-Small to 8.8B active (IBM: "9B active").

### Inferences
- Treat the always-on share as a first-class number. In Qwen3.5/3.6-35B-A3B, about 1.94B of the ~2.95B weights read per token are always-on (the 248K-vocab LM head alone is 0.51B), and Gemma 4's tied 262K-vocab head is 0.74B of 3.82B. Quantization schemes that keep attention and output layers at high precision (unsloth UD, Q4_K_M with Q6_K output) therefore raise per-token bytes more than the average bits-per-weight suggests. The "pessimistic" column in the speed section bounds this.
- At 32K context with f16 KV, every 3B-active candidate needs 0.2-3.3 GB of KV. That is under one P106's worth of memory, so long-context use (128K+) is realistic for Qwen3.5/3.6 (2.5 GB at 128K), Nemotron-H (0.75 GB at 128K) and Kimi-Linear (1 GB at 128K). Qwen3-30B-A3B at 128K needs 12 GB f16 or 6.4 GB q8_0.
- Gemma 4's head-dimension-512 global layers and the MLA models (GLM-4.7-Flash, kanana, Moonlight, V2-Lite, JoyAI, and the MLA layers in Kimi-Linear and Ling-3.0) hit attention-kernel shapes that are weak on Pascal (see the llama.cpp section).
- My Ling-3.0-tiny figures assume 18 KDA + 6 MLA layers, inferred from `layer_group_size: 4`. The computed 1.14B active is slightly below the card's 1.3B, which suggests a larger KDA or gate share than I assumed.

### Gaps
- MLA KV-cache layout in llama.cpp (latent-only vs separate V) was not verified against current source; it is a claim to test, readable from llama-server's KV-cache size log line.
- I did not read the Gemma 4 `-assistant` draft model's architecture, or whether llama.cpp's Gemma 4 MTP path works with the 26B-A4B (issue #30698 reports an abort with `--mmproj` plus MTP) — [llama.cpp #30698](https://github.com/ggml-org/llama.cpp/issues/30698)
- Hy-Memory's config reports moe_intermediate_size 1536 while Hy-MT2 reports 768 with an identical parameter total. I used 768; this is unresolved.

## Headline benchmark scores (model cards and independent leaderboards)

### Takeaway
Among ~3B-active models, the strongest published coding and agentic numbers belong to the Qwen3.6-35B-A3B lineage: vendor-reported SWE-bench Verified 73.4 and Terminal-Bench 2.0 51.5, with KAT-Coder-V2.5-Dev claiming 69.4 under a different harness. Next come Nemotron Cascade-2 and 3.5 Lightning, GLM-4.7-Flash, Gemma 4 26B-A4B and LFM2.5-24B-A2B. Artificial Analysis also ranked Qwen3.6-35B-A3B first among ~3B-active open models in April 2026. However, AA rescaled its index at least twice in 2026 (43, then 32, then 18 for the same model), so AA numbers are only comparable within one snapshot. Vendor tables use different harnesses and disagree by up to about 40 points on the same model (Gemma 4 26B-A4B SWE-bench Verified: 17.4 per Qwen, 35.8 per Kwaipilot, 57.40 per NVIDIA).

### Cited Findings
**Artificial Analysis Intelligence Index (independent; values are NOT comparable across dates)**
- About April 2026 (AA post on Qwen3.6): Qwen3.6 35B A3B (Reasoning) 43, "6 points ahead of Qwen3.5 35B A3B (37) and 13 points ahead of GLM-4.7-Flash (30)", with Gemma 4 26B A4B 31 and Nemotron Cascade 2 30B A3B 28. Non-reasoning: Qwen3.6 35B A3B 32, Qwen3.5 35B A3B 31. AA notes Qwen3.6 35B A3B used ~143M output tokens to run the index (~3.7x Gemma 4 31B) — [Artificial Analysis on X](https://x.com/ArtificialAnlys/status/2049881951260283097)
- 2026-08-11 (AA Lightning article): Nemotron 3.5 Lightning 24 (NVFP4 also 24), Nemotron 3 Nano 15, Qwen3.6 35B A3B 32, gpt-oss-120b 24. Time per index task: Lightning ~0.5 min vs Qwen3.6 35B A3B ~3.5 min. Terminal-Bench v2.1: Lightning 24% vs Nemotron 3 Nano 7% — [AA article](https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch)
- Current AA model page (2026-10-22): Qwen3.6 35B A3B scores 18 on "Artificial Analysis Intelligence Index v4.3.2" (median of comparable models: 8) — [AA Qwen3.6 35B A3B page](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)
- A search-engine summary of current AA pages reported Gemma 4 26B A4B at 17 (marked as an estimate) and Nemotron 3.5 Lightning at 13. I did not open those pages myself; treat as a claim to verify — [AA Gemma 4 26B A4B page](https://artificialanalysis.ai/models/gemma-4-26b-a4b), [AA Nemotron 3.5 Lightning page](https://artificialanalysis.ai/models/nemotron-3-5-lightning)
- Ling-3.0-tiny's card claims 25 on "Artificial Analysis Intelligence Index v4.1.1" and 16 on the AA Agentic Index, with >160 tok/s output in AA testing — [Ling-3.0-tiny card](https://huggingface.co/inclusionAI/Ling-3.0-tiny)

**Vendor model-card numbers (self-reported; harnesses differ)**
- Qwen3.6-35B-A3B: SWE-bench Verified 73.4, SWE-bench Multilingual 67.2, SWE-bench Pro 49.5, Terminal-Bench 2.0 51.5, MMLU-Pro 85.2, GPQA 86.0, LiveCodeBench v6 80.4, AIME26 92.7, HLE 21.4, TAU3-Bench 67.2. The same table lists Qwen3.5-35B-A3B at 70.0 / 60.3 / 44.6 / 40.5 and Gemma4-26BA4B at 17.4 / 17.3 / 13.8 / 34.2 (internal scaffold) — [Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- Qwen3.5-35B-A3B: MMLU-Pro 85.3, GPQA Diamond 84.2, SWE-bench Verified 69.2, Terminal Bench 2 40.5, LiveCodeBench v6 74.6, BFCL-V4 67.3, TAU2-Bench 81.2, IFEval 91.9, HLE w/ CoT 22.4 — [Qwen3.5-35B-A3B card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- KAT-Coder-V2.5-Dev, under a claude_code harness that it applies to all columns: SWE-bench Verified 69.40 (Qwen3.6-35BA3B 64.40, Qwen3.5-35BA3B 58.60, Gemma4-26BA4B 35.80, Qwen3-Coder-30B 31.80), SWE-bench Pro 45.96, Terminal-Bench 2.1 41.02, SciCode 44.20. It notes an "approximate 10 pp gap" from Qwen's official Qwen3.6 numbers — [KAT-Coder-V2.5-Dev card](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev)
- Qwen3-30B-A3B-Instruct-2507: MMLU-Pro 78.4, GPQA 70.4, AIME25 61.3, LiveCodeBench v6 43.2, Aider-Polyglot 35.6, LiveBench 20241125 69.0, Arena-Hard v2 69.0, BFCL-v3 65.1, TAU2-Retail 57.0, IFEval 84.7 — [card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- GLM-4.7-Flash, vs Qwen3-30B-A3B-Thinking-2507 and GPT-OSS-20B: AIME 25 91.6 (85.0, 91.7), GPQA 75.2 (73.4, 71.5), LCB v6 64.0 (66.0, 61.0), SWE-bench Verified 59.2 (22.0, 34.0), τ²-Bench 79.5 (49.0, 47.7), BrowseComp 42.8 (2.29, 28.3) — [GLM-4.7-Flash card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- Nemotron 3 Nano 30B-A3B: MMLU-Pro 78.3, AIME25 (no tools) 89.1, GPQA 73.0, LiveCodeBench 68.3, SWE-Bench (OpenHands) 38.8, TauBench V2 average 49.0, BFCL v4 53.8, Arena-Hard-V2 average 67.7, IFBench 71.5 — [card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- Nemotron 3.5 Lightning, columns Lightning / Qwen3.6-35B-A3B / Gemma 4 26B A4B / Nemotron 3 Nano / Nemotron 3 Super / GPT-OSS 20B: GPQA Diamond 75.44 / 83.40 / 79.61 / 74.05 / 78.60 / 71.46; SWE-bench Verified 51.56 / 70.12 / 57.40 / 34.08 / 63.08 / 52.44; Terminal-Bench 2.1 24.58 / 44.38 / 37.22 / 8.29 / 39.61 / 15.17; IFBench (loose) 71.88 / 63.71 / 77.25 / 72.17 / 71.92 / 68.50 — [Lightning card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- Nemotron-Cascade-2-30B-A3B: AIME 2025 92.4, AIME 2026 90.9, LiveCodeBench v6 87.2, MMLU-Pro 79.8, GPQA-Diamond 76.1, ArenaHard v2 average 83.5, IFBench 82.9, SWE Verified (OpenHands) 50.2, and a claimed 35 points on IMO 2025 (gold-medal level). The card says it "does not currently support OpenCode" — [Cascade-2 card](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)
- Gemma 4 26B A4B (Google card): MMLU Pro 82.6%, AIME 2026 no tools 88.3%, LiveCodeBench v6 77.1%, Codeforces ELO 1718, GPQA Diamond 82.3%, Tau2 (average over 3) 68.2%, BigBench Extra Hard 64.8% — [Gemma 4 26B A4B card](https://huggingface.co/google/gemma-4-26B-A4B-it)
- LFM2.5-24B-A2B, columns LFM2.5-24B / Qwen3.6-35B-A3B / Gemma-4-26B-A4B / Nemotron-3.5-Lightning: GPQA Diamond 74.42 / 83.69 / 74.90 / 72.42; SWE-Bench Verified (OpenCode) 51.00 / 66.20 / 25.80 / 53.40; BFCLv4 65.01 / 65.00 / 57.09 / 53.78; IFBench 74.69 / 64.21 / 47.44 / 71.67; AutomationBench 27.69 / 8.72 / 5.46 / 2.61. LiveClaw is claimed at 48.09 vs 38.19 for Qwen3.6 — [LFM2.5-24B-A2B card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- LFM2.5-8B-A1B vs LFM2-8B-A1B: IFEval 91.84 vs 79.44, IFBench 56.47 vs 26.00, AIME25 42.53 vs 20.00, BFCLv4 48.50 vs 25.52, Tau² Telecom 88.07 vs 13.60 — [LFM2.5-8B-A1B card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- kanana-2-30b-a3b-thinking-2601 vs Qwen3-30B-A3B-Thinking-2507: MMLU-Pro 74.2 vs 80.8, GPQA Diamond 57.8 vs 70.6, LiveCodeBench 58.8 vs 68.3, BFCL-v3 Live 75.9 vs 82.9 — [kanana-2 card](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601)
- Hy-Memory-30B-A3B scores 55.49 as a memory writer (macro-average over Mem0/Hy-Memory/MemOS/EverMemOS harnesses), vs Qwen3.6-35B-A3B 50.37. This is a narrow specialist benchmark — [Hy-Memory card](https://huggingface.co/tencent/Hy-Memory-30B-A3B)
- North-Mini-Code-1.0 reports SWE-Bench Verified/Pro and Terminal-Bench v2/Hard (SWE-Agent v1.1.0 harness, 3 seeds, T=1.0), but the numbers are in images that I could not extract — [North-Mini-Code card](https://huggingface.co/CohereLabs/North-Mini-Code-1.0)

### Inferences
- For "chat/coding/agent" single-user use, the candidates with the strongest self-reported agentic-coding results are Qwen3.6-35B-A3B and its derivative KAT-Coder-V2.5-Dev, followed by GLM-4.7-Flash, Nemotron 3.5 Lightning and LFM2.5-24B-A2B. Gemma 4 26B-A4B's agentic-coding numbers vary wildly by harness (17.4 to 57.4 on SWE-bench Verified), which suggests sensitivity to the tool-call format.
- Token efficiency matters a lot on a ~20-40 tok/s rig. AA reports Qwen3.6-35B-A3B at ~3.5 min per index task and ~143M output tokens for the index, versus Nemotron 3.5 Lightning at ~0.5 min. A model that "thinks" less may give better wall-clock answers here even with a lower index score. This is a claim to test with real prompts on the rig.

### Gaps
- I found no LMArena, LiveBench or Aider leaderboard entries for the 2026 small MoEs (only Qwen3-30B-A3B-2507's own card quotes Aider-Polyglot 35.6 and LiveBench 69.0). Independent coding leaderboards for Qwen3.6-35B-A3B, Gemma 4 26B-A4B, Nemotron 3.5 Lightning and LFM2.5-24B were not retrieved.
- No benchmarks were extracted for Trinity-Mini, Granite-4.0-H-Small, ERNIE-4.5-21B-A3B-Thinking, SmallThinker, Kimi-Linear (chat) or North-Mini-Code.
- AA index values for GLM-4.7-Flash, gpt-oss-20b, LFM2.5-24B and North-Mini-Code on the current v4.3.2 scale were not retrieved.

## GGUF availability, file sizes per quant, and fit in 26 GB of free disk and 54 GB of VRAM

### Takeaway
Every main candidate has GGUFs from unsloth, bartowski and/or ggml-org, except Hy-MT2/Hy-Memory, kanana-2, JoyAI, Moonlight, SmallThinker-21B and Megrez2, for which I found none from the main quantizers. At Q4-class quants, all ~3B-active models are 14-25 GB, so they fit the rig's 26 GB of free disk and load into VRAM with large KV headroom. Q8_0 of the 30-37B models (32-38 GB) and anything for Kimi-Linear above Q3_K_M do not fit the disk. Those need either a network mount from a separate, more powerful PC on the same LAN (a "workstation") or deleting files. The Nemotron-H family is an anomaly: every bartowski or unsloth quant from IQ2_M to Q4_0 is about 18-20 GB.

### Cited Findings
- unsloth Qwen3.6-35B-A3B-GGUF: UD-IQ2_XXS 10.76, UD-IQ3_XXS 13.21, UD-Q3_K_M 16.60, UD-IQ4_XS 17.73, MXFP4_MOE 21.71, UD-Q4_K_M 22.13, UD-Q4_K_XL 22.36, UD-Q5_K_M 26.46, UD-Q6_K 29.31, Q8_0 36.90, BF16 69.38 GB — [tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main). The MTP variant is ~0.5-0.9 GB larger (UD-Q4_K_XL 22.85, Q8_0 37.80) — [unsloth MTP tree](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/main); ggml-org also publishes Q8_0 37.80 / BF16 71.07 MTP files — [ggml-org MTP tree](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF/tree/main). bartowski Qwen3.6: IQ3_XXS 15.76, Q4_K_M 22.29, Q8_0 37.81 — [tree](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/tree/main)
- unsloth Qwen3.5-35B-A3B-GGUF: UD-IQ3_XXS 13.08, Q3_K_M 16.36, MXFP4_MOE 21.59, Q4_K_M 22.02, UD-Q4_K_XL 22.24, Q5_K_M 26.25, Q6_K 28.85, Q8_0 36.90 — [tree](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/main)
- unsloth Qwen-AgentWorld-35B-A3B-GGUF: UD-Q4_K_XL 22.32, Q8_0 36.90 — [tree](https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF/tree/main). bartowski KAT-Coder-V2.5-Dev: IQ3_XXS 14.87, Q3_K_M 16.23, IQ4_XS 18.81, Q4_K_M 21.39, Q5_K_M 25.02, Q6_K 30.05, Q8_0 36.91 — [tree](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/tree/main). bartowski MiniCPM-V-4.7-35B-A3B: Q4_K_M 22.32, Q8_0 36.91 — [tree](https://huggingface.co/bartowski/MiniCPM-V-4.7-35B-A3B-GGUF/tree/main)
- unsloth Qwen3-30B-A3B-Instruct-2507-GGUF: UD-IQ3_XXS 12.91, Q3_K_M 14.71, IQ4_XS 16.38, UD-Q4_K_XL 17.69, Q4_K_M 18.56, Q5_K_M 21.73, Q6_K 25.09, Q8_0 32.48 — [tree](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/tree/main). Qwen3-Coder-30B-A3B has identical sizes — [tree](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/tree/main). Tongyi-DeepResearch (bartowski): Q4_K_M 18.63, Q8_0 32.48 — [tree](https://huggingface.co/bartowski/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-GGUF/tree/main)
- gpt-oss-20b: ggml-org MXFP4 12.11 GB — [tree](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/main). unsloth quants are all 11.5-12.1 GB (Q3_K_S 11.46 up to Q8_0 12.11, F16 13.79) because the experts stay MXFP4 — [tree](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main)
- unsloth GLM-4.7-Flash-GGUF: UD-IQ3_XXS 12.91, Q3_K_M 14.61, IQ4_XS 16.27, MXFP4_MOE 16.97, UD-Q4_K_XL 17.52, Q4_K_M 18.31, Q5_K_M 21.41, Q6_K 24.69, Q8_0 31.84 — [tree](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/tree/main). unsloth also hosts a pruned GLM-4.7-Flash-REAP-23B-A3B (Q4_K_M 14.11, Q8_0 24.46) — [tree](https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/tree/main)
- Nemotron 3.5 Lightning (bartowski): IQ2_M 18.85, IQ4_XS 18.92, Q4_0 19.06, IQ3_XXS 19.80, Q3_K_M 19.82, Q4_K_M 25.48, Q5_K_M 26.96, Q6_K 34.31, Q8_0 35.00 — [tree](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/main). unsloth: UD-IQ2_M 19.43, MXFP4_MOE 23.21, UD-Q4_K_XL 25.51, Q8_0 35.00 — [tree](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/main). ggml-org: Q4_0 18.90, Q8_0 33.59 — [tree](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/main). Nemotron 3 Nano and Cascade-2 show the same pattern (e.g. bartowski Nano IQ2_M 18.03, Q4_K_M 24.66, Q8_0 33.59) — [Nano tree](https://huggingface.co/bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF/tree/main), [Cascade-2 tree](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/tree/main)
- unsloth gemma-4-26B-A4B-it-GGUF: UD-IQ3_XXS 11.42, UD-Q3_K_M 12.73, UD-IQ4_XS 13.60, MXFP4_MOE 16.55, UD-Q4_K_M 16.95, UD-Q5_K_M 21.15, UD-Q6_K 23.17, Q8_0 26.86 — [tree](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main). The QAT UD-Q4_K_XL is 14.25 — [tree](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/tree/main). ggml-org Q4_0 is 14.62 — [tree](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/tree/main). Google publishes its own QAT q4_0 GGUF repo — [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
- unsloth ERNIE-4.5-21B-A3B-Thinking: Q3_K_M 10.52, Q4_K_M 13.33, Q5_K_M 15.59, Q6_K 17.97, Q8_0 23.21 — [tree](https://huggingface.co/unsloth/ERNIE-4.5-21B-A3B-Thinking-GGUF/tree/main)
- LiquidAI: unsloth LFM2.5-24B-A2B UD-IQ3_XXS 8.96, UD-Q4_K_XL 14.88, UD-Q6_K_XL 21.86, Q8_0 25.49 — [tree](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF/tree/main). ggml-org Q4_K_M 14.53 — [tree](https://huggingface.co/ggml-org/LFM2.5-24B-A2B-GGUF/tree/main). bartowski LFM2-24B-A2B Q4_K_M 14.44, Q8_0 25.35 — [tree](https://huggingface.co/bartowski/LiquidAI_LFM2-24B-A2B-GGUF/tree/main). unsloth LFM2.5-8B-A1B UD-Q4_K_XL 5.35, Q8_0 9.01 — [tree](https://huggingface.co/unsloth/LFM2.5-8B-A1B-GGUF/tree/main). unsloth LFM2-8B-A1B Q4_K_M 5.04, Q8_0 8.87 — [tree](https://huggingface.co/unsloth/LFM2-8B-A1B-GGUF/tree/main)
- unsloth granite-4.0-h-small: Q4_K_M 19.57, Q8_0 34.26 — [tree](https://huggingface.co/unsloth/granite-4.0-h-small-GGUF/tree/main). granite-4.0-h-tiny: Q4_K_M 4.25, Q8_0 7.39 — [tree](https://huggingface.co/unsloth/granite-4.0-h-tiny-GGUF/tree/main)
- bartowski Ling-mini-2.0: Q4_K_M 9.94, Q8_0 17.31 — [tree](https://huggingface.co/bartowski/inclusionAI_Ling-mini-2.0-GGUF/tree/main). bartowski Ling-3.0-tiny: Q4_K_M 4.92, Q8_0 8.41, quantized with llama.cpp "release b10472" — [tree](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF/tree/main)
- bartowski Kimi-Linear-48B-A3B-Instruct: IQ3_XXS 19.65, Q3_K_M 22.68, IQ4_XS 26.46, Q4_K_M 30.06, Q5_K_M 35.10, Q6_K 40.46, Q8_0 52.25 — [tree](https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF/tree/main)
- bartowski Trinity-Mini: Q4_K_M 15.94, Q6_K 21.52, Q8_0 27.79. Trinity-Nano-Preview: Q4_K_M 3.79, Q8_0 6.53 — [Trinity-Mini tree](https://huggingface.co/bartowski/arcee-ai_Trinity-Mini-GGUF/tree/main), [Trinity-Nano tree](https://huggingface.co/bartowski/arcee-ai_Trinity-Nano-Preview-GGUF/tree/main)
- unsloth North-Mini-Code-1.0: UD-IQ3_XXS 11.71, MXFP4_MOE 18.66, UD-Q4_K_M 19.20, UD-Q6_K 25.51, Q8_0 32.44 — [tree](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF/tree/main)
- bartowski OLMoE-1B-7B-0924-Instruct: Q4_K_M 4.21, Q8_0 7.36 — [tree](https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/tree/main)
- Searches of unsloth, bartowski, ggml-org and lmstudio-community found no GGUF repos for Hy-MT2-30B-A3B or Hy-Memory-30B-A3B (unsloth only has Hy-MT2 1.8B/7B), kanana-2-30b-a3b, JoyAI-LLM-Flash, Moonlight-16B-A3B, SmallThinker-21B-A3B, Megrez2 or Phi-mini-MoE — [HF search API, e.g. unsloth Hy-MT2](https://huggingface.co/models?author=unsloth&search=Hy-MT2)
- Rig constraints: 9 x ~6068 MiB usable = ~57.3 GB decimal (53.3 GiB) — [hardware spec](../../mining_rig.md); ~26 GB free on the rig's 64 GB SSD (51 % full), measured on the rig on 2026-10-22; a workstation on the same gigabit LAN with room for a model library

### Inferences
- Disk is the binding constraint, not VRAM. Files up to ~24 GB load with a safety margin inside the 26 GB free, and files of 24-26 GB fit only tightly. That covers Q4 of every 30-36B model and Q5 of most of them, Q8_0 of ERNIE-21B (23.2 GB) and Q6_K of Gemma 4 26B-A4B (23.2 GB); Q8_0 of LFM2.5-24B (25.5 GB) and LFM2-24B (25.4 GB) is tight. Q8_0 of the 30-36B models (31.8-37.8 GB) fits VRAM comfortably (weights plus 32K KV under 38 GB) but not the disk. Since llama.cpp mmaps the GGUF only to copy it into VRAM, a read-only NFS or SSHFS mount from the workstation should work. Gigabit (~110 MB/s) gives roughly 5-6 min to load 37 GB; this is a claim to test, including behaviour with only 4 GB host RAM.
- Nemotron-H inference: `moe_intermediate_size` 1856 is not a multiple of 256 (1856 = 7.25 x 256), so super-block quant types (k-quants, i-quants with 256-element blocks) cannot be applied along that dimension. llama-quantize then falls back to other types for those expert tensors, which is consistent with every sub-Q4 Nemotron quant being ~18-20 GB. Practical effect: the smallest useful Nemotron file is ~19 GB and "Q4_K_M" is 25.5 GB, near the disk limit.
- VRAM budget assumption used in the table: about 50 GB for weights plus KV, i.e. 57.3 GB minus roughly 0.5 GiB per GPU for CUDA context, the existing Xorg context and compute buffers. This needs measuring; per-GPU compute buffers on long contexts can be larger.
- Several small models can be resident at once (e.g. gpt-oss-20b MXFP4 on 3 GPUs plus Qwen3.6 UD-IQ4_XS on the other 6), which matters if throughput becomes a goal.

### Gaps
- I did not verify file sizes for mradermacher or community quants (e.g. Hy-Memory oQ quants, vultr Nemotron GGUFs), or the exact quant-type mix inside unsloth UD files.
- Whether llama.cpp loads cleanly from NFS/SSHFS with 4 GB host RAM (page-cache pressure with mmap, or `--no-mmap` needing a RAM bounce buffer) is untested.

## llama.cpp support status per architecture (October 2026) and Pascal (sm_61) kernel issues

### Takeaway
All main candidate architectures are in mainline llama.cpp as of October 2026. Gated DeltaNet (Qwen3.5/3.6), KDA (Kimi-Linear, Ling-3.0), Mamba2 (Nemotron-H, Granite-H), LFM2 conv, MLA and MXFP4 all have CUDA kernels, and MTP speculative decoding exists for Qwen3.6 and Nemotron 3.5 Lightning. The Pascal-specific risks are in attention and matmul tuning rather than missing ops. Flash-attention for MLA head sizes (576/512) is "expected" to be slow on Pascal, there are open sm_61 MMVQ-fusion and MMQ-tile tuning issues, and Gemma 4's head-dim-512 global layers have had FA problems on other backends. GDN on Pascal is directly evidenced: Qwen3.5-122B-A10B runs on 4x P40 at 29.5 tok/s. The rig itself has already run Qwen3.8-27B, which uses the same GDN layers.

### Cited Findings
- Qwen3.5 (`qwen3.5moe`) on Pascal: "On 4x Tesla P40 (sm_61) ... `qwen3.5moe` (Qwen3.5-122B-A10B, 122B / 10B active) on the same GPUs and the same build decodes at ~29.5 tok/s" (UD-Q3_K_XL, llama.cpp b10666, driver 580.126.09). The same issue reports Qwen3.8-Flash-Next decoding slower (18 tok/s) and suspects the recurrent step on sm_61 — [llama.cpp #27980](https://github.com/ggml-org/llama.cpp/issues/27980)
- The CUDA Gated DeltaNet kernel exists and is being optimized. A merged PR (2026-10-08) assigns four GDN state columns per warp, "Applies to all S_v ... KDA and non-KDA ... one-token, and snapshot (keep_rs) paths", with pp +5% on an RTX 4090 and tg unchanged — [llama.cpp PR #30087](https://github.com/ggml-org/llama.cpp/pull/30087). A SYCL-only bug reports GDN "token soup on long prompts (Qwen3.6, Qwen3.8, Nemotron)" — [llama.cpp #29183](https://github.com/ggml-org/llama.cpp/issues/29183)
- Qwen3.6 MTP: bartowski's card says MTP layers are included "now that support has been added with release b9180", used via `--spec-type draft-mtp` — [bartowski Qwen3.6 GGUF card](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF). unsloth recommends `--spec-type draft-mtp --spec-draft-n-max 2` — [unsloth Qwen3.6 MTP GGUF card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Open issues: MTP with Qwen3.6 27B outputs repeated "////" after long sessions — [llama.cpp #23577](https://github.com/ggml-org/llama.cpp/issues/23577); speculative decoding (draft-mtp/draft-dspark) greedy output diverges from vanilla — [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618)
- Tensor-parallel granularity was fixed for "Qwen 3.5/3.6 + 3 GPUs" (merged 2026-05-30) — [llama.cpp PR #23843](https://github.com/ggml-org/llama.cpp/pull/23843). Backend-agnostic tensor parallelism (experimental) was merged 2026-04-09 — [llama.cpp PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378)
- Nemotron-H: MTP support was extended (merged 2026-09-17) — [llama.cpp PR #29018](https://github.com/ggml-org/llama.cpp/pull/29018); recurrent-state rollback for Nemotron on CUDA (merged 2026-08-14) — [PR #26623](https://github.com/ggml-org/llama.cpp/pull/26623); SSM_SCAN state size 96 support (merged 2026-09-26) — [PR #28717](https://github.com/ggml-org/llama.cpp/pull/28717); redundant copies after SSM_SCAN removed (merged 2026-10-09) — [PR #29807](https://github.com/ggml-org/llama.cpp/pull/29807). A fixed bug: "Nemotron 3.5 Lightning Q4_0: zero KV head metadata misclassifies Attention blocks as SSM" — [#27718](https://github.com/ggml-org/llama.cpp/issues/27718). bartowski's Lightning GGUF uses "release b10362", includes MTP layers ("stored at Q4_0 in the imatrix quants") and is run with `--spec-type draft-mtp` — [bartowski Lightning card](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)
- GLM-4.7-Flash on Pascal (Tesla P40, sm_61): llama-bench reports the model as `deepseek2` (MLA path) with Q4_0 (15.78 GiB) at tg128 41.22 tok/s and pp8192 411.80 tok/s with FA off. With FA on, prefill speed "is halved". A maintainer replied: "It's expected that the performance on Pascal is bad, this is also the case for other models with a head size of 576/512" — [llama.cpp #19020](https://github.com/ggml-org/llama.cpp/issues/19020). A PR adding MTP for GLM-4.7-Flash was closed without merge — [PR #25564](https://github.com/ggml-org/llama.cpp/pull/25564). A later merged PR fixes "MTP context kv cache allocation for deepseek2, glm4moe, c…" (2026-09-11) — [PR #28630](https://github.com/ggml-org/llama.cpp/pull/28630)
- Gemma 4: the global-attention head size 512 has had FA dispatch and performance fixes (a SYCL tile-kernel fix merged 2026-09-22, noting gqa_ratio 8 at D=512) — [PR #28450](https://github.com/ggml-org/llama.cpp/pull/28450). An open FA crash in the gemma4-assistant MTP path ("Query head dimension mismatch") — [#29419](https://github.com/ggml-org/llama.cpp/issues/29419). ROCm prefill is slow at head size 512 — [#30838](https://github.com/ggml-org/llama.cpp/issues/30838). Gemma 4 with `--mmproj` and MTP aborts after a regression — [#30698](https://github.com/ggml-org/llama.cpp/issues/30698)
- Pascal matmul tuning: "fusion is default on for P40 ... the rule covers (cc <= 600), fusion on P40 stays on"; turning fusion off plus forced small-K "gave about 12% at batch 1" on Q2_K (open as of 2026-10-11) — [llama.cpp #30334](https://github.com/ggml-org/llama.cpp/issues/30334). The Pascal DP4A MMQ config uses I=64 for Q8_0 and "I=128 gives +12% prompt processing" on sm_61 (open) — [#30235](https://github.com/ggml-org/llama.cpp/issues/30235). MMQ was unblocked for MoE on sm_60 (P100), merged 2026-08-26 — [PR #26264](https://github.com/ggml-org/llama.cpp/pull/26264). "CUDA: improve cuBLAS compatibility with Legacy GPUs" (open) — [PR #30727](https://github.com/ggml-org/llama.cpp/pull/30727). "CUDA: fix crash for MTP decoding on pre-Pascal GPUs" (closed) — [PR #28781](https://github.com/ggml-org/llama.cpp/pull/28781)
- Other architecture support: BailingMoeV2 (Ling/Ring-mini-2.0) merged 2025-10-20 — [PR #16063](https://github.com/ggml-org/llama.cpp/pull/16063); BailingMoE3 (Ling-3.0) merged 2026-08-17 — [PR #26608](https://github.com/ggml-org/llama.cpp/pull/26608); Hy3 / `hy_v3` with MTP merged 2026-07-13 — [PR #25395](https://github.com/ggml-org/llama.cpp/pull/25395); a dedicated Cohere2MoE (North Code) chat parser merged 2026-06-14 — [PR #24615](https://github.com/ggml-org/llama.cpp/pull/24615); GroveMoE merged 2025-09-25 — [PR #15510](https://github.com/ggml-org/llama.cpp/pull/15510); lfm2/lfm2moe tensor split merged 2026-08-21 — [PR #26993](https://github.com/ggml-org/llama.cpp/pull/26993); a SmallThinker fix merged 2025-10-26 (so the arch exists) — [PR #16782](https://github.com/ggml-org/llama.cpp/pull/16782); Megrez-MoE support still in an unmerged PR — [PR #17141](https://github.com/ggml-org/llama.cpp/pull/17141); "meta : enable tensor split for KDA hybrid models" closed 2026-10 — [PR #30277](https://github.com/ggml-org/llama.cpp/pull/30277)
- Quantization dates as hints for minimum builds: Ling-3.0-tiny was quantized with b10472, Nemotron 3.5 Lightning with b10362, and LFM2.5-24B-A2B with "release b11606" (October 2026) — [bartowski Ling-3.0-tiny](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF), [bartowski Lightning](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF), [bartowski LFM2.5-24B](https://huggingface.co/bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF). llama.cpp moved to version 0.6.0 on 2026-10-05 — [PR #29997](https://github.com/ggml-org/llama.cpp/pull/29997)
- Pascal multi-GPU reference points with Qwen3.6-35B-A3B: on 2x Tesla P100 with llama.cpp, UD-Q4_K_XL (22.9 GB) and `-sm layer` gives 91-95 tok/s (a user report in another project's issue) — [Strata #3294](https://github.com/Niko1221/Strata/issues/3294). On 2x P100 with `-sm tensor` and q8_0 KV, master gave S_TG 67.82 tok/s (before an MMQ change that mainly affects prefill) — [PR #26264](https://github.com/ggml-org/llama.cpp/pull/26264)
- Qwen3.8-27B has already run on the rig. It is `qwen3_5` with 48 linear-attention (GDN) + 16 full-attention layers — [Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)

### Inferences
- Since Qwen3.8-27B already ran on these P106s, the GDN CUDA op compiles and runs on sm_61 in the rig's existing build (llama.cpp commit 85d5703a3, 2026-09-04). Qwen3.5/3.6-35B-A3B add only the MoE FFN (standard `build_moe_ffn`), so they are the lowest-risk "new architecture" to try. Qwen3-30B-A3B, a plain GQA transformer, is the lowest-risk overall.
- MLA models (GLM-4.7-Flash, kanana-2, Moonlight, DeepSeek-V2-Lite, JoyAI, and the MLA layers of Kimi-Linear and Ling-3.0) should probably run with `-fa off` on Pascal for prefill. In llama.cpp, quantized V cache normally requires FA, so these may be limited to f16 KV. MLA's latent KV is small anyway (~53 KiB/token for GLM-4.7-Flash). This is a claim to test.
- gpt-oss-20b on sm_61 depends on the MXFP4 MMVQ/MMQ path (dp4a-based int8 conversion, no FP4 hardware) and on attention-sink support in whichever FA kernel Pascal uses. I found no direct Pascal report, so it is a claim to test.
- Mamba2 SSM_SCAN and the LFM2 conv ops are plain FP32 CUDA kernels and should not need FP16 or tensor cores. I found no Pascal-specific report for Nemotron-H, Granite-H or LFM2; Pascal behaviour is a claim to test. Issue #27980 hints that recurrent-step cost on sm_61 can dominate decode for hybrid models, and with only 5 SMs per P106 this could be worse than on a P40 (30 SMs).
- Builds newer than the rig's 2026-09-04 build may be needed for LFM2.5-24B-A2B (quantized with b11606) and for the latest sm_61 tuning fixes. Older builds should load the `lfm2moe` architecture, but chat templates and parsers may lag.

### Gaps
- I did not find the PR that first added Kimi-Linear (KDA) model support or the initial CUDA GATED_DELTA_NET kernel. Support is inferred from bartowski Kimi-Linear GGUFs, the KDA path in PR #30087, and BailingMoE3 support.
- No Pascal-specific measurements were found for gpt-oss-20b (MXFP4 plus sinks), Gemma 4 (head-dim 512 FA), Nemotron-H (SSM scan), LFM2/2.5 or Kimi-Linear on llama.cpp.
- No llama.cpp reports were found from P106, P104 or other 5-SM mining cards; the only hits for "P104-100" are unrelated multi-GPU issues — [#22926](https://github.com/ggml-org/llama.cpp/issues/22926)

## Theoretical batch-1 decode ceiling per candidate on this rig

### Takeaway
With layer-split pipelining (one GPU active at a time), the ceiling is about 173 GB/s divided by the bytes read per token. For the ~3B-active class at Q4 this gives about 80-100 tok/s on a "uniform bits" estimate and 57-73 tok/s if attention, shared and LM-head weights are kept at about 8.5 bits. Granite-4.0-H-Small (8.8B active) tops out around 32 tok/s at Q4_K_M, while the 1-1.5B-active models reach 150-390 tok/s. Real Pascal llama.cpp results run at roughly 24-43% of theoretical ceilings, so a planning range for Qwen3.6-35B-A3B UD-Q4_K_XL on this rig is roughly 20-40 tok/s. That range is a claim to test.

### Cited Findings
- Rig parameters: ~173 GB/s measured per-card bandwidth (192 GB/s theoretical), PCIe 1.0 x1 (~190 MB/s per card), no P2P — [hardware spec](../../mining_rig.md)
- Calibration data from other Pascal setups (llama.cpp): GLM-4.7-Flash Q4_0 at 41.22 tok/s on one P40 — [#19020](https://github.com/ggml-org/llama.cpp/issues/19020); Qwen3.6-35B-A3B UD-Q4_K_XL at 91-95 tok/s on 2x P100 with `-sm layer` — [Strata #3294](https://github.com/Niko1221/Strata/issues/3294); Qwen3.5-122B-A10B UD-Q3_K_XL at 29.5 tok/s on 4x P40 — [#27980](https://github.com/ggml-org/llama.cpp/issues/27980)

**Method.** I compute bytes per token = active parameters x effective bits-per-weight / 8. The effective bpw of each GGUF is 16 x (quant file size / BF16 file size from the same repo), so it includes the quantizer's mix of tensor types. The "pessimistic" column keeps routed experts at the file's effective bpw but counts always-on weights (attention, SSM, shared experts, LM head) at max(effective bpw, 8.5). The uniform column underestimates bytes for UD and K-quant files, which upweight those tensors. gpt-oss-20b is handled explicitly: experts at MXFP4 4.25 bpw, other weights at Q8_0 (8.5) or F16. KV@32K uses f16 KV plus fixed recurrent and SWA state. "~50 GB" is the assumed usable VRAM after per-GPU overheads. Rows marked "est" have no fetched GGUF size and use nominal bpw (Q4_K_M 4.9, Q8_0 8.5). File sizes are decimal GB.

| Model (file source) | Quant | File GB | eff. bpw | bytes/token, uniform (GB) | ceiling tok/s uniform | bytes/token, always-on at Q8_0 (GB) | ceiling tok/s (pessimistic) | fits 26 GB free disk (tight above ~24 GB) | weights+KV@32K f16 (GB) fits ~50 GB VRAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | UD-IQ3_XXS | 12.91 | 3.38 | 1.29 | 135 | 2.07 | 83 | yes | 16.1 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | Q3_K_M | 14.71 | 3.85 | 1.46 | 118 | 2.18 | 79 | yes | 17.9 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | IQ4_XS | 16.38 | 4.29 | 1.63 | 106 | 2.28 | 76 | yes | 19.6 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | UD-Q4_K_XL | 17.69 | 4.63 | 1.76 | 98 | 2.36 | 73 | yes | 20.9 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | Q4_K_M | 18.56 | 4.86 | 1.85 | 94 | 2.41 | 72 | yes | 21.8 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | Q5_K_M | 21.73 | 5.69 | 2.16 | 80 | 2.60 | 67 | yes | 25.0 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | Q6_K | 25.09 | 6.57 | 2.50 | 69 | 2.79 | 62 | tight | 28.3 yes |
| Qwen3-30B-A3B (unsloth Qwen3-30B-A3B-Instruct-2507) | Q8_0 | 32.48 | 8.51 | 3.23 | 53 | 3.23 | 53 | NO | 35.7 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-IQ3_XXS | 13.21 | 3.05 | 1.12 | 154 | 2.44 | 71 | yes | 13.9 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-Q3_K_M | 16.60 | 3.83 | 1.41 | 123 | 2.54 | 68 | yes | 17.3 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-IQ4_XS | 17.73 | 4.09 | 1.51 | 115 | 2.57 | 67 | yes | 18.5 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-Q4_K_M | 22.13 | 5.10 | 1.88 | 92 | 2.70 | 64 | yes | 22.9 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-Q4_K_XL | 22.36 | 5.16 | 1.90 | 91 | 2.71 | 64 | yes | 23.1 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | MXFP4_MOE | 21.71 | 5.01 | 1.84 | 94 | 2.69 | 64 | yes | 22.4 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-Q5_K_M | 26.46 | 6.10 | 2.25 | 77 | 2.83 | 61 | NO | 27.2 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | UD-Q6_K | 29.31 | 6.76 | 2.49 | 70 | 2.91 | 59 | NO | 30.0 yes |
| Qwen3.5/3.6-35B-A3B (unsloth Qwen3.6-35B-A3B) | Q8_0 | 36.90 | 8.51 | 3.13 | 55 | 3.13 | 55 | NO | 37.6 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | UD-IQ3_XXS | 12.91 | 3.45 | 1.54 | 112 | 2.71 | 64 | yes | 14.7 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | Q3_K_M | 14.61 | 3.90 | 1.75 | 99 | 2.80 | 62 | yes | 16.4 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | IQ4_XS | 16.27 | 4.35 | 1.94 | 89 | 2.90 | 60 | yes | 18.0 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | MXFP4_MOE | 16.97 | 4.53 | 2.03 | 85 | 2.94 | 59 | yes | 18.7 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | UD-Q4_K_XL | 17.52 | 4.68 | 2.09 | 83 | 2.97 | 58 | yes | 19.3 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | Q4_K_M | 18.31 | 4.89 | 2.19 | 79 | 3.02 | 57 | yes | 20.1 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | Q5_K_M | 21.41 | 5.72 | 2.56 | 68 | 3.20 | 54 | yes | 23.2 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | Q6_K | 24.69 | 6.59 | 2.95 | 59 | 3.39 | 51 | tight | 26.5 yes |
| GLM-4.7-Flash (unsloth GLM-4.7-Flash) | Q8_0 | 31.84 | 8.50 | 3.80 | 45 | 3.80 | 45 | NO | 33.6 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | IQ3_XXS | 19.80 | 4.81 | 1.94 | 89 | 2.79 | 62 | yes | 20.0 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q3_K_M | 19.82 | 4.82 | 1.94 | 89 | 2.79 | 62 | yes | 20.1 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | IQ4_XS | 18.92 | 4.60 | 1.85 | 93 | 2.76 | 63 | yes | 19.2 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q4_0 | 19.06 | 4.63 | 1.87 | 93 | 2.76 | 63 | yes | 19.3 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | MXFP4_MOE(unsloth) | 23.21 | 5.64 | 2.27 | 76 | 2.94 | 59 | yes | 23.5 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q4_K_M | 25.48 | 6.19 | 2.50 | 69 | 3.03 | 57 | tight | 25.7 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q5_K_M | 26.96 | 6.55 | 2.64 | 65 | 3.09 | 56 | NO | 27.2 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q6_K | 34.31 | 8.34 | 3.36 | 51 | 3.40 | 51 | NO | 34.6 yes |
| Nemotron-H 30B-A3B (bartowski Nemotron-3.5-Lightning) | Q8_0 | 35.00 | 8.50 | 3.43 | 50 | 3.43 | 50 | NO | 35.2 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | UD-IQ3_XXS | 9.55 | 3.50 | 1.47 | 118 | 2.37 | 73 | yes | 11.4 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | Q3_K_M | 10.52 | 3.86 | 1.62 | 107 | 2.45 | 71 | yes | 12.4 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | IQ4_XS | 11.82 | 4.33 | 1.81 | 95 | 2.57 | 67 | yes | 13.7 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | UD-Q4_K_XL | 13.40 | 4.91 | 2.06 | 84 | 2.70 | 64 | yes | 15.3 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | Q4_K_M | 13.33 | 4.89 | 2.05 | 85 | 2.70 | 64 | yes | 15.2 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | Q5_K_M | 15.59 | 5.71 | 2.39 | 72 | 2.90 | 60 | yes | 17.5 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | Q6_K | 17.97 | 6.59 | 2.76 | 63 | 3.10 | 56 | yes | 19.8 yes |
| ERNIE-4.5-21B-A3B (unsloth ERNIE-4.5-21B-A3B-Thinking) | Q8_0 | 23.21 | 8.51 | 3.56 | 49 | 3.56 | 49 | yes | 25.1 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | Q3_K_M | 4.00 | 3.83 | 0.75 | 232 | 1.09 | 159 | yes | 4.4 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | UD-Q4_K_XL | 4.74 | 4.54 | 0.88 | 196 | 1.18 | 147 | yes | 5.1 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | Q4_K_M | 5.04 | 4.83 | 0.94 | 184 | 1.21 | 143 | yes | 5.4 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | Q5_K_M | 5.92 | 5.68 | 1.10 | 157 | 1.31 | 132 | yes | 6.3 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | Q6_K | 6.85 | 6.57 | 1.28 | 135 | 1.42 | 122 | yes | 7.3 yes |
| LFM2-8B-A1B (unsloth LFM2-8B-A1B) | Q8_0 | 8.87 | 8.50 | 1.66 | 104 | 1.66 | 104 | yes | 9.3 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | UD-IQ3_XXS | 3.27 | 3.09 | 0.65 | 266 | 1.14 | 152 | yes | 3.7 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | UD-Q4_K_XL | 5.35 | 5.05 | 1.06 | 163 | 1.37 | 126 | yes | 5.8 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | UD-Q4_K_M | 5.32 | 5.02 | 1.06 | 164 | 1.37 | 126 | yes | 5.7 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | MXFP4_MOE | 5.35 | 5.05 | 1.06 | 163 | 1.37 | 126 | yes | 5.8 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | UD-Q6_K | 7.09 | 6.69 | 1.41 | 123 | 1.57 | 110 | yes | 7.5 yes |
| LFM2.5-8B-A1B (unsloth LFM2.5-8B-A1B) | Q8_0 | 9.01 | 8.51 | 1.79 | 97 | 1.79 | 97 | yes | 9.4 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | IQ3_XXS | 9.39 | 3.15 | 0.92 | 189 | 1.51 | 114 | yes | 10.1 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | Q3_K_M | 10.84 | 3.64 | 1.06 | 164 | 1.60 | 108 | yes | 11.5 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | IQ4_XS | 12.75 | 4.28 | 1.24 | 139 | 1.71 | 101 | yes | 13.4 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | Q4_K_M | 14.44 | 4.84 | 1.41 | 123 | 1.82 | 95 | yes | 15.1 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | Q5_K_M | 16.93 | 5.68 | 1.65 | 105 | 1.97 | 88 | yes | 17.6 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | Q6_K | 19.58 | 6.57 | 1.91 | 91 | 2.13 | 81 | yes | 20.3 yes |
| LFM2-24B-A2B (bartowski LFM2-24B-A2B) | Q8_0 | 25.35 | 8.50 | 2.47 | 70 | 2.47 | 70 | tight | 26.0 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | UD-IQ3_XXS | 8.96 | 2.99 | 0.92 | 189 | 1.62 | 107 | yes | 9.6 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | UD-Q3_K_XL | 10.95 | 3.65 | 1.12 | 154 | 1.74 | 99 | yes | 11.6 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | UD-Q4_K_XL | 14.88 | 4.96 | 1.52 | 114 | 1.97 | 88 | yes | 15.6 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | Q4_K_M(ggml-org) | 14.53 | 4.85 | 1.49 | 116 | 1.95 | 89 | yes | 15.2 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | UD-Q5_K_XL | 17.92 | 5.98 | 1.83 | 94 | 2.16 | 80 | yes | 18.6 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | UD-Q6_K_XL | 21.86 | 7.29 | 2.24 | 77 | 2.39 | 72 | yes | 22.5 yes |
| LFM2.5-24B-A2B (unsloth LFM2.5-24B-A2B) | Q8_0 | 25.49 | 8.50 | 2.61 | 66 | 2.61 | 66 | tight | 26.2 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | UD-IQ3_XXS | 13.70 | 3.40 | 3.74 | 46 | 6.95 | 25 | yes | 14.4 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | Q3_K_M | 15.49 | 3.85 | 4.23 | 41 | 7.15 | 24 | yes | 16.2 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | IQ4_XS | 17.43 | 4.33 | 4.76 | 36 | 7.38 | 23 | yes | 18.1 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | UD-Q4_K_XL | 18.76 | 4.66 | 5.12 | 34 | 7.54 | 23 | yes | 19.4 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | Q4_K_M | 19.57 | 4.86 | 5.34 | 32 | 7.63 | 23 | yes | 20.3 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | Q5_K_M | 22.95 | 5.70 | 6.27 | 28 | 8.03 | 22 | yes | 23.6 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | Q6_K | 26.53 | 6.59 | 7.25 | 24 | 8.45 | 20 | NO | 27.2 yes |
| Granite-4.0-H-Small (unsloth granite-4.0-h-small) | Q8_0 | 34.26 | 8.51 | 9.36 | 18 | 9.36 | 18 | NO | 34.9 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | IQ3_XXS | 6.55 | 3.22 | 0.45 | 387 | 0.86 | 200 | yes | 7.9 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | Q3_K_M | 7.54 | 3.71 | 0.51 | 336 | 0.89 | 194 | yes | 8.9 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | Q4_K_M | 9.94 | 4.89 | 0.68 | 255 | 0.96 | 179 | yes | 11.3 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | Q5_K_M | 11.62 | 5.71 | 0.79 | 218 | 1.01 | 171 | yes | 13.0 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | Q6_K | 13.39 | 6.58 | 0.91 | 189 | 1.07 | 162 | yes | 14.7 yes |
| Ling-mini-2.0 / Ring-mini-2.0 (bartowski Ling-mini-2.0) | Q8_0 | 17.31 | 8.51 | 1.18 | 146 | 1.18 | 146 | yes | 18.7 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | IQ3_XXS | 3.46 | 3.50 | 0.50 | 347 | 0.94 | 184 | yes | 3.7 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | Q3_K_M | 3.79 | 3.84 | 0.55 | 317 | 0.96 | 181 | yes | 4.0 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | Q4_K_M | 4.92 | 4.98 | 0.71 | 244 | 1.02 | 170 | yes | 5.2 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | Q5_K_M | 5.72 | 5.79 | 0.82 | 210 | 1.06 | 163 | yes | 6.0 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | Q6_K | 6.84 | 6.93 | 0.99 | 175 | 1.13 | 154 | yes | 7.1 yes |
| Ling-3.0-tiny (bartowski Ling-3.0-tiny) | Q8_0 | 8.41 | 8.52 | 1.21 | 143 | 1.21 | 143 | yes | 8.7 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | IQ3_XXS | 19.65 | 3.20 | 1.24 | 139 | 2.32 | 74 | yes | 20.0 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | Q3_K_M | 22.68 | 3.69 | 1.43 | 121 | 2.42 | 72 | yes | 23.0 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | IQ4_XS | 26.46 | 4.31 | 1.67 | 103 | 2.53 | 68 | NO | 26.8 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | Q4_K_M | 30.06 | 4.89 | 1.90 | 91 | 2.64 | 66 | NO | 30.4 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | Q5_K_M | 35.10 | 5.71 | 2.22 | 78 | 2.79 | 62 | NO | 35.4 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | Q6_K | 40.46 | 6.59 | 2.56 | 68 | 2.95 | 59 | NO | 40.8 yes |
| Kimi-Linear-48B-A3B (bartowski Kimi-Linear-48B-A3B-Instruct) | Q8_0 | 52.25 | 8.51 | 3.30 | 52 | 3.30 | 52 | NO | 52.6 NO |
| Trinity-Mini (bartowski Trinity-Mini) | IQ3_XXS | 10.54 | 3.23 | 1.24 | 140 | 2.26 | 77 | yes | 11.2 yes |
| Trinity-Mini (bartowski Trinity-Mini) | Q3_K_M | 12.10 | 3.70 | 1.42 | 122 | 2.35 | 74 | yes | 12.7 yes |
| Trinity-Mini (bartowski Trinity-Mini) | IQ4_XS | 14.16 | 4.33 | 1.66 | 104 | 2.47 | 70 | yes | 14.8 yes |
| Trinity-Mini (bartowski Trinity-Mini) | Q4_K_M | 15.94 | 4.88 | 1.87 | 93 | 2.57 | 67 | yes | 16.6 yes |
| Trinity-Mini (bartowski Trinity-Mini) | Q5_K_M | 18.64 | 5.71 | 2.19 | 79 | 2.73 | 63 | yes | 19.3 yes |
| Trinity-Mini (bartowski Trinity-Mini) | Q6_K | 21.52 | 6.59 | 2.52 | 69 | 2.89 | 60 | yes | 22.2 yes |
| Trinity-Mini (bartowski Trinity-Mini) | Q8_0 | 27.79 | 8.51 | 3.26 | 53 | 3.26 | 53 | NO | 28.4 yes |
| Trinity-Nano-Preview (bartowski Trinity-Nano-Preview) | Q4_K_M | 3.79 | 4.95 | 0.51 | 342 | 0.72 | 241 | yes | 4.3 yes |
| Trinity-Nano-Preview (bartowski Trinity-Nano-Preview) | Q6_K | 5.06 | 6.60 | 0.68 | 256 | 0.79 | 219 | yes | 5.6 yes |
| Trinity-Nano-Preview (bartowski Trinity-Nano-Preview) | Q8_0 | 6.53 | 8.52 | 0.87 | 198 | 0.87 | 198 | yes | 7.1 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-IQ3_XXS | 11.42 | 3.62 | 1.73 | 100 | 3.19 | 54 | yes | 12.3 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-Q3_K_M | 12.73 | 4.03 | 1.93 | 90 | 3.26 | 53 | yes | 13.6 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-IQ4_XS | 13.60 | 4.31 | 2.06 | 84 | 3.31 | 52 | yes | 14.5 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | QAT UD-Q4_K_XL | 14.25 | 4.51 | 2.16 | 80 | 3.35 | 52 | yes | 15.1 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | Q4_0(ggml-org) | 14.62 | 4.63 | 2.21 | 78 | 3.37 | 51 | yes | 15.5 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | MXFP4_MOE | 16.55 | 5.24 | 2.50 | 69 | 3.48 | 50 | yes | 17.4 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-Q4_K_M | 16.95 | 5.37 | 2.57 | 67 | 3.50 | 49 | yes | 17.8 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-Q5_K_M | 21.15 | 6.70 | 3.20 | 54 | 3.74 | 46 | yes | 22.0 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | UD-Q6_K | 23.17 | 7.34 | 3.51 | 49 | 3.85 | 45 | yes | 24.1 yes |
| Gemma-4-26B-A4B (unsloth gemma-4-26B-A4B-it) | Q8_0 | 26.86 | 8.51 | 4.06 | 43 | 4.06 | 43 | NO | 27.7 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-IQ3_XXS | 11.71 | 3.07 | 1.28 | 136 | 2.30 | 75 | yes | 12.9 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-Q3_K_M | 14.21 | 3.73 | 1.55 | 112 | 2.45 | 71 | yes | 15.4 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-IQ4_XS | 15.23 | 3.99 | 1.66 | 104 | 2.51 | 69 | yes | 16.4 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | MXFP4_MOE | 18.66 | 4.89 | 2.03 | 85 | 2.72 | 64 | yes | 19.8 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-Q4_K_M | 19.20 | 5.04 | 2.09 | 83 | 2.75 | 63 | yes | 20.4 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-Q5_K_M | 22.95 | 6.02 | 2.50 | 69 | 2.97 | 58 | yes | 24.1 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | UD-Q6_K | 25.51 | 6.69 | 2.78 | 62 | 3.12 | 55 | tight | 26.7 yes |
| North-Mini-Code-1.0 (unsloth North-Mini-Code-1.0) | Q8_0 | 32.44 | 8.51 | 3.54 | 49 | 3.54 | 49 | NO | 33.6 yes |
| Qwen3-30B-A3B (bartowski Tongyi-DeepResearch-30B-A3B) | Q4_K_M | 18.63 | 4.88 | 1.85 | 93 | 2.41 | 72 | yes | 21.9 yes |
| Qwen3-30B-A3B (bartowski Tongyi-DeepResearch-30B-A3B) | Q8_0 | 32.48 | 8.51 | 3.23 | 53 | 3.23 | 53 | NO | 35.7 yes |
| Phi-3.5-MoE (estimate (no size fetched)) | Q4_K_M(est 4.9bpw) | 25.60 | 4.89 | 3.98 | 43 | 4.65 | 37 | tight | 29.9 yes |
| Phi-3.5-MoE (estimate (no size fetched)) | Q8_0(est) | 44.50 | 8.51 | 6.92 | 25 | 6.92 | 25 | NO | 48.8 yes |
| Moonlight-16B-A3B (estimate) | Q4_K_M(est) | 9.80 | 4.92 | 1.58 | 109 | 2.14 | 81 | yes | 10.8 yes |
| Moonlight-16B-A3B (estimate) | Q8_0(est) | 17.00 | 8.53 | 2.75 | 63 | 2.75 | 63 | yes | 18.0 yes |
| DeepSeek-V2-Lite (estimate) | Q4_K_M(est) | 9.60 | 4.89 | 1.50 | 115 | 2.00 | 87 | yes | 10.6 yes |
| DeepSeek-V2-Lite (estimate) | Q8_0(est) | 16.70 | 8.51 | 2.61 | 66 | 2.61 | 66 | yes | 17.7 yes |
| kanana-2-30b-a3b (estimate) | Q4_K_M(est) | 18.80 | 4.91 | 2.06 | 84 | 2.96 | 58 | yes | 20.6 yes |
| kanana-2-30b-a3b (estimate) | Q8_0(est) | 32.60 | 8.51 | 3.56 | 49 | 3.56 | 49 | NO | 34.4 yes |
| Hy-MT2-30B-A3B / Hy-Memory-30B-A3B (estimate) | Q4_K_M(est) | 18.40 | 4.90 | 1.96 | 88 | 2.61 | 66 | yes | 21.6 yes |
| Hy-MT2-30B-A3B / Hy-Memory-30B-A3B (estimate) | Q8_0(est) | 31.90 | 8.49 | 3.40 | 51 | 3.40 | 51 | NO | 35.1 yes |
| JoyAI-LLM-Flash (estimate) | Q4_K_M(est) | 30.20 | 4.90 | 1.86 | 93 | 2.57 | 67 | NO | 31.7 yes |
| JoyAI-LLM-Flash (estimate) | Q8_0(est) | 52.40 | 8.50 | 3.23 | 54 | 3.23 | 54 | NO | 53.9 NO |
| SmallThinker-21B-A3B (estimate) | Q4_K_M(est) | 13.20 | 4.91 | 2.04 | 85 | 2.71 | 64 | yes | 16.7 yes |
| SmallThinker-21B-A3B (estimate) | Q8_0(est) | 22.90 | 8.52 | 3.54 | 49 | 3.54 | 49 | yes | 26.4 yes |
| Phi-mini-MoE (estimate) | Q4_K_M(est) | 4.70 | 4.92 | 1.37 | 126 | 2.03 | 85 | yes | 9.0 yes |
| Phi-mini-MoE (estimate) | Q8_0(est) | 8.10 | 8.47 | 2.36 | 73 | 2.37 | 73 | yes | 12.4 yes |
| OLMoE-1B-7B-0125 (bartowski OLMoE-1B-7B-0924) | Q4_K_M | 4.21 | 4.87 | 0.72 | 241 | 0.89 | 195 | yes | 8.5 yes |
| OLMoE-1B-7B-0125 (bartowski OLMoE-1B-7B-0924) | Q8_0 | 7.36 | 8.51 | 1.25 | 138 | 1.25 | 138 | yes | 11.7 yes |
| gpt-oss-20b | MXFP4 (ggml-org; non-expert Q8_0) | 12.11 | experts 4.25 | 2.56 | 67 | 2.56 | 67 | yes | 12.9 yes |
| gpt-oss-20b | F16 (unsloth; non-expert F16) | 13.79 | experts 4.25 | 3.71 | 47 | 3.71 | 47 | yes | 14.6 yes |

### Inferences
- Calibration: my uniform-bpw method gives a GLM-4.7-Flash Q4_0 ceiling of ~171 tok/s at the P40's 347 GB/s theoretical bandwidth, against 41.2 measured (24%). Qwen3.6-35B UD-Q4_K_XL gives ~385 tok/s at the P100's 732 GB/s theoretical, against 91-95 measured (24%). Qwen3.5-122B-A10B at about 4 bpw gives ~71 tok/s at 347 GB/s against 29.5 measured (about 41%). The higher fraction for the 10B-active model suggests a large fixed per-token overhead (kernel launches, routing, recurrent and norm ops) that matters most for ~3B-active models. Applying 24-45% to the P106 ceilings, which use measured rather than theoretical bandwidth so the real fraction could be a bit higher, gives about 20-40 tok/s for Qwen3.6/Qwen3-30B Q4, 15-30 for Gemma 4 Q4, 16-30 for gpt-oss-20b MXFP4 and 8-15 for Granite-H-Small Q4. These are hypotheses for experiments, not predictions.
- P106-specific risks to the per-token overhead are 5 SMs per GPU (slow non-matmul ops), 8 inter-GPU hops through host memory on a 9-GPU split, and the 2-core 1.8 GHz Celeron issuing all kernel launches. The hidden-state transfer per hop is small (hidden 2048-2880 values, 8-11 KB at f32, about 45-60 µs at 190 MB/s plus sync latency). Even so, 8 hops add roughly 0.5-1 ms per token. Using the fewest GPUs that hold the model plus KV (3-5 for 14-23 GB files) reduces hops without changing the bandwidth ceiling, since only one GPU reads at a time.
- Quant choice changes speed in direct proportion to bytes per token. For Qwen3.6, UD-IQ3_XXS (13.2 GB) has about 1.7x the uniform ceiling of UD-Q4_K_XL (22.4 GB), but under the pessimistic model only about 1.1x, because the always-on 1.94B dominates. Testing both pins down which model of per-token bytes holds for UD files.
- MTP or speculative decoding could help because decode is bandwidth-bound per GPU, but for MoE, verifying k draft tokens touches the union of their experts, so the cost per verified batch rises with k. Gains of perhaps 1.2-1.6x with `--spec-draft-n-max 2` are plausible but unverified on Pascal.
- Tensor parallelism (`-sm tensor`) would in principle let all 9 GPUs stream weights at once (a 9x bandwidth ceiling), but it needs per-layer all-reduces through host memory over PCIe 1.0 x1 with no P2P. Latency per all-reduce (likely hundreds of µs) times 2 per layer times 40-48 layers would likely cap it below layer split for batch 1. This is a claim to test; the [multi-GPU note](multigpu_pascal_performance.md) covers multi-GPU strategy.

### Gaps
- No measured tok/s exists for any model on P106-090 hardware; all rig numbers here are derived.
- The exact per-tensor quant mix (and so the true bytes per token) of the unsloth UD and MXFP4_MOE files was not inspected. `gguf-dump` on the workstation could give it exactly.
- Prompt-processing (prefill) speed was not modelled. It is compute-bound (dp4a INT8 about 8 TOPS peak per card, one card active at a time unless pipeline micro-batching overlaps cards) and matters for agentic use with long tool outputs.

## Experiment ideas: what to test first on the rig (options, not decisions)

### Takeaway
A sensible sequence goes from "known-good kernels" to "new kernels". Start with plain GQA (Qwen3-30B-A3B or Coder) as a control and Qwen3.6-35B-A3B, whose GDN path the rig has already exercised. Then test the hybrids with distinctive Pascal risks: Nemotron-H Mamba2, Gemma 4's head-512 FA, GLM-4.7-Flash MLA and gpt-oss MXFP4. Each experiment should record tg128 and pp512 against the ceilings above, so the efficiency factor is measured rather than assumed.

### Cited Findings
- Candidate quality signals: Qwen3.6-35B-A3B leads AA's ~3B-active cohort (April 2026 snapshot) — [AA on X](https://x.com/ArtificialAnlys/status/2049881951260283097); Nemotron 3.5 Lightning is much faster per index task (~0.5 min vs ~3.5 min) — [AA article](https://artificialanalysis.ai/articles/nemotron-3-5-lightning-launch); GLM-4.7-Flash reports τ²-Bench 79.5 and SWE-bench Verified 59.2 — [card](https://huggingface.co/zai-org/GLM-4.7-Flash); LFM2.5-24B-A2B reports BFCLv4 65.01 with 2.45B active — [card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- Known Pascal tuning levers: MMVQ fusion on sm_61 ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334)), Q8_0 MMQ tile I=128 ([#30235](https://github.com/ggml-org/llama.cpp/issues/30235)), and FA off for MLA head sizes ([#19020](https://github.com/ggml-org/llama.cpp/issues/19020))
- MTP usage flags: `--spec-type draft-mtp --spec-draft-n-max 2` — [unsloth Qwen3.6 MTP card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)

### Inferences
**Concrete experiments.** All use llama-bench (`-p 512 -n 128`) and llama-server with a fixed chat and coding prompt set. All record per-GPU memory (`nvidia-smi`), tg and pp tok/s, and the KV-size log line.

| # | Model / quant (file size) | What to measure | Expected result (hypothesis) | Effort / risk |
|---|---|---|---|---|
| E1 | Qwen3-Coder-30B-A3B or Qwen3-30B-A3B-Instruct-2507, Q4_K_M (18.56 GB) or UD-Q4_K_XL (17.69) | tg128 and pp512 with `-fa on` vs `off`, KV f16 vs q8_0; 4-GPU vs 9-GPU split (`-ts`) | 25-40 tok/s (ceiling 94 uniform / 72 pessimistic); a GPU-count change of under 5% means hop overhead is small | Low / low: mature GQA path. This is the efficiency-factor control |
| E2 | Qwen3.6-35B-A3B UD-Q4_K_XL (22.36 GB) and UD-IQ3_XXS (13.21 GB) | Same as E1, plus 32K-context decode; compare the two quants | 20-40 tok/s at Q4 (ceiling 91/64); IQ3 1.1-1.7x faster depending on how many always-on bytes UD keeps | Low / low-medium: GDN already ran on the rig via Qwen3.8-27B |
| E3 | Qwen3.6-35B-A3B MTP GGUF UD-Q4_K_XL (22.85 GB) | `--spec-type draft-mtp --spec-draft-n-max 2/3`: acceptance rate, tok/s, output equality at greedy | 1.0-1.6x speed-up; possible correctness issues (see #25618, #23577) | Low / medium |
| E4 | gpt-oss-20b MXFP4 (12.11 GB, ggml-org) | Does MXFP4 plus attention sinks run on sm_61? tg vs ceiling 67; reasoning-effort low/medium | 16-30 tok/s; only ~13 GB, so fits on 3 GPUs | Low / medium: MXFP4 and sink kernels unverified on Pascal |
| E5 | Nemotron 3.5 Lightning IQ4_XS (18.92 GB) or Q4_0 (19.06) | SSM_SCAN cost on 5-SM GPUs (compare tg to E1 at similar bytes per token), 128K context memory (KV ~6 KiB/token), MTP draft | 18-35 tok/s if the recurrent step is cheap; noticeably worse if #27980-style recurrent overhead appears | Low / medium |
| E6 | Gemma 4 26B-A4B QAT UD-Q4_K_XL (14.25 GB) or ggml-org Q4_0 (14.62) | FA on/off with head-dim-512 global layers; SWA cache size; tool-call formatting in llama-server | 15-30 tok/s (ceiling 78-80 uniform / ~52 pessimistic); FA may be slow or unsupported on Pascal | Low / medium |
| E7 | GLM-4.7-Flash UD-Q4_K_XL (17.52 GB) or Q4_0 | tg with `-fa off` vs `on` (prefill was halved on P40 with FA); KV size (expect ~53 KiB/token latent) | ~20-35 tok/s (P40 single-card Q4_0 got 41 tok/s at 2x the bandwidth) | Low / low-medium |
| E8 | LFM2.5-24B-A2B UD-Q4_K_XL (14.88 GB) and Q8_0 (25.49 GB, tight on disk) | Conv-layer path on Pascal; Q4 vs Q8 speed and quality; may need a build of at least b11606 | 30-50 tok/s at Q4 (ceiling 114/88), 20-30 at Q8 | Low-medium / medium (newest model, Oct 12) |
| E9 | Speed probes: Ling-3.0-tiny Q4_K_M (4.92 GB), LFM2.5-8B-A1B Q4_K_M (5.32), Trinity-Nano Q4_K_M (3.79) | Single-GPU vs 2-GPU tg, to measure the fixed per-token overhead floor | If tg stays far below the 160-340 tok/s ceilings (e.g. 40-60 tok/s), overhead dominates and the fix is launch overhead or CUDA graphs, not quant | Very low / low |
| E10 | Quant ladder on one model (Qwen3.6: UD-IQ3_XXS 13.2 → UD-Q4_K_XL 22.4 → Q8_0 36.9 over NFS) | tok/s vs bytes per token; check linearity | Linear fit gives the effective bandwidth and fixed overhead per token for the rig | Medium / low (needs the NFS mount) |
| E11 | Any of E1/E2 with patched builds | MMVQ fusion off for cc 6.1 (#30334), Q8_0 MMQ I=128 (#30235) | 0-12% tg change at batch 1 for some quants; pp +12% on Q8_0 | Medium (rebuild) / low |
| E12 | Loading over NFS/SSHFS from the workstation (Q8_0 of Qwen3.6 36.9 GB, or Kimi-Linear Q4_K_M 30.06 GB) | Load time, host RAM pressure with 4 GB, steady-state tok/s identical to a local load | ~5-6 min load at gigabit; no runtime penalty once in VRAM | Low / low-medium (mmap and page cache with 4 GB RAM) |
| E13 | Kimi-Linear-48B-A3B Q3_K_M (22.68 GB, local) or Q4_K_M (30.06, NFS) | KDA plus MLA on Pascal (FA off), long-context memory | 18-35 tok/s; tiny KV (~8 KiB/token) | Medium / medium |
| E14 | `-sm tensor` on 2-3 GPUs vs layer split, same model as E1 | Batch-1 tg and pp | Likely slower than layer split for batch 1 given no P2P and PCIe 1.0 x1, but worth one data point | Medium / medium (experimental feature) |

**An ordered list of models that look most worth testing first, and why (options):**
1. Qwen3.6-35B-A3B (UD-Q4_K_XL 22.36 GB; MTP variant 22.85 GB). It has the best published ~3B-active scores (SWE-V 73.4 and TB2 51.5 self-reported; top of AA's ~3B-active cohort in April 2026), an architecture the rig has effectively already exercised via Qwen3.8-27B, tiny KV (20 KiB/token), Apache-2.0, and MTP. The downside is high token usage per task.
2. Qwen3-Coder-30B-A3B / Qwen3-30B-A3B-Instruct-2507 (Q4_K_M 18.56 GB). This is the most mature kernel path, so it serves as the control for measuring the rig's efficiency factor. It is weaker than Qwen3.6 on agentic coding and has heavier KV (96 KiB/token).
3. Nemotron 3.5 Lightning (IQ4_XS 18.92 GB). It is fast per task (low token usage per AA), has 6 KiB/token KV for long context, built-in MTP and an open license (OpenMDW-1.1 per AA). The risks are the untested Mamba2 path on 5-SM GPUs and the ~19 GB minimum file. Nemotron-Cascade-2 is the same architecture and is stronger on math and code reasoning per its card.
4. Gemma 4 26B-A4B (QAT UD-Q4_K_XL 14.25 GB). It is Apache-2.0 and multimodal, and AA notes far fewer output tokens than Qwen. Its active 3.8B includes a 0.74B LM head, and the head-512 FA path on Pascal is a risk.
5. GLM-4.7-Flash (UD-Q4_K_XL 17.52 GB). It has strong self-reported τ² and SWE-V numbers and a measured 41 tok/s on a single P40, but needs `-fa off` (MLA on Pascal) and MTP is not confirmed in llama.cpp.
6. gpt-oss-20b (MXFP4 12.11 GB). It is the smallest well-known general model, would fit 3-4 copies later for throughput, and uses a proven chat format. Speed is limited by its large non-expert share, and MXFP4 on Pascal is unverified.
7. LFM2.5-24B-A2B (UD-Q4_K_XL 14.88 GB; Q8_0 25.49 GB fits the disk only tightly). It is the newest model (2026-10-12), has the lowest active count (2.45B) among the agent-oriented models, and claims strong tool-use and IFBench results. It is a reasoning-always model and may need a very recent build.
8. KAT-Coder-V2.5-Dev (Q4_K_M 21.39 GB). This is a coding-specialised Qwen3.6-architecture fine-tune that costs nothing extra once E2 works. Its claimed SWE-V of 69.4 under a claude_code harness beats Qwen3.6's 64.4 in the same table.
9. Kimi-Linear-48B-A3B (Q3_K_M 22.68 GB). Slightly over the size range, but it has 1M-context design and tiny KV. Its quality on chat and coding versus the 2026 models is unclear.
10. Small speed probes: Ling-3.0-tiny (4.92 GB, AA v4.1.1 index 25 per its card), LFM2.5-8B-A1B and Trinity-Nano. These are mainly for measuring overhead floors and later throughput workers.

Lower priority on current evidence: Granite-4.0-H-Small (8.8B active, so ceiling ~32 tok/s at Q4); ERNIE-4.5-21B-A3B, SmallThinker-21B (16K context), Moonlight (8K), Phi-mini-MoE and OLMoE (4K context), all older or short-context; Hy-Memory and Hy-MT2 (specialists, no mainstream GGUF); kanana-2 and JoyAI (no mainstream GGUF); Megrez2 (not in mainline llama.cpp); and Qwen-AgentWorld (a world-simulator model, not an assistant).

### Gaps
- All expected tok/s figures are derived from bandwidth arithmetic plus three Pascal data points from other card types. None are measured on the P106-090, so each is explicitly a claim to test.
- Quality on an owner's own chat, coding and agent tasks is unknown for every candidate; the benchmark numbers above are mostly self-reported and use inconsistent harnesses.
