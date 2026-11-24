# Upstream b12483..b12674: what touches the P106 path

Upstream llama.cpp's 191 commits between tags b12483 and b12674, read for anything that reaches a compute 6.1 card, the
server's speculation, the KV cache's removal, the batch API, or the architectures of the release files (`qwen35moe`
for Ornith-1.5-122B Pascal N2P GGUF, `qwen4exp` for Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, `flint` for Flint Pascal
Q5/Q8 GGUF; read from the files' headers). Each entry says what to watch on a P106 rig.

## CUDA code a P106 runs

| Change | What it does on compute 6.1 | Watch on the rig |
|---|---|---|
| #31352 Pascal DP4A MMQ I tile for J=64 | the prompt kernel (`mul_mat_q`) for 64-column passes takes 128-row tiles instead of 64 for Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS, IQ4_NL, IQ4_XS (the SASS shows each) | prompt time of the 64-token micro-batches (Flint's and Flash-Next's taper `short=…:64`): half as many blocks for a P106's 5 SMs; C1024 |
| #32374 64-bit dst offsets in `mul_mat_q` | every `mul_mat_q` kernel computes its output offsets in 64 bits | prompt time (register use) |
| #32505 duplicate expert ids | the MoE id helper (`mm_ids_helper`) rewritten: a shared-memory store dumped in pieces | MoE prompt passes; the fork's legacy layout (`GGML_CUDA_MMQ_MOE_LEGACY`) assumes distinct ids per token |
| #31345 MUL_MAT_ID precision in the per-expert fallback | the fallback keeps the op's precision setting | none expected (the fallback is not the P106 path) |
| #32239 no quantized matrix-vector fusion on Pascal | merged upstream; the fork keeps the fusion on compute 6.1 unless `GGML_CUDA_MMID_GLU_UNFUSE=2` | Ornith's preset sets `2` (its measured gain), Flash-Next's leaves it unset (unfused is a loss there); the default is C893's question |
| #30672 SKU-aware accumulation precision | NVML is opened at CUDA start (`dlopen("libnvidia-ml.so.1")`, `nvmlInit`) to read each card's brand; Pascal has no fast BF16, so nothing changes in the math | load time and host memory at start on a 4 GB host with nine cards; C1025 |
| #32454 fused NORM + MUL (+ ADD) | a new fusion for layer norms (not RMS norms) | none for the release files (they use RMS norms) |
| #32387 BF16 ROPE and SET_ROWS | new kernel instances | none (no BF16 activations on the release paths) |
| #32489 scatter kernel for set_n | `batch-get.cu` renamed `batch-copy.cu` | the fork's keep-alive quiesce moved with it |
| #32698 abort on undefined fastdiv | a host check for divisors above 2^31 | none expected |
| #32065, #32566 few-row MMQ for MUL_MAT_ID | Ampere and newer only (`ampere_mma_available`): not a P106, and not the 3090s running the sm_61 build | none |

## Graph, memory and server

| Change | What it does | Watch |
|---|---|---|
| #32669 `.scale` tensors applied automatically | `build_lora_mm` and `build_lora_mm_id` look up each weight's `<name>.scale`; the explicit scale arguments are gone | the release files carry no `.scale` tensors, so nothing changes for them; the fork's cold tier (Q238) gets its scales the same way |
| #32677 the MoE down projection's rescale | rescaled only when the precision policy keeps src1 below BF16; Flint (`flint`) is one of the four rescaled architectures | Flint's replies; the fork's `LLAMA_NO_MOE_DOWN_RESCALE` rule still applies |
| #32519 QSA scores only the ubatch's sequences' k-pools | `qwen4exp`'s indexer list shrinks to the sequences in the ubatch | with the k-pool tables on the cards (`LLAMA_QSA_KPOOL_DEV`, Flash-Next's preset) the fork keeps every sequence's pools; without them upstream's list |
| #32650 the fused lightning indexer always on | its switch and the unfused fallback are gone | none (the fork ran the fused one) |
| #32483 `qwen4exp` graph reallocation from the n-gram table scale | a graph that no longer reallocates when the scale changes | Flash-Next's graph reallocations (its file is `qwen4exp`) |
| #31913 sticky NaN in a recurrent state | merged upstream; the fork carried the same fill (Q1158) | none |
| #32434 slot scheduler: LoRA and recurrent models | merged upstream; the fork carried it since b12483 (C854) | none: the same code |
| #32636, #32648 graph node budget | the budget moves into the model classes, the MoE cache reserves its own nodes | none (the fork's budget was upstream's) |
| #32596 `llm_graph_params` without hparams | inputs read the model's hparams | none |
| #32713 non-causal batches in one ubatch | server only, embeddings | none |
| #32553 decision packing | server's decision tasks | none |
| #32429 iswa attention in a wrapper and an impl | the fork's device-built KQ mask moved into the impl | the device mask with sliding-window models |

Nothing in this range changes the batch API (`llama_batch_ext`, `llama_process`) or the server's speculation loop.
