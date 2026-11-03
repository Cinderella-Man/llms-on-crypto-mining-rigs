# Draft vs target: token embeddings and output head

- draft: `mtp-Qwen3.8-Flash-Next-draft.gguf`, architecture `qwen4exp`
- target: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, architecture `qwen4exp`
- draft: embedding_length 2560, block_count 49, nextn_predict_layers 1
- target: embedding_length 2560, block_count 48, nextn_predict_layers None

## Vocabulary

- `tokenizer.ggml.model`: draft `gpt2`, target `gpt2`
- `tokenizer.ggml.pre`: draft `qwen35`, target `qwen35`
- `tokenizer.ggml.tokens`: draft 248,320 entries sha256 `49930cc8a75188cf`, target 248,320 entries `49930cc8a75188cf`: **same**
- `tokenizer.ggml.token_type`: draft 248,320 entries sha256 `1f15beca7fa4bce5`, target 248,320 entries `1f15beca7fa4bce5`: **same**
- `tokenizer.ggml.merges`: draft 247,587 entries sha256 `2d0d7c583fcdbbcc`, target 247,587 entries `2d0d7c583fcdbbcc`: **same**

## Tensors

### `token_embd.weight`

- draft: Q8_0 [2560, 248320], 675,430,400 bytes; target: Q3_K [2560, 248320], 273,152,000 bytes
- per-row cosine, draft vs target: mean 0.98835, min 0.97979, p1 0.98686, p50 0.98840
- per-row relative RMS error |d - t| / |d|: mean 0.1532, p50 0.1530, p99 0.1630, max 0.2028; largest element difference 0.02344
- baseline, each draft row against a shuffled target row: mean cosine 0.04584
- rows with cosine < 0.9: 0 of 248320

### `output.weight`

- draft: Q6_K [2560, 248320], 521,472,000 bytes; target: Q5_K [2560, 248320], 437,043,200 bytes
- per-row cosine, draft vs target: mean 0.99913, min 0.99780, p1 0.99893, p50 0.99913
- per-row relative RMS error |d - t| / |d|: mean 0.0418, p50 0.0416, p99 0.0462, max 0.0664; largest element difference 0.00934
- baseline, each draft row against a shuffled target row: mean cosine 0.17220
- rows with cosine < 0.9: 0 of 248320

## Draft tensors outside the blocks

`output.weight`, `token_embd.weight`

## Verdict

- vocabulary identical: **True**
- `token_embd.weight`: same matrix (every row cosine > 0.9)
- `output.weight`: same matrix (every row cosine > 0.9)
