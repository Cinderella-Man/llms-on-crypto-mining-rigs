# Headers: A `mtp-Qwen3.8-Flash-Next-draft.gguf`, B `mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf`

| | A | B |
|---|---|---|
| GGUF version | 3 | 3 |
| tensors | 34 | 32 |
| metadata keys | 55 | 52 |
| header bytes | 10,946,957 | 10,946,608 |
| tensor bytes | 2,686,492,672 | 2,775,621,632 |

## Metadata: 47 of 56 keys equal

| key | A | B | equal |
|---|---|---|---|
| `general.architecture` | 'qwen4exp' | 'qwen4exp' | yes |
| `general.type` | 'model' | 'model' | yes |
| `general.sampling.top_k` | 20 | 20 | yes |
| `general.sampling.top_p` | 0.949999988 | 0.949999988 | yes |
| `general.sampling.temp` | 1 | 1 | yes |
| `general.name` | 'Bcd9F01Ddc9Cff2316Eb84281Bebcd5B058Bddce' | 'Ckpt_Q38' | **no** |
| `general.size_label` | '512x1.4B' | '512x95M' | **no** |
| `general.license` | 'other' | 'other' | yes |
| `general.license.name` | 'qwen-community-1.0' | 'qwen-community-1.0' | yes |
| `general.license.link` | 'LICENSE' | 'LICENSE' | yes |
| `general.base_model.count` | 1 | '(absent)' | **no** |
| `general.base_model.0.name` | 'Qwen3.8 Flash Next' | '(absent)' | **no** |
| `general.base_model.0.organization` | 'Qwen' | '(absent)' | **no** |
| `general.base_model.0.repo_url` | 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next' | '(absent)' | **no** |
| `general.tags` | str[1] md5 461cd2eb672c… first ['image-text-to-text'] | str[1] md5 461cd2eb672c… first ['image-text-to-text'] | yes |
| `qwen4exp.block_count` | 49 | 49 | yes |
| `qwen4exp.context_length` | 262144 | 262144 | yes |
| `qwen4exp.embedding_length` | 2560 | 2560 | yes |
| `qwen4exp.attention.head_count` | 24 | 24 | yes |
| `qwen4exp.attention.head_count_kv` | 2 | 2 | yes |
| `qwen4exp.rope.dimension_sections` | i32[4] md5 399985bb064b… first [11, 11, 10] | i32[4] md5 399985bb064b… first [11, 11, 10] | yes |
| `qwen4exp.rope.freq_base` | 10000000 | 10000000 | yes |
| `qwen4exp.attention.layer_norm_rms_epsilon` | 9.99999997e-07 | 9.99999997e-07 | yes |
| `qwen4exp.expert_count` | 512 | 512 | yes |
| `qwen4exp.expert_used_count` | 10 | 10 | yes |
| `qwen4exp.attention.key_length` | 256 | 256 | yes |
| `qwen4exp.attention.value_length` | 256 | 256 | yes |
| `qwen4exp.expert_feed_forward_length` | 640 | 640 | yes |
| `qwen4exp.expert_shared_feed_forward_length` | 640 | 640 | yes |
| `qwen4exp.nextn_predict_layers` | 1 | 1 | yes |
| `qwen4exp.ssm.conv_kernel` | 4 | 4 | yes |
| `qwen4exp.ssm.state_size` | 128 | 128 | yes |
| `qwen4exp.ssm.group_count` | 16 | 16 | yes |
| `qwen4exp.ssm.time_step_rank` | 48 | 48 | yes |
| `qwen4exp.ssm.inner_size` | 6144 | 6144 | yes |
| `qwen4exp.full_attention_interval` | 4 | 4 | yes |
| `qwen4exp.rope.dimension_count` | 64 | 64 | yes |
| `qwen4exp.hyper_connection.count` | 4 | 4 | yes |
| `qwen4exp.hyper_connection.low_rank` | 320 | 320 | yes |
| `qwen4exp.attention.indexer.head_count` | 4 | 4 | yes |
| `qwen4exp.attention.indexer.key_length` | 128 | 128 | yes |
| `qwen4exp.attention.indexer.top_k` | 2048 | 2048 | yes |
| `qwen4exp.attention.compress_ratios` | i32[49] md5 f66a3543e290… first [0, 0, 0] | i32[49] md5 ce91cbdd2999… first [0, 0, 0] | **no** |
| `tokenizer.ggml.model` | 'gpt2' | 'gpt2' | yes |
| `tokenizer.ggml.pre` | 'qwen35' | 'qwen35' | yes |
| `tokenizer.ggml.tokens` | str[248320] md5 9af5e456413c… first ['!', '"', '#'] | str[248320] md5 9af5e456413c… first ['!', '"', '#'] | yes |
| `tokenizer.ggml.token_type` | i32[248320] md5 dda3bdd8eae7… first [1, 1, 1] | i32[248320] md5 dda3bdd8eae7… first [1, 1, 1] | yes |
| `tokenizer.ggml.merges` | str[247587] md5 e22b72cf2f1d… first ['Ġ Ġ', 'ĠĠ ĠĠ', 'i n'] | str[247587] md5 e22b72cf2f1d… first ['Ġ Ġ', 'ĠĠ ĠĠ', 'i n'] | yes |
| `tokenizer.ggml.eos_token_id` | 248046 | 248046 | yes |
| `tokenizer.ggml.padding_token_id` | 248044 | 248044 | yes |
| `tokenizer.ggml.bos_token_id` | 248044 | 248044 | yes |
| `tokenizer.ggml.add_bos_token` | False | False | yes |
| `tokenizer.chat_template` | '{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content= | '{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content= | yes |
| `general.quantization_version` | 2 | 2 | yes |
| `general.file_type` | 15 | 7 | **no** |
| `qwen4exp.nextn_shared_target_tensors` | '(absent)' | True | **no** |

`qwen4exp.attention.compress_ratios` differs at 1 of 49 indices (A against B): 48: 4 against 0
A: [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 4]
B: [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0]

## Tensors: 21 of 34 with the same type and shape in both

| tensor | A type | A shape | A bytes | B type | B shape | B bytes | equal |
|---|---|---|---:|---|---|---:|---|
| `output.weight` | Q6_K | 2560×248320 | 521,472,000 | (absent) |  |  | **no** |
| `blk.48.nextn.hc_head_down.weight` | Q8_0 | 10240×320 | 3,481,600 | Q8_0 | 10240×320 | 3,481,600 | yes |
| `blk.48.nextn.hc_head_norm.weight` | F32 | 10240 | 40,960 | F32 | 10240 | 40,960 | yes |
| `blk.48.nextn.hc_head_up.weight` | Q8_0 | 320×10240 | 3,481,600 | Q8_0 | 320×10240 | 3,481,600 | yes |
| `token_embd.weight` | Q8_0 | 2560×248320 | 675,430,400 | (absent) |  |  | **no** |
| `blk.48.attn_k.weight` | Q4_K | 2560×512 | 737,280 | Q8_0 | 2560×512 | 1,392,640 | **no** |
| `blk.48.attn_k_norm.weight` | F32 | 256 | 1,024 | F32 | 256 | 1,024 | yes |
| `blk.48.attn_output.weight` | Q4_K | 6144×2560 | 8,847,360 | Q8_0 | 6144×2560 | 16,711,680 | **no** |
| `blk.48.attn_q.weight` | Q4_K | 2560×12288 | 17,694,720 | Q8_0 | 2560×12288 | 33,423,360 | **no** |
| `blk.48.attn_q_norm.weight` | F32 | 256 | 1,024 | F32 | 256 | 1,024 | yes |
| `blk.48.attn_v.weight` | Q6_K | 2560×512 | 1,075,200 | Q8_0 | 2560×512 | 1,392,640 | **no** |
| `blk.48.ffn_down_exps.weight` | IQ4_NL | 640×2560×512 | 471,859,200 | Q8_0 | 640×2560×512 | 891,289,600 | **no** |
| `blk.48.ffn_down_shexp.weight` | Q8_0 | 640×2560 | 1,740,800 | Q8_0 | 640×2560 | 1,740,800 | yes |
| `blk.48.ffn_gate_exps.weight` | Q4_K | 2560×640×512 | 471,859,200 | Q8_0 | 2560×640×512 | 891,289,600 | **no** |
| `blk.48.ffn_gate_inp.weight` | F32 | 2560×512 | 5,242,880 | F32 | 2560×512 | 5,242,880 | yes |
| `blk.48.ffn_gate_inp_shexp.weight` | F32 | 2560 | 10,240 | F32 | 2560 | 10,240 | yes |
| `blk.48.ffn_gate_shexp.weight` | Q4_K | 2560×640 | 921,600 | Q8_0 | 2560×640 | 1,740,800 | **no** |
| `blk.48.ffn_up_exps.weight` | Q4_K | 2560×640×512 | 471,859,200 | Q8_0 | 2560×640×512 | 891,289,600 | **no** |
| `blk.48.ffn_up_shexp.weight` | Q4_K | 2560×640 | 921,600 | Q8_0 | 2560×640 | 1,740,800 | **no** |
| `blk.48.hc_attn_down.weight` | Q8_0 | 10240×320 | 3,481,600 | Q8_0 | 10240×320 | 3,481,600 | yes |
| `blk.48.hc_attn_inject.weight` | Q8_0 | 10240×4 | 43,520 | Q8_0 | 10240×4 | 43,520 | yes |
| `blk.48.hc_attn_norm.weight` | F32 | 10240 | 40,960 | F32 | 10240 | 40,960 | yes |
| `blk.48.hc_attn_up.weight` | Q8_0 | 320×10240 | 3,481,600 | Q8_0 | 320×10240 | 3,481,600 | yes |
| `blk.48.hc_ffn_down.weight` | Q8_0 | 10240×320 | 3,481,600 | Q8_0 | 10240×320 | 3,481,600 | yes |
| `blk.48.hc_ffn_inject.weight` | Q8_0 | 10240×4 | 43,520 | Q8_0 | 10240×4 | 43,520 | yes |
| `blk.48.hc_ffn_norm.weight` | F32 | 10240 | 40,960 | F32 | 10240 | 40,960 | yes |
| `blk.48.hc_ffn_up.weight` | Q8_0 | 320×10240 | 3,481,600 | Q8_0 | 320×10240 | 3,481,600 | yes |
| `blk.48.indexer.k_norm.weight` | F32 | 128 | 512 | F32 | 128 | 512 | yes |
| `blk.48.indexer.k_proj.weight` | Q8_0 | 2560×128 | 348,160 | BF16 | 2560×128 | 655,360 | **no** |
| `blk.48.indexer.q_norm.weight` | F32 | 128 | 512 | F32 | 128 | 512 | yes |
| `blk.48.indexer.q_proj.weight` | Q8_0 | 2560×512 | 1,392,640 | BF16 | 2560×512 | 2,621,440 | **no** |
| `blk.48.nextn.eh_proj.weight` | Q8_0 | 5120×2560 | 13,926,400 | Q8_0 | 5120×2560 | 13,926,400 | yes |
| `blk.48.nextn.enorm.weight` | F32 | 2560 | 10,240 | F32 | 2560 | 10,240 | yes |
| `blk.48.nextn.hnorm.weight` | F32 | 10240 | 40,960 | F32 | 10240 | 40,960 | yes |

## Bytes per type

| type | A tensors | A bytes | B tensors | B bytes |
|---|---:|---:|---:|---:|
| BF16 | 0 | 0 | 2 | 3,276,800 |
| F32 | 11 | 5,430,272 | 11 | 5,430,272 |
| IQ4_NL | 1 | 471,859,200 | 0 | 0 |
| Q4_K | 7 | 972,840,960 | 0 | 0 |
| Q6_K | 2 | 522,547,200 | 0 | 0 |
| Q8_0 | 13 | 713,815,040 | 19 | 2,766,914,560 |
