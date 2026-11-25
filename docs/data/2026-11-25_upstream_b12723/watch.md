# Upstream b12674..b12723: what touches the P106 path

Upstream llama.cpp's 49 commits between tags b12674 and b12723, read for anything that reaches a compute 6.1 card, the
server's speculation, the recurrent state's rollback, or the architectures of the release files (`qwen35moe` for
Ornith-1.5-122B Pascal N2P GGUF, `qwen4exp` for Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, both with IMROPE; `flint`
for Flint Pascal Q5/Q8 GGUF, NEOX RoPE with Q and K norms). Each entry says what the candidate does and what to watch on
a P106 rig.

## Three upstream defaults the candidate keeps behind switches

Each would change how a release file runs, none was measured on a P106, so each waits behind a switch at the fork's
measured behaviour (as #32239 did in the b12674 merge), with its own registry row for the P106 measurement.

| Change | Upstream's default | The candidate's default | Switch for upstream's | Row |
|---|---|---|---|---|
| #31669 replay instead of rollback snapshots | the delta-net models (`qwen35`, `qwen35moe`, `qwen4exp`, Clover-MoE and the KDA models) keep one state and the last micro-batch's inputs, and the next micro-batch replays the kept tokens inside the gated delta net op; the K-snapshot branch is gone from the graph | a rollback restores one of `n_rs_seq + 1` state snapshots, as in b12674: Flash-Next's MTP draft (`n_rs_seq` 3) writes four snapshots a verify, through the fork's GDN fusions | `LLAMA_RS_REPLAY=1` (the fork's in-place state read stays off for the replay form) | Q1852 |
| #32500 fused RMS_NORM + MUL + ROPE for M-RoPE and IMROPE | Ornith's and Flash-Next's attention layers run their Q and K norms and RoPE as one launch | three launches, as in b12674 | `GGML_CUDA_NORM_ROPE_MROPE=1` | Q1853 |
| #32500 block sizes of that fused kernel | 32 to 256 threads by the row length, and fewer for launches of several tokens: a row's sum of squares is added in another order for prompt and verify passes than for one token | 256 threads a row below 1,024 columns, 1,024 from there, as in b12674: the same order for one token and for several | `GGML_CUDA_NORM_ROPE_BS=1` | Q1854 |

The one-token attention block (`GGML_CUDA_ATTN_BLOCK`, on by default on compute 6.1) reuses the fused kernel's
arguments and has no M-RoPE sections: it now refuses M-RoPE rows, so with `GGML_CUDA_NORM_ROPE_MROPE=1` an IMROPE
model's decode takes the fused per-op path instead of computing RoPE without its sections.

## CUDA code a P106 runs

| Change | What it does on compute 6.1 | Watch on the rig |
|---|---|---|
| #31669 the gated delta net kernel | a `replay` template argument and the replay arguments passed by value to every launch; the snapshot (`K > 1`) and plain instantiations keep their arithmetic; KDA's main loop computes its gates into registers first (KDA models only) | the GDN kernel's time at Flash-Next's verify (the SASS comparison shows the kernels' changes) |
| #32965 64-bit matrix offsets in `mul_mat_vec_f` | the F32/F16/BF16 matrix-vector kernel computes its row, channel and sample offsets in 64 bits | Flash-Next's F32 shared-expert gate at 4-8 tokens on `mul_mat_vec_f` (`GGML_CUDA_MMVF_F32_MT=1` in its preset): register use, the step time |
| #32854 size-1 strides in `bin_bcast` | a stride of a dimension of size 1 is set to 0 before the launch | none expected (the same elements) |
| #32897 cooperative softmax with more than 256 SMs | only above 256 SMs | none (5 SMs) |
| #32875 NVFP4 dequantization | 8 blocks per CTA | none (no NVFP4 file) |
| #32805, #32837 HIP MMQ and MMVQ choices | AMD only | none |

## Graph, memory, server and host

| Change | What it does | Watch |
|---|---|---|
| #32378 no CPU polling without OpenMP | the CPU threadpool pauses after a graph with several splits, in builds without OpenMP | the rig build links OpenMP, so none expected |
| #32859 faster BPE tokenization | the vocabulary's merge lookups | the same tokens (the replies check); prompt tokenization on the Celeron |
| #32655 softplus through `log1pf` | the CPU's softplus helper only (CUDA unchanged) | the processor's logits of recurrent models in `test-llama-archs` |
| #30997, #28992, #32828 server prompt cache | a failed restore retries after clearing idle slots; a cache lookup no longer skipped; an entry kept until its restore succeeds | the presets run `--cache-ram 0`, so none; with a RAM cache an entry's data now stays until the restore succeeds (host memory on a 4 GB rig) |
| #32064 MCP image and audio in tool results | server tool results | none |
| #32218, #32855, #32940, #32948 Inkling 2 (DSpark drafts, graph changes) | another architecture | none for the release files |
| #32941 Qwen3-Omni conversion, #32903 symlinked build folders, #32928 llguidance header, #32870 MSVC test builds | build and conversion | none |
| #32142, #30867 Arm CPU kernels (no dot product, SVE) | Arm only | none (x86) |
| web UI (#32945, #32952, #32949 and others) | Llama Chat runs apart from llama-server, named in config.json | the rig build pins the page to b12723 (`HF_UI_VERSION`) |
