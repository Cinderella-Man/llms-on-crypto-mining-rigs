# Miner Llama v2.0: every engine feature, how it was checked, what it measured

Miner Llama v2.0 is the fork's tag `miner-llama-v2.0` (commit `37e717ff6` on branch `pascal-p106-v2` of
[Cinderella-Man/llama.cpp](https://github.com/Cinderella-Man/llama.cpp)). This page lists each engine feature that v2.0
adds to v1, the registry row that measured it ([quants/experiments.md](../quants/experiments.md)), how its correctness
was checked, what it measured on the nine-card P106-090 rig, and whether v2.0 runs it by default. The switches
themselves are in [reproduce/engine.md](../release/flint-v2/reproduce/engine.md#switches); the user-facing summary is in
the [release notes](../release/notes-v2.0.md).

Speeds are llama-bench tg128 (tokens per second, one token at a time) on nine cards with Flint K2h-r16 (the file
released as Flint-Pascal-Q5) unless a row says otherwise. They are diagnostics: the release numbers are agentic speed
from [the agent check](../bench/agent_check/README.md), in [results.md](../release/flint-v2/results.md).

## Features

| Feature | Registry | Correctness check | Rig result | In v2.0 |
|---|---|---|---|---|
| FP32 activation sums in the Pascal one-token kernels, so the MoE down-projection rescale is dropped for one-token steps | Q76 | one-token perplexity identical with and without the rescale (23.2625, 4 × 512 tokens) | UD-Q4_K_XL on six cards 33.99 → 37.26 tok/s (+9.6 %) | on (`LLAMA_NO_MOE_DOWN_RESCALE` = `tg` on compute 6.1) |
| Shared expert fused with its activation | Q50 | exact | UD-Q4_K_XL on six cards 33.99 → 34.24 tok/s (+0.7 %) | on |
| Persistent kernel per MoE layer at one token | Q88 | bit-exact against the per-op path (`GGML_CUDA_PERSISTENT_MOE_CHECK=1`); pipelined output identical at depths 1–4 | 39.3 → 41.8 tok/s against the combined build (Q160); switching it off on the release candidate cost 8 % | on, for Flint's layer shapes |
| Pascal kernels for steps of 2–4 tokens | Q78 | bit-identical to the one-token products, natively and through the compute_61 PTX ([bench/mmvq_pascal/verify](../bench/mmvq_pascal/verify/)) | not timed on its own | on |
| Pascal one-token kernels for Q4_K and Q6_K | Q149, Q101 | `test-backend-ops` 536/536; Qwen3.6 end to end KLD 0.000000 with the kernels alone | see Q149 | on |
| Pipelined speculation in `llama-pipespec` and llama-server (`--pipe-depth`) | Q87, Q160 | every pipelined run identical to plain decoding (89 runs in the executor's tests; 34/34 jobs on the release build; server check 13/13) | served `--pipe-depth 3`: median 48.5 tok/s against 41.8 plain (Q160); [corrected per-depth tables](data/2026-10-24_rig_c7recheck/) | flag (`--pipe-depth 4` in the release command) |
| Fed tokens, KV-step stall and filler floor fixes for pipelined serving | Q153, Q154, Q156 | CPU identity | fed tokens +4.6 % at 4 conversations, −2.8 % at 8 | on (`--pipe-floor` is a flag) |
| Prompt chunks on CUDA streams below the generating tokens' priority | Q155 | `check_streams.sh` 84/84 with and without the second stream | 8 conversations 262.6 tok/s in total, exact | on (`llama-pipespec` streams mode) |
| Attention block of one token in two launches | Q152 | the release build.s identity runs | no measurable effect (36.5 off against 36.8 on, release candidate) | on |
| MoE prefill tiles sized by tokens per expert (upstream PR #31047) | Q108 | logits identical | prefill +22–29 % | on |
| Pascal prefill-attention kernel (head size 128 or 256), reading a Q8_0 cache directly | Q163 | `FLASH_ATTN_EXT` 4172/4172 with the kernel forced; 16k logits: Flint KLD 0.052, equal to stock's own Q8_0-cache floor; Qwen3.6 0.0021 | prefill +10 % at 8,192 tokens, +3 % at 32,768 | on (`GGML_CUDA_FATTN_TILE_PASCAL`) |
| GPU sampling for tool requests (one token checked against the grammar on the host) | Q67 | device against host sampling 32/32, host switch 16/16, forced rejections 16/16, pipelined to the tool call 16/16 | tool requests +13–15 % (T 0: 33.6 → 37.9; T 1: 39.6–40.0 → 45.3–46.0), same tool calls | on |
| Pascal decode-attention kernel (head size 128, 8 query heads per KV head) | Q151 | `FLASH_ATTN_EXT` 24/24; identity 15/15; Flint KLD 0.052 against a 0.047 summation-order floor | **slower**: −9 % at short context, and 15.0 against 25.4 tok/s after 32k tokens | built, off (`GGML_CUDA_FATTN_PASCAL=1`) |
| Embedding lookup on the first card | Q150 | identity | within the noise of the release candidate (37.8 off against 36.8 on) | built, off (`LLAMA_INPUT_ON_GPU=1`) |
| One packed upload per card and token in plain decoding | Q150 | identity; see [the crash](#the-pipelined-decoding-crash) | within the noise (37.4 off against 36.8 on) | built, off (`LLAMA_INPUT_UPLOAD=1`); the pipelined path packs by default (`LLAMA_PIPE_UPLOAD`) |
| Attention mask built on each card | Q110, Q116 | `LLAMA_GPU_KQ_MASK_CHECK` | decode +1.6 % at short context and +2.3 % after 32k, but long prompts 3.6× slower when built for every micro-batch | built, off (`LLAMA_GPU_KQ_MASK=N`) |
| Eight scheduler copies for prompts instead of four | Q157 | — | no gain (8k / 32k prompts 765 / 462 against 766 / 461 tok/s) | build option, not shipped |
| Prompt caching for agents (`--prefix-cache`, `--checkpoint-gen`) | Q111, Q132 | CPU bit-exact | 11 s of fixed time per task in the agent check and no gain | built, off |
| Screened output head (`LLAMA_SCREEN_HEAD`) | Q92, Q125 | exact | +1.5–2.3 % | built, no screen file shipped |
| Store of earlier answers (`--pipe-store`) and the staircase drafter (`--pipe-draft`) | Q161, Q20 | CPU identity; the drafter's full exactness set | see [docs/drafter.md](drafter.md) | built, files in a later release |
| Norm writes the Q8_1 activations and absorbs the residual add | Q100 | operator tests | turning it off (`GGML_CUDA_NORM_Q8=0`) aborted with a CUDA-pool assertion on the release candidate: leave it on | on |

## How the release defaults were chosen

The first v2 release candidate had every feature on and decoded slower than the unified build it came from (36.8
against 41.8 tok/s). Four rounds on the rig settled the defaults; the tables and scripts are in
[docs/data/2026-10-25_release_rig](data/2026-10-25_release_rig/).

1. **One switch off at a time** (llama-bench tg128, K2h-r16, nine cards; all on: 36.8): Pascal one-token kernels off
   31.9 (they give +15 %), persistent MoE kernel off 34.0 (+8 %), **decode attention off 40.3 (it cost 9 %)**,
   attention block off 36.5, packed uploads off 37.4, embedding on the host 37.8, card-built mask off 36.7 (no effect
   at short context). Flint Q8_0 decoded at 31.0 against v1's 32.9 and stock's 18.3.
2. **The slow pieces off together:** decode attention, embedding on the card and packed uploads off gave tg128 36.8 →
   40.3, 15.0 → 25.4 after 32k tokens, and pipelined depth 3 chat / code / edit / rewrite 41.5 / 42.8 / 46.1 / 70.0 →
   45.6 / 48.4 / 51.6 / 106.5 tok/s, 34 of 34 jobs identical to plain decoding. Flint Q8_0: 34.9 tok/s.
3. **Prompt reading:** a whole 8,192-token prompt (`-p 8192 -n 0 -b 8192 -ub 512`) read at 211 tok/s against 714 on the
   combined build and 593 on stock. A bisect over the builds (stock 602, combined 721, unified 720, the next base 738,
   first release candidate 188) put it on the card-built attention mask, built for every micro-batch; without it the
   release build read 764 tok/s. v2.0 turns the mask off unless `LLAMA_GPU_KQ_MASK` is set.
4. **v2.0 itself** (`37e717ff6`): pipelined jobs 34/34 identical, server checks 13/13, 13/13 and 6/6. K2h-r16 reads an
   8k prompt at 767 tok/s and decodes at 38.9; Flint Q8_0 801 and 33.3. Whole-prompt batches of 8k / 32k: 766 / 461 tok/s
   (prefill attention off: 699 / 447). The card-built mask, alternated with the same settings: decode off 40.5 / 40.3,
   on 41.3 / 40.8; after 32k off 25.1, on 25.7. Against 3.6× slower long prompts, it stays off.

## The pipelined-decoding crash

A release candidate crashed in pipelined decoding: llama-server with `--pipe-depth` died on the third reuse of a cached
prompt (a chat UI's "regenerate"), and `llama-pipespec` after three jobs, both with a segfault in
`ggml_gallocr_alloc_graph`. The cause: packed input uploads had become the default on compute 6.1. With two or more
cards the scheduler keeps four copies of each split input as graph leaves, packing gives the current copy its memory
before the allocator plans, and `ggml_gallocr_needs_realloc` checked only a graph's nodes, not its leaves, so a later
graph of the same shape with another current copy read a stale allocation. The fix in v2.0 (`2d53179aa`): the
allocator also checks the leaves. On the rig the fixed build ran 34 of 34 pipelined jobs identical to plain decoding and
the server's regenerate pattern 13 of 13. On other GPUs packed uploads were off by default, so the crash appeared only
on the P106 rig. The reproduction scripts are in [docs/data/2026-10-25_regen_crash](data/2026-10-25_regen_crash/).
