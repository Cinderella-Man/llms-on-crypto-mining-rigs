# Miner Llama {{VERSION}}: llama.cpp for P106 mining rigs and other Pascal GPUs

A llama.cpp build for NVIDIA GPUs of compute capability 6.1: the P106 and P104 mining cards, the GTX 10 series and the
Tesla P4/P40. Built from commit `{{COMMIT}}` of this fork, for Linux on x86-64 without AVX, so it runs on mining rigs'
Celeron hosts. This release is tuned and measured on Flint, on nine P106-090 cards; other compute-6.1 cards run the same
code but were not measured. Measurements, settings and the scripts behind every number:
[llms-on-crypto-mining-rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

## Speed

**Flint in near-lossless 8-bit on nine mining cards at the full 262,144-token context, about three times as fast as
stock llama.cpp for a coding agent.** Unsloth's Q8_0 (42.3 GB) picks the original model's next token 93.3 % of the
time (KL divergence 0.052 against the 16-bit weights), against 87.7 % (0.120) for the common 4-bit file UD-Q4_K_XL; it
runs 9–17 % slower than the 4-bit files. The fastest file in this test is the project's own
[Flint-Pascal-Q5](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) (29.4 GB, 88.8 %): closer to the original
than UD-Q4_K_XL, a little further than Unsloth's UD-Q5_K_XL (90.7 %), which ran at Q8_0's speed here (44.2 tok/s).

A coding agent ([pi](https://github.com/earendil-works/pi)) fixing twelve small bugs, Python and JavaScript, real test
runs, the context growing every turn, on nine P106-090 cards (one PCIe 1.0 lane each, a 2-core Celeron with 4 GB of
RAM). Generation in tokens per second over every request, and minutes per task (median):

| | Stock llama.cpp | **{{VERSION}}** |
|---|---|---|
| **Unsloth Q8_0**, `-ts 6,6,5,5,6,5,5,6,6` | 14.4 tok/s, 2.9 min (65,536-token context) | **44.6–45.9 tok/s, 1.1–1.4 min** (262,144) |
| Flint-Pascal-Q5 | 15.6 tok/s, 2.1 min | **53.8 tok/s, 1.2 min** (262,144; all twelve tasks in 17 minutes) |
| Unsloth UD-Q4_K_XL | 14.0 tok/s, 3.1 min | **50.7 tok/s, 1.1 min** (262,144) |
| Served chat, the store and the staircase drafter (`--pipe-guess staircase`) | — | **RIG-TBD** | <!-- only:drafter -->

Stock llama.cpp runs at 65,536 tokens because at 262,144 its attention mask, built on the host, does not fit 4 GB of
RAM. Pipelined speculation alone is worth +52 % in this test (29.3 → 44.6 tok/s on Q8_0). Results match stock
llama.cpp within rounding (the kernels change only the order of additions), and pipelined speculation gives exactly the
tokens of plain decoding.

## What is new since v1

**One conversation, faster per token:**
- **One persistent kernel per MoE layer** at one token, instead of 13 launches: +5.6–8.8 % per token on Flint. It covers
  Flint's layer shapes; other models take the normal path.
- **32-bit activation sums** in the Pascal kernels, so llama.cpp's MoE down-projection rescale is dropped for one-token
  steps automatically on compute 6.1 (+9.6 % on Flint UD-Q4_K_XL). `LLAMA_NO_MOE_DOWN_RESCALE=0` keeps the rescale.
- **A Pascal decode-attention kernel** for attention of Flint's shape (head size 128, 8 query heads per KV head), in the
  build but off by default (`GGML_CUDA_FATTN_PASCAL=1` turns it on): it slowed generation in the release test.
- **Pascal one-token kernels for Q4_K and Q6_K** as well, next to Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS: the bulk of
  the common Q4_K_M downloads of other models (`GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K` switches either off).
- **Pascal kernels for steps of 2–4 tokens**, and the shared expert fused with its activation (+0.7 %).
- **Less host work per token on compute 6.1:** the attention block of one token in two launches. The embedding row
  looked up on the first card and one packed upload per card and token are in the build but off by default
  (`LLAMA_INPUT_ON_GPU=1`, `LLAMA_INPUT_UPLOAD=1`): they slowed generation in the release test.

**Pipelined speculation in llama-server:** on a layer split only one card works at a time, so guessed tokens enter the
idle cards while the later cards finish the current token, and each guess is checked by sampling the real token: the
output is the same as without it.
- `--pipe-depth N` guesses in flight (4 on nine cards), `--pipe-guess none|copy|cont`, `--pipe-copy-min N`.
- `--pipe-floor N` keeps at least N guesses in flight, with a filler token when the guesser has none, so idle cards
  don't fall to their base clock.
- `--pipe-store FILE`: a store of earlier outputs that the guesser consults after the conversation itself. <!-- only:drafter -->
- For requests sampled on the GPU (`-bs`) without repetition penalties; tool requests run pipelined until the tool call.

**Agents:**
- **Tool requests sample on the GPU:** one token is checked against the grammar on the host, and the full row of scores
  crosses the PCIe link only when the grammar rejects it. Before, every tool request sampled on the CPU and decoded
  12–16 % slower than chat. `LLAMA_GRAMMAR_HOST_SAMPLING=1` restores sampling on the host.
- **Prompt caching for agents:** `--prefix-cache DIR` saves the state of the shared system and tool prompt to a file the first time it is read and restores it into any conversation that starts with it, across server restarts; one checkpoint per request holds the whole sliding-window ring; `--checkpoint-gen` also takes checkpoints while generating, at `</think>` and `<tool_call>`; and a request no longer re-reserves the scheduler before its first token. Off unless set: in the agent test it cost 11 s of fixed time per task and gained nothing. <!-- only:q163-q111 -->

**Prefill, long context and several conversations:**
- **MoE tiles sized by tokens per expert** (upstream PR #31047): prefill +22–29 %.
- **The attention mask built on each card** (`LLAMA_GPU_KQ_MASK=N`: for micro-batches of N tokens or more), in the build but off by default: built for every micro-batch it slowed long prompts 3.6× for +2 % in generation.
- **A Pascal prefill-attention kernel** (head size 128 or 256, 8 query heads per KV head) that skips fully masked key tiles and reads a Q8_0 KV cache directly, without an F16 copy: prefill +10 % at 8,192 tokens and +3 % at 32,768 on Flint-Pascal-Q5; on by default on compute 6.1 (`GGML_CUDA_FATTN_TILE_PASCAL=0` turns it off). <!-- only:q163-q111 -->
- **Several conversations through one pipeline** (the `llama-pipespec` tool's streams mode): conversations staggered
  through the cards, and a new request's prefill chunks on CUDA streams below the generating tokens' priority, so they
  no longer stall the conversations that are generating.

<!-- begin:drafter -->
**The staircase drafter, a trained guesser for pipelined speculation:**
- A small transformer guesses Flint's next token from the hidden states Flint has already computed for the tokens in flight: the newest token has passed only the first card, older ones are further along, a staircase of depths. It sits behind the copy lookup and runs on the first card in its own CUDA stream; the last card still samples every token, so the output is the same as without it.
- It ships as two side files (~0.27 GB each), trained on Flint's answers to prompts whose answers may be published (the project's own code prompts on permissively licensed files, and public prompt sets cleared for it; RIG-TBD: the release heads' mix): one for nine cards and one, depth-general, for six cards and any other split of 3–12 cards (RIG-TBD: their file names). Add `--pipe-guess staircase --pipe-draft <file>` next to `--pipe-depth 3`.
- The engine builds the drafter's window of hidden states from the run's own layer split, so no further setting is needed.
- Served on nine and six cards: RIG-TBD tok/s, against RIG-TBD without the drafter and RIG-TBD without guessing. Every answer stays identical to plain decoding.
- It guesses for a request that generates alone, like the rest of pipelined speculation.
<!-- end:drafter -->

**Coming later (v2.1):** a trained guesser, the staircase drafter (`--pipe-draft FILE`), and a store of Flint's earlier answers (`--pipe-store FILE`). The code is in this build; the files come with a later release ([how the drafter works](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/drafter.md)). <!-- only:nodrafter -->

## Install

Requirements: Linux with glibc 2.38 or newer (tested on Ubuntu 24.04) and NVIDIA's proprietary driver 580
(`nvidia-driver-580`, not `-open`). No CUDA toolkit: the second asset carries the CUDA runtime libraries.

```bash
sha256sum -c SHA256SUMS
tar xzf {{BIN_ASSET}} && tar xzf {{CUDART_ASSET}}
{{PREFIX}}-{{VERSION}}/llama-server --list-devices
```

Both archives unpack into `{{PREFIX}}-{{VERSION}}/`. `scripts/p106_check.sh` in the repository checks the driver, the
cards, host memory, the model's read speed and the shipped command lines. Flint Q8_0 on nine cards, chat with the full
context:

```bash
{{PREFIX}}-{{VERSION}}/llama-server -m flint-Q8_0.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --host 127.0.0.1 --port 8080
```

`-ts 6,6,5,5,6,5,5,6,6` splits the layers by memory so that Q8_0 and the full context fit; for a 4-bit file
(UD-Q4_K_XL, or this project's Flint-Pascal-Q5 on six cards) drop it.

With the drafter, add `--pipe-guess staircase --pipe-draft <the nine-card file>` to that line (RIG-TBD: its name). <!-- only:drafter -->
The repository's Flint page explains every flag, the six-card and agent variants, and the speed and quality behind them.

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `{{BIN_ASSET}}` | {{BIN_MB}} MB | `{{BIN_SHA256}}` |
| `{{CUDART_ASSET}}` | {{CUDART_MB}} MB | `{{CUDART_SHA256}}` |
<!-- begin:drafter -->
{{DRAFT_ROWS}}
<!-- end:drafter -->

`BUILDINFO.txt` in the first archive records the commit, the build image, and that the binaries contain no AVX
instructions and need glibc 2.38 at most. A third archive carries `test-backend-ops`, llama.cpp's operator tests, for
anyone who wants to check the kernels on their own cards.

## Licences

The binaries are llama.cpp and ggml with this fork's changes, all under the MIT licence (`LICENSE`,
`THIRD_PARTY_NOTICES.md`). The CUDA runtime libraries in the second archive are NVIDIA's, redistributed unmodified under
the NVIDIA CUDA Toolkit End User License Agreement (`NVIDIA_CUDA_RUNTIME_NOTICE.md`). Model files are not part of this
release and keep their own licences.
The drafter files are models of this project, trained on Flint's outputs and hidden states, under Apache-2.0. <!-- only:drafter -->
