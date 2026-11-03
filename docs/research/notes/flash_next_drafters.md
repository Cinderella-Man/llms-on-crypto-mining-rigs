# Drafters for Qwen3.8-Flash-Next on the P106 rig: DFlash2 first (2026-11-03)

Speculative decoding is the multiplier after the per-token fixes on this rig: decode is GPU-bound at 136 ms per token
([Q215](../../data/2026-11-03_flash_next_baseline/README.md)), and a draft lets one pass of the model check several
tokens. The first draft tried, agentionai's Gyro MTP draft, ran the rig's 4 GB host out of memory while it loaded
([Q226](../../data/2026-11-03_flash_next_mtp/README.md)): its 644 MiB of token embeddings stay in host memory, and its host
buffers were 788 MiB against 379 MiB without it. This note looks at every drafter published for Flash-Next and ranks
them for a 4 GB host. Nothing here was run on the rig yet; claims about llama.cpp are read from its source at the
fork's upstream base (`d77256999`, build 11820, 2026-10-22) and from the linked pages, and estimates are marked est.

## Summary

1. **Inco AI's DFlash2 drafter suits the 4 GB host.** It is 0.6B parameters in 5 layers and shares the target's token
   embeddings and output head, so it adds only est. 50–150 MiB of host memory. Gyro adds 644 MiB of embeddings plus
   ~400 MiB of buffers. Its files are 334–652 MB.
2. **llama.cpp runs it with one fix the fork lacks.** DFlash, DFlash2, recurrent rollback for `qwen4exp` and
   output-head sharing are all in the fork's base. The fix for Flash-Next's four-stream residual,
   [#30927](https://github.com/ggml-org/llama.cpp/pull/30927), merged two commits later. Without it the first request
   aborts at `GGML_ASSERT(row_floats == n_embd)`. The fix is one file (+19/−1).
3. **Est. 1.3–1.5× as a standard draft on this rig.** On MoE targets DFlash has done poorly (GPT-OSS-20B 0.61–1.27×,
   [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)), because a wide verify reads many more experts. On this
   rig, though, Flash-Next's experts are only 28 of the 136 ms per token.
4. **Est. 1.8–2.1× as the guesser for pipelined speculation**
   ([Q550](../../data/2026-11-03_flash_next_pipelined/README.md)). One draft pass gives the guesses that the idle cards
   run, and it needs no training.
5. **The stock n-gram guessers don't suit Flash-Next.** They keep no spare recurrent state, so the server saves a
   113 MiB checkpoint every round. Pipelined speculation is the draft-free route here.

## The drafters

| Repository | Size | Type | Shares embeddings and head | llama.cpp | Licence |
|---|---|---|---|---|---|
| [incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF) | Q4_K_M 362 MB … Q8_0 652 MB, BF16 1.22 GB | DFlash2, 0.6B | yes | `--spec-type draft-dflash -md` (needs #30927) | Apache-2.0 |
| [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF) `dflash-*` | Q4_0 351 MB, Q8_0 652 MB | the same drafter | yes | the same | other |
| [nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) | Q4_0 334 MiB, Q8_0 622 MiB | the same drafter | yes | the same | Apache-2.0 |
| ggml-org, [bartowski](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) `mtp-*` | Q4_0 2.20 GB, Q8_0 4.14 GB | MTP | no (est., from the sizes) | `draft-mtp` | other / not checked |
| [unsloth `MTP/`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) | Q4_K_M 2.79 GB; `shared-` Q4_K_M 1.91 GB | MTP | only the `shared-` files, which need Unsloth's fork | `draft-mtp` | qwen-community-1.0 |
| [agentionai Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 2.70 GB | MTP | no (644 MiB in host memory) | `draft-mtp` | not checked |
| [shadowpa0327/Qwen3.8-Flash-Next-DSpark](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark) | 0.66B, BF16 | DSpark, block 7, trained on coding-agent sessions | yes | `draft-dspark` exists; GGUF conversion untested | other |
| [PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash) | 0.5B, BF16 | DFlash, block 7 | yes | none (vLLM only) | other |

No EAGLE-3 drafter for Flash-Next is published, and Z-Lab publishes a DFlash2 drafter only for Qwen3.8-27B.

## How llama.cpp runs DFlash2

- **The drafter** ([card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
  [config](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/main/config.json)): 5 layers at hidden size
  2,560, trained with blocks of 8 (the card evaluates blocks of 4). It reads the residual rows of target layers
  5/15/25/35/45 and has a 2,048-token sliding window. The token embeddings and output head come from the target,
  through `ctx_other` (`src/llama-context.cpp`, `src/models/dflash.cpp`).
- **Already in the fork's base:** DFlash ([#22105](https://github.com/ggml-org/llama.cpp/pull/22105), 2026-06-28),
  DFlash2 ([#27816](https://github.com/ggml-org/llama.cpp/pull/27816)), rollback of `qwen4exp`'s recurrent state
  ([#28123](https://github.com/ggml-org/llama.cpp/pull/28123)), output-head sharing
  ([#30111](https://github.com/ggml-org/llama.cpp/pull/30111)) and `--spec-draft-n-ubatch`
  ([#30459](https://github.com/ggml-org/llama.cpp/pull/30459)).
- **Missing:** [#30927](https://github.com/ggml-org/llama.cpp/pull/30927) (upstream commit `d9ee007a0`). `qwen4exp`
  exposes its four-stream residual as the layer input, which the tap code does not expect.
- **Block width:** it comes from the GGUF key `dflash.block_size`. `--spec-draft-n-max` (default 3) is clamped to the
  block minus one, so up to 7 for this drafter (`common/speculative.cpp`).
- **Rollback costs VRAM:** `draft-dflash` keeps 1 + `n_max` recurrent-state planes per sequence, about 113 MiB of VRAM
  per drafted token over all cards (`common/common.h`, `src/llama-memory-recurrent.cpp`). Three drafted tokens need
  ~340 MiB more VRAM.
- **Host memory:** the tap buffers hold taps × hidden size × `-b` floats in the output card's pinned host buffer
  (`src/llama-context.cpp`): 100 MiB at `-b 2048` and 25 MiB at `-b 512`. A published measurement at `-ub 1024`
  (Metal) gives the Q4_0 drafter 324 MiB of weights, 60 MiB of KV cache and a 2,926 MiB compute buffer, against
  2,087 / 320 / 891 MiB for the MTP head ([nazeshinjite](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF)).
  The compute buffer scales with the vocabulary times the draft micro-batch, so est. `--spec-draft-n-ubatch 32–64`
  brings it to ~0.1–0.2 GB.

## Reported speedups (batch 1)

| Setup | Result | Source |
|---|---|---|
| DFlash2 in SGLang, GB300, temperature 1.0 | 1.63–1.82× (MTP 1.35–1.59×), acceptance length 2.83–3.26 (MTP 2.50–3.08) | [incoai](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| DFlash2 in llama.cpp, GB10, UD-Q3_K_XL, `n_max` 4 | 30.44 → 47.82 tok/s | [spiceai #4](https://github.com/spiceai/llama.cpp/pull/4) |
| DFlash2 against MTP, M5 Max, temperature 1.0 | +17 % geometric mean; −1 to −3 % on prose | [nazeshinjite](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) |
| DFlash2, #30927's own test | acceptance length 2.37; per position 0.65 / 0.42 / 0.30 | [#30927](https://github.com/ggml-org/llama.cpp/pull/30927) |
| MTP, RTX PRO 6000 | 108 → 183 tok/s on code, 144 on prose | [#28123](https://github.com/ggml-org/llama.cpp/pull/28123) |
| MTP, B200 | 1.34–1.67× | [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) |
| DFlash on an MoE (GPT-OSS-20B) | 0.61–1.27× | [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) |

**On the rig (est.):** a 4-token verify costs ~200–230 ms (dense weights read once, experts ×3.5) and a draft pass
~5–8 ms. At an acceptance length of ~2.4 that is 1.3–1.5× before any acceptance lost to the 2-bit target. The dense
fixes under way (BF16 kernel Q546, retype Q218) lower the dense share further, which favours wider verifies.

## The other options

- **MTP drafts:** none that loads in Miner Llama is smaller than Gyro's 2.7 GB. Unsloth's `shared-` heads leave out
  the embeddings and head, but Miner Llama requires `token_embd` in a `qwen4exp` MTP file. Gyro's 644 MiB can move to
  a card with `-otd 'token_embd=CUDA<n>'`, since tensor overrides apply to the input layer too
  (`src/llama-model-loader.cpp`), or with the fork's `LLAMA_INPUT_ON_GPU=1`. That makes MTP the fallback
  ([Q558](../../../quants/experiments.md)).
- **Stock n-gram guessers** (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`): they leave
  `n_rs_seq` at 0, so on Flash-Next the server saves a partial-state checkpoint of ~113 MiB every round
  (`tools/server/server-context.cpp`). Over PCIe 1.0 x1 that costs est. 0.15–0.6 s a round, more than a token. Mixing
  n-gram drafts with a draft model corrupts the rollback until
  [#31350](https://github.com/ggml-org/llama.cpp/pull/31350) (2026-10-29).
- **DSpark** is trained on coding-agent sessions (Open-SWE-Traces), which could suit agentic work, but it is published
  for SGLang only, and a GGUF conversion for llama.cpp's `draft-dspark` is untested.

## Ranked for the rig

1. **DFlash2 as a standard draft** (Q561): the #30927 fix on `pascal-p106-v3`, then `--spec-type draft-dflash
   --spec-draft-n-max 3 --spec-draft-n-ubatch 64 -b 512`, with the drafter on the card that holds the output head
   (`-devd`, `-ngld 99`). Est. +50–150 MiB of host memory, ~0.6 GB of VRAM on one card plus ~340 MiB of state planes;
   est. 1.3–1.5×.
2. **The same drafter as pipelined speculation's guesser** (Q554, Q271): one pass over [token, MASK × 3] gives three
   guesses for depth 3. Fork work, since the pipe keeps its own state groups. Est. 1.8–2.1×.
3. **MTP with its embeddings on a card** (Q558), as the fallback.
4. **Not** the stock n-gram guessers on Flash-Next (Q562, parked).

The registry rows are Q554, Q558, Q561–Q563 and Q566–Q568 in [quants/experiments.md](../../../quants/experiments.md).
