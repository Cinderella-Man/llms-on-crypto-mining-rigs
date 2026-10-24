# Pascal engine {{VERSION}}: llama.cpp for P106 mining rigs and other Pascal GPUs

A llama.cpp build for NVIDIA GPUs of compute capability 6.1: the P106 and P104 mining cards, the GTX 10 series and the
Tesla P4/P40. Built from commit `{{COMMIT}}` of this fork, for Linux on x86-64 without AVX, so it runs on mining rigs'
Celeron hosts. Measurements, settings per model and the scripts behind every number:
[llms-on-crypto-mining-rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

## Speed

Flint on nine P106-090 cards (one PCIe 1.0 lane each, a 2-core Celeron with 4 GB of RAM), tokens per second:

| | Stock llama.cpp | v1 engine | **{{VERSION}}** |
|---|---|---|---|
| Generation, Unsloth UD-Q4_K_XL (`llama-bench` tg128) | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Served chat, UD-Q4_K_XL | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Served chat, UD-Q4_K_XL, pipelined speculation (`--pipe-depth 3`) | — | — | **RIG-TBD** |
| Prompt processing, 512 tokens | RIG-TBD | RIG-TBD | **RIG-TBD** |

With a memory clock offset of RIG-TBD MT/s on every card, generation rises by RIG-TBD % on every engine (the
hardware-tuning section of the repository's README). The output does not change: the kernels change only the order of
additions, and pipelined speculation gives exactly the tokens of plain decoding.

## What is new since v1

- **One persistent kernel per MoE layer** at one token, instead of 13 launches: +5.6–8.8 % per token on Flint. It covers
  Flint's layer shapes and falls back to the normal path for other models.
- **Pipelined speculation in llama-server** (`--pipe-depth N`, `--pipe-guess none|copy|cont`): guessed tokens enter
  the idle cards of a layer split while the later cards finish the current token, each checked by sampling the real
  token. For requests sampled entirely on the GPU (`-bs`, no grammar, no repetition penalties).
- **32-bit activation sums** in the Pascal kernels, so llama.cpp's MoE down-projection rescale can go:
  `LLAMA_NO_MOE_DOWN_RESCALE=tg` (+9.6 % on Flint UD-Q4_K_XL).
- **Shared-expert projections fused** with their activation (+0.7 %).
- **MoE prompt tiles sized by tokens per expert** (upstream PR #31047): prompts +22–29 %.
- **The attention mask built on each card** for prompt batches.
- RIG-TBD: confirm or delete each of these after its own rig test: Pascal Q4_K and Q6_K one-token kernels (Q4_K_M
  files), a Pascal decode-attention kernel (`GGML_CUDA_FATTN_PASCAL`), a prompt-attention kernel, `--pipe-store`.

## Install

Requirements: Ubuntu 24.04 or later, or Debian 13 (glibc 2.38 or newer), and NVIDIA's proprietary driver 580
(`nvidia-driver-580`, not `-open`). No CUDA toolkit: the second asset carries the CUDA runtime libraries.

```bash
sha256sum -c SHA256SUMS
tar xzf {{BIN_ASSET}} && tar xzf {{CUDART_ASSET}}
{{PREFIX}}-{{VERSION}}/llama-server --list-devices
```

Both archives unpack into `{{PREFIX}}-{{VERSION}}/`. `scripts/p106_check.sh` in the repository checks the driver, the
cards, host memory, the model's read speed and the shipped command lines. The command line for each model is on its
page under `models/` in the repository; for Flint on nine cards:

```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg {{PREFIX}}-{{VERSION}}/llama-server -m flint-UD-Q4_K_XL.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 3 --host 127.0.0.1 --port 8080
```

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `{{BIN_ASSET}}` | {{BIN_MB}} MB | `{{BIN_SHA256}}` |
| `{{CUDART_ASSET}}` | {{CUDART_MB}} MB | `{{CUDART_SHA256}}` |

`BUILDINFO.txt` in the first archive records the commit, the build image, and that the binaries contain no AVX
instructions and need glibc 2.38 at most.

## Licences

The binaries are llama.cpp and ggml with this fork's changes, all under the MIT licence (`LICENSE`,
`THIRD_PARTY_NOTICES.md`). The CUDA runtime libraries in the second archive are NVIDIA's, redistributed unmodified under
the NVIDIA CUDA Toolkit End User License Agreement (`NVIDIA_CUDA_RUNTIME_NOTICE.md`). Model files are not part of this
release and keep their own licences.
