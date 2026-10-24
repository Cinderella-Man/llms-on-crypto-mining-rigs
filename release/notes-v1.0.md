# Pascal engine {{VERSION}}: llama.cpp for P106 mining rigs (the first release)

The engine as released on 2026-10-23, built from commit `{{COMMIT}}` of this fork for NVIDIA GPUs of compute capability
6.1, Linux on x86-64 without AVX. It doubled Flint's speed on nine P106-090 cards over stock llama.cpp: 27.97 against
14.20 tokens per second in a five-turn chat with Unsloth's UD-Q4_K_XL at the full 262,144-token context
([the benchmark](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/benchmarks/2026-10-23-flint-chat.md)).

## What it contains

- CUDA graphs on Pascal (upstream PR #27721).
- A launch shape for one-token products: 2 warps per block on wide rows, 4 on narrow ones.
- Pascal matrix-vector kernels for Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS.
- `LLAMA_NO_MOE_DOWN_RESCALE=1`, for Flint files whose expert down-projections are Q8_0, Q6_K, IQ4_XS, IQ4_NL or MXFP4
  only (never for UD-Q4_K_XL).
- `GGML_CUDA_PROFILE=<file>`: a CUDA-event profiler that works on mining cards, which NVIDIA's profilers refuse.

## Install

Requirements: Ubuntu 24.04 or later, or Debian 13 (glibc 2.38 or newer), and NVIDIA's proprietary driver 580. No CUDA
toolkit: the second asset carries the CUDA runtime libraries.

```bash
sha256sum -c SHA256SUMS
tar xzf {{BIN_ASSET}} && tar xzf {{CUDART_ASSET}}
{{PREFIX}}-{{VERSION}}/llama-server --list-devices
```

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `{{BIN_ASSET}}` | {{BIN_MB}} MB | `{{BIN_SHA256}}` |
| `{{CUDART_ASSET}}` | {{CUDART_MB}} MB | `{{CUDART_SHA256}}` |

## Licences

The binaries are llama.cpp and ggml with this fork's changes, under the MIT licence. The CUDA runtime libraries in the
second archive are NVIDIA's, redistributed unmodified under the NVIDIA CUDA Toolkit End User License Agreement.
