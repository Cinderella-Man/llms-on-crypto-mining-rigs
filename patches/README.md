# llama.cpp patches

Patches applied on top of upstream llama.cpp for the experiments in this repository. Build them with
[scripts/build_llamacpp_docker.sh](../scripts/build_llamacpp_docker.sh): copy this folder into the build work
directory (`~/llama-builds` by default) and pass the file, e.g.
`PATCHES=patches/moe-down-rescale-env.diff scripts/build_llamacpp_docker.sh <ref> --cuda-graphs-pascal`.
The CUDA-graphs patch for Pascal ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) is downloaded by the
build script itself (`--cuda-graphs-pascal`); when it no longer applies, the script makes the same one-line change.

The same changes are kept as one commit each on the `pascal-p106` branch of
[a llama.cpp fork](https://github.com/Cinderella-Man/llama.cpp/tree/pascal-p106), on top of upstream `d77256999`: CUDA
graphs on compute 6.1, the rescale switch, Flint's `prism.hadamard` support, `Q4_1_64` and the wide-rows launch shape.
New Pascal work lands there first and only there:
- **Pascal single-token matrix-vector kernels** for Q4_0, Q4_1, Q5_0 and Q8_0 (`ggml/src/ggml-cuda/mmvq-pascal.cu`,
  [design note](../docs/research/notes/pascal_mmvq_integration.md)). On by default on compute 6.1;
  `GGML_CUDA_MMVQ_PASCAL=0` switches them off, `=2` forces them on any NVIDIA GPU (for tests).
- **`GGML_CUDA_PROFILE=<file>`**: GPU time of every launch group from CUDA events, for the GPUs NVIDIA's profilers
  refuse (the CMP mining cards). Summarize with [scripts/cuda_profile_summary.py](../scripts/cuda_profile_summary.py).
- `test-backend-ops` cases at the shapes of a large MoE model.

Build it with `git -C ~/llama-builds/src remote add fork https://github.com/Cinderella-Man/llama.cpp` and
`TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106` (no `PATCHES`, no `--cuda-graphs-pascal`).

| Patch | What it does | Tested against |
|---|---|---|
| [moe-down-rescale-env.diff](moe-down-rescale-env.diff) | Lets the environment variable `LLAMA_NO_MOE_DOWN_RESCALE=1` skip the MoE down-projection rescale that llama.cpp applies to Flint, Mistral4 and Mellum. **Only safe when the expert down-projection is not Q4_0/Q4_1/Q5_0/Q5_1**: with Flint Q4_1 it produces NaN. | llama.cpp commit `d77256999` (2026-10-22); see [Flint benchmark](../docs/benchmarks/2026-10-22-flint.md#the-moe-down-projection-rescale) |
| [flint-prism-hadamard.diff](flint-prism-hadamard.diff) | Adds Flint to the architectures whose GGUFs may carry llama.cpp's `prism.hadamard` metadata (weights stored with a Hadamard folded into their input side; llama.cpp transforms their inputs at run time). Flint's graph sends every projection through the Hadamard-aware matmul helpers, as for the Qwen3 family already allowed. Used by [scripts/rotate_gguf.py](../scripts/rotate_gguf.py) `--online-down`. | llama.cpp commit `d77256999` (2026-10-22) |
| [q4_1_64-type.diff](q4_1_64-type.diff) | Adds `Q4_1_64`, Q4_1 with one scale and offset per 64 weights (4.5 bits per weight): CPU reference and dot product, CUDA matrix-vector product (MMVQ), dequantization and row lookup; no batched (MMQ) kernel, so prompt processing falls back to dequantize + cuBLAS. [scripts/q4_1_to_q4_1_64.py](../scripts/q4_1_to_q4_1_64.py) converts Flint's makers' Q4_1 to it without changing a weight. Passes `test-backend-ops` for `MUL_MAT`, `MUL_MAT_ID` and `GET_ROWS`. | llama.cpp commit `d77256999` (2026-10-22) |
| [mmvq-pascal-2warps.diff](mmvq-pascal-2warps.diff) | Uses 2 warps instead of 4 per block for single-token matrix-vector products (the idea of [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479), which no longer applies to this commit). It changes llama.cpp's generic parameter table, so build it for sm_61 only. | llama.cpp commit `d77256999` (2026-10-22) |
| [mmvq-pascal-2warps-widek.diff](mmvq-pascal-2warps-widek.diff) | Replaces the previous one: 2 warps per block at one token only for rows of 2048 values or more; narrower rows keep llama.cpp's 4 warps and its small-K rule, where 2 warps are slower (Q8_0 expert down-projection at K = 512: 81 against 43 GB/s). Measured tg128 on 6 cards: S4-L0 28.0 tok/s (26.3 with 2 warps everywhere), UD-Q4_K_XL 24.6 (24.8). Build it for sm_61 only. | llama.cpp commit `d77256999` (2026-10-22), applied after `q4_1_64-type.diff` |
