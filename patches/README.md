# llama.cpp patches

Patches applied on top of upstream llama.cpp for the experiments in this repository. Build them with
[scripts/build_llamacpp_docker.sh](../scripts/build_llamacpp_docker.sh): copy this folder into the build work
directory (`~/llama-builds` by default) and pass the file, e.g.
`PATCHES=patches/moe-down-rescale-env.diff scripts/build_llamacpp_docker.sh <ref> --cuda-graphs-pascal`.
The CUDA-graphs patch for Pascal ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) is downloaded by the
build script itself (`--cuda-graphs-pascal`).

| Patch | What it does | Tested against |
|---|---|---|
| [moe-down-rescale-env.diff](moe-down-rescale-env.diff) | Lets the environment variable `LLAMA_NO_MOE_DOWN_RESCALE=1` skip the MoE down-projection rescale that llama.cpp applies to Flint, Mistral4 and Mellum. **Only safe when the expert down-projection is not Q4_0/Q4_1/Q5_0/Q5_1**: with Flint Q4_1 it produces NaN. | llama.cpp commit `d77256999` (2026-10-22); see [Flint benchmark](../docs/benchmarks/2026-10-22-flint.md#the-moe-down-projection-rescale) |
| [flint-prism-hadamard.diff](flint-prism-hadamard.diff) | Adds Flint to the architectures whose GGUFs may carry llama.cpp's `prism.hadamard` metadata (weights stored with a Hadamard folded into their input side; llama.cpp transforms their inputs at run time). Flint's graph sends every projection through the Hadamard-aware matmul helpers, as for the Qwen3 family already allowed. Used by [scripts/rotate_gguf.py](../scripts/rotate_gguf.py) `--online-down`. | llama.cpp commit `d77256999` (2026-10-22) |
