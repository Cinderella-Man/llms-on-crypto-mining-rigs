# The v5 switch merge: C861, C865 and C866 in `pascal-p106-v5` (2026-11-17)

**Built on a CPU, not yet run on a rig.** The v5 line's three finished switch branches are merged into
`pascal-p106-v5`, each off by default, so every v5 engine change is in one branch and one rig build. With every new
switch unset, every kernel a P106 runs is v4.0's machine code. The three rig rows that test these switches
(C861's, C865's and C866's) can all use this one build. Every switch is in [docs/switches.md](../../switches.md).

- Fork branch [`pascal-p106-v5`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5) at **`a112712e3`**:
  three merge commits on `eb82916d3` (the v4.0 tag `miner-llama-v4.0`), a fast-forward of the branch.
- Rig build `llama.cpp-20261117-a112712e3-pascal-p106-v5.tar.gz` (838,929,497 bytes), md5
  **`d34dfe3a64d195d8dd068cfb3485c523`**. CUDA 12.9.2 image, `sm_61`, 0 AVX instructions in `llama-bench`, highest GLIBC
  symbol `GLIBC_2.38`, 34 compiler warning lines (the three branches' builds: 34 each).

## The merge

`git merge --no-ff` of each branch, in the order the branches were built. No conflicts.

| Commit | Merge of | Brings (off by default) | Registry |
|---|---|---|---|
| `355b3cf4a` | `pascal-p106-v5-kq-fp32scale` `abd6a2deb` ([C861](../2026-11-17_c861a_fp32_scale/README.md)) | `GGML_CUDA_MMQ_KQ_FP32SCALE` and `_TYPES`: the K-quant prompt kernels multiply each scale by its integer sum in FP32 | Q1549 |
| `b56fa04e7` | `pascal-p106-v5-mmid-glu-unfuse` `9550c62d6` ([C865](../2026-11-17_c865_glu_unfuse/README.md)) | `GGML_CUDA_MMID_GLU_UNFUSE`: the routed experts' gate, up and GLU as separate launches on Pascal | Q1559 |
| `a112712e3` | `pascal-p106-v5-keepalive-key` `087eb3b67` ([C866](../2026-11-17_c866_keepalive_key/README.md)) | `GGML_CUDA_KEEPALIVE_EAGER=3`: the keep-alive's new-shape key without the cache's length | Q1562 |

The merged tree is exactly the three branches' changes together:
- the 13 files only one branch changes are that branch's files byte for byte (C861: `mmq-vec-dot.cuh`, `mmq.cu`,
  `mmq.cuh`, `vecdotq.cuh`; C866: `fattn-sparse-pascal.cu`, `.cuh`, `fattn.cu`, `.cuh`, `keepalive-policy.h`,
  `keepalive.cu`, `top-k.cu`, `.cuh`, `tests/test-cuda-keepalive.cpp`);
- `ggml-cuda.cu`, which C865 and C866 both change, differs from C866's file by exactly C865's patch and from C865's
  file by exactly C866's patch (the same `git patch-id`): the two change different functions.

## The build

```bash
WORK=<build folder> TESTS=1 JOBS=8 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" \
  DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12360" \
  scripts/build_llamacpp_docker.sh fork/pascal-p106-v5
```

**Why `HF_UI_VERSION`:** the build embeds llama-server's web page from upstream's prebuilt archive
(Hugging Face bucket `ggml-org/llama-ui`), at the version `b<commit count>`. The fork's commit counts (12,840 for
`a112712e3`) are not upstream releases, so the build takes the archive `latest` instead. At 16:03 that archive had no
`workbox-*.js` file, which this tree's asset check (`scripts/ui-assets.cmake`) requires, and the build stopped at
`llama-ui-assets`; C866's build at 14:47 had passed the same check. `HF_UI_VERSION=b12360`
(the upstream release v4.0 sits on) makes the build take that release's archive (sha256 `6d98e0c6…`, with
`workbox-b3c04f83.js`). It changes only the web page llama-server serves. Making this pin the build script's default
is Q1577 in [quants/experiments.md](../../../quants/experiments.md).

| File in `bin/` | md5 |
|---|---|
| `llama-server` | `42f0b1f2ff2de02cd2e83836b5ff9841` |
| `test-backend-ops` | `cfd5c96a77f587c10e22071889e98107` |
| `test-llama-archs` | `f91fe740b65095ada20734ce828e3de9` |
| `test-cuda-keepalive` | `34d8497a3700670ab2d9768037bc85b9` |
| `llama-bench` | `097d816c146d6c9a8199d7f997c1b13b` |
| `llama-completion` | `3ea2f11d3e6b4dba268a7570618f1c24` |

A trial build of the same commit from a local branch, made before the branch was pushed, gave the same six files
byte for byte.

## The machine code with every switch unset

`run/sass_check.sh` disassembles v4.0's release `llama-server` and this build's with `cuobjdump -sass` (in the CUDA
image, on a CPU) and hashes each kernel's SASS with C861's `sass_compare.py`
([its run folder](../2026-11-17_c861a_fp32_scale/run/)). Full output: [sass_identity.txt](sass_identity.txt).

- **8,097 of v4.0's 8,797 kernels are byte for byte the same.** None of the 5,080 kernels over 200 instructions
  differs.
- **The other 700 are stubs that only report an error:** `mul_mat_q`'s `NO_DEVICE_CODE` branch, compiled for the
  template combinations that have no configuration on compute 6.1, at most 30 instructions each. They differ in the source line number they print
  (1083 in v4.0, 1102 here, because C861 added lines above it) and, in 132 of them, in the addresses of message
  strings. They are the same 700 that C861's own build changed, and their code is byte for byte C861's build's.
- **New kernels, only reached with `GGML_CUDA_MMQ_KQ_FP32SCALE` set:** 176 `mul_mat_q` kernels with C861's FP32 form
  (Q3_K 32, Q4_K 48, Q5_K 48, Q6_K 48), the same set as C861's build, with the same instruction counts
  ([kq_instructions.tsv](../2026-11-17_c861a_fp32_scale/kq_instructions.tsv)).
- C865 and C866 change host code only, as their own builds showed.

## Files

| File | What |
|---|---|
| [sass_identity.txt](sass_identity.txt) | the per-kernel comparison with v4.0, the 700 stubs' line-by-line classification, the comparison with C861's build |
| [run/sass_check.sh](run/sass_check.sh) | the comparison: `OUT=<folder> MNT=<folder holding both> BASE=<v4.0 llama-server> NEW=<this llama-server> C861=<C861's sass folder> bash run/sass_check.sh`, with `sass_compare.py` copied into `OUT` |

Everything ran on a build machine's CPU in Docker (`--cpu-shares=256`, `-j 8`, one build at a time); nothing ran on a
GPU. Licence: the scripts are this repository's.
