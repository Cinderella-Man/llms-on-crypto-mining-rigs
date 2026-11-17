# The v5 candidate on upstream b12410: `pascal-p106-v5-cand-b12410` (2026-11-17)

**Built on a CPU, not yet run on a rig.** Two v5 branches are joined into one candidate: `pascal-p106-v5` with its
three switches ([the switch merge](../2026-11-17_v5_merge_switches/README.md)) and C854's merge of upstream
llama.cpp b12410 ([C854 step one](../2026-11-17_c854_upstream/README.md)). With every switch unset, its GPU code is
C854's build kernel for kernel. `pascal-p106-v5` moves to this commit once the candidate passes its checks on a P106
(the kernel suite, the release files' greedy replies and launches a token against v4.0). Every switch is in
[docs/switches.md](../../switches.md).

- Fork branch [`pascal-p106-v5-cand-b12410`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-cand-b12410)
  at **`1f1110c67`**: one `--no-ff` merge of `pascal-p106-v5-b12410-up` (`b3dfb0d89`) on `pascal-p106-v5`
  (`a112712e3`). Its first parent is `pascal-p106-v5`'s tip, so moving `pascal-p106-v5` here is a fast-forward.
- Rig build `llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410.tar.gz` (840,761,714 bytes), md5
  **`5bdcfc21c6b4b0f59a9f7a1d8fed2c08`**. CUDA 12.9.2 image, `sm_61`, 0 AVX instructions in `llama-bench`, highest
  GLIBC symbol `GLIBC_2.38`, 34 compiler warning lines (the same lines as both parents' builds).

## The merge

No conflicts. Since the v4.0 commit (`eb82916d3`) the switches change 14 files and the upstream merge 177; two files
are changed by both, in different places:

| File | Upstream b12410's change | The switches' change |
|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the fused RMS norm's type checks and `supports_op` take BF16 (#32135) | C865's routed gate/up/GLU switch, C866's keep-alive key |
| `ggml/src/ggml-cuda/fattn.cu` | attention masks with a head dimension (#32217): no GQA shortcut for them, and the kernel choice refuses them only for row-list masks and soft-capped head sizes other than 128 and 256 | C866's `ggml_cuda_flash_attn_ext_length_choice`, which calls the kernel choice, so `GGML_CUDA_KEEPALIVE_EAGER=3`'s key follows the new rules |

The merged tree is exactly both sides' changes together:
- every file only one side changes is that side's file byte for byte (12 from the switches, 175 from the upstream
  merge);
- in each of the two shared files the merge differs from each side by exactly the other side's patch (the same
  `git patch-id --stable`).

The fork's compute-6.1 sparse attention kernel refuses a mask with a head dimension, as before, so upstream's per-head
masks never reach it. Every switch keeps its default (off).

## The build

```bash
WORK=<build folder> TESTS=1 JOBS=8 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" \
  DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12360" \
  scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-cand-b12410
```

`HF_UI_VERSION=b12360` pins the web page llama-server serves to upstream b12360's prebuilt archive, for the reason in
[the switch merge's README](../2026-11-17_v5_merge_switches/README.md#the-build). Upstream's web page source
(`tools/ui`) is the same at b12410 apart from its `.npmrc`. A trial build of the same commit from a local branch,
made before the push, gave the same files in `bin/` byte for byte.

| File in `bin/` | md5 |
|---|---|
| `llama-server` | `5b134f43f3a3e885bfba02538ffebcbe` |
| `test-backend-ops` | `1a09dae81d6ff2f3cc52d0ab91fe9eca` |
| `test-llama-archs` | `22fd68c5b53fd64ac2303fe950d154fb` |
| `test-cuda-keepalive` | `38ab0653c330e3949ff42af5497e4c1a` |
| `llama-bench` | `1544656d5c96d51949a5be62be15b19d` |
| `llama-completion` | `a3d479d72a7394faea02f1a74a3c6656` |

## The machine code with every switch unset

`run/sass_check.sh` disassembles C854's `llama-server` (`b3dfb0d89`) and the candidate's with `cuobjdump -sass` (in
the CUDA image, on a CPU) and hashes each kernel's SASS with C861's `sass_compare.py`
([its run folder](../2026-11-17_c861a_fp32_scale/run/)). Full output: [sass_identity.txt](sass_identity.txt).

- **8,135 of C854's 8,835 kernels are byte for byte the same.** None of the 5,117 kernels over 200 instructions
  differs, no kernel is missing and none is new with the switches off.
- **The other 700 are the stubs that only report an error:** `mul_mat_q`'s `NO_DEVICE_CODE` branch for the template
  combinations with no compute 6.1 configuration, at most 30 instructions each. They print the source line 1102
  instead of 1083, because C861 added lines above it, and 132 of them also load message strings from other addresses.
  They are the same 700 that the switch merge changed against v4.0: C854's 700 are v4.0's byte for byte, and the
  candidate's 700 are the switch merge's byte for byte.
- **New kernels, only reached with `GGML_CUDA_MMQ_KQ_FP32SCALE` set:** C861's 176 `mul_mat_q` kernels (Q3_K 32,
  Q4_K 48, Q5_K 48, Q6_K 48), each byte for byte the switch merge's, with the instruction counts in
  [kq_instructions.tsv](../2026-11-17_c861a_fp32_scale/kq_instructions.tsv).
- C865 and C866 change host code only, as their own builds showed.

## Files

| File | What |
|---|---|
| [sass_identity.txt](sass_identity.txt) | the per-kernel comparison with C854's build, the 700 stubs' line-by-line classification, the comparison with the switch merge's build |
| [run/sass_check.sh](run/sass_check.sh) | the comparison: `OUT=<folder> MNT=<folder holding the binaries> BASE=<C854's llama-server> NEW=<this llama-server> MERGE=<the switch merge's sass folder> bash run/sass_check.sh`, with `sass_compare.py` copied into `OUT` |

Everything ran on a build machine's CPU in Docker (`--cpu-shares=256`, `-j 8`, one build at a time); nothing ran on a
GPU. Licence: the scripts are this repository's.
