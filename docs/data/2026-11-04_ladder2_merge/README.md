# C86 / C79 (Q551, Q591, Q603): one tree for the night's Flash-Next engine branches (2026-11-04)

On 2026-11-03/04 six engine branches for Qwen3.8-Flash-Next on the nine-P106 rig were written side by side, each on
the speed ladder's engine `pascal-p106-v3-b12076-ladder` (or, for the grid types, on its base). Each one was built and
checked on its own; none had run on a GPU yet. This folder holds their merge into one fork branch,
`pascal-p106-v3-b12076-ladder2`, the checks that ran on the merged tree without a GPU, and the script that runs every
branch's GPU correctness on the merged build, each switch on and off. The registry rows are Q551 (the speed ladder),
Q591 (the k-token kernels) and Q603 (the IQ2_S/IQ3_XXS and Q2_0 kernels in one tree) in
[quants/experiments.md](../../../quants/experiments.md).

**Status:** merged, built for the rig and checked on the CPU: every branch's own CPU check passes on the merged tree,
and every kernel a branch changed or added comes out of the merge with that branch's sm_61 machine code. Not yet run
on a GPU; [run/gpu_checks.sh](run/gpu_checks.sh) does that next, one named run per switch.

## What is merged

Fork branch `pascal-p106-v3-b12076-ladder2` at `a330672c1`: six merge commits on `pascal-p106-v3-b12076-ladder`
(`98bbd6a2f`), each a branch tip as it stood at the merge, so later commits on a branch merge on top:

| Order | Branch, tip | What it adds | Its folder | Switches (defaults unchanged) |
|---|---|---|---|---|
| 1 | `-iqmmvq` `76788bf4a` (cut from `pascal-p106-v3-b12076` `b05ecc2d1`) | IQ2_S and IQ3_XXS in the Pascal matrix-vector kernels (Ornith's and Swift IQ3_XXS's grid types) | [upstream rebase](../2026-11-03_upstream_rebase/README.md) | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs` |
| 2 | `-kcol` `e1fa4c573` | k-column products of 1–8 tokens for the dense types (Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS), the BF16 kernel to 8 tokens | [kcol_dense](../2026-11-04_kcol_dense/README.md) | `GGML_CUDA_MMVQ_PASCAL_KCOL=0/1/2`, `_KCOL_SKIP`, `_R`, `_ITT`; `GGML_CUDA_MMVF_PASCAL=0`, `_SHAPE` |
| 3 | `-mtok` `c70687382` | Q2_0 expert products of 2–8 tokens that read each distinct expert once | [mtok_experts](../2026-11-04_mtok_experts/README.md) | `GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G` |
| 4 | `-gdnfuse` `766c92a22` | the recurrent layers' fusions: state read in place, `GGML_OP_GDN_CONV`, gated output norm | [gdn_fusions](../2026-11-04_gdn_fusions/README.md) | `LLAMA_GDN_FUSE=0`, `GGML_CUDA_GDN_FUSE=0` |
| 5 | `-ckpt` `b5ed2f4fe` | context checkpoints in device stores, checkpoints inside long prompts | [ckpt_cards](../2026-11-04_ckpt_cards/README.md) | `--ctx-checkpoints-vram N`, `--ctx-checkpoints-vram-sync`, `--ctx-checkpoints-spread N` (all off) |
| 6 | `-draftmem` `1e0caf230` | the MTP draft reads the main model's embeddings and head, a draft-head shortlist, settable load staging | [draft_mem](../2026-11-04_draft_mem/README.md) | `LLAMA_DRAFT_SHARE`, `LLAMA_DRAFT_SHORTLIST`, `LLAMA_DIO_STAGING` (all off) |

**The merge is the union of the branches' changes.** Of the 63 files the six branches touch, 56 are touched by one
branch: 54 of them are byte for byte that branch's file, and the two Pascal kernel files `-iqmmvq` changes differ from
`-ladder` by exactly `-iqmmvq`'s own change set on its base plus the union lines below. In the seven files two or more
branches touch (`qwen4exp.cpp`, `llama-context.{h,cpp}`, `llama.h`, `ggml-cuda.cu`, `mmvq.cu`,
`test-backend-ops.cpp`), the merged file's changed lines against `-ladder` are exactly the union of each branch's
changed lines (compared as sorted line sets).

### The one hand merge: Q2_0's tile staging and the grid types' tables (C79, Q603)

`-q2_0mv` (in `-ladder`) and `-iqmmvq` both change the Pascal matrix-vector kernels' shared memory and launch paths:
Q2_0 added a `tile` template parameter (a warp stages its whole rows, for 180-byte rows whose segments are not 16-byte
aligned) and sizes the segment buffer by `seg = staged && !tile`; the grid types added a shared-memory table
(`tab`, filled by `mmvqp_tabs_ready`), per-type launch bounds (`mmvqp_min_blocks<type>()`, three blocks per SM for
IQ2_S/IQ3_XXS, one for every other type as before) and a persistent row-block loop (`MMVQP_IQ_PERSIST`, off by
default). Eight conflict hunks, each resolved as the union:

| File, place | Kept from `-q2_0mv` | Kept from `-iqmmvq` |
|---|---|---|
| `mmvq-pascal-impl.cuh`, traits | `Q2_0` | `IQ2_S`, `IQ3_XXS`, the tuning macros, tables and their fill |
| `mmvq-pascal-impl.cuh`, `mmvqp_choose` | Q2_0's shapes (tile staging for down) | the grid types' shape |
| `mmvq-pascal.cu`, one-token rows | the segment buffer sized by `seg` | the table and its fill on a block's first row block |
| `mmvq-pascal.cu`, one-token kernel | the `tile` parameter, passed through | the launch bounds, the persistent loop |
| `mmvq-pascal.cu`, multi-token kernel | `tile`, the segment buffer by `seg` | the launch bounds, the table word count |
| `mmvq-pascal.cu`, launch by shape | `mmvqp_launch_tile`, Q2_0's tile cases first | the half-word types (Q6_K, IQ2_S, IQ3_XXS) together |
| `mmvq-pascal.cu`, host dispatch | `smem` (the tile's dynamic shared memory) | the type list, `mmvqp_halfword_type` |
| `test-backend-ops.cpp` | the BF16 and Q2_0 cases | the grid types' eval and perf cases |

Neither feature's logic changed: a grid type never takes tile staging (Q2_0 only), Q2_0 has no table (its unused
one-entry declaration is the one every non-grid type carries on `-iqmmvq`), and the QKV launch takes neither type.

### Who takes which product (precedence)

Three Pascal paths can see a quantized matrix-vector product; their conditions do not overlap, so the merge changes no
dispatch. In `ggml_cuda_mul_mat_vec_q`, in this order:

| Order | Path | Takes | Then falls through to |
|---|---|---|---|
| 1 | `ggml_cuda_mmvq_pascal_kcol` (`-kcol`) | plain `MUL_MAT` (no ids, no fusion) of 1–8 columns: Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS; at `KCOL=1` 2–8 columns plus Q3_K and short Q8_0 rows at one | 2, 3 |
| 2 | `ggml_cuda_mmvq_pascal_mtok` (`-mtok`) | Q2_0 `MUL_MAT_ID` (ids) of 2–8 tokens, fused gate/up included | 3 |
| 3 | `ggml_cuda_mmvq_pascal` (C3, `-iqmmvq`) | the rest of its types: one token, fused gate or bias, `MUL_MAT_ID` of other types at 2–4 tokens, IQ2_S and IQ3_XXS at 1–4 | llama.cpp's `mul_mat_vec_q` |

`-kcol` needs no ids, `-mtok` needs ids, so no product meets both. BF16 goes to `mmvf-pascal.cu` (1–8 tokens with
`-kcol`); the persistent MoE kernel is Flint-shaped and one-token only, unchanged. `GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0`
sends Q2_0 past paths 2 and 3; path 1 has its own `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP`.

## Checks on the merged tree (CPU only, no GPU)

The rig build (`TESTS=1`, sm_61 + compute_61 PTX, CUDA 12.9 in Docker): 346,367,872 bytes, no AVX instructions,
highest glibc symbol GLIBC_2.38; `test-llama-archs`, `test-llama-context` and `test-draft-mem` from the same tree in a
separate tarball. Each branch's own CPU check, re-run on the merged tree in a container without a GPU:

| Check | Branch | Result |
|---|---|---|
| decoding of the k-column kernels against ggml's reference dequantizers (`bench/mmvq_pascal/verify/kcol_unit_check.sh 40`; the bench's copy of the kernels' header is the merged tree's, byte for byte) | `-kcol` | 7/7 types pass, 240 cases each, worst error 0.0044 of the tolerance (as on the branch) |
| fused graph against unfused on a random qwen4exp, CPU logits after one batch and after steps of 1–8 tokens, without and with rollback snapshots | `-gdnfuse` | 0 of 32,768 logits differ in both settings |
| `test-llama-context` on random qwen4exp, qwen35 and mamba models, every case ([context_cases.tsv](context_cases.tsv)) | `-ckpt` | 54 OK, 6 SKIP, 0 FAIL; `store` and `store_p` OK on all three |
| llama-server's checkpoint arms on the random qwen4exp ([server_verdict.txt](server_verdict.txt)) | `-ckpt` | PASS: every device-store arm gives the host arm's tokens and top probabilities over five requests; the spread arm resumes P3 from the checkpoint at 2,048 tokens |
| `test-draft-mem` | `-draftmem` | 82 checks, 0 failed; the shortlist head's logits equal the full head's (max diff 0) |
| `test-llama-archs -b CPU`, every architecture | all | 148 of 148 passed |
| per-kernel SASS identity (below) | `-iqmmvq`, `-kcol`, `-mtok`, `-gdnfuse` | every kernel a branch changed or added carried unchanged |

### The machine code of every kernel, against the source branches' builds

R7's and C79's kernels cannot run without a GPU, but their code can be compared: `run/sass_identity.sh` takes the
sm_61 SASS of every kernel in the merged `test-backend-ops` (`cuobjdump`) and of the same binary from each source
branch's build, and compares kernel by kernel ([sass_summary.txt](sass_summary.txt),
[sass_kernels.tsv.gz](sass_kernels.tsv.gz), [sass_carried.tsv.gz](sass_carried.tsv.gz)):

| Kernels of the merged build (7,263) | Count |
|---|---|
| identical to `-ladder`'s | 6,720 |
| identical to `-iqmmvq`'s (the grid types' 36, and the Pascal kernels of the other types, whose parameter struct `-iqmmvq` extended) | 229 |
| identical to `-kcol`'s (k-column kernels, BF16 kernel to 8 tokens) | 227 |
| identical to `-gdnfuse`'s (recurrent-layer fusions) | 27 |
| identical to `-mtok`'s (multi-token expert kernels) | 12 |
| Q2_0's Pascal kernels: `-ladder`'s code with the kernel parameters 4 bytes further on (`-iqmmvq` added `nrowblocks` to the parameter struct) | 38 |
| Q2_0's multi-token kernels: the same opcodes in the same numbers, in another order | 4 |
| Q2_0's multi-token kernels: 2–10 integer address instructions (`IADD`, `MOV`, `XMAD`, `SHL`) differ, every `IDP.4A`, FP32, load, store and shuffle instruction the same in kind and number ([sass_q2_0_opdiff.tsv](sass_q2_0_opdiff.tsv)) | 6 |

Every kernel `-iqmmvq` (229), `-kcol` (227), `-mtok` (12) and `-gdnfuse` (27) changed or added against the tree it was
cut from is in the merged build with the same machine code; `-ckpt` and `-draftmem` change no kernel. The 48 Q2_0
kernels are the only ones that match no source build exactly, as expected where `-q2_0mv`'s kernel meets `-iqmmvq`'s
parameter struct; the GPU runs `q2_0` and `q2_0-stack` below test them.

### Q672: the sm_61 FP16-conversion trap

On compute 6.1 FP16 arithmetic runs at a small fraction of the FP32 rate, so an FP16-to-FP32 conversion that ptxas
emits as half arithmetic (`HADD2.F32 Rd, -RZ, Rs.H0_H0`) instead of the conversion instruction `F2F.F32.F16` can cost
many times more. [run/hadd2_audit.py](run/hadd2_audit.py) counts both per kernel in the merged build's SASS
([q672_families.tsv](q672_families.tsv), [q672_kernels.tsv.gz](q672_kernels.tsv.gz)):

- **No `HADD2` conversion anywhere** in the 7,682 function bodies; the binary holds no `HADD2` at all. Every FP16
  scale, block sum and router weight converts with `F2F.F32.F16` (5,188 in the Pascal quantized kernels, 271 in the
  k-column kernels, 204 in the multi-token kernels, 1,360 in the persistent MoE kernel, 3,264 in the BF16/F16
  matrix-vector kernels); BF16 needs no conversion instruction, its bits being FP32's upper half.
- **FP16 arithmetic** (`HFMA2`, `HMUL2`) is in 102 kernels ([q672_half_sites.tsv](q672_half_sites.tsv)): the
  F16-accumulating `mul_mat_vec_f<half, half>` (88 kernels, 1,480 instructions) and the persistent MoE kernel's F16
  router accumulation (12 kernels, 384), both chosen only where `fast_fp16_available(cc)`, which is false for a
  compute 6.1 build, so the P106 never runs them; and `dequantize_block_q8_0_f16` (2 kernels, 64 `HMUL2`), the Q8_0
  to F16 conversion for a cuBLAS product, which a P106 does run when a Q8_0 matrix goes to cuBLAS (prompt batches
  that MMQ does not take).

## Next: the GPU checks

[run/gpu_checks.sh](run/gpu_checks.sh) runs every branch's GPU correctness on the merged build, one named run per
switch setting, so that a failing switch is found by name. `MODE=ptx` runs it on a newer NVIDIA GPU through the
build's compute 6.1 PTX with every fork path forced on and the P106's launch shapes; `MODE=native` runs it on one
P106-090. Each run's line in `console.txt` gives the switch it tests, its summary, its FAIL lines and what passing
means:

| Runs | Switch | Ops or tool |
|---|---|---|
| `bf16`, `bf16-g8r8`, `bf16-g128r8`, `bf16-off` | `GGML_CUDA_MMVF_PASCAL`, `_SHAPE` | `MUL_MAT`, `MUL_MAT_VEC_FUSION` (BF16) |
| `q2_0` (k-token paths off), `q2_0-stack` (on), `q2_0-off` | `GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0` | `MUL_MAT`, `MUL_MAT_ID`, `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_SHARED` (Q2_0) |
| `hcsm` | — | `HC_STREAM_MEAN` |
| `iq`, `iq-off` | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs` | the same four ops (IQ2_S, IQ3_XXS) |
| `kcol`, `kcol-all`, `kcol-r1`, `kcol-r2`, `kcol-itt1`, `kcol-off` | `GGML_CUDA_MMVQ_PASCAL_KCOL`, `_R`, `_ITT` | `MUL_MAT` of the seven types at 1–8 columns |
| `mtok`, `mtok-cap1`, `mtok-cap3`, `mtok-rb1`, `mtok-rb8`, `mtok-gdir`, `mtok-off` | `GGML_CUDA_MMVQ_PASCAL_MTOK` | `MUL_MAT_ID`, `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_SHARED` (Q2_0) |
| `gdn`, `gdn-off`; `gdn-archs-rs{0,3}{,-off}` | `GGML_CUDA_GDN_FUSE`, `LLAMA_GDN_FUSE` | the fused and replaced ops; `test-llama-archs` qwen4exp against the CPU |
| `moe-check`, `moe-fault` | `GGML_CUDA_PERSISTENT_MOE_CHECK=1` (fault: its negative control) | `MOE_LAYER` |
| `archs`, `archs-off`, `archs-mask` | defaults; every fork path off; `LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1` | `test-llama-archs`, every architecture against the CPU |
| `ctx-*`, `ctx2-*` | the device stores (one card, two cards) | `test-llama-context` on random qwen4exp, qwen35, mamba |
| `all-on`, `all-off` | every fork path on / off | the whole `test-backend-ops` suite (`MODE=ptx` only by default) |
| `kcol-bench`, `mtok-bench` | — | the standalone benches' `--check-only` (when given) |

Expected failures: with Q2_0's Pascal kernel off (`q2_0-off`, `all-off`), Q2_0's large-activation cases, whose
blocks sum past FP16's range (llama.cpp's Q2_0 kernel reads the FP16 block sum); `moe-fault` must report differences.

```bash
# on a build machine with a free NVIDIA GPU (G), through the compute 6.1 PTX; WORK as for cpu_checks.sh
docker run --rm --gpus device=$G -v $WORK:$WORK -e MODE=ptx -e BIN=$WORK/<tree>/bin -e OUT=$WORK/gpu-ptx \
  -e MODELS=$WORK/checks/ctx/models nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  bash $REPO/docs/data/2026-11-04_ladder2_merge/run/gpu_checks.sh
# on the rig, one card, nothing else running (the tree unpacked into /opt/llama-builds with the extra tarball's tests)
ssh $RIG "MODE=native BIN=/opt/llama-builds/<tree>/bin OUT=<dir> MODELS=<dir> setsid nohup bash gpu_checks.sh > <dir>.out 2>&1 < /dev/null &"
```

After those, the speed ladder (Q551) runs its next step on this build: the rig's identity check, then the agent
check with the k-token kernels on and off.

## Files

| File | What it holds |
|---|---|
| [context_cases.tsv](context_cases.tsv) | `test-llama-context` per model and case: verdict and reason |
| [server_verdict.txt](server_verdict.txt) | llama-server's checkpoint arms against the host arm |
| [sass_summary.txt](sass_summary.txt) | the SASS comparison's summary: classes, carried kernels per branch, the flagged kernels |
| [sass_kernels.tsv.gz](sass_kernels.tsv.gz) | per kernel of the merged build: the builds with identical code, loose, same-mix and same-ops matches, class, registers, shared, stack and local bytes |
| [sass_carried.tsv.gz](sass_carried.tsv.gz) | per kernel a branch changed or added: carried into the merge unchanged or not |
| [sass_q2_0_opdiff.tsv](sass_q2_0_opdiff.tsv) | the six Q2_0 kernels beyond reordering: the opcode counts that differ from `-ladder`'s |
| [q672_families.tsv](q672_families.tsv), [q672_half_sites.tsv](q672_half_sites.tsv), [q672_kernels.tsv.gz](q672_kernels.tsv.gz) | Q672: FP16 conversions and FP16 arithmetic per kernel family, the kernels with FP16 arithmetic, every kernel |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the rig build of the merged branch, the extra tests from the same tree, and the six CPU checks |
| [run/sass_identity.sh](run/sass_identity.sh), [run/sass_identity.py](run/sass_identity.py) | the per-kernel SASS identity check of a merged build against its branches' builds (CPU, cuobjdump) |
| [run/sass_opdiff.py](run/sass_opdiff.py) | the opcode counts of the kernels the identity check flags, against a reference build |
| [run/hadd2_audit.py](run/hadd2_audit.py) | Q672: FP16 conversions (`HADD2` against `F2F`) and FP16 arithmetic per kernel |
| [run/gpu_checks.sh](run/gpu_checks.sh) | the GPU correctness of every merged switch, on and off (`MODE=ptx` or `native`) |

## Commands

```bash
# the CPU half (build machine, Docker, no GPU): WORK holds the llama.cpp clone with the fork as remote "fork"
WORK=<dir> REPO=<this repository> JOBS=6 LOCK=<shared lock file> \
  SASS_REFS="ladder=<-ladder tree> b12076=<base tree> iqmmvq:b12076=<-iqmmvq tree> kcol=<tree> mtok=<tree> gdnfuse=<tree> ckpt=<tree> draftmem=<tree>" \
  ARCHS_REF=<a base build's test-llama-archs dir> bash docs/data/2026-11-04_ladder2_merge/run/cpu_checks.sh
# the comparison again on the dumps cpu_checks.sh left in $WORK/checks/sass, then the flagged kernels' opcodes
R=docs/data/2026-11-04_ladder2_merge/run
docker run --rm -v $WORK/checks/sass:/s -v $PWD/$R:/r:ro python:3.12-slim-bookworm \
  python3 -I /r/sass_identity.py /s merged ladder iqmmvq:b12076 kcol mtok gdnfuse ckpt draftmem > $WORK/checks/sass/summary.txt
docker run --rm -v $WORK/checks/sass:/s -v $PWD/$R:/r:ro python:3.12-slim-bookworm \
  python3 -I /r/sass_opdiff.py /s ladder merged > $WORK/checks/sass/opdiff.tsv
# Q672 on the merged build's SASS
docker run --rm -v $WORK/checks/sass:/s:ro -v $PWD/$R:/r:ro -v $WORK/q672:/q python:3.12-slim-bookworm \
  python3 -I /r/hadd2_audit.py /s/merged.sass.gz /q/hadd2_kernels.tsv > $WORK/q672/hadd2_families.tsv
```

Provenance: the fork is Miner Llama (github.com/Cinderella-Man/llama.cpp, MIT like llama.cpp); the builds use
`nvidia/cuda:12.9.2-devel-ubuntu24.04` and, for the SASS comparison's Python, `python:3.12-slim-bookworm`. The random
test models are generated by `test-llama-archs`. Everything here is this repository's own work, MIT.
