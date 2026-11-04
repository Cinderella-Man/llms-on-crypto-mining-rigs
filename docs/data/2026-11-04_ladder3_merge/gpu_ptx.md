# C228 q1 (Q781): `-ladder3`'s switches on a GPU through the compute 6.1 PTX (2026-11-04)

Every switch merged into fork branch `pascal-p106-v3-b12076-ladder3` (`7bc1552f4`), run on and off on two RTX 3090s
through the rig build's compute 6.1 PTX: the kernels a P106-090 runs, with the fork's paths forced on (`=2`) and the
P106's launch shapes (`NSM=5`). Correctness only: `test-backend-ops` against the CPU backend, the model-level parts
comparing arms of one binary. The registry row is Q781 in [quants/experiments.md](../../../quants/experiments.md); the
P106 runs (`MODE=native`, one half per rig) are its second part.

**Result:** every switch passes. One failure, by name: `-gdnfuse` on llama.cpp's Meta backend, the device that splits
each tensor across GPUs (Q877); the cards' own rows pass, and the rig's layer split does not use that backend. One
check could not run: `test-llama-archs` on two cards stops at the first Meta row, because llama.cpp's all-reduce kernel
has no compute 6.1 code (Q879), so the staged hop's verdict rests on `hops/model` (bit for bit across a real two-card
split). Four gaps in the scripts themselves were found and fixed on the way (Q875, Q876, Q878).

## Verdict per switch

| Part | Branch (task) | Runs and result | Verdict |
|---|---|---|---|
| `ladder2` | `-bf16mv` (C54): `GGML_CUDA_MMVF_PASCAL`, `_SHAPE` | `bf16` 309/309, `bf16-g8r8` 303/303, `bf16-g128r8` 303/303, `bf16-off` 309/309 | pass |
| `ladder2` | `-q2_0mv` (C3), with `-kcol` and `-mtok` on | `q2_0`, `q2_0-stack`, `q2_0-off` 497/497 each (`q2_0-off` passes the large-activation cases too) | pass |
| `ladder2` | `HC_STREAM_MEAN` | 6/6 | pass |
| `ladder2` | `-iqmmvq` (C79): `GGML_CUDA_MMVQ_PASCAL_SKIP` | `iq` 602/602, `iq-off` 586/586 (its 16 repeated-id cases of 5–8 tokens left out, Q875) | pass |
| `ladder2` | `-kcol` (C70): `_KCOL=1/2`, `_KCOL_R`, `_KCOL_ITT` | `kcol`, `-all`, `-r1`, `-r2`, `-itt1`, `-off` 855/855 each | pass |
| `ladder2` | `-mtok` (R7): `GGML_CUDA_MMVQ_PASCAL_MTOK` | default, `cap1`, `cap3`, `rb1`, `rb8`, `gdir`, `off` 333/333 each | pass |
| `ladder2` | `-gdnfuse` (C18–C20): `GGML_CUDA_GDN_FUSE`, `LLAMA_GDN_FUSE` | ops `gdn`, `gdn-off` 499/499; qwen4exp fused against unfused at `n_rs_seq` 0 and 3: every row OK | pass on a card; **fails on the Meta backend** (Q877) |
| `ladder2` | persistent MoE kernel: `_CHECK`, `_FAULT` | `moe-check` 0 of 1 layers differed; `moe-fault` 1 differed (as the negative control must) | pass |
| `ladder2` | every architecture (`test-llama-archs`) | `archs-off` 411/411; `archs` and `archs-mask` stop at qwen4exp's Meta row (Q877); with the two GDN fusions off and every other forced switch on, 411/411, and with the card-built mask checked | pass, but for Q877 |
| `ladder2` | `-ckpt` (C22): device stores | `test-llama-context` on qwen4exp, qwen35, mamba: failed 0 | pass |
| `ladder2` | the whole suite, fork paths on / off | `all-on` 22,337/22,337; `all-off` 22,087/22,087 | pass |
| `ladder2` | the benches' own checks | `mmvq_kcol --check-only` and `mmvq_pascal --q2_0-mtok --check-only`: every variant matches (the multi-token kernel bit for bit per token) | pass |
| `kcol` | `-kcol` (C93, C84, C85) | `rig_kcol.sh`: six arms 855/855, BF16 309/303/303, `mmvq_kcol` matches; `rig_kcol_cov.sh`: `cov`, `_all`, `_r1`, `_r2`, `_itt1` 366/366, `q2f`, `q2f_r2` 157/157, `--q2_0` matches; `cov_off` 364/366: Q4_0's and Q5_0's 8-column cases with block sums past FP16, which with the k-column kernels off go to llama.cpp's MMVQ and overflow its FP16 sums (Q878) | pass |
| `iq2xs` | `-iq2xs` (C2): `GGML_CUDA_MMVQ_PASCAL_SKIP` | `on` and `skip`: five types × four ops, every case; memcheck 15 of 15 cases with 0 errors | pass |
| `iqkcol` | `-iqkcol` (C163): `_KCOL_R`, `_MTOK` caps and row blocks, `_SKIP` | `on` 1,813/1,813, `kcol_r2` 861/861, `cap1` 952/952, `rb1` 952/952, `off` and `skip` 1,765/1,765 (48 repeated-id cases left out, Q875); memcheck 30 of 30 with 0 errors; `mmvq_kcol --check-only` on the IQ types matches | pass |
| `qsa` | `-qsa` (C6): `GGML_CUDA_FATTN_SPARSE_PASCAL` 0/1/2, `_FAULT` | `sparse1`, `sparse2`, `sparse0` 48/48; faults 2 and 3 fail 29 of 48 cases; fault 1 (output × 1.01) 48/48, inside the limit (Q878); `fasp_check --gpu` 20 of 20 | pass |
| `fap` | `-fap` (C135): `GGML_CUDA_FATTN_PASCAL`, `_FAULT` | on 4,255/4,255; fault 1 22/22 (inside the limit); faults 2 and 3 fail 6 and 18 of 22; off 22/22 | pass |
| `hops` | `-hops` (C200, C201; C200g) | `CPY` 276/276 (the 18 hop shapes among them); `test-hop-cast --unit-only` exit 0; across two cards, 11 arms: every staged arm bit for bit the driver copy's logits, the F16, BF16 and Q8_0 casts within bound (8 contexts cast before layer 1, 14 staged rings); `test-llama-archs` on two cards stops at the first Meta row (Q879) | pass |
| `vgraphs` | `-vgraphs` (C151): `LLAMA_GRAPH_PER_SIZE`, `LLAMA_SCHED_PER_SIZE`, `LLAMA_GRAPH_META_CALLOC` | `spec_check.sh` PASS (every arm the base's tokens and top probabilities, 4 verify widths); `test-llama-context` at four settings × two models: failed 0 | pass |
| `ckpt` | `-ckptrc` (C154) with C22 | `server_check.sh` PASS without and with `--ctx-checkpoints-share` | pass |
| `server` | `-specdump` (Q688), `-adraft` (Q687), `-lightprof` (C142), C154's share, `-taper` (C218), `LLAMA_MOE_TAIL` | 12 llama-server arms: every arm's tokens equal the switch-off arm's; dump 48 lines, adapt logs 49 and 187, profiles written in all four modes, 6 taper lines; 0 errors (the adapt arms' rule cut every draft: reported) | pass |

`-ccflags` (C131) and `-bytesw` (Q618, Q619) have no runtime switch here: they ran in every part at their defaults.

## Findings

- **Q877, `-gdnfuse` on the Meta backend:** qwen4exp's Meta row stops at
  `GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` (`ggml-backend-meta.cpp`) with `LLAMA_GDN_FUSE` on, since
  the fused GDN op has no split rule there, and hits an illegal memory access with `GGML_CUDA_GDN_FUSE` on. Both are on
  by default. [run/probe_archs_meta.sh](run/probe_archs_meta.sh) turns each switch off and on alone: only these two
  matter.
- **Q879, the Meta backend across cards:** llama.cpp's all-reduce kernel (`ggml_cuda_ar_kernel`, `allreduce.cu`) has
  no compute 6.1 device code, so on two cards a compute 6.1 build stops at the first architecture's Meta row (the dense
  `llama` model). `test-llama-archs` adds the Meta row before the two-card layer-split rows ("Host experts", "MoE
  cache, layer split") and has no switch to leave it out, so those rows never run on a Pascal build with two cards.
- **Q875, repeated expert ids:** the fork's `test_mul_mat_id_dup` cases (a token naming the same expert more than once)
  at 5–8 tokens, with the Pascal kernels skipped, go to llama.cpp's MMQ, whose `quantize_mmq_q8_1` then reads past its
  input (compute-sanitizer: 2,817 errors, 16-byte reads up to ~152 KB past a 7.6 MB buffer; every IQ2_S case wrong;
  a plain run ends in an illegal-address fault). The fork's kernels pass the same cases with 0 memcheck errors.
  llama.cpp keeps its own repeated-id cases disabled for CUDA (`TAG_MUL_MAT_ID_DUPLICATE_ID`) and routers never repeat
  an expert, so no model reaches it; the PTX runs' stock arms leave these cases out
  ([run/probe_dup_ids.sh](run/probe_dup_ids.sh) isolates them). Whether a P106 faults the same way natively is open.
- **Q876, no forced JIT:** `CUDA_FORCE_PTX_JIT=1` also forces cuBLAS onto PTX, which cuBLAS does not ship for these
  cards, and every case cuBLAS takes aborted its run (`cublasCreate`: "the resource allocation failed"). The rig build
  holds only sm_61 code and compute_61 PTX, so a newer card JIT-compiles that PTX without the flag; the scripts take
  `PTX_JIT` (default 0).
- **Q878, verdict gaps:** `qsa_gpu.sh` counted a run with no cases as passing (it reads `DEV` as a backend name, and an
  exported card number matched no backend); it now fails such a run and is passed `DEV=CUDA0`. Its fault 1 (output ×
  1.01, an NMSE of ~1e-4) is inside `test-backend-ops`' 5e-4 for FLASH_ATTN_EXT and is now expected to pass, as FAP's
  fault 1 is. `cov_off`'s expectation names its two FP16 large-activation cases.
- The GPUs: eight application-level Xid lines in the kernel log, all from the test processes above: 13 and 31 from the
  repeated-id case and from `GGML_CUDA_GDN_FUSE` on the Meta row, 43 as channels closed when processes exited (four of
  them after the two aborted two-card runs). No Xid 79 or 154: no card left the bus.

## How it ran

The rig tarball `llama.cpp-20261104-7bc1552f4-pascal-p106-v3-b12076-ladder3` (md5 `58fe0038aa8bfd4294436a63a1b0b5c3`,
sm_61 code and compute_61 PTX only) with the extra tests copied into `bin/` (md5 `cede98594e6480ff5a708607b4d1d58f`),
[run/gpu_checks.sh](run/gpu_checks.sh) `MODE=ptx` inside `nvidia/cuda:12.9.2-devel-ubuntu24.04`, the random models of
`test-llama-archs -o` as `MODELS`, and the bench binaries the README lists. The parts ran in several containers side by
side, split by `PARTS` (and `-ladder2`'s `RUNS`/`SKIP`), on the same two cards; the CPU references made the run
CPU-bound (~4 h of wall time). A container whose CPU share was changed while it ran (`docker update` under cgroup v2
drops the devices NVIDIA's runtime hook added) found no CUDA device for every process it started afterwards; those
runs are left out and ran again in new containers, and a process that already held its card kept it.
[gpu_ptx_console.txt](gpu_ptx_console.txt) holds every valid run line, grouped by the container that ran it, and the
two probes' lines.

```bash
# one container per group of parts; the cards are the container's 0 and 1; DEV/DEVS2 pick them inside
docker run --rm --gpus '"device=<a>,<b>"' -v "$PWD":/r:ro -v "$WORK":"$WORK" -v "$BENCHES":"$BENCHES":ro \
  -e MODE=ptx -e BIN=$WORK/<tree>/bin -e OUT=$WORK/gpu-ptx-<n> -e MODELS=$WORK/checks/ctx/models \
  -e PARTS='iqkcol hops vgraphs ckpt server' -e DEV=1 -e DEVS2=0,1 \
  -e KCOL_BENCH=... -e MTOK_BENCH=... -e KCOL_BENCH2=... -e MMVQ_PASCAL=... -e FASP=... nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq curl jq > /dev/null; bash /r/docs/data/2026-11-04_ladder3_merge/run/gpu_checks.sh'
# the probes, one card, in the same image
BIN=$WORK/<tree>/bin OUT=$WORK/probe-dup bash run/probe_dup_ids.sh
BIN=$WORK/<tree>/bin OUT=$WORK/probe-archs bash run/probe_archs_meta.sh             # steps forced and rest
STEPS=nogdn BIN=$WORK/<tree>/bin OUT=$WORK/probe-archs/nogdn bash run/probe_archs_meta.sh
```
