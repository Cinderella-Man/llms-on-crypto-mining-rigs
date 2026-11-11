# FN68 (Q1280): C620's KV-only catch-up check in a container with one newer card (2026-11-11)

C620 ([design and CPU check](../2026-11-11_c620_draft_kv_catchup/README.md), Q1105) adds the switch
`LLAMA_MTP_KV_ONLY=1` to Miner Llama's branch `pascal-p106-v4-b12246-c620` (`ca3055285`): the MTP draft's catch-up
passes write only the draft layer's caches. Its check, `test-llama-archs --mtp-kv-only`, passed on a build machine's
CPU. FN68 ran the same check, from the rig build's own tree, in a container that holds one RTX 3080. The rig build
carries sm_61 machine code and compute_61 PTX only, so any kernel it runs on that card is the PTX a P106 would run,
JIT-compiled by the card's driver. The aim was the cut pass's identity on CUDA before the rig run R76 (Q1278).
Correctness only, no speed. The registry row is Q1280 in [quants/experiments.md](../../../quants/experiments.md).

**Result:** the three seeds pass, 15 of 15 runs (five configurations at seeds 1, 2 and 3): the draft's state after the
prompt, every draft and the state at the end are the same bytes with the switch off and on, with the same node counts
as on the CPU (1,458 / 531 for `f16 fa`). **The check ran on the CPU, not on the card.** `--mtp-kv-only` builds its
model with an empty device list (the test's help marks the check "CPU only"), so neither the model nor the MTP contexts
use the CUDA device, even when one is visible. Three readings show it:

- the card's used memory, read every 0.5 s, was at most 4 MiB in each run (1 MiB before the runs); with a model on the
  card the control below reached 293 MiB;
- the driver's JIT cache stayed empty after every run; after the control it held 24 files, 44,564,702 bytes;
- at `-v 4` every buffer of the run is on the CPU (1 model buffer of 31.38 MiB, 20 KV, 20 compute, 10 output and
  4 rotation buffers), and each of the 20 scheduler graphs has one split; no log line names a CUDA device besides the
  backend's own device list.

So this run says nothing about the cut pass on CUDA. The first check of it on CUDA is R76 (Q1278), step (a), on a
P106: the drafts of the control and the cut arm at 0 and 32,768 tokens. A device option for the MTP checks of
`test-llama-archs` (Q1370) would put the check's model and contexts on a named card; with it the check runs again
through the PTX on a build machine's card (FN76, under Q1280).

**The control:** the same binary, in the same container, with `-a qwen4exp -b CUDA0 -s 1` loads the same
architecture's fixture (its main model, without the MTP block) on the card and compares it with the CPU. It passes
(NMSE against the CPU 1.19·10⁻⁷, round trip OK, mixed batch 4.34·10⁻⁸): the card runs the rig build's PTX when a test
puts a model on it.

## Results

| Run | Arguments | Exit | Summary | Card memory, largest sample | JIT cache after it | Where it ran |
|---|---|---:|---|---:|---|---|
| `kvonly-s1` | `--mtp-kv-only -s 1` | 0 | all 5 passed | 4 MiB | empty | CPU |
| `kvonly-s2` | `--mtp-kv-only -s 2` | 0 | all 5 passed | 4 MiB | empty | CPU |
| `kvonly-s3` | `--mtp-kv-only -s 3` | 0 | all 5 passed | 4 MiB | empty | CPU |
| `kvonly-s1-v4` | `--mtp-kv-only -s 1 -v 4` | 0 | all 5 passed | 4 MiB | empty | CPU (every buffer; 1 split a graph) |
| control | `-a qwen4exp -b CUDA0 -s 1` | 0 | all 1 passed | 293 MiB | 24 files, 44,564,702 B | the card |

The `--mtp-kv-only` table, the same at each seed and the same as C620's CPU check:

| Config | KV after the prompt | Drafts | KV at the end | Catch-up nodes, off / on |
|---|---|---|---|---|
| `f16 fa` | identical | identical | identical | 1,458 / 531 |
| `f16` | identical | identical | identical | 1,503 / 549 |
| `q8_0 fa` | identical | identical | identical | 1,566 / 585 |
| `f16 fa top-k` | identical | identical | identical | 1,647 / 531 |
| `f16 fa dist` | identical | identical | identical | 2,043 / 2,043 |

The binary: `test-llama-archs` of the rig tree `llama.cpp-20261110-ca3055285-pascal-p106-v4-b12246-c620` (md5
`27cbad9f678e26435a8e0f9619876755`), 187 sm_61 cubins and 187 compute_61 PTX listings, nothing else
([archs.txt](archs.txt)). The card: RTX 3080 (compute capability 8.6), driver 595.99.02, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`. The fork's PTX-mode settings were set for every run (as in
[FN63](../2026-11-11_fn63_sparse_256_512/README.md)): `CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2
GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304
GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`. The run took 56 s from the
preflight to the container's end (16:52:45–16:53:41).

## How to run it

On a build machine with Docker and one free card, from a checkout of this repository, with the branch's rig tarball
unpacked (its `bin/` holds `test-llama-archs`, built into it by C620's `run/checks.sh`):

```bash
BIN=<rig tree>/bin OUT=<work dir>/out GPU="--device nvidia.com/gpu=<card>" MOUNTS=<the folder BIN and OUT lie on> \
DOCKER_ARGS="--cpu-shares=256" PREFLIGHT=<path of cuda_preflight.sh> \
setsid -f nohup bash docs/data/2026-11-11_fn68_c620_ptx/run/ptx_check.sh > <work dir>/driver.out 2>&1 < /dev/null
```

`OUT/done` marks the end; `OUT/steps.txt` has one line per step ([steps.txt](steps.txt) here) and `OUT/kvonly.tsv`
the table. Once the check takes a device (Q1370), its rerun (FN76) adds the device to the `--mtp-kv-only` runs.

## Files

| File | What |
|---|---|
| [runs.tsv](runs.tsv) | every run: arguments, exit code, summary, the card's memory samples and largest value, the JIT cache after it, where the work ran |
| [kvonly.tsv](kvonly.tsv) | `--mtp-kv-only`: one row per seed and configuration |
| [archs.txt](archs.txt) | the GPU code in the binary (`cuobjdump`) |
| [steps.txt](steps.txt) | the driver's steps, paths replaced by their variables |
| [run/ptx_check.sh](run/ptx_check.sh) | the run: preflight, the container, the runs, the readings, the control, the table |

## Provenance and licence

Every number here comes from random models that `test-llama-archs` builds from its seeds; no model file, prompt or
recorded text was read. The fork is MIT-licensed, as llama.cpp; these tables and the script are published with this
repository.
