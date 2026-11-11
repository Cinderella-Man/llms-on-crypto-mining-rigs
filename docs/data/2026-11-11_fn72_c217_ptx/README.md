# FN72 (Q1301): C217's `store_at` cases through the compute 6.1 PTX on one newer card (2026-11-11)

C217 ([design and CPU checks](../2026-11-11_c217_incall_checkpoints/README.md), Q770) adds `--ctx-checkpoints-incall`
to Miner Llama's branch `pascal-p106-v4-b12246-c217` (`68ea07e60`): a prompt's checkpoints are taken inside the decode
call, each state copied into its device store right after the micro-batch that ends at the checkpoint, on the stream of
the card that holds it, with no wait. Its `test-llama-context` cases `store_at` and `store_at_p` passed on a build
machine's CPU, where the copies are synchronous. FN72 ran them, from the rig build's own tree, in a container that holds
one RTX 3080. The rig build carries sm_61 machine code and compute_61 PTX only, so every kernel it runs on that card is
the PTX a P106 would run, JIT-compiled by the card's driver. Correctness only, no speed. The registry row is Q1301 in
[quants/experiments.md](../../../quants/experiments.md); on nine P106s the switch was already a gain with every reply,
checkpoint and restore unchanged ([R85](../2026-11-11_r85_c217_p106/README.md), Q1306).

**Result: both cases pass on the card, on all three random models (qwen4exp, Flash-Next's architecture; qwen35;
mamba), and the models and the state stores were on the card.** A state stored inside one call at a cut, with the copy
queued on the card's stream, is the cut calls' state byte for byte, with the same positions; the call ends in the cut
calls' end state and last logits; the first fill, restored, continues as the reference's restored state does; the cuts
alone end the same; no fill outlives its call. Whole states (`store_at`) and partial states (`store_at_p`, what
llama-server saves) alike, at both memory fills the test uses (0x00 and 0x3e).

**Where the models ran.** Three readings, each run against a run of the same binary with no device:

| Reading | The card (`row`, `both`) | No device (`cpu`) |
|---|---|---|
| card memory, largest sample (1 MiB before) | 399 MiB (`row`), 341 MiB (`both`) | 4 MiB |
| the driver's JIT cache (empty at the start) | 27 files, 47,611,492 B after `row` (30 s; `both`, with the kernels cached, 5 s) | unchanged |
| the per-model logs at `-lv 4` | each model: `offloaded 3/3 layers to GPU`, its weights in a `CUDA0` model buffer (4.17, 5.16 and 18.08 MiB for mamba, qwen35 and qwen4exp; beside it `CPU_Mapped` buffers of 0.12 and 0.25 MiB); all 120 KV, 120 RS and 120 compute buffers on `CUDA0`; all 40 state stores `allocated ... in 'CUDA0'` | every buffer and the 40 state stores on the CPU (its `offloaded 3/3 layers` line counts the layers; there is no device to put them on) |

A state store on `CUDA0` is what sends a fill down the asynchronous path: each row is copied with
`ggml_backend_tensor_copy_async` on the stream of the card that holds it and an event is recorded behind it
(`state_seq_fill_store_at`); the synchronous copy is taken only for rows with no device backend. No run logged a warning
or an error (no `could not be filled inside the call`, no failed store allocation).

**The controls report more on the card than on the CPU.** Each case also decodes, without requiring anything, one call
without the cuts, and with rollback snapshots the cuts without the snapshots' rule (one-token micro-batches again), and
reports whether they end in the cut calls' state ([controls.tsv](controls.tsv)):

| Control | Native CPU build (C217) | The card (`both`; `row` the same) | The rig build on the CPU (`cpu`) |
|---|---|---|---|
| one call without the cuts, no snapshots (`n_rs_seq 0`) | ends elsewhere, 6 of 6 | ends elsewhere, 12 of 12 | ends the same, 12 of 12 |
| one call without the cuts, snapshots for 3 drafted tokens (`n_rs_seq 3`) | ends the same, 4 of 4 | ends elsewhere, 8 of 8 | ends the same, 8 of 8 |
| the cuts without the snapshots' rule (`n_rs_seq 3`) | ends elsewhere, 4 of 4 | ends elsewhere, 8 of 8 | ends the same, 8 of 8 |

(Counts are case runs: models × fills × cases; C217's CPU control ran `store_at` alone, and mamba has no snapshots.) On the card every
control ends elsewhere: the card's kernels round the cut calls' micro-batches (16 + 1 and 13 + 4 tokens) differently
from an uncut call's 16 + 16 in every configuration, where the native CPU build rounded 13 + 4 as 16 + 16. So on the card
the cases see any change of the split, and their byte-identity cannot pass by a coincidence of rounding; the same passes
show that the card's results are repeatable for a given split. The rig build's own CPU code (built without AVX for the
rig's Celeron) gives the same bytes for every split, so on that build's CPU the controls cannot see a wrong split; C217's
CPU checks used a native build, which can.

## Results

| Run | Arguments | Exit | Verdicts | Card memory, largest sample | JIT cache after it | Where it ran |
|---|---|---:|---|---:|---|---|
| `row` | `-ngl 99 --models <models> --test '^store_at'` (FN72's command) | 0 | `store_at` OK on 3 of 3 models | 399 MiB | 27 files, 47,611,492 B | the card |
| `both` | `-ngl 99 -lv 4 -m <model> --test 'store_at(_p)?'`, each model | 0 | `store_at` and `store_at_p` OK, 6 of 6 | 341 MiB | unchanged | the card (every buffer and state store on `CUDA0`) |
| `cpu` | `-dev none -lv 4 -m <model> --test 'store_at(_p)?'`, each model | 0 | `store_at` and `store_at_p` OK, 6 of 6 | 4 MiB | unchanged | the CPU |

The test's `--test` filter matches whole case names, so FN72's `'^store_at'` selects `store_at` alone; `both` adds
`store_at_p`. The models table mode (`--models`) sets the log threshold to 0, so its per-model logs keep only the cases'
reports; one model at a time with `-lv 4` keeps the load and context lines that name each buffer's device (Q1380
would keep them in a table run).

The driver ran three times in a row (17:04, 17:05 and 17:08), the first two with `both` and `cpu` in table mode (`-lv 4`,
then `-v`), which kept no device lines; this folder is the third. In all three every case passed and every control
reported as above (the same 30 report lines), with the JIT cache at 27 files, 47,611,492 B after `row`. Card memory,
largest sample: `row` 439, 423 and 399 MiB; `both` 503 and 465 MiB (table mode) and 341 MiB (one model at a time);
`cpu` 4, 4 and 4 MiB.

The binary: `test-llama-context` of the rig tree `llama.cpp-20261111-68ea07e60-pascal-p106-v4-b12246-c217` (md5
`99e11c15741e625c2286b7e4d98f2107`, built into the tree by C217's `run/checks.sh`), 187 sm_61 cubins and 187 compute_61
PTX listings, nothing else ([archs.txt](archs.txt)). The models: C217's three random models (`test-llama-archs -a
'^<arch>$' -s 1 -o <dir>`: `qwen4exp-moe.gguf`, `qwen35-dense.gguf`, `mamba-dense.gguf`). The card: RTX 3080 (compute
capability 8.6), driver 595.99.02, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`. The fork's PTX-mode settings were set
for every run (as in [FN63](../2026-11-11_fn63_sparse_256_512/README.md) and
[FN68](../2026-11-11_fn68_c620_ptx/README.md)): `CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5
GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`. The third run took 81 s from the start to the
container's end (17:08:42–17:10:03), 33 s of it the preflight.

## How to run it

On a build machine with Docker and one free card, from a checkout of this repository, with the branch's rig tarball
unpacked (its `bin/` holds `test-llama-context`, built into it by C217's `run/checks.sh`) and C217's random models in a
folder:

```bash
BIN=<rig tree>/bin MODELS=<folder of the random models> OUT=<work dir>/out GPU="--device nvidia.com/gpu=<card>" \
MOUNTS=<the folder BIN, MODELS and OUT lie on> DOCKER_ARGS="--cpu-shares=256" PREFLIGHT=<path of cuda_preflight.sh> \
setsid -f nohup bash docs/data/2026-11-11_fn72_c217_ptx/run/ptx_check.sh > <work dir>/driver.out 2>&1 < /dev/null
```

`OUT/done` marks the end; `OUT/steps.txt` has one line per step ([steps.txt](steps.txt) here), `OUT/reports.tsv` the
controls' reports and `OUT/devices.tsv` the device lines.

## Files

| File | What |
|---|---|
| [runs.tsv](runs.tsv) | every run: mode, arguments, exit code, verdicts, the card's memory samples and largest value, the JIT cache after it, where it ran |
| [controls.tsv](controls.tsv) | each control per model, case and `n_rs_seq`: C217's native CPU build, the card, the rig build on the CPU |
| [reports.tsv](reports.tsv) | the cases' report lines: run, model, case, fill, report |
| [devices.tsv](devices.tsv) | the per-model logs' device lines of `both` and `cpu`, counted |
| [archs.txt](archs.txt) | the GPU code in the binary (`cuobjdump`) |
| [steps.txt](steps.txt) | the driver's steps, paths replaced by their variables |
| [run/ptx_check.sh](run/ptx_check.sh) | the run: preflight, the container, the three runs, the readings, the tables |

## Provenance and licence

Every number here comes from random models that `test-llama-archs` builds from its seed and from the token ids that
`test-llama-context` generates; no model file, prompt or recorded text was read. The fork is MIT-licensed, as
llama.cpp; these tables and the script are published with this repository.
