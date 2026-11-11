# FN78 (Q1386): C702's captured prompt graphs against eager launches at 48-token micro-batches over 8 chunks, through the compute 6.1 PTX on one newer card (2026-11-11)

[FN71](../2026-11-11_fn71_c702_ptx/README.md) (Q1298) checked C702's switch `GGML_CUDA_GRAPHS_FORCE=1` (Miner Llama
branch `pascal-p106-v4-b12246-c702`, `2ec3a0927`; [design](../2026-11-11_c702_prompt_graphs/README.md), Q1240) with the
rig build's own `llama-perplexity` on one RTX 3080, which can run only the build's compute_61 PTX. At 512-token
micro-batches that card never captures the graph that holds the expert layers: with FP16 tensor cores the host sends a
64-token or larger expert product to cuBLAS, whose fallback synchronizes and turns CUDA graphs off. At 48 tokens it
takes MMQ, the kernels a P106 runs, and there FN71's one chunk was captured and replayed with eager's bytes. FN78 is that
48-token pair over all 8 chunks. Correctness only, no speed. The registry row is Q1386 in
[quants/experiments.md](../../../quants/experiments.md).

**Result: with the switch on, the log-probabilities of every scored token over the whole vocabulary are the same bytes
as eagerly over 8 chunks, while the card's layers were captured 86 times and replayed 255 times.**

- **The base files:** identical, 4,064,632,788 B each, md5 `5ef0ff0d47b95427d9bea3369d9c5311` both
  ([kld.tsv](kld.tsv)).
- **The perplexity:** every chunk the same in every digit, `PPL = 4.8290 +/- 0.12306` both ([ppl.tsv](ppl.tsv)). Chunk
  1 is 4.5914, as in FN71's one-chunk pair.
- **The graphs** ([profile.tsv](profile.tsv), [prof-chunks.tsv](prof-chunks.tsv), [prof-graphs.tsv](prof-graphs.tsv)):
  - The card's 15 layers and the output run as one graph per micro-batch, under two keys that alternate (502 or 524
    launch groups under the first, 523 under the second).
  - Of the 342 evaluations written, 1 ran eagerly, 86 were captures and 255 replays. The one eager evaluation is chunk
    1's first micro-batch: each context's first batched graph runs eagerly.
  - Every chunk has the same sequence of 43 micro-batches: 11 captures and 32 replays (chunk 1: the eager one,
    10 captures, 32 replays; chunk 8: 41 written, since the last evaluation of each graph is never written).
  - A capture comes where a recorded property changed (the KV views grow with the cache); the micro-batches between
    replay the executables captured from the same properties.
  - Each chunk starts from an empty cache, whose first micro-batch has 524 launch groups. In chunks 2 to 8 that
    micro-batch and the next one are both captured, so no chunk replays executables from an earlier chunk.
  - No line was written as not ready or unmatched.

## Where the model ran

The model, Qwen3.6-35B-A3B Q4_K_M (`qwen35moe`: 40 layers, 256 experts, vocabulary 248,320; 20,419,565,568 B), does not
fit the card's 10,240 MiB. As in FN71, 16 of its 41 layers are on the card (`-ngl 16`: the output layer and layers 25
to 39), and `--no-op-offload` keeps the other 25 layers on the CPU ([devices.tsv](devices.tsv)).

| Reading | Both runs |
|---|---|
| load lines | `offloaded 16/41 layers to GPU`; `CUDA0 model buffer size = 7443.38 MiB`, `CPU_Mapped` 12,019.76 MiB; KV 16.02 MiB and RS 23.03 MiB on `CUDA0` (24.02 and 39.78 MiB on the CPU); compute buffer `CUDA0` 46.74 MiB, `CUDA_Host` 9.91 MiB |
| card memory, largest sample (1 MiB before) | 7,821 MiB off, 7,821 MiB on |
| the driver's JIT cache, empty at the start | 26 files, 148,957,472 B after the eager run; unchanged after the switched one |
| the switch's and the profiler's info lines | in the switched run's log |

## What this does not cover

- **The MMQ kernels at 512 tokens.** At 48 tokens MMQ picks its tile width for 48 columns. A P106 runs the 512-token
  tiles captured (R82, identity on nine cards); a newer card sends those products to cuBLAS. Q1387 would make the host
  choose kernels as for compute 6.1 on a newer card.
- cuBLAS's own kernels run as the card's native code, not as PTX.
- Decode and verify graphs (8 tokens or fewer) keep the warm-up by design; pipeline parallelism over several cards is
  R82's.
- The 48-token and 512-token eager runs are not the same bytes (other micro-batches, other kernels); each pair is
  compared within itself.

## How to run it

FN71's script, unchanged: [run/fn71_check.sh](../2026-11-11_fn71_c702_ptx/run/fn71_check.sh). On a build machine with
Docker and one free card, from a checkout of this repository, with the branch's rig tarball unpacked
(`llama.cpp-20261110-2ec3a0927-pascal-p106-v4-b12246-c702`, md5 of the tarball `2bb77d3a2324f1842fcd5a6a0240f784`), the
model and wikitext-2's `wiki.test.raw`:

```bash
BIN=<rig tree>/bin MODEL=<dir>/Qwen3.6-35B-A3B-Q4_K_M.gguf TEXT=<dir>/wiki.test.raw OUT=<work dir>/ub48-c8 \
GPU="--device nvidia.com/gpu=<card>" MOUNTS=<the folder BIN, TEXT and OUT lie on> MOUNTS_RO=<the model's folder> \
NGL=16 CHUNKS=8 UB=48 PPL_ARGS=--no-op-offload NAME=fn78-ptx-ub48 \
DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" PREFLIGHT=<path of cuda_preflight.sh> \
setsid -f nohup bash docs/data/2026-11-11_fn71_c702_ptx/run/fn71_check.sh > <work dir>/driver.out 2>&1 < /dev/null
```

`OUT/done` marks the end; `OUT/steps.txt` has one line per step ([steps.txt](steps.txt) here, paths replaced by their
variables). The script deletes the two 4 GB base files after their `cmp` and md5 (`KEEP_KLD=1` keeps them). The
per-chunk counts in [prof-chunks.tsv](prof-chunks.tsv) split the profile's evaluations (column `eval`, 1 to 343) into
chunks of 43 micro-batches (2,048 tokens at 48 a micro-batch).

The binary: `llama-perplexity` of the rig tree (md5 `847d9baa8db509d33d97e3f62150751a`), 187 sm_61 cubins and 187
compute_61 PTX listings. The card: RTX 3080 (compute capability 8.6), driver 595.99.02, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`. The fork's PTX-mode settings were set for both runs (as in FN71):
`CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`. Every other switch of the branch was off.

## Files

| File | What |
|---|---|
| [runs.tsv](runs.tsv) | both runs: arguments, switch, exit code, final estimate, the card's largest memory sample, the JIT cache, wall time, where it ran |
| [ppl.tsv](ppl.tsv) | every chunk's perplexity as printed |
| [kld.tsv](kld.tsv) | the base files' sizes, md5 and `cmp` result |
| [profile.tsv](profile.tsv) | the switched run's evaluations by graph, kind and launch groups |
| [prof-chunks.tsv](prof-chunks.tsv) | per chunk: eager, capture, replay and the sequence of both graphs |
| [prof-graphs.tsv](prof-graphs.tsv) | per graph: eager, capture, replay, lines not ready or unmatched, the sequence over 8 chunks |
| [devices.tsv](devices.tsv) | the load and context lines that name a device, and the switch's and profiler's info lines |
| [steps.txt](steps.txt) | the driver's steps, paths replaced by their variables |

## Provenance and licence

The text is the wikitext-2 test set (`wiki.test.raw`, CC BY-SA 3.0); nothing of it is republished here. The model is
Qwen3.6-35B-A3B in a Q4_K_M GGUF (Apache 2.0). The fork is MIT-licensed, as llama.cpp; these tables are published with
this repository.
