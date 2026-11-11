# FN71 (Q1298): C702's captured prompt graphs against eager launches through the compute 6.1 PTX on one newer card (2026-11-11)

C702 ([design and CPU check](../2026-11-11_c702_prompt_graphs/README.md), Q1240) adds the switch
`GGML_CUDA_GRAPHS_FORCE=1` to Miner Llama's branch `pascal-p106-v4-b12246-c702` (`2ec3a0927`): a prompt micro-batch's
graph of more than 8 tokens is captured as a CUDA graph at every change of its properties and replayed only from
executables captured from its current properties. On nine P106s it was exact and a gain
([R82](../2026-11-11_r82_c702_p106/README.md), Q1297). FN71 ran the rig build's own `llama-perplexity` with the switch
off and on, in a container that holds one RTX 3080 (10 GB). The rig build carries sm_61 machine code and compute_61 PTX
only, so every ggml kernel it runs on that card is the PTX a P106 would run, JIT-compiled by the card's driver. A kernel
that is not safe to capture, or that reads something at capture time, would give other bytes with the switch on.
Correctness only, no speed. The registry row is Q1298 in [quants/experiments.md](../../../quants/experiments.md).

**Result: with the switch on, every scored token's log-probabilities over the whole vocabulary are the same bytes as
eagerly, at 512-token and at 48-token micro-batches.**

- **The row's command (`-ub 512`, 8 chunks):** the two KL-divergence base files are identical (4,064,632,788 B each, md5
  `d3e0e84ccfcb5d5dd5f901d9d4036d47`), and every chunk's perplexity and the final estimate are the same in every digit
  (`PPL = 4.8454 +/- 0.12392`). The switch captured 628 graphs and replayed 6, all of them small (1 to 5 launch groups).
  **The graph that holds the card's 15 layers ran eagerly in every evaluation, with the switch on as off,** so at 512
  tokens the check did not capture the expert layers.
- **Why, on this card only:** the host picks the kernel for a quantized expert product (`MUL_MAT_ID`) from the card's
  real compute capability, not from the PTX. On a card with FP16 tensor cores (compute 7.0 and up),
  `ggml_cuda_should_use_mmq` leaves MMQ for 64 tokens or more and takes cuBLAS, and its expert fallback synchronizes the
  stream (`ggml_cuda_mul_mat_id_needs_sync`), so the backend turns CUDA graphs off for any graph that holds such a
  product. A P106 (compute 6.1, no FP16 tensor cores) takes MMQ at every size, and there the prompt graphs are captured
  (R82: 621 captures).
- **At 48 tokens (`-ub 48`, where this card takes MMQ too, 1 chunk):** the same layers' graphs were captured 9 times and
  replayed 31 times, and the base files are again identical (508,079,116 B each, md5
  `fd83eed66fda37a0a540bb8ae63bd065`), perplexity `4.5914 +/- 0.32867` both. So the expert products through MMQ (the
  P106's prompt kernels), with the attention, gated delta net and router layers in the same graphs, run captured and
  replayed through the PTX and give eager's bytes. The same pair over 8 chunks was stopped in its eager run to free the
  card for another job: Q1386.
- **Eager is repeatable on this card:** chunk 1 of the sizing trial (a separate process) and chunk 1 of the 512-token
  pair's eager run are the same bytes, so the identity above is not a coincidence of a nondeterministic path.

## Where the model ran

The model, Qwen3.6-35B-A3B Q4_K_M (`qwen35moe`: 40 layers, a full-attention layer every fourth and gated delta net
layers between, 256 experts, vocabulary 248,320; 20,419,565,568 B), does not fit the card's 10,240 MiB, so both pairs
load 16 of its 41 layers onto the card (`-ngl 16`: the output layer and layers 25 to 39); a one-chunk eager trial at
`-ngl 16` peaked at 9,477 MiB.

| Reading | 512-token pair | 48-token pair |
|---|---|---|
| load lines (every run) | `offloaded 16/41 layers to GPU`; `CUDA0 model buffer size = 7443.38 MiB`, `CPU_Mapped` 12,019.76 MiB; KV 16.02 MiB and RS 23.03 MiB on `CUDA0` (24.02 and 39.78 MiB on the CPU) | the same |
| compute buffer | `CUDA0` 505.00 MiB, `CUDA_Host` 26.12 MiB | `CUDA0` 46.74 MiB, `CUDA_Host` 9.91 MiB |
| the other 25 layers | op offload (the default) moves their 512-token products to the card, with the weights copied over for each micro-batch | `--no-op-offload`: computed on the CPU (at 48 tokens op offload would copy the experts for each of 43 micro-batches) |
| card memory, largest sample (1 MiB before) | 9,479 MiB off, 9,481 MiB on | 7,821 MiB off, 7,821 MiB on |
| the driver's JIT cache, empty at the start | 25 files, 146,501,338 B after the eager run; unchanged after the switched one | 26 files, 148,957,472 B; unchanged |
| the switch's and the profiler's info lines | in the switched run's log | the same |

## The graphs, from the switched runs' profiles

`GGML_CUDA_PROFILE_MODE=graph` writes one line per evaluation of each graph (key: its first node), with `eager`,
`capture` or `replay`; the last evaluation of each graph is never written ([profile.tsv](profile.tsv)).

| Pair | Graphs | Evaluations written | Eager | Captures | Replays | The large graph (the card's layers, 502 to 525 launch groups) |
|---|---:|---:|---:|---:|---:|---|
| 512 tokens | 928 | 13,920 | 13,286 | 628 | 6 | eager in all 30 (two keys alternating) |
| 48 tokens | 2 | 41 | 1 | 9 | 31 | the only graphs: the first evaluation eager (each context's first batched graph runs eagerly), then 9 captures and 31 replays |

- **512 tokens:** 926 small graphs (1 to 5 launch groups) are the parts of the CPU-held layers that op offload runs on the
  card, each evaluated 16 times (the keys alternate between micro-batches). 35 of them never ran eagerly (31 captured in
  all 15 written evaluations, 4 captured and replayed), 69 ran eagerly and were captured once or twice (41 and 27) or
  fourteen times (1), and 822 ran eagerly in every evaluation, as did the large graph
  ([prof-graphs-ub512.tsv](prof-graphs-ub512.tsv)). No graph that holds an expert product can be captured here (the
  synchronizing fallback above). No line was written as not ready or unmatched.
- **48 tokens:** the card's 15 layers and the output run as one graph per micro-batch, under two keys that alternate.
  Their sequences ([prof-graphs-ub48.tsv](prof-graphs-ub48.tsv)): `ecrrrcrrrrcrrrrrcrrr` and `ccrrrcrrrrrcrrrrcrrrr`
  (e eager, c capture, r replay): a capture where a recorded property changed (the KV views grow with the cache), a
  replay of the executables captured from the same properties otherwise.

## What this does not cover

- **The MMQ kernels at 512 tokens.** At 48 tokens MMQ picks its tile width for 48 columns; the 512-token tiles were not
  captured here. On a P106 they were (R82, identity on nine cards). Q1387 would make the host choose kernels as for
  compute 6.1 on a newer card, so a check like this one follows the P106's path at every size.
- cuBLAS's own kernels run as the card's native code, not as PTX (the 512-token pair's dense products).
- Decode and verify graphs (8 tokens or fewer) keep the warm-up by design; pipeline parallelism over several cards is
  R82's.
- The 48-token pair scored one chunk (1,023 positions); its 8-chunk run is Q1386.

## How to run it

On a build machine with Docker and one free card, from a checkout of this repository, with the branch's rig tarball
unpacked (`llama.cpp-20261110-2ec3a0927-pascal-p106-v4-b12246-c702`, md5 of the tarball
`2bb77d3a2324f1842fcd5a6a0240f784`), the model and wikitext-2's `wiki.test.raw`:

```bash
# the row's pair: 512-token micro-batches, 8 chunks, the base files compared and deleted
BIN=<rig tree>/bin MODEL=<dir>/Qwen3.6-35B-A3B-Q4_K_M.gguf TEXT=<dir>/wiki.test.raw OUT=<work dir>/ub512 \
GPU="--device nvidia.com/gpu=<card>" MOUNTS=<the folder BIN, TEXT and OUT lie on> MOUNTS_RO=<the model's folder> \
NGL=16 DOCKER_ARGS="--cpu-shares=256" PREFLIGHT=<path of cuda_preflight.sh> \
setsid -f nohup bash docs/data/2026-11-11_fn71_c702_ptx/run/fn71_check.sh > <work dir>/driver.out 2>&1 < /dev/null

# the 48-token pair: the same with
#   OUT=<work dir>/ub48 CHUNKS=1 UB=48 PPL_ARGS=--no-op-offload NAME=fn71-ptx-ub48
# the sizing trial: OUT=<work dir>/trial CHUNKS=1 RUNS=off KEEP_KLD=1
```

`OUT/done` marks the end; `OUT/steps.txt` has one line per step ([steps.txt](steps.txt) here, paths replaced by their
variables). The chunk-1 comparison of the trial against the 512-token eager run, with both base files kept
(`KEEP_KLD=1`; a file is a 20-byte header, `n_chunk × 2,048` token ids of 4 bytes, then per chunk 1,023 positions of
248,324 16-bit values):

```bash
cmp -n 8192 -i 20:20 trial/off.kld ub512/off.kld
cmp -n 508070904 -i 8212:65556 trial/off.kld ub512/off.kld
```

The binary: `llama-perplexity` of the rig tree (md5 `847d9baa8db509d33d97e3f62150751a`), 187 sm_61 cubins and 187
compute_61 PTX listings ([archs.txt](archs.txt)). The card: RTX 3080 (compute capability 8.6), driver 595.99.02, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`. The fork's PTX-mode settings were set for every run (as in
[FN72](../2026-11-11_fn72_c217_ptx/README.md)): `CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5
GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`. Every other switch of the branch was off.

## Files

| File | What |
|---|---|
| [runs.tsv](runs.tsv) | every run: arguments, switch, exit code, final estimate, the card's largest memory sample, the JIT cache, wall time, where it ran |
| [ppl.tsv](ppl.tsv) | every chunk's perplexity as printed |
| [kld.tsv](kld.tsv) | the base files' sizes, md5 and `cmp` results |
| [profile.tsv](profile.tsv) | the switched runs' evaluations by kind and launch groups |
| [prof-graphs-ub512.tsv](prof-graphs-ub512.tsv), [prof-graphs-ub48.tsv](prof-graphs-ub48.tsv) | per graph: eager, capture, replay, lines not ready or unmatched, the sequence |
| [devices.tsv](devices.tsv) | the load and context lines that name a device, and the switch's and profiler's info lines |
| [archs.txt](archs.txt) | the GPU code in the binary (`cuobjdump`) |
| [steps.txt](steps.txt) | the driver's steps of all four launches, paths replaced by their variables |
| [run/fn71_check.sh](run/fn71_check.sh) | the run: preflight, the container, the runs, the readings, the comparison, the profile's summary |

## Provenance and licence

The text is the wikitext-2 test set (`wiki.test.raw`, CC BY-SA 3.0); nothing of it is republished here. The model is
Qwen3.6-35B-A3B in a Q4_K_M GGUF (Apache 2.0). The fork is MIT-licensed, as llama.cpp; these tables and the script are
published with this repository.
