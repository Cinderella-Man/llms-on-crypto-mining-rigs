# Q230 + Q441: Flash-Next's QSA layers read only their selected cells on Pascal (2026-11-04)

Qwen3.8-Flash-Next has 12 attention layers, and each one attends only to the cells its indexer picks: the top 2,048
cells (512 pools of 4) plus the incomplete pool, 2,051 per query token (QSA). llama.cpp hands `FLASH_ATTN_EXT` that
selection as a mask with an `n_kv_max` hint, and only its Turing tensor-core kernel gathers the kept cells; on a
P106-090 the dense kernels read every K and V row of the cache and mask most of them away. At 32k cells that is
64 MiB per layer and token (F16 cache) where 4 MiB would do. This folder holds the Pascal path that reads only the kept
cells ([Q230](../../../quants/experiments.md)), the first part of Q441 (the per-token host inputs of the QSA layers),
the checks run without a GPU, and the commands that check and time it on GPUs.

**Status: written and checked on a CPU; not yet run on any GPU.** Miner Llama branch `pascal-p106-v3-b12076-qsa`
`bdb902d95` (3 commits on `pascal-p106-v3-b12076-ladder` `98bbd6a2f`), behind `GGML_CUDA_FATTN_SPARSE_PASCAL` (off by
default; then the branch runs `-ladder`'s kernels with the same inputs).

## The path

`ggml/src/ggml-cuda/fattn-sparse-pascal.cu` takes a `FLASH_ATTN_EXT` call with `n_kv_max > 0` on a GPU without Turing
tensor cores (compute capability below 7.5, or a build without Turing code), with head size 128 or 256 (K and V the
same), 4, 8, 12 or 16 query heads per KV head, F16 or Q8_0 K and V, an F16 mask, at most 8 query tokens, and no ALiBi,
softcap or `kv_rows`. Anything else goes to the stock kernels unchanged, as do prompts.

1. **Compaction:** upstream's SIMT kernel `flash_attn_mask_to_sparse_indices` (the Turing path uses the same one)
   lists each query token's finite mask columns in cache order, with their count.
2. **Gather** (`fattn-sparse-pascal-impl.cuh`): each list is split into equal shares of whole 32-entry tiles over
   about 4 blocks per SM (10 splits for one decode token on a 5-SM card, at least 64 entries a split). A block has one
   warp per 4 query heads of one KV head (Flash-Next: 3 warps for its 12), all on the same cells, so a K or V row comes
   from DRAM once and from the caches for the other warps. Lane l owns dimensions D/32·l to D/32·l + D/32 − 1: the
   query of its 4 heads, scaled by scale·log2(e), and their V accumulators stay in registers, and each K/V row is
   converted from F16 or dequantized from Q8_0 once per warp. Per tile, every lane holds one list entry (its cell and
   mask value); the scores of 8 keys × 4 heads are summed over the lanes with one transpose-reduce (31 shuffles for 32
   scores), an online softmax in base 2 runs per head over the 8 lanes that hold it, and P·V adds each V row once for
   the 4 heads.
3. **Merge:** each warp writes its partial output, maximum and sum; the last block of a column (query token × KV head)
   to finish, found with a counter, merges the splits, adds the sinks and writes the output. Two launches per layer.

FP32 throughout, so results match the dense kernels to rounding, not bit for bit. The kernels take 113–161 registers
with no stack frame and no spills ([registers.tsv](registers.tsv)). Written as `upper ? v[i] : v[i + n/2]` inside a
loop over two variables, the transpose-reduce compiled to indexed local-memory loads and stores (256 a kernel on
sm_61); with each step a template on its shuffle distance and the halves selected bitwise, every index is a constant
and the values stay in registers.

| Variable | Values |
|---|---|
| `GGML_CUDA_FATTN_SPARSE_PASCAL` | 0 off (default); 1 on GPUs without Turing tensor cores once the cache holds `_MIN_RATIO` × `n_kv_max` cells per query token; 2 on any NVIDIA GPU at any size (testing) |
| `_MIN_RATIO` | default 1.5: one decode token from 3,077 cells, a 4-token verify from 12,306 (each query token reads its own list; the dense kernels read the cache once for all of them) |
| `_NQ` | most query tokens per sequence, default 8 |
| `_SPLITS`, `_NSM` | splits per list; the SM count the splits assume (5 gives a P106-090's on another card) |
| `_FAULT` | testing the tests: 1 output × 1.01, 2 mask values ignored, 3 every split's last entry dropped |

The design is the fork's own: the warp layout, transpose-reduce and last-block merge follow its one-token Pascal decode
kernel (`fattn-pascal.cu`), and the compaction is upstream's. Strata's pre-Turing kernels (`attn_chunk_kernel_pre75`,
`attn_merge_kernel`; MIT) split the selection into chunks and merge them the same way; no Strata code is used.

## Q441: the QSA layers' host inputs

Q441 asked for the QSA layers to run without per-token host masks. What the source shows on `-ladder`:

- **The attention mask is already gone:** with every attention layer a QSA layer, llama.cpp leaves the hybrid cache's
  `kq_mask` unallocated (`llm_graph_input_mem_hybrid::set_input`), and the selection replaces it on the cards.
- **What is uploaded per ubatch** is the k-pool layout: `pool_cells` (4 bytes per pool), `pool_idxs` (16 bytes per
  pool), `pool_mask` (2 bytes per pool and token), `tail_idxs` and the new pools' members: at one token about 5.5 bytes
  per cell, 176 KiB at 32k cells. Every card that runs a QSA layer receives one copy (nine on the rig).
- **The scheduler copies a card's inputs when it reaches that card's split** (`ggml_backend_sched`: user inputs are
  copied right away, after the previous use of that copy slot), while the cards before it still compute. At ~13 ms of
  GPU time per card and token against ~0.9 ms for 176 KiB over PCIe 1.0 x1, only the first card's copy and the host's
  work to build the inputs are on a decoded token's path: est. ~1–1.5 ms at 32k, ~0.2 ms at 5k.
- **The host's part, done:** each padded new-pool entry takes the lowest cell that is no pool's rep, and
  `set_input_kpool` sorted every rep each ubatch to find them; marking the reps below n_pad + n_reps finds the same cells
  in linear time. The inputs are unchanged, bit for bit (the CPU check below).
- **The measurement, added:** `LLAMA_QSA_INPUT_STATS=N` logs every N ubatches the inputs' host time and bytes per card.
  Tomorrow's runs set it; a device-resident layout is [Q640](../../../quants/experiments.md), parked until that log and
  a profile put the inputs on the critical path. CUDA's `SET_ROWS` writes no I32, so per-layer stores would cost a cast
  and a few launches per layer (est. 0.2–0.4 ms a token at any context) against the copies they save.

## Expected on the rig

Dense reads per QSA layer and token are cells × 2 KiB with an F16 cache (2 KV heads × 256 × 2 bytes, K and V), at
the ~21 GB/s Q210 measured for these reads on a P106 ([Q210](../../../quants/experiments.md)). The sparse path reads
2,051 × 2 KiB = 4 MiB per layer and token plus 2 bytes per cell of mask, est. 0.08–0.15 ms per layer (two launches,
~35 µs of arithmetic and ~50 µs of gathered 512-byte rows at ~80 GB/s), 1–2 ms for the 12 layers.

| Cells in the cache | Dense, 12 layers, F16 | Sparse, 12 layers | Per decoded token |
|---:|---:|---:|---:|
| 5,120 | ~6 ms | ~1–2 ms | **−3 to −5 ms** |
| 16,384 | ~19 ms | ~1–2 ms | **−15 to −18 ms** |
| 32,768 | ~38 ms | ~1–2 ms | **−34 to −37 ms** |
| 131,072 | ~153 ms | ~1–2 ms | **−145 to −150 ms** |

A Q8_0 cache halves the dense column (and the gains). The agent check's turns end at 4.6k–7.5k tokens (Q210), so its
generation est. +3–6 % (about −3 to −6 ms of ~110 ms per token); sessions grow every turn, so the gain grows with them.
Under the switch's threshold (3,077 cells per query token) and for prompts the dense kernels keep running. A verify
batch of k tokens reads k lists: mode 1 takes it from 1.5 × 2,051 × k cells, and the perf run below (mode 2, forced)
shows where it starts to pay.

## Checks without a GPU

| Check | Result |
|---|---|
| Fork build (`TESTS=1`, sm_61 + compute_61 PTX) | compiles; 113–161 registers, no stack, no spills |
| `bench/fattn_sparse` CPU check: host model of the kernel against FP64 and ggml's CPU `FLASH_ATTN_EXT`, 10 shapes × F16/Q8_0 | **20 / 20 pass**: largest error 1.6·10⁻⁸–9.5·10⁻⁷ of the largest output against FP64, NMSE 4.9·10⁻⁹–1.2·10⁻⁵ against ggml ([cpu_check.md](cpu_check.md)) |
| The same with the kernel's 3 fault codes | each fails all 20 |
| The same with 4 source mutations of the header | Q8_0 rows without their scale: fails exactly the 10 Q8_0 cases; splits' sums merged unscaled: 14 (all but the single-split cases); every split but the first skipping an entry: 14; scores mapped to the wrong key: 20 |
| `test-llama-archs -a qwen4exp -s 42` (CPU; a random qwen4exp whose attention layers are all QSA layers) | OK on the branch and on `-ladder`, the same numbers: logits 0.00, state roundtrip OK, chunked decode against one batch NMSE 3.24·10⁻¹⁴ |
| `test-llama-context` on that model (CPU) | 17 passed, 1 skipped, 0 failed, on both |

## Files

| File | What |
|---|---|
| [cpu_check.md](cpu_check.md) | the CPU check's table per case, and the fault and mutation verdicts |
| [registers.tsv](registers.tsv) | per kernel instantiation: registers, stack, spills (ptxas, sm_61) |
| [run/qsa_gpu.sh](run/qsa_gpu.sh) | on one GPU: test-backend-ops `FLASH_ATTN_EXT` at modes 1, 2 and 0, the fault codes, `fasp_check --gpu`, the perf pairs (perf.tsv), `fasp_check --bench` |
| [run/long_decode.sh](run/long_decode.sh), [run/long_requests.py](run/long_requests.py) | on the rig: decode after 8k and 32k prompts with the switch off and on (greedy, so both arms generate the same text), optionally with the kernel profiler |
| [run/agent_arms.sh](run/agent_arms.sh) | the agent check, switch off and on, same build and flags |
| [run/cpu_e2e.sh](run/cpu_e2e.sh) | the CPU end-to-end check (test-llama-archs, test-llama-context) |
| [run/build.sh](run/build.sh) | the builds and CPU checks, as run |
| [bench/fattn_sparse](../../../bench/fattn_sparse/README.md) | the CPU check, the standalone GPU check and bench |

Test cases added to `tests/test-backend-ops.cpp`: eval at Flash-Next's QSA call (head 256, 2 × 12 heads, 2,051 cells)
at 1, 2, 3, 5 and 8 tokens and 8k/32k cells in the cache layout, two sequences, head 128 with sinks, 4 and 16 heads
per KV head, lists of one tile and shorter, F16 and Q8_0; perf at 1 and 4 tokens from 4k to 128k cells.

## Commands

The builds and CPU checks ([run/build.sh](run/build.sh)), on a machine with Docker, CPU only:

```bash
WORK=<work folder> REPO=<this repository> LOCK=<lock file> docs/data/2026-11-04_qsa_sparse_attn/run/build.sh
```

**1. Correctness through the compute_61 PTX on a newer GPU** (~20 min; `NSM=5` gives the P106's splits), from the
tarball's tree and `fasp_check` built as above:

```bash
docker run --rm --gpus device=<n> -v <work>:/w -e CUDA_FORCE_PTX_JIT=1 -e BIN=/w/<tarball tree>/bin \
  -e FASP=/w/bench/fattn_sparse/build/fasp_check -e OUT=/w/ptx -e NSM=5 -e PARTS="check fault fasp" \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 bash /w/qsa_gpu.sh
```

`console.txt` must show `ok` for `sparse1`, `sparse2`, `sparse0` (every case passing), `fault1` (output × 1.01, inside
the NMSE limit: every case passing), `fault2` and `fault3` (cases failing) and `fasp_gpu` (20 / 20).

**2. One P106** (~1.5 h, rig otherwise idle; the CPU references run on the rig's Celeron): the tarball unpacked on the
rig, then

```bash
ssh $RIG "BIN=<tree>/bin FASP=<dir>/fasp_check OUT=<dir>/out PARTS='check fault fasp perf bench' \
  setsid nohup bash <dir>/qsa_gpu.sh > <dir>/qsa_gpu.out 2>&1 < /dev/null &"
```

`perf.tsv` gives per cells × tokens × cache type the stock kernels' µs per call, the sparse path's, and the speed-up.

**3. Long contexts on the rig** (~1 h; the switch off and on, decode after 8k and 32k prompts):

```bash
ssh $RIG "BIN=<tree>/bin MODEL=<first shard> OUT=<dir>/long setsid -f nohup bash <dir>/long_decode.sh \
  > <dir>/long.out 2>&1 < /dev/null"
#   then with PROFILE=1 DEPTHS=32768 OUT=<dir>/prof for FLASH_ATTN_EXT's time per token (scripts/prof_summary.py)
```

**4. The agent check** (~45 min an arm), switch off (`sparse0`, the dense kernels as on `-ladder`) and on:

```bash
set -a; . <private env>; set +a   # agent_check.sh's environment, REL_BIN = the tarball's llama-server
ARGS='-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio' docs/data/2026-11-04_qsa_sparse_attn/run/agent_arms.sh <out>
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), branch `pascal-p106-v3-b12076-qsa`. The check's prompts on the rig are the
rig's own Python standard-library source (PSF licence), generated there and not stored. Model: Swift-1.5 GSQ-RCO
Qwen3.8-Flash-Next Q2_0 ([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open
License 1.0 and qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md).
