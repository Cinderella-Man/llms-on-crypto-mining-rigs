# fattn_sparse: checks for the sparse decode attention below Turing

Miner Llama's `fattn-sparse-pascal` path ([Q230](../../quants/experiments.md)) runs `FLASH_ATTN_EXT` calls that carry the
`n_kv_max` hint, as Qwen3.8-Flash-Next's QSA layers make them, on GPUs without Turing tensor cores: it reads only the
cells the indexer kept (about 2,051 per query) instead of the whole cache. This folder checks its arithmetic without a
GPU and runs the kernel alone on one.

`src/fasp_check.cu` includes the fork's `ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh` (the kernel and its per-lane
helpers, free of ggml) and links ggml's CPU backend from the same tree as the reference.

- **CPU mode** (default): a host model of the kernel runs the header's own functions (K/V row decoding for F16 and
  Q8_0, the split ranges, the split count, the merge of the splits, the sinks) and steps through the warp the way the
  kernel does: the transpose-reduce butterfly with its lane-to-value mapping, the max and sum butterflies over the lanes
  of a head, the per-head rescale, the order of P·V. Each case is compared with an FP64 softmax over the listed cells
  (K and V decoded by ggml's own `to_float`) and with ggml's CPU `FLASH_ATTN_EXT` over the whole masked cache.
- **`--gpu`**: also launches the kernel on device 0 with the lists built on the host (as the compaction would), twice
  (its merge counters must return to zero), and compares it the same way.
- **`--bench`**: times the kernel alone at Flash-Next's QSA call (head 256, 2 KV heads of 12 query heads, 2,051 cells)
  at 4k–128k cells and 1 or 4 query tokens: µs per call and GB/s of K/V rows read.
- **`--fault n`**: the kernel's `GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT` codes (1 output × 1.01, 2 mask values ignored,
  3 every split's last entry dropped), in the host model and the kernel.
- **`--rule`**: only the switch's rule (`fasp_rule_takes` in the header: which calls go to the kernel), on the host.
  With `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT` off it must take exactly the calls the switch took before the prompt rule
  (at most `_NQ` query tokens, in mode 1 from `_MIN_RATIO` × `n_kv_max` cells per query token), with it on the same
  decode and verify calls and the prompt calls its definition names, over a grid of 35,964,000 settings × calls. Then
  it prints the switch points at Flash-Next's 2,051 cells and, for each prompt case of test-backend-ops, the kernel it
  takes in each run of `scripts/record_checks.sh`'s suite `qsa-prompt`.
- `--nsm n` picks the split count of an `n`-SM card (default 5, the P106-090); `--threads n` the CPU reference's threads.

The 26 cases (each with an F16 and a Q8_0 cache): Flash-Next decode at 8k and 32k cells, a 5-token verify, 1 and 32
splits, two sequences (one with an empty list), head 128 with 8 query heads per KV head and sinks, head 128 with 4, 16
heads per KV head with a list shorter than a tile, a list of exactly one tile, prompt micro-batches of 16 and 128 query
tokens at Flash-Next's shape, and 512 query tokens over 2 KV heads (the kernel's 1,024 columns). Selections look like QSA's: pools of 4
consecutive cells, the incomplete pool at the end, single cells, mask values 0 or small negatives, a row with one entry.
A case passes within 2·10⁻⁵ of the FP64 reference (largest error over the largest output) and within test-backend-ops'
FLASH_ATTN_EXT tolerance (NMSE 5·10⁻⁴) of ggml's CPU kernel.

| File | What |
|---|---|
| `src/fasp_check.cu` | the check, the GPU run and the bench |
| `build.sh` | builds `build/fasp_check` in `nvidia/cuda:12.9.2-devel-ubuntu24.04`: ggml's CPU backend from `SRC` (static, no AVX, so it runs on the rig's Celeron too), the kernel as sm_61 code plus compute_61 PTX |
| `check.sh` | the CPU check: the tree's header must pass every case; the three fault codes and four source mutations of the header (Q8_0 rows read without their scale, the splits' sums merged unscaled, every split but the first skipping an entry, scores mapped to the wrong key) must each fail cases; `--rule` must pass, and three mutations of the rule (prompt calls by cells per query token, prompt calls taken with the prompt rule off, a call of exactly `_NQ` tokens by the prompt rule) must each fail it |

```bash
SRC=<llama.cpp tree with the path> DOCKER_ARGS="--cpu-shares 256" bench/fattn_sparse/check.sh   # CPU, ~5 min
build/fasp_check --gpu --nsm 5      # on a GPU: a P106 natively, a newer card with CUDA_FORCE_PTX_JIT=1
build/fasp_check --bench --nsm 5    # timings, on the card they are for
```

Results: [docs/data/2026-11-04_qsa_sparse_attn](../../docs/data/2026-11-04_qsa_sparse_attn/README.md); the prompt rule
and `--rule`: [docs/data/2026-11-11_c705_sparse_prompt_rule](../../docs/data/2026-11-11_c705_sparse_prompt_rule/README.md).
