# Q1722: how small an attention-kernel error does the per-stretch quality check see on Ornith-1.5-122B? (2026-11-24)

[C976](../2026-11-23_c976_ornith_q8_cache_kld/README.md) judged a Q8_0 attention cache on Ornith-1.5-122B Pascal N2P
GGUF (`valid1`) by KLD and perplexity. Two of its measures behaved very differently. A single window's KLD could not
tell a 5 % error in the Pascal prompt kernel's output from rounding, while the paired difference over 12 stretches of
8,192 tokens resolved 0.000590 ± 0.000317. This folder calibrates that per-stretch check with deliberate kernel errors,
so that later attention-kernel checks on Ornith know what size of error they can and cannot see
([Q1722](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

**Result: pending (the run starts 2026-11-24 ~00:45).**

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C976's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096`, `-ub 512` unless the arm says otherwise; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only), so each log names the kernel that took the prompt passes |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before every pass |

### The faults

The Pascal prompt attention kernel (`fattn-tile-pascal.cu`) has a fault switch for testing the tests,
`GGML_CUDA_FATTN_TILE_PASCAL_FAULT`. Only that kernel reads it, so a fault that moves the output also shows that the
kernel took the call. On a 3090 the kernel runs only when forced (`GGML_CUDA_FATTN_TILE_PASCAL=2`; its default `1`
asks for compute 6.1 exactly). It takes FLASH_ATTN_EXT at head 128 or 256 with F16 or Q8_0 K/V and at least 3 query
tokens with Q8_0 (9 with F16), so it takes every prompt pass here: Ornith's attention is head 256, and every
micro-batch is 512 tokens (128 in `f16u`).

- **`=1`, output × 1.05:** every output of the kernel scaled by 1.05, a uniform 5 % error in every attention layer's
  output on every token.
- **`=3`, last key tile dropped:** each KV split of a block skips its last key tile (128 keys at head 256). In a
  prompt pass of 512 tokens the kernel walks the whole cache written so far, so the dropped tiles are the newest keys,
  the micro-batch's own. A token near the end of a micro-batch loses its most recent keys, itself included; a token
  earlier in it loses nothing, since those keys were masked anyway. The number of splits comes from the card's
  occupancy and SM count, so how many keys a 3090 drops differs from a P106. A large error on a few tokens.

### The arms

| Pass | Arm | Cache | `-ub` | Fault | Scored against |
|---|---|---|---:|---|---|
| g | `f16` | F16 | 512 | | writes the reference: the v5 preset's cache and micro-batch |
| g | `q8` | Q8_0 | 512 | | the F16 cache |
| g | `q8f1` | Q8_0 | 512 | `=1` | the F16 cache |
| g | `q8f3` | Q8_0 | 512 | `=3` | the F16 cache |
| g | `f16u` | F16 | 128 | | the F16 cache: the rounding yardstick, the same sums in another order |
| h | `q8`, `q8f1`, `q8f3` | Q8_0 | 512 | none, `=1`, `=3` | Ornith-1.5-122B Q8_0's own log-probabilities on the same 12 stretches (C976's pass a); C976's `f16` there is the F16 cache's run |

## The check and its rule (fixed before the run)

**The paired per-stretch check.** For each stretch i (one chunk), take an arm's own KLD, top-token disagreement
(100 − same top %) and ln PPL ratio, and subtract the paired arm's on the same stretch: d_i. The check reports the mean
d̄ over the 12 stretches and its standard error SE = sd(d_i)/√12, and t = d̄/SE. Each stretch is its own stretch of
text with its own context, so the spread across stretches holds even though neighbouring tokens are correlated.
`llama-perplexity` prints running means over the chunks at five decimals; a chunk's own value is k·M_k − (k−1)·M_{k−1},
which carries up to about 0.00006 of printing error a stretch at chunk 12. That error is part of the spread, so the SE
includes it.

**"Seen" means d̄ at least 3 SE above zero** in KLD (a fault on top of an arm can only add KLD on average), and
|d̄| ≥ 3 SE in ln PPL ratio. With 11 degrees of freedom, noise alone gets 3 SE above zero 0.6 % of the time.

The pairs:
- **The fault against its own cache:** `q8f1 − q8` and `q8f3 − q8`, in pass g (against the F16 cache) and pass h
  (against the Q8_0 model). This is what the check sees of the fault alone.
- **The fault against rounding:** `q8f1 − f16u` and `q8f3 − f16u` in pass g; a later kernel check compares a new
  kernel with rounding, since any kernel that sums in a new order moves the output by rounding.
- **The fault the way C976 compared caches:** `q8f1 − f16` and `q8f3 − f16` against the Q8_0 model, with C976's
  `f16` run of pass a (the same build, settings, stretches and reference).

**The smallest scale error seen, est.:** a small error in the kernel's output changes the logits in proportion to its
size, and KLD is second order in a small logit change, so the fault's KLD grows with the square of the error. If
`q8f1 − q8` is seen with mean d̄ and error SE, the smallest output scale error the check sees is
est. 0.05 × √(3 SE / d̄). It is an estimate from one size, and it is marked so.

**Validity:**
- **The repeats:** in pass h, `q8` repeats C976's pass a `q8` on every stretch, and in pass g its first stretch repeats
  C976's pass b `q8` at 8,192 (0.009892), the same tokens with the same settings; otherwise C976's runs are not paired
  with this run's.
- **The kernel:** every arm's log names the compute 6.1 prompt attention kernel at head 256 with the arm's cache type.

## Files

Filled in after the run: the tables, `run/c989_table.py`, `runs.tsv`.

| File | What |
|---|---|
| [run/run.sh](run/run.sh) | the two passes, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for pass g's
reference; the bin folder unpacked from the rig tarball):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  AGENT_BASE=<Ornith-1.5-122B-Q8_0's agent reference, 2026-10-27_ornith_kld> \
  MOUNTS="<the folders of the build, model, text and reference>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # passes g and h; $W/DONE at the end
```

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT). The Q8_0 reference is the makers' Q8_0.
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
