# Q1765: a rounding yardstick on Ornith-1.5-122B that changes only the attention kernel (2026-11-25)

[C994](../2026-11-24_c994_attn_fault_selfref/README.md) set the form a check of a new attention kernel takes on
Ornith-1.5-122B Pascal N2P GGUF (`valid1`): the new kernel's output scored against the old kernel's on the same Q8_0
cache, stretch by stretch, beside a yardstick for what summing in another order does. Its yardstick, the Q8_0 cache
read in micro-batches of 128, reorders every sum in the model and is itself as large as an est. 4.25 % attention error,
so the check sees errors only from est. about 4.4 %. This folder measures a yardstick that changes only the attention:
the same cache and micro-batch with the fork's compute 6.1 prompt attention kernel switched off, so the build's
generic tile kernel takes the prompt passes ([Q1765](../../../quants/experiments.md)). It is a correctness check on
three RTX 3090s through the rig build's compute 6.1 PTX; no time on these cards means anything for a P106.

**Result:** the run is in progress; this section is filled from [verdict.tsv](verdict.tsv) when it ends.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `3dfad876f` (C908's rig tarball `llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`; its `llama-perplexity` md5 `08187b308bee6912f435442b8d7f95ef`), C976's, C989's and C994's build |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`) |
| Stretches | C976's 12: `-c 8192 --chunks 12`, the text's first 98,304 tokens, each chunk its own stretch with its own context, scored on its second half (positions 4,096–8,191) |
| Settings in every run | C994's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -ub 512 -ctk q8_0 -ctv q8_0`; `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's other Pascal kernels on as a P106 runs them, with a P106-090's launch shapes (`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`); `GGML_SCHED_DEBUG_REALLOC=1`; plus `-lv 4` (log level only) |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight before the pass |

### The arms (one pass, s)

| Arm | Prompt attention kernel | Fault | What it is |
|---|---|---|---|
| `q8` | the fork's compute 6.1 kernel (`GGML_CUDA_FATTN_TILE_PASCAL=2`) | | writes the reference: C994's `q8` |
| `q8g` | the build's generic tile kernel (`GGML_CUDA_FATTN_TILE_PASCAL=0`) | | the attention-only yardstick |
| `q8f1` | the fork's compute 6.1 kernel | `=1` | C994's 5 % error (every output of the kernel × 1.05), again |

**What `q8g` changes.** The build holds only compute 6.1 code, so no tensor-core attention kernel exists in it, and a
prompt pass of 512 tokens goes to the tile kernel. With `GGML_CUDA_FATTN_TILE_PASCAL=0` the fork's kernel declines it
and the build's generic tile kernel (`ggml_cuda_flash_attn_ext_tile`, the one upstream llama.cpp runs on Pascal) takes
it. That kernel reads F16 K/V, so the Q8_0 cache is first converted to F16 copies in the compute buffer, rounding
each value to F16's 11 bits. Everything outside the attention (the matrix products, the recurrent layers, the
micro-batch) is the same as in `q8`. The generic kernel logs nothing, so the log shows which kernel ran by what is
missing: the fork's kernel logs one line the first time it takes a pass (C994's logs: `compute-6.1 prompt attention
kernel: head 256, q8_0 K/V`), and `q8g`'s log has none; its compute buffers are larger by the F16 copies.

## The check and its rule (fixed before the run)

For each stretch i, an arm's KLD against `q8`, its top-token disagreement (100 − same top %) and its ln PPL ratio,
read from `llama-perplexity`'s running means as in C994: G_i, H_i for `q8g`; F_i, T_i for `q8f1`; U_i, V_i for C994's
`q8u` (the Q8_0 cache at `-ub 128`, scored against the same reference, whose identity this run checks). Mean over the
12 stretches, standard error SE = sd/√12, t = mean/SE.

- **The answer to the row:** `q8g`'s KLD over all tokens against C994's 0.007963, and per stretch G_i − U_i (two-sided)
  and G_i/U_i − 1.
- **The new-kernel check with the new yardstick:** F_i − G_i (KLD, top-token disagreement, ln PPL ratio) and
  (F_i − G_i)/G_i (KLD, top-token disagreement). "Seen" is a mean at least 3 SE above zero in KLD and top-token
  disagreement, |t| ≥ 3 in ln PPL ratio, as in C994.
- **The smallest error seen, est.:** (a) the error whose KLD equals the yardstick's, 0.05 × √(yardstick KLD / 0.011030)
  over all tokens (C994's yardstick: 4.25 %), and per stretch 0.05 × √(mean G / mean F); for top-token disagreement
  0.05 × mean H / mean T. (b) As in C994, F_i scaled by (ε/0.05)² and T_i by ε/0.05 in each form against `q8g`: the
  smallest ε at which t reaches 3 (C994 against its yardstick: 4.58 % and 4.43 % in KLD, 5.09 % and 4.97 % in
  top-token disagreement). Both are estimates from one size of error and are marked so.
- **Validity:**
  - **The same reference as C994:** `q8g`'s and `q8f1`'s reference perplexity (read from `q8`'s file) equals C994's
    1.577859, and `q8f1` repeats C994's `q8f1` on every stretch (KLD to 0.00001, same top to 0.001 %) and in the mean
    KLD (0.011030). Then C994's U_i and V_i are on the same reference and can be set beside G_i.
  - **The kernels:** `q8` and `q8f1` log the fork's prompt kernel at head 256 with Q8_0 K/V; `q8g` logs no such line.

## Files

| File | What |
|---|---|
| [run/run.sh](run/run.sh) | the pass, with each arm's settings |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |
| [run/c1032_table.py](run/c1032_table.py) | the tables and the rule from the logs and C994's tables |

## Commands

On a build machine with Docker and three free cards (`W` a work folder on a disk with ~30 GB free for the reference;
the bin folder unpacked from the rig tarball):

```bash
export WORK=$W BIN=<3dfad876f tarball>/bin GPUS=0,1,2 \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> TEXT=<agent-heldout.txt> \
  MOUNTS="<the folders of the build, model and text>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<a CUDA check of the cards, optional>
setsid -f nohup bash run/run.sh > $W/driver.log 2>&1 < /dev/null      # pass s; $W/DONE at the end
# once $W/DONE exists, from this folder:
python3 -I run/c1032_table.py $W ../2026-11-24_c994_attn_fault_selfref .
```

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
