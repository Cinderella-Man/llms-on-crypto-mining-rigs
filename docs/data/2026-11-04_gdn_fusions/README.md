# C18–C20 (Q436–Q438): fewer launches in Flash-Next's recurrent layers (2026-11-04)

Qwen3.8-Flash-Next (`qwen4exp`) is a hybrid: most of its layers are Gated DeltaNet recurrent layers, which keep a
conv history and a 128 × 128 state per value head instead of a KV cache. On nine P106-090 cards, Q215's profile of
Miner Llama v3 put 33.5 of the 136 ms of GPU time per decoded token (summed over the cards) into operations without
weights: copies, norms, scales, sigmoids, concats and the recurrent update
([decode_profile_v3.txt](../2026-11-03_flash_next_baseline/decode_profile_v3.txt)). Each of them is a kernel launch on
a 5-SM card, and the recurrent state is copied twice per layer and token. This folder holds the code change that
removes most of those launches and copies in the recurrent layers, ported from Strata's kernels
([strata.md](../../research/notes/strata.md), ranks 1–3), and its checks. The registry rows are Q436–Q438 in
[quants/experiments.md](../../../quants/experiments.md).

**Status:** written, built for the rig, and checked on the CPU: the fused graph gives the unfused graph's logits bit
for bit. Not yet run on a GPU; the card and rig commands are below.

## The change

Fork branch `pascal-p106-v3-b12076-gdnfuse` at `766c92a22`, seven commits on the speed ladder's engine
`pascal-p106-v3-b12076-ladder` (`98bbd6a2f`). Per recurrent layer and decoded token:

| Part | Before (the b12076 base) | After |
|---|---|---|
| Conv window (C20) | `GET_ROWS` (the history into a temporary), `CONCAT` (history + new tokens), one `CPY` per rollback slot (the new history back), `SSM_CONV`+`SILU`, then `RMS_NORM`+`SCALE` twice (the q and k L2 norms) | one `GDN_CONV` that reads the history from its cache row and writes the new histories into the cache |
| Recurrent state (C19) | `GET_ROWS` (the 3 MiB state into a temporary), then `GATED_DELTA_NET`, which already writes its state back into the cache | `GATED_DELTA_NET` reads the state from its cache row and updates it in place: no temporary, 3 MiB less written and read |
| Output norm (C20) | `RMS_NORM`+`MUL`, `SIGMOID`, `MUL` | one launch |
| Launches (one rollback slot) | 11 | 3 |

- **C18 (Q436) is in the base already:** upstream #31443 (`749db7f03`), which writes the conv history's tail with one
  copy instead of a `cont` and a copy, is part of upstream b12076. In v3's profile that `cont` was the
  `CONT _(view)_(cont)` launch group, 2.56 ms per token over the nine cards. The base-against-v3 run (R35) measures it.
- **`GGML_OP_GDN_CONV`** is a new ggml op (CPU and CUDA): the history read from the cache row by id, the depthwise
  causal conv, SiLU, the first channels L2-normalized per head, and the new history for each rollback slot in the
  output, for one copy back. The CUDA backend fuses that copy, so the kernel writes the histories into the cache.
  The CPU code keeps the sums of `ssm_conv` and `rms_norm` in the same order; the CUDA kernel sums the squares of a
  128-wide head the way `rms_norm`'s vector path does (a float4 per lane, then one warp sum).
- **The state read in place** is a CUDA-backend fusion: when a `GATED_DELTA_NET`'s state comes from `build_rs`'s
  gather of one row, its snapshot copy into the cache fuses too, and the ubatch holds one sequence with no extra states,
  the kernel reads the cache row itself and the gather is skipped. Each block reads its whole part of the state before
  it writes any of it, so the row it reads may be the one it updates. It applies to every Gated DeltaNet model
  (`qwen35`, `qwen35moe` as well), since the graph is unchanged.
- **The gated output norm** is a CUDA-backend fusion of four consecutive nodes, with the same sums and products in the
  same order as the `RMS_NORM`+`MUL` fusion, `SIGMOID` and `MUL` it replaces. `build_norm_gated` puts the gate's
  producers into the graph first so the four nodes are adjacent.
- **Every step handles 1–8 tokens and any number of rollback slots**, so decode steps and k-token verify passes take
  the fused path; prompt ubatches (512 tokens) do too.
- **Switches, for A/B runs on one binary:** `LLAMA_GDN_FUSE=0` builds the conv window and the norm as before;
  `GGML_CUDA_GDN_FUSE=0` turns the CUDA fusions off. Both together give the unfused arm.

Expected gain, from the porting note's budget: −3 to −4 ms per token for the state (Q437) and −2 to −3 ms for the conv
window and norm (Q438) **[estimate]**; the rig profile replaces these.

## Checks so far

**CPU, 2026-11-03** (a build machine without a GPU in the container): `test-llama-archs` on a random `qwen4exp` model
(a Gated DeltaNet layer with the n-gram conv history, an attention layer with the sparse-attention indexer,
hyper-connections), 128 tokens as one batch and then decoded in steps of 1, 2, …, 8 tokens, with the unfused graph
writing the reference logits and the fused graph compared with them ([cpu_check.tsv](cpu_check.tsv)):

| Rollback snapshots | One batch against steps of 1–8 (nmse) | Fused against unfused |
|---:|---:|---|
| 0 | 9.4e-13, both graphs | all 32,768 logits identical |
| 3 (four slots) | 9.4e-13, both graphs | all 32,768 logits identical |

The fused run's graph has `GDN_CONV` nodes and no `SSM_CONV`, `CONCAT` or L2-norm nodes in the recurrent layers, and
its node order is the one the CUDA matchers need: the history copy is the first non-empty node after `GDN_CONV`, only
compute-buffer nodes sit between the state gather and `GATED_DELTA_NET`, whose snapshot copy follows it, and the
norm's `RMS_NORM`, `MUL`, `SIGMOID`, `MUL` are consecutive.

**Not run yet:** any GPU. The quant machine's GPUs were busy and the rigs off. `test-backend-ops` has cases for every
fusion at 1–8 tokens (and 33–512) with 1–4 rollback slots, reading the written row, another cell's row or a rollback
slot's row: `GATED_DELTA_NET_STATE_ROWS`, `GDN_CONV`, `GDN_CONV_CACHE_FUSION`, `GATED_RMS_NORM`.

## Files

| File | What |
|---|---|
| `cpu_check.tsv` | the CPU check: per setting, the conv window's ops, one batch against steps, fused against unfused |
| `run/build_and_cpu_check.sh` | the rig build (`TESTS=1`), `test-llama-archs` from the same tree, then the CPU check in a container without a GPU |
| `run/card_checks.sh` | one CUDA card: `test-backend-ops` (fusions on and off) and `test-llama-archs` CPU against the card (fused and unfused graph, with and without rollback snapshots) |

## Commands

The build and the CPU check, on a build machine with Docker, from a work folder that holds this repository as `repo/`
and a llama.cpp clone with the fork as remote `fork` as `src/` (see `scripts/build_llamacpp_docker.sh`):

```bash
WORK=<work dir> LOCK=<lock file, if the machine is shared> JOBS=6 DOCKER_ARGS='--label <label>' \
  setsid nohup bash <work dir>/repo/docs/data/2026-11-04_gdn_fusions/run/build_and_cpu_check.sh \
  > <work dir>/logs/chain.log 2>&1 < /dev/null &
```

It printed the rig tarball `llama.cpp-20261103-766c92a22-pascal-p106-v3-b12076-gdnfuse.tar.gz`
(304,797,393 bytes, md5 `223798575be09cd73c238bb9f136ab2d`, no AVX, GLIBC_2.38, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`) and `test-llama-archs` (md5 `38fd5c4db66ac39329932722d1c966cf`).

Next, on one P106 with the rest of the rig idle (correctness, minutes; `test-llama-archs` copied next to the tarball's
`bin/`):

```bash
BIN=<build>/bin OUT=<dir> DEV=0 bash docs/data/2026-11-04_gdn_fusions/run/card_checks.sh
```

Then on the rig, nine cards, with [fn_line.sh](../2026-11-03_flash_next_baseline/run/fn_line.sh) from Q215's folder
(Swift Q2_0, the settings of that folder): the two identity prompts at T 0 for the ladder build, this build, and this
build with `ENVS='LLAMA_GDN_FUSE=0 GGML_CUDA_GDN_FUSE=0'` (`MODE=speed`; the three output hashes should be equal),
then the diagnostic line with `GGML_CUDA_PROFILE` for this build fused and unfused (`MODE=diag`), summarized by
`scripts/prof_summary.py`: launches and milliseconds per token without weights, against the 33.5 ms above. Speed is
then measured on the agent check, as the speed ladder's step for these fusions (Q551).

## Provenance and licence

Ports of the techniques of Strata's `gdn_step_norm_multi_kernel`, `gdn_conv_l2_kernel` and `gdn_step_norm_kernel`
(MIT); the code is new, in the fork (MIT). The checks use a model with random weights that `test-llama-archs` makes;
no data set is involved.
