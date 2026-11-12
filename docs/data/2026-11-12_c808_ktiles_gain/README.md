# C728's K tiles only where they gain (C808, Q1414, 2026-11-12)

What this is: a narrower rule for C728's K tiles (`GGML_CUDA_MMVQ_PASCAL_DMT_KT`), after
[R120b](../2026-11-12_r120b_kern_speed/README.md) timed them on a P106-090. Against C722's read-once kernel the tiles
lost at IQ4_XS 2,560 × 6,144 at 4 tokens and at Q4_K at 3 tokens, and gained only at Q4_K at 4 tokens. In the served
verify pass IQ4_XS `ssm_out` cost 0.360 ms more, so the switch as a whole was a loss and stayed out of the v4 candidate
line. With this change `_DMT_KT=1` takes only the case that gained, at two row blocks a tile round (`_DMT_KT_M=2`, now
the default), so the switch can join the line once a P106 confirms it.

The change is on the fork branch `pascal-p106-v4-b12246-c808`, merged into `pascal-p106-v4` as `a1347d9dd`. It is host code only: no
GPU function changes. This folder holds the design, the checks on a build machine's CPU, why `_DMT_KT_M=2` seemed to move
two products that have no tile, and the rig run that times it (R133). Registry row
[Q1414](../../../quants/experiments.md); the switch is in [docs/switches.md](../../switches.md).

## What R120b measured

µs per product on a P106-090, pass 1 / pass 2, at 2,560 × 6,144 (`ssm_out`, `attn_output`), from
[R120b's perf table](../2026-11-12_r120b_kern_speed/perf/wide.tsv). `mt` is C722's read-once kernel; the tiles are
`_DMT_KT=1`, at one row block a round (M = 1, the default then) and at two (M = 2):

| Product | Tokens | Warps a block, whole activations | `mt` | Tiles, M = 1 | Tiles, M = 2 | Class at M = 2 |
|---|---:|---:|---:|---:|---:|---|
| IQ4_XS | 4 | 4 | 112.60 / 113.33 | 132.96 / 133.04 | 128.48 / 128.06 | loss |
| Q4_K | 3 | 5 | 98.07 / 99.10 | 110.40 / 110.94 | 106.91 / 106.79 | loss |
| Q4_K | 4 | 3 | 142.29 / 141.75 | 136.07 / 135.90 | 129.45 / 129.39 | gain |

The tiles always run 6 warps a block. They gain only where C722's kernel, staging the whole activations, is down to 3
warps a block; from 4 or 5 warps the extra barriers and staging cost more than the warps bring. M = 2 is faster than
M = 1 in all three cases.

## The change

In `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu` (`dmt_kt_nw`, `dmt_kt_m`):

- **`_DMT_KT=1` takes Q4_K's products of 4 tokens only,** and only where C722's kernel with the whole activations holds
  at most 3 warps a block (`DMT_KT_FULL_MAX`) and the tiles hold more. At Flash-Next's widths that is K = 6,144:
  `ssm_out` and `attn_output`. Everything else keeps C722's kernel (or the k-column kernel) as with the switch off.
- **`_DMT_KT_M` defaults to 2** (two row blocks per warp and tile round, half the tile staging per row). `_DMT_KT_M=1`
  keeps one.
- **`_DMT_KT=2` is unchanged** (the tiles wherever they fit: IQ4_XS and Q4_K, 2–4 tokens, for the checks), but its M
  follows the new default.
- `_DMT_KT_NW` and `_DMT_KT_FAULT` are unchanged. The tile kernels themselves are C728's, and R120a found them exact on a
  P106 at M = 1 and M = 2 ([R120a](../2026-11-12_r120a_kern_exact/README.md)).

The rule uses the warps a block, not the row width, so that other models' Q4_K products get the tiles only from as
few warps as the gain was measured at.

## Why `_DMT_KT_M=2` moved two products with no tile

R120b's arm `kernm2` (`_DMT_KT_M=2`) also changed two products that get no tile:

| Product | Tokens | `kern` (M unset) | `kernm2` (M = 2) | Products timed per pass |
|---|---:|---:|---:|---|
| Q6_K 2,560 × 6,144 | 4 | 226.89 / 226.23 | 234.83 / 234.85 | 4,770 in both arms |
| Q4_K 2,560 × 6,144 | 5 | 172.45 / 172.86 | 177.19 / 176.69 | 6,360 against 5,724 |

**It is not a choice of kernel.** `_DMT_KT_M` is read only inside the tiles' launcher, after the tiles have taken the
product, and `kern_pick.cu` shows that `kernm2` changes no line of either tree but the tiles' own M (see "The checks").
Both products run the same kernel with the same launch in both arms.

**Where it could come from.** In R120b each arm was one `test-backend-ops` process, with the cases in the same order.
The Q6_K product is case 44 of 88 and the Q4_K product case 51; the tiles ran before them in cases 29, 36 and 40, at
M = 1 in `kern` and M = 2 in `kernm2`. The two processes also differ by one variable in the environment. Neither changes
what the two products launch, so the difference is in the card's time for the same work. The 5,724 against 6,360 is a
consequence: the tool repeats a graph of 636 products until a second has passed, and at 177 µs nine graphs pass it.

**The rig run tells which** (R133, `run/rig_c808.sh`):
- the perf arms `kern`, `kernm1` and `kernm2`: `kernm2` makes the same choices as `kern` (M = 2 is now the default)
  with one more variable in the environment, and `kernm1` has as many variables as `kernm2` with the tiles at M = 1;
- the `alone` step runs each of the two products, and the tile product, in a process of its own, so no tile runs before
  them.

A move that follows the earlier tiles shows in `perf` and not in `alone`. A move that follows the environment shows
`kernm2` against `kern` in both.

## The checks

On a build machine's CPU ([run/checks.sh](run/checks.sh)), 2026-11-12 18:01-19:28, every step in a container without
GPUs at a low CPU share; the merged tree's step 19:49-19:52. The base is `pascal-p106-v4` `d21820340`, whose ggml is the
`pascal-p106-v4-b12246-c803` build's (`bd632176f`; the two differ in `tests/test-backend-ops.cpp` only), so that checked
build stands for the base's GPU code and CPU logits. Each check's line is in [summary.tsv](summary.tsv).

| Check | Result |
|---|---|
| rig tarball | `llama.cpp-20261112-6c24e2025-pascal-p106-v4-b12246-c808.tar.gz`, 678,343,653 bytes, md5 `8e7b83ff793714741daf7662b30b0f6e`; `TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most; `llama-server` md5 `10728043a4e12d184485a633941adf06`, `test-backend-ops` md5 `7e184fe650416b34516bbe8a6e45e2b6` |
| every GPU function | `cuobjdump -sass` and `-ptx` of `llama-server` against the base build's, function by function: 8,540 of 8,540 the same in each listing, none new, none missing; `mmvq-pascal-dmt.cu`'s object against the base's: 146 functions (48 of them the tiles' kernels), SASS and PTX the same |
| CPU logits, every switch off | the base's static `test-llama-archs` writes 146 architectures' logits; the branch's are the same, 146 of 146, with no switch, with `GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1` and with `_DMT_KT_M=1` added |
| CPU, static | every target; ctest 70 of 70 (the 7 tests that need a download, the network or torch not run), the network test 1 of 1, `test-llama-archs` 151 of 151 cases with its 146 logits files written and compared, and the hand tests: `test-hop-cast` 54 checks, `test-route-once` 223 (once and 30 times), `test-bytes-switches` 535 and 536, `test-moe-ids` 13, `test-thread-safety`, `test-llama-context`, `llama-eval-callback`, 0 failed |
| CPU, shared libraries | the same, all passed |
| the kernel choices | `kern_pick.cu` under 48 arms, 1,344 products each, on both trees: the table below |
| `multitoken_bitid` | linked against the rig build's tree, modes `dmtkt`, `dmt` and `q6k`: md5 `ad6a2c8d38250ca60e0e6e5be22d6d71` (it runs on the rig) |
| merged tree | the `--no-ff` merge into `pascal-p106-v4` (`a1347d9dd` on `d7cdd032a`, the line's tip after C809's merge, which changed the one-warp rule in the same file; it adds exactly the branch's diff), checked before it was pushed: a static CPU build of every target; `test-llama-archs` 146 of 146 the base's logits with no switch and with `_DMT=1 _DMT_KT=1`; `kern_pick.cu` on its `mmvq-pascal-dmt.cu`, its copy of the one-warp rule (`PICK_Q6K1`) set to C809's, against this branch's lines: 48 arms, 285 lines differ, each a one-warp launch at 5-8 tokens that C809's rule leaves to the k-column kernel (15 in each of the 19 arms with `_RO8=1`), none at 2-4 tokens, 0 unexpected |

The change is host code: the device code is the base's, and the tile kernels are C728's, which R120a ran on a P106 with
the v4 kernel merge's build: `multitoken_bitid dmtkt` at `_DMT_KT=1`, `=2` and `=2` with `_DMT_KT_M=2`, 78 comparisons
each, 0 differ from each token's one-token product, and the replay arms at M = 1 and 2, 345 of 345 cases each.

### The kernel choices

[kern_pick.cu](../2026-11-11_v4_kernel_merge/run/kern_pick.cu) is the v4 kernel merge's host-side copy of the kernel
choice of `mmvq-pascal-dmt.cu`, compiled with `nvcc` against each tree's own file (`-DTREE_KERN`, no GPU). It prints, for
every product it covers (IQ4_XS, Q4_K and Q6_K, 2-8 tokens, rows of 1 to 64 superblocks, K = 256 to 16,384: 1,344 lines
an arm), the kernel and its launch. The arms are the v4 kernel merge's 37
([pick_arms.txt](../2026-11-11_v4_kernel_merge/pick_arms.txt)), R120b's `kern`, `kernm2` and `kernm1`, and the settings of
`_DMT_KT` and `_DMT_KT_M` ([pick/arms.txt](pick/arms.txt)). For each arm with `_DMT_KT=1` the base also ran without the
tiles' switches, which gives the kernel a product keeps when the tiles leave it and its warps a block.

| Arms | Lines | Lines that changed | What a changed line must be | Unexpected |
|---|---:|---:|---|---:|
| `_DMT_KT` unset, `0` or `3` (23) | 30,912 | 0 | nothing changes | 0 |
| `_DMT_KT=2` (6) | 8,064 | 816 | the tiles' M, 1 to 2, in the four arms that leave `_DMT_KT_M` unset (1,228 tile lines in each tree) | 0 |
| `_DMT_KT=1` (19) | 25,536 | 1,629 | a tile the base took goes to the kernel without the tiles, unless it is Q4_K at 4 tokens where that kernel holds 1 to 3 warps; a tile kept has the new M (1,689 tile lines in the base, 181 in this tree) | 0 |
| R120b's `kernm2` against `kern` | 1,344 a tree | base 102, this tree 0 | only a tile's M | 0 |

Per arm: [pick/counts.tsv](pick/counts.tsv); every changed line, base => this tree: `pick/changes.tsv.gz`.

Under `kern` (R120b's arm) the tiles now take 10 products, all Q4_K at 4 tokens: K = 6,144, 6,400, 6,656 and 6,912
(where C722's kernel holds 3 warps a block), 7,168, 7,424 and 7,680 (2 warps), 7,936, 8,448 and 8,704 (1 warp). At 8,192
one warp covers a row and the tiles do not fit, in either tree. In the base `kern` sent 102 products to the tiles, IQ4_XS
and Q4_K at 2-4 tokens ([pick/kern_tiles.tsv](pick/kern_tiles.tsv)). At Flash-Next's widths:
[pick/flashnext_kern.tsv](pick/flashnext_kern.tsv) (each product's line in the base and in this tree, side by side), and
the products R120b timed at 2,560 × 6,144 under each of its three arms ([pick/r120b_cases.tsv](pick/r120b_cases.tsv)):

| Product | Tokens | `kern`, base | `kern`, this tree | `kernm2`, base | `kernm2`, this tree |
|---|---:|---|---|---|---|
| IQ4_XS | 4 | tiles, M = 1, 6 warps | C722's, 4 warps | tiles, M = 2 | C722's, 4 warps |
| Q4_K | 3 | tiles, M = 1, 6 warps | C722's, 5 warps | tiles, M = 2 | C722's, 5 warps |
| Q4_K | 4 | tiles, M = 1, 6 warps | tiles, M = 2, 6 warps | tiles, M = 2 | tiles, M = 2 |
| Q4_K | 5 | C722's, 2 warps | C722's, 2 warps | C722's, 2 warps | C722's, 2 warps |
| Q6_K | 4 | C722's, 2 warps | C722's, 2 warps | C722's, 2 warps | C722's, 2 warps |

Under `kernm1` this tree's Q4_K at 4 tokens is the tiles at M = 1; every other line is `kern`'s.

## The rig run (R133)

[run/rig_c808.sh](run/rig_c808.sh) on a P106-090 rig, the rig idle, with this branch's tarball, Flash-Next and its MTP
draft:

1. **perf:** `test-backend-ops perf` at Flash-Next's IQ4_XS, Q4_K, Q5_K and Q6_K shapes, 1-8 tokens (R120b's cases),
   arms `off` (the preset's k-column switches), `mt` (+ C722's and C690's read-once kernels), `kern` (+ `_Q6K_MT=1`,
   `_DMT_KT=1`, `_RO8=1`: the tiles now at Q4_K's 4-token products only, M = 2), `kernm1` (`kern` at M = 1) and `kernm2`
   (`kern` with `_DMT_KT_M=2` set, the same choices), two passes, each pass by value. At K = 6,144 `kern` runs C722's kernel
   again for IQ4_XS at 4 tokens and Q4_K at 3, so it must match `mt` there (R120b: 112.60 / 113.33 and 98.07 / 99.10
   µs), and Q4_K at 4 tokens must gain over `mt` as R120b's M = 2 did (129.45 / 129.39 against 142.29 / 141.75 µs).
2. **alone:** Q6_K 2,560 × 6,144 at 4 tokens, Q4_K at 5 and Q4_K at 4, each in a process of its own, arms `kern`,
   `kernm1` and `kernm2`, two passes: which of the two causes above moved R120b's two products.
3. **served:** R49's profile of a served verify pass
   ([verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh), through the v4 kernel merge's
   [rig_kern.sh](../2026-11-11_v4_kernel_merge/run/rig_kern.sh)) on the Flash-Next preset with the MTP draft, both
   arms with `_DMT=1 _Q5K_MT=1`, `on` adding `GGML_CUDA_MMVQ_PASCAL_DMT_KT=1`, loads `off on on off`: Q4_K `ssm_out` and
   `attn_output` a pass must fall (est. −0.05 ms a pass) and IQ4_XS `ssm_out` must be `off`'s.
4. **bitid** (optional): `multitoken_bitid dmtkt` at `_DMT_KT=1`, at `_DMT_KT=2` (M = 2 and 1) and the fault arm, which
   must differ.

`_DMT_KT=1` is then a gain, negligible or a loss by the served pass, each load by value.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the build machine's checks, one line each (work folders shown as `$WORK`) |
| [pick/counts.tsv](pick/counts.tsv) | `kern_pick.cu`: per arm, lines, lines changed, unexpected lines, tile lines in each tree |
| `pick/changes.tsv.gz` | every line that changed, base => this tree, per arm |
| [pick/r120b_cases.tsv](pick/r120b_cases.tsv) | R120b's products at K = 6,144 under `kern`, `kernm2` and `kernm1`, both trees |
| [pick/kern_tiles.tsv](pick/kern_tiles.tsv) | every product `kern` sends to the tiles, both trees |
| [pick/flashnext_kern.tsv](pick/flashnext_kern.tsv) | Flash-Next's widths (K = 2,560 and 6,144) under `kern`: base and this tree side by side |
| [pick/merged_by_arm.tsv](pick/merged_by_arm.tsv) | `kern_pick.cu` on the merged tree against this branch, per arm: lines that differ, of them one-warp launches C809 declines, unexpected |
| [pick/arms.txt](pick/arms.txt) | the 48 arms (switches without the `GGML_CUDA_MMVQ_PASCAL_` prefix) |
| [run/checks.sh](run/checks.sh) | the build machine's checks |
| [run/rig_c808.sh](run/rig_c808.sh) | the rig run (R133) |

## Commands

On an x86-64 build machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image, no GPU, from this
repository's root (`BASE_TREE` the base's unpacked rig tarball, `BASE_OBJ` its build's `ggml-cuda.dir`, `BASE_CPU` the
`bin/` of a static CPU build of the base; `SEED_SRC` a llama.cpp clone with the fork as remote `fork`; `LOCK` a lock
file shared with other builds on the machine):

```bash
M=$WORK/c808 REPO=$PWD LOCK=$LOCK BASE_COMMIT=d21820340 BASE_TREE=$BASE_TREE BASE_OBJ=$BASE_OBJ BASE_CPU=$BASE_CPU \
    SEED_SRC=$SRC JOBS=8 THREADS=8 bash docs/data/2026-11-12_c808_ktiles_gain/run/checks.sh
# the merge, checked before it was pushed to the fork (the merge commit put into $M/main/rig/src as refs/c808/merge):
M=$WORK/c808 REPO=$PWD LOCK=$LOCK BASE_COMMIT=d21820340 BASE_CPU=$BASE_CPU STEPS=merged FINAL_REF=refs/c808/merge \
    bash docs/data/2026-11-12_c808_ktiles_gain/run/checks.sh
```

On the rig (the tarball unpacked; `rig_c808.sh`'s header names every variable):

```bash
BIN=<tarball>/bin OUT=<out> RIG_KERN=docs/data/2026-11-11_v4_kernel_merge/run/rig_kern.sh \
    MODEL=<Flash-Next GGUF> DRAFT=<MTP draft GGUF> PRESET=release/flash-next-v3/settings.env \
    RIG_SERVER=bench/agent_check/rig_server.sh REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
    VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
    setsid nohup bash docs/data/2026-11-12_c808_ktiles_gain/run/rig_c808.sh > console.txt 2>&1 < /dev/null &
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c808` at `6c24e2025` (one commit on `pascal-p106-v4` `d21820340`), merged into `pascal-p106-v4`
as `a1347d9dd`. The tests' inputs are generated (fixed seeds), not model weights. R120b's numbers are that row's
([its data](../2026-11-12_r120b_kern_speed/README.md)). Everything here is this repository's licence.
