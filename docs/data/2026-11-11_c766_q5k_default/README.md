# C690's Q5_K kernel at one row per lane group, without the prefetch, by default (C766, Q1347, 2026-11-11)

What this is: a change of defaults on the v4 line. `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1` sends Q5_K's plain products of 2–4
tokens (Qwen3.8-Flash-Next's output layer in a verify pass) to C690's kernel
([design](../2026-11-11_c690_head_four_rows/README.md)), which gives every token the one-token kernel's sums bit for
bit. C690 made two rows per lane group (R = 2) with a register prefetch its default. On a P106-090
[R75](../2026-11-11_r75_c690_p106/README.md) measured both choices as slower: at the head's four rows R = 2 took
5,486.46 and 5,326.59 µs (two passes), R = 1 with the prefetch 3,401.63 and 3,412.48, and R = 1 without it 3,340.78 and
3,346.73, the fastest at 2, 3 and 4 rows. R75's served step therefore ran `=2 _R=1`.

The change is on the fork branch `pascal-p106-v4-b12246-c766` (`9cb1b4a14`, one commit on
`pascal-p106-v4-b12246-c748` `5b329df16`). It touches only the host code of `ggml/src/ggml-cuda/mmvq-pascal-q5k.cu`
(two defaults) and the comments of it and its header.

## Result

**`GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1` now launches, at every token count, the kernel that `=2 _R=1` launched before, with
the same launch shape, and that kernel's code is the same instruction for instruction; so are all 8,952 kernels of the
build.** R75's numbers for `r1nopf` (`=2 _R=1`) are therefore `=1`'s on this branch: in the served verify pass the
output head takes 3.149, 3.155, 3.149 and 3.151 ms against 8.105, 8.125, 8.115 and 8.120 with the switch off, and the
warm loads' step 116.37 and 116.25 ms against 122.11 and 121.81 (second request). No new rig run is needed.

- **The device code is unchanged** ([sass_objects.tsv](sass_objects.tsv), [sass_binary.tsv](sass_binary.tsv)). CUDA
  12.9's `cuobjdump` of the branch's and the base's rig builds (sm_61): in each of the 193 CUDA objects every function's
  SASS is the same, 8,952 of 8,952, and so is each object's compute_61 PTX (193 of 193); `bin/llama-server` holds 8,847
  kernels, the same 8,847. The SASS dumps are byte for byte the same; the Q5_K object differs in its host
  code (`mmvq-pascal-q5k.cu.o` md5 `3bbbfc9e…` → `c87b6330…`).
- **The 12 instantiations** `mul_mat_vec_q5_K_pascal_mt<NT, R, PF>` (NT = 2, 3, 4 tokens; R = 1, 2; prefetch off, on)
  are the same code and registers in both builds ([sass_q5k.tsv](sass_q5k.tsv)). The one `=1` launches now is
  `<NT, 1, false>`: 126, 136 and 139 registers at 2, 3 and 4 tokens, no stack or local memory
  ([sass_counts.tsv](sass_counts.tsv), C690's [sass_check.sh](../2026-11-11_c690_head_four_rows/run/sass_check.sh),
  the same counts in both builds).
- **Which kernel each setting launches** ([sass_settings.tsv](sass_settings.tsv)), from the host code before and after:

  | Setting | Before (`5b329df16`) | Since `-c766` | Same kernel as before at |
  |---|---|---|---|
  | `=1` | R = 2, prefetch | R = 1, no prefetch | `=2 _R=1` |
  | `=2` | R = 2, no prefetch | R = 1, no prefetch | `=2 _R=1` |
  | `=1 _R=1` | R = 1, prefetch | R = 1, no prefetch | `=2 _R=1` |
  | `=2 _R=1` | R = 1, no prefetch | R = 1, no prefetch | itself |
  | `=1 _R=2` | R = 2, prefetch | R = 2, no prefetch | `=2` |
  | `=2 _R=2` | R = 2, no prefetch | R = 2, no prefetch | itself |
  | `=1 _PF=1` | R = 2, prefetch | R = 1, prefetch | `=1 _R=1` |
  | `=1 _R=1 _PF=1` | R = 1, prefetch | R = 1, prefetch | itself |
  | `=1 _R=2 _PF=1` | R = 2, prefetch | R = 2, prefetch | itself (C690's default) |

  Everything the host passes besides the kernel (warps per block, staging bytes, grid, shared memory) is computed from
  R, the token count and the row width alone, so a setting that picks the same R and prefetch launches the same way.
- **With the switch at 0 nothing changes:** `test-llama-archs -b CPU -s 1` gives the base's logits on 146 of 146 random
  models (151 tests pass), **ctest 70 of 70** (the static CPU build; the tests that need a download, the network or
  torch left out, as in C565) ([summary.tsv](summary.tsv)).
- **The rig tarball builds:** `llama.cpp-20261111-9cb1b4a14-pascal-p106-v4-b12246-c766.tar.gz`, 656,347,588 B, md5
  `4e3279d67065721434fe9b66977d5c7b` (`TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38,
  `test-llama-archs` in the tree's `bin/`).

## The switches now

- `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1`: the kernel at one row per lane group (R = 1, as the one-token kernel), without the
  register prefetch.
- `=2`: the same, with the prefetch off even under `_PF=1`. Settings written before keep their kernel with `=2`: `=2
  _R=1` and `=2 _R=2` run what they ran.
- `_R=2`: two rows per lane group, which share the activation loads (default 1; any other value gives 1).
- `_PF=1` (new, with `=1`): each warp loads its next weight segments into registers before it computes the current ones.
- `_NW`: warps per block, unchanged (default the most that fit: 6 at R = 1, 3 at R = 2).
- C690's first default is `=1 _R=2 _PF=1`. [docs/switches.md](../../switches.md) has the row.

**The replay sets.** [scripts/record_checks.sh](../../../scripts/record_checks.sh)'s `q5kmt` runs (also in `v4night`)
now name R and the prefetch in every run (`q5kmt-on` is `=1 _R=2 _PF=1`, `q5kmt-nopf` `=2 _R=2`, `q5kmt-r1`
`=1 _R=1 _PF=1`, the `nw` runs likewise), so each run launches the same kernel on builds before and after this change;
`q5kmt-defaults` (`=1` alone) runs the build's own default. A set recorded before keeps the old strings in its
`runs.tsv`: on this branch or later those runs launch R = 1 without the prefetch, all but the warp counts the same
kernel. `STEPS=runs` rewrites a recorded set's run lists from the script (the cases do not change).

## Files

| File | What it is |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict: the rig tarball, the SASS comparison, the CPU build, the logits identity, ctest |
| [sass_objects.tsv](sass_objects.tsv) | per CUDA object of the build tree: functions in each build, the same, different, in one only; the PTX the same or not |
| [sass_binary.tsv](sass_binary.tsv) | the same for the rig tarball's `bin/llama-server` |
| [sass_q5k.tsv](sass_q5k.tsv) | the 12 instantiations of `mul_mat_vec_q5_K_pascal_mt<NT, R, PF>`: a hash of each build's code, the same or not, registers |
| [sass_settings.tsv](sass_settings.tsv) | per setting and token count: the instantiation each build launches, whether it is the same code, and the base's settings with the branch's kernel |
| [sass_counts.tsv](sass_counts.tsv) | C690's `sass_check.sh` on the branch's Q5_K object (the base's is the same file): registers, spills, instruction counts of one super-block, the last two FFMA |
| [run/checks.sh](run/checks.sh) | the build machine's steps: the rig tarball, the SASS dumps and comparison, the CPU build, the logits identity, ctest |
| [run/compare_sass.py](run/compare_sass.py) | compares two builds' `cuobjdump` output function by function and writes the `sass_*.tsv` tables |

## Commands

On the build machine, CPU only (Docker with `--cpu-shares=256`, `nice -n 19`, `-j 8`, no GPU, `cuobjdump` from the
`nvidia/cuda:12.9.2-devel-ubuntu24.04` image), the work folder `WORK` holding in `rig/src` a clone of llama.cpp with
the fork as remote `fork`. The base's builds are those of [C748](../2026-11-11_c748_kvonly_kpool/README.md)
(`5b329df16`: its rig build tree and tarball, and its static CPU build):

```bash
WORK=$WORK REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c766 \
  BASE_CPU=<bin/ with 5b329df16's test-llama-archs> \
  BASE_OBJ=<5b329df16's rig build tree>/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir \
  BASE_BIN=<5b329df16's rig tarball tree>/bin \
  LOCK=<a flock file> JOBS=8 DOCKER_ARGS='--label pascal-quants=1' bash docs/data/2026-11-11_c766_q5k_default/run/checks.sh
```

It ran 2026-11-11 13:20:03–13:35:58, most of the SASS step's time spent waiting for the lock.

## Provenance and licence

Everything here comes from building the fork's code; no model or data file is read beyond `test-llama-archs`' own random
models. The speeds quoted are R75's, measured on a P106-090 ([R75](../2026-11-11_r75_c690_p106/README.md)). The tables
and scripts here are this repository's licence.
