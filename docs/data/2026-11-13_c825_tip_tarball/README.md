# C825: the rig tarball of `pascal-p106-v4` at `65ac9a47d`

Registry: [Q1445](../../../quants/experiments.md). One P106 rig tarball of Miner Llama's v4 line at its tip,
`65ac9a47d` (2026-11-12 22:23), built on a build machine's CPU on 2026-11-13. The tip holds every change merged on
2026-11-12, each behind its switch and off by default, so a rig can run all of them, alone or together, from one build.

## What the tip holds

The 13 merges into `pascal-p106-v4` since the last tarball of the branch (C796's `d097d6d88`), oldest first:
C805 (`GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT`), C802 (`LLAMA_SAMPLING_TOP_K_ROWS`), C806 (the replay's `NOREC` verdict,
test code), C803 (`LLAMA_SCHED_PER_SIZE_RESERVE`), C811 (`GGML_CUDA_KEEPALIVE_EAGER`), C807 (`GGML_CUDA_MMVF_F32_MT`
rows), C812 (`LLAMA_PP_MULTI_SPLIT`), C809 (`_RO8`'s one-warp launches), C808 (`GGML_CUDA_MMVQ_PASCAL_DMT_KT`), C779
(`GGML_CUDA_GRAPHS_PATCH`), [C815](../2026-11-12_c815_adapt_while_drafting/README.md) (`LLAMA_SPEC_ADAPT_ONLINE`),
[C817](../2026-11-12_c817_keepalive_held/README.md) (inside `GGML_CUDA_KEEPALIVE_EAGER`) and
[C813](../2026-11-12_c813_hrow_half/README.md) (`LLAMA_MTP_HROW_LAZY`). Every switch is in
[docs/switches.md](../../switches.md).

The `-c813` rig tarball (`cb1ae3145`, built on C779's merge `b9a1c1fa7`) is the last rig tarball on the same base. The
tip adds C815's and C817's merges on top of that base: 7 files, all host or test code (`common/speculative.cpp`, `.h`,
`tools/server/server-context.cpp`, `tests/test-spec-adapt.cpp`, `ggml/src/ggml-cuda/keepalive-policy.h`,
`keepalive.cu` and `tests/test-cuda-keepalive.cpp`).

## The tarball

`llama.cpp-20261113-65ac9a47d-pascal-p106-v4.tar.gz`, 807,543,830 bytes, md5 `c6c9496354a3fb25e0857e15cce3b987`.
Built by [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) `fork/pascal-p106-v4` with
`TESTS=1` and `EXTRA_TARGETS="test-cuda-keepalive test-llama-archs"`: the extra tests of the `-c811`, `-c817` and
`-c813` rig tarballs, so one tarball serves those rig checks. The build is CUDA 12.9 for compute 6.1 only, with no AVX.

| `bin/` file | md5 |
|---|---|
| `llama-server` | `9cbea9859c17e3355c9eb51d14a4a863` |
| `test-backend-ops` | `450390d8469c4c549a6fedc3c3defee2` |
| `test-cuda-keepalive` | `0f763d45485f80d42d9e3d32c5694e42` |
| `test-llama-archs` | `c0024459a1cdf93b05eb7cafb53753af` |

Every `bin/` file's md5 is in [md5.txt](md5.txt), next to the `-c813` build's.

## Checks

| Check | Result |
|---|---|
| `BUILDINFO.txt` | commit `65ac9a47d6933025bc5d11db82158700d1352ddb` (2026-11-12 22:23:28 +0100), image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, 0 AVX instructions in `llama-bench`, highest glibc symbol `GLIBC_2.38` |
| the build | rc 0, 34 warning lines (the `-c813` build also had 34) |
| switch list (C770's `switch-names.cpp`) | 870 names, against 868 in the `-c813` build's; added `LLAMA_SPEC_ADAPT_ONLINE` and `LLAMA_SPEC_ADAPT_ONLINE_FAULT` (C815); none gone |
| `llama-server` SASS against `-c813`'s | **the same**: 8,540 functions in each; the masked dumps are byte for byte the same file (md5 `55ea1dcc6d520e8b4297630f3f41885c`), and the same as the dump the C813 checks made |
| `llama-server` PTX against `-c813`'s | **the same**: 8,540 functions in each; the masked dumps are the same file (md5 `5e8ba5d5c28cc949528b5451327cb837`), and the same as C813's dump |
| `bin/` against `-c813`'s | all 13 files differ in bytes. That is expected: C815's and C817's host code links into every program |

C817 changes `keepalive.cu`, a CUDA source file, but only in its host code: the build's GPU code is the `-c813` build's,
function for function. So the tip's GPU code is that of `b9a1c1fa7`, whose GPU functions C813's and C817's own checks
matched.

## Commands

On the build machine, under one build lock, in a container with a low CPU share (`--cpu-shares=256`, `JOBS=8`, the
driver under `nice -n 19`):

```bash
M=<work root> REPO=<this repository> LOCK=<flock file> BRANCH=pascal-p106-v4 \
  EXPECT=65ac9a47d6933025bc5d11db82158700d1352ddb SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> \
  SEED_CCACHE=<the -c813 build's ccache> REF_TREE=<the -c813 rig tree> REF_LABEL=c813 REF_SRC=<the -c813 build's src> \
  REF_DUMPS=<the -c813 checks' sass folder> CUOBJDUMP=/usr/local/cuda-12.9/bin/cuobjdump JOBS=8 \
  nice -n 19 bash docs/data/2026-11-13_c825_tip_tarball/run/build.sh
```

The build container's own processes do not inherit `nice` from the docker client, so they were reniced to 19 from the
host as soon as it started (`docker top <container> -eo pid`, then `renice -n 19 -p <pid>`). The compile ran after that.
With the `-c813` build's ccache, the build took 2 min 40 s. The two `cuobjdump` passes over each `llama-server` took
5 min 25 s.

On a rig (the ssh alias in `RIG`; `/opt/llama-builds` as on this project's rigs):

```bash
cat llama.cpp-20261113-65ac9a47d-pascal-p106-v4.tar.gz | ssh "$RIG" 'cat > /opt/llama-builds/tip.tar.gz &&
  [ "$(md5sum < /opt/llama-builds/tip.tar.gz | cut -d" " -f1)" = c6c9496354a3fb25e0857e15cce3b987 ] &&
  tar xzf /opt/llama-builds/tip.tar.gz -C /opt/llama-builds && rm /opt/llama-builds/tip.tar.gz && sync &&
  md5sum /opt/llama-builds/llama.cpp-20261113-65ac9a47d-pascal-p106-v4/bin/llama-server'
```

## Files

| File | What it is |
|---|---|
| [checks.tsv](checks.tsv) | every check's verdict and detail, in the order the script ran them (one build machine folder in angle brackets) |
| [md5.txt](md5.txt) | the tarball's size and md5, and each `bin/` file's md5 next to the `-c813` build's |
| [run/build.sh](run/build.sh) | the build machine's steps: the rig tarball, `BUILDINFO` and the md5s, the switch list, the GPU code against a reference build |

## Limits

- Only the CPU builds and checks this tarball. Its GPU code is `-c813`'s, function for function; the speed and the
  outputs on a P106 are the rig checks' (R136, R139, R140 and the `v4c3` agent check).
- The SASS and PTX comparison masks the numbers in anonymous namespaces and internal names, as the C813 and C779 checks
  did, since those numbers change between builds of the same code.

## Provenance and licence

Everything here was made for this repository: the build, the script and the tables. The engine is Miner Llama, the
fork of llama.cpp (MIT). The scripts are under this repository's licence.
