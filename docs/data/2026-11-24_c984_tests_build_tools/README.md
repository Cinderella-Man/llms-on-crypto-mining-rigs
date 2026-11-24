# Miner Llama's check tools in every `TESTS=1` tarball (C984; 2026-11-24)

Result: registry [Q1717](../../../quants/experiments.md) (from Q1687/C957; new Q1744, Q1745).

Besides llama.cpp's own tests, Miner Llama has check tools that are run by hand on a model or on the cards: the
pipelined-decoding check `llama-pipe-recurrent-check`, `test-moe-ids` (each expert at most once per token),
`test-hop-cast` and `test-hop-mailbox` (the card-to-card hops), the three pool tests `test-kpool-dev`,
`test-kpool-stage` and `test-pool-mask-dev`, and `test-llama-archs`, which writes the random models the others load.
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) built them only when a caller listed them
in `EXTRA_TARGETS`, so most `TESTS=1` tarballs could not run them and a check had to build its base again.

- **The script now builds each of the eight with `TESTS=1`**, when the tree defines it as a target: it asks CMake for
  the configured tree's targets (`cmake --build build --target help`), so a tree without a tool still builds, and
  `BUILDINFO.txt` names the tools built and the ones the tree lacks. `CHECK_TOOLS=0` leaves them out.
- **What it costs: 0.54 GB.** Each tool is a static binary with every CUDA kernel, 96 to 107 MB unpacked. The tarball of
  Miner Llama v5's tip is 1,252,901,378 bytes with them against 708,062,047 without (+544,839,331 bytes, +77 %).
- **The other binaries are unchanged:** `llama-server`, `test-backend-ops` and the other nine have the same md5 as the
  script before the change gave for the same commit, with and without `CHECK_TOOLS=0`.
- [scripts/package_release.sh](../../../scripts/package_release.sh) builds with `CHECK_TOOLS=0` unless told otherwise:
  it packs every file in `bin/`, so a release's assets would otherwise grow by the same amount.

## The builds

Six builds on a build machine's CPU (CUDA 12.9.2 image, compute 6.1 code, no AVX, `-j 8`), one after another with one
compiler cache. A cache seeded from an earlier build of the same commit makes the four builds of the tip take about a
minute each. [tarballs.tsv](tarballs.tsv):

| Build | Tree | Script and settings | Exit code, seconds | Tarball bytes | Files in `bin/` | Check tools built |
|---|---|---|---|---|---|---|
| old | v5 tip `8a9629942` | before the change | 0, 53 | 708,062,047 | 11 | (not a feature yet) |
| new | v5 tip `8a9629942` | changed | 0, 80 | 1,252,901,378 | 19 | all 8 |
| off | v5 tip `8a9629942` | changed, `CHECK_TOOLS=0` | 0, 52 | 708,062,576 | 11 | none: "not built (TESTS=1, CHECK_TOOLS=0)" |
| dup | v5 tip `8a9629942` | changed, `EXTRA_TARGETS="test-llama-archs test-cuda-keepalive"` (what earlier briefs passed) | 0, 76 | 1,318,839,641 | 20 | all 8, and `test-cuda-keepalive` |
| v3 | `miner-llama-v3.0` (`1551f5fe1`) | changed | 0, 440 | 956,074,494 | 16 | 5; the tree has no pool tests |
| v2 | `miner-llama-v2.0` (`37e717ff6`) | changed | 0, 50 | 320,560,684 | 12 | `test-llama-archs` only; the tree has none of the other 7 |

- **Every tool in the new tarball:** `llama-pipe-recurrent-check`, `test-moe-ids`, `test-hop-cast`,
  `test-hop-mailbox`, `test-kpool-dev`, `test-kpool-stage`, `test-pool-mask-dev`, `test-llama-archs`
  ([binaries.tsv](binaries.tsv): every file of each tarball's `bin/` with its bytes and md5).
- **The same binaries:** [compare.tsv](compare.tsv) puts each binary of the old build beside the new, off and dup
  builds: all 11 the same md5 in all three (`llama-server` `1741fbce759489b052cf0d06ebec2e09`, `test-backend-ops`
  `f089e54a68e7a4051fc126abf5f41439`). These two are also the md5s of the same commit's earlier build with the old script
  and `EXTRA_TARGETS="test-llama-archs test-cuda-keepalive"`. The tarballs' own md5s differ: `BUILDINFO.txt` gains a line
  and the archive holds the build's time.
- **Older trees:** the v3.0 and v2.0 tags have the tools their trees define and nothing else, and `BUILDINFO.txt` lists
  the missing ones ([buildinfo.txt](buildinfo.txt)): for v3.0 "(not in this tree: test-kpool-dev test-kpool-stage
  test-pool-mask-dev)". The v4.0 tag already has all eight. A tool named both ways (dup) builds and is in the tarball once.
- **No new warning:** old, new, off and dup print the same 36 lines with "warning" in them.

The bit-for-bit result holds because the change adds build targets and copies but no compile flag: every object of the
shared binaries is compiled with the same command, so the cache serves the same objects. Two builds that compile from
scratch differ anyway, in three PTX files where nvcc names a shared variable per build
([C937](../2026-11-24_c937_v3_rebuild/README.md)).

## Is the size worth it

A `TESTS=1` tarball is a check build: it goes to a build machine or to a rig for a check, where 0.54 GB more costs
seconds (est. 6 s on a gigabit link) and the disk has room. Leaving the tools out cost a whole build each time one was
needed. So the tools are in by default, and `CHECK_TOOLS=0` gives the smaller tarball where only `test-backend-ops` or
the server is wanted. The release package keeps its earlier content; whether a release's tests archive should carry the
check tools is Q1744.

## Files

| File | What |
|---|---|
| [tarballs.tsv](tarballs.tsv) | one row per build: tree, settings, exit code, seconds, tarball name, bytes, md5, files in `bin/`, the "check tools" line of `BUILDINFO.txt` |
| [binaries.tsv](binaries.tsv) | every file in each tarball's `bin/`: bytes and md5 |
| [compare.tsv](compare.tsv) | the old build's binaries against the new, off and dup builds |
| [buildinfo.txt](buildinfo.txt) | each build's `BUILDINFO.txt` lines for its name, commit, web page and check tools |
| [run/builds.sh](run/builds.sh) | the six builds and the listings |
| [run/tables.py](run/tables.py) | the tables from `builds.sh`'s output |

## Commands

```bash
# the script before the change: public v5 at 098bb851 (md5 b24d4ad2e44a05dca650a604a3d29141)
git show 098bb851:scripts/build_llamacpp_docker.sh > old.sh
# WORK holds src/ (a llama.cpp clone with the fork as remote "fork" and the v2.0/v3.0 tags) and ccache/
WORK=<build folder> OLD=old.sh NEW=scripts/build_llamacpp_docker.sh TIP=8a9629942a4635ea0b2711b72e59556b96a63f86 \
  LOCK=<a lock file> JOBS=8 DOCKER_ARGS='--cpu-shares=256' bash docs/data/2026-11-24_c984_tests_build_tools/run/builds.sh
python3 -I docs/data/2026-11-24_c984_tests_build_tools/run/tables.py <build folder>/out docs/data/2026-11-24_c984_tests_build_tools
```

The tip builds pass `HF_UI_VERSION=b12483`, the web page the tip's earlier builds embedded; the v3.0 and v2.0 builds
take the script's own pick (b12091, b11819). The script that ran differs from the committed one in one header comment
(the size, filled in after the run; md5 `5b495dd2c5b1a56766f3c8173fe59923` ran).

Provenance: Miner Llama ([Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama)), branch
`pascal-p106-v5` at `8a9629942` and the tags `miner-llama-v3.0` and `miner-llama-v2.0`, built 2026-11-24 06:49–07:02.
Licence: the tables are this repository's; the binaries are not published here.
