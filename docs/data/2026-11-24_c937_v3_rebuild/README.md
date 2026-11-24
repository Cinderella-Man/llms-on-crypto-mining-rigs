# Rebuilding Miner Llama v3.0 and v2.0 from their tags: the web page (C937; 2026-11-24)

Result: registry [Q1677](../../../quants/experiments.md) (from Q1577; new Q1732, Q1733).

llama-server's web page (its built-in chat UI) is downloaded during the build from upstream's archive on Hugging Face
(bucket `ggml-org/llama-ui`), one archive per upstream release, `b<N>`. Since C874 the build script asks for the release
the tree sits on. Upstream has no archive for b12092, the release v3.0 sits on, nor for v2.0's b11820. For those the
tree takes `latest`, which since 2026-11-17 has a new layout without three files these trees require.

- **Before this change, v3.0 did not build from its tag:** `TESTS=1 scripts/build_llamacpp_docker.sh miner-llama-v3.0`
  stopped after 6 minutes at `llama-ui-assets` (run 1).
- [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) now walks down to the newest release at
  or below the tree's that has an archive: b12091 for v3.0, b11819 for v2.0. **Both build from their tags again** (runs 2
  and 4).
- Each release shipped the page that was upstream's newest on its build day, not its own release's: v2.0 b11887, v3.0
  b12128, v4.0 b12359. Passed that page and the release's 9-character commit id, a rebuild has **the released build's
  host code and data and all its cubins** (runs 3 and 5). The files still differ, because nvcc names a function-local
  shared variable differently in each build (3 PTX files of 187; Q1733).

The pages are named in [release/flint-v2/reproduce/engine.md](../../../release/flint-v2/reproduce/engine.md).

## Which releases have an archive

[releases.tsv](releases.tsv): every upstream release number from b11700 to b12500, whether it is an upstream tag
(`git ls-remote` of upstream), whether the bucket has a folder for it, and the answer to `dist.tar.gz.sha256` (what a
tree downloads first).

| | Count |
|---|---|
| upstream release tags b11700 to b12500 | 616 |
| with an archive | 519 |
| without: no folder | 96, among them b11820 (v2.0) and b12092 (v3.0) |
| without: HTTP 500 once | 1, b12383, which answered 200 on two later requests, so it has one |
| archives that are no upstream tag | 0 |

[folders.tsv](folders.tsv) lists all 2,198 top-level folders of the bucket (b9156 to b12667, and `latest`) with their
upload times, as of 2026-11-24 ~03:05.

`latest` (uploaded 2026-11-24 01:57 UTC) has 293 files and no `bundle*.js`, `bundle*.css` or `workbox*.js`. The archives
of b12090, b12091 and b12100 have 72 files each, these three among them. The v2.0 and v3.0 trees stop the build when
any of eight files is missing: `index.html`, `manifest.webmanifest`, `sw.js`, `build.json`, `version.json`,
`bundle[hash].js`, `bundle[hash].css`, `workbox[hash].js`.

## The page each release shipped

A release's `llama-server` holds its page's file names, and the bundle's hashed name identifies the archive. Each one
matches the folder that was the newest upload when the release was built:

| Release | Upstream release under it | Its archive | The script now takes | Release built | `bundle` in the released `llama-server` | Archive with that bundle |
|---|---|---|---|---|---|---|
| v2.0 (`miner-llama-v2.0`, `37e717ff6`) | b11820 | none | b11819 | 2026-10-25 (rc4's tarball) | `bundle.B2aMpQof.js` | b11887 (uploaded 2026-10-25 18:38 UTC; the next, b11888, at 23:32) |
| v3.0 (`miner-llama-v3.0`, `1551f5fe1`) | b12092 | none | b12091 | 2026-11-05 17:12 UTC | `bundle.DYKZKbOs.js` | b12128 (15:33 UTC; the next, b12129, at 18:52) |
| v4.0 (`miner-llama-v4.0`) | b12360 | yes | b12360 | 2026-11-13 12:55 UTC | `bundle.DG7eroDT.js` | b12359 (10:25 UTC, after b12360 at 10:06) |

Runs 3 and 5 confirm v3.0's and v2.0's: with those pages `llama-server`'s `.rodata` section equals the release's. The
released binaries also embed the commit id as git printed it then: 9 characters for v2.0 and v3.0 (`1551f5fe1`), 10 for
v4.0 (`eb312b7bce`). Git's default length grows with the clone, and today's clone prints 10.

v4.0's release assets were built from `eb312b7bc`; its tag points at `eb82916d3`, so a rebuild from the tag cannot carry
the released code whichever page it embeds. v4.0 was not rebuilt here.

## The builds

Each one on a build machine's CPU, in NVIDIA's `nvidia/cuda:12.9.2-devel-ubuntu24.04` image (the tag the releases name;
this pull was built by NVIDIA on 2026-05-18, before every release), with `TESTS=1` and 8 compile jobs. Nothing else was set unless the table says so.
The compiler cache was empty before run 1, so the times after it are with a warm cache; a cold build takes about 8
minutes (est.: run 1's 6 minutes plus run 2's 2). The release comparisons are with the released v3.0 build (`fork/pascal-p106-v3-b12076-ladder5` at the same
commit, built 2026-11-05, `llama-server` md5 `c735e40f184f3de61f16ffd5e2ab5ea5`) and with the binaries in v2.0's release
archive `llama-p106-v2.0-bin-ubuntu-cuda12.9-sm61.tar.gz` (sha256 `f213259e…baec1`, as its `SHA256SUMS`), which
`scripts/package_release.sh` gave RPATH `$ORIGIN`.

| Run | Tag | Script, settings | Page asked, taken | Commit id | Build | Against the release |
|---|---|---|---|---|---|---|
| 1 | v3.0 | before this change (public `v5` at `e98387a8`) | b12092 (404), then `latest` | 10 characters | **stopped** after 6 min 8 s: `llama-ui-assets` failed, "missing required asset(s): bundle[hash].js, bundle[hash].css, workbox[hash].js" | no tarball |
| 2 | v3.0 | this change | b12091, taken b12091, 70 assets embedded | 10 | rc 0, 1 min 48 s | md5: 0 of 11 the same ([md5_run2.tsv](md5_run2.tsv)) |
| 3 | v3.0 | this change, `HF_UI_VERSION=b12128`, `core.abbrev 9` | b12128, taken b12128 | 9 | rc 0, 54 s | md5: 1 of 11 (`llama-gguf`, CPU only). Each of the 10 CUDA binaries: every section the same (`.text`, `.rodata` and `.data` among them) but 7, which nvcc's names explain (below); 187 of 187 cubins, 184 of 187 PTX files, the other 3 differing only in nvcc's names ([compare_run3/](compare_run3/)) |
| 4 | v2.0 | this change | b11819, taken b11819, 70 assets embedded | 10 | rc 0, 5 min 4 s (with any wait for the build lock) | md5: none of the 10 in the archive the same. 147 of 147 cubins and 144 of 147 PTX files (the other 3 only nvcc's names) in each CUDA binary. The host code differs as the page and the commit id do: only `llama-cli`'s and `llama-server`'s `.rodata` change size ([compare_run4/](compare_run4/)) |
| 5 | v2.0 | this change, `HF_UI_VERSION=b11887`, `core.abbrev 9` | b11887, taken b11887 | 9 | rc 0, 45 s | md5: none of the 10 in the archive the same. Each CUDA binary: every section the same but run 3's 7 and `.dynstr`, which holds the release's RPATH; 147 of 147 cubins, 144 of 147 PTX files. `llama-gguf`: every section but `.dynstr` ([compare_run5/](compare_run5/)) |

Why no binary but `llama-gguf` matches byte for byte: in 3 of the 187 (v3.0) or 147 (v2.0) device code objects, the
flash-attention tile instances for head sizes 64, 128 and 256, nvcc gives the function-local `__shared__` variable
`KQ_max_new_shared` of `flash_attn_tile_iter` a name with a part that changes from build to build:
`_INTERNAL_afa861a5_35_fattn_tile_instance_dkq128_dv128_cu_9dce18f5_4736…` in the release,
`…_cu_69d4b2b6_4721…` in the rebuild. That part is the same in all three files of one build. Only the PTX carries the
name; the compute 6.1 cubins, which the P106 runs, are the same. The fatbin is compressed, so its size moves (by 272
bytes in v3.0) and with it the relocations (`.rela.dyn`, `.nvFatBinSegment`), the unwind tables (`.eh_frame`,
`.eh_frame_hdr`, `.gcc_except_table`) and the build id.

## The script's choice, checked without a build

[run/logic_test.sh](run/logic_test.sh): C874's checks plus the walk-down, with a stand-in `curl` that answers each
release from a table: **all pass** ([logic_test.txt](logic_test.txt)).
- A tree whose own release has an archive asks for it with one request, as before.
- A 404 walks down: b12092, then b12091. b12100, on a branch the tree does not contain, is never asked.
- A 500 or no answer is asked twice, then stops the walk at the tree's own release with a warning.
- No archive at or below gives the same warning.
- A given `HF_UI_VERSION` is used as is.
- A tree that does not download its page by release makes no request.

## Files

| File | What |
|---|---|
| `releases.tsv` | upstream releases b11700 to b12500: tag, folder, archive |
| `folders.tsv` | the bucket's top-level folders and their upload times |
| `md5_run2.tsv` … `md5_run5.tsv` | each binary's md5 against the released one; in runs 4 and 5 the reference is v2.0's release (`check_build.sh` labels the column "released v3.0" in all runs) and `test-backend-ops` is in its separate tests archive, so "missing" |
| `compare_run3/`, `compare_run4/`, `compare_run5/` | `sections.tsv` (every ELF section's size and whether its content is the same), `device_code.tsv` (cubins and PTX files the same), `ptx_diff.tsv` (each PTX file that differs: changed lines, and whether they differ only in nvcc's per-build names) |
| `logic_test.txt` | the logic test's output |
| `run/list_archives.sh` | the bucket listing, one request per release folder, the tag list |
| `run/check_build.sh` | one build of a release tag and the md5 comparison (runs 1 and 2 ran it before the `ABBREV` lines were added, which do nothing when `ABBREV` is unset) |
| `run/compare_build.sh` | sections and device code of two builds' binaries |
| `run/logic_test.sh` | the logic test |

## Commands

```bash
# which releases have an archive (b11700..b12500 by default), on any machine with curl, python3 and git
OUT=$PWD/archives bash docs/data/2026-11-24_c937_v3_rebuild/run/list_archives.sh

# the page a llama-server embeds, and the archive that holds that bundle
strings -n 8 <tree>/bin/llama-server | grep -E 'bundle\.[A-Za-z0-9_-]+\.(js|css)|workbox-[0-9a-f]+\.js' | sort -u
curl -s 'https://huggingface.co/api/buckets/ggml-org/llama-ui/tree/b12128/_app/immutable?recursive=false' | grep -o 'bundle\.[A-Za-z0-9_-]*\.js'

# the logic test, no build, no network
T=$PWD/logic SCRIPT=scripts/build_llamacpp_docker.sh bash docs/data/2026-11-24_c937_v3_rebuild/run/logic_test.sh

# the builds: SEED is a llama.cpp clone with upstream's tags and the fork's release tags, REF_BIN the released binaries,
# REPO a folder holding build_llamacpp_docker.sh and check_build.sh (run 1: the script before this change); detached,
# done when $W/DONE_run<N> exists
V3="REF=miner-llama-v3.0 HEAD_SHA=1551f5fe12424d7006748c1628b0df9550f20171"
V2="REF=miner-llama-v2.0 HEAD_SHA=37e717ff65f641a2b365eebeef33fc1e1292e707"
COMMON="W=$WORK/c937 SEED=<clone> REPO=<folder> LOCK=<build lock> JOBS=8"
env -u HF_UI_VERSION $COMMON $V3 REF_BIN=<released v3.0 bin/> RUN=1 bash check_build.sh
env -u HF_UI_VERSION $COMMON $V3 REF_BIN=<released v3.0 bin/> RUN=2 bash check_build.sh
env -u HF_UI_VERSION $COMMON $V3 REF_BIN=<released v3.0 bin/> RUN=3 UI_VERSION=b12128 ABBREV=9 bash check_build.sh
env -u HF_UI_VERSION $COMMON $V2 REF_BIN=<released v2.0 folder> RUN=4 bash check_build.sh
env -u HF_UI_VERSION $COMMON $V2 REF_BIN=<released v2.0 folder> RUN=5 UI_VERSION=b11887 ABBREV=9 bash check_build.sh

# sections and device code against the release, after a run (CUDA's cuobjdump)
A=<released bin/> B=$WORK/c937/rig/llama.cpp-<date>-<commit>/bin OUT=$PWD/compare_run3 bash compare_build.sh
```

## Provenance

The fork's commits and release assets are public on [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama).
The web page archives are upstream llama.cpp's (MIT), listed through the bucket's public API without a token. The check
builds' tarballs were checks only and were deleted after the comparison. Raw build logs are not published.
