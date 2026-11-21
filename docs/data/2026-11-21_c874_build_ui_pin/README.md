# The build script pins llama-server's web page by itself (C874; 2026-11-21)

Result: registry [Q1577](../../../quants/experiments.md) (follow-up Q1677).

llama-server's web page (its built-in chat UI) is not in recent llama.cpp sources. The build downloads it from
upstream's archive on Hugging Face (bucket `ggml-org/llama-ui`) by release, `b<N>`; without a version it asks for
`b<the tree's commit count>`, and when that has no archive it takes `latest`. A Miner Llama branch's commit count is
no upstream release, so fork builds took `latest`, which need not fit an older tree: from 2026-11-17 it lacked a file
the fork's tree requires and every fork build stopped at `llama-ui-assets` until `HF_UI_VERSION` was passed by hand.

[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) now does it itself:

- inside the container, after the checkout, it sets `HF_UI_VERSION` to the upstream release the tree sits on,
  `git describe --tags --abbrev=0 --match 'b[0-9]*'` (b12483 on `pascal-p106-v5`, b12360 on `miner-llama-v4.0`,
  b12092 on v3.0, b11820 on v2.0);
- an `HF_UI_VERSION` in the environment, or a `-e HF_UI_VERSION=...` in `DOCKER_ARGS`, wins (`DOCKER_ARGS` first);
- a clone without upstream's `b<N>` tags gets a warning and the unpinned behaviour;
- it prints the version it asked for, the one the build took (the tree's own `resolved:` line), a warning when they
  differ, and writes the pair into `BUILDINFO.txt` as `web page: asked b12483 (...), took b12483`.

## Checks

| Check | Result |
|---|---|
| [run/logic_test.sh](run/logic_test.sh): the script's choices without a build, a stand-in `docker` that prints its arguments | **18 of 18 pass** ([logic_test.txt](logic_test.txt)): nothing set passes no `HF_UI_VERSION`; the environment's reaches the container; `DOCKER_ARGS`' alone when both are set; the container script parses; in a repository without tags, or with only a `miner-llama-*` tag, nothing is pinned and the warning is printed; with `b12483` below `HEAD` it is pinned and exported, no warning; a given one is kept; the `took` parser reads `b12483`, `latest` and a failed download |
| [run/check_build.sh](run/check_build.sh): one rig build of `fork/pascal-p106-v5` (`3dfad876f`) with no `HF_UI_VERSION` anywhere, `TESTS=1` plus `test-llama-archs` and `test-cuda-keepalive`, on a build machine's CPU | **rc 0** in 59 s (a warm compiler cache). The script printed `web page: HF_UI_VERSION=b12483 (the upstream release this tree sits on)`; the tree's own line: `UI: HF download succeeded, stamp updated (ggml-org/llama-ui\|b12483, resolved: b12483)`, 72 assets embedded |
| Every binary against the build of the same commit that passed `HF_UI_VERSION=b12483` by hand ([C908](../2026-11-21_c908_v5_merged_build/README.md)) | **13 of 13 the same md5** ([md5.tsv](md5.tsv)), `llama-server` `5b330eccef892b7dd9f4faf1027319c3` in both: the build is reproducible. `BUILDINFO.txt` differs only in name, date, the new `web page:` line and `ldd`'s load addresses, which change on every run. The compiler warnings are the same 38 distinct lines |
| Which releases have an archive (one request for `dist.tar.gz.sha256` each) | [archives.tsv](archives.tsv): b12360 (v4.0), b12483 (v5) and `latest` have one; **b11820 (v2.0) and b12092 (v3.0) do not**, nor do b12150 and b12300 of the releases probed. For those the tree still takes `latest` and the script warns: Q1677 |

## Files

| File | What |
|---|---|
| `md5.tsv` | each binary of the check build against C908's build of the same commit |
| `archives.tsv` | upstream web page archives by release, 2026-11-21 ~05:40 |
| `logic_test.txt` | the logic test's output |
| `run/logic_test.sh` | the logic test (`T=<scratch dir> SCRIPT=<build_llamacpp_docker.sh> bash logic_test.sh`) |
| `run/check_build.sh` | the check build and the md5 comparison |

## Commands

```bash
# the logic test, on the build machine (no build, no network)
T=$PWD/logic SCRIPT=scripts/build_llamacpp_docker.sh bash docs/data/2026-11-21_c874_build_ui_pin/run/logic_test.sh

# the check build: SEED is a llama.cpp clone with upstream's tags and the commit, CCACHE a warm compiler cache,
# REF_BIN the bin/ folder of C908's build of 3dfad876f; detached, done when $W/DONE exists
env -u HF_UI_VERSION W=$WORK/c874 SEED=$WORK/c908/rig/src REPO=<folder with build_llamacpp_docker.sh and check_build.sh> \
  REF_BIN=$WORK/c908/rig/llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off/bin \
  CCACHE=$WORK/c908/rig/ccache LOCK=<build lock file> JOBS=8 bash check_build.sh

# which releases have an archive
for v in b11820 b12092 b12100 b12150 b12200 b12250 b12300 b12340 b12360 b12483 latest; do
  printf '%s ' $v; curl -s -o /dev/null -w '%{http_code}\n' -L \
    "https://huggingface.co/buckets/ggml-org/llama-ui/resolve/$v/dist.tar.gz.sha256?download=true"; done
```

## Provenance

The fork's commits are public on [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama); the web
page archives are upstream llama.cpp's. The check build's tarball was a check only and was deleted after the
comparison. Raw logs are not published.
