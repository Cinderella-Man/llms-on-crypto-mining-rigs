# The switches in llama-server's log (C770, Q1355, 2026-11-11)

What this is: a change to llama-server's start-up log and its check on a build machine's CPU. Every engine change on
the v4 line sits behind an environment switch, and a rig run needs to show from the server's log which switches the
server ran with. The engine's own lines that announce a switch (for example C691's "one top-k over the output rows")
are logged at the library's INFO level, and llama-server prints the library's lines only from `-lv 4` on: its default
verbosity is 3. So R83 counted 0 such lines in every load and had to show its switch acting from the profiles
([data](../2026-11-11_r83_c691_p106/README.md)). Registry [Q1355](../../../quants/experiments.md).

## The change

**At load, llama-server logs one line at its default verbosity with every `LLAMA_` and `GGML_` variable in its
environment.** Fork branch `pascal-p106-v4-b12246-c770`, two commits on `pascal-p106-v4-b12246-c748` `5b329df16`
(`101406a84`, `76b3c410f`), in `common/` only:

```
I cmn  common_param: common_params_print_info: switches: none set (this build reads 815 LLAMA_ and GGML_ names)
I cmn  common_param: common_params_print_info: switches: 8 set: GGML_CUDA_KEEPALIVE=decode GGML_CUDA_MMVQ_PASCAL_SKIP="" LLAMA_ARG_NO_UI=1 LLAMA_GPU_KQ_MASK=1 LLAMA_SAMPLING_TOP_K_ROWS=1; not read by this build: LLAMA_C770_NOTE="two words" LLAMA_C770_TOKEN=(hidden) LLAMA_SAMPLNG_TOP_K_ROWS=1 (this build reads 815 LLAMA_ and GGML_ names)
```

- **Where:** `common_switches_info()` builds the line and `common_params_print_info` logs it with `COM_INF`, right after
  the "verbosity = 3" line. That is `common`'s INFO level, which the server prints at verbosity 3; the library's INFO
  maps to verbosity 4 (`common_log_get_verbosity`).
- **What it lists:** one scan of the process environment at that moment, so whatever is set is listed, a switch added
  tomorrow included. Names are sorted. A value that is empty or holds a space is in double quotes, control characters
  become `?`, and a name with `KEY`, `TOKEN`, `SECRET` or `PASSWORD` shows `(hidden)` (`LLAMA_API_KEY` is read by the
  server, for example).
- **"Not read by this build":** a variable that no source of the build names, which is a typo or a switch of another
  branch. The names come from the sources when the build is compiled: `common/switch-names.cmake` collects every quoted
  `"LLAMA_..."` or `"GGML_..."` literal in the C, C++ and CUDA files under `src`, `common`, `tools` and `ggml/src` into a
  generated `switch-names.cpp`, and CMake runs it again whenever one of those files changes or a new one appears. A new
  switch is recognised at the next build with no edit to any list. `common_arg`'s negative forms count as read
  (`LLAMA_ARG_NO_<X>` for a read `LLAMA_ARG_<X>`).
- **The count of names** is wide on purpose: it includes the variables of backends this build does not compile (Vulkan,
  CANN, SYCL and others), the server's `LLAMA_ARG_` names, and a few op names that some backends quote as strings
  (`"GGML_OP_ADD"`). None of those is set by accident, so the only cost is that such a name would not be flagged.
- **No compute change:** the line is text; nothing reads it, and no switch, kernel or graph changes. It touches no CUDA
  path, so no rig row is needed: the next rig run on a build that holds it shows the line in its server log.

## Results

Every build and test ran on a build machine's CPU in a container at `nice -n 19`, `--cpu-shares=256` and `-j 8`, one at
a time under a lock, with no GPU.

| Check | Verdict | What it shows |
|---|---|---|
| CPU build of the branch (static, native, every target) | OK | the generated list holds 815 names; 0 errors; the 3 warnings are in `sampling.cpp`, not in this change |
| names ([names.tsv](names.tsv), [docs.tsv](docs.tsv)) | OK | the build's list is the script's scan, and the branch reads the same 815 names as `5b329df16`; each of today's branches that adds a switch shows its names (C771 `LLAMA_MTP_EMBD_FIRST`, C726 five `GGML_CUDA_MMVQ_PASCAL_Q3K_MT*`), and the day merge's tree reads 847; every variable in [switches.md](../../switches.md) is read by some branch's scan (126 by this one, the other 21 by their own branches and the day merge); the 3 that no scan finds are op names (`GGML_OP_KQ_MASK`, ...), not variables |
| server ([lines.tsv](lines.tsv), [server.tsv](server.tsv)) | PASS | the base's llama-server (`5b329df16`) and the branch's on the CPU, at the default verbosity, with the random qwen4exp model and its MTP-only draft: with no switch, with five the branch reads, and with those and three it does not (a typo, a value with a space, a name with `TOKEN`). In each arm the branch's 64 greedy and 64 sampled tokens, their top-3 probabilities and its draft counts (180 and 183 drafts) equal the base's. The branch logs one line in each arm, each variable in the right list, the hidden value nowhere in the log; the base logs none, and the library's own `LLAMA_SAMPLING_TOP_K_ROWS` line is in neither log |
| rescan | OK | with a comment naming `LLAMA_C770_LATER_SWITCH` added to `src/llama-context.cpp`, rebuilding llama-server ran the scan once and compiled the list once, and the line moved the variable from "not read" to read (815 → 816 names); with the comment removed, back to 815 |
| every switch off against `5b329df16` (`test-llama-archs -b CPU -s 1`) | SAME | 146 of 146 logits files the same; all 151 tests pass |
| ctest | OK | 70 of 70 (without the 7 that need a download, the network or torch) |
| rig tarball | OK | `llama.cpp-20261111-76b3c410f-pascal-p106-v4-b12246-c770.tar.gz`, 656,483,819 B, md5 `d7b381e7790786fa51645232d495b372`; sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most; the line's text is in its `llama-server` |

The scan took 0.62 s on `-c748`'s tree and 0.63 s on this branch's on that machine ([names.tsv](names.tsv), the `scan_s` column).

[summary.tsv](summary.tsv) holds two server rows. The first run started the servers without the model's draft, which
this test model needs to load, and set `LLAMA_ARG_NO_WEBUI`, which no source reads (the server's variable is
`LLAMA_ARG_UI`, so its negative form is `LLAMA_ARG_NO_UI`): no server loaded the model, and the branch's line listed
`LLAMA_ARG_NO_WEBUI=1` under "not read by this build". The second run, with `-md` and `LLAMA_ARG_NO_UI=1`, is the one
in the tables. In that run the rescan ran before `off` and `ctest` (the order of the steps in the script then); it
puts the file back and rebuilds only llama-server, and both later steps run binaries built before it.

**What the line does not cover:** output that a switch writes while it runs. C707's `LLAMA_QSA_INPUT_STATS` counts and
C753's clock anchor of `GGML_CUDA_PROFILE_HOST` are library lines at the INFO level too; the line shows that those
switches are set, but their data still needs a level the server prints, or a file
(registry [Q1256 and Q1329](../../../quants/experiments.md)).

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail, as the run wrote them |
| [lines.tsv](lines.tsv) | each server arm's switch lines: the base's and the branch's count, the library's own `LLAMA_SAMPLING_TOP_K_ROWS` line, the branch's line |
| [server.tsv](server.tsv) | each server arm's requests: prompt and generated tokens, drafts, the tokens themselves |
| [names.tsv](names.tsv) | each fork branch scanned: its commit, the names it reads, the scan's seconds, the names it adds and lacks against `-c748` |
| [docs.tsv](docs.tsv) | each name in `docs/switches.md`: read by `-c748`, by this branch, by the day merge, and which branches read it |
| [rescan.tsv](rescan.tsv) | the line before, with and after the added comment |
| [run/checks.sh](run/checks.sh) | the whole check: the CPU build, the names, the server, every switch off, ctest, the rescan, the rig tarball |
| [run/server_check.sh](run/server_check.sh) | llama-server's three arms, base and branch, and their verdict |
| [run/line_check.py](run/line_check.py) | the check of each arm's switch line |
| [run/names.sh](run/names.sh) | the scan on each branch and the documented names against it |

## Commands

```bash
# on a build machine with Docker (CPU only), from this repository's root, with a llama.cpp clone that has the fork as
# remote "fork" (SRC), the base's static native CPU build (BASE_CPU: bin/ with test-llama-archs; BASE_SRV: bin/ with
# llama-server, both of pascal-p106-v4-b12246-c748 5b329df16) and the random qwen4exp model with one MTP block and its
# MTP-only draft (MODELS: written by test-llama-archs -a '^qwen4exp$' -s 1 -o <dir> --mtp-draft on
# pascal-p106-v4-b12246-c762 950844caf, md5 b67f1d9c... and 8ba549ab...)
WORK=$WORK REPO=$PWD SRC=$SRC BRANCH=pascal-p106-v4-b12246-c770 BASE_BRANCH=pascal-p106-v4-b12246-c748 \
  DAY=pascal-p106-v4-b12246-day BRANCHES="pascal-p106-v4 pascal-p106-v4-b12246-night pascal-p106-v4-b12246-c693 ..." \
  BASE_CPU=$BASE_CPU BASE_SRV=$BASE_SRV MODELS=$MODELS JOBS=8 LOCK=$WORK/lock \
  bash docs/data/2026-11-11_c770_switch_log/run/checks.sh
```

`BRANCHES` held `pascal-p106-v4`, `-night` and today's branches `-c693`, `-c701` to `-c705`, `-c711`, `-c719`, `-c720`,
`-c722`, `-c726`, `-c735`, `-c738`, `-c740`, `-c741`, `-c749`, `-c750`, `-c754`, `-c757`, `-c758`, `-c761`, `-c762`,
`-c766`, `-c767` and `-c771` (the list is in [names.tsv](names.tsv)).

To read a server's switches from its log on the rig: `grep -a 'common_params_print_info: switches:' server.log`.

Provenance: the model files are random weights written by the fork's own test; the tables are this check's output.
Licence: as this repository.
