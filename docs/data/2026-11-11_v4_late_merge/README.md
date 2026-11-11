# The v4 late merge: four more engine branches in `pascal-p106-v4` (2026-11-11)

On 2026-11-11 the day's engine branches were merged into one branch, `pascal-p106-v4-b12246-day`
([the day merge](../2026-11-11_v4_day_merge/README.md), `18d0fa309`), and `pascal-p106-v4` was moved to it: every
switch of the v4 line goes into `pascal-p106-v4`, off by default, whether or not it has proven a gain yet. Four
branches finished after the day merge was built, each on `pascal-p106-v4-b12246-c748` (`5b329df16`) and each checked on
a CPU by its own session. This folder merges them into `pascal-p106-v4`, builds the result once for the rig (compute
6.1) and runs every merged branch's own published check on the merged tree, CPU only, plus the `test-llama-archs`
identity of the whole tree with every switch off against the day merge, the switches of all three merges on together,
and the GPU code of the merged build function by function against the day merge's. The registry row is Q1369 in
[quants/experiments.md](../../../quants/experiments.md); every switch is in [docs/switches.md](../../switches.md).

**Result:** the merged tree builds for the rig and every merged branch's own check passes on it. With every switch off
it computes what the day merge computes: `test-llama-archs`' logits are the same bit for bit on all 146 models, and the
CPU test set passes (ctest 70 of 70, with static and with shared libraries). With every switch of the three merges on together (28
switches; the CUDA ones change nothing on a CPU) the logits are still the day merge's on 146 of 146 models, and
llama-server's MTP-drafted replies with every exact host switch and the three late ones are the base's: the same tokens,
probabilities, draft counts and 406 verify steps (with the rows of the drafted tokens the server cuts left out, as C744's
cap leaves out the passes that drafted them). Two pairs no branch had run do too: C744's cap with C742's switch, and C771's
switch with C762's draft table on its layer's card, where C771 changes nothing, as its session estimated. Every GPU
function of the merged rig build is the day merge's (8,483 of 8,483 in the SASS and in the PTX): none of the four branches
changes GPU code, so the day merge's replay set and its rig checks hold for this build as they stand. Nothing ran on a
GPU; the speed of each switch is its rig row's.

## The branch

`pascal-p106-v4` at **`08634ce00`**: one merge commit per branch on `18d0fa309` (`git merge --no-ff`, no rebase), pushed
first as `pascal-p106-v4-b12246-late` for the build machine to fetch, and `pascal-p106-v4` moved to the same commit once
every check below had passed. The order is the order the branches finished in (C771 at 14:50, C770 at 15:45, C744 at
15:55, C742 at 16:15). It also puts C744 before C742: both change the MTP drafter's loop in `common/speculative.cpp`,
and C742 was written knowing C744 (its merge note is made against `-c744`). None of the four changes GPU code.

| Commit | Merge of | Cut from | Brings (off by default) | Conflicts and their resolution |
|---|---|---|---|---|
| `db89152a9` | `-c771` `fa7035917` ([C771](../2026-11-11_c771_draft_kpool_split/README.md)) | `-c748` | `LLAMA_MTP_EMBD_FIRST=1`: the MTP draft's token-row gather built first, so with the k-pool tables on the cards (`LLAMA_QSA_KPOOL_DEV`) each draft evaluation runs on the layer's card once, not twice | none |
| `c4117f7a1` | `-c770` `76b3c410f` ([C770](../2026-11-11_c770_switch_log/README.md)) | `-c748` | no switch: llama-server logs one line at load with every `LLAMA_` and `GGML_` variable set and, apart, those no source of the build reads | none |
| `511739d6c` | `-c744` `4b0e2730a` ([C744](../2026-11-11_c744_draft_cap/README.md)) | `-c748` | `LLAMA_MTP_DRAFT_CAP=1`: a greedy MTP draft stops at the slot's draft cap | `common/speculative.cpp`: the end of the MTP drafter's constructor, where `-c744` adds `cap_init()` and the day merge has C615's `LLAMA_DRAFT_PREFETCH` block: both kept, `cap_init()` first (C744's merge note). The merge's added and removed lines are exactly `-c744`'s against `-c748` |
| `08634ce00` | `-c742` `08cc9bf95` ([C742](../2026-11-11_c742_last_draft_row/README.md)) | `-c748` | `LLAMA_MTP_SKIP_LAST_ROW=1`: a step's last MTP drafting pass copies no hidden row to the host | none (`common/speculative.cpp`, `src/llama-context.{cpp,h}`, `src/llama-ext.h` merged without a conflict; each file's added and removed lines are exactly `-c742`'s) |

Where two switches meet, on together:
- C744 with C742: C742 marks a pass as the last when every drafting sequence reaches `--spec-draft-n-max` tokens after
  it, so a step that C744's cap stops earlier still copies its last pass's row, as without C742: in the checks below,
  with both on, 808 rows are copied and 398 left out in 406 drafts, against 812 and 406 with C742 alone (the 8 capped
  drafts keep their last row), with the same replies. Marking by C744's per-sequence stop instead would save those rows
  too (Q1377, planned).
- C744 with C738 (`LLAMA_VERIFY_PREPARE`): C738 prepares the verify pass after the drafter's last pass at
  `--spec-draft-n-max`, and its server gate refuses a cap below the draft, so a capped step prepares nothing either way
  (C744's merge note); both are on in the combination's `late` arms.
- C771 with C762 (`LLAMA_DRAFT_EMBD_DEV=layer`): C762 puts the draft's token table on its layer's card, so the gather
  and the k-pool update are on one card in either order: every draft evaluation is one split on that card with C771's
  switch and without it (C771 left this pair unrun; `c771x762` below).
- C770 with every switch: its line names every switch a server's environment sets; with this build each of the 16 host
  switches of the combination's `late` arms is under "set", none under "not read by this build" (`combo` below).

## The checks

`run/checks.sh` ran every step on a build machine's CPU, in Docker containers at `--cpu-shares=256` and `nice 19`, one
build or test at a time, no GPU, on `08634ce00` (2026-11-11 17:00–18:34; the final tarball 18:34–18:36). The base is the day merge, `18d0fa309`: its
static CPU build made here, its rig tarball the day merge's. The builds the branches' scripts share were made once in
`main` and hardlinked into each check's folder; each branch's script then ran its own steps there, with the merged
branch as its branch and the base's build as its base. Every verdict line is in [summary.tsv](summary.tsv) (its first
column is the row). The first start stopped before `main` on a fault of the driver, not of the code: an apostrophe inside
a `${VAR:?message}` made bash read the base step's tarball path on to the main step's message; its line is kept there as
`main-try1`, and the base's CPU build of that start served the second.

| Branch | Its check, run on the merged tree | Result |
|---|---|---|
| base | C744's `cpu`: a static native CPU build of `18d0fa309`, every target | built, 0 error lines (the base of every comparison below) |
| builds | C771's `cpu` and the rig build: a static native CPU build of every target, a `GGML_RPC=ON` build of `llama-server` and `ggml-rpc-server`, the rig tarball (CUDA 12.9, sm_61, no AVX) with `test-llama-archs` and `test-backend-ops` | all three built, 0 error lines (the base's 3 warning lines); the tarball has 0 AVX instructions and needs GLIBC 2.38 at most |
| every switch off | C771's `off`: `test-llama-archs -b CPU -s 1` on every architecture against `18d0fa309`'s logits | 146 of 146 models the same bit for bit (151 of 151 tests pass in both builds); the same with `LLAMA_MTP_EMBD_FIRST=1`, and on qwen4exp with it and the k-pool tables on the devices, staged and checked; C744's and C742's `off` the same unset and with their switch |
| every switch off | ctest in the static CPU build | 70 of 70 (the tests that need a download, the network or torch left out, as before) |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md), static and shared libraries | both built (0 error lines); ctest 70 of 70, `test-arg-parser` with network, `test-llama-archs` written and compared (146 files, none differ), the nine hand tests, in each |
| every switch on | `combo`: `test-llama-archs -b CPU -s 1` with the 28 switches of the three merges on together (the day merge's 25 with C767's, and C771's, C744's and C742's) | 146 of 146 models the same as `18d0fa309` |
| the stacked run on llama-server | `combo`: [run/server_combo.sh](run/server_combo.sh), a random qwen4exp model with an MTP block and a mapped n-gram table, the MTP draft at 3 tokens, four requests (two greedy, two sampled), `18d0fa309`'s llama-server as the reference | R90's host switches, every exact host switch of the night's and the day's merges with `--ctx-checkpoints-incall` (with and without the backend sampler), and those with the three late switches: each the base's tokens, top probabilities, draft counts and 406 verify steps (the late arms with the cut rows left out). In the late arms C744's cap left out 12 drafting passes in 8 drafts and C742 left out 398 hidden-row copies in 406 drafts; C770's line names each arm's switches (1, 4, 14 and 17 with `LLAMA_SPEC_DUMP`), none under "not read by this build"; the base logs no line |
| C771 | its `server` (CPU arms and six RPC pairs on two devices laid out as cards 4 and 8, with the scheduler's trace), `kvonly` (C748's `--mtp-kv-only` with the switch, 7 switch sets × 3 seeds) | every pair the same tokens, probabilities, draft counts and 406 verify steps; with the tables on the devices all 1,661 draft evaluations go from 3 to 2 splits and from 2 to 1 on the layer's device, the same nodes per device, the target's 429 evaluations the same splits; 21 of 21 runs pass |
| C771 with C762 | `c771x762`: C771's `server` with `LLAMA_DRAFT_EMBD_DEV=layer` in every arm | every pair the same replies and 406 verify steps; every one of 1,661 draft evaluations is one split, on the layer's device, with C771's switch and without it |
| C770 | its `names`, `server`, `rescan` | the merged tree's list (850 names) = its sources' scan; it reads the day merge's 847 and the three late switches, and every name each late branch adds; llama-server: no line in the base's log, the line exact with none, five and eight variables set (the three unread ones listed apart), the same tokens and probabilities; a comment naming a new variable: 850 → 851 → 850 names, one scan and one compile of the list per rebuild. C770's own `names` verdict line reads FAIL by its design (it wants the branch's names to be its base's); the line `names on the merged tree` checks what applies here |
| C770 | `line`: its build-line check on the merged rig tree | the line's text and `LLAMA_MTP_SKIP_LAST_ROW` (a late switch, from the list) in `llama-server` and every binary built with `common` |
| C744 | its `server`, `off` | every arm its reference's replies and verify steps (the cut rows left out where capped); the cap leaves out 12 passes in 8 drafts (greedy, also with the backend sampler), 6 of 11 with a probabilistic draft and 4 of 12 with `n_min` 2 (the refused drafts keep theirs); two slots: both replies alone's, 6 passes left out |
| C744 with C742 | `c744x742`: C744's `server` with `LLAMA_MTP_SKIP_LAST_ROW=1` in every arm | the same verdicts as C744's alone |
| C742 | its `server`, `off` | every arm its reference's replies and verify steps; hidden rows copied 3.00 → 2.00 a draft (also with the backend sampler and a probabilistic draft), 1.73 with a `p_min` cut; two slots 378 → 252 |
| GPU code | `sass`: the day merge's [sass_attrib.py](../2026-11-11_v4_day_merge/run/sass_attrib.py) on `cuobjdump -sass` and `-ptx` of `llama-server` from the merged rig build and the day merge's | 8,483 of 8,483 functions are the day merge's, in the SASS and in the PTX |

Left out, as the day merge did: each script's own CPU build, ctest and rig build (main's); C770's `cpu`, `off`, `ctest`
and `build` (main's), C744's and C742's `cpu`, `ctest` and `build` (main's). No replay set and no bit-identity build:
the GPU code is the day merge's, function for function, so its replay set `v4day`, its fault replays and its
`multitoken_bitid` stand for this build.

## The rig tarball

`llama.cpp-20261111-08634ce00-pascal-p106-v4.tar.gz`, 671,310,531 bytes, md5 `2435f1072847805a3082c4c5bfe65066`
(`TESTS=1`, CUDA 12.9 image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61, 0 AVX instructions, highest glibc symbol
GLIBC_2.38, `test-llama-archs` and `test-backend-ops` in `bin/`), built from `pascal-p106-v4` once it was moved to the
checked commit (`final`). Its 12 files in `bin/` are the same bytes as those of the build the checks ran,
`llama.cpp-20261111-08634ce00-pascal-p106-v4-b12246-late.tar.gz` (671,310,418 bytes, md5
`a15c26976ac8dc27dd66e5084135a476`, built from the side branch: the two differ only in the folder's name). Every switch
is off by default, so with nothing set it runs the day merge's code; the stacked run (R90) sets the same switches on it
as on the day merge's tarball, and the late switches join with `LLAMA_MTP_EMBD_FIRST=1`, `LLAMA_MTP_DRAFT_CAP=1` and
`LLAMA_MTP_SKIP_LAST_ROW=1`.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the driver: every step above, each branch's script run on the merged tree |
| [run/server_combo.sh](run/server_combo.sh) | llama-server with the switches of the three merges on together against the base |
| [summary.tsv](summary.tsv) | every verdict line (row, check, verdict, detail) |
| [server_combo.tsv](server_combo.tsv) | the combination's server arms: per request the prompt and generated tokens and the draft counts |
| [server_lines.tsv](server_lines.tsv) | per server arm of the combination: C770's line, C744's and C742's totals |
| [sass_attrib.txt](sass_attrib.txt) | `sass_attrib.py attrib`'s output for the SASS and the PTX |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`,
the day merge's rig tarball at `BASE_TGZ` and C762's two `qwen4exp-moe-mtp` model files in `MODELS`:

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> BASE_TGZ=<the day merge's rig tarball> SEED_SRC=<clone> MODELS=<dir> \
bash docs/data/2026-11-11_v4_late_merge/run/checks.sh
# once every check has passed and pascal-p106-v4 is moved to the checked commit:
M=<work folder> REPO=$REPO LOCK=<flock file> SEED_SRC=<clone> STEPS=final bash docs/data/2026-11-11_v4_late_merge/run/checks.sh
```

## What it decides

- Every switch of the v4 line so far is in one branch, `pascal-p106-v4`, off by default: the night's 17 branches, the
  day's 17 and these four. R90 (the gains together) and every open rig row can take this tarball with only its own
  switch set.

Provenance and licence: the models are `test-llama-archs`' random-weight models; everything here is this repository's
MIT licence.
