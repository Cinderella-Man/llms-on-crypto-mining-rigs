# C372 / Q931: native kernel checks on a mining rig without the rig's CPU (test-backend-ops record and replay, 2026-11-04)

Every new GPU kernel of the fork is checked with llama.cpp's `test-backend-ops`, natively on a P106-090: the quant
machine's newer cards can only run the P106 code through the build's compute_61 PTX, which catches logic errors but
not everything the real sm_61 build does. `test-backend-ops` builds each case's inputs, quantizes them and computes
the reference on the CPU, then runs the GPU and compares. On the rig's 2-core Celeron the CPU side is nearly all the
time: 1,961 k-column cases took 58 minutes, and each half of the `-ladder3` checks 3–5 hours, mostly in the IQ types'
quantization and the reference products.

Record and replay moves that CPU work to a build machine. The build machine records each case once, its tensors and
the CPU backend's results. The rig restores them, runs only the GPU side and compares with the same thresholds and
the same output lines. Fork branch `pascal-p106-v3-b12076-replay` (one commit on `pascal-p106-v3-b12076-kshape`
`d27845b67`), `tests/test-backend-ops.cpp` and a new `tests/test-backend-ops-replay.h`; nothing in ggml changes.
Scripts: [scripts/record_checks.sh](../../../scripts/record_checks.sh) (build machine) and
[scripts/replay_checks.sh](../../../scripts/replay_checks.sh) (rig).

**Status:** built (the rig tarball and an sm_86 `test-backend-ops`) and exact on the build machine: live and replayed
verdicts and every node's error agree on the same inputs, the fault codes fail the same cases, and the rig build's
references equal the record's bit for bit. Four suites are recorded (3–22 GB each). The first runs on the rigs
themselves are their own tasks (C440 on rig 1, C441 on rig 2).

## How it works

`test-backend-ops test` gains two options (and environment variables for the replay, so that the scripts that call
it, each branch's `gpu_checks.sh`, replay unchanged):

- **`--record DIR`** writes, per case, `cases/<id>.in` (the bytes of every tensor the case allocates, as its
  `initialize_tensors` left them: the inputs, the sentinels around them, the outputs' initial contents),
  `cases/<id>.ref` (the reference CPU backend's result of every node the live check compares, in graph order, after
  the case's own `host_reference`; for nodes without an op, the sentinels and parameters, their bytes as well) and
  `cases/<id>.meta` (the case string, the test class, the threshold, the size and md5 of both blobs, each tensor's
  name, type, shape and size, each compared node). With `-b CPU` only the reference runs; with a GPU backend the live
  check runs too and its verdict and each node's error go into the meta. A case already in the folder is kept, so
  several filters share one folder and a stopped record resumes.
- **`--replay DIR`** (or `GGML_TEST_REPLAY=DIR`) restores the tensors instead of drawing them, computes on the tested
  backend alone, node by node as the live check does (`ggml_graph_view` of one node, or the whole graph and its test
  nodes for fusion cases), and runs each node through the live check's own comparison: the sentinels byte for byte,
  NaN and infinity checks, the case's `err()` against its `max_err()`. Support is checked on the card as in a live
  run, so unsupported cases print `not supported` as before. The case lines, the `N/N tests passed` summary, the exit
  status and the FAIL lines are those of a live run, plus one `replay:` line with the counts.
- **Record ids:** `<id>` is the md5 of the case's class, its parameter string and how many earlier cases in the full
  list have both. A record therefore stays valid when a branch adds or removes other cases. Before using a record, the
  replay checks the case string, every tensor's type, shape and size, the compared nodes and both md5s; a mismatch
  fails that case (`record mismatch: …`).
- **The reference is the rig's own:** the rig build (`scripts/build_llamacpp_docker.sh`) compiles its CPU code
  without AVX, FMA or F16C (SSE4.2 at most), and the reference backend runs llama.cpp's reference paths. Recorded with
  the rig tarball's own `test-backend-ops`, the stored references are what the Celeron itself computes. `--cpu-check N`
  shows it on the rig: about one replayed case in N (inputs under 32 MB) also runs the live check against the host's
  CPU, compares its reference with the stored one value for value, and fails the run if the live and replayed
  verdicts differ.
- **The inputs are the record's draw.** `test-backend-ops` draws inputs from `std::random_device`, so a live run
  checks one random draw per case; a replay always checks the recorded draw, and all switch settings of a suite
  (e.g. nine k-column launch settings) see the same inputs.
- **Caps:** `--sample N` records about one case in N (chosen by a hash of the case string, the same cases every run),
  `--record-max-case-mb` and `--record-budget-gb` leave cases out, `--record-dry` prints the sizes and writes nothing.
  Replaying a filter that selects cases the record lacks prints `not recorded` lines and fails the run, unless
  `--replay-allow-missing` (`GGML_TEST_REPLAY_ALLOW_MISSING=1`) is given.

## The suites and their sizes

[scripts/record_checks.sh](../../../scripts/record_checks.sh) defines four suites. Each is a few filters (the union of
the cases its runs select) and a `runs.tsv` with each run's switches, filters and expected outcome, taken from each
branch's own `gpu_checks.sh`. All switch settings of a suite replay one record, because the reference does not depend
on the card's switches. Sizes from `--record-dry` (`sizes.tsv`; GB = 10⁹ bytes):

| Suite | What the runs check | Cases | Files | Largest case |
|---|---|---|---|---|
| `kshape` | `-kshape`'s 18 runs: the k-column products of 16 types at 1–8 columns in 9 launch settings, the large-activation cases (FP32 sums), the Q2_0 experts in 6 `MTOK` settings, the grid types' and IQ1_M's experts | 1,961 + 138 + 354 + 994 (118 of the 138 also in the first) | 22.3 GB | 473 MB (Q2_0, 512 experts) |
| `ladder3-rest` | the parts of `-ladder3`'s native checks left on the rig: the grid types and IQ1_M in the k-column and multi-token kernels in 6 settings (every shape: the native runs left out the large ones for the Celeron), the one-token grid-type kernels' last runs, the sparse decode attention with its three faults, 45 memcheck cases | 1,855 + 61 | 13.2 GB | 134 MB |
| `flashnext-moe` | the experts at Flash-Next's shapes (512 experts, 10 used: MUL_MAT_ID, the fused gate/up, the shared expert), every type, `MTOK` on and off | 109 | 3.0 GB | 473 MB |
| `fap` | every FLASH_ATTN_EXT case with FAP on, its three faults, FAP's own cases with it off | 5,654 | 21.3 GB | 329 MB |

Recording took 57 minutes for `kshape` on the build machine's 12-core CPU, shared with other jobs at a low priority
(`--cpu-shares=256`, 4 workers): 7 minutes for the k-column filter, 3 for the Q2_0 experts, 46 for the grid types'
experts (their quantization is most of the CPU work, as on the rig). `ladder3-rest` took 8 minutes (1,734 of its 1,916 cases hardlinked from `kshape`), `flashnext-moe` 30
seconds, `fap` 15 minutes.

A fifth suite, `n2p`, holds the native checks of the N2P types on the v3.0 release build: 3,655 cases, 37.4 GB,
recorded in 35 minutes with 8 workers. It also writes `runs-ptx.tsv`, the run list for a newer card through the
compute 6.1 PTX (`RUNS_FILE` of `replay_checks.sh`), and a run list may carry a seventh field naming cases that are
known to fail, which `replay_checks.sh` counts apart ([n2p_flash](../2026-11-04_n2p_flash/README.md#the-check-set-for-a-p106)).

The inputs are almost all of it (the quantized weights; the references are 0.5–11 %), and they are random data that do
not compress, so a set is stored as a plain tar. Every set fits a rig: rig 1 keeps its models on a 4.5 TB USB disk, rig
2 on a 57 GB and a 115 GB stick. No cap or shape filter was needed, though both exist (`MAX_CASE_MB`, `BUDGET_GB`,
`SAMPLE`, `EXTRA_P`). A replay run reads its filter's files once per run: the nine k-column runs read 4.0 GB each, a
Q2_0 expert run 7.0 GB. The rig's USB disk reads ~120 MB/s and its PCIe 1.0 x1 links carry ~190 MB/s, so a run takes
about a minute rather than the Celeron's tens of minutes to hours.

## Exactness

A replayed pass has to mean what a live pass means. Four tests, all on the build machine, none on a rig:

**On a CPU alone** (`run/cpu_tests.sh`, `cpu_tests.txt`; `-b CPU` as the tested backend, the reference CPU backend as
its reference; 49 MUL_MAT cases of Q4_0, Q8_0 and IQ2_XXS): 7 of 7 pass.

| Test | Result |
|---|---|
| t1 record, then replay | 49/49 OK, as the live run of the same filter |
| t2 the references re-recorded from the replayed inputs | 0 of 49 `.ref` blobs differ: the record-only path computes the live check's reference bit for bit |
| t3, t4 one byte of a case's `.in` or `.ref` flipped | that case fails (`record mismatch: … md5`), the other 48 pass, exit 1 |
| t5 a filter wider than the record | `not recorded` lines, a FAIL summary line, exit 1; with `--replay-allow-missing` exit 0 |
| t6 `--cpu-check 1` | every case's live reference identical to the stored one, verdicts equal |
| t7 a reference-only record | no live line in its metas; the replay log has one line per case |

**The rig binary's references** (`run/rigref_check.sh`, `rigref.txt`): the sets were recorded with a CPU-only build of
the branch, compiled with the rig build's CPU flags. About one case in 20 was replayed into the rig tarball's own
`test-backend-ops` and recorded again: 154 of 154 `kshape` and 100 of 100 `ladder3-rest` `.ref` blobs are identical
bit for bit. The stored references are therefore what the rig build computes, which `--cpu-check` shows on the rig
itself.

**Live against replayed, on the same inputs** (`run/gpu_exact.sh`, `exact_sm86.txt`, `exact_ptx.txt`; one RTX 3090,
the fork's P106 kernel paths forced on with a P106's launch shapes, as each branch's `gpu_checks.sh MODE=ptx` runs
them). A live run on the card also records (`--record` with a GPU backend writes its verdict and each node's error),
then the replay of that record runs under the same switches.

| Filter (switches) | Cases | sm_86 build: same verdict / same errors | Rig build through the PTX: same verdict / same errors | Failing cases, live = replayed |
|---|---|---|---|---|
| k-column products, `KCOL=2` | 1,961 | 1,961 / 1,961 | 1,961 / 1,961 | 0 = 0 |
| Q2_0 experts, defaults | 340 | 340 / 340 | 340 / 340 | 0 = 0 |
| sparse attention, `SPARSE=2 NQ=128` | 48 | 48 / 48 | 48 / 48 | 0 = 0 |
| sparse attention, fault 2 (negative control) | 48 | 48 / 48 | 48 / 48 | 29 = 29 |
| sparse attention, fault 3 (negative control) | 48 | 48 / 48 | 48 / 48 | 29 = 29 |
| FAP's cases, fault 2 (negative control) | 22 | 22 / 22 | 22 / 22 | 6 = 6 |
| FAP's cases, fault 3 (negative control) | 22 | 22 / 22 | 22 / 22 | 18 = 18 |

"Same errors" means every compared node's error (the case's own `err()`, printed to 9 digits) is equal, not only the
verdict. Each fault's record replayed without the fault passes every case (48/48, 48/48, 22/22, 22/22): the stored
references are the CPU's, untouched by the card's fault. The sm_86 build runs native Ampere code and the rig build JIT
compiles its compute_61 PTX: the same branch, two builds, the same results.

**The recorded rig sets replayed on the card** (`scripts/replay_checks.sh` with the PTX-mode switches, every run of
`runs.tsv`; `sets.tsv` has every run):

| Build | Set | Runs as expected | Not as expected |
|---|---|---|---|
| sm_86 | `kshape` | 17 of 18: k-column 1,961/1,961 in 8 settings, large activations 138/138 twice, Q2_0 experts 340/340 in 6 `MTOK` settings, IQ experts 952/952 | `kcol` (defaults): a suite-definition bug (an empty switch field written as `:`, which `env` ran as a command), fixed in `record_checks.sh` and in the recorded `runs.tsv` |
| rig build, PTX | `kshape` | 18 of 18, the same counts | — |
| rig build, PTX | `ladder3-rest` | 61 of 63: the grid types' and IQ1_M's k-column and multi-token runs in 4 settings, C2's runs, the sparse attention's 6 runs (faults 2 and 3 fail cases, as expected), 45 memcheck cases with 0 errors | `iqkcol-off`, `iqkcol-skip`: an illegal memory access in llama.cpp's own MMQ on the repeated-expert-id cases (5–8 tokens, `dup=1`) when the fork's kernels are skipped, through the PTX only (C315, Q875; the PTX-mode scripts leave those cases out); the rig runs them natively |

The pass counts are those of the live PTX runs of the same filters (the `-kshape` folder's `checks.tsv`), and a fresh
live run of the k-column filter beside the replay gave 1,961/1,961 again. A flipped byte in a blob was tested on the CPU
(t3, t4); the md5 check runs before anything computes, so it does not depend on the backend.

## Commands

```bash
# build machine: the rig tarball and an sm_86 test-backend-ops (no GPU needed)
WORK=<build folder, src/ a llama.cpp clone with the fork as remote "fork"> REPO=<this repository> SRC=<checkout of the branch> \
  bash docs/data/2026-11-04_replay_checks/run/build.sh
# build machine: record a suite with the rig tarball's test-backend-ops (in a container without GPUs) and tar it
OUT=<records> SUITE=kshape BIN=<unpacked rig tarball>/bin bash scripts/record_checks.sh
# a suite that shares cases with a recorded one: those are hardlinked, not drawn and quantized again
OUT=<records> SUITE=ladder3-rest BIN=<bin> LINK_FROM=<records>/kshape bash scripts/record_checks.sh
# rig: unpack the tar and the tarball, then every run of the suite on card 0
SET=<records>/kshape BIN=<unpacked rig tarball>/bin OUT=<out> bash scripts/replay_checks.sh
# or an existing script, unchanged
GGML_TEST_REPLAY=<records>/kshape MODE=native RUNS="kcol-all mtok-iq" BIN=<bin> OUT=<out> bash docs/data/2026-11-04_kshape/run/gpu_checks.sh
```

## Files

| File | What |
|---|---|
| `sizes.tsv` | each suite's filters: cases, input and reference GB, the largest case (`--record-dry`) |
| `cpu_tests.txt` | t1–t7 on a CPU alone |
| `rigref.txt` | the rig binary's references against the record's, a 1-in-20 sample of two sets |
| `exact_sm86.txt`, `exact_ptx.txt` | live against replayed on the same inputs, the fault controls, the sets (one line per check) |
| `sets.tsv` | every replayed run of the sets: build, set, run, switches, expected outcome, cases passed, FAIL lines, exit, memcheck errors, verdict |
| `run/build.sh` | the rig tarball and the sm_86 `test-backend-ops` of the branch |
| `run/cpu_tests.sh` | the record and replay paths tested on a CPU alone |
| `run/rigref_check.sh` | the reference check with the rig binary |
| `run/gpu_exact.sh` | the tests on one newer card (sm_86 build, or the rig build through the compute_61 PTX) |
| `run/sets_table.sh` | `sets.tsv` from `replay_checks.sh`'s console lines |

The two runs on the card: `gpu_exact.sh` with `PARTS="same faults corrupt sets" SUITES=kshape SAME="kcol-all mtok
qsa-sparse2"` and `BIN` the sm_86 build; then `SUITES="kshape ladder3-rest" LIVE=kcol-all` with `BIN` the rig
tarball's `bin/`. The first was stopped during its last check, a fresh live run of the IQ experts, which is CPU-bound;
the second during the same check, after the sets.

Provenance: the fork branch's own test cases; the inputs are random draws; nothing here comes from a model or a data
set. Licence: as the repository.
