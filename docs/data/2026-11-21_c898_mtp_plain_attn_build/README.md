# Flash-Next's draft runs a layer marked plain attention: the build and CPU checks (2026-11-20)

**Built and checked on a CPU; not yet run on a P106.** Registry [Q1623](../../../quants/experiments.md) (C898). The
published distilled draft head for Qwen3.8-Flash-Next ([Q1545](../2026-11-19_fn115_flash_draft_head/README.md)) marks
its one layer plain attention (compress ratio 0), while the engine's draft code assumed the trunk's sparse attention
(QSA) and stopped on `GGML_ASSERT(buffer)` before the first request. The fix builds the sparse attention's indexer cache
and k-pool inputs for the draft only when the draft's own layer is a QSA layer. The preset's own head (ratio 4) must
build exactly the graph it built before.

- Fork branch [`pascal-p106-v5-mtp-plain-attn`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-mtp-plain-attn)
  at **`c9c3134d4`**, one commit on `pascal-p106-v5` `602691153` (upstream b12483 on the v5 line). No switch: the
  change acts only on a draft layer at ratio 0.
  - `src/llama-model.cpp` (`create_memory`): the MTP context gets its indexer cache only when its block's compress
    ratio is above 0.
  - `src/models/qwen4exp.cpp` (`graph_mtp`): the k-pool inputs only then.
  - `tests/test-llama-archs.cpp`: the new `--mtp-plain-attn` test.
- Rig build `llama.cpp-20261120-c9c3134d4-pascal-p106-v5-mtp-plain-attn.tar.gz` (841,346,134 bytes), md5
  **`9f0ab5f1cb5eb21cb57e1c2cd6a15014`**: CUDA 12.9.2 image, `sm_61`, `TESTS=1`, 0 AVX instructions in `llama-bench`,
  highest GLIBC symbol `GLIBC_2.38`, the web UI upstream b12483's archive (the same embedded bytes as the v5
  candidate's build).

## Result

| Check | Pass rule | Result |
|---|---|---|
| CPU build of every target, new (`c9c3134d4`) | builds, no new warning | **built**, 0 errors, 30 warning lines: 27 `-Wdeprecated-declarations` (upstream's old batch calls in eight of the fork's own tools and tests) and 3 `-Wmissing-field-initializers` in `common/sampling.cpp`; none in the three changed files |
| CPU build, base (`602691153`) | builds | **built**, the same 30 warning lines, line for line |
| `test-llama-archs -a qwen4exp -s 1`: `--mtp-kv-only`, the same with the preset's three QSA switches, `--mtp-graph-keep`, `--mtp-replan`, `--mtp-catchup-overlap` | every test passes on both trees, and the preset head's passes (MTP block at ratio 4) give the base's node counts and outputs | **passed: every output the same bytes on both trees** apart from its first line, a timestamp (5, 5, 8, 4 and 5 cases) |
| `--mtp-plain-attn` (new tree only) | the ratio-4 block keeps its indexer cache, QSA ops and k-pool reads; the ratio-0 block has none of them and drafts | **passed, 5 of 5**: ratio 4 one indexer cache, 84 and 57 k-pool reads, 60 and 24 QSA ops (KV-only passes off and on); ratio 0 none, with and without flash attention and with the switches; drafts finite and the same bytes with KV-only passes off and on |
| `test-llama-archs -s 1`, every architecture | as before | 153 of 153 on both trees, the same bytes after the timestamp line |
| `--mtp-plain-attn` on the base's code (`602691153` with only the new test file) | the test catches the bug | **it does:** the two ratio-4 arms give the new tree's counts, then the first plain arm stops on `GGML_ASSERT(buffer)` (`ggml-backend.cpp:347`), exit 134 |
| Rig tarball | builds, 0 AVX, GLIBC_2.38 | **built**; its 12 compiler warning lines are the v5 candidate's rig build's, line for line |

Every table is in [tests.md](tests.md). In the `--mtp-plain-attn` table, `Idx` counts the "creating indexer KV cache"
lines of the draft context, `Dev` its k-pool tables on the device, `K-pool in` the graph ops over all of the sequence's
passes that read a `kpool_*` input, and `QSA ops` the ops named as sparse-attention ops (`indexer*`, `kpool*`,
`kq_mask_qsa`); each as "KV-only passes off / on". On the base's code the first plain arm stops on the same
`GGML_ASSERT(buffer)` as the published head did on the rig.

What the CPU checks cannot show: the draft's dense attention over the whole cache on a P106 (every attention layer of
Flash-Next's trunk is QSA, so this kernel path has not run on these cards with this model), and whether the published
head fits on the draft's card (est. 32 MiB less than before the fix, from its indexer cache). Both are the rig check,
the next step: Q1545's probe and lane on this build, where the preset's head must repeat
[R205](../2026-11-19_r205_flash_glu/requests.csv)'s replies and accepted drafts on `602691153` request for request.

## Commands

[`run/build_check.sh`](run/build_check.sh) drives a build machine over ssh (`BUILDER`, `WORK`, `SEED`, `LOCK` from the
environment); every step runs in Docker on its CPU, one container at a time under a lock, `--cpu-shares=256`, `-j 8`,
no network for the CPU builds and tests. `SEED` was the v5 candidate's build folder (its llama.cpp clone and caches).
The whole run took 23:37 to 23:46 on 2026-11-20, the caches making every build short.

```bash
B=run/build_check.sh
bash $B seed
bash $B cpu fork/pascal-p106-v5-mtp-plain-attn new && bash $B wait cpu-new/RC
bash $B cpu 602691153 base && bash $B wait cpu-base/RC
bash $B test new
bash $B test base
bash $B compare
bash $B archs
HF_UI_VERSION=b12483 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" bash $B rig fork/pascal-p106-v5-mtp-plain-attn final
bash $B wait rig/RC-final
TEST_FROM=fork/pascal-p106-v5-mtp-plain-attn bash $B cpu 602691153 basetest && bash $B wait cpu-basetest/RC
ONLY=--mtp-plain-attn bash $B test basetest
```

- The CPU builds: image `pq-c565-cpu:24.04` (Ubuntu 24.04, gcc 13.3.0, cmake 3.28.3), `-DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF`,
  ccache. `basetest` is the base's archive with `tests/test-llama-archs.cpp` replaced by the new branch's.
- `archs`: every architecture, `test-llama-archs -s 1` on each tree in the same image.
- The rig build: [`scripts/build_llamacpp_docker.sh`](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1 JOBS=8`,
  `HF_UI_VERSION=b12483` (the web UI archive the v5 line pins,
  [why](../2026-11-17_c854b_upstream_newest/README.md#the-builds)).
- [`run/tables.sh`](run/tables.sh) writes `tests.md` from the test outputs (`OUT=<folder> bash run/tables.sh`).

| File in the tarball's `bin/` | md5 |
|---|---|
| `llama-server` | `554c5313306e89ab97bbf09eaada002e` |
| `test-llama-archs` | `93a8c15fd969b164fae0b3fb51ef5e4f` |
| `test-backend-ops` | `6acde993c4fedbc001e71dc90a3bf2c5` |
| `test-cuda-keepalive` | `18086425e4abc6dfde60321244381b37` |
| `llama-bench` | `4f273175e23fbe0abc4dbedb2f644277` |
| `llama-completion` | `644711b90168ac565b3b66ac6230cf7d` |

## Files

| File | What |
|---|---|
| [tests.md](tests.md) | every test's table (new tree), whether the base tree's output is the same bytes |
| [run/build_check.sh](run/build_check.sh) | the build and test driver |
| [run/tables.sh](run/tables.sh) | `tests.md` from the test outputs |

Provenance: the fork's code, built from GitHub; the tests use random models made by `test-llama-archs` (no model
file). MIT, as llama.cpp.
