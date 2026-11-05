# Every fork behaviour behind a switch (C536, Q1025)

The engine branch `pascal-p106-v3-b12076-flags` (`b4d6d427b`, one commit on `-ladder4` `e3b30303d`) gives every
behaviour the fork adds to upstream llama.cpp b12076 a run-time switch whose default is the behaviour it had. The
reference page of all switches is [docs/switches.md](../../switches.md); the registry row is Q1025 in
[quants/experiments.md](../../../quants/experiments.md).

## What was done

1. **The audit** ([audit.md](audit.md)): the fork's 197 commits and 33 merges between upstream b12076 (`20a51f24a`) and
   `-ladder4`, read one by one. For each: what it adds, its switch and default, and whether its default path differs
   from upstream's without a switch. 29 items were on the default path without one.
2. **The switches:** 20 new environment variables and two extended ones cover 23 of those items. The other six: three are
   fixes whose other value would bring a crash or a corrupt cache entry back, two change only output (a backend feature
   entry; server timing fields and log levels), and one tuning has a build-time switch only (a run-time one is Q1040).
   Of one switched item, the device-side -1 id checks stay: a comparison never taken for valid ids. Each default is
   the engine's behaviour:
   - the upstream variants of kernels are extra template instantiations, so the default kernels are the ones before;
   - every new variable reads "unset or 1 = the feature on", except the existing `GGML_CUDA_MMQ_MOE_LEGACY`.
3. **The checks** below, on a build machine, CPU only.

## Results

Every check passes. [summary.tsv](summary.tsv) has every verdict with its detail; [summary.run1.tsv](summary.run1.tsv)
has the first run of the last two steps. In that run the context step's parser missed one verdict line that a log line
was printed into, and the server requests restored no checkpoint; both were corrected for the second run.

| Check | Result |
|---|---|
| Rig build (sm_61 + compute_61 PTX, `TESTS=1`) and the extra tests | built, no new warnings (the same 12 as `-ladder4`'s build), 0 AVX in `llama-bench`, GLIBC_2.38 |
| Identity at the defaults, `test-llama-archs -b CPU`, against `-ladder4` | rig builds and CPU builds: 143 of 143 compared architectures bit for bit, the same summary (148 tests, 7 counted failed in both: the dense and MoE variants of llama, refact, minicpm, granite, granitemoe, mistral3 and k2-horizon share one logits file in the compare mode, so the second variant's compare fails; every result column is OK or SKIP) |
| Each graph and cache switch flipped, `test-llama-archs` | `LLAMA_ATTN_K_LAST=0`, `LLAMA_KV_HEAD_EMPTY=0`, `_TAIL=0`, `_FREE=0`, `LLAMA_KV_RESTORE_CANONICAL=0`, `LLAMA_GPU_KQ_CELLS=0`, `LLAMA_QSA_PAD_LINEAR=0`, `GGML_SCHED_CTX_PER_GRAPH=0`: all 143 the same as the defaults. Upstream's QSA pad search gives the same cells, so qwen4exp's logits are unchanged |
| The nine CUDA switches flipped together, `test-llama-archs` on the CPU | the same as the defaults (no CUDA code runs; their GPU check is Q1041) |
| `test-llama-context` on random llama, gemma3, qwen35, qwen4exp and mamba models | base and branch at the defaults: 93 OK, 27 SKIP, 0 FAIL, the same verdict per case; each of the seven cache and graph switches flipped: the same 93 / 27 / 0 |
| llama-server on a random gemma3 (sliding window 512), five requests ([run/server_arms.sh](run/server_arms.sh)) | the default gives the base's tokens, cached tokens, restore points and checkpoint sizes; `LLAMA_STATE_NOINIT=0` the same. Every flipped arm gives the default's tokens, and each shows its rule: see the next table |

The server arms. Each request reads `read` tokens and takes `cached` from the cache; checkpoints are 5 per arm:

| Arm | P3 (diverges after C1) | P4 (diverges inside C1's window) | Checkpoints |
|---|---|---|---|
| default, base, `LLAMA_STATE_NOINIT=0` | C1 restored, resumes at 2,996: 520 read | C1 restored past the divergence: 616 read | 5 × 0.759 MiB |
| `LLAMA_SERVER_CKPT_RESUME_NEXT=0` | resumes at 2,995 (stock): 521 read | as the default | 5 × 0.759 MiB |
| `LLAMA_SERVER_CKPT_RING_PAST=0` | as the default | no checkpoint serves it: 3,516 read | 5 × 0.759 MiB |
| `LLAMA_SERVER_CKPT_RING=0` | as the default | no ring checkpoint, so none serves it: 3,516 read | 5 × 0.506 MiB (the last window only) |
| `LLAMA_KV_HEAD_*=0` and `LLAMA_KV_RESTORE_CANONICAL=0` together | as the default | as the default | one of 0.684 MiB, 4 × 0.759 MiB (other cells in the ring) |

Not covered by these checks: the CUDA switches at their other values on a GPU (Q1041), and any switch on a P106.

## Files

| File | What |
|---|---|
| [audit.md](audit.md) | the audit: the resolution of each item, then every commit and merge with its switch, default and reading |
| [summary.tsv](summary.tsv) | every check's verdict and detail (check, verdict, detail) |
| [summary.run1.tsv](summary.run1.tsv) | the first run's context and server verdicts (superseded, see above) |
| [run/checks.sh](run/checks.sh) | the build and the checks (steps `basecpu build cpubuild identity flips context server`); the build, CPU build and identity steps call the `-ladder4` merge's [run/checks.sh](../2026-11-04_ladder4_merge/run/checks.sh) |
| [run/server_arms.sh](run/server_arms.sh) | the server step: one llama-server per arm, five requests, tokens, restores and checkpoint sizes |

## Commands

On the build machine, from a work folder whose `src` is a llama.cpp clone with the fork as remote `fork` (https fetch
URL), and with `-ladder4`'s unpacked rig tarball (extras in its `bin/`):

```bash
WORK=<work dir> REPO=<this repository> BASE_BUILD=<-ladder4 tree> JOBS=8 THREADS=8 LOCK=<build lock> \
  DOCKER_ARGS='--label <label>' bash docs/data/2026-11-05_switches/run/checks.sh
# the second run of the last two steps
WORK=<work dir> REPO=<this repository> BASE_BUILD=<-ladder4 tree> THREADS=8 STEPS='context server' \
  bash docs/data/2026-11-05_switches/run/checks.sh
```

Every container runs with `--cpu-shares=256` and no GPU; the builds take the build lock. The random models are written
by the branch's `test-llama-archs -o` with `LLAMA_ARCHS_N_CTX=4096`.

## Build

| File | Bytes | md5 |
|---|---:|---|
| `llama.cpp-20261105-b4d6d427b-pascal-p106-v3-b12076-flags.tar.gz` (`TESTS=1`, sm_61 + compute_61 PTX, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`) | 608,288,139 | `fcc5dcf26e8f4c5f1d21988469209f8f` |
| `extra-b4d6d427b.tar.gz` (`test-llama-archs`, `test-llama-context`, `test-draft-mem`, `test-hop-cast`, `test-bytes-switches`, `test-moe-ids`, `test-spec-adapt`, `test-hop-mailbox`, `test-route-once`) | 468,465,412 | `584a75b921320b78c496abce4893d9b8` |

Provenance: the fork's public branch, upstream llama.cpp (MIT); random-weight models only, no model files or text
corpora.
