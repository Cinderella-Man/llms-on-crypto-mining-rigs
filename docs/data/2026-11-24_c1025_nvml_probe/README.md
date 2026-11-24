# Upstream's NVML probe skipped where no card reads it (C1025, Q1758; 2026-11-24)

What this is: the code half of registry row Q1758 in [quants/experiments.md](../../../quants/experiments.md), and its
correctness check. Upstream llama.cpp's #30672, merged into the v5 candidate with tag b12674
([C912](../2026-11-24_c912_upstream_b12674/README.md)), opens NVML, the NVIDIA driver's management library, at every
CUDA start: it loads `libnvidia-ml.so.1`, calls `nvmlInit`, and asks each card with fast BF16 for its brand. Quadro-
and RTX-branded cards then multiply quantized weights in cuBLAS with BF16 inputs and FP32 sums instead of FP16. Only a
card with fast BF16 (compute 8.0 and newer; GB10 has its own rule) reads the answer. A P106-090 is compute 6.1, so on
a rig the probe runs at every start and its answer is never used.

The fork branch `pascal-p106-v5-nvml-probe` (`88b0db403`, one commit on the candidate `pascal-p106-v5-cand-b12674` at
`eee615cdb`) opens NVML only when a card would read it. The switch `GGML_CUDA_NVML_PROBE=1` runs the probe always, as
upstream does, so a rig can measure both in one build. It was built and checked on a build machine's RTX 3090s
(compute 8.6, which do have fast BF16). Correctness only: nothing here is timed. What the probe costs a nine-card P106
load is [the rig check](#the-rig-check) below.

## Result in plain words

- **The probe now runs only where its answer is used, and nothing else changes.** On the 3090s, with the switch unset,
  the probe still ran and every card chose as the candidate chose. With `GGML_CUDA_NVML_PROBE=0` the library was never
  loaded. Every greedy reply was the candidate's, token for token, in every arm.
- **cuBLAS made the same calls in every arm.** A prompt of over 64 tokens sends these cards' quantized products to
  cuBLAS, where the choice is made. The candidate and the new build, unset, `=0` and `=1`, each made 128
  `cublasGemmEx` calls with FP16 compute (`CUBLAS_COMPUTE_16F`), 62 `cublasSgemm` calls and the same setup calls.
  A GeForce card takes FP16 either way; the branded cards that take BF16 are not in the build machine.
- **The card code is the candidate's.** All 9,508 kernel names, 10,009 copies, are the same machine code. The change
  is host code only, and no compiler warning is new.
- **What the probe costs on the build machine's one card:** it added 19.4 MiB of resident memory while it ran and
  0.1 MiB after it closed. On a P106 rig, where the host has 4 GB, the rig check measures it with nine cards.
- **What it does not show:** the time and memory the probe costs a P106 rig. That is [the rig check](#the-rig-check),
  est. 30 min.

## The change

- **Where NVML's answer is read.** `ggml_cuda_init` decides `prefer_fp32_acc` for each card in
  `should_use_fp32_acc`. It returns true for GB10 without NVML, and it asks NVML only for a card where
  `fast_bf16_hardware_available(cc)` holds: NVIDIA compute 8.0 and newer. Every other card gets false, with or without
  NVML.
- **The rule.** While `ggml_cuda_init` sums the cards' memory, it notes whether any card is such a card (fast BF16 and
  not GB10). If none is, it does not load NVML, and every card's choice is the same as with the probe. On a machine of
  P106 cards, or of any cards older than compute 8.0, the probe no longer runs.
- **The switch, `GGML_CUDA_NVML_PROBE`.** Unset: the rule. `1`: the probe always runs, upstream's path, to measure it.
  `0`: never; a card with fast BF16 then chooses as it does when NVML is missing. Any other value is ignored with a
  warning. HIP and MUSA builds never opened NVML and are unchanged.
- **It says what it did.** One line at the library's INFO level at CUDA start (llama-server prints it from `-lv 4`
  on, and `-lv 4` must come before the arguments that start CUDA):
  `NVML probe skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it)`, or
  `NVML probe ran (...): library loaded, nvmlInit ok; open and init <t> ms, shutdown and close <t> ms; host RSS
  +<m> MiB after init, +<m> MiB after close; BF16 accumulation for quantized cuBLAS on devices: <list or none>`.
  The memory figures read `/proc/self/statm` (Linux).

## The checks

Everything ran on the build machine's CPU and one RTX 3090 (driver 595, CUDA 12.9 runtime in the build image). The
check step's model is Granite 3.1 3B-A800M Instruct in Q4_K_M (md5 `c8e8b8bc7890057906d99643dbf8063b`), a small MoE
model whose attention projections are dense quantized matrices.

| Check | Result |
|---|---|
| Rig build (`TESTS=1`, web page b12674) | `llama.cpp-20261124-88b0db403-pascal-p106-v5-nvml-probe.tar.gz`, 1,403,241,029 B (the candidate's 1,403,235,928 B), md5 `ebcca32b6fb01dcda2798ff4589547ce`; `llama-server` md5 `3154dba02422da775b4a554ca5b28a1c`, `test-backend-ops` md5 `8271121a02bb55de34299843493cfd37`; 0 AVX instructions, GLIBC_2.38; the candidate's 21 binaries |
| Warnings against the candidate's build | 32 lines and 32; none new |
| Card code (`cuobjdump -sass`, the compute 6.1 code a P106 runs) | 9,508 of 9,508 kernel names the candidate's bytes (10,009 copies each); none missing, none new |
| Device list (`llama-server -lv 4 --list-devices`), the candidate | `libnvidia-ml.so.1` loaded |
| Device list, unset | loaded; the probe ran ("a device has fast BF16"); BF16 accumulation on no device |
| Device list, `=0` | not loaded; "NVML probe skipped (GGML_CUDA_NVML_PROBE=0)" |
| Device list, `=1` | loaded; the probe ran (`GGML_CUDA_NVML_PROBE=1`) |
| Device list, `=2` | a warning that the value is invalid, then the rule: loaded, the probe ran |
| One load each: the candidate, unset, `=0`, `=1` | every load up; two greedy replies of 64 tokens each (a prompt of over 64 tokens and a short one), the same token ids and text as the candidate's in every arm |
| cuBLAS's API log (`CUBLAS_LOGINFO_DBG=1`) in each load | 194 calls in each arm, the same functions and compute types: 128 `cublasGemmEx` with `CUBLAS_COMPUTE_16F`, 62 `cublasSgemm_v2`, and one each of create, math mode, stream and workspace |
| llama-server's switch line | `switches: 1 set: GGML_CUDA_NVML_PROBE=1` (and `=0`): the build reads 961 names, the candidate 960 |

"Loaded" is the dynamic loader's own trace (`LD_DEBUG=files`), which names every library a process opens. Every
container ran on one card, shared for minutes with another session's checks; nothing was timed. The probe's line
prints times, and they are not results on this machine.

[list.tsv](list.tsv) has the device-list arms, [load.tsv](load.tsv) the loads and replies, and
[cublas.tsv](cublas.tsv) the cuBLAS calls by function and compute type in each load. In the two tables the probe's
times are replaced by `<t>`.

The rig script below also ran once on the build machine, on one card with this model and `REPS=2`, with
`GGML_CUDA_NVML_PROBE=0` standing in for the rule a P106 takes: it read the skip line and the probe line in the
device lists and in both loads, and the two loads gave the same reply.

## The rig check

**Planned:** one rig, this build, two arms that differ only in `GGML_CUDA_NVML_PROBE`: unset (no P106 has fast BF16,
so the probe is skipped) and `1` (upstream's probe). [run/rig_probe.sh](run/rig_probe.sh) starts
[run/rig_probe.py](run/rig_probe.py) on the rig, detached, waits for it and collects it:

1. **The device list, five pairs**, the order of each pair alternating: `llama-server -lv 4 --list-devices` with the
   preset's environment. Each run's wall time and peak resident memory, and from the probe's line its own time to open
   and initialise NVML, to shut it down, and the memory it added.
2. **One load each way** of Flint Pascal Q8 GGUF on its v5 preset over nine cards, with `-lv 4` first: the time from
   the start to `/health`, the log's own times of "found 9 CUDA devices", of the first card's line and of the probe
   line, the server's peak resident memory, the host's `MemAvailable` before the load and its lowest value during it
   (every 0.25 s; a load stops below 60 MiB), and one 16-token greedy reply in each arm, compared.

Each figure is read by value, nothing averaged. The probe's own time and memory decide whether the rule, now the
default, is a gain or negligible.

```bash
RIG=<ssh alias> WORK=<folder on the rig> BIN=<this build's llama-server on the rig> \
  MODEL=<Flint Pascal Q8 GGUF on the rig> PRESET=release/flint-v5/settings.env \
  STEP=start bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
RIG=<ssh alias> WORK=<folder on the rig> STEP=wait bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
RIG=<ssh alias> WORK=<folder on the rig> OUT=<folder here> STEP=collect bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
```

## Files

| File | What it holds |
|---|---|
| [list.tsv](list.tsv) | the device list per arm: exit code, how often the loader opened `libnvidia-ml`, the probe line |
| [load.tsv](load.tsv) | one load per arm: up, `libnvidia-ml` opened, the probe line, each reply's token count and whether its token ids and text are the candidate's |
| [cublas.tsv](cublas.tsv) | every cuBLAS call in each load, by function and compute type |
| [run/run.sh](run/run.sh) | the build machine's driver: steps `src`, `rig`, `sass`, `check`, `tables` |
| [run/c1025_tables.py](run/c1025_tables.py) | the three tables from the check step's outputs |
| [run/rig_probe.sh](run/rig_probe.sh), [run/rig_probe.py](run/rig_probe.py) | the rig check: started from a machine that only runs ssh, run on the rig with Python's standard library |

## Commands

The build machine (a 12-core CPU, three RTX 3090s, Docker with the NVIDIA toolkit), from a copy of this repository:

```bash
W=<work dir> SEED=<llama.cpp clone holding eee615cdb> REPO=<this repository> \
CAND_RIG=<the candidate's rig build folder> CAND_RIG_LOG=<its build log> RIG_CCACHE=<its compiler cache> \
JIT_SEED=<a PTX JIT cache> GPU=2 PORT=18925 MOUNTS=<the model's folder> MODEL=<granite-3.1-3b-a800m-instruct-Q4_K_M.gguf> \
PREFLIGHT=<cuda check> LOCK=<build lock> JOBS=8 bash docs/data/2026-11-24_c1025_nvml_probe/run/run.sh
```

`run.sh` reuses `docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py` and
`docs/data/2026-11-24_c912_upstream_b12674/run/sass_classify.py` for the card code, and `scripts/build_llamacpp_docker.sh`
for the build.

## Provenance and licence

- The engine: [miner-llama](https://github.com/Cinderella-Man/miner-llama), branch `pascal-p106-v5-nvml-probe` at
  `88b0db403`, on `pascal-p106-v5-cand-b12674` at `eee615cdb` (upstream llama.cpp b12674 merged into v5); MIT, as
  llama.cpp.
- The check model: IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) in Q4_K_M.
- The tables are this project's own output (MIT, as the repository).
