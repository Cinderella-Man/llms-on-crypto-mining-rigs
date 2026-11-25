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
(compute 8.6, which do have fast BF16). Correctness only: nothing on that machine is timed. What the probe costs a
nine-card P106 rig, in time and host memory, is [the rig check](#the-rig-check) below (2026-11-25).

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
- **On a P106 rig, skipping the probe takes time off every CUDA start and costs nothing**
  ([the rig check](#the-rig-check), 2026-11-25, nine P106-090 cards). In five pairs of device lists the CUDA start was
  shorter by 81.733, 82.938, 88.335, 82.935 and 82.808 ms without the probe, and by 118.611 ms in a model load. That is
  a gain at the start itself. Against the load's 541.148 s it is negligible, and agentic speed does not change, since a
  server starts once. Memory: negligible. The probe's 14.2 MiB are freed 0.1 s into the start, before any model file is
  read, and 0.1 MiB stays after it. Both loads gave the same greedy reply.

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

Run on 2026-11-25, 15:03 to 15:22, on one rig of nine P106-090 cards (driver 580.178.04, 4 GB of host memory). The
cards ran at their daily clock offsets ([bench/mem_offset](../../../bench/mem_offset/README.md)) with the kit's
watchdog on: every card at its offsets, 0 Xid before and after. The build is this one
(`llama-server` md5 `3154dba02422da775b4a554ca5b28a1c`). Two arms differ only in `GGML_CUDA_NVML_PROBE`: unset (no
P106 has fast BF16, so the probe is skipped) and `1` (upstream's probe). [run/rig_probe.sh](run/rig_probe.sh) starts
[run/rig_probe.py](run/rig_probe.py) on the rig, detached, waits for it and collects it:

1. **The device list, five pairs**, the order of each pair alternating: `llama-server -lv 4 --list-devices` with the
   preset's environment. Each run's wall time and peak resident memory, and from the probe's line its own time to open
   and initialise NVML, to shut it down, and the memory it added.
2. **One load each way** of Ornith-1.5-122B Pascal N2P GGUF (`valid1`), the rig's daily model, on its v5 preset
   ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)) over nine cards, with `-lv 4` first. It
   records the time from the start to `/health`, the log's own times, and the server's peak resident memory. It also
   records the host's `MemAvailable` before the load and its lowest value during it (every 0.25 s; a load stops below
   60 MiB), and one 16-token greedy reply in each arm, compared. The probe runs at CUDA start, before any model file is
   read, so the model does not change what it costs. The same scripts take Flint's preset
   (`PRESET=release/flint-v5/settings.env`).

"CUDA start" below is the log's own time from "found 9 CUDA devices" to `ggml_cuda_init`'s last line, the probe's
line, which prints whether the probe ran or was skipped. It covers the device loop and the probe. Each figure is one
run, nothing averaged.

**The device lists** (each a whole `llama-server` process):

| Pair (order) | CUDA start, skipped ms | CUDA start, probe ms | probe's open and init / shutdown and close ms | wall, skipped s | wall, probe s | peak RSS, skipped KiB | peak RSS, probe KiB |
|---|---|---|---|---|---|---|---|
| 1 (skipped first) | 3.265 | 84.998 | 19.1 / 62.8 | 2.9965 | 2.5959 | 1,293,372 | 1,293,384 |
| 2 (probe first) | 2.891 | 85.829 | 19.4 / 63.4 | 2.4383 | 2.4755 | 1,293,596 | 1,293,444 |
| 3 (skipped first) | 2.889 | 91.224 | 19.0 / 69.2 | 2.4138 | 2.4850 | 1,293,736 | 1,293,400 |
| 4 (probe first) | 2.886 | 85.821 | 19.1 / 63.7 | 2.4647 | 2.4843 | 1,293,732 | 1,293,952 |
| 5 (skipped first) | 2.880 | 85.688 | 19.1 / 63.6 | 2.4324 | 2.5596 | 1,293,624 | 1,293,700 |

The probe held 14.2 MiB of resident memory after `nvmlInit` in all five runs and 0.1 MiB after it closed. Pair 1 is
the first CUDA start of the session, which is why its skipped run is the slowest list.

**The loads** (the skipped arm first):

| | probe skipped | probe (`=1`) |
|---|---|---|
| CUDA start, ms | 2.912 | 121.523 |
| probe's open and init / shutdown and close, ms | | 28.0 / 90.3 |
| probe's resident memory after init / after close, MiB | | +14.0 / +0.1 |
| tensor read ("loading model tensors" to "constructing llama_context"), s | 535.619 | 539.705 |
| start to `/health`, s | 541.148 | 546.849 |
| server's peak resident memory (at `/health`, the same at exit), KiB | 2,705,340 | 2,996,188 |
| host `MemAvailable` before the load, KiB | 3,022,548 | 3,273,128 |
| host `MemAvailable`, lowest during the load, KiB | 1,407,412 | 1,415,944 |
| 16-token greedy reply | the same token ids | the same token ids |

The two logs differ only in the probe's line, the switch list (one name more) and the times. Every buffer they print is
the same size.

**The verdict, by the project's classes** (gain, negligible or loss):

- **Time: a gain at every CUDA start, negligible against a load.**
  - Skipping the probe shortened the CUDA start by 81.733, 82.938, 88.335, 82.935 and 82.808 ms in the five pairs, and
    by 118.611 ms in the load.
  - Within each arm, the five starts without the probe differ by at most 0.385 ms, and the five with it by at most
    6.226 ms.
  - In the whole device list, the probe run was slower in pairs 2 to 5 (by 37.2, 71.2, 19.6 and 127.2 ms). In pair 1
    the skipped run was slower by 400.6 ms, as the session's cold start.
  - In a load the 118.611 ms is inside the 5.701 s between the two loads' times to `/health`. Of those 5.701 s,
    4.086 s were the tensor read from the model's USB drive.
  - A server starts once per session, so agentic speed does not change.
- **Host memory: negligible.**
  - The probe's 14.2 MiB (14.0 in the load) are held only while it runs, 0.1 s into the start, before any model file
    is read. 0.1 MiB stays after it closes.
  - The device lists' peaks interleave (probe minus skipped: +12, −152, −336, +220 and +76 KiB).
  - In the loads, the lowest `MemAvailable`, which decides whether a load fits the 4 GB host, was 8,532 KiB higher with
    the probe.
  - The server's peak was 290,848 KiB higher with the probe. The server maps the model file (`load_mode = mmap`), so
    its peak counts the file pages it holds, and the second load started with 250,580 KiB more memory available. One
    load each way does not split that peak into anonymous and file pages.

A bare nine-card device list held 1,293,372 KiB or more at its peak in all ten runs (each value in the table). Where
that host memory goes is registry row Q565's question: host memory per CUDA context.

```bash
# DRY=1 with STEP=start prints the two load lines and starts nothing
RIG=<ssh alias> WORK=<folder on the rig's model drive> BIN=<this build's llama-server on the rig> \
  MODEL=<Ornith-1.5-122B Pascal N2P GGUF on the rig> PRESET=release/ornith-v5/settings.env \
  STEP=start bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
RIG=<ssh alias> WORK=<folder on the rig's model drive> STEP=wait bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
RIG=<ssh alias> WORK=<folder on the rig's model drive> OUT=<folder here> STEP=collect bash docs/data/2026-11-24_c1025_nvml_probe/run/rig_probe.sh
python3 -I docs/data/2026-11-24_c1025_nvml_probe/run/rig_tables.py <folder here>/out docs/data/2026-11-24_c1025_nvml_probe/rig
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
| [rig/summary.md](rig/summary.md), [rig/results.csv](rig/results.csv) | the rig check as `rig_probe.py` wrote it: every run by value, with each load's reply token ids |
| [rig/rig_runs.tsv](rig/rig_runs.tsv) | every rig run with its place in the pair, the CUDA start from the log's own times and, for the loads, the tensor read |
| [rig/rig_pairs.tsv](rig/rig_pairs.tsv) | probe minus skipped for each device-list pair and for the two loads |
| [run/rig_tables.py](run/rig_tables.py) | the two `rig_*.tsv` tables from `rig_probe.py`'s output folder (results and logs) |

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
- The rig check's model: Ornith-1.5-122B Pascal N2P GGUF (`valid1`, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`),
  this project's file; the tables hold times, memory figures and reply token ids, no text. Measured on the project's
  own rig on 2026-11-25.
- The tables are this project's own output (MIT, as the repository).
