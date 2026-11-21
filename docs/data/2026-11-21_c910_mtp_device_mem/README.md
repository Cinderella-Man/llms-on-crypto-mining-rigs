# `--mtp-device` made safe for a 4 GB host: the head copy counted, freed and copied in pieces (C910, Q1647; 2026-11-21)

What this is: a fix to the v5 engine's `--mtp-device N` (contributed in pull request 7, off by default), built and
checked on a build machine's CPU. With the switch set, the model's MTP (nextn) layers go to card N and the output head
and its norm are copied there for the MTP graph. Pull request 7's review found three problems with that copy, all with
the switch set only: the load's memory fit did not count it, it was never freed, and the load briefly held the whole
head in host memory, which on a mining rig's 4 GB host is several hundred MiB for Ornith-1.5-122B. Fork branch
`pascal-p106-v5-mtp-device-mem` (`f94aa552b`, two commits on `pascal-p106-v5` `3dfad876f`). Registry row Q1647 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **All three are fixed, and nothing changes with the switch unset.** With `--mtp-device` unset the new code does not
  run: every check against the base gave the base's result (logits of 148 random models, ctest test by test, the
  server's replies, and the GPU code, all 8,836 kernels byte for byte).
- **The fit now counts the copy.** In the fit's own dry run the model's memory grew by exactly the copy's size, on the
  chosen device. Before, the fit saw no difference, so the card could be overfilled by the head's size.
- **The copy is freed with the model.** Loading and freeing the model twice in one process left the devices where they
  started. Before, each load left one head's worth of device memory behind.
- **No host buffer of the head's size any more.** The largest host allocation of the load was one 16 MiB piece when
  the head moves between two cards, and nothing extra when the head comes from host memory. Before, it was the whole
  head (254,279,680 bytes on the test model).
- **The copy is right.** Built on the copy, the MTP head's guesses were identical to those built on the original head:
  every drafted token, and the top 8 candidates with their probabilities at all 1,218 drafted positions.
- **Not yet on a P106.** The rig check (Ornith with `--mtp-device 0` on rig 2's cards and 4 GB host) is a separate row;
  its driver is [run/c910_rig.sh](run/c910_rig.sh).

## What changed in the code

One file, `src/llama-model.cpp`, all of it inside `if (params.nextn_device >= 0 ...)` blocks:

| Problem (pull request 7 review) | Before | Now |
|---|---|---|
| The fit does not count the copy (finding 2) | the copy's buffer was made after the loader's dry-run return (`no_alloc`), so the fit never saw it | the copy gets its room before that return, as a weight buffer in the model's buffer list (a size-0 placeholder in the dry run, as the other weights get), so the fit's dry run counts it on card N |
| The copy is never freed (finding 3) | its `ggml_context` and device buffer were kept outside the model ("the process exit releases it") | both live in the model's buffer list (`pimpl->ctxs_bufs`) and `llama_model_free` frees them with the other weights |
| A host buffer of the head's size at load (finding 4) | `std::vector<uint8_t>` of the whole head, read from the source card and written to card N | `llama_tensor_copy_bounded`: from host memory (or to it) directly; card to card in whole rows through a host buffer of at most 16 MiB |

The copy's data is still written once all weights are loaded (after the Hadamard step), and the log line is the same:
`MTP output head copied to <device>, <MiB> MiB (--mtp-device)`. The load log now also lists the copy's buffer among the
model buffers, as one more `<device> model buffer size = <MiB> MiB` line.

## How it was checked without cards

The build machine runs no GPU work, so two local `ggml-rpc-server` processes, each serving its CPU, stood in for two
cards: llama.cpp treats RPC devices as GPUs, so with all layers on the devices the output head sits on the second
device and `--mtp-device 0` really copies it from one device to the other. The test model is a random `qwen35moe` MTP
model (Ornith's architecture) with a real 248,320-token vocabulary, written by the repository's `test-llama-archs`
with its MTP filter widened to `qwen35moe` (the one-line change is in `run/run.sh`, step gen), then with its own MTP
head and head norm removed ([run/strip.py](run/strip.py)), so that its MTP graph reads the main output head and norm,
as real Qwen MTP files do, and with `--mtp-device` their copies. Its head is 254,279,680 bytes (F32, 248,320 × 256) and
its norm 1,024.

[run/mtp_device_check.cpp](run/mtp_device_check.cpp), built against each tree, does three things per setting:
- **the fit's view:** loads the model with `no_alloc` and makes a context on it, as the fit's dry run does
  (`common/fit.cpp`), and reads the model bytes per buffer type (`llama_get_memory_breakdown`);
- **two loads and frees in one process**, reading the RPC servers' resident memory before each load, after it and after
  the free (a device buffer that is never freed stays resident on its server);
- **the largest host allocation during each load:** every `malloc`-family call of the process goes through the
  program's own definitions, which record the largest request (`std::vector` and `operator new` use `malloc`).

### The fit's view ([fit.tsv](fit.tsv))

Model bytes the fit counts, all layers on the two devices:

| Build | `--mtp-device` unset | `--mtp-device 0` | Difference |
|---|---|---|---|
| base `3dfad876f` | 776,190,336 | 776,190,336 | 0: the copy is not counted |
| C910 `f94aa552b` | 776,190,336 | 1,030,471,040 | **254,280,704 = the head and its norm**, all on the first device (521,910,656 there against the base's 267,629,952) |

With no layers on the devices (`-ngl 0`, the head in host memory) the same: 776,190,336 on both settings on the base,
776,190,336 and 1,030,471,040 on C910. The MTP layer's move to the first device shows on both builds (its weights,
with its own token embedding, leave the second device or the host); only C910's total grows.

### Loads and frees ([loads.tsv](loads.tsv))

| Build | Layers on the devices | `--mtp-device` | Largest host allocation, load 1 and 2 (bytes) | RPC servers resident after each free (KiB; 39,208 or 39,060 before the first load) |
|---|---|---|---|---|
| base | all | unset | 9,932,800 and 9,932,800 | — (the unset runs came first) |
| base | all | 0 | **254,279,680 and 254,279,680** | **303,920, then 552,244** |
| C910 | all | 0 | **16,777,216 and 16,777,216** | **22,804, then 30,832** |
| base | none | 0 | 254,279,680 and 254,279,680 | 296,328, then 544,652 (39,208 before) |
| C910 | none | 0 | 9,932,800 and 9,932,800 (the unset runs' largest) | 31,600, then 31,600 (30,832 before) |

- On the base each second load with the switch set left 248,324 KiB more resident on the servers after its free, the
  head and norm's 248,321 KiB; each first load left 264,712 (all layers on the devices) and 257,120 (none). The
  servers' heaps also keep some memory after their first large loads, as the unset runs show: 12,180 then 39,208 KiB
  on the base, 12,032 then 39,060 on C910. On C910 nothing stays after a load with the switch set.
- 9,932,800 bytes is the largest allocation of any load without the copy (the loader's own); 16,777,216 is the copy's
  16 MiB piece. On C910 no allocation of 64 MiB or more happened in any load; on the base one in each load with the
  switch set.

### Server replies and the copy's contents ([server.tsv](server.tsv))

Five requests per server (greedy, sampled at seed 7, two JSON-schema requests, greedy again; [run/client.py](run/client.py)),
token ids compared, and for MTP-draft runs every verify step's drafted tokens and the MTP head's top-8 candidates with
their probabilities (the server's `LLAMA_SPEC_DUMP`, [run/compare.py](run/compare.py)):

| Run against reference | Same tokens | Drafts | Top-8 candidates and probabilities |
|---|---|---|---|
| Rig builds, dense model, `-bs`, nothing set | 5 of 5 | — | — |
| Rig builds, dense model, `-bs --pipe-depth 3`, nothing set | 5 of 5; the same 370 guesses, 37 right, 113 wrong | — | — |
| Rig builds, the MTP model, `-bs --spec-type draft-mtp`, nothing set | 5 of 5 | identical, 406 verify steps, 1,203 drafted tokens | identical at all 1,218 positions |
| CPU builds on the two RPC devices, MTP draft, `--mtp-device` unset: C910 against base | 5 of 5 | identical | identical |
| `--mtp-device 0` against unset, base | 5 of 5 | identical | identical |
| `--mtp-device 0` against unset, C910 | 5 of 5 | identical | identical |
| `--mtp-device 0`, C910's copy against the base's | 5 of 5 | identical | identical |

The `--mtp-device 0` servers' logs (`-lv 4`) carry the copy line, and C910's lists the copy's 242.50 MiB buffer on the
first device ([load_lines.txt](load_lines.txt)). The base's pipelined server exited with code 1 when stopped after its
fifth reply ("Received second interrupt": the stop signal arrived twice); its replies and counts were complete.

### Build checks

| Check (C910 `f94aa552b` against base `3dfad876f`) | Result |
|---|---|
| Rig build (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, sm_61, `HF_UI_VERSION=b12483`) | rc 0; 38 warning lines on both, the same set |
| Native CPU build of every target ([cpu_summary.tsv](cpu_summary.tsv)) | rc 0 on both; 26 warning lines on both, the same set |
| ctest, 75 tests, no network ([ctest.tsv](ctest.tsv)) | 71 pass on both, test by test the same; `test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` fail on both (the base's known four) |
| `test-llama-archs -b CPU -s 1`, native | 153 of 153 three times on both; 148 of 148 logit files the base's bit for bit |
| `test-llama-archs`, C910's rig build against the base's rig build | 153 of 153; 148 of 148 files identical |
| Hand-run tests (hop-cast, route-once ×2, bytes-switches ×2, moe-ids, thread-safety, context-model, eval-callback) | all pass on both |
| `cuobjdump -sass` of `llama-server` (`1df013f4…` against `5b330ecc…`, [sass_summary.txt](sass_summary.txt)) | 8,836 of 8,836 kernels identical, none new or missing; the 176 `kq_fp32` kernels identical |

Rig tarball `llama.cpp-20261121-f94aa552b-pascal-p106-v5-mtp-device-mem.tar.gz`, 841,411,915 bytes, md5
`3f26fdb33d6d9465040a048ff6a5560f` (`bin/llama-server` md5 `1df013f4a0567b916d5547a6bdc0c09a`).

## Limits

- The head copy runs through host memory in 16 MiB pieces even where a backend could copy card to card itself (CUDA's
  peer copy). On the rigs, with no peer-to-peer and every card on a PCIe 1.0 x1 link, both cross the host; the
  difference is est. a few seconds of a load that takes minutes.
- With `--mtp-device` set, the copy is made whenever the file has MTP layers, also when they are not loaded (a server
  without an MTP draft); then it only takes card N's memory (by reading the code; registry Q1662).
- `llama-fit-params` does not take `--mtp-device` (a server option), so its printed estimate leaves the copy out
  (registry Q1663). The server's own fit, which the server runs at start, counts it.

## Files

| File | What |
|---|---|
| [fit.tsv](fit.tsv) | the fit's dry run: model bytes per buffer type, per build, layer placement and setting |
| [loads.tsv](loads.tsv) | two loads and frees per setting: the largest host allocation, allocations of 64 MiB or more, the RPC servers' resident memory |
| [server.tsv](server.tsv) | every server comparison: tokens, drafts, the MTP head's candidates, the pipeline's counts |
| [load_lines.txt](load_lines.txt) | the load log's copy, MTP-layer and model-buffer lines of each run |
| [ctest.tsv](ctest.tsv), [cpu_summary.tsv](cpu_summary.tsv) | ctest per test; the CPU test set's verdicts |
| [sass_summary.txt](sass_summary.txt) | the card code comparison |
| [run/run.sh](run/run.sh) | the whole check on a build machine (steps src, rig, cpu, gen, archs, server, rpc, sass) |
| [run/mtp_device_check.cpp](run/mtp_device_check.cpp) | the fit view, load-and-free and host-allocation check |
| [run/compare.py](run/compare.py), [run/client.py](run/client.py), [run/strip.py](run/strip.py) | the server comparison, its client (C908's), the GGUF copy without the MTP model's own head (pull request 7's review's) |
| [run/c910_rig.sh](run/c910_rig.sh), [run/rig_table.py](run/rig_table.py) | the rig check on Ornith (three loads with `bench/deep_lane`) and its table |

## Commands

On the build machine, from a folder holding the scripts (`REPO`), with the base's clone, rig build and build log of
[2026-11-21_c908_v5_merged_build](../2026-11-21_c908_v5_merged_build/README.md) and any GGUF with a tokenizer as
`VOCAB`; `cpu_tests.sh` is [2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh),
`sass_compare.py` and `sass_classify.py` are C861's and C854's:

```bash
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> BASE_RIG=<3dfad876f's rig build folder> \
  BASE_RIG_LOG=<its build log> VOCAB=<vocab.gguf> REPO=<scripts> LOCK=<flock file> JOBS=8 \
  STEPS="src gen rpc" bash run.sh
# then the rest (the server step reads gen's models):
STEPS="src rig cpu archs server sass" ... bash run.sh
```

Every container carried `--cpu-shares=256` and `--memory=16g`; everything that ran the built code ran without network.
The rig check, from the repository's root on a machine with ssh to the rig:

```bash
RIG=<rig> WORK=<run folder on the rig> BIN=<C910's llama-server> OLD_BIN=<3dfad876f's llama-server> \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig> TEXT_LOCAL=<recorded agent text> \
  STEP=start bash docs/data/2026-11-21_c910_mtp_device_mem/run/c910_rig.sh
# STEP=wait, then STEP=collect OUT=<here>; then:
python3 -I docs/data/2026-11-21_c910_mtp_device_mem/run/rig_table.py <OUT>
```

## Provenance

The fork's commits are public on [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama). The test
models are random (`test-llama-archs`, with the vocabulary-only GGUF pull request 7's review and C908 used). The base's
rig build and its build log are C908's. Raw logs are not published.
