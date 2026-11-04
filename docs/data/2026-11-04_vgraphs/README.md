# C151 / Q701: CUDA graphs and schedulers per verify width (upstream #31156 and #31274) on Miner Llama (2026-11-04)

With a draft, every step of llama-server verifies the sampled token and the drafted ones in one batch. On the nine-P106
rig the DFlash2 line (Q561) drafts two tokens, so a verify batch is 1, 2 or 3 tokens wide, and the width changes from
step to step as drafts are accepted or cut. llama.cpp keeps one graph arena for every batch with outputs, so each width
change rebuilds the graph in that arena, and since the CUDA graph cache is keyed by the graph's first node (the same
address in the same arena), each change also invalidates the captured CUDA graph: the next steps run uncaptured and
then capture again. Upstream's draft [#31156](https://github.com/ggml-org/llama.cpp/pull/31156) gives batches of up to
8 tokens one arena per token count (and, behind a switch, one scheduler each); its follow-up
[#31274](https://github.com/ggml-org/llama.cpp/pull/31274) sizes the scheduler's scratch for the graph at hand and
allocates each arena's metadata with `calloc`, so its unused pages stay out of memory. This folder holds their port to
the fork, the checks that ran without a GPU, and the script that measures them on the rig. The registry row is Q701
(Q52 is the older plan it implements) in [quants/experiments.md](../../../quants/experiments.md); the source is idea 1
of [flash_next_news_2026-11.md](../../research/notes/flash_next_news_2026-11.md).

**Status:** ported, built for the rig and checked on the CPU. Not yet run on a GPU; [run/rig_vgraphs.sh](run/rig_vgraphs.sh)
measures it on the rig next.

## What is ported

Fork branch `pascal-p106-v3-b12076-vgraphs` at `a476ab83f`, four commits on `pascal-p106-v3-b12076-ladder2`
(`a330672c1`, [the merged engine](../2026-11-04_ladder2_merge/README.md)):

| Commit | From | What it does |
|---|---|---|
| `d86fc66c2` | #31156 `7d229e479` (ported by hand) | one graph arena per token count for batches of up to 8 tokens; `LLAMA_SCHED_PER_SIZE=1` gives each of them its own scheduler |
| `66ba1f654` | #31274 `9f084f355` (cherry-pick, clean) | the scheduler sizes its scratch context for the graph it splits, not for the largest graph it could ever see; the optimizer's allocation dependencies get a context of their own |
| `601525521` | #31274 `8337f4fa1` (cherry-pick, clean) | each arena's metadata buffer is `calloc`'d once and kept across resets, so pages its graph never reaches are never made resident |
| `a476ab83f` | the fork | switches for both behaviour changes, a log line per setting and per scheduler, the per-size schedulers in the memory breakdown |

**The switches** (environment variables of llama-server and every llama.cpp tool; defaults as the pull requests have
them):

| Switch | Default | What the other setting does |
|---|---|---|
| `LLAMA_GRAPH_PER_SIZE` | 1: an arena per token count up to 8 (#31156) | 0: the two arenas of before #31156 (batches with and without outputs), so every verify width shares one arena and one CUDA graph key, as on `-ladder2` |
| `LLAMA_SCHED_PER_SIZE` | 0: one shared scheduler (#31156's default) | 1: each arena of up to 8 tokens gets its own scheduler, so a width that comes back reuses its graph instead of rebuilding and re-splitting it; each scheduler allocates compute buffers of its own on first use. Needs `LLAMA_GRAPH_PER_SIZE=1` (ignored with a warning otherwise) |
| `LLAMA_GRAPH_META_CALLOC` | 1: `calloc`'d metadata (#31274) | 0: one byte per page written when the buffer is allocated, so every page is resident, as with the `std::vector` before #31274 |

The scheduler's scratch sizing (`66ba1f654`) has no switch: it changes how much address space the scratch reserves,
not what any graph computes. At creation each context logs `graphs per batch size up to 8 tokens: on|off, schedulers
per size: on|off`, and with `LLAMA_SCHED_PER_SIZE=1` it logs each per-size scheduler it creates.

**Where the fork differs from upstream's llama-context, and what the port does there:**
- The fork reuses a graph under pipeline parallelism only while nothing is in flight, and its input upload
  (`LLAMA_INPUT_UPLOAD`) and caller-sync scope are per scheduler: the port applies both to whichever scheduler the
  batch uses. A per-size scheduler is created without pipeline parallelism (one input copy), as in #31156, so when an
  earlier micro-batch of the same decode call is still computing it waits for the devices first.
- The staircase drafter's `on_ubatch` reads the scheduler of the graph it was given; its init and reset invalidate
  every arena's graph, not only the shared one's.
- `memory_update()`, `graph_reserve()` and `opt_epoch_iter()` select the shared scheduler before they build graphs
  (the K-shift, reserve and training graphs use it; upstream gets this from `decode()` and `encode()` alone).

**Why the CUDA graph keys separate without Q52's first layer.** The CUDA backend keys its graph cache by the first node
of the graph a split hands it, and a split's graph is a view of the arena's nodes. With one arena per width, each width's
splits start at that arena's own tensors, so each width keeps its own CUDA graphs. Upstream's narrower
[#31004](https://github.com/ggml-org/llama.cpp/pull/31004) (Q52's first layer: the key extended by the first node's
shape) is not in the fork and is not needed for widths up to 8.

**What it costs, by construction.** Each arena holds `ggml_tensor_overhead() × max_nodes` plus the graph's own arrays,
with `max_nodes = max(40 × n_tokens, 32 × n_tensors)` for `qwen4exp`. With the default shared scheduler the extra
arenas cost only host memory, of which `calloc` keeps only the pages a graph reaches; device memory grows by the
captured CUDA graphs of each width. With `LLAMA_SCHED_PER_SIZE=1` every width also allocates its compute buffers on
each card, on first use (not at load): upstream measured ~400 MiB more VRAM for DFlash2 and MTP with up to 5 drafted
tokens on a GB300.

## Checks without a GPU

The rig build (`TESTS=1`, sm_61 + compute_61 PTX, CUDA 12.9 in Docker):
`llama.cpp-20261103-a476ab83f-pascal-p106-v3-b12076-vgraphs.tar.gz`, 346,403,082 bytes, md5
`8bb2ed8ebdec541ba2290e0e08632659`, no AVX instructions, highest glibc symbol GLIBC_2.38; `test-llama-archs`,
`test-llama-context` and `test-draft-mem` from the same tree in `extra-a476ab83f.tar.gz` (md5
`f9a177a988e2d2f06a096b8c4a4a4344`). Then a CPU-only build of the same commit ran every check, in containers without a
GPU ([checks.tsv](checks.tsv)):

| Check | Result |
|---|---|
| `test-backend-ops -b CPU` (every op of the CPU backend) | 25,247 of 25,247 passed |
| `test-opt` and `test-alloc` (they drive the scheduler whose scratch #31274 resizes, and the optimizer's allocation dependencies it moves to their own context) | both pass |
| `test-llama-context` on random-weight qwen4exp (MoE), qwen35 and mamba models, every case, at the defaults, `LLAMA_GRAPH_PER_SIZE=0`, `LLAMA_SCHED_PER_SIZE=1` and `LLAMA_GRAPH_META_CALLOC=0` ([context_runs.tsv](context_runs.tsv), [context.tsv](context.tsv)) | the same at every setting: 54 passed, 0 failed, 6 skipped (qwen4exp 19 / 0 / 1, qwen35 19 / 0 / 1, mamba 16 / 0 / 4) |
| `test-llama-archs -b CPU`, every architecture, at the defaults and with `LLAMA_SCHED_PER_SIZE=1` | 148 of 148 passed, both |
| llama-server's speculative decoding, the random qwen4exp drafting for itself, five requests, five arms ([run/spec_check.sh](run/spec_check.sh); [spec_verdict.txt](spec_verdict.txt), [spec_requests.tsv](spec_requests.tsv), [spec_widths.tsv](spec_widths.tsv)) | PASS: `gps0`, `sps1`, `calloc0` and `noreuse` give `base`'s tokens and top-3 probabilities on every request; 295 verify steps at widths 1 (3 steps), 2 (116), 3 (11) and 4 (165), the same in every arm; `sps1` created 8 schedulers (widths 1–4, in the target's context and in the draft's) |
| C22's checkpoint arms on llama-server ([../2026-11-04_ckpt_cards/run/server_check.sh](../2026-11-04_ckpt_cards/run/server_check.sh); [ckpt_verdict.txt](ckpt_verdict.txt), [ckpt_requests.tsv](ckpt_requests.tsv)) | PASS: every device-store arm gives the host arm's tokens and top probabilities, the spread arm resumes from its checkpoint at 2,048 tokens |

**Host memory on the CPU** ([spec_memory.tsv](spec_memory.tsv)): llama-server's resident memory after the five requests,
with the random qwen4exp as both target and draft. Here the compute buffers are host memory too, so `sps1`'s extra
schedulers show up in it; on the rig they are on the cards.

| Arm | resident (MiB) | anonymous (MiB) | anonymous against `base` (MiB) |
|---|---:|---:|---:|
| `base` (defaults) | 91.8 | 43.1 | — |
| `gps0` (`LLAMA_GRAPH_PER_SIZE=0`) | 88.3 | 39.5 | −3.5 |
| `sps1` (`LLAMA_SCHED_PER_SIZE=1`) | 100.0 | 51.2 | +8.1 |
| `calloc0` (`LLAMA_GRAPH_META_CALLOC=0`) | 112.4 | 63.6 | +20.5 |
| `noreuse` (`LLAMA_GRAPH_REUSE_DISABLE=1`) | 92.1 | 43.1 | +0.1 |

On this model the eight extra arenas (widths 1–4 in two contexts) cost 3.5 MiB of pages their graphs touch, and
`calloc` keeps 20.5 MiB of the arenas' metadata out of memory. Both scale with `max_nodes`, which for Flash-Next is set by
its tensor count; the rig run measures them there.

## Next: the rig

[run/rig_vgraphs.sh](run/rig_vgraphs.sh) loads S18's DFlash2 line (`-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio
--no-host -md <drafter> --spec-type draft-dflash --spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99`,
[Q561](../2026-11-03_flash_next_dflash2/README.md)) once per arm with `GGML_CUDA_GRAPH_DEBUG=1` and `-lv 5`, sends
Q561's two smoke requests, and records each card's memory and the host's once loaded and after each request:

| Arm | Build | Settings |
|---|---|---|
| `ladder2` | `-ladder2` (`a330672c1`) | as it is: one arena for every width |
| `vg` | this build | defaults: an arena per width, one scheduler, `calloc` |
| `vg-gps0` | this build | `LLAMA_GRAPH_PER_SIZE=0`: `-ladder2`'s arenas with #31274's memory changes |
| `vg-sps1` | this build | `LLAMA_SCHED_PER_SIZE=1`: a scheduler per width |

[run/graph_counts.py](run/graph_counts.py) then counts, per arm and request, the CUDA graph events the debug log
prints (`warmup complete`: a capture; `warmup reset`: a captured graph changed, so the next steps run uncaptured;
`update failed`: a capture re-instantiated; `reused`: a reused graph launched without a check), the steps per verify
width and the width changes, and tabulates host and card memory. **What passing looks like:** on `ladder2` and
`vg-gps0` a capture or reset follows most width changes; on `vg` and `vg-sps1` captures stop once each width has been
seen; `vg` and `vg-gps0` hold more usable host memory than `ladder2` once loaded; `vg-sps1`'s cards show its extra
compute buffers, which have to fit beside the 54 GB of weights. The timed agent check (`bench/agent_check`) then takes
the arms that pay.

## Files

| File | What it holds |
|---|---|
| [checks.tsv](checks.tsv) | one line per CPU check: verdict and detail |
| [context_runs.tsv](context_runs.tsv) | `test-llama-context` per setting and model: the runner's passed, failed and skipped |
| [context.tsv](context.tsv) | `test-llama-context` per setting, model and case: verdict and reason (the recurrent cases twice, the second time against a pre-filled cache) |
| [spec_requests.tsv](spec_requests.tsv) | `spec_check.sh` per arm and request: prompt and generated tokens, drafted and accepted tokens, decode milliseconds, the generated token ids |
| [spec_widths.tsv](spec_widths.tsv) | `spec_check.sh` per arm: verify steps per width |
| [spec_memory.tsv](spec_memory.tsv) | `spec_check.sh` per arm: llama-server's resident, anonymous, file-backed and peak memory after the requests |
| [spec_verdict.txt](spec_verdict.txt) | `spec_check.sh`'s verdict: each arm against `base`, the widths, the per-size schedulers, each arm's settings line |
| [ckpt_requests.tsv](ckpt_requests.tsv), [ckpt_verdict.txt](ckpt_verdict.txt) | C22's checkpoint arms on this build |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the rig build, the CPU build and every CPU check above |
| [run/spec_check.sh](run/spec_check.sh) | llama-server's speculative decoding with each switch, on the CPU |
| [run/rig_vgraphs.sh](run/rig_vgraphs.sh) | the rig run: S18's DFlash2 line per arm with the graph debug log, memory per stage |
| [run/graph_counts.py](run/graph_counts.py) | the rig run's tables: CUDA graph events and verify widths per request, host and card memory, the memory breakdown, the replies |

## Commands

```bash
# the build machine (Docker, no GPU): WORK holds the llama.cpp clone with the fork as remote "fork" (https)
WORK=<dir> REPO=<this repository> JOBS=6 THREADS=8 LOCK=<shared lock file> DOCKER_ARGS="<docker run labels>" \
  bash docs/data/2026-11-04_vgraphs/run/cpu_checks.sh
# the rig: both builds unpacked into /opt/llama-builds, the rig quiet; from this repository's root
RIG=<ssh alias> BENCH=bench/agent_check MODEL=<Swift-1.5 GSQ-RCO Q2_0, first shard> DFLASH=<DFlash2 Q8_0 drafter> \
  RIG_PORT=<port> LOGDIR=<folder on the rig> GUARD=docs/data/2026-11-03_flash_next_mtp/run/mem_guard.sh \
  BIN_ladder2=/opt/llama-builds/llama.cpp-20261103-a330672c1-pascal-p106-v3-b12076-ladder2/bin/llama-server \
  BIN_vg=/opt/llama-builds/llama.cpp-20261103-a476ab83f-pascal-p106-v3-b12076-vgraphs/bin/llama-server \
  bash docs/data/2026-11-04_vgraphs/run/rig_vgraphs.sh out/vgraphs
# the tables, where the files are
docker run --rm -v "$PWD/out/vgraphs:/d" -v "$PWD/docs/data/2026-11-04_vgraphs/run:/r:ro" python:3.12-slim-bookworm \
  python3 -I /r/graph_counts.py /d
```

## Provenance and licence

The ported code is upstream llama.cpp's (MIT): #31156 by Aman Gupta, #31274 by Joel Fernandes, read from the pull
requests' heads on 2026-11-04 (`7d229e479`; `9f084f355`, `8337f4fa1`). The random-weight test models come from the
fork's `test-llama-archs`; no model file or text dataset is used. The tables hold test verdicts, token ids of random
models, counts and memory figures.
