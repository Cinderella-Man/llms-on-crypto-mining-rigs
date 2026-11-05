# C200 / C201: cheaper hops between cards, the residual in F16 or Q8_0 and a staged copy through pinned slots (2026-11-04)

On the nine-P106 rig every decoded Flash-Next token crosses eight card boundaries in a layer split, and each crossing
carries the residual: four hyper-connection streams of 2,560 FP32 values, 40,960 bytes. The cards have no peer access,
so every hop goes card → host → card over two PCIe 1.0 x1 links at ~200 MB/s each: 0.41 ms of bytes alone, an
estimated 0.5–0.9 ms with the driver's fixed part, 4–7 ms of a ~121 ms token; a 4-token verify pass carries four
times the bytes, and a 512-token prompt micro-batch 21 MB. [The card-hops note](../../research/notes/card_hops_x1.md)
ranks what makes this cheaper; this folder holds the engine work for two of its ideas, both behind switches that are
off by default, the checks that ran without a GPU, and the scripts that measure them on GPUs and on the rig:

- **Fewer bytes per hop** (the note's D1, D3 and P2; registry Q750, Q231): the residual crosses each boundary in Q8_0
  (10,880 bytes a token, est. −2.4 ms a token) or F16 (20,480 bytes, est. −1.6 ms), cast on the card before the
  boundary and back to F32 on the card after it.
- **A manual staged hop** (D4 and P3; Q751, Q755, and the destination half of Q284's copy stream): the copy between
  two cards without peer access goes through a small pinned ring of its own in two chunks, so the second chunk's
  card-to-host leg overlaps the first chunk's host-to-card leg, in place of the driver's own staging of
  `cudaMemcpyPeerAsync`.

The registry rows are Q750, Q231, Q751, Q755 and Q284 in [quants/experiments.md](../../../quants/experiments.md), with
the new Q756–Q759.

**Status:** written, built for the rig and checked on the CPU; the KL divergence of the casts passed (C200k) and the rig
ran the hop bench and all five server lines (C200r): **the Q8_0 cast is the hop setting for the release line**
(`LLAMA_HOP_CAST=q8_0`, decode +11–15 %, prompts +21–32 %); the staged hop adds nothing measurable on top of it (a
negligible gain, kept behind its switch), but it is a gain where the hops stay F32. The results are below. Still to
come: the two-card check ([run/gpu_check.sh](run/gpu_check.sh)) and the agent-check arm of the cast.

## The build

Fork branch `pascal-p106-v3-b12076-hops`, two commits on `pascal-p106-v3-b12076-ladder2` (`a330672c1`,
[the merged engine](../2026-11-04_ladder2_merge/README.md)):

| Commit | What it does |
|---|---|
| `8ff12e276` | `LLAMA_HOP_CAST` (the casts at card boundaries, the graph builder's `build_hop_cast`, called by `qwen4exp`), `GGML_CUDA_HOP_*` (the staged copy in the CUDA backend's `cpy_tensor_async`), `tests/test-hop-cast.cpp`, `test-backend-ops` copy cases at the hop shapes |
| `5b69b0574` | `test-hop-cast`: `-ts` for a layer split over several cards; per boundary the source's range and the round trip's error |

## The switches

Environment variables of llama-server and every llama.cpp tool. With none of them set, the graph and the copies are
those of `-ladder2`, node for node and byte for byte.

| Switch | Default | Values |
|---|---|---|
| `LLAMA_HOP_CAST` | off | `f16`, `bf16` or `q8_0`: the residual crosses every boundary between two GPUs of a layer split in that type. Unset, empty, `0`, `off` or `f32`: off |
| `LLAMA_HOP_CAST_AT` | the layer split's boundaries | a comma-separated list of layers (each the first layer after a boundary): the round trip happens before those layers instead, whatever the devices; for a rig's boundaries in a run on fewer cards or on the CPU |
| `GGML_CUDA_HOP_STAGED` | 0 | 1: the staged hop between two cards without peer access; 2: between any two cards (for checks on boards with peer access). 0: `cudaMemcpyPeerAsync`, as before |
| `GGML_CUDA_HOP_CHUNKS` | 2 | chunks per hop, 1–16; a hop larger than that many slots takes more chunks |
| `GGML_CUDA_HOP_SLOTS` | 4 | pinned host slots per boundary, 2–64 |
| `GGML_CUDA_HOP_SLOT_KB` | 256 | KiB per slot, 4–65,536. A boundary holds slots × KiB of pinned host memory, allocated on its first hop: 1 MiB by default, 8 MiB for nine cards |
| `GGML_CUDA_HOP_STREAM` | 1 | where the host-to-card leg runs. 0: on the destination's compute stream; 1: there for hops that fit the ring (decode, verify passes) and on a copy stream of the destination for larger ones (prompt micro-batches); 2: on the copy stream always |

At context creation `LLAMA_HOP_CAST` logs `card hops in <type> before layers <list> (the layer split |
LLAMA_HOP_CAST_AT)`. Each boundary of the staged hop logs `card hop CUDAi -> CUDAj staged through N pinned slots of K
KiB, C chunks, copy stream mode M` when its ring is made, and its counts (hops, MiB, chunks, hops on the copy stream)
when the context is freed.

## Where the cast goes, and why there

The residual could be narrowed in three places. **The graph builder was chosen** (`llm_graph_context::build_hop_cast`,
`src/llama-graph.cpp`):

1. **The graph builder (chosen).** At the top of each layer that `cparams.hop_layer` marks (the first layer on a new
   card), the residual gets two ordinary `CPY` nodes: F32 → narrow type, then back to F32. The narrow node is named
   `hop_narrow-<il-1>` and the wide one `hop_wide-<il>`, and llama-context's graph callback pins each to its layer's
   device, the mechanism that already keeps `norm` and `l_last` on their layer's card. The scheduler then finds a
   Q8_0 (or F16) tensor as the next card's input and copies that: 10,880 bytes a token instead of 40,960. Nothing in
   the scheduler or the backends changes; the casts run as two small kernels on the two cards (inside their CUDA
   graphs at decode), with the fork's existing `CPY` kernels (F32 ↔ F16, BF16, Q8_0). They act on every token row
   alike, so decode, k-token verify passes and prompt micro-batches take the same path. Because the boundaries are
   data, `LLAMA_HOP_CAST_AT` can put the rig's eight boundaries into a KL-divergence run on a machine with another
   split, or into a CPU run. The cost: one line in each architecture's layer loop; `qwen4exp` (Flash-Next) has it,
   the others are Q756.
2. **The scheduler** (`ggml-backend.cpp`, where it copies a split's inputs from other backends). It would have to add a
   cast node to the end of one split's graph and another to the start of the next, and it copies every tensor a split
   reads from another device, not only the residual: it cannot tell which may be rounded.
3. **The CUDA backend's copy** (`ggml_backend_cuda_cpy_tensor_async`). Transparent to everything above it, but it would
   round every F32 tensor that any copy between two cards carries, it needs device scratch for the narrow form on
   both cards, and it exists only where two CUDA cards do: no CPU check and no KLD with the rig's boundaries elsewhere.

The staged hop, by contrast, belongs in the CUDA backend's copy: it changes how bytes move, not which bytes.

## The staged hop

Per boundary (source card → destination card) the destination's backend context keeps a ring of `GGML_CUDA_HOP_SLOTS`
pinned, portable host slots of `GGML_CUDA_HOP_SLOT_KB`, an event per slot on each card, and (on first use) a copy stream
on the destination. A hop of B bytes is cut into chunks of ⌈B / chunks⌉ rounded up to 256 bytes (at most a slot); per
chunk, in the next slot of the ring:

1. the source card's compute stream waits on the slot's "drained" event (its previous host-to-card copy is done), then
   copies the chunk card → slot and records the slot's "filled" event;
2. the destination's stream waits on "filled", copies slot → card and records "drained".

The card-to-host leg stays on the source card's compute stream, as the driver's copy did, so the source's next
micro-batch cannot overwrite the residual before it has left; with two chunks the second chunk's first leg overlaps the
first chunk's second leg (est. 0.31 ms instead of 0.41 for 40 KB). A hop that needs more chunks than the ring has slots
(a prompt micro-batch: 21 MB in F32, 5.6 MB in Q8_0) puts its host-to-card legs on the destination's copy stream
(`GGML_CUDA_HOP_STREAM=1`): on the compute stream they would queue behind the destination's current micro-batch, and the
source's next card-to-host leg would wait for them through the slot events. On the copy stream they first wait on the
event the scheduler made the destination's compute stream wait on just before the copy (`ggml_backend_cuda_event_wait`
keeps it as the backend's `hop_guard`): the guard that no earlier micro-batch still reads that input copy. The
compute stream then waits on the copy stream's last leg. Afterwards the destination's stream is ordered after the
copy, as the scheduler expects of `cpy_tensor_async`.

Pinned memory: 1 MiB per boundary by default, 8 MiB for nine cards; `GGML_CUDA_HOP_SLOT_KB=64` makes it 2 MiB on a
host with little to spare. A decoded token's hop (10,880–40,960 bytes) and a 4-token pass's (≤ 164 KB) fit the default
ring in two chunks.

## The checks without a GPU

The build machine ran [run/cpu_checks.sh](run/cpu_checks.sh) in containers without a GPU: the rig tarball, the test
binaries from its tree, CPU-only builds of the branch and of its base `a330672c1`, the hop microbenchmark, and these
checks ([checks.tsv](checks.tsv)). Every check passes:

| Check | Result |
|---|---|
| Rig build | `llama.cpp-20261104-5b69b0574-pascal-p106-v3-b12076-hops.tar.gz`: 346,398,437 bytes, md5 `430948087084794d9e935308a88e348f`, sm_61 code and compute_61 PTX, no AVX, GLIBC_2.38; the test binaries from its tree in `extra-5b69b0574.tar.gz` (md5 `516ca5a265a3c19009f5a829c4a36ac4`) |
| `test-backend-ops -b CPU -o CPY` | 555 / 555, the 18 new cases at the hop shapes among them ([ops_cpu.txt](ops_cpu.txt)) |
| Round trips (`test-hop-cast --unit-only`) | F16, BF16 and Q8_0 at T = 1, 4 and 512: data the type represents comes back bit for bit; residual-like data (normal, with channels at 8× and 60×) stays within each type's bound, at worst 1.000, 0.996 and 0.970 of it ([unit.txt](unit.txt)) |
| The model (`test-hop-cast`, random qwen4exp) | the reference twice, the staged hop, and F16 and Q8_0 without a boundary (the CPU has none): the same logits bit for bit. The round trip before layer 1: max \|Δlogit\| 3.5e-5 (F16), 2.6e-4 (BF16), 8.1e-4 (Q8_0); mean KLD 3.2e-11, 2.1e-9, 2.2e-8; top-1 the same at all 58 positions; each of the 10 hops per arm (three prompt micro-batches, the 4-token pass, six single steps) within the bound, with an error of 2.1e-4, 1.7e-3 and 5.4e-3 of the residual's rms; layer 0's output unchanged in every arm. 55 of 55 checks ([model.txt](model.txt), [model.tsv](model.tsv)) |
| Identity (`test-llama-archs -b CPU`, every architecture) | with the switches off all 136 architectures' logits equal the base build's bit for bit; with `GGML_CUDA_HOP_STAGED=1 LLAMA_HOP_CAST=q8_0` (no boundary on the CPU) the same; with `LLAMA_HOP_CAST=q8_0 LLAMA_HOP_CAST_AT=1` only qwen4exp changes (nmse 2.7e-5, max difference 8.5e-4). The 7 failures in each run's summary are the test's own: it builds a second model for 7 families (llama, granite and others) whose logits file has the first model's name, so even the base run compares the two; every run repeats the base's numbers for them exactly ([identity.tsv](identity.tsv), [identity_summary.txt](identity_summary.txt)) |
| `test-llama-context` | the random qwen4exp: 19 passed, 1 skipped, 0 failed at the default and with the Q8_0 round trip before layer 1 ([ctx.txt](ctx.txt)) |
| llama-server | speculative decoding with verify passes of 1–4 tokens (392 tokens drafted) and prompts in 32-token micro-batches: `off`, `staged` and `q8auto` give the base build's tokens and top-3 probabilities in all four requests; `q8at1` and `f16at1` log the round trip before layer 1 in the target's and the draft's contexts and complete every request, with tokens that differ from the base's (a random model's near-tied logits) ([server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt)) |
| Hop microbenchmark | `bench/megakernel_p106` with `--hop-chunks` builds (md5 `f47c545dd6b0f9f9d6fe194e16877ae7`, sm_61, compute_61 PTX, sm_86, no AVX) |
| RTX 30 build (`STEPS=sm86`) | the same commit for compute capability 8.6: `llama-perplexity`, `test-backend-ops`, `test-hop-cast`, `test-llama-archs` |

What the CPU cannot check: the staged hop itself (it runs only between two CUDA cards; the checks above show it
changes nothing where it is not used), the casts' CUDA kernels at the hop shapes (the same `CPY` kernels as before;
`test-backend-ops` on a card covers them), and the cost of either.

## Next: two cards, the KLD, the rig

1. **Two cards** ([run/gpu_check.sh](run/gpu_check.sh), any two cards of a build machine, ~30 min with the JIT of the
   rig build's PTX): `test-backend-ops -o CPY` on each card; the round trips on both; `test-hop-cast` on the random
   qwen4exp split over the two cards (`-ts 1,3`: one boundary, before layer 1), every staged-hop arm (1, 2 and 4
   chunks, 4 KiB slots so a prompt micro-batch wraps the ring on the copy stream, the copy stream always and never)
   bit for bit the driver copy's logits, the casts within bound; `test-llama-archs` on every architecture with and
   without the staged hop; optionally a real model's perplexity, which must not change.
2. **The KLD** ([run/kld_hops.sh](run/kld_hops.sh), all seven cards, ~45 min an arm): Swift-1.5 Flash-Next Q8_0 with
   the round trip at the rig's eight boundaries (`LLAMA_HOP_CAST_AT=6,11,16,22,27,32,38,43`, the layers of
   `-ts 6,5,5,6,5,5,6,5,6`) in F16 and in Q8_0, against the quality chain's bases
   ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)), with its `kld.sh` unchanged; the arm
   `off` (the switches off) is this build's floor against the bases' build. Gate: within noise of `off`. `test-hop-cast`
   on the same file gives each boundary's range (F16 stops at 65,504) and round-trip error (Q757).
3. **The rig** ([run/rig_hops.sh](run/rig_hops.sh), ~1.5 h, nothing else running): the hop microbenchmark between
   four pairs of cards (`bench/megakernel_p106 --hop`, now with the manual staged hop in 1, 2 and 4 chunks and the
   switches' sizes: 10,880 / 20,480 / 40,960 bytes a token, 43,520 / 81,920 / 163,840 a 4-token pass), then one
   llama-server line per arm (no switch, Q8_0, F16, the staged hop, both) with decode, ~1k / 8k / 32k prompts and two
   identity prompts whose hash must agree between the arms that copy the same bytes. Then the agent check, one arm
   (S18's DFlash2 line on this build with the switches the lines favour) against S18's 9.5 tok/s.

## Files

| File | What it is |
|---|---|
| [checks.tsv](checks.tsv) | one line per CPU check: verdict and detail |
| [ops_cpu.txt](ops_cpu.txt) | the hop-shape `CPY` cases of `test-backend-ops -b CPU` and its totals |
| [unit.txt](unit.txt) | `test-hop-cast --unit-only` on the CPU: the round trips F32 → F16 / BF16 / Q8_0 → F32 at [2560, 4, T] |
| [model.tsv](model.tsv), [model.txt](model.txt) | `test-hop-cast` on the random qwen4exp, per arm: identity with the reference, logit differences, KLD, top-1 agreement, hops and their bound; per boundary the source's range and the error |
| [identity.tsv](identity.tsv), [identity_summary.txt](identity_summary.txt) | `test-llama-archs` logits of every architecture, the base's own lines and the branch's at three settings against them (`base_self` 1: a line the base run itself printed); each run's summary line |
| [ctx.txt](ctx.txt) | `test-llama-context` totals at the two settings |
| [server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt) | llama-server's speculative decoding, the base build against the branch's arms |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the build-and-check chain (steps 1–12) |
| [run/server_check.sh](run/server_check.sh) | the llama-server check |
| [run/gpu_check.sh](run/gpu_check.sh) | the two-card check |
| [run/kld_hops.sh](run/kld_hops.sh) | the KLD arms on the quality chain's bases |
| [run/rig_hops.sh](run/rig_hops.sh) | the rig run: the hop microbenchmark, then a server line per arm |
| [rig_hop_bench.txt](rig_hop_bench.txt) | C200r: the hop bench on the second rig, four card pairs, every size and transfer |
| [rig_lines.tsv](rig_lines.tsv) | C200r: the six server lines on the second rig (`f32`, `q8` and `q8b` (a second load of `q8`), `f16`, `staged`, `q8staged`), one row per line and request: the switches, start time, prompt and cached tokens, prompt tok/s, first token, decode tok/s, free host memory, the identity replies' hashes |
| [run/rig_lines_table.py](run/rig_lines_table.py) | builds `rig_lines.tsv` from the lines' `line-<arm>.out` files and prints the comparison against the first arm and the identity hashes |

The hop microbenchmark is [bench/megakernel_p106](../../../bench/megakernel_p106/README.md) (`--hop`, `--hop-chunks`).

## Commands

```bash
# build and CPU checks (a machine with Docker; WORK holds a llama.cpp clone with the fork as remote "fork")
WORK=<work dir> REPO=<this repository> JOBS=6 bash docs/data/2026-11-04_card_hops/run/cpu_checks.sh
WORK=<work dir> REPO=<this repository> STEPS=sm86 bash docs/data/2026-11-04_card_hops/run/cpu_checks.sh   # the RTX 30 build
# two cards (the rig tarball unpacked, a random qwen4exp from test-llama-archs -a '^qwen4exp$' -s 1 -o <dir>)
TREE=<unpacked build> MODEL=<qwen4exp gguf> OUT=<dir> GPUS=<a>,<b> bash docs/data/2026-11-04_card_hops/run/gpu_check.sh
# the KLD (the quality chain's environment: MOUNTS, WIKI, AGENT, CODE, TAG; BASES = its kld folder)
PPL=<sm86 build>/bin/llama-perplexity BASES=<bases> MODEL=<Q8_0 first shard> OUT=<dir> GPUS=0,1,2,3,4,5,6 \
  MOUNTS=... WIKI=... AGENT=... CODE=... TAG=hops bash docs/data/2026-11-04_card_hops/run/kld_hops.sh
# the rig (the bundle: megakernel_p106, rig_hops.sh, fn_line.sh, fn_requests.py), detached on the rig
BUILDS=<folder of builds> BUILD=<this build> MODEL=<Flash-Next first shard> setsid nohup bash rig_hops.sh > console.log 2>&1 < /dev/null &
# C200r's lines: the bench skipped (HOPS=), chosen arms, a kernel setting on every line, under the RAM watchdog
LOG=/dev/shm/rw.tsv PROC=llama-server MIN_AVAIL_MB=60 chrt -f 10 bash ram_watchdog.sh & w=$!
BUILDS=<folder of builds> BUILD=<-ladder3 build> MODEL=<pmix-q2-fast first shard> OUT=<dir> HOPS= ARMS="f16 staged q8staged" \
  ENV_ALL=GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 bash rig_hops.sh; kill $w
# the table (the lines' output files side by side; the first arm is the reference)
python3 run/rig_lines_table.py rig_lines.tsv f32=line-f32.out q8=line-q8.out q8b=line-q8b.out f16=line-f16.out \
  staged=line-staged.out q8staged=line-q8staged.out
```

Provenance: the fork's code and this folder's scripts are this project's; the random-weight models are generated by
`test-llama-archs`; no model was downloaded for this work.

## C200k: what the hop casts cost (2026-11-04)

Swift Q2_0 (the Q8_0 reference needs ~124 GB of VRAM, more than five cards hold), FN1c's Swift-1.5 Q8_0 bases,
`LLAMA_HOP_CAST_AT=6,11,16,22,27,32,38,43` (the rig's nine-card split), two RTX 3090s, each arm paired against its own
`off` on the same cards ([`kld_hops_q2_0.tsv`](kld_hops_q2_0.tsv)):

| Cast | agent Δ KLD | wiki Δ KLD | code Δ KLD |
|---|---|---|---|
| f16 | −0.0001 ± 0.0012 | +0.0002 ± 0.0006 | −0.0007 ± 0.0004 |
| q8_0 | −0.0011 ± 0.0012 | +0.0002 ± 0.0006 | −0.0005 ± 0.0004 |

**Passed:** neither cast moves the KLD beyond its paired noise, so the Q8_0 hop cast goes on the release line (S35).

## C200r: the hop bench and the server lines on a rig (2026-11-04)

The second rig (nine P106-090, `kho=off`), from the bundle's `run/rig_hops.sh` with the lines on Miner Llama's
`-ladder3` (it holds this branch) and pascal-mix's pmix-q2-fast, `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768` on every line
(`ENV_ALL`, as the day's agent arms ran), S12's no-draft split `-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off`, under
the RAM watchdog (floor 60 MiB). Only the `f32` and `q8` lines ran: the rig was switched off for the night before `f16`,
`staged`, `q8staged` and the timed agent arm.

**The hop bench** ([rig_hop_bench.txt](rig_hop_bench.txt), four card pairs: behind one switch, across the switches, to
the card on its own port and back): every byte arrived in every size and chunk count, no mailbox timeouts. The manual
staged hop in two chunks takes 118.7 µs at 10,240 B against the driver's peer copy's 144.2, and 357.5 against 455.8 µs
at 40,960 B (cards 0 → 1); in four chunks it is faster still from ~20 KB up (317.3 µs at 40,960 B, 1,119 against 1,702
at 163,840 B), as the first rig showed.

**The lines** ([rig_lines.tsv](rig_lines.tsv)), generation and prompt speed per request, `f32` (no switch) against `q8`
(`LLAMA_HOP_CAST=q8_0`):

| Request | Prompt tokens | Prompt tok/s, f32 → q8 | Decode tok/s, f32 → q8 |
|---|---:|---|---|
| short | 65 | 12.0 → 12.6 | 14.5 → 16.2 (+12 %) |
| ~1k | 1,382 | 61.0 → 74.5 (+22 %) | 16.2 → 18.3 (+13 %) |
| ~8k | 7,183 | 140.6 → 184.1 (+31 %) | 13.7 → 15.8 (+15 %) |
| ~32k | 32,674 | 155.2 → 189.0 (+22 %) | 9.2 → 10.1 (+10 %) |
| ~31k from a checkpoint | 31,161 | 156.7 → 192.2 (+23 %) | 9.8 → 10.8 (+10 %) |

**The Q8_0 hop cast gives +10–15 % decode and +22–31 % prompt speed** on this line, at no measurable KLD (C200k above).
Each boundary crosses the PCIe 1.0 x1 links in a quarter of F32's bytes, and prompts carry the most bytes per hop. The
identity hashes of `f32` and `q8` differ, as a cast's must. The agent check of the cast is the next step.

## C200r: the other three lines and a second Q8_0 load (2026-11-05)

The same rig, build, model, line and watchdog as above, the next morning: the lines `f16`, `staged` and `q8staged`, then
`q8` again (`q8b`), so that two loads of one setting a day apart give the spread. The hop bench was not repeated.
Every line ran to its end, and the watchdog stopped nothing. [rig_lines.tsv](rig_lines.tsv) holds all six lines, one row
per line and request ([run/rig_lines_table.py](run/rig_lines_table.py)).

**Decode tok/s** (the change against `f32`):

| Request | Prompt tokens | f32 | q8 | q8b | f16 | staged | q8staged |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 65 | 14.5 | 16.2 (+12 %) | 16.1 (+11 %) | 15.5 (+7 %) | 15.2 (+5 %) | 15.9 (+9 %) |
| ~1k | 1,382 | 16.2 | 18.2 (+13 %) | 18.2 (+12 %) | 17.9 (+11 %) | 16.9 (+5 %) | 18.2 (+13 %) |
| ~8k | 7,183 | 13.7 | 15.8 (+15 %) | 15.6 (+14 %) | 15.5 (+13 %) | 15.1 (+10 %) | 15.8 (+15 %) |
| ~32k | 32,674 | 9.2 | 10.1 (+10 %) | 10.6 (+16 %) | 10.6 (+16 %) | 10.0 (+9 %) | 10.6 (+15 %) |
| ~31k from a checkpoint | 31,161 | 9.8 | 10.8 (+10 %) | 9.6 (−2 %) | 9.5 (−3 %) | 10.2 (+4 %) | 10.7 (+10 %) |

**Prompt tok/s** (the change against `f32`):

| Request | Prompt tokens | f32 | q8 | q8b | f16 | staged | q8staged |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 65 | 12.0 | 12.6 (+5 %) | 12.8 (+7 %) | 12.6 (+5 %) | 12.3 (+2 %) | 12.8 (+7 %) |
| ~1k | 1,382 | 61.0 | 74.5 (+22 %) | 73.5 (+21 %) | 72.8 (+19 %) | 61.7 (+1 %) | 75.2 (+23 %) |
| ~8k | 7,183 | 140.6 | 184.1 (+31 %) | 185.6 (+32 %) | 180.4 (+28 %) | 144.8 (+3 %) | 186.7 (+33 %) |
| ~32k | 32,674 | 155.2 | 189.0 (+22 %) | 189.5 (+22 %) | 186.3 (+20 %) | 158.6 (+2 %) | 191.5 (+23 %) |
| ~31k from a checkpoint | 31,161 | 156.7 | 192.2 (+23 %) | 189.2 (+21 %) | 188.2 (+20 %) | 158.4 (+1 %) | 193.7 (+24 %) |

**The identity replies agree where they must:** `staged` gives `f32`'s hashes (`4819c10f6138`, `3882b483f7c4`), and
`q8staged` and both `q8` loads give the same pair (`146f0d55ff53`, `470c86f69d04`), so the staged hop moves the same
bytes as the driver's copy. `f16` has its own pair (`9c8cb9974778`, `8033b769c24a`), as a different cast must.

**The spread between two loads** (`q8` against `q8b`, a day apart, the same tokens): 0.2–1.6 % in prompt speed on
every request, and 0.3–1.3 % in decode at the short, ~1k and ~8k requests. Decode at ~32k and after the restore (64–128
tokens at ~10 tok/s) and on the two identity replies (192 tokens) differs by 4–12 % between the two loads, too much to
class a change of a few per cent, so the classes below use the short, ~1k and ~8k decodes and all five prompts.

| Setting | Against | Decode (short, ~1k, ~8k) | Prompts (five requests) | Class |
|---|---|---|---|---|
| `LLAMA_HOP_CAST=q8_0` (`q8`, `q8b`) | `f32` | +11 to +15 % | +21 to +32 % (the four long ones) | **gain**: the release line's hop setting |
| `GGML_CUDA_HOP_STAGED=1` on Q8_0 hops (`q8staged`) | the mean of `q8` and `q8b` | −1.6, +0.1, +0.1 % | +0.7 to +1.6 %, the same sign on all five | **negligible**: the size of the spread, not slower |
| `GGML_CUDA_HOP_STAGED=1` on F32 hops (`staged`) | `f32` | +4.5, +4.8, +9.8 % | +1.1 to +3.0 % | **gain** where the hops stay F32 |
| `LLAMA_HOP_CAST=f16` (`f16`) | the mean of `q8` and `q8b` | −4.2, −1.9, −1.4 %, below both loads | −0.7 to −2.4 %, below both loads on four of five | **loss** against Q8_0 (a gain against F32: decode +7 to +13 %, prompts +19 to +28 %) |

- **The hop setting for the stacked line is `LLAMA_HOP_CAST=q8_0`.** Both casts passed C200k's KLD, and Q8_0 carries
  half of F16's bytes and is faster on every request that the spread lets us read.
- **The staged hop adds nothing measurable over the Q8_0 cast alone.** Prompts run 0.7–1.6 % faster, with the right sign
  on all five, but that is no more than two loads of the same line differ; decode is unchanged. A Q8_0 hop is 10,880
  bytes, where the bench saves 25 µs a hop (118.7 against 144.2 µs at 10,240 B): 0.2 ms over eight boundaries, 0.3 % of a
  ~62 ms token. It stays behind its switch as a negligible gain, to be tested with the others as one group.
- **On F32 hops the staged hop is a clear gain:** decode +4.5–9.8 %, which is 2.8–6.5 ms a token. The hop bench
  predicted only ~0.8 ms (eight boundaries at ~100 µs saved each on 40,960 B), so most of what the driver's peer copy
  costs in the server is on the host (its staging on the two Celeron cores), not on the links. This applies to every
  model whose hops stay F32: Flint, Qwen3.6 and Ornith until their layer loops get the cast.
- `f32` and `q8` ran on the evening of 2026-11-04 (models loaded in 457–459 s); `q8b`, `f16`, `staged` and `q8staged`
  ran on the morning of 2026-11-05, an hour after a boot (396–399 s). The two `q8` loads span that change, so the spread
  above includes it.
- The switches' own log lines (`card hops in …`, `card hop … staged through …`) do not reach llama-server's log at its
  default verbosity, so the `hops-<arm>.txt` files the script writes are empty; the identity hashes and the speeds show
  which path ran.
