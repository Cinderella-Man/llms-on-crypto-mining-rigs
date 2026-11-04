# C143: card hops through a mailbox in mapped host memory (Q682, Q270), 2026-11-04

On the nine-P106 rig every decoded Flash-Next token crosses eight card boundaries in a layer split. The cards have no
peer access, so each hop goes card → host → card over two PCIe 1.0 x1 links. Until now the host moved every hop: the
driver's `cudaMemcpyPeerAsync`, or the fork's staged hop (`GGML_CUDA_HOP_STAGED`,
[2026-11-04_card_hops](../2026-11-04_card_hops/README.md)), which enqueues per chunk a card-to-host copy, an event, a
host-to-card copy and another event. The staged hop with the Q8_0 cast took the first rig's decode from 14.98 to
17.06 tok/s (S35). That is −8.1 ms a token, three times what the bytes alone predicted, so the host's part of a hop
counts.

The mailbox takes the host out of the copy path, and the result stays exact. A hop becomes one small kernel on each
card. The source card's kernel writes the tensor chunk by chunk into a ring of slots in pinned host memory that both
cards map, and raises a sequence-number flag after each chunk. The destination card's kernel polls the flags, reads each
chunk straight from the ring as soon as it arrives, and acknowledges the slot. The destination reads chunk c over its
own link while the source writes chunk c + 1 over its link. There is no `cudaMemcpy`, no event and no host wait per
hop.

C140's microbenchmark ([2026-11-04_megakernel_rig](../2026-11-04_megakernel_rig/README.md)) had timed a one-shot
mailbox: the whole hop written, then one flag, then the whole hop read. It took as long as two host-staged copies
(436 µs for 40 KB), because the two links still ran one after the other. Streaming in chunks with a flag per chunk is
what this build adds; the hop bench now times it beside the staged hop.

The registry rows are Q682 and Q270 in [quants/experiments.md](../../../quants/experiments.md), with the new Q965 and
Q966.

**Status:** built for the rig and checked, with the results below. On a CPU every check passes. On two RTX 3090s
running the rig build's compute 6.1 PTX, the bytes arrive exactly in nine ring settings. Every mailbox setting gives
the driver copy's logits bit for bit, and the Q8_0 cast through the mailbox gives the Q8_0 cast through the staged
hop's logits bit for bit. The rig run ([run/rig_mailbox.sh](run/rig_mailbox.sh), C143r) is next: it measures what the
mailbox saves.

## The build

Fork branch `pascal-p106-v3-b12076-mailbox`, one commit on `pascal-p106-v3-b12076-kshape` (`d27845b67`,
[2026-11-04_kshape](../2026-11-04_kshape/README.md)):

| Commit | What it does |
|---|---|
| `5aebbddd9` | `ggml/src/ggml-cuda/hop-mailbox.cu` / `.cuh` (the ring, the two kernels, the host side), the branch in `ggml_backend_cuda_cpy_tensor_async`, the mailbox fields of the CUDA backend context, `tests/test-hop-mailbox.cpp` |

## The switches

Environment variables of llama-server and every llama.cpp tool. With `GGML_CUDA_HOP_MAILBOX` unset the code is never
reached and every copy is the base's.

| Switch | Default | Values |
|---|---|---|
| `GGML_CUDA_HOP_MAILBOX` | 0 | 1: copies between two cards without peer access go by mailbox; 2: also between cards with peer access (for checks on boards that have it). It comes before `GGML_CUDA_HOP_STAGED`: the copies it does not take use the staged hop or the peer copy as before |
| `GGML_CUDA_HOP_MAILBOX_CHUNKS` | 8 | chunks per hop, 1–256; a chunk is at least 256 bytes and at most a slot |
| `GGML_CUDA_HOP_MAILBOX_SLOTS` | 16 | slots per boundary, a power of two, 2–256 |
| `GGML_CUDA_HOP_MAILBOX_SLOT_KB` | 32 | KiB per slot, 1–4,096. A boundary holds slots × KiB of pinned, mapped host memory, allocated on its first hop: 512 KiB by default, 4 MiB for nine cards |
| `GGML_CUDA_HOP_MAILBOX_MAX_KB` | 0 (no limit) | hops larger than this take the staged hop or the peer copy, e.g. 256: decode and verify hops by mailbox, prompt micro-batches staged |
| `GGML_CUDA_HOP_MAILBOX_STREAM` | 1 | where the reading kernel runs. 0: on the destination's compute stream; 1: there for hops that fit the ring (decode, verify passes) and on a copy stream of the destination for larger ones (prompt micro-batches); 2: on the copy stream always |
| `GGML_CUDA_HOP_MAILBOX_THREADS` | 256 | threads of each kernel, 64–1,024 |
| `GGML_CUDA_HOP_MAILBOX_POLL_US` | 4 | the longest pause between two polls of a flag, in µs |
| `GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS` | 60,000 | a kernel that waits longer traps, so the process stops instead of computing with stale data |

Copies the mailbox does not take: sizes or addresses that are not multiples of 16 bytes, and hops above `_MAX_KB`.
Every residual the hop casts produce qualifies (F32, F16, BF16 and Q8_0 at 2,560 × 4 values a token, and the Q6_K
and IQ4_NL casts of Q963). Each boundary logs `card hop CUDAi -> CUDAj by mailbox: N slots of K KiB in mapped host
memory, C chunks, T threads, polls at most P us apart, copy stream mode M` on its first hop, and its counts (hops, MiB,
chunks, hops on the copy stream) when the context is freed.

## How a hop goes

Per boundary (source card → destination card) the destination's backend context keeps a ring of `_SLOTS` slots of
`_SLOT_KB` in pinned host memory allocated portable and mapped, plus control words in the same kind of memory: per slot
a flag line (the sequence number of the chunk the slot holds, written by the source card) and an acknowledgement line
(the sequence number of the chunk last read from it, written by the destination), each in its own 128-byte line.
Sequence numbers count chunks per boundary from 1, chunk q lives in slot q mod slots, and they are compared by their
signed 32-bit difference, so they may wrap. A hop of B bytes is cut into ⌈B / chunks⌉ rounded up to 256 bytes (at most a
slot) and takes the next n sequence numbers. In `cpy_tensor_async`, where the scheduler copies a split's input from
another card, the host launches two one-block kernels and nothing else:

1. **The writer**, on the source card's compute stream, after the work that wrote the tensor. It first reads the
   acknowledgements of the hop's first lap of slots in one round trip (one thread per slot). For each chunk it waits,
   if needed, until the slot's previous chunk (q − slots) is acknowledged, copies the chunk from the card into the slot
   (16-byte stores over PCIe), runs `__threadfence_system()` in every thread, then one thread writes q into the slot's
   flag.
2. **The reader**, on the destination card's stream. One thread polls the flag of the next chunk. It polls once at
   once, then with pauses that double up to `_POLL_US`. When the flag reaches q, warp 0 reads the flags of the next
   up to 31 chunks in one round trip. The whole block then reads every chunk that has arrived straight from the ring,
   with loads that bypass the caches (`ld.global.cv`), and writes the destination tensor. After a barrier, so that all
   of those loads have returned, the reader writes each chunk's sequence number into its slot's acknowledgement.

The destination's stream is ordered after the copy because the reader runs on it, as the scheduler expects of
`cpy_tensor_async`. A hop larger than the ring (a prompt micro-batch: 21 MB in F32, 5.6 MB in Q8_0 at 512 tokens)
streams through the ring with the writer waiting on acknowledgements. Its reader runs on a copy stream of the destination
(`_STREAM=1`), so the source card is not held while the destination still computes the previous micro-batch. On the copy
stream the reader first waits on the event the scheduler made the destination's compute stream wait on just before the
copy: the guard that no earlier micro-batch still reads that input copy, the same guard the staged hop uses. The
compute stream then waits on the reader.

**Why it cannot deadlock or reuse a slot early.** Every wait points to work the host enqueued earlier. A reader waits
for its own writer, enqueued just before it. A writer waits for the reader of the slot's previous chunk, which belongs
to an earlier hop or to the same one. Stream order and events point backwards in enqueue order too. So the waits form
no cycle, and a kernel waiting on one card never waits for work queued behind it. A slot is written again only after
its acknowledgement, and the acknowledgement comes only after every load from the slot has returned. A flag rises only
after the fence that follows the chunk's stores, and posted PCIe writes from one card arrive in order. The bytes are
copied exactly, so logits through the mailbox must equal those through the peer copy or the staged hop bit for bit.

**What it costs.** Two kernel launches a hop on the host instead of the staged hop's per-chunk copies and events. The
waiting card keeps one block of one SM busy while it polls, and during decode a token's later cards wait most of the
time. They poll their links at most every `_POLL_US`. The card does not drop its clocks while it waits; Q966 is what
that costs in power. Pinned, mapped host memory: 512 KiB a boundary by default.

**Where it sits, and why there.** The mailbox changes how bytes move, not which bytes, so it belongs in the CUDA
backend's copy beside the staged hop. It is transparent to the scheduler and to every architecture, and it composes
with the hop casts of the graph builder (`LLAMA_HOP_CAST`), which only make the bytes fewer. Fusing the writer into the
source card's CUDA graph as its last node and the reader into the destination's graph as its first would remove the
two launches too; that needs the scheduler to tell the graphs which inputs cross a boundary (Q965).

## The checks

[run/build_checks.sh](run/build_checks.sh) ran on a build machine in containers without a GPU, and
[run/gpu_check.sh](run/gpu_check.sh) on two of its RTX 3090s with the rig build itself, whose compute 6.1 PTX the
driver compiles for those cards, so the checks run the code the rig runs ([checks.tsv](checks.tsv)). Every check
passes:

| Check | Result |
|---|---|
| Rig build | `llama.cpp-20261104-5aebbddd9-pascal-p106-v3-b12076-mailbox.tar.gz`: 515,770,046 bytes, md5 `0d49aa0d8a1dd74ed84e46480c325389`, sm_61 code and compute_61 PTX, no AVX, GLIBC_2.38; the test binaries from its tree in `extra-5aebbddd9.tar.gz` (md5 `fd490362ffcf9d72869024b81620d0a6`) |
| The kernels' code | `k_hop_mb_send` and `k_hop_mb_recv` in sm_61 code and compute_61 PTX: 18 and 21 registers, no local memory; `__threadfence_system()` is `MEMBAR.SYS`, the flag polls and the ring reads are `LDG.E.CV` (volatile, past the caches), the ring reads 128 bits wide ([sass.txt](sass.txt)) |
| CPU build | the same commit without CUDA; `test-hop-mailbox` finds no CUDA card and exits 0; `test-hop-cast --unit-only` 18 checks, 0 failed |
| Bytes (`test-hop-mailbox`, two cards) | 96 rounds a setting, A → B and A → B → A, at 16 B to 5.6 MB, offsets 0, 256 and 4 (the last left to the other paths), every few rounds a matrix product holding one card's stream: every round's bytes exact in all nine settings: the peer copy; the mailbox's defaults; 1 chunk; a ring of 4 slots of 1 KiB (41,016 chunks through 4 slots) with the reader on the copy stream and on the compute stream; the copy stream always; 64 threads without a poll pause; 1,024 threads, 3 chunks and a 50 µs pause; a 16 KiB limit beside the staged hop. The hops each context logged as made by mailbox equal the copies the mailbox takes (96 a setting, 42 with the limit); no timeout; 27 checks, 0 failed ([mailbox_gpu.txt](mailbox_gpu.txt)) |
| Logits, F32 hops (`test-hop-cast`, two cards) | a random qwen4exp split over the two cards (`-ts 1,3`, one boundary), a 48-token prompt in three micro-batches in flight, a 4-token pass and six single steps: the staged hop and seven mailbox arms (the defaults; 1 chunk; the 4 × 1 KiB ring on the copy stream and on the compute stream; the copy stream always; 64 threads without a poll pause; decode-sized hops by mailbox and the rest staged) give the driver copy's logits bit for bit at every position; 99 checks, 0 failed ([model_f32.tsv](model_f32.tsv)) |
| Logits, Q8_0 hops | against `LLAMA_HOP_CAST=q8_0` through the staged hop (S35's hops): the cast through the mailbox, at the defaults and through the 4 × 1 KiB ring, gives identical logits (largest difference 0, KLD 0, top-1 1.0000), each of the 10 hops within the Q8_0 bound; 70 checks, 0 failed ([model_q8.tsv](model_q8.tsv)) |
| The mailboxes in use | 18 made by the model runs' contexts, all used, 6 with prompt micro-batches through the copy stream behind the scheduler's guard |
| Hop bench | `bench/megakernel_p106 --hop` with the streamed mailbox (md5 `caef20da7b9d5908965e29a43e4408cf`): on the two RTX 3090s (their own sm_86 code: a correctness run, not the rig's times) every byte arrives at 10,880 B to 1 MiB in 1, 4, 8 and 16 chunks, and through a 4 × 4 KiB ring in 1 to 256 chunks with 64 threads and no poll pause; no timeout |

What these checks cannot show: the speed. On the RTX 3090s' fast links the cost of each chunk dominates: one chunk is
fastest at 10,880 bytes, and 16 chunks are slowest. On PCIe 1.0 x1 a 10 KB hop spends ~40 µs on each link, so the
overlap weighs more there, and the right chunk count is the rig's to find (the hop bench, then
`GGML_CUDA_HOP_MAILBOX_CHUNKS` in the lines). The
script's `archs` step (`test-llama-archs` on every architecture, the mailbox against the default to the last digit)
waits for C319: through the compute 6.1 PTX the sweep stops at its first Meta row, before the two-card rows, because
llama.cpp's all-reduce has no compute 6.1 code (Q879).

## Next: the rig

[run/rig_mailbox.sh](run/rig_mailbox.sh) (~1.5 h, nothing else running):

1. The hop bench between four pairs of cards: the streamed mailbox in 1, 4, 8 and 16 chunks beside the staged hop
   and the peer copy, at 10,880 B to 1 MiB.
2. S35's llama-bench line (decode tg128, and 1-, 2- and 4-token passes) with `f32`, `q8staged`, `q8mbox`, `q8mboxst`
   and `mbox`.
3. llama-server lines with decode, ~1k / 8k / 32k prompts and identity hashes for `q8staged`, `q8mbox` and
   `q8mboxst`. The mailbox arms must hash the same as `q8staged`.

The agent check of the best arm comes after, on the release line.

## Files

| File | What it is |
|---|---|
| [checks.tsv](checks.tsv) | one line per check: where it ran, verdict, detail |
| [sass.txt](sass.txt) | the two kernels in the rig build: names, registers, fences and volatile loads |
| [mailbox_gpu.txt](mailbox_gpu.txt) | `test-hop-mailbox` on two cards: each boundary's mailbox as made and its counts when freed, one line per setting |
| [model_f32.tsv](model_f32.tsv), [model_q8.tsv](model_q8.tsv) | `test-hop-cast` on two cards, per arm: identity with the reference, logit difference, KLD, top-1 agreement, hops and their bound |
| [run/build_checks.sh](run/build_checks.sh) | the build-and-check chain on a build machine (steps 1–5) |
| [run/gpu_check.sh](run/gpu_check.sh) | the two-card check |
| [run/rig_mailbox.sh](run/rig_mailbox.sh) | the rig run: the hop bench, llama-bench per arm, a server line per arm |

The hop microbenchmark is [bench/megakernel_p106](../../../bench/megakernel_p106/README.md) (`--hop`, `--hop-mbox`).

## Commands

```bash
# build and CPU checks (a machine with Docker; WORK holds a llama.cpp clone with the fork as remote "fork")
WORK=<work dir> REPO=<this repository> JOBS=8 bash docs/data/2026-11-04_mailbox_hops/run/build_checks.sh
# two cards (the rig tarball unpacked with the extra tests, a random qwen4exp from test-llama-archs -a '^qwen4exp$' -s 1 -o <dir>)
TREE=<unpacked build> MODEL=<qwen4exp gguf> OUT=<dir> GPUS=<a>,<b> bash docs/data/2026-11-04_mailbox_hops/run/gpu_check.sh
# the rig (the bundle: megakernel_p106, rig_mailbox.sh, fn_line.sh, fn_requests.py, ram_watchdog.sh), detached on the rig
BUILDS=<folder of builds> BUILD=<this build> MODEL=<Flash-Next first shard> \
  ENV_ALL='GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0' WATCHDOG=./ram_watchdog.sh \
  setsid nohup bash rig_mailbox.sh > console.log 2>&1 < /dev/null &
```

Provenance: the fork's code and this folder's scripts are this project's; the random-weight models are generated by
`test-llama-archs`; no model was downloaded for this work.
