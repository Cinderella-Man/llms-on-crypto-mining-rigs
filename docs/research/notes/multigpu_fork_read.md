# The multi-GPU fork, read commit by commit against Miner Llama v4 (2026-11-10)

[lukolszewski/llama.cpp-multigpu](https://github.com/lukolszewski/llama.cpp-multigpu) serves Qwen3.8-Flash-Next
(UD-Q4_K_XL) on five and six RTX 3090s over x8, x4 and x2 links. Its branch `multigpu` (`e4054f726`) is 47 commits on
upstream `df03399b8`, each behind its own switch ([patches.md](https://github.com/lukolszewski/llama.cpp-multigpu/blob/multigpu/docs/multigpu/patches.md)).
Its author reports prompts 568 → 2,127 tok/s and generation 25.7 → 41.9 tok/s at 50k tokens, one session, upstream
against patched ([second review](flash_next_review2_2026-11.md), section 3.1).

This note reads each commit against Miner Llama's v4 engine branch `pascal-p106-v4` at `84f7bd3d2` (upstream b12246
merged on the v3.0 tree) and ranks what to take by what two rig runs measured on nine P106-090 cards:
[R42's prompt timeline](../../data/2026-11-10_prompt_timeline/README.md) and
[R43's prompt lane](../../data/2026-11-10_prompt_lane/README.md). The commits were read as their patches on GitHub;
nothing of the fork was built or run.

## The short answer

1. **The fork's base is two months older than ours.** `df03399b8` is upstream tag b10902 (2026-09-10); Miner Llama v4
   carries b12246, 1,344 upstream commits later. Of the fork's four causes of idle cards in a prompt, two are not ours
   (pipeline parallelism is on in the preset, and the sort is stream-ordered in our base), one does not show for the
   main model (no reallocation at 4,097 tokens), and the fourth, a full launch queue, is the pattern R42 measured.
2. **The forced graph capture is the one port that matches the largest measured share.** R42's host is blocked inside
   a call's third queued micro-batch for 29.5 % of a fresh 4,097-token prompt (51.9 % with the taper); the fork names
   the same stall ("kernel-launch queue stalls capped the pipeline depth at about 1.5 cards") and lifts it by
   capturing every prompt split as a CUDA graph. Ported as `GGML_CUDA_GRAPHS_FORCE` (off by default) on
   [`pascal-p106-v4-b12246-c702`](https://github.com/Cinderella-Man/miner-llama/commits/pascal-p106-v4-b12246-c702)
   (`692bbbc9a`, 20 lines in `ggml-cuda.cu`), not yet built.
3. **The fork's drain-free checkpoints do not help one agent.** `fcb21fe5f` takes the synchronous path whenever a
   draft model is loaded, and the preset always loads the MTP draft; it still cuts the call at each checkpoint; and the
   slot waits for its own save, so only other slots gain. What R42 and R43 measured (the call cut at the checkpoint,
   18.10 % of prompt time with device stores) needs the checkpoint taken inside the call, between two micro-batches,
   on the cards' own streams: the design registry Q770 already describes.
4. **26 of the 47 commits were built for several sessions at once:** 13 change nothing with one slot (decode groups,
   output regions, prefill admission and chunking, sequence compaction), 6 stage inputs only for pipelined
   multi-sequence decode, and 7 make re-planning drain-free, measured only with several sessions. They wait behind
   single-agent work.

## 1. What R42 and R43 measured

A fresh 4,097-token prompt on the v3.0 release line, taper off, 29,983.0 ms (R42; 1.42 of the nine cards busy on
average):

| Part of the window | ms | share |
|---|---:|---:|
| the host waits for the cards after each of the four calls (2,048, 1,533, 512 and 4 tokens) | 13,760.4 | 45.9 % |
| of which the 512-token call between the two checkpoints, one micro-batch through nine cards in turn | 5,988.7 | 20.0 % |
| the host blocked inside a call's third queued micro-batch on cards 1 to 7 | 8,851.0 | 29.5 % |
| the MTP draft's catch-up (32 passes of 128 rows on cards 4 and 8) | 2,944.1 | 9.8 % |
| after the catch-up until the next call, of which checkpoint saves 1,352.9 | 2,671.7 | 8.9 % |
| the host before card 0 inside calls | 1,379.0 | 4.6 % |

The main model logs no compute-buffer reallocation in that prompt; the draft logs 34 (4 with a changed size), about
one per catch-up pass. With the taper the blocked splits grow to 15,483.3 of 29,807.6 ms.

The agent check's 62 prompts replayed (R43): 509.16 and 499.93 s a pass with the preset's checkpoints, 349.57 without
(the checkpoints cost 30.72 % of prompt time), 437.54 and 416.17 with `--ctx-checkpoints-vram 4`, which takes a
checkpoint in 17.6 ms on the cards against 641.5 ms through the host and still leaves 77.28 s a pass, 18.10 % of that
line's prompt time.

## 2. Every commit

"In our base" means the same effect is in `pascal-p106-v4`: from upstream between b10902 and b12246, or Miner Llama's
own. "Several slots" means it changes nothing with `-np 1`. "Fork fix" means it repairs code that exists only in the
fork. No upstream commit after b12246 (to `582eab718`, 2026-11-10) on the files these commits touch carries one of
their ideas.

| # | Commit | What it does (switch) | Here | Port |
|---:|---|---|---|---|
| 1 | `b22bc5b68` | decode at depth: sparse attention given the selection's width, pooled keys kept and re-pooled only where blocks changed, a block-level top-k; a 256-wide tensor-core sparse shape (always on) | in our base: upstream's k-pool selection with pooled keys kept across batches (#29805, #30480, #30613, #31243, #30965) and the Pascal sparse decode kernel (Q230, `GGML_CUDA_FATTN_SPARSE_PASCAL`); the shape is tensor-core only | — |
| 2 | `4d6ef5ad4` | decode attention over a compact gather of the selected cells, so a q8_0 cache keeps the sparse speed (always on) | tensor-core only: it works around the tensor-core kernel's f16-only sparse path; the Pascal kernel reads the selected cells itself | — |
| 3 | `b75e380e0` | busy slots kept on contiguous sequence ids so several sessions decode in one micro-batch (`--seq-compact`); the fork's block-key book reset on sequence copies | several slots; the reset is a fork fix | Q1164 |
| 4 | `02e136bef` | prefill admission: slots and long prompts prefilling at once (`--prefill-max-partial`, `--prefill-long-threshold`, `--prefill-max-long`), wait metrics | several slots | C716 (parked) |
| 5 | `c8832a5f0` | prompt micro-batches overlapping across cards: pipeline parallelism despite a tensor override (`LLAMA_PIPELINE_PARALLEL`), re-reserve once per shape against a full-context memory, a stream-ordered sort (`GGML_CUDA_SEGMENTED_SORT` restores the old one), forced graph capture (`GGML_CUDA_GRAPHS_FORCE`), reallocation triggers logged (`GGML_SCHED_LOG_REALLOC`); 859 → 2,500 tok/s | pipeline parallelism is on in the preset already (the only override, `-otd`, is the draft's); the sort is in our base (bitonic argsort for expert routing, radix top-k for the indexer: #28713, #28957); no main-model reallocation at 4,097 tokens; **the forced capture applies** | forced capture **ported** (C702); the re-reserve for the draft: C712; the trigger log: C701 |
| 6 | `0f5aaf45d` | `GGML_SCHED_N_COPIES`: fewer input copies, less compute-buffer memory, a shallower pipeline | applies: memory on 6 GB cards against depth | C713 (9 lines, low risk), after C702 |
| 7 | `a4e856f1a` | the causal KQ mask built on each device from cell positions (`LLAMA_KQ_MASK_DEVICE`); 84 MB a micro-batch at 82k on its rig | in our base: Flash-Next allocates no KQ mask (upstream #30965; C23), other models have Miner Llama's `LLAMA_GPU_KQ_MASK` | — |
| 8 | `9f5a3202e` | the sparse layers' block bias built on each device (`LLAMA_QSA_BIAS_DEVICE`) | in our base in another form: the selection's mask is one fused op on the device (#30640); the k-pool inputs still go up from the host every step, which C100 (Q640) moves onto the cards | C100 |
| 9 | `4660a295e` | a multi-sequence decode batch split into per-sequence micro-batches (`LLAMA_DECODE_PIPELINE=1`) | several slots; Miner Llama has its own pipelined decoding (`--pipe-depth`, Q550) | Q1164 |
| 10 | `fb8897947` | a stream-agnostic decode graph for those micro-batches | several slots | Q1164 |
| 11 | `1567041ef` | a per-micro-batch trace (`LLAMA_UBATCH_TRACE`): reused, reserved, preparation and submission time; per split the host's wait-and-copy and launch time | applies | C701 (39 lines, off by default) |
| 12 | `1ca80a510` | scheduler inputs staged through rotating pinned buffers and copied on the compute stream (`GGML_SCHED_STAGE_INPUTS`) | after #17 only for pipelined multi-sequence decode; Miner Llama's `LLAMA_INPUT_UPLOAD=2` stages a split's host inputs when nothing is in flight (no measurable gain on the rig), and reuses a graph only when nothing is in flight (`LLAMA_GRAPH_REUSE_WAIT`) | C717 (parked) |
| 13 | `5110faa51` | one staging offset per backend per compute (streams read each other's inputs) | fork fix | with #12 |
| 14 | `6b7bb59fe` | staging slots grown geometrically (`cudaHostAlloc` churn: five 164k prefills 581 s staged against 355 s unstaged); `GGML_SCHED_STAGE_WAIT` | fork fix | with #12 |
| 15 | `850db8aec` | a hop copy's source stream waits for the destination's previous split (`GGML_SCHED_NO_HOP_WAIT=1` turns it off) | fixes a reused graph overwriting its own inputs in flight, which Miner Llama avoids by rebuilding | with #12 |
| 16 | `21910bb3b` | staging only for reused graphs | with #12 | with #12 |
| 17 | `66a32a7f8` | staging only for pipelined decode micro-batches; prompts and single-sequence decode keep plain copies | with #12 | with #12 |
| 18 | `fa532f399` | per-phase `update_slots` timing (`LLAMA_SERVER_TIMINGS`) | applies (diagnostic; one synchronize a decode when on) | with C701 |
| 19 | `53fcc9f87` | several batches in flight: an output region per batch with its event, server decode groups (`LLAMA_SERVER_GROUPS`) | several slots (groups refuse fewer than two) | Q1164 |
| 20 | `d59899454` | parked batches swapped member by member (a double free) | fork fix | — |
| 21 | `9109ac64c` | single-sequence decode on the stream-agnostic graph with a cache-wide n_kv floor (`LLAMA_DECODE_PIPELINE=2`) | several slots | — |
| 22 | `c89c7fc89` | trace: why a hybrid graph was not reused | applies (diagnostic) | with C701 |
| 23 | `ec1b7bf52` | the decode pipeline's level read live | fork fix | — |
| 24 | `2c9807e15` | a trace of full synchronizes while output regions are in use | several slots | — |
| 25 | `00b3703f0` | sampled-token getters wait only for the selected output region | with one region it is a full synchronize; Miner Llama's `synchronize()` returns at once when nothing is in flight, but only under `LLAMA_INPUT_UPLOAD` | C714: that skip on its own switch |
| 26 | `f8c8a9ce0` | the common sampler calls `llama_output_synchronize` | same as #25 | same row |
| 27 | `ef7ffc7e2` | groups without work skipped; the per-stream graph while one slot is active | several slots | — |
| 28 | `07728cbfa` | prefill chunked to one micro-batch and one slot while others generate | several slots | C716 (parked) |
| 29 | `da3fbb65d` | reserve once per shape class: a list of reserved shapes, a smaller shape counts as covered | pipeline-parallel contexts only | C712 |
| 30 | `6ccace4e8` | no drain when the graph's layout changes: the allocator synchronizes only to grow a buffer; the first compute after a re-plan synchronizes per input (`GGML_SCHED_REPLAN_SYNC=1` drains) | the main model logs no re-plan at 4,097 tokens; the draft keeps one input copy, where the fork still drains | C717 (parked) |
| 31 | `7419cd87c` | eager prefill chunks; a 12-slot staging ring allocated up front, 16 MiB a slot | several slots; 192 MiB pinned a card does not fit a 4 GB host | — |
| 32 | `442b4e349` | a synchronize before an unstaged host copy after a re-plan; eager prefill off (it corrupted decoders under CUDA graphs) | with #30 | C717 (parked) |
| 33 | `270b27d53` | the drain on re-plan back on by default (a decoder rebuilt after a deep prefill micro-batch was corrupted) | with #30 | C717 (parked) |
| 34 | `c73c59364` | completion events recorded again after the output reads (the corruption's cause: a copy overwrote logits before they were read); no-drain re-plan on again; 256-token chunks | with #30 | C717 (parked) |
| 35 | `ca0faad46` | the re-plan's safe paths kept for 12 computes (an illegal access when a second sequence joined) | with #30 | C717 (parked) |
| 36 | `93764d163` | each event carries its plan's epoch, so one plan's micro-batches stay pipelined; chunking only while decoders outnumber prefills | with #30 | C717 (parked) |
| 37 | `a1405f2f7` | asynchronous per-sequence state jobs (`llama_state_seq_get_data_begin`, `llama_state_seq_job_step`) on a worker thread | the base of #38 | with #38 |
| 38 | `fcb21fe5f` | context checkpoints behind one event per backend (`llama_fence_record`), copied by a worker through pinned staging (`LLAMA_SERVER_XFER_STAGE_MB`); restores without a drain; idle saves one a round (`LLAMA_SERVER_CKPT_SYNC=1` restores the old path) | not as written: with a draft model the server takes the synchronous path; the call is still cut at the checkpoint; the slot is held until its own save is done | C217 on Q770's design (section 3) |
| 39 | `1b85f6429` | transfers finished while no slot processes | several slots | — |
| 40 | `2866c72a2` | a task waits for its slot's save instead of reading 164k tokens again in another slot | prompt cache, several slots | — |
| 41 | `c8bef79c2` | a prompt-cache save starts in at most 1 ms: shared checkpoint bytes, frees on a reaper thread, `madvise` in 64 MiB pieces | prompt cache (the preset runs `--cache-ram 0`) | C715 (parked) |
| 42 | `4fb700cf5` | staged state copies on several threads (`LLAMA_STATE_XFER_THREADS`, 4): a 2.4 GiB restore 1.83 → 0.99 s | applies to checkpoints through the host and the prompt cache: each card has its own x1 link | same row |
| 43 | `6a33bc322` | a cancelled request's output region released | several slots | — |
| 44 | `efef36982` | n-gram drafts with decode groups; the recurrent state's rollback point kept as a spare cell, metadata only, with no copy and no drain | the rollback is in our base another way (rollback rows per draft step); the spare cell is one way to take C217's in-call checkpoint | with C217 |
| 45 | `83f09b865` | the n-gram map reset when a prompt does not extend the last one; drafting capped by users (`LLAMA_SERVER_SPEC_MAX_USERS`) | the reset applies to one slot with lookup drafts (Q1161); the cap is several slots | line in Q1161 |
| 46 | `7c29e21b8` | drafting stopped while the accepted tokens a round stay below 2.5 per user (`LLAMA_SERVER_SPEC_MIN_ACCEPT`) | in our base in another form: the adaptive draft's cost rule (`LLAMA_SPEC_ADAPT`, Q1132) | — |
| 47 | `e4054f726` | `std::max` made portable in `output_reserve` (gcc 13) | fork fix | — |

By kind: 4 in our base or our own (1, 7, 8, 46; also 5's parallelism and sort), 1 tensor-core only (2), 13 for several
slots only (3, 4, 9, 10, 19, 21, 24, 27, 28, 31, 39, 40, 43), 6 in the input-staging chain (12 to 17, which ends
staging only pipelined multi-sequence decode), 7 in the parked re-plan chain (29, 30, 32 to 36), 3 fork fixes (20, 23,
47), and 13 that apply to one agent on nine x1 cards: 5's forced capture, 6, 11, 18, 22, 25, 26, 37, 38, 41, 42, 44's
spare-cell idea and 45's map reset.

## 3. The port list

Ranked by the share of prompt time R42 and R43 measured that each one acts on. Every gain below is an estimate until
its rig run.

| Rank | Row | What | Acts on (measured) | Cost, risk |
|---:|---|---|---|---|
| 1 | C702 (Q1240) | `GGML_CUDA_GRAPHS_FORCE=1`: a prompt split captured as a CUDA graph from its first call and again after every change, so it reaches its card as one launch per graph chunk instead of one per kernel | the blocked third micro-batch: 29.5 % of a 4,097-token prompt (8,851.0 ms), 51.9 % with the taper (15,483.3 ms) | ported, 20 lines, off by default. Risk: a capture and an executable update per split per micro-batch on the 2-core host (unmeasured), and prompt kernels that are not capture-safe; the split's graph is checked as for decode (`ggml_cuda_graph_check_compability`). Next: build, the CPU and PTX checks, then R42's driver with the switch against R42's loads and R62 |
| 2 | C217 (Q770) | the prompt's checkpoints taken inside the call: the server stops cutting the batch at a checkpoint; `llama_decode` ends a micro-batch at the checkpoint position and enqueues the copy of the recurrent rows into the device store on each layer's card stream (or keeps the rollback point as a spare cell, `efef36982`'s snapshot), with no host wait; the fork's per-backend fence (`fcb21fe5f`) orders any host read after it | the cut: 18.10 % of prompt time with device stores (77.28 s a pass), 30.72 % with the release preset's host copies; in R42 the lone 512-token call alone is 20.0 % | own design, est. 3 to 5 days (llama API, memory, server); the fork's code does not apply as written. Risk: restores must give the same replies (R43's greedy-identity check) |
| 3 | C701 (Q1238) | the per-micro-batch trace (`1567041ef`, `c89c7fc89`) plus the allocator's reallocation triggers from `c8832a5f0` and the server's phase timing (`fa532f399`) | confirms rank 1's cause (the split's wait-and-copy against its launch time) and finds the draft's re-plans | about 80 lines, off by default; no risk. Our tree lacks the fork's reserve fields, so the `reserved` column is left out |
| 4 | C712 (Q1264) | the MTP draft's 34 re-plans a 4,097-token prompt: find the trigger (`GGML_SCHED_DEBUG_REALLOC=1`, on the quant machine), then either reserve the draft's catch-up shape once against a full-context memory (`c8832a5f0`, `da3fbb65d`, opened to a context without pipeline parallelism) or give the draft's own 1-to-3-token steps their own scheduler (`LLAMA_SCHED_PER_SIZE`, Q701) | part of the draft's catch-up, 9.8 % (2,944.1 ms); each re-plan of a one-copy scheduler drains the draft's cards | 40 lines; low risk (shapes only). Beside C703, which moves the catch-up into the next call |
| 5 | C713 (Q1265) | `GGML_SCHED_N_COPIES` (`0f5aaf45d`): three input copies instead of four if the pipeline never reaches a fourth micro-batch; a fit probe of what it frees on each card | memory, not time: R42's depth stops at the third micro-batch today | 9 lines, off by default; after C702, which may deepen the pipeline |
| 6 | C714 (Q1266) | the per-token synchronize skip without `LLAMA_INPUT_UPLOAD` (the aim of `00b3703f0` and `f8c8a9ce0` for one slot): `synchronize()` returns at once when nothing is in flight | est. four sweeps over nine idle streams a token, in the negligible class (C534) | a few lines; risk: the in-flight flag must stay exact outside `LLAMA_INPUT_UPLOAD` |
| 7 | C715 (Q1267) | prompt-cache transfers off the token's path: per-card copy threads (`4fb700cf5`), save starts without copies (`c8bef79c2`), asynchronous state jobs (`a1405f2f7`) | none in the preset (`--cache-ram 0`, device stores); reopens with a prompt cache (Q774) or a preset without device stores | parked; 300 lines |
| 8 | C716 (Q1268) | several agents on one rig: prefill admission (`02e136bef`), prefill chunked while others generate (`07728cbfa`, `93764d163`'s rule), sequence compaction (`b75e380e0`) | several slots (Q1164) | parked behind single-agent work |
| 9 | C717 (Q1269) | parked: the re-plan without a drain (`6ccace4e8` to `93764d163`) and input staging for reused graphs (`1ca80a510` to `66a32a7f8`, `850db8aec`) | nothing at one slot: the main model does not re-plan at 4,097 tokens, the draft keeps one copy, graphs are reused only with nothing in flight | reopens if C701 shows main-model re-plans in long prompts, or after C702 if the trace shows the input copies waiting; two corruptions in the fork's history make it the riskiest chain |

Two rows outside the fork act on the same window: R63 (`-b 4096`, the first two calls in one) and C703 (the draft's
catch-up run during the next call's fill).

## Sources

The fork, read 2026-11-10 as GitHub's commit patches (`https://github.com/lukolszewski/llama.cpp-multigpu/commit/<id>.patch`)
for all 47 commits listed in [patches.md](https://github.com/lukolszewski/llama.cpp-multigpu/blob/multigpu/docs/multigpu/patches.md),
and its [README](https://github.com/lukolszewski/llama.cpp-multigpu). Its author's numbers are from five and six RTX
3090s (sm_86 only) and were not reproduced here.

Miner Llama, [`pascal-p106-v4`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v4) at `84f7bd3d2`:
`src/llama-context.cpp` (the pipeline-parallel gate, `LLAMA_INPUT_UPLOAD`, `LLAMA_GRAPH_REUSE_WAIT`,
`LLAMA_SCHED_PER_SIZE`, the device stores' `llama_state_seq_get_data_store`), `ggml/src/ggml-backend.cpp` (input
copies and their slots), `ggml/src/ggml-cuda/ggml-cuda.cu` (the graph warm-up), `ggml/src/ggml-cuda/argsort.cu`,
`top-k.cu`, `src/llama-graph.cpp` (`LLAMA_GPU_KQ_MASK`, the expert routing), `src/models/qwen4exp.cpp`,
`src/llama-memory-hybrid-idx.cpp`, `tools/server/server-context.cpp` (`create_checkpoint`). Upstream
[llama.cpp](https://github.com/ggml-org/llama.cpp) from b10902 to `582eab718`.

Project data: [R42](../../data/2026-11-10_prompt_timeline/README.md), [R43](../../data/2026-11-10_prompt_lane/README.md);
notes: [second review](flash_next_review2_2026-11.md), [prefill](flash_next_prefill.md).
