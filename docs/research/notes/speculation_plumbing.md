# Speculation plumbing for Flint on the P106 rig: CUDA graphs per verify length and copy-gated n-gram drafting (2026-10-23)

Scope: everything between "a drafter proposes k tokens" and "the GPUs verify them" for Flint (40B-A2.5B MoE, 49 layers,
256 experts, top-6, no MTP head) in llama-server on nine P106-090 cards with a 2-core Celeron host. Two fork changes are
designed here: a CUDA-graph and graph-reuse cache that keeps one graph per verify length (registry Q52), and an n-gram
drafter that drafts long blocks only while the model copies its context (Q53). The note traces the code in our llama.cpp
fork (branch `pascal-p106`, head `eeaa59660` on upstream `d77256999`) and in upstream master (`36e36e6`, 2026-10-23),
lists the upstream pull requests on both topics, builds a cost model for this rig and ranks the experiments. It extends
sections 1–2 of [HyperQwen lessons](hyperqwen_lessons.md). Conventions:
- "Claim to test" marks anything not measured on this rig.
- "(my calc)" marks this note's own arithmetic; "(est.)" marks an estimate with its basis given next to it.
- "P" and "S" in source lists mean primary (code, docs, measurements) and secondary.
- `file:line` refers to the fork head. `llama-context.cpp`, `llama-graph.h`, `ggml-backend.cpp`, `llama-kv-cache.cpp`,
  `ngram-map.cpp`, `sampling.cpp` and `common.h` are identical on master. Where master differs, its line follows as
  "master :N".

Starting point, measured: S4-L0 (the quality-bar mix) decodes 37.4 tok/s in llama-bench tg128 on six cards with the fork's
Pascal kernels, 26.8 ms per token (T1 below), and 29.2 tok/s (34.25 ms) with the kernels off. In a five-turn chat at
temperature 1.0 on nine cards with backend sampling (`-bs`) it generates 31.6–32.2 tok/s.

## Headline findings

1. **Every change of batch shape costs three separate things today, and speculation changes the shape at every draft
   decision.**
   - llama-context reuses a built graph only for an identical ubatch shape. Otherwise the 2-core host rebuilds the
     ~3,500-node graph, re-splits it across the cards and re-allocates it (est. 3–5 ms).
   - The CUDA backend keeps one captured graph per split, keyed by the address of the split's first node. A 1-token and
     a k-token graph get the same key, so the switching step runs uncaptured (~5 ms; graphs are worth +14 % for Flint),
     and the step after re-captures with a fresh instantiation, because the two shapes launch different kernels
     (est. 2–10 ms).
   - Every rebuilt graph also pays a property comparison of 1,056 bytes per node, ~3.7 MB per token (my calc).
   - One shape switch costs ~10–20 ms, 0.4–0.75 of a token (est.). An isolated draft between plain tokens costs two
     switches plus a re-capture, ~0.7–1.1 of a token on top of its verify cost.
2. **Upstream has the CUDA half, open since 2026-10-22, and a maintainer wants it solved one level up.**
   - PR #31004 (open, 2026-10-22) keys the CUDA graph cache by the first node's address plus its shape, keeps up to 8
     graphs per card with frequency-based admission, and measured +3.7–5.2 % for 7-token DFlash2 drafts. Its
     predecessor #28666 measured +18 % for a draft model (issue #28652: 68 warm-up resets per request; speculation
     was slower than none until the key changed).
   - On 2026-10-23 a maintainer asked for separate graph arenas per small token count in llama-context, as the merged
     MTP fix #28549 did, and ideally a scheduler per graph. Nothing upstream keeps several built graphs in llama-context.
   - #31004 enables graphs only from Volta up. The fork must keep its compute-6.1 gate when it takes the patch.
3. **None of llama.cpp's draft-model-free drafters fits Flint.**
   - Draft lengths vary with the match (`ngram-simple`, `ngram-mod`, `ngram-cache`) or only shrink after misses
     (`ngram-map-k`). Nothing grows a block while a copy runs, and the first non-empty draft wins.
   - Only `ngram-mod` gates on quality: a 24-token key and 48–64-token drafts, which fire on verbatim recall and almost
     nowhere else (PR #27210's data: 292.9 against 29.9 tok/s on recall, unchanged on reasoning, prose and code).
   - Ungated drafting is expensive on MoE targets. The open SuffixDecoding port (PR #26283) took an MoE model from
     185 to 110 tok/s on normal prompts until a 3-token minimum draft was added.
   - Verification is exact at temperature 1.0 (sample-and-match), and the accepted count is on the host after sampling
     anyway, so a controller costs nothing to run.
4. **Design: an `ngram-copy` drafter with three fixed verify lengths.**
   - No draft below a 6-token match.
   - 2 drafted tokens on a match (a 3-token verify, inside every format's matrix-vector limit).
   - 7 drafted tokens (an 8-token verify) after two consecutive fully accepted steps, while the match has 7 tokens of
     continuation or repeats with a short period; kept for 2 missed steps.
   - The match is the longest one up to 12 tokens, the newest on ties, and may overlap its own suffix.
   - About 250 lines in `common/`, no server change. With verify lengths of 1, 3 and 8 only, Q52 has three shapes to
     keep per card.
5. **Expected speed (my calc, at the 32 tok/s chat baseline, Q52 in place):**
   - chat 32 tok/s (−1 to +1 %), new code 32–34 (−1 to +5 %);
   - edits that keep context 43–65 (1.3–2.0×), file rewrites 53–121 (1.65–3.8×; above 3.3× only with 15-token
     blocks);
   - agent sessions 36–43 (1.1–1.35×).

   The spread is the verify cost of 2–16 tokens (Q19). 15-token blocks would beat 7-token blocks only with real expert
   locality, and Flint has little (Q51: 8 consecutive tokens touch 80 % of the experts random routing would).
6. **Q52 matters little for the gated drafter and a lot for everything else.**
   - The gate keeps the batch shape stable for long stretches, so the model puts Q52's gain for `ngram-copy` at
     +2–6 % (+12 % at the high end of the host costs).
   - Ungated drafters (`ngram-simple`-like) gain +8–31 % outside file rewrites. Q52 is required for drafting with
     `-np 2` and for learned drafters, whose catch-up batches vary in length.
   - The CUDA half (cherry-pick #31004) gives most of it; per-shape graphs in llama-context add 1–2 % (my calc).
7. **Two host-side costs grow with every verified position on this rig.**
   - Without backend sampling (every request with tools or a grammar) each verified row copies 600 KB of logits over
     the last card's x1 link, ~2.8 ms per row. Agent sessions lose ~4 points of their gain (1.22× → 1.18×, my calc).
   - The KQ masks are graph inputs placed on the CPU backend and copied synchronously to every card at every step:
     n_kv × n_tokens × 2 bytes per card. That is 64 KiB per card at 32k context and one token, 512 KiB at 32k and
     8 tokens (~2.7 ms per x1 link), 4 MiB at 128k and 16 tokens (~22 ms per link). Long blocks may need a depth cap
     (claim to test).
8. **The cheapest decisive experiment needs no rig.** Replaying drafters offline on transcripts Flint generated at
   temperature 1.0 gives exact acceptance for any controller and setting, because each realized token is a sample from
   the target, which is what sample-and-match compares against (new row Q64). With Q19's step costs and Q51's locality it
   fixes the draft lengths and the gain per workload before any fork code is written.

## 1. How a decode step reaches the GPUs

### 1.1 llama-context: rebuild or reuse

- **The reuse test.** `process_ubatch` reuses the previous graph only if it is the active one and `can_reuse` accepts the
  new parameters (`llama-context.cpp:1530-1547`). Otherwise it resets the result and the scheduler, calls
  `model.build_graph` and `ggml_backend_sched_alloc_graph` (`:1548-1573`).
- **What has to match** (`llama-graph.h:839-910`):
  - n_tokens, n_seq_tokens, n_seqs, n_seqs_unq and, for equal-sequence splits, the same sequence ids;
  - n_outputs, the samplers and each output row's sequence;
  - every graph input's own shape, e.g. the KQ mask's width n_kv (`llama-graph.cpp:49-66`).
- **Two arenas, one active graph.** Graph results live in two arenas chosen by `n_outputs > 0` (`get_gf_res_prev`,
  `llama-context.cpp:2572-2578`; `llama-context.h:415-418`). The second arena came with PR #28549 for MTP's no-output
  batches. A verify step and a plain decode step both produce outputs, so they share one arena, and only the last built
  graph is kept (`gf_res_prev_active`).
- **n_kv steps.** The mask width is padded to 256 cells (`llama-kv-cache.cpp:1285-1299`), so plain decoding rebuilds once
  per 256 tokens. A rejected draft frees its cells, so near a 256 boundary the width can move back and forth.
- **Switches for testing.**
  - `LLAMA_GRAPH_REUSE_DISABLE=1` rebuilds at every step (`llama-context.cpp:378-383`).
  - `LLAMA_GRAPH_RESULT_DEBUG=2` logs which input refused reuse (`llama-graph.cpp:1385`).
- **Backend sampling chains.** With `-bs`, every sampler builds `n_outputs_max_per_seq` chains in every graph, so the
  topology stays fixed across ubatches (`llama-graph.cpp:4242-4250`, PR #30223). The server sets that number to 1 + the
  largest configured draft (`server-context.cpp:46-66`).
  - `ngram-mod`'s default of 64 therefore puts 65 sampler chains into every graph, plain steps included.
  - A 7-token cap puts 8.

### 1.2 The scheduler

- **Splits.** It builds one split per run of layers on a card and gives every split a new uid at each re-split
  (`ggml-backend.cpp:1749-1752`). A reused graph keeps its uids.
- **Inputs live on the host.** Graph inputs (token ids, positions, KQ masks, KV row indices) are placed on the CPU
  backend (`:1092-1097`). The scheduler copies them to each card before that card's split runs, and it copies user
  inputs synchronously: it waits for the card, then copies (`:1823-1837`, called from `:1886-1897`).
- **The masks cross every link.** Every card that holds attention layers receives, at every step:
  - the full-attention mask, n_kv × n_tokens in f16 with flash attention (`llama-graph.cpp:30-47`);
  - the sliding-window mask, at most ~1,000 columns wide;
  - and the host blocks on each of these copies in turn.

### 1.3 The CUDA backend: one graph per split

| Stage | Code (fork head; master) | Behaviour |
|---|---|---|
| Key | `ggml-cuda.cu:2759-2761`; master :2758 | The address of the split's first node |
| Cache | `common.cuh:1468-1497`; master :1463-1490 | Map from key to graph; graphs unused for 10 s are dropped in a sweep every 5 s |
| Architecture gate | `:5001-5015`; master :4794-4808 | The fork allows compute 6.1 and up (`fedac87b2`, the same change as PR #27721); master only Volta and up, so master runs Pascal without graphs |
| Fast path | `:2769-2775` | If the split's uid equals the cached one (llama-context reused the graph), no property check |
| Property check | `:2777-2804` | Copies and compares, per node, the tensor header plus every source's data pointer, `ne` and `nb`: 336 + 80 + 320 + 320 = 1,056 bytes (my calc) |
| Warm-up | `:5036-5068`; master :4825-4858 | A new or changed graph runs eagerly. The next call with unchanged properties captures. A change after warm-up resets it, so that step runs eagerly too |
| Capture | `:5070-5078`, `:4888-4906` | Stream capture of the whole split |
| Instantiate or update | `:4908-4918`, `:2807-2833` | The first capture instantiates. Later captures try `cudaGraphExecUpdate` and re-instantiate if the update fails |
| Compatibility | `:2726-2757`, `:2034-2063` | Only `MUL_MAT_ID` paths that synchronize turn graphs off. Quantized experts on the matrix-vector or MMQ path keep them |

### 1.4 What differs between a 1-token and a k-token step

- **Shapes and addresses.** Every activation's second dimension, every input (tokens, positions, mask rows, KV indices),
  and every allocation offset in the compute buffer.
- **Kernels:**
  - the fork's Pascal kernels run only for one column (`mmvq-pascal.cu:607`);
  - the dense matrix-vector fusions also need one column (`ggml-cuda.cu:1913-1916`);
  - the expert products stay on the matrix-vector path up to 6 tokens for Q4_0, Q4_1, Q5_0 and IQ4_NL, 5 for Q4_K,
    Q5_K and IQ4_XS, and 4 for Q8_0 and Q6_K, and move to MMQ above that (`mmvq.cu:153-177, 288-297`);
  - dense products stay on it up to 8 tokens.

  For S4-L0 that means: the Q8_0 down-projections leave the matrix-vector path above 4 tokens, the Q5_0 gate/up above 6.
- **Consequence.** The kernel lists differ, so `cudaGraphExecUpdate` fails between the shapes and the backend
  instantiates again. A change that keeps the topology, such as n_kv crossing a 256 boundary, updates in place.
- **One key for both shapes.** The first node of a split is built at the same position in the same arena whatever the
  token count, so the 1-token and the k-token graph share its address. Issue #28652's reproducer prints the same key for
  both variants.

### 1.5 What a shape sequence costs today

R = graph rebuild, U = uncaptured step, C = capture with re-instantiation (values in section 1.6).

| Step | Tokens | llama-context | CUDA backend | Extra cost |
|---|---|---|---|---|
| 1 | 1 | reuse | replay | — |
| 2 | 3 (2 drafted) | rebuild | properties changed: warm-up reset, eager | R + U |
| 3 | 1 | rebuild | changed again: eager | R + U |
| 4 | 1 | reuse | unchanged: capture, re-instantiate | C |
| 5 | 1 | reuse | replay | — |

- **An isolated draft** costs 2R + 2U + C ≈ 18–30 ms, 0.7–1.1 T1, on top of its own verify cost (est.).
- **Drafts on every other step** run every step uncaptured and rebuilt.
- **The same effect upstream:**
  - Issue #28652, a draft model on an RTX 4060 Ti: 68 warm-up completions and 68 resets per request. Speculation ran at
    173.5 tok/s against 188.7 without it, and at 205.3 with per-shape keys.
  - PR #28549: MTP's catch-up and draft batches re-captured each other at every iteration; separate arenas gave +4–5 %
    on an RTX 5090.

### 1.6 Host-side costs on the rig

| Cost | Value | Basis |
|---|---|---|
| U: one step without CUDA graphs | ~4.9 ms, 0.18 T1 | Measured on the pre-fork build: makers' Q4_1 on 5 cards, 24.56 → 27.92 tok/s with graphs ([Flint benchmark](../../benchmarks/2026-10-22-flint.md)); claim to test on the fork build |
| C: a capture step with re-instantiation | 2–10 ms (est.) | Capturing ~300–600 kernels per card and instantiating them on a 1.8 GHz core, mostly hidden behind the previous card's work except on the first card |
| R: graph rebuild | 3–5 ms (est.) | PR #14482 saved 0.2–0.3 ms per token by reusing 1.5–3B dense graphs on an M2 Ultra core; Flint's graph is ~3× larger (~3,500 nodes; Qwen3.6-35B-A3B's is 3,576–3,727 on the rig); the Celeron 3865U is ~3–4× slower per core |
| Property check after a rebuild | 0.5–1 ms (est.) | 3.7 MB copied and compared per token (my calc); PR #21764 measured 0.3 ms per token for Qwen3.5-35B-A3B on a desktop CPU |
| KQ mask copies, per card and step | n_kv × n_tokens × 2 B at ~190 MB/s | 0.34 ms per token row at 32k, 1.35 ms at 128k (my calc); the host copies card after card |
| Logits without backend sampling | ~2.8 ms per output row | 150,144 × 4 B over the last card's x1 link at ~213 MB/s; host sampling adds to it ([HyperQwen lessons](hyperqwen_lessons.md) §4) |
| Activations between cards | ~0.04 ms per token row per hop | 8 KB per token row at ~190 MB/s (my calc) |

For scale: the host launches all six cards' graphs within ~1.9 ms of a token's start while each card computes 4–8 ms
(Q49), and busy CPU cores cost small-MoE models 4–11 % ([small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md)). The host is not the limit within a captured
token, but every rebuild, copy and capture lands between tokens, where nothing overlaps it.

## 2. Upstream work on graph caching

| PR / issue | State (2026-10-23) | What it does | Measured |
|---|---|---|---|
| [#14482](https://github.com/ggml-org/llama.cpp/pull/14482) | merged 2025-07 | llama-context reuses the previous graph when the ubatch shape matches | +2–4 % tg on an M2 Ultra |
| [#21764](https://github.com/ggml-org/llama.cpp/pull/21764) | merged 2026-04-16 | Passes reuse to the CUDA backend (the uid fast path) to skip the O(n) property check | +2–7 % on an RTX 5090 |
| [#28549](https://github.com/ggml-org/llama.cpp/pull/28549) | merged 2026-09-16 | Second graph-result arena for MTP's no-output batches, so both shapes get stable, distinct CUDA keys | +4–5 %, Qwen3.6-35B-A3B MTP-3, RTX 5090 |
| [#28652](https://github.com/ggml-org/llama.cpp/issues/28652) | open issue | Alternating variants under one key reset the warm-up at every switch; model-free reproducer | 173.5 → 205.3 tok/s with shape keys |
| [#28666](https://github.com/ggml-org/llama.cpp/pull/28666) | closed 2026-09-09 | Key = first node's address plus its `ne` | +2–6 % for DFlash2/MTP-7 on RTX 4090, RTX PRO 6000, GB300 (later comment) |
| [#31004](https://github.com/ggml-org/llama.cpp/pull/31004) | open, 2026-10-22, no review yet | #28666 rewritten: the same key, at most 8 graphs per card, admission by use frequency over the last 512 calls, 10 s eviction kept | +3.7–5.2 % DFlash2-7; +0.15 % plain decoding |
| [#29768](https://github.com/ggml-org/llama.cpp/pull/29768) | open | Re-capture at once after a stable replay instead of an eager step | +1.1 % batched; a reviewer expects more failed updates when the verify width varies |
| [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | open | Splits a CUDA graph into chunks to overlap the launch | +1–2 % tg |
| [#30223](https://github.com/ggml-org/llama.cpp/pull/30223) | merged 2026-10-09 | Backend-sampling chains fixed at `n_outputs_max_per_seq` per sampler | topology stays the same across ubatches |
| [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) | open | CUDA graphs on compute 6.1; in the fork as `fedac87b2` | +14 % for Flint on the rig |

The maintainers' position, from the #31004 thread (2026-10-23): rebuilds happen at the scheduler, llama-context and CUDA
levels; graphs of materially different shape should come from different arenas; a separate arena still misses the uid
fast path because the scheduler re-splits; the fix is a scheduler per graph or a scheduler that stores several graphs.
A comment on PR #27210 says the same about adaptive draft depth: graphs per depth "could be saved and re-used, but that's
a much bigger change".

## 3. Q52 in the fork: one graph per verify length

### 3.1 Layer 1: per-shape CUDA graphs (cherry-pick #31004)

- **Change.** The graph key becomes (first node address, first node `ne`) in `common.cuh` and `ggml-cuda.cu`. Each verify
  length then keeps its own captured graph, and a switch costs one llama-context rebuild and one property check, with no
  eager step and no capture.
- **Fork-specific edits:**
  - #31004 replaces `ggml_cuda_graph_set_enabled` with a check for `cc >= GGML_CUDA_CC_VOLTA`. The fork keeps
    `GGML_CUDA_CC_DP4A` there.
  - The fork's profilers key their per-graph labels by the old pointer key (`ggml-cuda.cu:4675-4688, 4791`). They take
    the new key, so each shape keeps its own labels.
- **Capacity.** 8 graphs per card covers plain decoding, the two verify lengths, `-np 2` shapes and prompt batches. The
  frequency admission keeps prompt-batch shapes from evicting decode shapes.
- **Memory.** A captured graph of ~300–600 kernels per card costs at most a few MB of device memory per shape (claim to
  test: `nvidia-smi` before and after warming three shapes).
- **Alternative with the same effect.** Separate graph-result arenas per token count up to 16 in llama-context, as the
  maintainer suggests, give each shape its own node addresses and so its own key, without touching the CUDA backend.
- **Effort.** Half a day, plus the tests in section 3.5.

### 3.2 Layer 2: per-shape built graphs in llama-context

- **Data structure.** A small LRU of {graph result, scheduler} entries for decode-sized ubatches (n_tokens ≤ 16), keyed
  by (n_tokens, n_seqs_unq, n_outputs, equal_seqs). Prompt batches keep the main scheduler.
- **Why a scheduler per entry.** Each entry's scheduler is reserved once for its shape and never re-split, so its split
  uids survive, the CUDA fast path skips the property check, and switching costs only `set_inputs` and the graph launch.
  Each entry owns its buffers, and every input is set again before each step.
- **Invalidation.** Every place that resets `gf_res_prev` today drops the cache too (`llama-context.cpp:751-755,
  992-997, 2650-2655, 3892-3897`), and so does `sched_reserve`. An entry whose n_kv grew rebuilds in its own scheduler
  once per 256 tokens.
- **Touch points.** The code that finds output tensors through `sched` (`:1684-1748, 2066-2176, 2429`) must use the
  active entry's scheduler.
- **Memory per entry (est.):**
  - per card: activations under 1 MB at 16 tokens; the copied f16 mask, n_kv × n × 2 B (8 MiB at 262k context and 16
    tokens); logits rows on the last card, 0.6 MB each;
  - host: the scheduler mallocs `graph_size × 30 × 2` tensor headers (`ggml-backend.cpp:1991`), ~20 KB per node of
    capacity, ~145 MB of address space for Flint's ~900 tensors, mostly never touched (my calc), plus pinned input
    buffers.

  Cap the cache at 3 entries on the 4 GB host.
- **Value.** +1–2 % on top of Layer 1 for `ngram-copy` (my calc), more for `-np 2` and drafter models.
- **Effort.** 2–3 days.

### 3.3 Verify lengths, slots and splits

- **Three lengths.** Q53 verifies only 1, 3 or 8 tokens. When the context end or `max_tokens` leaves less room, it drops
  to the next cached length or to no draft, never to an odd length.
- **Two slots (`-np 2`):**
  - Long blocks only while one slot is generating. HyperQwen restricts its long-block hold to one request for the same
    reason: the block costs every request in the batch step time.
  - Two slots drafting 2 tokens each make a 6-token batch, above Q8_0's expert limit of 4, so with two active slots
    only one slot drafts per step.
  - Shapes then: one slot {1, 3, 8}, two slots {2, 4}: 5 cached graphs.
- **Unify the KV cache for drafting with two slots.** With per-slot caches (no `-kvu`) the sliding-window cache needs
  equal splits (`llama-kv-cache-iswa.cpp:160-215`). A step whose slots verify different numbers of tokens then becomes
  several ubatches, i.e. several graph evaluations.

### 3.4 Host time saved (my calc)

With R ≈ 3–5 ms including the property check, U ≈ 5 ms and C ≈ 2–10 ms:

| Pattern | Today | Layer 1 | Layers 1 + 2 |
|---|---|---|---|
| One isolated draft (1 → 3 → 1) | 2R + 2U + C ≈ 18–30 ms | 2R ≈ 6–10 ms | ~0 |
| One copy run (1 → 3 → 3 → 8 … 8 → 1), 3 switches | 3(R + U) + 3C ≈ 30–60 ms | ~9–15 ms | ~0 |
| Shape changes at every step | R + U ≈ 8–10 ms per step | R ≈ 3–5 ms per step | ~0 |

### 3.5 Tests and measurements

- **Correctness:**
  - With a fixed draft pattern, greedy output must be identical with and without graphs on the same build
    (`GGML_CUDA_DISABLE_GRAPHS=1`): graphs replay the same kernels. Layer 1 and Layer 2 must not change a token.
  - `test-backend-ops` is unaffected.
  - Port #28666's `test-cuda-graph-cache` (alternating shapes, counts warm-up resets) into the fork's tests.
- **Counting re-captures.** `GGML_CUDA_GRAPH_DEBUG=1` with `llama-server -v` logs "CUDA graph warmup complete" and
  "warmup reset". Per request they should drop from one per switch to single digits with Layer 1.
- **Counting rebuilds.** The server's timing line prints `graphs reused`. With Layer 2 it should approach the step count.
- **Host timeline.** `GGML_CUDA_PROFILE_HOST=<file>` logs host entry and exit and GPU time per graph evaluation and card.
  - The gap between the last card's exit and the first card's entry of the next step holds sampling, drafting, the
    rebuild and `set_inputs`.
  - The gap between one card's exit and the next card's entry holds that card's input copies.
  - Its events slow the token itself (37.4 → 23.4 tok/s for S4-L0), so compare only runs that both use it.
- **Alternating-shape microbenchmark** (~80 lines, part of the Q52 work). It calls `llama_decode` with fixed patterns
  (1, 3, 1, 3 …; 10 × 1 then 10 × 8; 1, 1, 8) on the real model, rolls back with `seq_rm`, and times each step. It
  isolates the switching cost from acceptance.
- **Reading Q19's numbers.** The synth part of `verify_cost.sh` changes the verify width whenever the 1-token key finds a
  shorter continuation, so until Layer 1 lands its step times include re-captures. The `bench` part (llama-bench pp1…pp8)
  keeps one width per test and runs captured, but computes the output head for one row only.

## 4. Draft-model-free speculation in llama.cpp today

### 4.1 The drafter types

| Type | Lookup | Draft length | Defaults and flags | Code (fork; master) |
|---|---|---|---|---|
| `ngram-simple` | Most recent earlier occurrence of the last `size_n` tokens (incl. the sampled one); never overlaps the suffix; position 0 never matches | min(`size_m`, tokens after the match); no draft if fewer than `size_n` follow | 12 / 48; `--spec-ngram-simple-size-n/-m` | `ngram-map.cpp:49-112`; `speculative.cpp:1877-1920` (master :1934) |
| `ngram-map-k`, `ngram-map-k4v` | Hash of `size_n`-grams; k4v keeps up to 4 continuations and needs the top one to be ≥ 2× the sum of the others | min(`size_m`, the key's last accepted count): shrinks after a miss, never grows past the last draft | 12 / 48 / min-hits 1 | `ngram-map.cpp:221-516`, accept `:518-536` |
| `ngram-mod` | 4 M-entry hash of 24-token contexts → next token, chained token by token, so it can follow its own output; reset after 5 rounds below 25 % acceptance | Chains until a miss; no draft if under `n_min` | 24 / 48 / 64; `--spec-default` turns it on | `speculative.cpp:1976-2149` (master :2033) |
| `ngram-cache` | 1–4-gram counts over the context, a dynamic cache file and a static one (`-lcs`, `-lcd`), with minimum counts and shares per level | Up to 8, hard-coded; stops at the first token without enough support | `ngram-cache.cpp:60-63` | `speculative.cpp:2151-2297` (master :2208) |
| SuffixDecoding (`ngram-suffix`) | Online suffix tree of the request, most frequent continuation | min(`n_max`, match length × `max_factor`), stops below `min_prob`; at least `n_min` | max depth 24, `n_max` 24, `n_min` 3, factor 1.0, `min_prob` 0.1 | open PR #26283 |

Learned drafters (`draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`) follow in priority after the
n-gram types and need `-md`.

### 4.2 Dispatch

- **Priority.** The enabled types are tried in a fixed order: `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
  `ngram-mod`, `ngram-cache`, then the learned drafters (`speculative.cpp:2748-2762`, master :2808). The first non-empty
  draft wins (`:2959-2989`, master :3016).
- **Truncation.** Drafts longer than `dp.n_max`, the room left before the context end or `max_tokens`, are cut
  (`:2963-2973`; `server-context.cpp:565-584`).
- **Feedback.** The accepted count goes to the drafter that produced the draft. The others get `is_other = true`
  (`speculative.cpp:3008-3043`).
- **Statistics.** The server keeps acceptance per draft position and prints `draft acceptance`. The API returns
  `draft_n` and `draft_n_accepted` in `timings`, which [server_bench.sh](../../../scripts/server_bench.sh) already reads.

### 4.3 Verification and exactness

- **Sample-and-match.** n-gram drafts carry no distribution, so the server verifies them by sample-and-match: sample each
  position from the target and keep the draft token while they agree (`sampling.cpp:690-722`; `server-context.cpp:4808-4821`).
  That is exact speculative sampling for a point-mass proposal at any temperature: the draft token is kept with
  probability p(draft), and a mismatch is a sample from the target without the draft token.
- **Grammars and end of generation.** A grammar applies inside the same sampling call. After an end-of-generation token
  the rest of the draft is dropped (`sampling.cpp:707-711`).
- **A fixed regression.** Since PR #29924 (merged 2026-10-04, in the fork's base), a truncated n-gram draft at T > 0 no
  longer switches the verifier to rejection sampling against empty candidate lists.
- **Rollback.** Flint's sliding-window cache supports partial sequence removal (the server's probe,
  `common.cpp:1574-1612`), so rejected tokens are removed in place, without checkpoints or replay.

### 4.4 Where the acceptance arrives, and where a controller lives

- **When.** After the verify step the server samples on the host, which already synchronized the context, and calls
  `common_speculative_accept(spec, slot, n_accepted)` (`server-context.cpp:4866`, master :4889) before it builds the next
  batch. The next `draft()` call (`:3782`) sees the result with no device read.
- **Where.** A controller is a `common_speculative_impl` subclass: per-sequence state, `draft()` (`speculative.cpp:210`)
  and `accept()` (`:212`). PR #27210 and #26283 both work this way, and so does Q53.

### 4.5 Controllers upstream and in published work

- **Adaptive MTP depth**, [PR #27210](https://github.com/ggml-org/llama.cpp/pull/27210):
  - State: open since 2026-08-17, last updated 2026-10-18. An automated review on 2026-09-08 flagged an undisclosed
    change in `delta-net-base.cpp`; no maintainer review since.
  - Mechanism (`common/speculative-adaptive.h` in the PR): depth climbs one step after 2, 4, 10, 6, 3 or 2 consecutive
    fully accepted verifies at depth 1, 2, 3, 4, 5 and 6+; misses add n_draft − n_accepted to a pressure that drops a
    step at max(5 × depth, 20). Per sequence, MTP only, and every depth change is a new verify shape.
  - Results, Qwen3.8-27B Q8_0 on two Radeon R9700, greedy: adaptive 3–12 against fixed 3 gives 51.4 / 56.6 / 78.0 /
    147.5 against 51.7 / 55.9 / 69.3 / 81.9 tok/s on reasoning, prose, code and recall.
  - On an RTX 5090, graphs on against off changed the adaptive runs by under 3 %.
  - On two MI50 (gfx906), the adaptive range lost on fresh code: 70.4 → 48–50 tok/s.
  - The author's hindsight-optimal depth per round on reasoning and prose was "statistical white-noise".
- **[PR #25726](https://github.com/ggml-org/llama.cpp/pull/25726)** (open): grow or shrink the MTP draft by one after N
  successes or failures in a row.
- **SuffixDecoding**, [PR #26283](https://github.com/ggml-org/llama.cpp/pull/26283):
  - State: open since 2026-07-29, waiting for review since 2026-09-07.
  - Mechanism: a per-request suffix tree, no cross-request corpus.
  - A tester's numbers: without a minimum draft length, decode fell 42 → 31 tok/s on a dense 27B and 185 → 110 on an
    MoE; with `n_min = 3` it was back to 41 and 185.
- **HyperQwen's lookup controller** (`patches/dflash2-lookup-drafting.patch` at `ce9615c`):
  - Lookup: the longest match up to 12 tokens, the newest on ties. A cap of 32 preferred old long matches over new short
    ones and measured 2.69 against 3.21 tokens per step on quote-and-explain work.
  - The match may overlap the suffix.
  - The long block needs two consecutive qualifying steps (a full block emitted and a lookup hit), because one
    saturated step happens inside ordinary prose.
  - It is held 3 steps after the flag drops; without the hold a copy measured 13.9 instead of 15.2 tokens per step.
  - The decision is read one step stale from pinned memory, because a synchronous read cost 5 %.
- **SuffixDecoding paper** ([arXiv 2411.04975](https://arxiv.org/html/2411.04975)):
  - Draft cap: α × match length, α = 1–4 for agentic work.
  - AgenticSQL: 5.3×, 6.3 accepted tokens per step, 42 % of drafted tokens accepted.
  - SWE-Bench offline: 2.5×, 7.8 per step, 30 % accepted.
  - Spec-Bench without a learned drafter: 1.66×.
  - Lookup cost: ~11–12 µs per token.
- **SwitchSD** ([arXiv 2609.20186](https://arxiv.org/abs/2609.20186)): probes on the target's hidden states detect
  "copy intent" (AUC > 0.99) and switch between copying and EAGLE-3, +15 % over EAGLE-3. It needs hidden states and a
  trained probe; the acceptance-history gate needs neither.

## 5. Q53 design: the `ngram-copy` drafter

### 5.1 Parameters

| Parameter | Default | Flag | Why |
|---|---|---|---|
| `n_strong` | 6 | `--spec-ngram-copy-n-strong` | Shortest match that may draft. HyperQwen's value; PR #26283 shows that drafting on any match loses ~40 % on an MoE |
| `n_longest` | 12 | `--spec-ngram-copy-n-longest` | Match lengths are compared up to this; newest on ties (HyperQwen: 12 beat 32) |
| `k_short` | 2 | `--spec-ngram-copy-k-short` | Verify 3: on the matrix-vector path for every format; protects chat against false matches (section 6) |
| `k_long` | 7 | `--spec-ngram-copy-k-long` | Verify 8: dense products still on the matrix-vector path; 11 or 15 would need expert locality, which Flint lacks (Q51) |
| `n_enter` | 2 | `--spec-ngram-copy-enter` | Fully accepted steps in a row before the long block |
| `n_hold` | 2 | `--spec-ngram-copy-hold` | Missed long steps tolerated before falling back to short (HyperQwen held 3 where long blocks cost little) |
| `search` | 0 | `--spec-ngram-copy-search` | Tokens searched back; 0 = all |

### 5.2 Lookup

`common_ngram_copy_draft()` in `common/ngram-map.cpp`, on the history `hist` plus the sampled token `last`:

```cpp
// longest match (n_strong..n_longest) of the suffix of hist + [last], newest on ties; the match may overlap the
// suffix, and the draft reads past the end of the history into itself, so short-period text (indentation, list
// markers, repeated rows) continues with its period
static bool common_ngram_copy_draft(const llama_tokens & hist, llama_token last, int n_strong, int n_longest,
                                    int search, int k, llama_tokens & out, int & n_avail) {
    const int t  = (int) hist.size();                        // index of `last` in hist + [last]
    auto      at = [&](int i) { return i < t ? hist[i] : last; };
    const int lo = search > 0 ? std::max(n_strong - 1, t - search) : n_strong - 1;
    int best_len = 0, best_end = -1;
    for (int e = t - 1; e >= lo; --e) {                       // candidate match ends, newest first
        if (at(e) != last) continue;
        int len = 1;
        while (len < n_longest && e - len >= 0 && at(e - len) == at(t - len)) ++len;
        if (len >= n_strong && len > best_len) {
            best_len = len; best_end = e;
            if (len == n_longest) break;
        }
    }
    if (best_end < 0) return false;
    n_avail = t - best_end;            // continuation inside hist + [last]; if < k the draft repeats it (period n_avail)
    out.resize(k);
    for (int i = 0; i < k; ++i) {
        const int src = best_end + 1 + i;
        out[i] = src <= t ? at(src) : out[src - t - 1];
    }
    return true;
}
```

- **Cost.** The scan is O(history) per step, ~0.1 ms at 32k on the Celeron (est.; SuffixDecoding measures ~11 µs per
  lookup with a tree). A hash of each 6-gram's last position, like `ngram-mod`'s table, makes it constant if 128k-token
  agent contexts need it.

### 5.3 States

| State | On a match | On no match | After the verify |
|---|---|---|---|
| short | draft `k_short` | no draft (1-token step) | `n_full` += 1 if all accepted, else 0; at `n_enter` go long with `hold = n_hold` |
| long | draft `k_long` if `n_avail ≥ k_long` or the match overlaps its own suffix (period `n_avail` ≤ match length), else `k_short` | no draft; back to short if `hold` is 0, else `hold` −= 1 | all accepted: `hold = n_hold`; otherwise back to short if `hold` is 0, else `hold` −= 1 |

- **Slots.** With more than one slot drafting in the same call (`dparams` has several `drafting` entries), only one
  slot drafts, with `k_short`, taking turns (section 3.3).
- **Room.** If `dp.n_max < k`, use the next smaller fixed length that fits, or no draft.
- **Reset.** `begin()` resets the state for a new request.

### 5.4 Code changes (fork)

- **`common/common.h`:** `COMMON_SPECULATIVE_TYPE_NGRAM_COPY` before `_COUNT`; `common_params_speculative_ngram_copy`
  with the fields above; a member in `common_params_speculative`.
- **`common/arg.cpp`:** the six flags, next to the other n-gram flags (`:4315-4481`).
- **`common/speculative.cpp`:**
  - the name `ngram-copy` in the type map (`:73-85`);
  - `static_assert(COUNT == 12)` (`:2747`) and the new type first in the priority list (`:2751`);
  - `k_long` in `common_speculative_n_max` (`:2443-2477`), which sizes `n_outputs_max_per_seq` to 1 + `k_long`;
  - the factory case;
  - `common_speculative_impl_ngram_copy` with a per-sequence {mode, n_full, hold, n_last}: `draft()` as in 5.3, ticking
    the hold on no-match steps itself (the server calls `accept()` only after a draft); `accept()` as in 5.3;
    statistics per state (steps, drafted, accepted) in `common_speculative_print_stats`.
- **`common/ngram-map.{h,cpp}`:** the lookup.
- **`tests/test-ngram-copy.cpp`:** the lookup (newest-on-ties, overlap, periodic continuation, search window) and a
  scripted state sequence. PR #27210's `test-speculative-adaptive.cpp` is a template.
- **Size:** ~250 lines; no server change. Use it alone: `--spec-type ngram-copy`.

## 6. Cost model

### 6.1 Inputs

Step cost v(n) for verifying n tokens, in T1 (one plain token on the Pascal kernels). Today every n ≥ 2 runs on llama.cpp's
stock multi-token kernels, so its base is the kernels-off token, 34.25 / 26.76 = 1.28 T1.

| Curve | v(2) | v(3) | v(4) | v(8) | v(16) | Basis |
|---|---|---|---|---|---|---|
| optimistic | 1.35 | 1.43 | 1.50 | 1.80 | 2.40 | +2 ms per extra token: two slots cost 1.03–1.13× one on the pre-fork build ([flint_speedups](flint_speedups.md) §1) |
| mid | 1.49 | 1.70 | 1.91 | 2.75 | 4.43 | +5.6 ms per extra token: ~0.7 GB of extra experts at 120–150 GB/s plus a dense column ([HyperQwen lessons](hyperqwen_lessons.md)) |
| pessimistic | 1.68 | 2.07 | 2.46 | 3.90 | 6.40 | 1.28 × the bytes read with random routing (2.24 GB → 11.2 GB at 16 tokens) at the stock kernels' efficiency |
| locality | 1.50 | 1.68 | 1.87 | 2.41 | 3.08 | 1.28 × the bytes with Qwen3-Coder-30B-A3B's measured expert sharing (51 % of random at 8 tokens) |

For comparison, Qwen3.6-35B-A3B Q4_0 (8 of 256 experts) measured 1.49 / 1.82 / 2.15 / 3.82 for 2 / 3 / 4 / 8 tokens with
llama-bench pp-n on the pre-fork build ([speed-ups](../../benchmarks/2026-10-22-speedups.md)).

Other inputs:
- **Graph costs:** U = 0.18, C = 0.35, R = 0.15 T1 (section 1.6), applied per the warm-up logic of section 1.3 for
  "today", only R on a switch for Layer 1, and once per shape for Layers 1 + 2.
- **Acceptance inside a copy:** each drafted token kept with p = 0.95 at temperature 1.0 (0.90–0.98 tried).
- **Matches in fresh text:** a 6-token match on 3–8 % of fresh steps, first drafted token accepted with 35–40 %.
- **Workloads**, as shares of output tokens in copy runs and mean run lengths (assumptions for Q64 to replace):

| Workload | Tokens in copy runs | Mean copy run | 6-token matches in fresh text |
|---|---|---|---|
| chat | 3 % | 20 | 3 % of steps |
| new code | 10 % | 25 | 6 % |
| edit that keeps context | 75 % | 80 | 8 % |
| file rewrite (2–4k tokens, one change) | 97 % | 400 | 8 % |
| agent session | 40 % chat, 15 % tool calls (30 % copy, runs of 15), 45 % edits | | |

The edit share sits at the low end of Copy-as-Decode's 74–98 % of gold tokens reachable by line-level copying on edit
benchmarks ([arXiv 2604.18170](https://www.alphaxiv.org/abs/2604.18170.md), S).

Method: a Monte-Carlo replay of 20,000-token synthetic transcripts per workload through the controller of section 5,
step by step, with the curve's step costs and the graph penalties (my calc).

### 6.2 Results

Speed relative to plain decoding with `k_short = 2`, `k_long = 7`, Q52 Layers 1 + 2 (my calc):

| Curve | chat | new code | edit | file rewrite | agent session |
|---|---|---|---|---|---|
| optimistic | 1.01 | 1.05 | 1.94 | 3.27 | 1.32 |
| mid | 1.00 | 1.02 | 1.61 | 2.27 | 1.22 |
| pessimistic | 0.99 | 0.99 | 1.34 | 1.65 | 1.13 |
| locality | 1.00 | 1.03 | 1.71 | 2.55 | 1.25 |
| mid, with 2–4-column Pascal kernels | 1.01 | 1.04 | 1.64 | 2.29 | 1.24 |

- **Absolute speed.** At the 32 tok/s chat baseline: chat ~32, new code 32–34, edits 43–62, rewrites 53–105 tok/s,
  agent sessions 36–42 tok/s; with `k_long` = 15 up to 65, 121 and 43.
- **`k_long` = 15.** Rewrites 3.79× (optimistic), 3.08× (locality), 2.30× (mid), 1.66× (pessimistic). The longer block
  pays only with locality or a cheap verify.
- **`k_short`.** 1, 2 or 3 changes copy workloads by under 1 %. A shorter short block protects chat and code: with
  three times the false matches, chat goes to 0.97 with `k_short = 3`, and at the pessimistic curve chat is 0.976 with
  3 against 0.986 with 2.
- **The hold.** No hold costs 2–5 % on edits and rewrites. A hold of 3 is no better than 2.
- **`n_strong`.** 4, 6 or 8 moves results by ≤ 2 %; the gate does the work.

Graph caching at the mid curve (today / Layer 1 / Layers 1 + 2):

| Drafter | chat | new code | edit | file rewrite | agent session |
|---|---|---|---|---|---|
| `ngram-copy` | 0.97 / 0.99 / 1.00 | 0.97 / 1.01 / 1.02 | 1.53 / 1.59 / 1.61 | 2.22 / 2.26 / 2.27 | 1.17 / 1.21 / 1.22 |
| ungated 4-token drafts on any short match | 0.76 / — / 0.88 | 0.81 / — / 0.93 | 1.43 / — / 1.54 | 2.03 / — / 2.05 | 1.00 / — / 1.12 |

With the high end of the host costs (U 0.25, C 0.8, R 0.4 T1) `ngram-copy` today drops to 0.94 / 0.92 / 1.44 / 2.16 /
1.12, and Q52 is worth +3–12 %.

Without backend sampling (0.11 T1 per output row, every row of every step), relative to plain decoding under the same
condition: edits 1.48×, rewrites 1.94×, agent sessions 1.18× (mid). Q55's GPU top-k path would return most of the gap.

### 6.3 What changes the answer

- **Q19 stage 1, the verify cost of 2–16 tokens on the fork build.** It picks the curve: rewrites 1.65× at pessimistic,
  3.3× at optimistic.
  - If v(3) ≥ 2.0 T1, short blocks lose on chat and code, and `k_short` = 1 or drafting only in the long state is next.
  - If v(8) ≥ 4 T1, `k_long` drops to 5 (verify 6, still on the matrix-vector path for Q5_0 gate/up).
- **Q51, expert overlap of neighbouring tokens (measured).**
  - Over 65,600 generated tokens (chat, code, whole-file edits, agent turns), 2, 4, 8 and 16 consecutive tokens touch
    11.1, 20.1, 35.6 and 59.7 distinct experts per layer, against 11.9, 23.2, 44.2 and 80.8 with random routing: 80 % of
    random at 8 tokens, where Qwen3-Coder-30B-A3B is at 51 %. Whole-file edits route no differently (36.1 at 8 tokens).
  - So the "locality" curve does not apply, `k_long` stays at 7, and a deduplicating multi-token `MUL_MAT_ID` kernel
    would save only 8–26 % of the routed-expert bytes of a 2–8-token verify.
- **Q64, copy share and run lengths of Flint's own output.** It replaces the workload table. The agent result moves
  roughly linearly with the copy share: 1.22× at 45 % edits.
- **Q55, backend sampling for tool requests.** Without it, agent sessions lose ~4 points of the gain.
- **Q65, mask copies at depth.** At 32k context an 8-token step copies 512 KiB to each of six cards, card after card; at
  128k with 16 tokens the copies alone may exceed the GPU work. Long blocks may need a depth cap, or the masks a GPU-side
  build.

## 7. Experiments, ranked by value for effort

| Rank | Experiment | Where, effort | Expected | Decides |
|---|---|---|---|---|
| 1 | **Q64** offline replay of drafters on Flint transcripts (with Q51's router dumps on the same text) | quant machine 1–2 h generation; any PC, ~1 day of script | statistic | whether Q53 is worth building; `k_short`, `k_long`, `n_strong`, hold; per-workload gain with Q19's costs |
| 2 | **Q19 stage 1 + Q65** step costs v(1…16) at depth 0 and 32k; U and R by A/B; mask-copy gaps | rig ~2.5 h | measurement | the cost curve; Layer 2 or not; depth cap for long blocks |
| 3 | **Q52 Layer 1**: cherry-pick #31004 with the compute-6.1 gate, profiler key ported, alternation microbenchmark | fork 0.5–1 day, rig 1 h | +2–6 % with `ngram-copy` (up to +12 %), +8–31 % for ungated drafting, ~0 plain | whether the CUDA half suffices |
| 4 | **Q53** `ngram-copy` | fork 1.5–2 days, rig 3 h | chat ±1 %, edits 1.3–2.0×, rewrites 1.65–3.8×, agent sessions 1.1–1.35× | the default speculation setting for Flint serving |
| 5 | **Q52 Layer 2**: per-shape graphs and schedulers in llama-context | fork 2–3 days | +1–2 % on top of Layer 1; more with `-np 2` and drafter models | only if R × switches per token ≥ 2 % (rank 2) |
| 6 | **Q55** GPU top-k for tool requests (existing row) | fork ~2 days | ~4 points of agent gain, plus the plain-token gain | agent sessions |
| 7 | Mask built on the GPU from positions (only if Q65 shows ≥ 5 % at 32k) | fork 1–2 days | depth-dependent | long blocks at long context |

### Commands

Placeholders: `B` is the fork build's `bin/` on the rig, `M` the S4-L0 file, six cards. S4-L0 is rescale-free, so every
run sets `LLAMA_NO_MOE_DOWN_RESCALE=1`.

**Rank 2: step costs and host costs (Q19 stage 1, Q65).**

```bash
B=/opt/llama-builds/<fork build>/bin; M=/mnt/models/flint-S4L0.gguf; G=0,1,2,3,4,5
E="LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=$G"
# v(n) at depth 0 and 32k, one width per test (captured); pp-n computes the head for one row only
ssh rig "$E $B/llama-bench -m $M -ngl 99 -fa on -ub 128 -p 1,2,3,4,6,8,12,16 -n 0 -d 0 -r 5"
ssh rig "$E $B/llama-bench -m $M -ngl 99 -fa on -ub 128 -p 1,4,8,16 -n 0 -d 32768 -r 2"
# U (graphs off) and R (rebuild every step; graphs stay captured because the rebuilt graph is identical)
for env in "" GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_GRAPH_REUSE_DISABLE=1; do
  ssh rig "$env $E $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 5"
done
# mask copies: host timeline of 8-token steps at depth 0 and 32k (gaps between cards = input copies)
ssh rig "mkdir -p /tmp/q63 && $E GGML_CUDA_PROFILE_HOST=/tmp/q63/d0.csv $B/llama-bench -m $M -ngl 99 -fa on -ub 128 \
    -p 8 -n 0 -d 0 -r 10 && $E GGML_CUDA_PROFILE_HOST=/tmp/q63/d32k.csv $B/llama-bench -m $M -ngl 99 -fa on \
    -ub 128 -p 8 -n 0 -d 32768 -r 3"
# real verify steps (n-gram drafts with synthetic acceptance) for widths 1, 3 and 8, as in Q19
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M GPUS=$G PARTS='bench synth' WIDTHS='1 3 8' TAG=q19_s4l0 \
    bash -s" < bench/eagle3_flint/verify_cost.sh
```

**Rank 3: Q52 Layer 1, before and after.**

```bash
# warm-up resets and reused graphs, with a 3-token-key n-gram drafter that changes the verify width often;
# B52 is the same build with Layer 1
for run in "before|$B" "after|$B52"; do
  ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 GGML_CUDA_GRAPH_DEBUG=1 BIN=${run#*|} MODEL=$M GPUS=$G TAG=q52_${run%%|*} \
      PARTS=ngram NGRAM_ARGS='--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 4' \
      SERVER_ARGS='-v' bash -s" < bench/eagle3_flint/verify_cost.sh
done
ssh rig 'for d in /tmp/eagle3_flint/q52_*; do echo $d; grep -c "warmup reset" $d/server_*.log; grep "graphs reused" $d/server_*.log | tail -1; done'
```

**Rank 4: Q53 against the existing drafters.** Chat at Flint's own sampling (no-regression check) and the edit prompt:

```bash
for cfg in "none|--spec-type none" "simple|--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 4" \
           "mod|--spec-type ngram-mod" "copy|--spec-type ngram-copy"; do
  ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M GPUS=$G TAG=q53c_${cfg%%|*} PROMPTS=chat MAX_TOKENS=2048 \
      SERVER_ARGS='-c 32768 -b 512 -ub 128 -bs --ctx-checkpoints 4 ${cfg#*|}' bash -s" < scripts/server_bench.sh
  ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M GPUS=$G TAG=q53e_${cfg%%|*} PARTS=ngram \
      NGRAM_ARGS='${cfg#*|}' bash -s" < bench/eagle3_flint/verify_cost.sh
done
```

`verify_cost.sh` falls back to its default drafter when `NGRAM_ARGS` is empty, hence `--spec-type none` for the
baseline; its edit prompt runs at temperature 0. `server_bench.sh` still needs two prompt sets for this: a 2–4k-token file returned with one function renamed, and a
replayed agent session with tool output. Both must run at temperature 1.0, and the agent set with and without `tools`
(backend sampling off and on).

**Rank 1: Q64 offline replay** (a new script under `bench/`):
- **Transcripts.** Generate at Flint's default sampling with fixed seeds on the quant machine through llama-server's
  `/completion` with `"return_tokens": true`. Cover: the four chat prompts, the five-turn chat, rewrite prompts, edit
  prompts, and two or three replayed coding-agent tasks with their recorded tool outputs. Save prompt and output token
  ids.
- **Replay.** For each output position, draft from prompt + output so far, accept the longest prefix that equals the
  realized continuation, advance by 1 + accepted, and log the verify width and acceptance. Run it for `ngram-simple`
  3/4 and 12/48, `ngram-mod`, `ngram-cache` and `ngram-copy` over a parameter grid.
- **Outputs:**
  - tokens per step and the width histogram per workload;
  - copy share and run lengths;
  - with Q19's v(n) in ms, the predicted tok/s for each drafter and setting;
  - with the router dumps (Q51), distinct experts per actually verified block.

## Sources

The fork, branch `pascal-p106` at `eeaa59660` on upstream `d77256999`, and upstream master at `36e36e6` (2026-10-23) (P):
- `ggml/src/ggml-cuda/ggml-cuda.cu`: graph key, property check, warm-up, capture, update, compatibility, fusion rules,
  the fork's profilers.
- `ggml/src/ggml-cuda/common.cuh` (graph cache), `mmvq.cu` (expert batch limits), `mmvq-pascal.cu` (one-column gate).
- `ggml/src/ggml-backend.cpp`: input placement, synchronous input copies, split uids, context buffer size.
- `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-graph.cpp`, `src/llama-graph.h`: reuse, arenas, sampling
  chains, mask construction.
- `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp`: n_kv padding, ubatch splits.
- `common/speculative.cpp`, `common/speculative.h`, `common/ngram-map.cpp`, `common/ngram-cache.cpp`,
  `common/sampling.cpp`, `common/common.h`, `common/common.cpp`, `common/arg.cpp`.
- `tools/server/server-context.cpp`, `tools/server/server-schema.cpp`.

llama.cpp pull requests and issues (P): [#14482](https://github.com/ggml-org/llama.cpp/pull/14482),
[#21764](https://github.com/ggml-org/llama.cpp/pull/21764), [#25726](https://github.com/ggml-org/llama.cpp/pull/25726),
[#26283](https://github.com/ggml-org/llama.cpp/pull/26283) (with its review thread),
[#27210](https://github.com/ggml-org/llama.cpp/pull/27210) (with its benchmark thread),
[#27721](https://github.com/ggml-org/llama.cpp/pull/27721), [#28549](https://github.com/ggml-org/llama.cpp/pull/28549),
[#28652](https://github.com/ggml-org/llama.cpp/issues/28652), [#28666](https://github.com/ggml-org/llama.cpp/pull/28666),
[#29768](https://github.com/ggml-org/llama.cpp/pull/29768), [#29924](https://github.com/ggml-org/llama.cpp/pull/29924),
[#30223](https://github.com/ggml-org/llama.cpp/pull/30223), [#30804](https://github.com/ggml-org/llama.cpp/pull/30804),
[#31004](https://github.com/ggml-org/llama.cpp/pull/31004) (with the 2026-10-23 maintainer comment).

Other (P unless marked S):
- HyperQwen at `ce9615c`: `patches/dflash2-lookup-drafting.patch` (lookup kernel, `next_num_draft_tokens`, the NMAX,
  STICKY and stale-flag measurements) and `docs/gotchas.md` (CUDA graphs for both block lengths, 27.9 against 25.9 ms
  per step).
- [arXiv 2411.04975](https://arxiv.org/html/2411.04975): SuffixDecoding.
- [arXiv 2609.20186](https://arxiv.org/abs/2609.20186) (S, abstract only): SwitchSD copy-intent probes.
- [arXiv 2604.18170](https://www.alphaxiv.org/abs/2604.18170.md) (S): Copy-as-Decode, reachable tokens on edit
  benchmarks.

This repository:
- [HyperQwen lessons](hyperqwen_lessons.md): the verify-cost table, the copy-gated controller, CUDA graphs per length,
  backend sampling for tool requests.
- [flint_speedups.md](flint_speedups.md): n-gram types and flags, the two-slot verify-cost estimate.
- [pascal_mmvq_integration.md](pascal_mmvq_integration.md): the one-token kernels, per-launch profile, host timeline (Q49).
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md): CUDA graphs +14 %, depth speeds, two slots.
- [speed-ups](../../benchmarks/2026-10-22-speedups.md): pp-n step costs of Qwen3.6 and Qwen3.8.
- [small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md): graph sizes and CPU sensitivity on the rig.
- [quants/experiments.md](../../../quants/experiments.md): Q19, Q38, Q49, Q51–Q55.
- [verify_cost.sh](../../../bench/eagle3_flint/verify_cost.sh), [server_bench.sh](../../../scripts/server_bench.sh).
