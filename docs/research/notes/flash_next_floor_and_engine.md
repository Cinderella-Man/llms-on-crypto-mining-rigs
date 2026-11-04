# Flash-Next on nine P106s: the bandwidth floor, where the gap goes, and whether an own engine would close it (2026-11-04)

Qwen3.8-Flash-Next (Swift-1.5 GSQ-RCO Q2_0) decodes a token in 82.4 ms on nine P106-090 cards with the night's Pascal
kernels (`-ladder3`; [k-curve](../../data/2026-11-04_kcurve_p106/README.md)). This note computes the memory-bandwidth
floor of that token from the file's tensor census, splits the distance to the floor into slices with the evidence behind
each, and prices three ways to shrink it: more fusion inside the fork, a Flash-Next-only executor inside the fork, and a
standalone engine. The question is the owner's: whether to rework the CUDA side outright, since "we don't need it to
support all the cards in the world". It reads [Strata](strata.md), [uzu](uzu_lessons.md),
[the slim-fork study](pascal_slim_fork.md), [the megakernel note](megakernel.md), [ninfer](engine_infer.md) and
[HyperQwen](hyperqwen_lessons.md) against the measurements of 2026-11-03/04.

Conventions:
- "est." marks this note's arithmetic or inference; the basis is next to it. Numbers without it are measured or quoted
  from the linked source.
- Nothing ran for this note. The k-pass profile (registry Q822, `docs/data/2026-11-04_kpass_profile/` once it lands)
  was running on the rig when the note was written. Section 2.5 lists the numbers that wait for it, and every ledger
  row below is a bottom-up estimate that the profile replaces.
- Hardware, from [the hardware spec](../../mining_rig.md) and the bench folders: 5 SMs and 1.5 MB of L2 per card; 192 GB/s
  theoretical and ~173 GB/s measured streaming (178–179 in the benches); 1.96 TFLOPS FP32 at the 1,531 MHz boost clock
  (~2.2–2.4 at the 1,700–1,911 MHz the cards reach under load); dp4a, no tensor cores; PCIe 1.0 x1 links at ~189 MB/s
  host to GPU and ~213 MB/s GPU to host; no peer access; a 2-core Celeron host.
- `-ladder3` at its defaults holds the Pascal Q2_0 expert kernel (Q223), the Pascal BF16 kernel (Q546), the k-column
  and multi-token kernels (Q591; at one token only Q3_K and short Q8_0 rows take the k-column path), the recurrent-layer
  fusions (Q436–Q438), the IQ-type kernels and per-size graphs
  ([merge](../../data/2026-11-04_ladder3_merge/README.md)); the hop casts and the sparse-attention kernel are off.

## Summary

- **The floor is 24.0 ms of weights, 28.7 ms with the recurrent state and the hops' bytes (34.8 tok/s).** A token reads
  4,155 MB (attention and recurrent projections 34 %, hyper-connection projections 31 %, routed experts 16 %, output
  layer 11 %, router 6 %, shared expert 3 %), and a layer split runs one card at a time, so one card's 173 GB/s is the
  whole bandwidth: 24.0 ms at 173 GB/s, 21.6 at the 192 GB/s peak. The state read and written in place adds 1.4 ms, the
  residual's bytes over the x1 links 3.3 ms. With the residual in Q8_0 the floor is 26.4 ms (37.9 tok/s). At 8–32k of
  context the KV cache adds est. 0.4–1.5 ms to the floor (and 6–38 ms to the dense kernel that runs today).
- **82.4 ms is 2.9× the floor (3.4× the weights alone).** The cards' own timelines hold est. ~66 ms and stream at ~36 %
  of 173 GB/s. The k-curve has the same shape: a fixed part of 49.6 ms a pass (floor est. 21.6) and a slope of 33.2 ms a
  token (floor est. 8–11).
- **The slices of the 53.7 ms gap (est.):**
  1. **Weight products below bandwidth, ~24 ms** (range 21–27): at their measured kernel speeds the products take
     48.4 ms, 50 % of bandwidth on average. Two thirds of the excess is one part: the attention and recurrent
     projections take ~23.8 ms for 8.1 ms of bytes, and their 475 MB of Q3_K run at ~35 GB/s on the k-column kernel's
     one-column path at its default launch shape ([k-column bench](../../data/2026-11-04_kcurve_p106/README.md), Q823).
  2. **Outside the cards, ~13 ms above the hop bytes** (range 12–16): the logits copy of llama-bench (4.7 ms, absent
     from a served token, which samples on the card), the hops beyond their bytes (0.4), and est. 7–10 ms of host work
     and waits that nothing has split yet.
  3. **Operations without weights, ~9 ms** over the state's bytes (range 7–12), of which launches only ~2–3.
  4. **In-model excess, ~7 ms** (range 3–12): what the cards' timeline holds beyond the products' bench speeds and the
     weightless work, about 8 µs a weight call (Flint's llama.cpp kernels showed 13–33 µs; candidates: clock state,
     cache state, graph scheduling).
- **Launch and node cost is spread over those slices and is the smallest honest lever.** ~2,900 kernels at 1.7–2.3 µs
  a graph node are 4.9–6.7 ms in all; the best persistent kernel measured takes 72 % of a graph node's time, so a
  persistent design nets ≤ 2.3 ms ([C140](../../data/2026-11-04_megakernel_rig/README.md)). The one persistent kernel
  built so far (Flint's MoE layer) saved exactly its removed launches and nothing more.
- **Ranking (est., k = 1, short context, against 82.4 ms):**
  - first, kernel work every option needs: **−6 to −14 ms, 1–3 weeks**;
  - (a) more fusion in the fork: **−4 to −8 ms, 3–5 weeks, low risk**;
  - (b) a Flash-Next-only executor inside the fork: **a further −3 to −7 ms, 6–10 weeks, medium-high risk**;
  - (c) a standalone engine: **a further −1 to −3 ms, 20–40 weeks, high risk**.

  The largest slice is kernel speed, which an own engine would have to write again before it gained anything.
  Host-side generality, the thing a rewrite removes first, cost under 1 ms on Flint once CUDA graphs ran. The only
  no-matrix-core port of a dedicated engine (ninfer on gfx906) ran at 0.52–0.66× llama.cpp's speed, and its 4-token
  check cost 2.3× a plain token, where this rig's costs 2.1×.
- **Recommendation:** do the kernels and (a) now, in the order of the per-op profile, starting with Q823's launch shapes
  (the Q3_K products alone est. −6 ms). Build (b) only after two one-card measurements pass: the in-model excess
  reproduced in a chained bench (Q866), and a fused real layer ≥ 15 % faster than the fork's graph for the same layer
  (Q867). Keep (c) parked (Q466); it reopens only if Q869 finds ≥ 5 ms of host time a pass that the fork cannot remove.
- **Larger levers sit outside the engine question:** sparse attention at agent depth (est. −6 ms at 8k, −15 at 16k, −34
  at 32k), the Q8_0 hop (est. −2.6 ms), and the slope of the k-curve, which decides what a draft pays.

## 1. The floor

### 1.1 Bytes a token reads, by part

Swift-1.5 Q2_0, one decoded token, from the GGUF headers ([file choice](../../data/2026-11-04_file_choice/README.md),
[token_bytes.tsv](../../data/2026-11-04_file_choice/token_bytes.tsv)): every dense tensor once, the routed experts at 10
of 512 per layer, 48 layers (36 Gated DeltaNet, 12 attention).

| Part | Tensors | MB a token | Floor at 173 GB/s (ms) | Share |
|---|---|---:|---:|---:|
| Attention and recurrent projections | `attn_*`, `ssm_*`: Q3_K 475, IQ4_XS 428, Q4_K 221, Q6_K 82, Q5_K 80, BF16 57, Q2_0 43, F32 6 | 1,392 | 8.05 | 33.5 % |
| Hyper-connection projections | `hc_*` BF16, four a layer (10,240 ↔ 320), injects | 1,283 | 7.42 | 30.9 % |
| Routed experts | 10 of 512: gate/up Q2_0 442, down Q2_0 221 | 664 | 3.84 | 16.0 % |
| Output layer | Q5_K, 248,320 × 2,560 | 437 | 2.53 | 10.5 % |
| Router | `ffn_gate_inp` F32, 2,560 × 512 | 252 | 1.46 | 6.1 % |
| Shared expert | Q3_K 24, Q2_0 23, Q4_0 15, Q4_K 11, IQ4_XS 10, Q5_0 8, IQ4_NL 7, Q8_0, Q5_K and Q6_K 10 | 106 | 0.61 | 2.6 % |
| Per-layer embedding projections | BF16, Q2_0 | 21 | 0.12 | 0.5 % |
| **Weights** | | **4,155** | **24.02** | |

The dense tensors are 84 % of the bytes; the routed experts that give the model its name are 16 %. The hyper-connection
projections alone read twice the routed experts.

### 1.2 What else moves

| Traffic | Per token | Floor | Notes |
|---|---|---:|---|
| Gated DeltaNet state and conv window, read and written in place | 36 layers × 3.27 MB × 2 = 236 MB | 1.36 ms | 3.15 MB of FP32 state (48 heads × 128 × 128) and 0.12 MB of conv window a layer. Before the fusions the state's gather copy added a third pass over it, which Q437 removes |
| Residual between cards | 8 boundaries × 40,960 B, two x1 legs each | 3.3 ms | 192 µs GPU to host plus 217 µs host to GPU a boundary at the measured link rates; Q8_0 blocks (10,880 B) bring the bytes to 0.9 ms |
| N-gram table rows | 16 rows × ~90 B, in 4 KiB pages | 0.15–0.87 ms of SSD latency | 3.6–5.7 pages read from disk a generated token with the page cache ([variants §5](flash_next_variants.md)); latency, not bandwidth, and absent from llama-bench |
| Logits to the host | 993 KB (248,320 × 4 B) | 4.7 ms at 213 MB/s | llama-bench pays it every pass; a served token samples on the card (backend sampling, and Q67's GPU sampling for tool requests) and copies a token id |
| Token embedding row | 2,560 values | ~0 | |
| KV and indexer cache | see below | see below | zero at the k-curve's empty context |

The KV term grows with context. The 12 attention layers cache 2 KiB of K and V a cell each (2 KV heads × 256 × 2 B × 2),
and the indexer the rest of the ~30 KiB a token. The dense Pascal kernels read every cell and mask most away; the
indexer keeps 2,051 cells a query ([QSA](../../data/2026-11-04_qsa_sparse_attn/README.md)). Q210 measured the dense
path at ~21 GB/s, 1.17 ms per 1,000 cells over the 12 layers:

| Context | K and V of every cell (12 layers) | Dense floor at 173 GB/s | Dense kernel today (est., 21 GB/s) | Only the selected 2,051 cells + indexer scan, floor (est.) | After Q230 (its est.) |
|---:|---:|---:|---:|---:|---:|
| 5k | 126 MB | 0.73 ms | 6.0 ms | 0.34–0.47 ms | 3.0 ms (−3) |
| 16k | 403 MB | 2.3 ms | 19 ms | 0.43–0.87 ms | 4 ms (−15) |
| 32k | 805 MB | 4.7 ms | 38 ms | 0.58–1.45 ms | 4 ms (−34) |
| 128k | 3,221 MB | 18.6 ms | 153 ms | 1.45–4.9 ms | 8 ms (−145) |

The selected cells are 48 MiB (0.29 ms); the indexer's scan is 1.5 KiB a cell if it reads pooled keys and up to 6 KiB
if it reads every cell's row. At agent contexts of 8–32k the attention floor adds ≤ 1.5 ms to the weights' 24; the
dense kernel adds 10–38 ms. That gap is kernel work (Q230), not an engine property. The attention mask the host copies
to each card grows with context too (2 B a cell and card, est. ~3 ms a token at 32k on nine cards; the device mask of
Q116 and Q550 removes it).

### 1.3 The floor in milliseconds

| | ms | tok/s |
|---|---:|---:|
| Weights at 173 GB/s | 24.0 | 41.6 |
| Weights at the 192 GB/s peak | 21.6 | 46.3 |
| + state in place (1.4) + F32 hop bytes (3.3) | **28.7** | **34.8** |
| The same with the residual in Q8_0, staged (the measured hop, 1.0) | 26.4 | 37.9 |
| At 192 GB/s, Q8_0 hops | 23.8 | 42.0 |
| **Measured, `-ladder3`, one token (llama-bench)** | **82.4** | **12.1** |

The floor is one card's bandwidth, not nine: the layer split runs the cards one after another for a token, and tensor
parallelism over x1 links costs more than it saves. Nine cards would read the same bytes in 2.7 ms if the links allowed
it; they do not, so what the other eight cards could add is idle capacity for guesses (pipelined speculation,
[idle cards](idle_cards.md)), not a lower floor for one token. The floor itself moves only with the bytes: the
hyper-connections in a cheaper format (Q604, Q620), a shortlist output layer (Q619) or fewer experts (Q618, Q860). The
router in a 16-bit type (Q865) halves its bytes and, more to the point, moves it onto a fast kernel.

### 1.4 The k-curve through the same lens

A least-squares line through T(1)–T(8) (82.4, 122.1, 149.4, 173.0, 217.1, 249.1, 280.1, 319.1 ms) is **49.6 ms + 33.2
ms a token**.

| | Fixed part of a pass | Each extra token |
|---|---:|---:|
| Measured | 49.6 ms (44.9 without the logits copy) | 33.2 ms |
| Floor (est.) | 21.6: dense weights 3,492 MB at 173 GB/s (20.2) + state 1.4 | 8–11: ~630 MB of new experts (3.6), hop bytes (3.3, or 0.9 in Q8_0), weightless work (~1–2), k-column arithmetic (~0–2) |
| Ratio | 2.1× | 3–4× |

llama-bench's passes use random token ids, so each extra token routes independently: the union grows to 48.1 experts a
layer at k = 5 ([MoE speculation](moe_speculation_union.md)), ~9.5 new ones a token. Consecutive tokens of real text
share more (est. 68–77 % of independent routing at 4 tokens), so a served verify's expert term is ~1 ms a token lower.

Both parts sit 2–4× above their floors, so the gap is not specific to the one-token case. Two consequences for any
engine decision. A saving that is fixed per pass (launches, the host, the logits copy where it applies) is worth 1/k per
verified token, ~1/3 at DFlash2's ~3 tokens a step. And the slope, not the fixed part, limits what a draft pays: T(4) =
173 ms is 43 ms a token, and the asymptote is ~33 ms. The k-column bench puts the dense products' growth at ~7–8 ms a
token at today's launch shapes (on the retype's shapes, k = 2–8,
`kcol_best.tsv` in the [k-curve folder](../../data/2026-11-04_kcurve_p106/README.md)); experts, hops, weightless work and the recurrent
update share the rest, which the k = 5 profile splits.

## 2. Where the 82.4 ms goes

### 2.1 The ledger

Bottom-up from the old base's profile, the kernel benches and the measurements of 2026-11-04. The k-pass profile
replaces it.

| Slice | est. ms | Range | Basis |
|---|---:|---:|---|
| **Floor** | **28.7** | | section 1.3: weights 24.0, state 1.4, hop bytes 3.3 |
| A. Weight products below bandwidth, at their measured kernel speeds | 24.4 | 21–27 | 48.4 ms of products against 24.0 of bytes (50 % of 173 GB/s on average); per part in 2.2 |
| B. In-model excess beyond the bench speeds | 7 | 3–12 | what the cards' timeline leaves after the products, the weightless work and the floor (cross-check below) |
| C. Operations without weights, above the state's byte floor | 9 | 7–12 | 2.3 |
| D1. Hops beyond their bytes | 0.4 | 0.3–1 | 0.455 ms measured a hop (3.64 a token) against 0.41 ms of bytes |
| D2. Logits copy to the host (llama-bench only) | 4.7 | 4.6–5.3 | 993 KB at 189–213 MB/s |
| D3. Host work and waits outside the cards' timelines | 8.2 | 7–10 | the old base's wall minus its event-corrected card time, minus D1, D2 and the hop bytes |
| **Sum** | **82.4** | | |

**Cross-check from the wall.** On the old base, llama-bench's one-token pass took 136.0 ms (profiler off) against a
profiled card-time sum of 139.6 ms ([verify cost](../../data/2026-11-03_verify_cost/README.md)). The profiler's events
cost 7.6–8.5 µs a launch group ([C140](../../data/2026-11-04_megakernel_rig/README.md)), 19–21 ms over ~2,520 groups,
so the cards' own timelines held 118–120 ms and **16–18 ms of the wall sat outside them**: the hops 3.6, the logits copy
4.7, and 7–10 of host work and waits. These terms run the same code on `-ladder3`, so its cards' timeline is est. 82.4 −
16.6 ≈ 66 ms. The products at their bench speeds take 48.4 of it and the operations without weights est. ~10.4 (the
state's 1.4 included), which leaves ~7 ms for B. The light profiler's `graph` mode measures the cards' timeline directly
(section 2.5).

The file-choice model's "timed" column (48.97 ms of products plus R13's profiled 33.48 ms without weights) lands on
82.4 too: the profiler's event cost inside that constant (~13 ms over 1,658 groups) stands in for the hops, the logits,
the host and B, which the model leaves out.

### 2.2 A and B: the weight products

Floor against the speed each kernel family reaches on a P106, in a bench or in the model (the routed experts and the
BF16 tensors at their bench speed × 0.9, the share earlier Pascal kernels kept in the model):

| Part | Floor (ms) | Est. now (ms) | Share of 173 GB/s | Kernel and evidence |
|---|---:|---:|---:|---|
| Attention and recurrent projections | 8.05 | 23.8 | 34 % | Q3_K (475 MB) on the k-column kernel's one-column path: 13.7 ms in the bench at the default launch shape (~35 GB/s), 7.6 at the best one (`kcol_best.tsv` in the [k-curve folder](../../data/2026-11-04_kcurve_p106/README.md), Q823). IQ4_XS and K-quants (917 MB) on v2's one-token kernels at 87–91 GB/s in the model: 10.1 ms |
| Hyper-connection projections | 7.42 | 9.1 | 82 % | Pascal BF16 kernel: 157–159 GB/s in the bench (91 %), 8.93 ms for all BF16 tensors against 25.5 on llama.cpp's ([kernel timing](../../data/2026-11-03_kernel_timing/README.md)) |
| Routed experts | 3.84 | 7.0 | 55 % | Pascal Q2_0 kernel: gate/up 132 GB/s (76 %), down 82 GB/s (47 %) in the bench, 6.0 ms for the routed bytes; 95.2 GB/s in the model |
| Output layer | 2.53 | 3.0 | 83 % | Q5_K at 144 GB/s |
| Router | 1.46 | 3.0 | 48 % | F32 at ~83 GB/s on llama.cpp's F32 kernel |
| Shared expert | 0.61 | 2.4 | 26 % | sixteen small products in ten types; Q3_K, Q4_0, Q5_0, IQ4_NL at 23–36 GB/s |
| Per-layer embeddings | 0.12 | 0.2 | | |
| **Total** | **24.0** | **48.4** | **50 %** | |

- **The attention and recurrent projections are the largest excess (+15.8 ms), and Q3_K is most of it (+11.0).** The
  one-column path is the k-column kernel run at k = 1, built for 2–8 columns: at its default shape (2 rows per lane
  group, the largest activation tile in 48 KB) it reads Q3_K no faster than llama.cpp's kernel did (33 GB/s). One row
  per lane group or a 32 KB tile reads it at ~62 GB/s in the bench, est. −6.1 ms a token (Q823, its in-model arms
  queued); a Q3_K one-token kernel at the K-quants' ~89 GB/s would take 2.3 ms more (Q224). The other 917 MB run at
  87–91 GB/s (51–53 %), where the same kernels reached 90–110 GB/s at Flint's shapes in the bench (Q46); nobody has
  timed them alone at Flash-Next's 2,560-wide shapes, which Q866's bench does. This is a kernel and file matter: an
  engine cannot change the bytes, and an own engine would start from the same kernels.
- **The routed experts are instruction-bound, not launch-bound.** At 2.25 bits and 173 GB/s a card must unpack ~0.6 T
  weights a second on ~1 T lane-instructions a second, a budget of ~1.5 instructions a weight
  ([uzu note](uzu_lessons.md), §4). The gate/up kernel at 132 GB/s sits near that limit, so the expert floor in practice
  is ~76 % of bandwidth until the core clock rises, which is where B meets the clock below.
- **The router is a kernel miss, not a byte problem.** Its 2,560 × 512 shape is the indexer's, which runs in BF16 on
  Q546's kernel; in F32 it runs on llama.cpp's F32 kernel at ~83 GB/s. As a 16-bit copy on Q546's kernel (Q218's F16
  copy was exact) it would read 126 MB at ~142 GB/s: est. 3.0 → 0.9 ms (Q865). The retype's F16 router gained nothing
  because it stayed on llama.cpp's F16 kernel (41.8 GB/s).
- **B is the least understood slice.** On the old base it was large: llama.cpp's BF16 `hc_*_up` took ~330 µs a call in
  the model against 197 µs in the bench, and Flint's kernels ran 13–33 µs a call slower in the model than in the bench.
  Q99 tested page walks (no cost: 175–184 ns a dependent load from 16 MiB to 5 GiB) and left clock state, the cache
  state the neighbouring operations leave, and graph launch scheduling as candidates
  ([Q99](../../data/2026-10-28_q99_tlb/README.md)). On `-ladder3` the wall leaves room for only est. ~7 ms of it over
  ~860 weight-bearing launch groups (~8 µs a call), so most of the old excess went with llama.cpp's kernels. Clock state
  is the one candidate with a size: idle cards fall to the 1,354 MHz base clock between their turns, Flint ran 34.0
  against 38.5 tok/s at base against boost, and Ornith, decoding one token at a time, sat at base 50 % of the time in an
  agent check (Q154). A card computes ~8 % of a Flash-Next token, so est. 2–5 ms of B is clock.
- **The weight kernels' average sizes make fixed costs matter.** ~860 calls share 4,155 MB: 4.8 MB a call, 28 µs at the
  floor. A launch ramp, a tail wave on 5 SMs and a dependent first load of ~5–10 µs each are 20–35 % of such a call,
  which is the first-order reason the small products run at half the bandwidth and the 6.6 MB hyper-connection products
  at 91 %.

### 2.3 C: the operations without weights

R13's trace counted 1,658 launch groups without weights and 30.9 ms for them, profiled
([census](../../data/2026-11-04_megakernel_census/README.md)). The profiler's own cost takes much of that:

| Piece (old base, per token) | Profiled | Est. real | Basis |
|---|---:|---:|---|
| Near-zero groups (norms, scales, gates, adds, routing, hyper-connection mixing) | 14.5 ms, 1,265 groups | 2.9–4.9 ms | 2.3 µs a graph node (census), or profiled minus 7.6–8.5 µs a group |
| State, cache and attention groups (`GET_ROWS`, `CONT`, `CONCAT`, `SSM_CONV`, `CPY`, `SET_ROWS`, delta net, flash attention) | 16.4 ms, 393 groups | ~13 | profiled minus ~8 µs a group |
| **All, before the fusions** | 30.9 ms | **~17–18** | |

On `-ladder3` the recurrent-layer fusions (Q437, Q438) remove 8 of 11 launches a recurrent layer and the state's
gather copy, and upstream's single-copy conv history (Q436, in the b12076 base) removes a 2.6 ms `CONT` group. What
remains (est.): ~2.3–4 ms of near-zero groups (~290 fewer), the delta-net update at ~3.3 ms against its 1.4 ms floor
(~90 µs a layer for a 6.6 MB read and write), flash attention 0.7 ms at an empty context, ~2–4 ms of remaining gathers,
copies and concats, and ~1.5 ms of routing, top-k and indexer groups: **~10.4 ms in all (8–13), ~9 above the state's
floor**. In Q583's k-token passes the weightless time was nearly fixed per pass (+2.1 ms a token on the old base), so it
also scales with 1/k per verified token.

**Launch cost is a minor part of C.** A small kernel costs 1.70–2.3 µs as a graph node on a P106 (6.4 µs eager), and a
grid barrier 1.05–1.32 µs. The census bounds every launch removed at no cost to 6.7 ms of ~2,900 kernels; barriers take
it to 0.9–3.8 ms. The best persistent variant on the census's recurrent-layer mix took 72 % of a graph's time a stage
(1.98 against 2.74 µs), so ≤ 2.3 ms ([C140](../../data/2026-11-04_megakernel_rig/README.md)). Flint's persistent MoE
layer, the one built, saved 27 µs a layer (198.5 → 171.3) for 12 removed launches: 2.3 µs each, exactly the launch
cost and nothing from overlap (Q88, [megakernel §2](megakernel.md)). If decode did not replay its CUDA graphs on some
cards (Q680, open), each kernel there would cost ~6.4 µs eager instead, and fixing that would come before any of this.

### 2.4 D: the cards' surroundings

- **Hops.** One token's residual takes **0.455 ms** between two cards unprofiled (the driver's staged peer copy), ~3.6 ms
  over eight boundaries; the profiler's 1.7–2.2 ms a hop included its synchronize and CSV writing. A hop is
  bandwidth-bound (~26 µs fixed + ~10 µs a KB, both x1 legs in series); a mailbox in mapped memory takes the same time
  (436 µs). Fewer bytes and overlapped legs help: the fork's staged copy in 4 chunks takes 317 µs (−30 %), and Q8_0
  blocks in 2 chunks 127 µs (−72 %): **3.6 → 1.0 ms a token**. All of it is outside the CUDA kernels and
  engine-independent ([card hops](../../data/2026-11-04_card_hops/README.md)).
- **Logits.** llama-bench copies the last row's logits (993 KB) every pass: est. 4.7 ms of each T(k). A served token
  samples on the card and copies a token id: v2.0 kept tool requests on the GPU path too (Q67). So a served one-token
  step is est. ~78 ms on today's build, and every saving below is the same in both. The head-bound shortlist (Q619)
  does not help yet: on Swift Q2_0 the bound excludes nothing and 99.7 % of rows fall back to the full head.
- **Host and waits** (est. 7–10 ms, by difference). The scheduler copies the residual with `cudaMemcpyPeerAsync`, so
  the host can queue every card's work ahead of the cards. What can still sit on the critical path: the work before
  the first card (the graph's build or its reuse check, est. 2–5 ms on the Celeron if a pass rebuilds it,
  [speculation plumbing §1.6](speculation_plumbing.md); the inputs' uploads, ~26 µs each plus their bytes), each card's
  own synchronous input copies and graph launch where the host has not run ahead, and the end-of-pass synchronization.
  With graphs replayed, the host makes ~250 driver calls a token and overlaps 75–90 % of its time with the cards
  ([slim fork](pascal_slim_fork.md)); all the host-side slimming that study found came to **under 1 ms on a 25–29 ms
  Flint token**. Nothing has split the 7–10 ms; per-phase host timers (Q869) and the `graph` profile do.

### 2.5 What waits for the k-pass profile (Q822, C266)

`kpass_prof.sh` runs `llama-bench -p 1,5 -n 0` under the light profiler's four modes with the k-column and multi-token
kernels on and off, and `kpass_parts.py` subtracts C140's 7.6 µs a span. It replaces:

| Number here | Replaced by | Mode |
|---|---|---|
| The cards' timeline (A + B + C + floor, est. 66 ms), and D3 by difference | the cards' evaluation times against the unprofiled wall | `graph` |
| The split by layer type and card; whether the recurrent and attention layers fuse equally | ms per layer and card | `layer` |
| A, B and C per part (µs a call, weight and weightless groups) | every launch group timed once in 8 passes, event cost subtracted; compared with the bench's µs at the same shapes | `sample=8` |
| Cross-check of the three light modes | the default mode with its cost subtracted | `group` |
| The slope's floor and its excess by part | the same parts at k = 1 and 5 | all |
| Whether decode replays captured graphs (Q680, Q765) | the `how` column: replay, capture or eager per card and token | all |
| The k-column and multi-token kernels' share of the slope | on against off | `on`/`off` arms |

Q823's in-model arms (`GGML_CUDA_MMVQ_PASCAL_KCOL_R=1`, `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, queued on the same
rig) measure the Q3_K term of A in the model. Until both land, nothing below the floor in section 1 is a measurement,
and ranks that depend on B and D3 are conditional.

## 3. What a dedicated engine would change, part by part

"Dedicated" means a Flash-Next-only design with geometry fixed at compile time (2,560 wide, 48 layers, 512 experts, 10
used), as Strata compiles the model's shapes into its kernels. The columns say what the sources show, not what each
option would do alone.

| Part | est. ms now | What the outside engines show | What a Flash-Next-only design can add |
|---|---|---|---|
| Attention and recurrent projections | 23.8 | uzu's `INPUT_ROW_TILE` (one weight read for 1–8 rows) is in the fork as the k-column kernels | The products at bandwidth are kernel work in either engine: Q823's launch shapes, a Q3_K one-token kernel (Q224), and Flash-Next's 2,560-wide shapes as compile-time constants (the shape-class templates of Q147). Concatenating `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` (one normed input) into one launch: 108 fewer launches and better SM fill, est. −0.5 to −1 ms (Q868) |
| Hyper-connection mixers and projections | 9.1 weights + ~1.1 of launch | Strata's fused read: 672 launches a token to 192 (`fused_gr`, Q439); the Pascal BF16 kernel already runs at 91 % in the bench | Fusion of the read, the projection and the post-mix removes ~480 launches and the L2 round trips: est. −1 to −2.5 ms. The bytes (1.28 GB) change only with the format (Q604, Q620), not the engine |
| Routed experts | 7.0 | Strata's grouped Q2_0 kernel; uzu fuses down + combine for several tokens; the fork's Q88 persistent kernel fused 13 launches into 1 | Gate/up, SwiGLU, down and weighted sum in one launch with the activation in shared memory, est. −1 to −2 ms; the down projection (82 GB/s) is the target. Skipping dropped experts (Q660) is separate |
| Recurrent state and conv | ~5 (state, conv, gathers) | Strata updates the state in place and fuses conv, L2 norms and the gated norm; uzu keeps no per-token snapshots (replay of the accepted path) | In place and fused already (Q437, Q438). The delta-net kernel at its state floor (~90 → ~40 µs a layer) est. −1.5 to −2 (Q868); one kernel from conv to `ssm_out` with the state in registers est. −0.5 more |
| Attention (12 layers) | 0.7 at empty context; 6–38 at 5–32k | Strata's sparse SIMT attention; uzu's GQA-packed decode | Kernel work in either engine (Q230); no engine advantage |
| Router, top-k, softmax | 3.0 + ~1 | uzu: fusing the router GEMV and top-k into one threadgroup lost on one of two tries | A 16-bit router on Q546's kernel, est. −2.1 (a file change); router + top-k + normalise in one kernel, est. −0.3 to −0.5 (both Q865) |
| Output layer | 3.0 | uzu and HyperQwen both shortlist the head | at 83 % of bandwidth already; a bounded shortlist excludes nothing so far |
| Launch nodes | ≤ 6.7 ceiling | Hazy's and Mirage's megakernels win where a token is ~1 ms and kernels are µs long; uzu's open discussion points to a megakernel | The kernels here average 28 µs, so the saving is bounded by the census, not by the design: ≤ 2.3 ms net as persistent stages, ~2.3 µs a launch as plain fusions |
| Overlap of a product's weights with the small ops before it | part of B and C | none of the engines read publishes it for decode | The next product's first ~1 MB prefetched into the 1.5 MB L2 from a parallel graph branch while the norms and gates run: est. −2 to −4 if the L2 holds it (Q867's prefetch arm; possible inside the fork's graph, with no executor) |
| Hops | 3.6 | Strata's host hand-off is 51.2 KB a row against the fork's 40 KB | x1-bound: Q8_0 staged (1.0 ms) is a fork switch; a mailbox saved nothing |
| Host and waits | 7–10 (unsplit) | HyperQwen: graphs per verify length +14 % on Flint, −7 % when short steps fall back; slim fork: < 1 ms of host generality in all | `-vgraphs` (#31156) is in `-ladder3`. What a standalone engine adds depends on what Q869 finds: inputs staged on the cards and graph reuse are fork changes too |
| Verify and speculation | slope 33.2 | uzu and Strata verify from the committed state and replay the accepted path; ninfer's lead is cheap verification on tensor cores, absent here | The state ring and replay (Q686) fit a fork or an own engine equally; the k-column kernels exist |
| Memory plan | compute buffers 100–670 MiB a card by `-ub` | Strata's fit leaves ~138 MiB at 64k | A static plan frees part of the buffers and could let IQ3_XXS hold 64k at a larger micro-batch (no ms; fit and prompts) |

Three findings from other engines bear on the decision:
- **Strata** wins where VRAM is short and host RAM deep; with a model that fits, llama.cpp ties or wins in its own
  tracker, and on Pascal its sm_61 numbers have been flat since 0.1.44. Its value to the rig is the code around the
  weights ([strata.md](strata.md), ranks 1–5), which the fork has taken.
- **ninfer on gfx906**, a no-matrix-core card, ran at 0.52–0.66× llama.cpp's speed while reading about half the bytes,
  and its 4-token check cost 2.3× a plain token ([port](https://github.com/JCraigWasTaken/ninfer-gfx906);
  [engine_infer.md](engine_infer.md)). Here T(4)/T(1) is 2.1: a dedicated engine on hardware without matrix cores met
  the same verify wall. Its lead elsewhere comes from tensor cores and cheap verification, neither of which the P106
  has. The calibration on fast hardware: ninfer-all's maker reports 136 tok/s for Flash-Next Q2_0 on two RTX 5090s
  (7.4 ms against a ~2.3 ms bandwidth floor, ~3×; speculation status not stated, a claim to test), so a dedicated engine
  at ~3× its floor would be unremarkable.
- **uzu** has no CUDA or Vulkan backend and no MoE on `main`; what it contributes (multi-row matvec, replay) is already
  in the fork.

## 4. Three options, ranked

All figures are est., one token at short context unless marked, against 82.4 ms. Savings of (b) and (c) are beyond the
previous option, so they cannot be added to an option skipped. The launch savings overlap (all options share the
6.7 ms ceiling), which is why the totals are below the sums.

**First, the kernel work every option needs.** The largest slice, A, is the products' speed, and an own engine would
have to rebuild these same kernels before it saved anything:

| Change | Registry | est. ms | First measurement |
|---|---|---:|---|
| Q3_K's one-column path at the bench's best launch shape | Q823, C267 | −4 to −6.1 | the in-model arms on T(1) and T(4) (queued) |
| Router as a 16-bit copy on Q546's BF16 kernel | Q865, C305 | −2.1 | T(1) on the file with `ffn_gate_inp` rewritten; KLD (expected 0) |
| Q3_K one-token kernel at the K-quants' speed | Q224, Q812 | −2.3 beyond Q823 | the kernel's µs at the four Q3_K shapes against `kcol_best.tsv` |
| IQ4_XS and K-quant one-token kernels at Flash-Next's shapes | Q866 | −2 to −3 (if they reach ~120 GB/s) | Q866's bench times them alone |
| The shared expert's small products (Q612's launch shape, Q602's fold into the routed launch) | Q612, Q602 | −0.3 to −0.5 | T(1) with each |
| **Together** | | **−6 to −14** | T(1) est. 68–76 ms, 1–3 weeks, low risk (the two unbuilt kernels may give less) |

**The three options:**

| | (a) More fusion in the fork | (b) A Flash-Next-only executor inside the fork | (c) A standalone engine |
|---|---|---|---|
| What it is | Fusions and merged launches behind switches, one at a time (the worklist below) | For `qwen4exp` decode and verify (≤ 8 tokens): a static program per card built once from the GGUF, 4–8 fused kernels a layer, a fixed buffer plan, stages as device functions of the fork's kernels; prompts keep the ggml path | A new program: loader, kernels, executor, server, caches |
| est. ms saved | **−4 to −8** | **−3 to −7 more** | **−1 to −3 more** |
| Where the ms come from | ~1,500 launches (3.4 ms ceiling) + the delta-net kernel near its floor (−1.5 to −2) + the hyper-connection traffic | ~1,400 launches left (net 1.5–2.5 after barriers); overlap of each product's weights with the small ops before it (part of B and C, est. 1.5–4.5, unproven); state in registers; device-fed tokens | host-side generality (≤ 1 ms), a memory plan, a scheduler built for pipelining; more only if Q869 finds host time that is structural to ggml |
| Weeks (one engineer) | 3–5 | 6–10 | 20–40 to reach the fork's agent-check parity |
| Risk | low: each fusion bit-exact against the unfused graph, behind a switch | medium-high: registers (Q88 needed 128 and two blocks of 8 warps a SM; one block a SM lost), barrier cost, k-shaped stage tables, coexistence with pipelined speculation, checkpoints and the fit planner | high: every serving feature of the agent path must be rebuilt first; the fork's ms are lost until then |
| est. ms per week | 1–1.6 | 0.3–1.2 | 0.03–0.15 |

With the kernels and (a), T(1) comes to est. 60–72 ms; with the Q8_0 hop as well, 58–70; with (b), 51–67. A served
step, without the logits copy, is ~4.7 ms less in each case.

**(a) The worklist**, in the order the per-op profile should set:

| Fusion | Registry | est. ms |
|---|---|---:|
| Hyper-connection read in two kernels (Strata `fused_gr`) | Q439 | −1 to −2.5 |
| Delta-net kernel to its state floor (3.3 → ~1.5 ms) | Q868, C308 | −1.5 to −2 |
| Single-op groups and activation quantizations folded into neighbours' prologues | Q683, Q100 | −0.5 to −1.5 |
| `attn_qkv` + `attn_gate` + `ssm_alpha` + `ssm_beta` as one product | Q868, C308 | −0.5 to −1 |
| Router + top-k + normalise in one kernel | Q865, C305 | −0.3 to −0.5 |
| Injects folded into the down projection | Q557 | −0.35 |

Needs: Q822's per-op table for the order; a `test-backend-ops` case per fusion; the bit-exact comparison the fork
already runs. Proof first: each fusion's ms in T(1) and T(4) on a plain A/B of its switch; stop a fusion that saves less
than its launches × 2.3 µs.

**(b) The executor.** Needs a layer descriptor from the GGUF (shape-class templates per Q147), stage tables for k = 1–8
(the k-column kernels already take k at run time), counters rather than grid barriers between dependent stages (a
barrier costs about a node), bit-exact checks of every stage against the ggml graph (Q88's method), a switch with the
ggml path as the fallback, and the fit planner's knowledge of the executor's buffers. Proof first, in this order:
1. **Reproduce B in a bench (Q866, C306).** Chain the five heaviest real weight products with the small operations
   between them in a CUDA graph on one card, against the same products alone (`bench/mmvq_pascal`,
   `bench/megakernel_p106`); then bisect with the clock held, L2 flushed between calls, and eager against graph. If no
   excess appears, (b)'s overlap term is zero and (b) shrinks to the launch savings (≤ 2.5 ms beyond (a)): stop.
2. **A real layer (Q867, C307).** One recurrent layer with Flash-Next's actual products (synthetic weights, real shapes
   and formats) and its weightless chain as the fork's kernels in a graph, against 4–6 fused kernels, with an arm that
   prefetches the next product's first megabyte with `prefetch.global.L2` from a parallel graph branch (the plain form
   exists from sm_50, [CCCL's PTX wrapper](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/prefetch.html);
   the eviction-priority forms need sm_80; the L2 is 1.5 MB, so only the head of each product can be staged). Gate: ≥
   15 % less time a layer (~0.2 ms × 48 = est. 9 ms a token); under 8 % stops it. If the prefetch arm alone pays, it
   goes into the fork's graph without an executor.

**(c) The standalone engine** needs everything of (b) plus a GGUF loader and the dequantizers of every type in the file
for prompts, the tokenizer, chat template, tool grammar and reasoning budget, an OpenAI-compatible server with prompt
caching and the 113 MiB state checkpoints the agent check depends on, the n-gram table reads, the drafters (DFlash2,
MTP), sampling and the fit planner. Strata's own fork estimate for phases A and B alone is 12–27 engineer-weeks and
parity with this project's features 31–65 ([strata.md](strata.md)). What it adds over (b) is the host's serial work
(≤ 1 ms after the input layer moves to the card, [slim fork](pascal_slim_fork.md)), per-token synchronizations and a
memory plan. The agent check drives an HTTP endpoint, so serving compatibility is not optional. **Reopen only if** (b)'s
benches pass and Q869 shows ≥ 5 ms of host time a pass that the fork's scheduler cannot remove, or if the owner makes
Flash-Next the sole target for good. Strata's fork conditions (under ~5 tok/s with kernels as the proven cause) do not
hold at 12.1 tok/s.

**What the options leave alone:** the file's bytes (the floor), the x1 hops (Q8_0 staged is a fork switch), the prompt
path (19–21 % of an agent task's time on Flint), and the idle capacity of eight cards a token (speculation).

## Ideas for the rig

Ranked by the middle of the est. ms saved per one-token pass at short context, except sparse attention, ranked at 8k
(its saving grows with depth); lossy ideas and per-pass ones are marked. The executor ranks below the fusion worklist
and the excess hunt because it depends on both. Registry ids are rows of
[quants/experiments.md](../../../quants/experiments.md).

| # | Idea | est. ms | First measurement | Registry |
|---:|---|---|---|---|
| 1 | **Sparse attention at agent depth**: the Pascal gather kernel for the 12 QSA layers, reading 2,051 cells instead of the cache (depth) | −3 at 5k, −6 at 8k, −15 at 16k, −34 at 32k | `bench/fattn_sparse/check.sh` on one card (written, never run on a GPU), then `llama-bench -d 8192,32768` with `GGML_CUDA_FATTN_SPARSE_PASCAL=0` against `1` on `-ladder3` | Q230, Q441 |
| 2 | **Fusion worklist (a)**: the hyper-connection read, the delta-net kernel at its floor and the merged recurrent products, single-op groups, router + top-k, the injects | −4 to −8 | Q822's per-op table ranks them; each fusion's switch A/B on T(1) and T(4) | Q439, Q868, Q683, Q865, Q557 |
| 3 | **Q3_K's one-column path at the best launch shape** (1 row per lane group or a 32 KB tile) | −4 to −6.1 | T(1) and T(4) with `GGML_CUDA_MMVQ_PASCAL_KCOL_R=1` and with `_SMEM=32768` (queued on rig 1) | Q823, C267 |
| 4 | **Fused-layer executor (b)**: stages 1–3 of the megakernel plan on real layers, after the benches of 5 and 7 | −3 to −7 beyond 2 | Q867's real-layer bench on one card, ~1 day | Q867, Q670 |
| 5 | **Find the in-model excess (B)**: a chained-kernel bench of real products, then bisect clock, cache state and graph scheduling; the same bench times the dense one-token kernels alone at Flash-Next's shapes | recoverable 2–6 (of 3–12) | Q822's `sample=8` table against the bench µs at the same shapes (no new run), then the chained bench | Q866, C306 |
| 6 | **Split the host and waits** (7–10 ms by difference): host timers per decode phase (graph build or reuse, input uploads, the enqueue of each card, the end-of-pass sync, the output read) | recoverable 2–6; decides (c) | `llama-bench -p 1 -n 0` and a served token with the timers on, against the `graph` profile's card time | Q869, C309 |
| 7 | **L2 prefetch of the next product's weights** on a parallel graph branch, inside the fork's graph, with no executor | −2 to −4 (speculative; part of 5) | Q867's bench with only the prefetch branch added | Q867, C307 |
| 8 | **Residual hops in Q8_0, staged in 2 chunks** (4 above ~16 KB) | −2.6 (3.6 → 1.0) | T(1) and T(4) with `LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1`, then KLD on the quality chain's bases and one agent-check arm | Q750, Q231, Q751, Q820 |
| 9 | **The IQ4_XS and K-quant one-token kernels at Flash-Next's shapes**, tuned if Q866's bench shows them below ~120 GB/s alone | −2 to −3 | Q866's bench | Q866 |
| 10 | **Keep the cards at the boost clock**: a light keep-alive load per card, or pipelined guesses that keep them busy | 0–5 (overlaps 5) | a 100 ms clock log per card during `llama-bench -p 1 -n 0 -r 20` and during an agent turn, with and without a keep-alive kernel on the idle cards; power measured | Q154 |
| 11 | **A Q3_K one-token kernel at the K-quants' speed** | −2.3 beyond 3 | its µs at the four Q3_K shapes against `kcol_best.tsv` | Q224, Q812 |
| 12 | **Router as a 16-bit copy on Q546's BF16 kernel** (exact; the indexer's shape) | −2.1 | T(1) on the file with only `ffn_gate_inp` rewritten; KLD against the original | Q865, C305 |
| 13 | **Hyper-connection projections below BF16**: needs a short-row Q8_0 kernel at ≥ 140 GB/s; the best k-column variant reads those rows at 48–80 GB/s, slower than the BF16 kernel (the hcq8 retype takes 98.8 ms against 82.4 at k = 1). Lossy | 0 to −4 | a short-row Q8_0 kernel timed on one card against the BF16 kernel's 157–159 GB/s | Q604, Q620, Q812 |
| 14 | **Skip the low-weight experts of confident tokens**. Lossy | −0.5 to −1 at p = 0.9 | `LLAMA_MOE_TAIL` hist on the agent check; text differs in 10 of 12 requests at p = 0.9 today | Q618, Q660 |
| 15 | **Route once per verify block**: tokens 2..k pick only among the first token's experts. Lossy, per pass | −15 to −26 per 5-token pass (est.: 38 fewer expert reads a layer on llama-bench's random ids, 2.5 GB at 95–173 GB/s; less on real text; none at k = 1) | KLD by k = 2–5 on the quality chain's bases, then T(5) | Q860, C300 |
| 16 | **Close the slope of the k-curve** (33.2 ms a token against a floor of 8–11): which part of the extra token's cost is experts, hops, k-column arithmetic or the recurrent update | up to −20 a verified token | Q822's k = 5 against k = 1 parts | Q822, Q591 |
| 17 | **A standalone Flash-Next engine**: parked | −1 to −3 beyond 4 | none until 4's benches pass and Q869 shows ≥ 5 ms of host time the fork cannot remove | Q466 (Strata fork, parked) |

## Sources

Repository (measurements of 2026-10-24 to 2026-11-04):
[k-curve and k-column bench](../../data/2026-11-04_kcurve_p106/README.md) ·
[file choice and bytes a token](../../data/2026-11-04_file_choice/README.md) ·
[kernel timing, Q2_0 and BF16](../../data/2026-11-03_kernel_timing/README.md) ·
[megakernel census](../../data/2026-11-04_megakernel_census/README.md) ·
[megakernel bench and the hop](../../data/2026-11-04_megakernel_rig/README.md) ·
[light profiler](../../data/2026-11-04_light_profiler/README.md) ·
[verify cost](../../data/2026-11-03_verify_cost/README.md) ·
[Flash-Next baseline](../../data/2026-11-03_flash_next_baseline/README.md) ·
[ladder3 merge](../../data/2026-11-04_ladder3_merge/README.md) ·
[recurrent-layer fusions](../../data/2026-11-04_gdn_fusions/README.md) ·
[sparse attention](../../data/2026-11-04_qsa_sparse_attn/README.md) ·
[card hops](../../data/2026-11-04_card_hops/README.md) ·
[bytes switches](../../data/2026-11-04_bytes_switches/README.md) ·
[Q99](../../data/2026-10-28_q99_tlb/README.md).
Notes: [Strata](strata.md), [uzu](uzu_lessons.md), [slim Pascal fork](pascal_slim_fork.md), [megakernel](megakernel.md),
[ninfer](engine_infer.md), [HyperQwen](hyperqwen_lessons.md), [MoE speculation](moe_speculation_union.md),
[card hops](card_hops_x1.md), [host decode loop](host_decode_loop.md), [speculation plumbing](speculation_plumbing.md),
[Flash-Next variants](flash_next_variants.md), [idle cards](idle_cards.md).
Outside: [ninfer-gfx906](https://github.com/JCraigWasTaken/ninfer-gfx906) (a no-matrix-core port, 0.52–0.66× llama.cpp);
[Strata](https://github.com/Niko1221/Strata); [CCCL PTX `prefetch`](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/prefetch.html).
