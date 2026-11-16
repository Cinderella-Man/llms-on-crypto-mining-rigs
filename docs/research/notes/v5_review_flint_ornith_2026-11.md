# Flint and Ornith-1.5-122B after v4.0: decode, hops, drafts and files (v5 review, 2026-11-16)

One of five reviews for the v5 order: the two models that write one token a step, their hops between cards, a
draft for each, and the model files. The method is [the third Flash-Next review](flash_next_review3_2026-11.md)'s.

- "est." marks my arithmetic or inference, with its basis beside it. Every other number is measured, from the
  linked folder or [registry](../../../quants/experiments.md) row, one run or load a value.
- The files are Flint Pascal Q8 GGUF (`p065`), Flint Pascal Q5 GGUF and Ornith-1.5-122B Pascal N2P GGUF (`valid1`).
  Deep-lane steps are the second request's. Rig hours are est. Nothing ran for this note.

## The short answer

- **Neither model has a per-part profile of its release file on v3.0 or v4.0.** Prices below rest on kernel
  timings and deep-lane steps. The first v5 row is the map of a one-token step.
- **Ornith's largest lever is attention at depth.** Its step grows from 53.45 ms at the start to 63.89 at 8,192
  tokens and 115.22 at 48,640. A kernel that reads the cache at card speed is worth est. 8.1 ms at 8,192.
- **The cheapest sure gain is in Ornith's expert kernel.** Fused, the gate and up products take 278.5 µs where the
  two alone take 2 × 105.2: est. 3.2 ms a token.
- **The 8-bit hop is small in decode for both** (est. 0.7 ms on Ornith, est. 0.5 on Flint's full pass). It needs one
  call in each graph and a quality check first.
- **A draft for Ornith is not ready for another agent run.** Its step must fall from 165.1 ms to under est. 152.1,
  and its prompt path still loses.

## Where the time goes now

**Flint Pascal Q8 GGUF, v4.0's preset** (`--pipe-depth 4`):

| Part | ms a token | Source |
|---|---|---|
| A coding agent's token | est. 20.28 and 21.28 (49.32 and 46.99 tok/s) | [R159](../../data/2026-11-14_r159_flint_release_agent/README.md) |
| At the start, and at 49,152 tokens | 18.09 and 18.09; 41.64 and 41.63 | [R168](../../data/2026-11-16_r168_flint_f4t_deep/README.md) |
| What staged hops save at the start | est. 0.42 (17.54 and 17.57 against 18.02 and 17.92) | [R165](../../data/2026-11-16_r165_flint_stack_split/README.md) |
| The eight hops' bytes, 8,192 B each | est. 0.86 (the bench's fit: 26 µs and 10 µs a KB) | [C140](../../data/2026-11-04_megakernel_rig/README.md) |

On v2.0, 15,658 of 46,460 generated tokens came from right guesses ([Flint's plan](flint_v3_plan.md)). Most of
Flint's time is full passes after a miss. Its last split into layers, head and hand-offs is from v2.0's traces
([idle cards](idle_cards.md)).

**Ornith-1.5-122B Pascal N2P GGUF, v4.0's preset:**

| Part | ms a token | Source |
|---|---|---|
| A coding agent's token | est. 60.53 and 62.00 (16.52 and 16.13 tok/s) | [R166](../../data/2026-11-16_r166_ornith_stack_no_taper/README.md) |
| At the start | 53.50 and 53.45 | [R167a](../../data/2026-11-16_r167_keepalive_depth/README.md) |
| At 4,096, 8,192 and 48,640 tokens | 58.71 and 58.68; 63.98 and 63.89; 115.36 and 115.22 | R167a |
| Routed experts | est. 18.3: gate and up 12.83, down 5.47 (kernel times at the file's layers) | [Q188](../../data/2026-10-28_lowbit_rig/README.md) |
| The eight hops' bytes, 12,288 B each | est. 1.17 | C140's fit |
| Dense products, recurrent layers, head, host | est. 34 together, no split | none |

Without the keep-alive Ornith decodes 3.4 % slower (R167a). Its newest per-part profile is of Unsloth's UD-Q2_K_XL
on an October build (`baf2e8ded`): experts 41.94 and dense products 36.25 of 98.81 ms of profiled GPU time, and
at 39,448 tokens attention 45.95 of 142.87
([Q210](../../data/2026-10-27_iq_kernels/README.md)).

## What v4 proved and what lost here

- **Proved, prompts:** Ornith reads at 103.0 and 105.9 tok/s against 61.1 and 58.4, Flint at 219.7 and 204.1 against
  179.2 and 181.3 (R166, R159).
- **Level, decode:** over six depths Flint writes 40.58 and 40.57 tok/s against 40.24 and 40.00 (R168).
- **Lost on Flint, the keep-alive:** prompts −16.5 % and decode −3.0 % at depth (R165).
- **Split, staged hops:** Flint decodes 40.99 and 41.03 tok/s against 40.42 and 40.40, but reads the 8,192-token
  chunk at 446.1 and 437.4 against 478.2 and 479.2 (R165). Ornith's stack with them decodes 12.94 and 12.93 against
  12.92 and 12.92 ([R162](../../data/2026-11-14_r162_ornith_release_deep/README.md)).
- **Lost on v3.0, Ornith's MTP draft:** 165.1 ms a step for 2.14 tokens against 61.1 ms a token, prompts at 39.4
  against 60.1 tok/s ([data](../../data/2026-11-06_ornith_mtp_rig2/README.md)).
- **Lost, the better Flash-Next file:** row 9.

## Candidates for v5

| # | Row | Model | What | Evidence | Worth | Cost | Needs | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | NEW | both | The one-token step's map on v4.0; for Ornith also passes of 2 to 4 tokens | no profile of either release file exists | prices rows 2 to 10 | none; 1 h each | — | first |
| 2 | Q350 | Ornith | The fused gate and up product, one matrix staged after the other | measured: 278.5 µs fused against 2 × 105.2 (N2P), 256.0 against 2 × 95.5 (N3P) (Q188) | est. 3.2 ms, 6.0 % (24 layers of each type) | small; 2 h | — | first |
| 3 | Q702, Q416 | Ornith | Decode attention that reads the cache at card speed | measured growth: 10.44 ms at 8,192, 61.77 at 48,640 (R167a) | est. 8.1 ms at 8,192 (12.7 %), 48.2 at 48,640 (41.8 %), at Q210's est. 78 % off | large; 2 h | row 1 | first |
| 4 | Q1125 | both | The Q8_0 hop cast in both graphs | Flash-Next: 14.98 to 17.06 tok/s ([S35](../../data/2026-11-04_s35_stack/README.md)); here est. by bytes | Ornith est. 0.7 ms (1.3 %); Flint est. up to 0.5 (2.4 %); prompts unpriced | small; 3 h | the quality check | next |
| 5 | NEW | Flint | Staged hops only below a size cap | measured: decode +1.5 % over six depths (R165) | est. 0.42 ms at the start (2.3 %) | small; 1.5 h | — | next |
| 6 | Q634, Q633 | Ornith | Flash-Next's fused conv window, norm and gate inputs in Ornith's graph | est. only: the rows' own, for Flash-Next's 36 recurrent layers; Ornith has 36 | est. 2.5 to 4 ms (4.7 to 7.5 %) | small; 1.5 h | row 1 | next |
| 7 | Q675 | Flint | The decode attention kernel without local memory | code done, never on a GPU; Flint Pascal Q5 GGUF writes 50.79 tok/s at the start, 34.60 at 32,768 ([data](../../data/2026-11-06_flint_q5_depth/README.md)) | ceiling est. 9.2 ms at 32,768 | none; 1 h | a GPU check | next |
| 8 | Q202 | Flint | The answer store (`--pipe-store`) on v4 | measured on v2.0: 22.55 against 22.87 ms a token | est. 0.3 ms (1.4 %) | none; 1.5 h | — | next |
| 9 | Q1409 | Flash-Next's file | Where `gmix-q2`'s 18.06 ms a step go, per switch | measured: 125.83 and 125.86 ms against 107.80 and 107.77 ([R115](../../data/2026-11-12_r115_gmix_step/README.md)); est. 14.3 gate and up, 3.8 dense ([Q1451](../../data/2026-11-13_c827_gmix_iq_readonce/README.md)) | decides the file | none; 1 h | — | next |
| 10 | Q1281 | Ornith | The draft's catch-up cut to its caches, then its step with probabilistic verification | Flash-Next's prompts: 101.4 and 101.8 against 92.2 and 92.6 tok/s ([Q1278](../../data/2026-11-11_r76_c620_p106/README.md)) | break-even under est. 152.1 ms a step | small; 2 h | row 1 | later |

**The hop (row 4).** `LLAMA_HOP_CAST` acts through `build_hop_cast`, and only Flash-Next's graph builder calls it
(Q750). It takes one call at the top of Flint's and Ornith's layer loops. A hop then carries est. 2,176 B instead
of 8,192 on Flint and est. 3,264 instead of 12,288 on Ornith. Flash-Next gained three times its bytes' worth,
since staging also removed host work; Ornith already runs staged hops with no decode gain, so I price bytes
only. The check first: the cast against F32 hops on a P106 by
[FN93c](../../data/2026-11-16_fn93c_ornith_taper_kld/README.md)'s method, a repeat as the floor. Ornith's arms also
go against the makers' Q8_0: another rounding alone moved it by a KLD of 0.047. Before that, each model's residual
range at its eight boundaries.

**Ornith's draft (row 10).** v4 adds probabilistic verification: on Flash-Next 3.283 and 3.304 tokens a step against
3.043 and 3.072 ([Q1173](../../data/2026-11-10_prob_verify_acceptance/README.md)). Offline Ornith's own layer is
accepted 0.843 and 0.740 on drafts one and two ([Q397](../../data/2026-10-28_ornith_mtp_accept/README.md)): est.
2.49 tokens a step against 2.14 served. Kernels that read N2P and N3P once for 2 to 4 tokens exist
(`GGML_CUDA_MMVQ_PASCAL_MT`, off by default) and save little: at 3 tokens the fused gate and up cost 2.60 times one
token and the down product 3.67 ([Q211](../../data/2026-10-28_multitoken_kernels/README.md)), since each token
picks its own experts. The checkpoints on the cards free host memory, not the card memory the draft needs.
Break-even is est. 152.1 ms a step (2.49 × 61.1); v3.0's step took 165.1.

**Flint's draft.** Pipelined guesses are its draft. Left are row 8 and a lighter last card (Q277: est. 0.4 ms for
a cycle a fifth shorter). The trained head lost as served, 23.59 against 22.85 ms a token (Q193); a better head
is training, the sister project's. Chapter 9's "+8 to 15 %" for hand-offs is stale: row 5 is the measured part.

**Files.** A better file ships when its step cost is measured per switch on the line it runs (row 9), the solver
takes that cost as its budget, and the agent check at equal work is not slower beyond the spread. Otherwise it is
a second, slower preset (Q1196). Open for Flint: Q431, Q273, Q97. Open for Ornith: Q249, Q250, Q301, Q351. Cutting
or merging experts, and any training, belong to the sister project.

## Rows to close, merge or re-rank

- **Q756 (C203):** duplicates Q1125 (C640); merge.
- **Q1029 (C557):** superseded by R165 and R162; the only Flint or Ornith row in "Mining rig 2".
- **Q751:** done; its class is per model: negligible on Flash-Next, a decode gain on Flint, level on Ornith.
- **Q231, Q963:** stay closed; F16 reopens only if Q8_0 fails a model's check. **Q81:** close; C140 measured the hop.
- **Q270:** superseded by Q682; its est. +8 to 15 % rests on a v2.0 fit. Q682's rig run goes below row 5.
- **Q162:** park until Q593: on Flash-Next pipelined guesses wrote 4.91 against 7.88 tok/s, a wrong guess costing
  372 ms ([data](../../data/2026-11-03_flash_next_pipelined_rig/README.md)).
- **Q341, Q1129, Q1130, Q1131, Q1132:** stay parked with row 10; priced on v3.0.
- **Q351:** a stale price (October's prompt path); time it on v4's.
- **Q219:** done; its ms prices are v3.0's one-token kernels'. **Q1392** goes below Q1409: the cost blocked the
  file, not the KLD prediction. **Q1433** waits for Q1409.
- **Q168:** close; shipped as Flint Pascal Q8 GGUF.
- **Q260:** done, negligible on v3.0 (2.59 against 2.60 minutes a task); untimed on v4.
- **Q267, Q268, C668:** park; they shorten or steer the reasoning.
- **Q133:** a stale price (a v1 token); the hardware review's row.
- **Q1126 (C641), Q1134 (C648):** close; v3.0's drift is within noise against the Q8_0 reference, and v4 reads
  every deep chunk faster than v3.0 (R168).
- **C11, C12:** superseded by the taper and the checkpoints on the cards.
- **C16 (Q195):** park; it serves a head that lost.
- **C89, C166 (Q623, Q716):** wait for row 1's passes of 2 to 4 tokens. **C19, C20:** in the build, on by default
  (`GGML_CUDA_GDN_FUSE`); row 6 brings Ornith's graph the rest.
- **C10, Q1133 (C647), C136, C351, C75, C131:** keep, unpriced on v4.
- **"Negligible gains":** `-t 1 -tb 1` and eight scheduler copies were timed on v2.0; time them as one group on
  v4 with rows 5 and 8.

## New ideas

- **Rows 1 and 5** above.
- **Q162 split in two:** its card-built mask for Ornith's kind of model is built and exact. Time it alone at depth.
- **Probabilistic verification for Ornith's own draft:** Q731 covers Flash-Next's drafts only.
- **A dense-only pick for Flash-Next:** `denseiq31` alone holds 0.0193 of `gmix-q2`'s 0.0432 gain in agent KLD
  ([data](../../data/2026-11-04_fmt_mixes/README.md)), and the dense changes are est. 3.8 of the 18.06 ms.

## What I could not settle from the files

- The split of a step on either release file.
- Whether Ornith's growth with depth is the attention kernel or the mask built on the host.
- Which kernels Ornith's 3-token check takes on v4.0. v3.0's passes of 1 to 4 tokens took 104.6, 183.9, 208.7 and
  293.3 ms: no smooth curve.
- Whether [the host ladder](../../data/2026-11-12_r122_host_ladder/README.md)'s switches act on Ornith's MTP layer.
- Why the keep-alive costs Flint its deep prompts, and why staged hops lose one chunk.
- The residual's range in Flint and Ornith, and so the cast's quality.
- Flint's residual width: I took the hops note's 8 KB. Ornith's 12,288 B follows from its hidden size of 3,072.
