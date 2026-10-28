# Qwen3.8-Flash-Next on the P106 rig: which version to run and how to make it fast (2026-10-28)

Qwen3.8-Flash-Next (Qwen, 2026-08-24) is a 125B mixture-of-experts model with about 6B active parameters per token,
plus a 51B-parameter [n-gram table](../../../GLOSSARY.md). Hundreds of versions of it are published: quants, pruned
models, fine-tunes, merges. This note answers four questions for the rig:
- which of those versions people rate best for agentic coding;
- which fit on nine 6 GB cards with 4 GB of host RAM;
- how a version can be compared with the others;
- which one has the best odds of becoming [Pascal-friendly](../../../GLOSSARY.md) and fast in Miner Llama.

Sources were read on 2026-10-27/28: the Hugging Face Hub API and model cards, community reports, upstream llama.cpp
and the Miner Llama v3 source. A header census of ten local copies of the files gives every size and format in
section 1. Nothing was run on the rig; numbers marked **est.** are arithmetic, not measurements. The census tables and
scripts are in [docs/data/2026-10-27_flash_next_variants/](../../data/2026-10-27_flash_next_variants/README.md).

## Summary

1. **The rig can hold the model's transformer, not its n-gram table, and llama.cpp is built for that.** The table
   (28.8 GB in IQ4_NL) stays memory-mapped on disk; each token reads 16 of its rows (about 90 bytes each), and the CPU
   gathers them. The cards hold the rest: 30–47 GB in the files that fit.
2. **No file that people rate near the full model for agents fits.** The community favourite is ISTA-DASLab's
   GSQ-RCO IQ3_S (SWE-bench Verified 82.0 against 82.8 for the base model, by its maintainer), whose transformer is
   54.8 GB. The one independent agentic test found the Q2 class degrading and IQ3_S on par with the original.
3. **The best-evidenced files that fit and run in Miner Llama are from the same GSQ-RCO family:** IQ3_XXS (47.0 GB,
   the best quality, at the very edge of the cards), IQ2_XS (39.2 GB), Q2_0 (37.6 GB) and the 256-expert Coder
   (29.6 GB). ISTA-DASLab and ukisai (Swift 1.5, a fine-tune that thinks less) each publish the first three. Other
   files that fit (IQ1 quants, prunes, Tinfield-1 Mini) have less evidence behind them.
4. **Dense tensors, not experts, decide decode speed.** Each token reads 3.4–4.8 GB of dense tensors and only
   0.5–1.0 GB of routed experts. The expert format decides what fits; the dense formats decide most of the speed. The
   cheapest gains are rewriting the BF16 hyper-connection tensors as Q8_0 and, in the Swift files, the F32 router as
   F16.
5. **Q2_0 is the most Pascal-friendly expert format in these files.** It is a simple 2.25-bit format, est. 3–6 ps per
   weight on a P106, against 20.5 measured for IQ3_XXS. A Q2_0 file also leaves room for 262k of context and the MTP
   head.
6. **Recommendation:** make the GSQ-RCO **Q2_0** file the first Pascal-friendly target, with the **IQ3_XXS** file as
   the quality reference that the v3 IQ kernels speed up. KLD on the quality machine, then the agent check on the rig,
   picks between them and between the ISTA and Swift builds. Est. decode for Q2_0 at short context is 12–15 tok/s
   before MTP and the dense fixes.
7. **The table needs an SSD, not a flash stick.** With only 0.3–0.7 GB of page cache, 11–14 of a token's 16 lookups
   go to disk: ~0.3 ms per token on a SATA SSD, 4–8 ms on a USB flash stick, which also adds 50–95 % to prefill (est.).
8. **Miner Llama is the only engine for this rig.** Every dedicated Flash-Next engine needs 32–80 GB of host RAM, a
   newer GPU, or both.

## 1. What the model is and what the rig must hold

From the GGUF headers (`qwen4exp`):

| Part | Value |
|---|---|
| Layers | 48: 36 Gated DeltaNet, 12 attention with a sparse-attention indexer (4 heads × 128, top 2,048 tokens) |
| Width | 2,560; 24 query and 2 KV heads of 256 |
| Experts | 512 routed (top 10) + 1 shared, 640 wide |
| Hyper-connections | 4 residual streams, low rank 320 |
| N-gram table | `per_layer_token_embd`, 160 × 320,001,536: 16 hashed heads of 2- and 3-token sequences, feeding layer 1 |
| MTP | one layer, shipped separately as GGUF drafts (1.9–5.2 GB) |
| Context | 262,144 |

What the cards must hold, against 9 × 6,068 MiB (53.3 GiB):
- **The transformer:** everything but the n-gram table (next table).
- **KV cache plus indexer cache:** 30 KiB per token in F16, ~16 KiB in q8_0: 3.75 / 2.0 GiB at 128k, 7.5 / 4.0 GiB at
  262k. The 36 DeltaNet layers keep a fixed state instead.
- **Compute buffers and the desktop:** about 0.5 GiB per card. Ornith-1.5-122B's 47.6 GB file reaches 122,880 tokens
  of F16 cache only with a 128-token micro-batch
  ([Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md#fitting-it-on-the-cards)).
- **The MTP draft, if used:** about 2.7 GB on one card.

The local files, from the [census](../../data/2026-10-27_flash_next_variants/census/summary.tsv) (GB):

| File | Experts | On the cards | N-gram table | Routed experts | Dense | Read per token |
|---|---|---|---|---|---|---|
| ISTA GSQ-RCO-Coder ("IQ1_M") | 256 | 29.6 | 28.8 | 25.2 | 4.1 | 5.10 |
| agentionai Gyro-S (trellis) | 512 | 29.7 | 28.8 | 25.8 | 3.4 | 3.86 |
| agentionai Gyro-M (trellis) | 512 | 37.6 | 54.4 | 32.1 | 4.8 | 5.42 |
| ukisai Swift-1.5 GSQ-RCO Q2_0 | 512 | 37.7 | 28.8 | 34.0 | 3.5 | 4.16 |
| ukisai Swift-1.5 GSQ-RCO IQ2_XS | 512 | 39.3 | 28.8 | 35.5 | 3.6 | 4.24 |
| ukisai Swift-1.5 GSQ-RCO IQ3_XXS | 512 | 47.2 | 28.8 | 42.9 | 4.0 | 4.81 |
| ISTA GSQ-RCO IQ3_S | 512 | 54.8 | 28.8 | 50.3 | 4.2 | 5.16 |
| ukisai Swift-1.5 GSQ-RCO IQ3_S | 512 | 54.9 | 28.8 | 50.3 | 4.3 | 5.28 |
| agentionai Gyro-L (trellis) | 512 | 53.9 | 54.4 | 48.4 | 4.8 | 5.74 |
| TakeOnMe Q4_K (merge) | 512 | 82.5 | 54.4 | 77.0 | 4.8 | 6.31 |

"Read per token" counts every dense tensor once and the routed experts at 10/512 (10/256 for the Coder). ISTA's own
Q2_0, IQ2_XS and IQ3_XXS builds are 37.6, 39.2 and 47.0 GB on the cards
([card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)). In Swift Q2_0, 1.35 GB of the dense
bytes are in BF16, 1.26 GB of them the hyper-connection projections, and 0.25 GB is the router in F32. ISTA's builds
keep the router in BF16 (0.13 GB).

## 2. The published versions

The Hub search for "Flash-Next" hits its 500-result cap; 389 of the results are GGUF repos. Qwen publishes only BF16
(360 GB) and FP8: every GGUF, imatrix and MTP draft comes from someone else. Fewer than ten repos report
agentic-coding scores with a named harness. Abliterated and "uncensored" versions are left out of this note.

| Group | Versions | For this rig |
|---|---|---|
| Standard quants of the base | [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ1_S … BF16, [bartowski](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) (28 sizes, imatrix and its calibration text), [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF), mradermacher, [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF), [AesSedai](https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF), agentionai AP, mudler APEX | Only unsloth UD-IQ1_S/IQ1_M (≈44–46 GB on the cards) and APEX Nano (43.9) fit; IQ1 files are worse per GB than GSQ-RCO IQ2_XS |
| GSQ-RCO (Gumbel-Softmax learned rounding with a per-tensor format search, standard GGUF formats; method in [sub_1_bit_quants.md](sub_1_bit_quants.md), section 2) | [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) and [-Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) (512 → 256 experts); [Swift-1.5 GSQ-RCO](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | The candidates: Q2_0, IQ2_XS, IQ3_XXS and the Coder fit and load in Miner Llama |
| Custom formats in their own forks | [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) and Gyro-Coder (QTIP-family trellis), [Mach-2](https://huggingface.co/SyzygyResearch/Mach-2-Additive-Medium) (1.70 bpw, CUDA-only fork), [Victoria](https://huggingface.co/rmonsurate/Victoria) (288 experts, 4-bit retrained), Niwaki, UltraLite | Do not load in Miner Llama; no BF16 to rebuild from |
| Fine-tunes and merges | [Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next) (thinks less), [Tinfield-1](https://huggingface.co/badtheorylabs/Tinfield-1) (agentic SWE), ThinkingCap, [Groot-65B-A6B](https://huggingface.co/GrootTeam/Groot-65B-A6B-261023) (256 experts, post-trained), Signal, TakeOnMe (60/40 merge of Swift 1.5 and Tinfield-1) | Swift 1.5 has rig-sized GSQ-RCO files; Tinfield-1 Mini (37.3 GB) fits but is unbenchmarked |
| Expert prunes | REAP-288/320/256, REAM-60Pct, RAZOR, Swift REAP K224–K320, Coder-192E | 23–40 GB on the cards; no agentic evaluation; pruning loss adds to quantization loss |
| MTP drafts | unsloth `MTP/`, bartowski and ggml-org `mtp-*`, a DFlash2 drafter (ggml-org, incoai), heads cut to 256 experts for the Coder | Load in llama.cpp with `--spec-type draft-mtp` |
| N-gram table sidecars | [Delgon](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF) (BF16, Q8_0, IQ4_NL), mduca771 | Need a `--ple`-style loader patch |

BF16 sources exist for the base model, Swift 1.5, Tinfield-1, ThinkingCap, Groot-65B and several prunes, but not for
the GSQ-RCO files, Gyro, Mach-2 or Victoria. Importance matrices are published by unsloth, bartowski, mradermacher,
AtomicChat (with its corpus), AesSedai and ukisai; ISTA's is not published, though its per-tensor allocations are.

Licences: the base is `qwen-community-1.0`. Files built on Swift 1.5 add the Swift Open License 1.0, free below
US$1M of yearly revenue. ISTA tags its repos Apache-2.0, but its card says the weights inherit the base licence.

## 3. What people rate best for agentic coding

### The evidence

- **ISTA GSQ-RCO IQ3_S is the community's pick.** In a thread about it, the maintainer reports SWE-bench Verified
  82.0 against 82.8 for the base model, harness not stated
  ([discussion #26](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/26)). It has 7.3M
  downloads, more than any other Flash-Next repo, including Qwen's own. Hacker News and Hugging Face users agree.
- **The only independent agentic test** ([Kaitchup #163](https://kaitchup.substack.com/p/agentic-coding-with-qwen38-flash),
  2026-10-10) tested four GGUFs. UD-Q4_K_XL and GSQ-RCO IQ3_S were on par with the original, and UD-Q2_K_XL degraded
  significantly. On non-agentic tasks every file held up, down to the IQ1 class. Its per-file numbers are paywalled.
- **ISTA's own task scores** (xhigh effort; [card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)):

  | File | On the cards | AIME25 | GPQA-D | LiveCodeBench v6 | Mean of the three |
  |---|---|---|---|---|---|
  | BF16 | — | 100.00 | 91.92 | 87.43 | 93.12 |
  | IQ3_S | 54.8 GB | 100.00 | 92.93 | 86.86 | 93.26 |
  | IQ3_XXS | 47.0 GB | 100.00 | 91.41 | 86.29 | 92.57 |
  | IQ2_XS | 39.2 GB | 96.67 | 87.37 | 83.43 | 89.16 |
  | Q2_0 | 37.6 GB | 96.67 | 89.39 | 81.14 | 89.07 |

- **KLD against Swift's BF16** (development set, 512 tokens of context;
  [card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)): IQ3_S 0.145, IQ3_XXS 0.240,
  IQ2_XS 0.341, Q2_0 0.424. On code: 0.069, 0.119, 0.174, 0.235. Swift's IQ2_XS beats ISTA's on 7 of 8 sets; Swift's
  Q2_0 is worse than ISTA's on 6 of 8, and the card itself prefers IQ2_XS at that size.
- **IQ3_XXS sits level with UD-Q2_K_XL** on the one KLD set that has both, agentionai's: 0.311 against 0.334
  ([Gyro card](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF); a competing maker's measurement).
  That is the file Kaitchup saw degrade, so whether IQ3_XXS holds up in agent work is open.
- **A Q8_0 spine helps agents.** One user replaced the dense tensors of the IQ3_XXS file (a 3.83 GiB mix of BF16,
  Q6_K, Q5_K, IQ4_XS and IQ3_S) with unsloth's Q8_0 ones (5.09 GiB). On 9 seeded bugs × 2 runs with the Pi agent,
  both solved 18/18, but tool calls per task fell from 18.2 to 15.2 (p ≈ 0.04) and wrong claims per fix from 0.72 to
  0.28 (p ≈ 0.09) ([discussion #65](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/65)).
- **Low effort costs more on these quants.** The ISTA maintainer says medium-effort performance "has degraded
  significantly" ([#18](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/18)); testers
  run IQ2_XS and IQ3_XXS at xhigh with a reasoning budget of 12k tokens or more.
- **The Coder trades knowledge for size.** Keeping 256 of the 512 experts costs 3.6 points of SWE-bench Verified
  (79.2 against 82.8, harness not stated) with LiveCodeBench unchanged. On 4 × P100 one user measured factual recall
  falling from 63 % to 33 %, with more confident wrong answers
  ([discussion #19](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/19)). Its
  users split between "fine as a coding subagent" and "loops on long tasks".
- **Swift 1.5 thinks less at equal quality, at xhigh.** The maker reports 31–57 % fewer thinking tokens and Terminal-
  Bench 2.1 69.66 against 67.64. An independent Aider polyglot run gave 41.1 % against 40.2 % on the first try at
  about 40 % of the tokens and time per case
  ([r/LocalLLaMA via mirror](https://prismix.dev/news/e4614f039e53)). Two users found Swift's IQ3_XXS loopier or
  needier than ISTA's in agents
  ([#8](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/8),
  [#16](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/16)).
- **The n-gram table tolerates 4.5 bits.** With only the table quantized, KLD against BF16 is 0.0255 for IQ4_NL and
  0.0162 for Q8_0 ([Delgon](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF)). AtomicChat's 6-bit and
  8.5-bit tables differ by 0.0005 in otherwise identical builds, and one user's HumanEval difference between IQ4_NL and
  Q8_0 tables was not significant. Nobody has tested it below 4 bits.
- **Pruning stacks on quantization.** Removing 37.5 % of the experts costs about as much KLD (0.39) as going from
  BF16 to Q2_0 (0.42), and the two add up
  ([90-008 REAP card](https://huggingface.co/90-008/Swift-1.5-Qwen3.8-Flash-Next-REAP-Pruned-GGUF)).

### Ranked for agentic coding, among the files that fit

1. **GSQ-RCO IQ3_XXS** (ISTA 47.0 GB; Swift 47.2 GB): the best-evidenced file that fits, close to the full model on
   short tasks; agentic behaviour unproven.
2. **GSQ-RCO IQ2_XS** (Swift 39.3 GB, ISTA 39.2 GB): about 4 points lower on the task mean; loops in thinking at
   times (7 of 10 seeds in agentionai's test of ISTA's build).
3. **GSQ-RCO Q2_0** (ISTA 37.6 GB, Swift 37.7 GB): the same task mean as IQ2_XS, higher KLD, fewer loops.
4. **GSQ-RCO-Coder** (29.6 GB): code close to the full model, general knowledge halved.
5. **Gyro-M** (37.6 GB, trellis): the best KLD per GB by its maker's numbers, but it needs agentionai's fork and has
   no independent evidence.
6. **unsloth UD-IQ1_S / UD-IQ1_M**, **REAP and REAM prunes**: no evidence that they beat the files above.

### Known problems in agent use

These come from the base model and show up on every quant:
- **Thinking:** xhigh effort (the default) is very verbose: 240M output tokens to run Artificial Analysis's index,
  against a median of 140M.
- **Prompt cache:** changing `reasoning_effort` within a session invalidates it, because the chat template puts the
  effort first.
- **Language:** the model switches into Chinese mid-answer, most often in Q2_0.
- **Empty turns:** it sometimes ends a turn empty right after a tool result.
- **Invented context:** it sometimes invents tasks or instructions.

Many bad reports come from the Strata engine (malformed tool calls, greedy defaults), not from the files
([Strata #2685](https://github.com/Niko1221/Strata/issues/2685)).

### How to compare the versions

- **No independent leaderboard scores any quantized, pruned or fine-tuned Flash-Next.** Artificial Analysis rates
  only the full model (Intelligence Index 40, #10 of 122).
- **Makers' KLD sets are not comparable with each other.** The same file scores 2–3× apart on different sets
  (AtomicChat's Q4_K_M: 0.228 on agentionai's set, 0.084 on its own).
- **The project compares files with its own two steps, as for Ornith:** KLD against a Q8_0 reference on the same
  three texts (wikitext-2, held-out agent sessions, code), on the quality machine; then the
  [agent check](../../../bench/agent_check/README.md) on the rig for the two best files.

## 4. Engines

| Engine | Runs on compute 6.1? | Host RAM it needs for Flash-Next | Verdict |
|---|---|---|---|
| **Miner Llama** (upstream llama.cpp's `qwen4exp`) | yes | page cache only: the table stays on disk | **the path** |
| Strata 0.2 | experimental; puts a copy of the dense weights and ~2 GB of buffers on every card, and turns off cards of 8 GB or less | experts in RAM + ~10 GB | blocked |
| tqLLM, ninfer, capy-flash, ninfer-offload | no (tensor-core or sm_89+/sm_120 code) | 8–80 GB | blocked ([engine_infer.md](engine_infer.md), [tqllm.md](tqllm.md)) |
| ninfer-all | no build below sm_70 | ~1.1 GB peak with experts and table on disk | blocked by architecture |
| agentionai's llama.cpp fork (Gyro) | probably: dp4a trellis kernels without architecture guards (not built) | like llama.cpp, plus a table-on-disk option | runs, but est. 2–3.5× slower per expert weight than Q2_0 |
| vLLM, DwarfStar, colibri, AirLLM | no, or 8–95 GB of RAM | — | blocked |

How llama.cpp runs this model on Pascal (read in the v3 source):
- **The table:** at 28.8 GB it is a lazy tensor (`--lazy-mode auto`, the default, above 4 GiB). It stays
  memory-mapped with random-access advice, is never preloaded or locked, and lives in a CPU buffer whatever `-ot`
  says.
- **Per token:** the host hashes the 16 row numbers, asks the kernel to prefetch them, and the CPU gathers the rows;
  10 KB per token then goes to the card with layer 1. `--lazy-mode off` would load the whole table into RAM and must
  not be used on the rig.
- **Hyper-connections:** CUDA runs them as fused ops in F32. The BF16 projections use a software-converting
  matrix-vector kernel for 1–8 tokens and BF16 cuBLAS for 9–128 tokens, which is unmeasured on Pascal.
- **The indexer:** it runs on sm_61 through SIMT kernels. The sparse attention gather that reads only the ~2,051
  selected cells needs Turing tensor cores, so on Pascal the selection becomes a mask, and the 12 attention layers read
  the whole cache every token: 768 MiB at 32k, 3 GiB at 128k (F16).
- **MTP:** it works since b11330 with a separate draft file. On 4 × Tesla P40 (Pascal) one user measured 21.6 tok/s
  without speculation, and 30.6 (prose) / 24.5 (code) with Unsloth's Q4_K_M draft at an acceptance of 0.45; prefill
  ran at 320–330 tok/s ([llama.cpp #29174](https://github.com/ggml-org/llama.cpp/issues/29174)).

## 5. The n-gram table on the rig

- **Page cache:** during agent runs the rig keeps 495–691 MB of memory available
  ([Ornith](../../benchmarks/2026-10-27-ornith-122b.md), [answer store](../../benchmarks/2026-10-27-answer-store.md)).
  A 4 KiB page holds one useful 90-byte row, because the hash scatters rows. With
  [ninfer-all's measured hot-row curve](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md)
  (33.8 % hits for a 256 MiB row cache, 49.6 % at 4 GiB and more), that gives an est. 15–30 % hit rate: **11–14 reads
  of 4 KiB from disk per token.**
- **Cost per medium** (random 4 KiB reads measured by [ssd-tester](https://ssd-tester.com/); per-token costs est.):

  | Medium | 4 KiB read at queue depth 1 | Queueing | Per decoded token | Per prompt token | Cold 13.7k-token first turn |
  |---|---|---|---|---|---|
  | SATA SSD | 88–107 µs | 53–56k IOPS | ~0.2–0.3 ms | ~0.25 ms | ~4 s |
  | USB 3 SSD with UAS | ~180 µs | helps | ~0.3–1 ms | ~0.5–1.3 ms | ~7–18 s |
  | USB 3 flash stick | 367–588 µs | no gain (~2.7k IOPS) | **~4–8 ms** | **~5 ms** | **~65 s** |

  Put the table on an SSD (internal or USB with UAS). A flash stick costs 5–11 % of decode, 50–95 % of prefill, and
  most of what MTP gains, because a verify step reads (k+1) × 16 rows.
- **The table's format:** rows are 160 wide, so only 32-weight-block formats fit (Q4_0, IQ4_NL, Q5_0, Q8_0, F16).
  Every format costs one 4 KiB page per row read, so a smaller table saves no I/O, and a Q8_0 table (54.4 GB) costs
  only disk space.
- **Ideas for later:**
  - A hot-row sidecar: the most-used rows packed together, so the page cache holds ~45 useful rows per page instead of
    one.
  - A row cache with direct reads, as in agentionai's `llama-ple-disk.cpp`.
  - Hot rows kept on a card.
  - Prefetching the rows of MTP drafts.

## 6. Formats on Pascal and the time per token

Speed per expert weight on a P106 (from the [low-bit note](pascal_lowbit_format.md) and the v3 kernel work; est. where
marked):

| Format | Bits per weight | ps per weight | Why |
|---|---|---|---|
| Q4_0 (reference) | 4.5 | 4.2 measured | fork's Pascal kernel |
| **Q2_0** | 2.25 | est. 3–6 | `__byte_perm` + dp4a, no table lookups; llama.cpp's generic kernel only, no Pascal-tuned one yet |
| IQ2_S | 2.56 | 10.8 measured; 6.4–9.2 with v3's IQ kernels | grid lookups |
| IQ2_XS | 2.31 | est. 10–12 | grid and sign-table lookups |
| IQ3_XXS | 3.06 | 20.5 measured (llama.cpp's kernel) | grid lookups; v3's kernels cut Ornith's IQ expert time per token from 31.6 to 17.3 ms ([data](../../data/2026-10-27_iq_kernels/README.md)) |
| IQ3_S | 3.44 | est. 18–21 | grid lookups |
| IQ4_NL | 4.5 | est. 6–10 | 16-entry table in registers |
| Gyro TQ2_T (trellis) | 2.125 | est. 9–15 | a 32-bit multiply and two shared-memory lookups per 4 weights |
| IQ1_M | 1.75 | est. 9–14 | no batched kernel: slow prefill |

The 640-wide `ffn_down_exps` rows rule out 256-weight-block formats (K-quants, IQ2/IQ3, IQ4_XS) for the down
projections, which is why the GSQ-RCO files use Q2_0 or IQ4_NL there.

Time per token for the Swift Q2_0 file at short context (est.):

| Part | Per token | Time on nine P106 |
|---|---|---|
| Dense tensors other than hyper-connections and router | ~1.8 GB | ~12 ms |
| Hyper-connection projections, BF16 | 1.26 GB | ~11–14 ms |
| Router, F32 | 0.25 GB | ~3.5 ms |
| Routed experts, Q2_0 | 0.66 GB (2.36 G weights) | ~7–14 ms |
| Operations without weights (DeltaNet, attention, indexer, mixes, norms) | — | ~30–40 ms (Ornith: 26 ms) |
| Four residual streams between cards | 40 KB per card boundary | ~2.5–3 ms |
| N-gram table on a SATA SSD | 11–14 page reads | ~0.3 ms |
| **Total** | ~4.2 GB | **~66–86 ms: 12–15 tok/s** |

The same file with IQ3_XXS-class experts spends est. 30–48 ms on experts instead of 7–14 (est. 9–11 tok/s); v3's IQ
kernels bring part of that back.

## 7. Which version to make Pascal-friendly

| Rank | Candidate | Quality for agents | Fits (cache in q8_0) | Route to Pascal-friendly | Odds |
|---|---|---|---|---|---|
| 1 | **GSQ-RCO Q2_0**, ISTA and Swift builds | 3rd of the files that fit; Swift thinks less | 262k and MTP, with ~5 GB to spare | Dense fixes (router F16, hyper-connections Q8_0, optionally a Q8_0 spine, +1.3 GB), a Pascal-tuned Q2_0 expert kernel, MTP | **High**: every format already runs in Miner Llama, and the experts are already in the fastest low-bit format |
| 2 | **GSQ-RCO IQ3_XXS**, ISTA build | 1st | about 128k, no room for MTP; needs a fit probe | Hyper-connections in Q8_0 (which also frees ~0.6 GB), v3's IQ2/IQ3 kernels | **Medium**: the best quality, but the slowest experts and the tightest fit |
| 3 | GSQ-RCO IQ2_XS, Swift build | 2nd | 262k, MTP likely | as rank 2 (mostly IQ2_S experts, Q2_0 down) | Medium: IQ2_S kernels exist; thinking loops |
| 4 | GSQ-RCO-Coder | 4th | 262k and MTP easily | as rank 2; mixed IQ formats | Medium: compact, but general knowledge halved |
| 5 | Own file from BF16 (base or Swift 1.5) | unknown | chosen to fit | Experts in Q2_0 or the N2P/N3P formats from [route 2](pascal_lowbit_format.md), dense in fast formats, built on the quality machine's CPU from the 354 GB BF16 | Medium-low: llama-quantize's own Q2_0 quantizer is unusable for this, so it needs GSQ-style rounding or the new formats; highest ceiling |
| 6 | Gyro-M | 5th (maker-only evidence) | 262k | Port the trellis formats into Miner Llama | Low: est. 2–3.5× slower per weight than Q2_0; no BF16 and no published encoder |

How far below these files the experts can go, and why sub-1-bit methods do not help: [sub_1_bit_quants.md](sub_1_bit_quants.md).

Not pursued: unsloth IQ1 files (worse per GB than IQ2_XS), REAP/REAM prunes (no agentic evidence; their loss adds
to quantization's), Mach-2, Victoria and Niwaki (custom forks, no BF16 source), ThinkingCap and Tinfield-1 (no file
in the rig's range except Tinfield-1 Mini, unbenchmarked).

**Recommendation:** run the first two ranks side by side. Q2_0 is the file most likely to become fast on the rig and
leaves room for 262k and MTP; IQ3_XXS is the quality reference, and if KLD and the agent check show Q2_0 losing too
much, IQ3_XXS with v3's IQ kernels is the fallback. Compare the ISTA and Swift builds at the Q2_0 rung, since Swift's
shorter thinking matters on a ~15 tok/s rig.

## 8. Speed levers, in order of gain per effort

1. **The router as F16 instead of F32, in the Swift files.** F32 there is an exact copy of the BF16 source, so F16
   loses nothing in practice. It saves 0.13 GB and ~1.5–2 ms per token. ISTA's files already store it in BF16.
   llama-quantize never touches `ffn_gate_inp`, so this takes a GGUF rewrite.
2. **Hyper-connection projections as Q8_0.** They save ~0.6 GB per token (est. 5–8 ms) and avoid the BF16 cuBLAS path
   in short prefills. ninfer-all ships them as Q8_0. llama-quantize accepts `--tensor-type hc_=q8_0` for qwen4exp.
   Needs a KLD check.
3. **MTP.** On P40s it gave +41 % on prose and +13 % on code. On the rig a verify step reads the dense part once while
   expert reads grow with the draft length: est. +15–40 %. Only with the table on an SSD.
4. **A Pascal-tuned Q2_0 expert kernel**, in the style of the fork's Q4_0 kernel. For the IQ files, the v3 IQ2_S and
   IQ3_XXS kernels apply as they are.
5. **A Pascal sparse-attention kernel** that walks the indexer's existing list of selected cells: est. −5 ms per token
   at 32k and −20 ms or more at 128k. This is the largest long-context lever.
6. **Smaller hops between cards:** F16 for the four-stream residual halves its 40 KB per boundary; needs a KLD check.
   Fewer cards also help.
7. **Hybrid-model work from v3:** Flash-Next is a hybrid model like Ornith, so the recurrent-state replay and
   pipelined speculation planned for hybrids apply to it as well.
8. **The table:** a hot-row sidecar (section 5).

## 9. What settles the open questions

| Step | Where | What it settles |
|---|---|---|
| Q8_0 reference of the base model and KLD of ISTA Q2_0, IQ2_XS, IQ3_XXS, the Swift builds and the Coder on wikitext, agent sessions and code | quality machine (Q8_0 transformer ~134 GB across seven GPUs) | quality rank under the project's own method |
| KLD of the dense fixes: router F16, hyper-connections Q8_0, Q8_0 spine | quality machine | whether levers 1–2 are free |
| Replay of agent-check sessions through the `qwen4exp` hash, with simulated page and row caches | quality machine, CPU only | real table reads per token; value of a hot-row sidecar |
| `fio` 4 KiB random reads on the internal SSD, a stick and a USB SSD | rig, minutes | the per-medium cost |
| Q2_0, IQ4_NL and the BF16 hyper-connection shapes in `bench/mmvq_pascal` | rig, one card | ps per weight of the formats that matter |
| `--fit` memory probe ([scripts/fit_probe.sh](../../../scripts/fit_probe.sh)) at 128k and 262k for Q2_0 and IQ3_XXS | rig | the real context limits |
| The agent check of the two best files | rig | the choice |

Each step and each lever of section 8 is a row of the [registry](../../../quants/experiments.md#qwen38-flash-next)
(Q213–Q232).

## Sources

- Census and arithmetic: [docs/data/2026-10-27_flash_next_variants/](../../data/2026-10-27_flash_next_variants/README.md).
- Model cards: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  [ISTA Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
  [Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next),
  [Swift GSQ-RCO](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF),
  [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF),
  [Delgon table sidecar](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF); Hub API listings
  ([search](https://huggingface.co/api/models?search=Flash-Next&limit=500&sort=downloads&direction=-1)).
- Independent tests: [Kaitchup #163](https://kaitchup.substack.com/p/agentic-coding-with-qwen38-flash),
  [ISTA #65](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/65),
  [ISTA Coder #19](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/19),
  [HN 49953495](https://news.ycombinator.com/item?id=49953495),
  [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-flash-next).
- llama.cpp: [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (qwen4exp),
  [#27837](https://github.com/ggml-org/llama.cpp/pull/27837) (lazy table),
  [#29599](https://github.com/ggml-org/llama.cpp/pull/29599) (row prefetch),
  [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) (sparse attention, Turing+),
  [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) (MTP),
  [#29174](https://github.com/ggml-org/llama.cpp/issues/29174) (P40 numbers),
  [#25707](https://github.com/ggml-org/llama.cpp/pull/25707) (Q2_0 on CUDA); in the v3 source
  `src/models/qwen4exp.cpp`, `src/llama-model-loader.cpp` (lazy reads), `ggml/src/ggml-cuda/fattn.cu` (sparse gather
  condition), `src/llama-quant.cpp` (router exclusion).
- Engines: [Strata docs](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md),
  [ninfer-all Flash-Next doc](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md),
  [agentionai llama.cpp fork](https://gitlab.com/agentionai-group/llama.cpp).
