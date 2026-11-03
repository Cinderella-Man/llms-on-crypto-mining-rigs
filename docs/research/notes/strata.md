# Strata on Pascal rigs: why it leads for Qwen3.8-Flash-Next, what to port, and why not to fork it (2026-11-02)

This note answers three questions about [Strata](https://github.com/Niko1221/Strata), the engine users recommend for
[Qwen3.8-Flash-Next](flash_next_variants.md): why it leads, whether its techniques can move into Miner Llama, and
whether a fork for Pascal cards is worth making. Sources were read on 2026-11-02: Strata's source at `c62d5e3` (v0.2.9),
Miner Llama at `37e717ff6`, issue trackers, model cards and third-party benchmarks. Nothing was built or run.

**Do not fork Strata: keep Miner Llama as the one engine for Pascal rigs, port Strata's kernel techniques into it, and let R13's profile set the order.** Strata earned its reputation on one case: a single consumer GPU with far more host RAM than VRAM, where same-machine tests put it 1.8–4.9× ahead of llama.cpp on decode and about 3× on prompts. That lead comes from an expert tier spread across VRAM, RAM and CPU, plus MTP drafting and fused kernels. "Go-to" is a community verdict, and neither Qwen nor the quant makers recommend it. A nine-card P106 rig removes that tier: 4 GB of RAM, no AVX, and PCIe 1.0 x1 links that take an estimated 7 ms per streamed expert. The source shows nothing that makes a run impossible. A fork would still cost an estimated 12–27 engineer-weeks for one tuned model family, could not keep every Flash-Next expert in VRAM below about nine 6 GB cards, and would track a codebase taking about 1,000 commits a week. Strata's value here is the code around the weights: in-place recurrent state, fused launches, sparse attention without tensor cores and verify/commit replay, which together may cut a short-context token from about 76 ms to 52–58 ms before speculation. Every rig figure in this report is an estimate, because no Flash-Next token has run on a P106; R13 must replace them before they guide engineering order.

*Tags: **[code]** read in source, Strata `c62d5e3` (v0.2.9) or Miner Llama `37e717ff6`. **[maker]** stated in Strata's docs, notes or comments. **[independent]** third-party measurement. **[tracker]** third-party measurement filed in Strata's own repo, a venue that selects for its users. **[project]** measured by this project. **[estimate]** arithmetic or judgement, never run.*

| Evidence class | What it settles | What it leaves open |
|---|---|---|
| [code] | sm_61 is a real build target; every decode and prompt kernel has a path without tensor cores; the layer split needs no P2P; the 8 GB rule is setup policy, not an engine limit; n-gram reads bypass the page cache | Speed, and host memory at 4 GB |
| [maker] | RAM needs experts + ~10 GB by default; stage trim, mapped experts and pipeline windows exist; old cards are community paths the maintainers test only on a P100 | Anything on 5-SM cards, nine stages or x1 |
| [independent], [tracker] | 1.8–4.9× on one GPU; about 1.2–1.8× on Pascal once quants are allowed for; agent reliability reports conflict | Any P106, any x1-only rig, any host under 16 GB |
| [project] | Miner Llama on Ornith 122B over nine cards: 9.7 tok/s on v2.0, 14.5 with the valid1 file on v3 kernels | Strata against llama.cpp on one file; Strata on any Pascal card |
| [estimate] | Nothing yet: the token budget, port gains, fit sums and engineer-weeks all wait on a real run | |

## Strata wins where VRAM runs short and host RAM runs deep

The label comes from users. No source found uses the phrase "go-to", and Strata reached 38,311 stars 39 days after its repository was created ([GitHub](https://github.com/Niko1221/Strata)). Qwen's card leaves it out ([Qwen card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)), ukisai's Swift 1.5 card recommends only llama.cpp ([Swift card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), and ISTA-DASLab lists llama.cpp first and Strata as a third-party option ([ISTA card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)). It was not first either: llama.cpp merged `qwen4exp` on 2026-08-27 ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)), a month before Strata existed.

The design treats Flash-Next as a placement problem. On a Q2_0 file a token reads about 3.5 GB of dense weights but only 0.66 GB of the 34 GB of routed experts, so Strata keeps dense weights on the GPU, pins every expert in RAM, and splits each layer's ten routed experts between VRAM hits, DMA copies and CPU workers **[maker]** ([paper](https://github.com/Niko1221/Strata/blob/c62d5e3/docs/paper/Strata-Paper.pdf)). The code compiles the model's shapes into its kernels ([geometry.hpp](https://github.com/Niko1221/Strata/blob/c62d5e3/include/strata/core/geometry.hpp)) and leaves recurrent state untouched until accepted tokens replay ([verify.hpp](https://github.com/Niko1221/Strata/blob/c62d5e3/include/strata/core/verify.hpp)) **[code]**. MTP adds 1.6–1.8× **[maker]** ([paper](https://github.com/Niko1221/Strata/blob/c62d5e3/docs/paper/Strata-Paper.pdf)). A Python server with turn-boundary checkpoints and client-specific prompt pinning sits on top, and that server work probably matters more than kernels after the first agent turn **[estimate]**.

| Date | Hardware and file | llama.cpp decode | Strata decode | Ratio | Limits of the test |
|---|---|---|---|---|---|
| 10-05 | A100, whole card, same IQ2_XS file | 63.0 | 110.7 | 1.8× | No MTP or int8 KV on llama.cpp |
| 10-05 | A100 capped at ~12.3 GiB, 64 cores | 23.4 | 62.9 | 2.7× | Same |
| 10-05 | Same cap, 8 cores | 8.8 | 43.1 | 4.9× | Strata's core count unverified |
| 10-10 | RTX 3090, IQ3_XXS | 17.5–19.1 | 77.6–78.0 drafts off; ~119 as shipped | 4.1–4.5× drafts off | Old llama.cpp, no expert cache |
| 10-18 | V100 32 GB, 16 GB RAM, same Coder IQ1_M files | 41.7–42.9 | 98–101; 60.4–70.9 drafts off | 2.4×; ~1.5× drafts off | Tracker report |

Rows one to three are [independent] ([SOTAAZ](https://sotaaz.com/post/strata-qwen-flash-next-llamacpp-en)), row four is [independent] ([InsiderLLM](https://insiderllm.com/guides/strata-vs-llama-cpp-rtx-3090/)), and row five is [tracker] ([#3082](https://github.com/Niko1221/Strata/issues/3082)). All three show prompts about 3× faster even against tuned llama.cpp, the lead most likely to last for agents. All predate 2026-10-19, and most ran llama.cpp without its best features. Upstream has since merged MTP ([#29761](https://github.com/ggml-org/llama.cpp/pull/29761)), an expert cache ([#29887](https://github.com/ggml-org/llama.cpp/pull/29887)) and a Strata-derived attention selection ([#30965](https://github.com/ggml-org/llama.cpp/pull/30965)). **No same-machine rerun of current llama.cpp against Strata 0.2.x exists**, so the size of today's gap is unknown.

The same evidence bounds the claim. When the model fits in VRAM, llama.cpp ties or wins: Qwen3.6 on a V100 gave Strata 59.7 tok/s against 87.3 at a 16K prompt ([#2464](https://github.com/Niko1221/Strata/issues/2464)), and 2 × P100 gave 58 against 91–95 ([#3294](https://github.com/Niko1221/Strata/issues/3294)) **[tracker]**. Agent reliability is bimodal: one RTX 5070 run scored 0 of 10 tasks on Strata against 5 of 5 clean on llama.cpp with the same quant, the issue was closed in a tracker reset without a fix ([#2685](https://github.com/Niko1221/Strata/issues/2685)), and other users ran 20 of 20 and 66 of 72. Speed is partly bought with changed output: under VRAM pressure only 12 of 30 temperature-0 repeats matched, and matching all 30 cost a drop from 62.9 to 46.3 tok/s **[independent]** ([SOTAAZ](https://sotaaz.com/post/strata-qwen-flash-next-llamacpp-en)).

On Pascal the multiplier shrinks. A GTX 1070 Ti 8 GB at this commit, with a 12-core CPU and 126 GB RAM, gave 30.6–31.8 tok/s decode, about 250 tok/s prompts and 66.6 s to first token at 16K, and reported no sm_61 gain from 0.1.44 to 0.2.9 **[tracker]** ([#4920](https://github.com/Niko1221/Strata/pull/4920)). 2 × P40 reached 36–40 tok/s ([#3616](https://github.com/Niko1221/Strata/pull/3616)) **[tracker]**. 4 × P40 on llama.cpp reached 21.6, or 30.6 prose and 24.5 code with MTP, at 320–330 tok/s prefill **[independent]** ([#29174](https://github.com/ggml-org/llama.cpp/issues/29174)). Files differ, so the resulting 1.2–1.8× is an **[estimate]**. Strong hosts carry those Pascal numbers: on a 1080 Ti, 58–84 % of experts ran on AVX2 cores ([#2684](https://github.com/Niko1221/Strata/issues/2684)).

This project has no same-file comparison. Strata has run on three RTX 3090s of a 7-GPU quant machine, but never against llama.cpp on one file, every preset ran with its prompt cache off, and no `bench/agent_check` run exists, so none of those runs counts as agentic speed.

## Where the plan stands

The planned files are Swift-1.5 GSQ-RCO Q2_0 and IQ3_XXS (R11, Q213), and R13 runs Q2_0. Neither is an easy Strata candidate. Setup's model list offers Q2_0 for the original model only, with a comment that the pack tool cannot yet handle a layer split across Swift's shards ([setup.py:411-413](https://github.com/Niko1221/Strata/blob/c62d5e3/setup.py#L411-L413)) **[code]**. The pack tool's header at the same commit describes split layers ([iq_pack.py:26-33](https://github.com/Niko1221/Strata/blob/c62d5e3/tools/iq_pack.py#L26-L33)) **[code]**, so this looks like a setup-flow limit; nobody has repeated such a pack on v0.2.9. IQ3_XXS packs normally but cannot stay fully resident on nine 6 GB cards, about 7.5 GiB short **[estimate]**. IQ1_S has no Strata kernels **[code]**, and it is parked here anyway (Q243).

No Flash-Next token has run on a P106, in this project or in any report found, and no Strata build has run on a Pascal card in this project. The one P106 mention is [#4739](https://github.com/Niko1221/Strata/issues/4739): an owner of 9 × P106-100 who runs a small 7B model under Ollama, with no test following. A six-card layer split has run, on 6 × V100 16 GB with 88 GB RAM ([#4687](https://github.com/Niko1221/Strata/issues/4687)) **[tracker]**; nine stages are untested. Cards of 8 GB or less start Off only beside a bigger usable card or below 4 GB ([gpu_roles.py:116-128](https://github.com/Niko1221/Strata/blob/c62d5e3/strata_control/gpu_roles.py#L116-L128)) **[code]**.

## A fork would inherit the kernels and lose the tier that justifies Strata

**[code]** Below sm_75 a CMake flag admits the build ([CMakeLists.txt:257-262](https://github.com/Niko1221/Strata/blob/c62d5e3/CMakeLists.txt#L257-L262)), and tensor-core, cp.async and PDL paths all have SIMT fallbacks, so Pascal costs speed, not correctness. A tracker user ran this exact commit on sm_61 with CUDA 12.9 and driver 580.178.04 on Ubuntu 24.04 ([#4920](https://github.com/Niko1221/Strata/pull/4920)). The dense copy per card has an off switch ([generate.cpp:4476-4486](https://github.com/Niko1221/Strata/blob/c62d5e3/src/program/generate.cpp#L4476-L4486)), and the "~2 GB of buffers" is mostly borrowed from the expert cache ([generate.cpp:3681-3703](https://github.com/Niko1221/Strata/blob/c62d5e3/src/program/generate.cpp#L3681-L3703)). The split hands off 51.2 KB per row through pinned host memory against llama.cpp's ~40 KB, about 17 ms across eight boundaries for a 4-row window on this rig **[estimate]**, which is small next to the window itself. With every expert resident a stage runs with no CPU doorbells at all ([verify.cpp:1138-1156](https://github.com/Niko1221/Strata/blob/c62d5e3/src/core/verify.cpp#L1138-L1156)). Of the audit's 18 blockers, eight are flags or launcher work.

What remains is one hard constraint: every expert in VRAM and almost nothing in RAM. A streamed Q2_0 expert takes 7.3 ms on x1, or 3.5 s for all 480 experts of a token **[estimate]**, and the CPU pool falls back to one worker without AVX. Ready-made engines need AVX2, so a custom SSE4.2-floor build is mandatory ([CMakeLists.txt:35-39](https://github.com/Niko1221/Strata/blob/c62d5e3/CMakeLists.txt#L35-L39)) **[code]**. Mapped experts exist, but nobody has reported Strata below about 16 GB of host RAM, and a 9,589-line Python server adds its own footprint. Fit on a 6,068 MiB card shows how little room is left **[estimate]**:

| Per-card item, MiB, Q2_0 at 64K | Default-ish | Tightest sane |
|---|---:|---:|
| Six layers of experts and dense, 770 each | 4,620 | 4,620 |
| VRAM reserve | 700 | 500 |
| Verify windows | 96 | 96 |
| Own prompt buffers | 884 (chunk 512) | 714 (chunk 256) |
| Total against 6,068 | 6,300, over | 5,930, 138 spare |

That spare comes before the kernel image, the n-gram tensors and attention KV, and the 839 MiB drafter sits on the last card. Forty-eight layers need only three of nine cards at six layers, so placement helps, but 500 MiB of headroom is about where the engine is reported to hang ([#2065](https://github.com/Niko1221/Strata/issues/2065)) **[tracker]**. Small chunks also hurt the metric that matters: a setup comment records prompts 6.2× slower when a chunk fell to 512 tokens ([setup.py:1640-1642](https://github.com/Niko1221/Strata/blob/c62d5e3/setup.py#L1640-L1642)) **[code]**. By the same sum, all-resident Q2_0 needs about 9 × 6 GB, 6 × 8 GB or 5 × 10 GB cards. **A 3-card rig of 6 GB cards cannot run Flash-Next on Strata at all**, and Strata has no smaller model to offer it. Miner Llama's Flash-Next floor is also high: Swift Q2_0 needs seven 6 GB cards at 64k and eight at 128k, and IQ3_XXS all nine ([fit probe](../../data/2026-11-03_flash_next_fit/README.md)). But it runs smaller models on small rigs, and every port below works per layer or per card.

| Fork phase | Work | Engineer-weeks [estimate] |
|---|---|---:|
| A | sm_61 + SSE4.2 build, nine-stage bring-up, ISA fixes | 2–4 |
| B | Host memory, reserve hardening, prompt path, 5-SM tuning, x1 hand-off, 3–12 card planner | 10–23 |
| C | N2P/N3P, Q5_0 dense, plus IQ1_S (1–2 weeks, now droppable) | 3–6 |
| D | Flint, Ornith, drafter, answer store, screened head, tool sampling | 16–32 |

A+B is 12–27 weeks and full parity is 31–65, with 0.25–0.5 weeks a month on a pinned tag or 1–2 tracking `main` **[estimate]**. Only Phase A's range is defensible; the rest depends on a first run. Tuning would not carry over either: tables were fitted on 30-SM P40s, the strongest tricks are gated to 48-SM GP100, and nothing is measured on 6.1 ([OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/c62d5e3/docs/OLDER_GPUS.md)) **[maker]**, while a P106-090 has 5 SMs. One `__restrict__` once halved P40 decode, 36.4 to 17.7 tok/s ([#1469](https://github.com/Niko1221/Strata/issues/1469)) **[tracker]**. Geometry is compile-time, so Flint and Ornith are refused ([model_family.hpp](https://github.com/Niko1221/Strata/blob/c62d5e3/include/strata/artifact/model_family.hpp#L15-L19)) **[code]**, while Miner Llama already runs Flint at 44.6–45.9 tok/s for a coding agent, 12 of 12 tasks, and Ornith at 9.7 tok/s on v2.0 and 14.5 with the valid1 file ([README](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/v3/README.md)) **[project]**. With `generate.cpp` taking 547 commits in a month and upstream testing nothing on 6.1, a pinned fork gets none of Strata's future tuning for free, which removes much of the reason to fork.

The best case for a fork deserves stating. Strata's kernels were written for exactly this model and plan on the device with no host step per layer, so on Flash-Next alone it could beat Miner Llama. The closest precedent is 3 × CMP 50HX 20 GB mining cards at Gen2 x8 on an old Xeon, all experts resident, at 80–82 tok/s and 2,100–2,400 tok/s prompts ([FASTCHIP](https://github.com/FASTCHIP/cmp50hx-llama-research)) **[independent, unverified]**, though that is 60 GB across three far faster cards. **Reopen the fork question only if R13 shows Flash-Next under about 5 tok/s on Miner Llama with kernels as the proven cause, if the owner makes Flash-Next the sole target, or if the rig gains much more RAM and a stronger CPU.** A RAM upgrade alone would remove the host-memory risk but not the CPU or x1 limits. Mining boards add one caution: a V100 on an X99 mining board dropped off the bus under long prefill after a package change ([#4085](https://github.com/Niko1221/Strata/issues/4085)) **[tracker]**.

## Pascal's token goes to launches and copies, which Strata's kernels remove

Strata's headline idea targets the part of a token that matters least here. The project's budget for Swift Q2_0 on nine P106 cards puts experts at 7–14 ms and weightless operations at 30–40 ms of a 66–86 ms token ([flash_next_variants.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/v3/docs/research/notes/flash_next_variants.md)) **[estimate]**. Ornith's profile shows the same shape: GET_ROWS 3.93, RMS_NORM 2.44, CPY 1.20 and CONCAT 0.89 ms outweigh the recurrence op's 4.65 ms **[project]**. Strata stops copying with in-place state, stops launching through fusion (hyper-connections from 672 launches per token to 192), and stops reading the whole cache with sparse attention on plain SIMT kernels **[code]**. Its recurrence kernel is no faster than the one Miner Llama already has **[estimate]**; the wrapper is the win. Attention is the long-context cliff: Q210 measured ~1.17 ms per 1,000 cells on Ornith **[project]**, against 1.59 ms per window for selection plus attention over all 12 layers on a 30-SM P40 at 7K **[maker]** ([P40 report](https://github.com/Niko1221/Strata/blob/c62d5e3/bench/results/2026-10-12-p40-sm61-kernels/README.md)).

| Scenario | Basis | Figure [estimate] |
|---|---|---|
| Miner Llama, Swift Q2_0, short context, no MTP | Part-by-part budget; Ornith and Flint proxies | 66–86 ms, 12–15 tok/s |
| Same, with the port stack | Ranks 1–5, 8, 9 plus Q218, overlaps not summed | 52–58 ms, 17–19 tok/s |
| Strata fork, nine P106, serial | GP100 window data × 4–9.6 (low confidence) | 300–700 ms per window, 4–10 tok/s, ~2× more pipelined |
| Cross-check of the fork figure, mine | P40 56 ms window at 1.72 tokens, × 2 (bandwidth) to × 6 (SMs) | 66–195 ms per token, 5–15 tok/s |

The nearest real number is Ornith 122B on these cards at 98.8 ms per short-context token **[project]**, slower than the Flash-Next budget assumes. Once the bases align, the two engines' ranges overlap, so **the estimates do not show either engine ahead on a P106**. Strata's 37–38 tok/s on one 24 GB P40 is a lead for kernel work and proves nothing here.

The list below keeps the ranks in the porting note, which weigh agentic gain against effort. Gains are [estimate], against a 76 ms mid-case. Each port must become a shape-class template under Q147, so Ornith benefits and 3–12 card rigs inherit it. Strata is MIT ([LICENSE](https://github.com/Niko1221/Strata/blob/c62d5e3/LICENSE)); keep its notices.

| Rank | Technique (Strata source) | Gain [estimate] | Effort | Row |
|---:|---|---|---|---|
| 1 | Drop the redundant `cont` on the conv-state tail (upstream #31443; present at `qwen4exp.cpp:1354`, checked) | ~1 ms | Hours | Q436 |
| 2 | GDN state updated in place (`gdn_step_norm_multi_kernel`) | 3–4 ms | 3–5 days | Q437 |
| 3 | Fused conv window and output norm (`gdn_conv_l2_kernel`, `gdn_step_norm_kernel`) | 2–3 ms | 3–5 days, with rank 2 | Q438 |
| 4 | Sparse attention without tensor cores (`block_scores_kernel`, `block_topk_kernel`, `attn_chunk_kernel_pre75`, `attn_merge_kernel`) | −3 ms at 5k, −15 at 16k, −34 at 32k, −145 at 128k, F16 cache | 2–3 weeks | Q230 |
| 5 | Two-kernel hyper-connection read (`fused_gr`) | 3–5 ms; 9–12 with Q218 | 1–2 weeks | Q439 |
| 6 | Verify reads state, commit replays (`gdn_step_commit_kernel`, `gdn_state_replay_kernel`) | Enabler: MTP +15–40 %, Ornith pipelined 2.4–2.9× | 1–2 weeks | R8 |
| 7 | On-device checkpoints (`STRATA_CKPT_ASYNC`, `_VRAM_SLOTS`, `_GATHER`) | −0.6 to −1.3 s per turn; frees 113 MB of host RAM per checkpoint | ~1 week | Q440 |
| 8 | Pascal Q2_0 expert kernel (`s2_expert_grouped.cu`) | 2–4 ms; Q225 first | 3–5 days | Q223 |
| 9 | Int8 K/V without rotation | 1–2 ms with a q8_0 cache; needs agent-text KLD | Flag + KLD | Q442 |
| 10 | Cost-aware draft window (`DraftPolicy`) | +2–5 % over fixed k | 2–3 days | Q409 |
| 11 | Row cache and unbuffered reads (`ple_reader`) | ≤0.2 ms on SATA; −3 to −6 ms on a USB stick, where it jumps to rank 2 | 3–5 days | Q229 |
| 12 | Tool-call recovery for forms outside the grammar | ≤1–2 % of turns | 2–3 days | Q444 |
| 13 | Control tokens in message text stay text (`mark_think_literals`) | Correctness | 2–3 days | Q443 |
| 14 | Attention step records in device memory | ~0.5 ms at 5k, 3–4 at 32k | With rank 4 | Q441 |
| 15 | Prompt chunks sized for the pipeline (`--prefill-pipe`) | Part of Q264's ~5 % of a task | 3–5 days | Q363 |
| 16 | Device-side wait kernels (`wait_flag_ge_kernel`) | Q157's own +40–90 % at 8k prefill | 1 week | Q157 |
| 17 | Weight-stationary prompt experts (`STRATA_MMQ_WS`) | 10–20 % of prefill GPU time | 2–3 weeks | Q446 |
| 18 | Side branches in the window graph (`STRATA_SIDE_MIXER=2`) | 0–1.5 % | Flag test | Q445 |
| 19 | Raw n-gram rows sent to the card (`STRATA_PLE_COMPACT`) | 50 µs per token; 1–2 % of prefill | 2–3 days | Q447 |
| 20 | Gap-keeping checkpoint eviction (`--prompt-cache-spread`) | ~0 on linear sessions | 1–2 days | Q448 |
| 21 | Incremental prompt tokenization | Tens of ms per turn | 1–2 days | Q449 |
| 22 | Non-finite logits guard | Insurance | 1–2 days | Q451 |
| 23 | Long-context select (`block_scores_dx_kernel`) | ~0 on the agent check | With rank 4 | Q230 |

Sequence them as Q436 first, then Q437 and Q438 on the hybrid branch, then Q230 with Q441, with Q440 landed before any live Flash-Next agent check, because ~0.6 s per checkpoint on every turn would distort the numbers. Three risks sit behind the top ranks. Strata's attention kernel has only run on cards with 28+ SMs, and its thread and shared-memory shape may behave differently on 5 SMs. Replay at depth 4 may crowd the register limit. Miner Llama's own head-128 Pascal attention was slower on the P106, and the reason is unknown.

Several Strata techniques should stay out. KVA failed agent runs (0 of 10 at split 24) and refuses a layer split. The route prior gained speed but scored 4.5 against 5.6 of 10 agent tasks with repetition loops, so its maintainer removed the default. Tail skip, expert caches, CPU experts, PCIe shares and KV streaming all need misses, RAM or fast links the rig lacks. Nibble slots cost 175–350 MiB per layer. Miner Llama also leads Strata where the rig most needs it: `--pipe-depth` to 4 on nine cards against two windows, a staircase drafter, answer store and screened head, text reuse, generation checkpoints, lazy tool grammar where Strata has no grammar decoder ([structured.py](https://github.com/Niko1221/Strata/blob/c62d5e3/serve/structured.py)), and identity gates.

## R13 must replace every estimate before it guides the order

R13 (Q215) runs Swift Q2_0 stock against Miner Llama v3, decode at short and 32k context, then the agent check, after R10, R11 and a table on an SSD (R19 or a USB SSD). It needs a readable copy of Swift's shards first, and an unmodified v3 build, so each port's change is measurable afterwards. As written it yields speed. It must also record the following, on the rig under the usual quiet rules:

| R13 must record | Replaces | Decides |
|---|---|---|
| Per-op profile under `GGML_CUDA_PROFILE` at short, 16k and 32k context, with launches per token | 30–40 ms weightless budget; ~1,850 counted launches; 5–7 µs per launch from Flint; the GET_ROWS attribution | Order of ranks 1–5 |
| ms per token for BF16 hyper-connections, router and Q2_0 experts (Q225 shapes) | 11–14, ~3.5 and 7–14 ms | Q218, Q223, Q439 |
| Attention time against cells on `qwen4exp` | Ornith's 1.17 ms per 1,000 cells | Q230's priority |
| Agent check: time per task, turns, first token per turn, whole-session tok/s, tasks passed, context at each turn | Ornith's 4.6k–7.5k turn ends; "Flash-Next is usable" | The release metric; whether long context matters |
| Stock against Miner Llama ratio on this model | Assumed carry-over of Pascal kernels to `qwen4exp` | Whether engine work beats file work |
| Checkpoint save and restore time for the ~113 MB state on nine cards; host memory available | ~0.6 s; 495–691 MB available | Q440, Q362 |
| Table reads and ms per token on the real medium; cold first-turn time | 11–14 reads; 0.2–0.3 ms on SATA; 10–20 min on a hard disk | Q229, Q434 |
| Prompt tok/s at 1k, 8k and 32k with per-card busy share | Q157 and Q363's estimates | Prompt ports |
| Free VRAM per card in a real run, against R11's projection | `--fit` sums; 138 MiB-style margins | IQ3_XXS, Q226's draft size, Q460 |
| Identity hashes and tool-call parse failures per session | Assumed exactness; the 92.9 % grammar-free baseline | Q444 and every later port's gate |

Until those exist, no P106 tok/s figure for Flash-Next should enter public docs as a result, and any figure should be labelled an estimate. The gate for the fork question is the first row combined with the speed result: under about 5 tok/s with kernels as the proven cause reopens Q466; anything else leaves it parked.

## Registry

Every port and parked idea above is a row of the [registry](../../../quants/experiments.md#qwen38-flash-next): Q436–Q471, with Q230, Q229, Q228, Q223, Q218, Q226, Q409, Q363, Q157, Q416 and Q162 carrying Strata's references. The fork is Q466, parked with the reopen conditions above.

## Conclusion

Strata's popularity answers a question this rig never asks: how to run a model bigger than VRAM on a host with RAM to spare. The rig asks how to spend fewer milliseconds around weights already on the cards, and Strata's best code for that sits in files its users never mention. The research also moved the fork question. "Can Pascal run Strata?" is now probably yes, with effort; the live question is what a single-model engine would cost a multi-model project, and that cost is structural whatever R13 shows. Strata's flat sm_61 record since 0.1.44 means waiting will not improve the trade.

The most portable thing Strata offers may be a warning. Its route prior passed token-rate tests and failed agent tasks, and #2685 shows that a speed path without bit-reproducibility cannot diagnose its own quality loss. Keeping identity hashes and `test-backend-ops` as gates for every port matters as much as any kernel on the list. The flow can also run the other way: a 9 × P106 owner has already asked Strata how little RAM it needs, and Miner Llama's many-card, low-host work is the answer that rig is looking for.
