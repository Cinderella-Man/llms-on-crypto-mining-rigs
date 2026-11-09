# Flash-Next after v3.0, second review: the real workload, what others already built, and the order of work (2026-11-09)

This note continues [the first review](flash_next_review_2026-11.md) of the same day. It checks that review's main
claims against the data folders and the release source, reads the owner's bench at the level of its settings, reads
what upstream llama.cpp and other forks merged or published since the fork's base, and gives one order of work. Where
the two notes differ, this one holds: section 1 lists every difference.

Conventions:
- "est." marks my arithmetic or inference, with its basis beside it. Every other number is measured and quoted from the
  linked folder or page, one run unless repeats are listed.
- "The release line" is [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env) on build
  `1551f5fe1`: UkisAI's Swift-1.5 GSQ-RCO Q2_0, nine P106-090 cards, the MTP draft, a 65,536-token context. Its
  agent-check arm is `t1swift` ([data](../../data/2026-11-05_release_flashnext/README.md)).
- "The bench" is the [Elixir SWE Bench](https://github.com/Cinderella-Man/elixir-swe-bench): 20 repository-level
  cases under the pi agent. Its numbers here are from its published `results.json` and from the sister repo's read
  of it ([E14](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs/tree/main/docs/data/2026-11-09_bench_runs_read)).
- Nothing ran on any machine for this note: file reads, source reads of the release tree, and web pages.

## The short answer

- **A solved real task costs est. 100 to 123 minutes of rig time today, and more than half of that is spent on
  attempts that end at the generation budget.** The bench keeps the whole history in a 262,144-token window with no
  compaction, and a Flash-Next attempt writes 40,557 and 47,293 tokens on average (the two files it has run; Swift's
  file writes fewer and has no bench run yet). The release preset holds 65,536 tokens, so it cannot hold a typical
  attempt at all. Section 2 has the model and its inputs.
- **The headline number should be minutes per solved task at depth,** with the bug-fix check kept as the regression
  test. Every lever below is priced on it.
- **Depth is the first engine problem, and three parts of it were never measured:** speed beyond 32k, the draft's
  value at depth (in two other engines' published runs the MTP draft loses at depth), and a 131,072-token line. The
  sparse Pascal attention is the right fix for the measured slope; it has still never run on a GPU.
- **The prompt path's two structural problems are solved in a public fork on a similar machine.** A six-card fork for
  starved PCIe links took prompt reading from 859 to 2,500 tok/s by removing three causes of serial cards, and takes
  checkpoints without draining the pipeline. The registry had no row for it. Read it before writing Q770.
- **Upstream merged or opened this week what three planned rows would build:** the MTP draft reading the target's
  hidden rows on the device (#31927, merged 2026-11-09), replay instead of rollback snapshots for the recurrent state
  (#31669, open), and two recurrent-state bug fixes. The rebase (`-ladder6`) comes before that code.
- **The release tree carries two upstream recurrent-state bugs, and a third exactness problem starts the moment a
  draft outgrows its snapshots.** The build resets the recurrent state by multiplying by zero, so a NaN stays for
  good (upstream #31913). Its slot finder swaps cells without their snapshot count (#31656; by reading, one slot does
  not reach it). The sampling skew this project found (Q808) does not touch the preset, whose three drafts stay
  within the rollback snapshots. It does touch longer drafts and lookup drafts, so `LLAMA_SPEC_KEEP_DRAWS` goes on
  before those are measured.
- **Reasoning effort is not the first lever.** The model card says lower effort "does not always reduce overall task
  completion time", and the GSQ-RCO maintainer reports medium effort degraded on these quants. It is a measurement
  with a real downside, and the sister repo already has its row (E16).
- **File quality: measure the two ends before the neighbours.** Swift Q2_0 against Swift IQ3_S, three seeds each. One
  run of 82 tasks cannot separate Q2_0 from IQ2_XS, and IQ2_XS loops in thinking in two independent reports. The
  sister repo's E1 and E13 are that bracket on the bench; one lane serves both repos.
- **Two rigs were moved down on a stale estimate.** Re-priced on v3.0's token, Swift IQ3_S on 18 cards is est. 11
  tok/s plain, not 5.5 to 6.3. Whether that pays is the bracket's answer.
- **Several agents at once is the cheapest multiplier nobody measured:** two slots decode two tokens in the 77.9 and
  78.4 ms a 2-token pass takes, est. +31 % aggregate with no new code.
- **The order:** section 8. On each rig the first row is a map (the step's parts, the depth curve), then the arms
  that need no code. Code starts with a read of what already exists.

## 1. The first review, checked

| Claim of the first review | Check | Verdict |
|---|---|---|
| The release check is shallow: 5.5 turns, ~2,260 generated tokens, under 10k of context | [report_table.md](../../data/2026-11-05_release_flashnext/report_table.md), `agent_t1.tsv` | stands |
| A served step takes 150.4, 151.4 and 154.6 ms against a 4-token pass of 100.2, 101.6, 105.5 and 106.3 ms | `agent_t1.tsv` (tokens a step × ms a token), `tk_alt.tsv` | stands |
| The gap may be the clock: the keep-alive gave llama-bench 23.0 to 30.4 ms on T(4) and the served step 13.2 ms | the release run's per-card log during a served decode: all nine cards at 1,860 to 1,911 MHz in every one of 24 samples a card over four minutes, 33.7 to 41.8 W | **not the clock.** The cards hold boost while serving. The gap is the draft's four passes, their crossings, the server's work and the extra output rows (section 5) |
| Decode slows from 54.9 ms a token at ~1k to 94.3 and 99.0 at ~32k | [card hops](../../data/2026-11-04_card_hops/README.md): 18.2 and 18.2 tok/s at 1,382 tokens, 15.8 and 15.6 at 7,183, 10.1 and 10.6 at 32,674, no draft, `-ladder3` | stands. The slope is est. 1.26 and 1.41 ms per 1,000 cells; Ornith's dense read measured 1.17 (Q210), so the dense attention read explains it and Q230 is the right fix |
| Prompts are ~30 % of a check task, mostly fixed cost | the prefill note's fit, the release log's bins | stands; the causes are now known elsewhere (section 3.1) |
| The shipped file has the worst agent KLD of the one-rig files; IQ2_XS is "a better file at the same line" | [quality data](../../data/2026-11-03_flash_next_quality/README.md); [variants note](flash_next_variants.md) | KLD stands. **IQ2_XS is not a free upgrade:** it looped in thinking on 7 of 10 seeds in one maker's test and in a user report where Q2_0 stayed stable ([thread](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/45)). Its loop rate decides, not its KLD |
| Lever 1: reasoning effort `medium` or a budget, est. −20 to −50 % of a real task | the [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); the variants note, §3 | **corrected.** The card: lower effort "does not always reduce overall task completion time" (more failures and retries). The GSQ-RCO maintainer: medium effort "has degraded significantly" on these quants. Measure it; do not plan on it |
| Upstream's kv-cells scan and compact-mask fixes "land on newer bases" | [#28128](https://github.com/ggml-org/llama.cpp/pull/28128), [#28130](https://github.com/ggml-org/llama.cpp/pull/28130) | **corrected.** Both were closed unmerged on 2026-09-01. Upstream fixed depth another way before the fork's base (issue #28734 closed 2026-10-07; the k-pool selection, #31243). The rebase does not fix depth on Pascal; Q230 does |
| The task-quality lane starts with Q2_0 against IQ2_XS, one seed of 82 tasks | the project's own power table ([task-level checks](task_level_quality_checks.md), §2): paired tasks resolve 5 points at ~340 tasks | **changed.** ISTA's own LiveCodeBench v6 puts that pair 2.29 points apart (83.43 against 81.14); one seed cannot see a difference of that size. Bracket first (section 6) |
| Two rigs wait until a one-rig file can no longer gain | [two-rig note](flash_next_two_rigs.md) | **changed.** Its speed estimate predates every v3 kernel (section 6.3) |

Not in the first review: the bench's settings (section 2), the multi-GPU fork and this week's upstream work (section
3), the exactness problems (section 4), the draft at depth, lookup drafts and several slots (section 5), and the
overlap with the sister repo's rows of the same day (section 6.1).

## 2. The real workload

### 2.1 What the bench runs

From the bench's published `results.json` (every local series): the pi agent, `window_tokens` 262,144, `compaction`
false, the full history kept, `response_tokens` 8,192 a reply, `max_tokens` 65,536 generated per attempt, effort
`xhigh`, temperature 1.0, top-k 20, top-p 0.95. pi sends each reply's reasoning back, as the model card asks
(`preserve_thinking` is on by default), so an attempt's context holds everything it generated plus every tool result.

| Flash-Next series (one run of 20 cases) | Passed | Failed at the budget | Failed below it | Generated tokens, all 20 attempts | Mean per attempt | Tokens per passed case | Calls per passed case |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISTA GSQ-RCO-Coder IQ1_M | 10 | 7 | 3 | 945,855 | 47,293 | 39,421 | 78.9 |
| Agention Gyro-L TQ3_0 | 11 | 5 | 4 | 811,146 | 40,557 | 32,639 | 61.9 |

- **Context:** a passed case ends with at least its 32,639 or 39,421 generated tokens in context, plus the tool
  results of 61.9 or 78.9 calls. A budget-cut attempt holds 65,536 generated tokens plus its results. The preset's
  65,536-token context cannot hold these attempts; the bench ran them in 262,144. Peak prompt tokens per attempt are
  not in the published file (a number the bench's raw evidence has).
- **Throughput for scale:** the Gyro-L series ran its 20 cases in 1 h 49 min on the bench's five RTX 3090s
  (811,146 tokens, est. 124 tok/s over the whole run, prompts included). A series takes an afternoon there.
- **These files are not Swift's.** Swift 1.5 writes fewer thinking tokens (its card: 63.4 % fewer at `xhigh`), so the
  shape holds and the sizes are upper bounds for Swift.

### 2.2 Minutes per solved case on the rig (est.)

No rig measurement exists beyond 12k of context with the draft or beyond 32k without it, so this is a model. It
exists to rank levers, and each input names the row that replaces it.

| Input | Value | Basis | Replaced by |
|---|---|---|---|
| A served step at 2.8–10k | 154.6 ms for 3.15 tokens | `t1swift` | — |
| What depth adds to a step | est. 1.75 ms per 1,000 cells: 1.3 for the verify, which reads the cache once for its four tokens, and 0.44 for the draft's own attention layer over four passes | the no-draft slope (est. 1.26 and 1.41); the draft's layer is one of thirteen dense reads | the deep lane with the draft (R39) |
| Acceptance at depth | unchanged | none: DSpark's card reports 4.10 accepted below 32K and 3.46 above; EngramHalo's MTP line fell from 31.7 tok/s at ~13K to 15.0 at ~26K, under its no-draft 16.8 | R39's draft arm |
| A prompt of N new tokens | 2.4 s + 14 ms × N up to ~460 tokens | [prefill note](flash_next_prefill.md) §1.2 (v3's base; the release's short prompts take 2.42 to 2.90 s) | the prompt lane (R43) |
| New tokens a call | est. 400 | the check's later turns are 28–351 tokens, its file reads 598–3,312 | the bench's raw evidence |
| Mean depth | est. 35k in a passed case, 50k in a budget-cut one | tokens generated so far plus results | the bench's raw evidence |

- **A passed case:** est. 32,639 × 68.5 ms (a 216 ms step at 35k) = 37.3 min of decoding, plus est. 61.9 × 8.0 s =
  8.3 min of prompts: **est. 46 min.**
- **A budget-cut attempt:** est. 65,536 × 76.9 ms (a 242 ms step at 50k) = 83.9 min, plus est. 110 calls × 8.0 s =
  14.7 min: **est. 99 min.**
- **Per solved case:** with the Coder's outcome mix (10 passed, 7 at the budget, 3 below it at est. 25 min),
  (10 × 46 + 7 × 99 + 3 × 25) / 10 = **est. 123 min**; with Gyro-L's mix (11, 5, 4), **est. 100 min**.
- **Where it goes:** in the Coder's mix, est. 693 of 1,228 minutes (56 %) are budget-cut attempts. Decoding is est.
  82 % of a passed case.

What follows from the model:
1. **Fewer budget hits beat any kernel.** A file or setting that turns two of seven budget failures into passes (the
   distance between the two series above) removes est. 24 % of the time per solved case (est. 123 → 94 min in the
   Coder's mix). That is file quality and tokens per solved case, the sister repo's standing measure.
2. **Depth sets the decode cost.** At 35k, est. 61 ms of a 216 ms step is the dense cache read. The sparse attention
   removes most of it (section 5.1).
3. **The fixed cost per call matters with 62 to 79 calls:** est. 2.5 to 3.2 min of pure fixed cost per passed case.
4. **A stopped loop is worth a budget-cut attempt:** est. 99 min each. The loop census (Q1138) prices this.

## 3. What others already built

### 3.1 A fork for starved links: prompt overlap and checkpoints without a drain

[lukolszewski/llama.cpp-multigpu](https://github.com/lukolszewski/llama.cpp-multigpu) serves Flash-Next on six RTX
3090s on x8, x4 and x2 links, four of them behind USB4 docks: the closest published topology to nine x1 cards. It is
47 commits on upstream `df03399b8`, each with its switch, listed in `docs/multigpu/patches.md`. Its numbers (UD-Q4_K_XL,
five 262k slots, q8_0 cache, upstream against patched, one session):

| | 5k | 50k | 250k |
|---|---|---|---|
| Prompt reading, tok/s | 715 → 1,249 | 568 → 2,127 | 263 → 2,111 |
| Generation, tok/s | 38.5 → 45.9 | 25.7 → 41.9 | 10.2 → 33.7 |

What its commits say they fix, against what this project measured:

| The fork's finding (commit) | This project's matching observation | Row |
|---|---|---|
| Prompt micro-batches did not overlap across cards for three reasons: pipeline parallelism turned off by any tensor override; compute buffers reallocated as the context grew, each reallocation synchronizing every backend; a host-blocking sort and a full launch queue. "About one card busy on average", 859 → 2,500 tok/s with 2.9 cards busy (`c8832a5f0`) | "The cards are idle 60–75 % of the time in a long prompt (est.), cause unknown"; the period grows from 0.4–1.0 s to 2.0–3.3 s as a call grows ([prefill note](flash_next_prefill.md), items 2 and 6) | Q157, Q510, Q284; new Q1156 (the timeline), Q1157 (the read) |
| Checkpoints called `synchronize()` twice per prompt and once per restore, 100–800 ms each with a full pipeline; now taken by a worker behind a per-backend fence, the copy staged through pinned memory (`fcb21fe5f`) | the 112.6 MiB save drains the nine cards, 0.67 s; every prompt over 516 tokens ends with a lone 512-token pass (prefill note §1.3) | Q770 |
| Scheduler inputs staged through rotating pinned buffers and copied asynchronously on the compute stream; a hop's source stream waits for the destination's previous split (`1ca80a510`, `850db8aec`) | "synchronous input copies queued behind compute" is one of the prefill note's suspects | Q759 |
| The sampler waits only for its own output batch instead of a full synchronize per token (`00b3703f0`, `f8c8a9ce0`) | est. 7–10 ms of host work and waits a token, never split ([floor note](flash_next_floor_and_engine.md) §2.4) | Q869 |
| Several sequences decode as overlapping per-sequence batches (`4660a295e`, `fb8897947`, `53fcc9f87`): five sessions at 250k, 2.3 → 27.3 tok/s each | never measured for Flash-Next (section 5.4) | new Q1164 |
| The MTP draft "did not help, slower in most cases", left unsupported; n-gram lookup "about 2.5×" on a refactoring task | the rig's draft is +6 % against plain decoding at depth 0 by two tools; lookups parked (section 5.3) | new Q1153, Q1161 |

What does not carry over: its attention path is the tensor-core sparse kernel (Pascal needs Q230's), and only sm_86
was ever run. One of its three prompt causes is already gone from the fork's base: the release tree's top-k takes a
stream-ordered radix select for many rows and a bitonic sort up to 1,024 columns, so the host-blocking segmented sort
is left only for a few rows of wide inputs outside a graph capture (`top-k.cu`, `argsort.cu` at `1551f5fe1`). The
other two have never been tested here. A per-card timeline of one long prompt settles which apply (Q1156).

### 3.2 Upstream since the fork's base (b12076 plus 16; upstream is at b12233)

| PR | State on 2026-11-09 | What it does | Here |
|---|---|---|---|
| [#31927](https://github.com/ggml-org/llama.cpp/pull/31927) | merged 2026-11-09 | the MTP draft reads the target's hidden rows from a device buffer; prompt reading +6 to +9 % on Flash-Next (RTX PRO 6000), decode within 0.5 % | this is Q970 (C426). Take it with `-ladder6`; the draft and the last layers share card 8, so no peer copy is needed for that leg |
| [#31669](https://github.com/ggml-org/llama.cpp/pull/31669) | open, WIP | replay instead of rollback snapshots for the Gated DeltaNet state: 598.5 → 163.2 MiB of recurrent memory a sequence at three drafted tokens (Qwen3.8-27B); decode on par | the main model keeps 12.3 instead of 3 MiB of state per recurrent layer with the draft loaded: est. 335 MiB back across the cards (36 × 9.3). Room for `-ub`, context or a larger file (Q1159) |
| fix `824170e` for [#31656](https://github.com/ggml-org/llama.cpp/issues/31656) | on #31669's branch | the slot finder moves `rs_max` with the sequence when it swaps cells; without it, aborts at `-np 4` and, by the author's CPU test, logits wrong by up to ~12 | section 4 |
| [#31913](https://github.com/ggml-org/llama.cpp/pull/31913) | open, one approval | the recurrent state is reset by a fill, not a multiply by zero, so a NaN cannot stay | section 4 |
| [#31660](https://github.com/ggml-org/llama.cpp/pull/31660) | merged 2026-11-05 | the MTP conv snapshots of every recurrent model in one `set_rows` | fewer launches per verify; with `-ladder6` |
| [#31419](https://github.com/ggml-org/llama.cpp/pull/31419), [#31185](https://github.com/ggml-org/llama.cpp/pull/31185) | merged 2026-11-09 | inputs no node reads are not uploaded; the n-gram table from its own file | as the first review says |
| [#31959](https://github.com/ggml-org/llama.cpp/pull/31959) | open | BF16 matrices of few rows (the 320 × 10,240 hyper-connection projections, the indexer's) on a one-block-per-row kernel at 2 to 8 tokens: up to 3.61× at two tokens on an RTX PRO 6000, 1.00 on low-SM cards by its gate | the fork has its own Pascal BF16 kernel; one bench line against it at k = 2 to 5 says whether the idea adds anything on five SMs (Q1167) |
| [#31941](https://github.com/ggml-org/llama.cpp/pull/31941) | open, draft | DSpine block drafts (Tencent): accepted length 6.08 against DFlash2's 4.79 on GSM8K for Qwen3.8-27B, speed equal (29.5 against 29.6 tok/s) because its step costs more | no Flash-Next weights exist; nothing to do |
| [#29738](https://github.com/ggml-org/llama.cpp/pull/29738), [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | open | slot save and restore keyed by session; prompt prefix cells shared between slots | the 10–15k-token prefixes of real agents (Q774, Q1135) |

### 3.3 The model's makers and other engines

- **The paper** ([arXiv 2608.30320](https://arxiv.org/abs/2608.30320)): the MTP module's attention layers use QSA too,
  and multi-step MTP "reuses top-k indices across prediction steps". SGLang's version saves the selection of the last
  accepted row and reuses it for every draft step, and reports the accepted length unchanged (3.3 with three draft
  tokens, [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)). On Pascal each of the draft's four passes
  reads its layer's whole cache; the same reuse plus Q230's kernel on the draft's context removes that term (Q1153).
- **The residual branches in FP8:** the paper says FP8 storage "halves the bytes moved for the residual state
  relative to BF16, with almost no loss in quality". The rig's Q8_0 hop cast is that idea on the links, already in
  the preset.
- **EngramHalo.cpp** (a Strix Halo fork): `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min
  0.75`, a Q8_0 draft. Decode with the draft: 29.3 tok/s at ~185 tokens, 38.5 at ~3.4K, 31.7 at ~13K, 15.0 at ~26K;
  without it 20.9, 20.5, 18.4, 16.8. At ~1K and ~6.6K the pairs are 28.2 against 21.2 and 23.5 against 19.4. The
  draft wins by 21 to 88 % up to 13K and loses at 26K.
- **Community evidence on files has not moved:** no independent agentic comparison of the GSQ-RCO Q2_0, IQ2_XS and
  IQ3_XXS files exists. One 30-task check (math, code, QA; one sample; a 2,048-token budget) had the GSQ-RCO Q2_0
  hit its token cap on all 11 of its misses ([thread](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/71)):
  the same budget effect as on the bench.

## 4. Exactness of the speculative path

Miner Llama promises the model's own answers. Three places where a draft line may not give them, two of them in
the release tree as upstream bugs and one that the preset avoids only by its draft length:

| Problem | Evidence | In `1551f5fe1`? | What settles it |
|---|---|---|---|
| A NaN in the recurrent state stays for good: the reset multiplies the state by zero | [#31913](https://github.com/ggml-org/llama.cpp/pull/31913), found on Flash-Next; the symptom reported is logits that come out as "//////…" | yes: `src/llama-graph.cpp` resets with `ggml_scale_inplace(ctx0, state_zero, 0)` | take the fix; the loop census counts replies that end in one repeated token, which separates this from the model's own loops (Q1158, Q1138) |
| The slot finder swaps two cells' position, source and sequence but not `rs_max`, so the next rollback of that sequence reads the wrong snapshot count | [#31656](https://github.com/ggml-org/llama.cpp/issues/31656): aborts within seconds at `-np 4` with `draft-mtp`; wrong logits on CPU by the fix's author | yes: `llama_memory_recurrent::find_slot` swaps `pos`, `src`, `seq_id` only. By reading, a swap needs two cells, so the preset's single slot should not reach it | take fix `824170e`; a CPU run of the preset's shape (one slot, the draft, a prefix restore) with the check that the swap never fires (Q1158). It is a precondition for any line with two or more slots |
| Rejection sampling reuses its uniforms after a checkpoint replay: total variation 0.095 on a JSON tool call under a grammar, p = 6·10⁻⁹ | this project's [upstream fix](../../upstream_fixes.md) and [data](../../data/2026-11-04_block_verify/README.md) | the fix is in the build behind `LLAMA_SPEC_KEEP_DRAWS`, off by default. **The preset is not affected:** its three drafts stay within `n_rs_seq`, so no step replays a checkpoint (Q808; the release session restores state only at task starts) | on for every line whose draft can outgrow the snapshots: a draft of four or more tokens, lookup drafts, DFlash2 blocks. One paired replay arm shows its cost (it changes which uniforms are drawn, not the work, so none is expected); then it is the default (Q1160) |

The token-forcing check (Q1014, C525) is the instrument that would show any of the three on a rig line: each
recorded reply replayed through the same file with and without the draft, sampled arms tested against the sampler's
support. Until it has run, a quality number from a stock build without the draft is evidence about the file, not
about the rig's line with the draft.

## 5. Speed: the parts nobody has measured

### 5.1 Depth, with the draft and beyond 32k

- **The measured slope is the dense read.** Twelve attention layers read 2 KiB a cell each: 805 MB at 32k, at the
  ~21 GB/s these reads reach on a P106. Q230's kernel reads the 2,051 selected cells instead and takes up to 8 query
  tokens, so a verify batch is covered. Its mode 1 switches a 4-token verify to the sparse path from 12,306 cells
  ([QSA data](../../data/2026-11-04_qsa_sparse_attn/README.md)).
- **The dense kernel reads the cache once for all four verify tokens; the sparse one reads four lists.** So on the
  rig, unlike on stock engines, the draft should gain value with depth while attention is dense, and fall back to
  its shallow value once attention is sparse. Whether acceptance holds at depth is the unknown that decides both.
- **Never measured:** the release line beyond 12k with the draft; anything beyond 32k; the draft's own passes at
  depth; the host's k-pool inputs at depth (176 KiB a card at 32k, est. 1 to 1.5 ms a token; `LLAMA_QSA_INPUT_STATS`
  logs it); a 131,072-token line, which the fit probe projects to fit with the draft at `-ub` 64, 128 and 256.
- **So the deep lane (Q1143) takes two more arms** (Q1153): the draft off against on at every depth with acceptance
  and tokens a step logged per depth, and depths 65,536 and 120,000 on a 131,072-token line (Q1154).

### 5.2 The step

The known parts of the ~50 ms between a served step and llama-bench's 4-token pass, newest measurement of each:
draft GPU time 24.8 ms with the full head (`-ladder3`; the preset's shortlist head was never timed), four crossings
and turnarounds est. 4.4 ms, server work after the last draft pass 7.03 ms, three extra output rows est. 5 to 7 ms,
attention at the session's depth a few ms. They sum to est. 41 to 48 ms: no part is missing, and none is measured on
the release build. Two rows measure them (R41, S37) and should run before any arm, because every estimate in the
queue is priced on an older token. After `-ladder6`, #31927 has already removed the hidden rows' host round trip.

A draft chained past its window on its own card while the first cards verify (Q1109) has an offline gate that needs
no machine time on a rig: the share of steps with all three drafts accepted and the bonus token equal to a fourth
draft, from existing dumps. From the release's acceptance (2.15 of 3 drafts accepted on average) that share is est.
40 to 50 %, well above the row's 10 % parking line.

### 5.3 Lookup drafts beside the MTP draft

`--spec-type` takes a comma-separated list in the release build, and `ngram-mod` is in it. Lookup drafts cost no
draft pass and can run 8 or more tokens when the model repeats context: an edit that quotes a file, a tool call
that repeats a path. The copy-span study found spans too short on the bug-fix check (median 11 tokens; 0 to 6 %
gain) and parked the idea (Q617), and the lookup guessers were parked for the checkpoint each round saved over the
x1 link (Q562). The first reason is weaker on real tasks, which edit several files; nobody has counted spans there.
The second still holds for a lookup longer than `n_rs_seq`: a partly accepted draft then restores the state, 0.63 s
over the link. Three things remove it: the checkpoint kept on the card (`--ctx-checkpoints-vram`, built, never run on
a rig), lookups capped at `n_rs_seq`, or replay instead of snapshots (#31669). So the order is: count copy spans in
recorded real attempts (CPU), then one replay arm with on-card checkpoints and kept draws (Q1161).

### 5.4 Several slots

Miner Llama's multi-token kernels make a pass of k tokens cheap whether the tokens are one sequence's drafts or k
sequences' next tokens. On the release build a 2-token pass takes 77.9 and 78.4 ms and a 4-token pass 101.6 and
106.3 ms (two loads), against 51.1 and 52.3 ms a token decoding alone. Two agents decoding together would get est.
25.6 tok/s between them (+31 % over one at 19.6) and four est. 37.6 and 39.4 (the two loads), before server
overheads, with no draft and no new code: `-np 2` or `-np 4` and a context that holds every slot. It does nothing
for one agent's wait. It raises what a rig finishes in a day when there are several tasks, a bench series or
training sessions to write. The fork of section 3.1 goes further with overlapping per-sequence batches. Needs fix
`824170e` first (section 4). Row Q1164.

## 6. Quality

### 6.1 One lane, two repos

On the same day the sister repo registered the bench runs that the first review's lane duplicates:

| Question | Sister repo (the bench, 20 cases, three seeds) | This repo's row |
|---|---|---|
| The rig's file | E1: Swift Q2_0 and Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (`pmix-q2-fast`) | Q1146 (82 polyglot tasks) |
| The high-precision end | E13: Swift IQ3_S | — |
| Effort | E16 | Q1147 |
| Budget-cut attempts at twice the budget | E15 | — |
| Loops and failure classes | E20, by Q1138's rules | Q1138, Q1152 |

One instrument should answer each question once. The bench is the owner's measure and its runs are planned, so:
- **The bracket is E1 and E13.** By the sister repo's own estimate, two three-seed series separate 2.5 to 3.0 cases.
- **Q1146 becomes the second opinion on the same two files** (Swift Q2_0 and IQ3_S on the polyglot tasks), run only
  if the bench's bracket lands inside its resolution, since 82 tasks a seed resolve less than they seem to: by the
  project's power table a 5-point difference needs ~340 paired tasks.
- **Near neighbours (IQ2_XS, IQ3_XXS, the v3.1 mix, the N2P files) are scored only after the bracket separates.** If Q2_0 and IQ3_S
  cannot be told apart on tasks, no file between them can, and the file is chosen on speed, tokens per solved case
  and loop rate.
- **Q1147 points at E16.** Note for that row: the card lists `xhigh`, `medium` and `low`; there is no `high`.

### 6.2 Quality where real tasks live

Every Flash-Next KLD is on texts of at most 8,192 tokens against Swift's Q8_0. Real attempts spend most of their
tokens beyond 16k, where each query sees 2,051 of the cells and the indexer's own tensors (BF16 or retyped) pick
them. Whether a 2-bit file's distance from the reference grows with depth has never been measured. KLD by position
bucket (0–8k, 8–16k, 16–32k, 32–64k) on long agent sessions answers it for Swift Q2_0, IQ2_XS, IQ3_XXS and IQ3_S
(Q1162). A file whose KLD climbs with depth would explain budget hits better than its 8k KLD does.

### 6.3 Two rigs, re-priced (est.)

The two-rig note priced Swift IQ3_S on 18 cards at 5.5 to 6.3 tok/s plain from v3's base, when a token took 128 to
136 ms. On the release build a token takes 51.1 ms. Re-priced: the same dense work plus est. +5 ms for IQ3_S's
larger dense tensors, est. +16 to +21 ms for its experts (2.36 G weights at est. 10 to 12 ps on the v3 IQ kernels
against Q2_0's ~3.3), est. +9.5 to +17 ms for the second rig over RPC (the note's figure, unmeasured): **est. 81 to
94 ms a token, 10.7 to 12.3 tok/s plain**, before any draft. IQ3_S is the file its community calls level with the
full model (SWE-bench Verified 82.0 against 82.8 by its maker).

The decision is the bracket's. If IQ3_S passes clearly more cases and writes fewer tokens per pass than Q2_0, a
two-rig line at half the tok/s can still solve a case in fewer minutes, since budget-cut attempts are est. 56 % of
the time. If the bracket shows nothing, two rigs are two replicas (two agents), not one better model. The same
two-rig runtime is what a larger model than Flash-Next would need, which the sister repo keeps parked (its Q502).

## 7. The levers, re-priced

Est. effect on minutes per solved real case (section 2.2), with what the estimate rests on and the first
measurement. Ordered by expected value.

| # | Lever | Est. effect | Rests on | Works at all | First measurement |
|---|---|---|---|---|---|
| 1 | **Fewer budget-cut attempts:** the file (bracket), tokens per pass, loops stopped | est. −24 % for two of seven converted | the two bench series; est. 99 min an attempt | open: the bracket decides | E1 and E13; Q1138 |
| 2 | **Sparse attention at depth, and a line that holds a real attempt** | est. −25 % of decode at 35k (a 216 ms step → est. 162) | the measured slope; Q230's CPU checks | high | C441, C313, R39 with the draft arm, then Q1154 |
| 3 | **Prompt structure:** no drain for checkpoints, overlap of micro-batches | est. −50 to −60 % of prompt time; −9 % of a passed case | the fit; the fork's 859 → 2,500 | high | R43 (no code), Q1156, then Q1157's ports |
| 4 | **The step outside the verify pass** | est. −15 to −25 ms of 154.6 | `-ladder3`'s parts; #31927 | high | R41, S37, after `-ladder6` |
| 5 | **Probabilistic verification** | est. −4 to −5 % of decode | the V100 fork's same model and draft | high | R38 |
| 6 | **Several slots** | est. +31 % aggregate at two, 1.9× and 2.0× at four (two loads); nothing for one agent | T(2), T(4) on the release build | high for plain decode; with the draft open | Q1164 |
| 7 | **Lookup drafts beside MTP** | 0 to −10 % of decode on edit-heavy work | other forks; spans unmeasured on real tasks | open | Q1161, C234 on real traces |
| 8 | **Draft chained past its window** | est. −10 % of a step if the gate's share is ~45 % | Q1109's note; acceptance | medium | the offline gate |
| 9 | **Two rigs with IQ3_S** | fewer budget hits at est. 0.5 to 0.6× the tok/s | the bracket; est. 81–94 ms a token | open | the bracket, then C260's plain T(k) over RPC |
| 10 | **Verify kernels at k = 4 to 5** | est. −4 to −7 % | the `-moeskip` profile | medium | R41 |
| 11 | **Reasoning effort, `preserve_thinking` off, compaction** | large either way | the card's caution; no data | open | E16; Q1166 |

Moved down or parked, with the reason: pipelined one-token speculation (a 4-token verify costs 2.0 plain tokens
now); a persistent kernel (≤ 0.77 ms beyond the clock); lossy acceptance (changes answers); IQ2_XS as the preset's
file before its loop rate is known; `medium` effort as a preset variant before E16.

## 8. The order of work

Three machines in parallel, one session per rig, one row per session. A row marked "sister" is the sister repo's and
runs wherever the owner places it.

| Lane | Order | Row | Answers | Est. time |
|---|---|---|---|---|
| Desk and code | 1 | Q1157: the multi-GPU fork's commits read against `1551f5fe1` | which of its prompt, checkpoint, staging and multi-slot commits apply; the port list | a day of reading |
| | 2 | C657: the deep lane's driver, with the draft arm and a second context size | — | small |
| | 3 | Q1150 (`-ladder6`) with #31927, #31660, #31419, #31185, fix `824170e`, #31913's fill | the base for all later code | a merge and its checks |
| Rig A (the step) | 1 | R41: per-part profile at k = 1 and k = 4 | every ms estimate re-based | 45 min |
| | 2 | S37 on the release build | the step outside the verify, part by part | 1.5 h |
| | 3 | R38 with a kept-draws arm (Q1160); the lookup arm once spans are counted (Q1161) | +5 points of acceptance; what kept draws cost | 2 h |
| | 4 | R43: the prompt lane, switches only | the drains' cost; tail checkpoints, on-card checkpoints, the taper | 2 h |
| | 5 | Q1156: one long prompt's per-card timeline | why cards idle in prompts | 30 min |
| Rig B (depth) | 1 | C441, then C313 | the sparse kernel: correct, then T(k) at 8k and 32k | 1.2 h |
| | 2 | R39 with Q1153's arms | the release line at 0 to 49k: draft off and on, sparse off and on, acceptance by depth | 2.5 h |
| | 3 | Q1154: the 131,072-token line | fit, speed at 65k and 120k | 1.5 h |
| | 4 | Q1164: two and four slots | aggregate tok/s, plain | 1 h |
| Quant machine | 1 | Q1138 with the repeated-token class (Q1158) | loops, and whether any are the NaN | 1 h, CPU |
| | 2 | sister: E1 and E13 (the bracket) | whether file quality moves real tasks | est. 12 h of series |
| | 3 | Q1109's offline gate; Q806; copy spans in recorded real attempts (Q787) | what a longer, chained or lookup draft can pay | 1 to 3 h |
| | 4 | Q1162: KLD by depth bucket | quality where tasks live | 3 h |
| | 5 | Q710's reshard, C326's code text, Q552's KLD, the two N2P files' checks and KLD (Q928) | files ready for whatever the bracket says; whether a Pascal-format file beats IQ2_XS's KLD without its loops | 1 + 2.5 + 0.75 + 1.5 h |

**Decision rules**, so that each result moves the queue without a new review:
- R41 or S37 puts a part of the step above 5 ms → that part's row is next in code (Q1148, Q960, Q1104, Q969).
- R38: the kept-draws arm within the spread of the preset → on by default for every draft line. Probabilistic
  verification or the lookup arm a gain on the paired replay → one agent-check pair, then the preset.
- C441 passes and C313 or R39 shows `GGML_CUDA_FATTN_SPARSE_PASCAL=1` a gain from 8k → on by default for
  `qwen4exp`, and the 131,072-token line becomes a second preset.
- R39: the draft a loss beyond some depth → a depth gate for the draft (Q1153) before any other draft work.
- R43: `--ctx-checkpoints 0` against the preset is the drains' whole cost. Above 15 % of prompt time → the
  drain-free checkpoint (Q770, by Q1157's port) is the first engine code; any switch arm that gains goes into the
  preset.
- Q1156: under half the cards busy in a long prompt → Q1157's overlap commits before the taper's agent arm.
- The bracket: under 2.5 cases between Swift Q2_0 and IQ3_S over three seeds → the file is chosen on speed, tokens
  per solved case and loop rate, and Q710, Q227, Q889 and two rigs stay down. 3 cases or more → IQ3_XXS on the rig
  (Q227) and C260's plain T(k) over RPC move to the top of their lanes.
- Q1138: any reply ending in one repeated token → fix #31913 before every other row.

**What the release reports next:** minutes per solved case on recorded real attempts replayed on the rig (Q1144's
lane, built from the bench's saved attempts where its evidence allows, since replays need no new tasks), the deep
lane's table, and the bug-fix check as before.

## 9. New registry rows

In [quants/experiments.md](../../../quants/experiments.md):

| Row | Idea | Status |
|---|---|---|
| Q1153 | The MTP draft by depth: off against on with acceptance per depth; the draft's own attention through Q230's kernel and with the selection reused across its passes; a depth gate if it loses | planned |
| Q1154 | A 131,072-token line: fit on the release build, speed at 65k and 120k, as a second preset | planned |
| Q1155 | The prompt lane: the release session's prompts replayed with `--ctx-checkpoints 0`, the tail-checkpoint rule, on-card checkpoints and the taper, one switch an arm | planned |
| Q1156 | One long prompt's per-card timeline and host waits on the release line | planned |
| Q1157 | The multi-GPU fork's 47 commits read against the release tree; the port list | planned |
| Q1158 | Recurrent-state exactness: #31913's fill and fix `824170e` carried; the repeated-token class in the loop census; the swap check on CPU | planned |
| Q1159 | Replay instead of rollback snapshots (#31669) when it merges: est. 335 MiB back | planned |
| Q1160 | `LLAMA_SPEC_KEEP_DRAWS` on by default before any draft that can outgrow its snapshots; one paired arm for its cost | planned |
| Q1161 | Lookup drafts beside the MTP draft (`draft-mtp,ngram-mod`) | planned |
| Q1162 | KLD by depth bucket on long agent sessions | planned |
| Q1163 | Minutes per solved case as the headline, and section 2.2's model with each input's row | planned |
| Q1164 | Two and four slots on the release line; then overlapping per-sequence batches | planned |
| Q1165 | The bracket before the neighbours; one lane with the sister repo | planned |
| Q1166 | `preserve_thinking` off and compaction as depth and budget levers | planned |
| Q1167 | #31959's one-block-per-row BF16 path against the Pascal BF16 kernel at k = 2 to 5 | planned |
| Q1168 | Swift's own MTP head from its BF16 against the draft in use | planned |
| Q1169 | Two rigs re-priced on v3.0's token; the first measurement is plain T(k) over RPC | planned |
| Q1170 | A checkpoint taken in the idle gap after a reply instead of inside the next prompt | planned |

## Sources

Project data: [release](../../data/2026-11-05_release_flashnext/README.md),
[keep-alive](../../data/2026-11-05_keepalive/README.md), [card hops](../../data/2026-11-04_card_hops/README.md),
[sparse attention](../../data/2026-11-04_qsa_sparse_attn/README.md),
[quality](../../data/2026-11-03_flash_next_quality/README.md), [fit probe](../../data/2026-11-03_flash_next_fit/README.md),
[block verification](../../data/2026-11-04_block_verify/README.md); notes: [first review](flash_next_review_2026-11.md),
[prefill](flash_next_prefill.md), [floor and engine](flash_next_floor_and_engine.md), [variants](flash_next_variants.md),
[news](flash_next_news_2026-11.md), [two rigs](flash_next_two_rigs.md), [task-level checks](task_level_quality_checks.md).
The release tree `1551f5fe1` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama): `ggml/src/ggml-cuda/top-k.cu`,
`argsort.cu`, `src/llama-graph.cpp`, `src/llama-memory-recurrent.cpp`, `common/arg.cpp`.

Outside, read 2026-11-09:
- the bench: [results.json](https://cinderella-man.github.io/elixir-swe-bench/results.json), and the sister repo's
  [E14](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs/tree/main/docs/data/2026-11-09_bench_runs_read);
- the multi-GPU fork: [README](https://github.com/lukolszewski/llama.cpp-multigpu),
  [patches](https://github.com/lukolszewski/llama.cpp-multigpu/blob/multigpu/docs/multigpu/patches.md), commits
  `c8832a5f0`, `fcb21fe5f`, `850db8aec`, `b22bc5b68`, [discussion #30071](https://github.com/ggml-org/llama.cpp/discussions/30071);
- llama.cpp: [releases](https://github.com/ggml-org/llama.cpp/releases) (b12233),
  [issue #28734](https://github.com/ggml-org/llama.cpp/issues/28734), the pull requests and issue of sections 3.2 and 4,
  [unslothai/llama.cpp#165](https://github.com/unslothai/llama.cpp/pull/165);
- the model: [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [paper](https://arxiv.org/html/2608.30320v1),
  [SGLang](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/),
  [Swift's GSQ-RCO card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  [ISTA's discussions](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions);
- [EngramHalo.cpp's write-up](https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/).

Not verified: web numbers came through summarising fetches of the pages; the fork's commits were read as their
messages and diffs' summaries, not built; section 2.2 and every "est." are models.
