# Flash-Next after v3.0: a review of the evidence, the gaps in the thinking, and the order of work (2026-11-09)

The owner's next goal is "pushing Qwen Flash Next even further". Before adding ideas, I read the project as someone
taking it over would: the release data, the registry, the queues, the research notes and the sister repo's first
bench reads. I checked the main claims against their data folders, and read what changed upstream and in the
literature since v3.0. This note says where a task's time goes today, what the project measures well and what it does
not, which beliefs rest on no data, and the order of work that gives Flash-Next the best chance of a large gain in
speed or quality.

**The order of work in section 5 is replaced by [the second review](flash_next_review2_2026-11.md), section 8.**
That note checks this one claim by claim (its section 1) and changes three things: reasoning effort is not the
first lever (section 3.6 and lever 1 here), the upstream depth fixes named in section 3.8 were closed unmerged, and
the first files to score on tasks are Swift Q2_0 and Swift IQ3_S, not Q2_0 and IQ2_XS. Sections 1 to 3 here stand
as the evidence.

Conventions:
- "est." marks my own arithmetic or inference; the basis is next to it. Every other number is measured and quoted
  from the linked folder, one run unless the folder lists repeats.
- "The release line" is [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env) on build
  `1551f5fe1`, Swift-1.5 GSQ-RCO Q2_0, nine P106-090 cards. Its agent-check arm is `t1swift`
  ([2026-11-05_release_flashnext](../../data/2026-11-05_release_flashnext/README.md)).
- Nothing ran for this note. It read files and web pages only.

## The short answer

- **The release check measures a task the user rarely runs.** Its 12 bug fixes take 5.5 turns, ~2,260 generated
  tokens and end below 10k tokens of context. Real repository tasks under the same agent took 61.9 and 78.9 model
  calls and 32,639 and 39,421 generated tokens per passed case, for two Flash-Next files (the sister repo's read of the
  Elixir SWE Bench). There, the tokens the model writes and the speed at 20–60k tokens decide the wait, and the check
  sees neither.
- **Speed falls hard with depth, and the fix is built but has never run on a GPU.** Without the draft, a token took
  54.9 ms at ~1k tokens of context, and 94.3 and 99.0 ms in two loads at ~32k. The sparse Pascal attention (Q230) is
  in the release build behind a switch.
- **Speculation pays little because a third of every step is outside the verify pass.** A served step took 150.4,
  151.4 and 154.6 ms in three arms for ~3.15 tokens; the 4-token verify alone took 100.2, 101.6, 105.5 and 106.3 ms in
  four loads. The ~45–50 ms between them (est.) has not been measured since `-ladder3`, and the profile that would
  split it (S37) is ready and never ran.
- **Prompt reading is a third of a task and mostly fixed cost:** ~7 s at the end of every prompt over 516 tokens and
  ~2.4 s for every short tool result. The two built fixes (the taper, checkpoints on the cards) never ran on a rig.
- **The quality half of the goal has no task-level measurement.** The shipped Swift Q2_0 has the worst agent-text KLD
  of the files that fit one rig (0.2851; IQ2_XS 0.2248 fits the same line; IQ3_XXS 0.1755). No Flash-Next file has a
  pass rate beyond a 12/12 check that cannot separate files.
- **Tokens per solved task were never a lever for Flash-Next,** although the model's default reasoning effort is its
  most verbose and v2 found shorter reasoning the largest wall-clock lever for Flint.
- **The speed instrument is noisier than the decisions it makes:** two runs of one line differ by 3.7 % of task time,
  and the "negligible" changes summed to a loss. Engine changes should be decided on paired, fixed-text passes.
- **The order:** a week of measurements on what exists (task-quality lane, effort, loops, S37's step profile,
  probabilistic verification, sparse attention at depth, the prompt fixes, IQ2_XS on the rig), then code in the order
  those answers price it: prompt structure, the step outside the verify pass, the upstream rebase, verify kernels,
  longer drafts. Section 5.

## 1. What a task costs today, and what a real task costs

### 1.1 The release check

`t1swift` ran the 12 bug-fix tasks in 32.2 minutes, 66 turns in all
([agent_t1.tsv](../../data/2026-11-05_release_flashnext/agent_t1.tsv),
[report_table.md](../../data/2026-11-05_release_flashnext/report_table.md)):

| Per task (mean of 12) | Seconds | Share |
|---|---:|---:|
| Reading prompts (prefill) | 48 | 30 % |
| Generating (decode, with the MTP draft) | 111 | 69 % |
| Other (fixed, tools) | 3 | 2 % |

- About 2,260 tokens generated per task (est.: 111 s × 20.40 tok/s) over 5.5 turns.
- About 3,300 new prompt tokens per task at 69.4 tok/s; 87.6 % of the prompt was a cache hit.
- The contexts stay small: the agent's own prefix is ~1,549 tokens and the tasks end at 2.8–10k tokens (Q787).

### 1.2 A real coding task

The sister repo read the Elixir SWE Bench's published runs
([E14 data](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs/tree/main/docs/data/2026-11-09_bench_runs_read)):
repository-level tasks under the same agent (pi), a 65,536-token generation budget, one run of 20 cases.

| Flash-Next file (one run) | Passed | Failures that hit the budget | Tokens per passed case | Model calls per passed case |
|---|---:|---:|---:|---:|
| Agention Gyro-L TQ3_0 | 11 of 20 | 5 of 9 | 32,639 | 61.9 |
| ISTA GSQ-RCO-Coder IQ1_M | 10 of 20 | 7 of 10 | 39,421 | 78.9 |

A real task asks for about 14 times the generated tokens and 11 times the turns of a release-check task (est.:
32,639 against ~2,260 tokens, 61.9 against 5.5 calls). At the release line's speeds one passed case of that kind is
est. 26.7 minutes of decoding (32,639 / 20.40 tok/s) plus est. 7 minutes of prompt reading (61.9 calls × the 6.89 s
median first token), before any slowdown with depth. A case that fails at the budget costs est. 53.5 minutes of
decoding alone. These files are not Swift's (UkisAI reports 31–57 % fewer thinking tokens for its fine-tune), so the
numbers bound the shape of the problem, not Swift's exact cost.

**What this changes:** on real work, the number of tokens the model writes per solved task, and the speed at 20–60k
tokens of context, decide the user's wait. The release check measures neither.

## 2. Where a decode step's time goes

No per-part profile exists for a build with the release's switches (the Q8_0 hop cast and the keep-alive are known
only as whole-pass deltas). The newest pieces, with their builds:

| Piece of a served step | ms | Build and source |
|---|---:|---|
| Served step, release line | 150.4, 151.4, 154.6 | `1551f5fe1`, `1f7215948`, `f6c1c6a4a`; agent arms |
| llama-bench T(4), one logits row | 100.2, 101.6, 105.5, 106.3 | four alternated loads of `f6c1c6a4a` and `1551f5fe1`, [tk_alt.tsv](../../data/2026-11-05_release_flashnext/tk_alt.tsv) |
| Draft GPU per step (catch-up 3.63 + 3 × 7.06), full output head | 24.8 | `-ladder3`, C338 ([draft one card](../../data/2026-11-04_draft_one_card/README.md)) |
| Draft host crossings and turnarounds | ~4.4 (est.: 4 × ~1.1) | `-ladder3`, C338 |
| Server work between the draft's end and the next verify | 7.03 | `-ladder3`, C338 |
| The verify's three extra output rows | ~5–7 (est.) | v3's base, S21: 10.4 ms for 4 rows against 3.5 for 1 |
| Attention at the session's depth | a few (est.) | the long release task ran 160.2 ms a step against ~150 |

The draft's GPU time with the shortlist head, now in the preset, was never measured; `-draft1card` (S37, in the
release build behind its switches) never ran. On `-ladder3` the parts outside the verify pass summed to ~41 ms of a
179.2 ms step ([verify host](../../data/2026-11-04_verify_host/README.md)).

The verify pass itself, on the newest profile (`-moeskip` with the switch off, F32 hops, no keep-alive, real agent
text, GPU ms summed over the nine cards, [moe_skip](../../data/2026-11-04_moe_skip/README.md)):

| Part | k = 1 | k = 5 | Growth |
|---|---:|---:|---:|
| Attention and recurrent projections | 20.08 | 49.85 | +29.8 |
| Routed experts (Q2_0) | 8.94 | 30.55 | +21.6 |
| Hyper-connections (BF16) | 10.65 | 22.82 | +12.2 |
| Shared expert | 3.33 | 6.41 | +3.1 |
| Router (F32) | 2.88 | 6.06 | +3.2 |
| Gated DeltaNet state and conv | 3.52 | 5.45 | +1.9 |
| Other operations without weights | 6.75 | 8.89 | +2.1 |
| Attention (≤ 1.7k context) | 0.83 | 1.97 | +1.1 |
| **Cards' GPU time** | **57.75** | **133.41** | |
| Hops, host and gaps (unprofiled pass − GPU) | ~8.6 | ~39.9 | |

The projections grow most with k: they hold 33.5 % of a token's bytes and run at about a third of the card's
bandwidth ([floor](flash_next_floor_and_engine.md)). The bandwidth floor of one token is 26.4 ms with Q8_0 hops; the
release line's plain token is 51.1 ms.

**Depth** (no draft, `-ladder3`, pmix-q2-fast, Q8_0 hops, single sampled requests,
[card_hops](../../data/2026-11-04_card_hops/README.md)): 54.9 ms a token at ~1k, 63.3 and 64.1 at ~8k, 94.3 and 99.0 at
~32k. At 32k a token costs 1.72 and 1.80 times what it costs at 1k (est., per load).

**Prompts** (the release session's server log, 66 requests, 574.2 s for 39,863 new tokens):

| New tokens | Requests | Seconds each | Share of prompt time |
|---|---:|---|---:|
| 28–57 | 18 | 2.42–2.90 | 8 % |
| 83–480 | 24 | 3.24 (83 tokens) to 9.71 (480) | 25 % |
| 561–3,312 | 24 | 10.53 (577) to 28.75 (3,312) | 67 % |

Every prompt above 516 tokens that logs its progress ends with a 6.69–7.41 s tail after the last full call: a lone
512-token pass, a recurrent-state checkpoint, the 4-token end and the draft's catch-up, none of them split by a
measurement. The 24 long prompts pay est. ~168 s of tail, ~29 % of the session's prompt time. Each task's start also
restores a 112.61 MiB checkpoint in 622.0–722.5 ms.

## 3. Gaps in the thinking

Each gap below says what the project believes or does, what the data says, and what follows. They are ordered by how
much they change the plan.

### 3.1 The headline workload is shallow and short

- **Belief:** the 12-task bug-fix check is "realistic agentic speed".
- **Data:** its tasks end at 2.8–10k tokens of context after 5.5 turns and ~2,260 generated tokens (section 1). Real
  repository tasks under the same agent take 61.9 and 78.9 model calls and 32,639 and 39,421 tokens per passed case (E14),
  and real agents carry larger prefixes (OpenCode est. 10–15k tokens, [agent workload check](agent_workload_check.md)).
- **What hides there:**
  - Speed at depth. Flint Pascal Q5 GGUF on v3 decodes 50.79 tok/s at depth 0 and 34.60 at 32,768 (T2h, one load,
    [2026-11-06_flint_q5_depth](../../data/2026-11-06_flint_q5_depth/README.md)). Flash-Next's 12 attention layers
    read every cached cell through the dense Pascal kernel and mask most of them away; Q230 put that at est. −15 ms a
    token at 16k and −34 ms at 32k ([QSA](../../data/2026-11-04_qsa_sparse_attn/README.md)). The sparse Pascal kernel
    that removes it is in the release build behind `GGML_CUDA_FATTN_SPARSE_PASCAL` and has never run on a GPU.
  - The first request at each new depth. Flint's first of two identical requests ran 47.15 against 54.56 tok/s at
    depth 0 (Q1133). Every agent turn is such a first request.
  - Tokens per solved task. A runaway that hits the 12-minute cap costs as much as est. 4.5 mean tasks (161 s each),
    and at the budget of a real task a failure costs est. 53.5 minutes (section 1.2).
- **What follows:** keep the bug-fix check for continuity, and add a deep lane: fixed-depth decode and prompt passes at
  0, 8k, 16k, 32k and 48k on recorded agent text (T2h's method, whose loads repeat within 0.04–0.43 tok/s), and an
  agent lane with a realistic prefix and multi-file tasks. Report the headline at depth beside the bug-fix number.
  New rows: Q1143, Q1144.

### 3.2 Speculation pays little, and nobody owns the whole step

- **Belief:** the MTP draft is a large part of v3's speed.
- **Data:**
  - A served step of the release line took 150.4, 151.4 and 154.6 ms for 3.14 or 3.15 tokens (`s35ka`, `t1base`,
    `t1swift`: ms a token × tokens a step). llama-bench's T(4) on the same builds took 100.2, 101.6, 105.5 and
    106.3 ms in four loads. The ~45–50 ms between them (est.) has not been measured on any build after `-ladder3`
    (section 2).
  - The served line writes a token every 48.2 ms (`s35ka`); plain decoding on the same build takes 51.1 ms a token at
    depth 0 (tg128). The agent decodes at 2–12k, where plain decoding is slower, so the draft is worth more than that
    6 % says, but no no-draft agent arm exists after `-ladder3`, where the draft gave +12 % (14.3 against 16.0 tok/s).
  - The keep-alive took llama-bench's T(4) from 131.48 and 124.08 ms (two loads) to 101.17 and 101.11, but the
    served step only from 163.6 to 150.4 ms (`s35kaoff` against `s35ka`), so the part of the step outside the verify
    pass grew.
  - The model's makers built the MTP layer for this: the paper reports a mean accepted length of 4.07 under four-step
    speculation (arXiv 2608.30320, Table 4), and SGLang's three-token line accepts 3.3 a step on a B200.
- **What follows:** the step, not the verify pass, is the unit to optimize. First S37's profile of one served step on
  the release build, then the fixes it points at (Q1148, Q970, Q960, Q1104), and the exact gain that needs no code
  (Q731, probabilistic verification).

### 3.3 Prompt reading is a third of a task, and most of it is fixed cost

- **Data:** 48 of 161 s per task (section 1.1). A prompt of N ≤ 460 new tokens takes about 2.4 s + 14 ms × N, and
  every prompt above 516 tokens ends with a lone 512-token pass and two recurrent-state saves of 112.6 MiB (0.67 s
  each): ≈ 9 s whatever the prompt's length ([Flash-Next prefill](flash_next_prefill.md), §1, on the 2026-11-03 base).
  On real tasks with 60–80 calls, the fixed part alone is est. 2.5–3 minutes per task (62–79 × 2.4 s).
- **What follows:** the note's top two fixes (checkpoints without a drain, one call for the whole prompt; a taper of
  small micro-batches) are est. 2.0–3.1× on prompt time and are not built. They move up (Q770, C218r, S30).

### 3.4 The speed instrument is noisier than the decisions it makes

- **Data:** two runs of one line on the release build differ by 3.7 % in task time on all 12 tasks and 1.6 % on 9
  (`s35ka` against `t1base`); every other arm is one run. `t1neg`'s +5.3 % "loss" is 1.6 points beyond one pair's
  spread. The "negligible" group, each member within the noise alone, summed to a loss.
- **Why:** at temperature 1.0 each run writes different text, so acceptance (0.708–0.718 across the release arms),
  tokens per task and turns per task all move. The equal-work column removes the token count, not the acceptance.
- **What follows:** decide engine changes on paired, fixed-text measurements: T2h-style fixed-depth passes, and replays
  of recorded agent requests at temperature 0 with alternated loads (≥ 4 per arm). Keep the agent check as the
  confirmation of a stacked line, two runs per arm. "Stack small gains" stays, but a member joins the stack only on a
  paired measurement that is not a loss (Q1145).

### 3.5 The quality half of the goal has no task-level measurement

- **Belief:** the one goal is "as high quality Swift Qwen 3.8 Flash Next ... as possible", and "even an IQ1_S
  Flash-Next file is better than Ornith-1.5-122B at Q5".
- **Data:**
  - The shipped Swift Q2_0 has the worst agent-text KLD of the files that fit one rig: 0.2851, against IQ2_XS 0.2248
    and IQ3_XXS 0.1755 ([2026-11-03_flash_next_quality](../../data/2026-11-03_flash_next_quality/README.md)). ISTA's
    own card prices the gap on LiveCodeBench v6: BF16 87.43, IQ3_XXS 86.29, IQ2_XS 83.43, Q2_0 81.14.
  - IQ2_XS fits the release line unchanged (328 MiB free on the fullest card with the MTP draft at 64k, `-ub 512`) and
    waits only for a one-hour reshard (C160). The IQ3_XXS agent check (S13) never ran, with the shards on rig 1.
  - No Flash-Next file has a pass rate on any task set beyond the bug-fix check, which sits at 12/12: a 12/12 result
    only says the pass rate is above 77.9 % (one-sided 95 %).
  - Every Flash-Next KLD uses Swift's Q8_0 as the reference (itself ~0.022 from BF16 by the maker's card) on texts of
    at most 8,192 tokens; the 512-token wiki and code texts never exercise the sparse attention's selection.
  - The IQ1_S-beats-Ornith-Q5 statement is not recorded in this project. It matches the owner's Elixir SWE Bench,
    where one run of 20 cases gave ISTA's Flash-Next Coder IQ1_M 10 passes and Ornith-1.5-122B UD-Q5_K_XL 7 (E14).
    The sister repo's read says one run of 20 cases cannot rank near neighbours, so the premise behind running
    Flash-Next rests on one run of one task family.
- **What follows:** a task-quality lane on the quant machine (stock CUDA, no Pascal kernels needed): the 82 polyglot
  tasks, several seeds, Swift Q2_0 against IQ2_XS against IQ3_XXS, with pass rate, tokens per solved task and loop
  rate (Q1146). Its result prices every speed-for-quality trade, including the file the rig should run. IQ2_XS goes to
  the rig at once (Q710: the reshard C160, then rig row R40).

### 3.6 Tokens per task were never a lever for Flash-Next

- **Data:** Flash-Next's default reasoning effort is `xhigh`, which Artificial Analysis found very verbose (240M
  output tokens for its index against a median of 140M, [variants](flash_next_variants.md)). The preset does not set
  an effort. v2 named shorter reasoning the largest wall-clock lever for Flint (registry Q85, Q93); nothing carried it
  to Flash-Next. Loops grow at 2 bits (arXiv 2606.02011); the loop census (Q1138) is ready and not run.
- **What follows:** measure tokens per solved task and pass rate at `xhigh`, `medium` and `low` effort and with a
  thinking budget, in Q1146's lane (Q1147). If `medium` keeps the pass rate, a preset variant cuts task time by more
  than any kernel left (est.: decode is 69 % of a release-check task and est. ~80 % of a real one).

### 3.7 Built, never measured

Work that exists and would answer a question in under a day of machine time:
- the sparse Pascal attention (Q230), in the release build behind a switch;
- probabilistic verification (`--spec-draft-sampling probabilistic`, Q731): on the Qwen-V100 fork's Flash-Next MTP
  line, greedy draft sampling cost 5 points of acceptance and ~7 % of tok/s against it
  ([V100 note](qwen_v100_llamacpp.md)); C243r, its rig run, is "ready" since 2026-11-04;
- the N2P Flash-Next files (C368, C394): built, no KLD, no rig run;
- the two-rig file's agent and code KLD (FN18), stopped by a GPU fault on 2026-11-04;
- C326's code-text KLD, the last input of the v3.1 file pick;
- the IQ3_XXS agent check (S13) and the IQ2_XS reshard (C160).

A day spent on these answers more than a week of new kernels. They go first (section 5).

### 3.8 The fork drifts from upstream

- **Data:** the fork's base is b12076 plus 16 commits; upstream is at b12233 on 2026-11-09. Two of that day's merges
  touch Flash-Next: #31419 skips uploading inputs no node reads (pp512 +2.1 to +4.5 % on a two-GPU layer split, greedy
  output identical) and #31185 loads the n-gram table from its own file (`--ple`), as `gguf_reshard.py` does by hand.
  Upstream's own Flash-Next work (MTP #29761, the depth fix for issue #28734, the gather-based decode in
  unslothai/llama.cpp#165 with its kv-cells scan fix, "about 58 % of decode CPU at 141K") lands on newer bases.
- **What follows:** one rebase onto the newest tag per major cycle, as `-ladder5` did (Q1150). The kv-cells scan and
  mask-upload fixes matter more on a 2-core Celeron than on the hosts they were measured on.

## 4. The levers, priced

Each gain is est., on two workloads: the release check's mean task (161 s: 48 s of prompts, 111 s of decoding) and a
real task shaped like E14's (est. ~80 % decoding at 20–60k tokens of context, 60–80 prompts). "Evidence" says what the
estimate rests on. Ordered by expected value on real tasks, with the probability that the lever works at all.

| # | Lever | Evidence | Release-check task | Real task | Works at all | Cost |
|---|---|---|---:|---:|---|---|
| 1 | **Fewer tokens per solved task:** reasoning effort `medium` or a thinking budget, loops stopped (Q1138, Q1139) | `xhigh` default, very verbose; E14's failures mostly at the budget; Flint's v2 note | est. −10 to −30 % | est. −20 to −50 % | medium: quality decides | quant machine first, no code |
| 2 | **Speed at depth:** the sparse Pascal attention (Q230) | dense kernel reads every cell; 54.9 ms a token at 1k, 94.3 and 99.0 at 32k | est. −1 to −3 % | est. −15 to −35 % of decode at 20–60k | high: built, CPU-checked | GPU check, one rig run |
| 3 | **The step outside the verify pass** (~45–50 ms of ~152) | §2; `-ladder3`'s split; the keep-alive's half-delivered gain | est. −8 to −15 % | the same share | high that half is removable | a profile (S37), then code |
| 4 | **Prompt structure:** checkpoints without a drain, the taper, checkpoints on the cards | ~7 s tail on every long prompt, ~2.4 s on every short one | est. −10 to −15 % | est. −8 to −15 % (more prompts, deeper) | high: the note's model reproduces the log | two built, one to write |
| 5 | **Probabilistic verification** (Q731) | V100 fork, same model and draft: ~7 % tok/s | est. −4 to −5 % | the same | high: exact, in the base | one rig run |
| 6 | **A better file at the same line:** IQ2_XS | KLD 0.285 → 0.225; ISTA: LCB 81.14 → 83.43 | speed cost unknown | quality up, fewer retries (est.) | high that it fits; speed unknown | 1 h reshard, one rig run |
| 7 | **Verify-pass kernels**, the projections at k = 4–5 (+29.8 ms from k = 1 to 5) | profile §2; Q3_K at ~35 GB/s on the k-column path (Q823) | est. −4 to −7 % | the same | medium | kernel work |
| 8 | **A draft of 4–5 tokens**, after #3 | MTP trained multi-step (4.07 at four steps); positions 4–7 unmeasured on agent text (Q806 for the MTP draft, Q804 for DFlash2) | est. 0 to −8 % | the same | medium: pays only with a cheap draft | offline first |
| 9 | **The upstream rebase** to b12233 or newer | #31419 +2.1–4.5 % pp512 on a layer split; the kv-cells scan and mask fixes for depth | est. −1 to −2 % | more at depth | high | a merge and the checks |
| 10 | **IQ3_XXS or a v3.1 mix** with the room left on the cards | KLD 0.176; 9,617 MiB free across the cards; experts moved for 64k with the draft | slower (est. +16–19 ms a token on v3's base, stale) | quality up | medium | fit, then one rig run |

Not in the table, with the reason:
- **Pipelined speculation on the idle cards:** 1.6× slower on v3's base (4.9 against 7.9 tok/s, a wrong guess cost
  +372 ms), and by reading broken with the default fused conv (Q1042). Its premise, one-token steps cheaper than a
  verify, no longer holds: a 4-token verify costs 2.0 one-token passes on the release line (101.6 against 51.1 ms).
- **A persistent kernel or an own engine:** launches are ≤ 2.3 ms a token ([floor](flash_next_floor_and_engine.md)),
  and the persistent-pipeline bench found 0.77 ms a token beyond the clock, which the keep-alive already holds.
- **Lossy acceptance** (AdaptiveSpec, arXiv 2609.02897: up to +56 % over EAGLE-3 with "93 % to fully lossless" task
  accuracy): it changes the answers, which the project promises not to do. Parked as Q1149 until the task lane can
  price it.

## 5. The order of work

The principle: **measure what is built before building more, and measure what users feel before tuning what they do
not.** Three machines run in parallel (rig 1, rig 2, the quant machine), one session per rig, each row its own session.
The private queue holds the same order with each row's brief.

### Step 1: answers from what exists (no new engine code; about a week of machine time)

| Order | Row | Machine | What it answers | Est. time |
|---|---|---|---|---|
| 1 | Q1138 (loop census) | quant, CPU | how many runaways are loops; sizes Q1139 | ~1 h |
| 2 | Q710 (IQ2_XS reshard and fit probe) | quant, CPU | IQ2_XS ready for the rig | ~1 h |
| 3 | Q552's agent and code KLD (FN18) | quant, GPUs | the two-rig file's quality, open since a GPU fault | ~45 min |
| 4 | C326's code text | quant, GPUs | the last input of the v3.1 pick | ~2.5 h |
| 5 | Q1146 (task-quality lane), pilot | quant, GPUs | pass rate and tokens per solved task: Swift Q2_0 against IQ2_XS, 82 polyglot tasks, one seed | ~5 h |
| 6 | Q1147 (reasoning effort), first arm | quant, GPUs | Swift Q2_0 at `medium` in the same lane | ~4 h |
| 7 | Q731 on the MTP line | rig | probabilistic verification against the preset, paired replay | ~1.5 h |
| 8 | S37 on the release build | rig | the served step split per pass (draft, crossings, server work, output rows) | ~1.5 h |
| 9 | C441, then C313 | rig | the sparse Pascal attention: native correctness, then T(k) at 8k and 32k | ~1.2 h |
| 10 | Q1143 (deep lane) | rig | the release line's decode and prompt speed at 0–48k, first and second pass, sparse attention off and on | ~2 h |
| 11 | C218r and S30 on the release build | rig | the taper and the checkpoints on the cards, on the release session's 66 recorded prompts | ~2 h |
| 12 | Q710's rig arm | rig | IQ2_XS on the release line: speed and pass rate on the bug-fix check | ~1.5 h |
| 13 | Q806 (the MTP draft), then Q804 (DFlash2) | quant, GPUs | acceptance at draft positions 1–7 on agent text: what a longer draft could pay | ~1–3 h each |
| 14 | per-part profile of the release line (rig row R41) | rig | k = 1 and k = 4 by part with the hop cast and the keep-alive on; every ms estimate in the queue rests on older profiles | ~45 min |
| 15 | Q1152 (reflection and reasoning census) | quant, CPU | whether Flash-Next's reasoning has slack beyond the effort setting | ~1 h |

### Step 2: code, in the order the answers price it

1. **Prompt structure** (Q770, the checkpoint without a drain; with C218r and S30 if step 1 shows them gains). The
   largest measured fixed cost: ~7 s on every long prompt, ~2.4 s on every short one.
2. **The step outside the verify pass,** in the order S37's profile ranks its parts: the draft passes fed on the device
   (Q1148), the main model's hidden rows to the draft without the host (Q970), the server work between draft and
   verify (Q960, Q1104).
3. **The sparse attention on by default** for Flash-Next if C441 and C313 pass, then the indexer's own cost at depth.
4. **The upstream rebase** (Q1150) onto the newest tag, carrying every switch, as `-ladder5` did.
5. **Verify-pass kernels** for the projections at k = 4–5, the largest growing part of the verify pass.
6. **A draft of 4–5 tokens** once the step is cheap and Q806 shows the MTP draft's acceptance at positions 4–5.

### Step 3: the file, once the task lane prices quality

- If IQ2_XS keeps the speed and the task lane shows a pass-rate gain: it becomes the preset's file, and the release
  table gets its own row.
- If IQ3_XXS's gain over IQ2_XS is large on tasks: S13 at 64k with experts moved, then the v3.1 mix (`galloc.py`)
  priced at the same line.
- Two rigs (C260) only after a single-rig file can no longer gain, with the estimate redone on today's speeds (its
  est. 5.5–6.3 tok/s rests on v3's base at 128–136 ms a token).

### What the release check becomes

The bug-fix check stays as the continuity number. Each Flash-Next release also reports the deep lane (Q1143) and,
once built, the deep agent lane (Q1144), and the task lane's pass rate and tokens per solved task for its file and
effort (Q1146, Q1147).

## 6. New registry rows from this review

In [quants/experiments.md](../../../quants/experiments.md), "Qwen3.8-Flash-Next (continued)":

| Row | Idea | Status |
|---|---|---|
| Q1143 | The deep lane: the release line's decode and prompt passes at fixed depths on recorded agent text, first and second pass, paired switches | planned |
| Q1144 | A deep agent lane: a realistic agent prefix and multi-file tasks that reach 20–60k tokens | planned |
| Q1145 | Engine decisions on paired fixed-text passes; `t1neg`'s group again with the build's own T(k) table | planned |
| Q1146 | The task-quality lane for Flash-Next files on the quant machine | planned |
| Q1147 | Reasoning effort and a thinking budget for Flash-Next, priced in Q1146's lane | planned |
| Q1148 | The MTP draft's passes fed on the device (Q961's second way out) | planned |
| Q1149 | Lossy acceptance (AdaptiveSpec) | parked |
| Q1150 | `-ladder6`: the upstream rebase onto the newest tag | planned |
| Q1151 | Flash-Next against Ornith-1.5-122B in Q1146's lane, to back the premise that the rigs run Flash-Next | planned |
| Q1152 | Tokens per task beyond the effort setting: a reflection and reasoning census of Flash-Next's replies, then a first-finish race and early-stop rules priced offline | planned |

Existing rows this review moves up: Q1138 (loops), Q710 (IQ2_XS), Q552 (FN18's KLD), Q731 (probabilistic
verification, now on the MTP line), Q961 and Q960 (S37, C405), Q230 and Q873 (C441, C313), Q770, Q771 and Q440 (the
prompt fixes), Q227 (S13), Q806 and Q804 (acceptance at positions 1–7),
Q681 (the light profiler, for the release line's per-part profile).

## Sources

Project data, all in this repository:
- [2026-11-05_release_flashnext](../../data/2026-11-05_release_flashnext/README.md): the release arms, `tk_alt.tsv`,
  `report_table.md`;
- [2026-11-05_keepalive](../../data/2026-11-05_keepalive/README.md),
  [2026-11-04_s35_stack](../../data/2026-11-04_s35_stack/README.md),
  [2026-11-04_stacked_gains](../../data/2026-11-04_stacked_gains/README.md);
- [2026-11-04_moe_skip](../../data/2026-11-04_moe_skip/README.md) (the newest per-part profile),
  [2026-11-04_kpass_profile](../../data/2026-11-04_kpass_profile/README.md),
  [2026-11-04_verify_host](../../data/2026-11-04_verify_host/README.md),
  [2026-11-04_draft_one_card](../../data/2026-11-04_draft_one_card/README.md);
- [2026-11-04_card_hops](../../data/2026-11-04_card_hops/README.md) (decode at ~1k, ~8k and ~32k);
- [2026-11-04_qsa_sparse_attn](../../data/2026-11-04_qsa_sparse_attn/README.md),
  [2026-11-06_flint_q5_depth](../../data/2026-11-06_flint_q5_depth/README.md);
- [2026-11-03_flash_next_quality](../../data/2026-11-03_flash_next_quality/README.md),
  [2026-11-04_file_choice](../../data/2026-11-04_file_choice/README.md),
  [2026-11-04_fmt_mixes](../../data/2026-11-04_fmt_mixes/README.md);
- notes: [Flash-Next prefill](flash_next_prefill.md), [floor and engine](flash_next_floor_and_engine.md),
  [acceptance](flash_next_acceptance.md), [variants](flash_next_variants.md), [Qwen-V100](qwen_v100_llamacpp.md),
  [agent workload check](agent_workload_check.md).

Sister repo: [E14, the Elixir SWE Bench's published runs](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs/tree/main/docs/data/2026-11-09_bench_runs_read).

The model and upstream, read 2026-11-09:
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (model card) and
  [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320) (QSA trained jointly; MTP accepted
  length 4.07 at four steps; branches stored in FP8 "with almost no loss in quality");
- [LMSYS: Qwen3.8-Flash-Next in SGLang](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/) (IndexShare MTP, the
  n-gram rows prefetched per token, accept length 3.3);
- llama.cpp: [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (qwen4exp, QSA as a mask),
  [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) (qwen4exp MTP, merged 2026-10-01),
  [issue #28734](https://github.com/ggml-org/llama.cpp/issues/28734) (decode slowing with context),
  [#31419](https://github.com/ggml-org/llama.cpp/pull/31419) (inputs no node reads),
  [#31185](https://github.com/ggml-org/llama.cpp/pull/31185) (external n-gram table),
  [unslothai/llama.cpp#165](https://github.com/unslothai/llama.cpp/pull/165) (gather-based QSA decode);
- speculation: [SuffixDecoding](https://arxiv.org/abs/2411.04975), [AgSpec](https://arxiv.org/abs/2610.01108),
  [AdaptiveSpec](https://arxiv.org/abs/2609.02897), [PipeSpec](https://arxiv.org/abs/2505.01572).
