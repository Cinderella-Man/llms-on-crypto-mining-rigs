# 12. Long conversations: speed at depth

> **In short**
> - On v3, writing slowed as a conversation grew. One step took 123.21 ms at the start and 301.90 ms with 49,152
>   tokens of context: 2.45 times as long.
> - The cause was one fallback: on these cards the attention layers read their whole cache, although the model itself
>   picks only about 2,051 cells of it for each token.
> - A kernel that reads only the picked cells brought the deep step to 180.77 ms. Four more changes went after what
>   still grew. With the first two of them and the kernels of [chapter 13](13-the-check-pass.md), the step at 49,152
>   tokens is 149.81 and 150.10 ms.
> - The draft keeps paying at depth. A 131,072-token window works on one rig, and 196,608 is the largest that fits.

Facts in this chapter are from 2026-11-11. The model is Qwen3.8-Flash-Next on the v3 release preset with the draft
on, measured on the deep lane of [chapter 11](11-after-v3.md): a *step* is one check pass plus the draft's passes,
and its time is taken from the second of two identical requests at each depth. *Depth* is the number of tokens
already in the conversation.

## 12.1 Why writing slows as the conversation grows

[Chapter 2](02-how-an-llm-makes-text.md) introduced the KV cache: every token leaves a key and a value in each
attention layer, and every later token reads them. Twelve of Flash-Next's 48 layers work this way. The longer the
conversation, the more each new token has to read.

On the v3 release, two loads at each depth:

| Depth (tokens) | One step, ms | Against the start |
|---:|---|---:|
| 0 | 123.21 and 123.23 | |
| 8,192 | 153.00 and 152.97 | 1.24× |
| 16,384 | 182.25 and 182.98 | 1.48× |
| 32,768 | 241.65 and 242.49 | 1.96× |
| 49,152 | 301.90 and 301.48 | 2.45× |

That is est. 3.64 ms more per step for every thousand tokens of history (the first load's two ends). A coding agent on a real task reaches these
depths ([chapter 11](11-after-v3.md)), so by the time it matters most, the model writes at less than half its
starting speed.

It should not be this way, and the model's makers saw to that. Flash-Next's attention is *sparse*. Beside the cache
it keeps a small index: one short key for every group of four cache cells. For a new token the model scores all
index keys against the token, keeps the best 2,051 cells, and attends only to those. The cost of attention should
then stay flat however long the conversation gets.

On a P106 it did not, for a plain reason. llama.cpp's code that reads only the picked cells needs a newer GPU. On
Pascal it falls back to reading the whole cache and ignoring most of it: 64 MiB per layer and token at 32,000 tokens
where 4 MiB would do ([chapter 6](06-gpu-code.md)).

## 12.2 Story: the kernel that had never run

**The idea.** Give Pascal its own attention kernel that reads only each token's picked cells. I had written it on
2026-11-04. It shipped in v3 behind a switch, off, checked on a CPU, and had never run on any GPU.

**How it should work.** The index hands the kernel a list of cells per token. The kernel walks that list and loads
only those keys and values. The work per token no longer depends on the length of the cache, only on the 2,051
cells.

**What I did.** Three checks first, in the order of [chapter 11](11-after-v3.md): the whole model on newer cards
with the rig's own GPU code, with the switch off and on, after prompts of 8,192, 32,768 and 60,000 tokens; then the
kernel's recorded test cases replayed on one P106, with two planted bugs; then the deep lane on a rig.

**What happened.** The replay passed 48 of 48 cases, and each planted bug failed 29 of them. Then the timing:

| Depth (tokens) | Whole cache read, ms a step | Picked cells only, ms a step | Change |
|---:|---|---|---:|
| 0 | 123.21 and 123.23 | 123.36 and 123.38 | +0.15 ms |
| 8,192 | 153.00 and 152.97 | 150.40 and 150.34 | −2.62 ms |
| 16,384 | 182.25 and 182.98 | 149.73 and 150.80 | −32.35 ms |
| 32,768 | 241.65 and 242.49 | 165.86 and 165.74 | −76.27 ms |
| 49,152 | 301.90 and 301.48 | 180.77 and 180.96 | −120.82 ms |

At 49,152 tokens the model wrote 17.81 and 14.68 tokens per second in two loads, against 10.59 and 11.54 without
the kernel. At the very start it costs 0.15 ms a step.

**Why.** Reading fewer bytes was always the lever on this rig. Here the bytes were being read and thrown away, so
nothing of the model's output had to change to stop reading them.

**What the first check taught.** Comparing the written text, switch off against on, looked like a strict test. It
is a blunt one. Deep into a conversation this model is so sensitive to rounding that the same kernel, adding the
same numbers in another order, moves the next token's log-probabilities by 0.0945, 0.109 and 0.11 in three
settings. Two loads of one unchanged setting did not repeat at 32,768 and 60,000 tokens: their log-probabilities
differed by up to 0.564 and 0.474, and one load at 60,000 wrote another text. And a deliberate 1 % error in the
kernel's output left the text unchanged at both of those depths. So a matching text proves little and
a differing text proves nothing. The quality check that counts is the KL divergence of
[chapter 3](03-measuring.md), with recorded test cases and planted bugs for the kernel itself.

## 12.3 Story: where to switch kernels

**The puzzle.** At 8,192 tokens the new kernel gained only 2.62 ms, and at 16,384 it gained 32.35. Agent turns end
at about 2,800 to 10,000 tokens, right in that gap.

**Why.** The switch came with a rule: stay on the old kernel while the cache holds fewer than 1.5 times the cells
the new one would read. For a four-token check that is 12,306 cells. At 8,192 tokens the check still ran on the old
kernel, and only the draft's passes used the new one.

**What I did.** I ran five settings of that rule on a rig at 0, 4,096, 8,192 and 12,288 tokens: the factor at 1.5,
1.0, 0.5 and 0.25, and no rule at all.

**What happened.** A factor of 1.0 was best: 143.43 and 144.13 ms a step at 8,192 tokens against 151.37 and 150.30
with 1.5, and within 0.35 ms of it at the other three depths. The factors 0.5 and 0.25 and no rule at all saved about as much at 8,192 (6.90, 6.30 and 7.09 ms)
but lost 2.50, 2.71 and 2.70 ms at 4,096.

**Why.** The new kernel has a fixed cost: it works from a list, one token at a time. On a small cache the old
kernel's plain scan is cheaper. There is a crossover, as between an index lookup and a table scan in a database,
and it has to be measured, not assumed. A kernel that serves a check's four tokens from one shared read is written
and waits for its rig run; whether it moves the crossover lower is open.

## 12.4 Story: what still grew

**The puzzle.** With the new kernel the step still grew 30.44 ms between 16,384 and 49,152 tokens. The attention
that had caused the slope was fixed. What was left?

**What I did.** Two profiles at depth: the GPU's time by part, then the host's side of every hand-off.

**What happened.**

| What grows from 16,384 to 49,152 tokens | ms a step |
|---|---:|
| Copying every index key into a fresh buffer before scoring it | 9.71 |
| Scoring the index keys | 3.41 |
| The attention itself, on its picked cells | 0.25 |
| Other GPU work | 0.61 |
| **GPU time** | **13.98** |
| **Outside the GPU: host work and uploads** | **est. 16.46** |

The attention was no longer the problem: 0.25 ms. More than half of the growth was not on any card. The host
profile found it. On every pass the host rebuilt the small tables that say which cells form each group of the
index, and uploaded them whole to every card. Those tables grow with the cache, and the uploads crossed the links
at 186.5 and 187.7 MB/s: the link speed of [chapter 1](01-the-rig.md).

**Why.** Each of these was invisible while the attention cost ten times as much. Removing the largest cost is what
makes the next ones measurable. And the pattern is familiar from backend work: a client that re-sends a whole
table on every request, when one row changed.

## 12.5 Four fixes for what grew

Each fix went through the checks of [chapter 11](11-after-v3.md), gives the same output bit for bit, and was timed
on a rig on its own loads. The step at 49,152 tokens, without and with each:

| Change | What it does | Step at 49,152 tokens, ms |
|---|---|---|
| Read the index keys in place | The scoring reads each key where it sits in the cache; the copy disappears | 168.94 and 169.16, against 179.58 and 180.15 |
| Keep the index tables on the cards | Each pass sends only the rows that changed: 3,072 bytes a card for a step, where the whole tables are 247,040 bytes | 173.97, against 179.96 |
| Send five small inputs inside that update | Two copies to a card per pass instead of seven | 173.89, against 174.00 |
| Build the index mask on the cards | 8 bytes a token cross the link, instead of 98,816 bytes a check pass | 172.04, against 173.87 |

The second row's base differs from the first's, and the last two were measured on top of the second: each row
compares one change against the same build without it.

Two honest notes. Reading the keys in place makes the scoring itself slower, 8.454 ms a step against 5.392 on the
copied keys, because the reads now land scattered across the cache; the copy it removes cost more, so the change is
a net gain, and the slower scoring is on the list. And the second fix was estimated at 12.43 ms and gave 5.99: part
of what grows outside the GPU grows with or without the uploads. What that part is has not been measured yet.

The mask fix also frees memory: each card's work buffer is 53.05 or 36.67 MiB smaller.

## 12.6 Does guessing ahead still pay at depth?

**The worry.** In two other engines' published runs, a draft of this kind loses at depth. If the check pass gets
dearer as the cache grows while the guesses get no better, the draft should be switched off past some depth.

**What I did.** The deep lane with the draft on and off, two loads each, and a count of how many guesses the model
accepts at each depth.

**What happened.** Over the five depths the draft wrote 18.87 and 19.06 tokens per second against 16.37 and 16.30
without it, 16.1 % more. It was a gain at 0, 16,384 and 49,152 tokens, and inside the spread of its own two loads
at 8,192 and 32,768; it lost at no depth. The share of guesses accepted did not fall with depth: 0.633 at 8,192
tokens, 0.762 at 16,384, 0.627 at 32,768 and 0.637 at 49,152.

**The catch is in reading, not writing.** The draft is a small model that must also read every prompt token to keep
up. With it on, the rig read prompts 28.4 % slower on an empty cache and 41.3 % slower on 32,188 cached tokens.
[Chapter 14](14-reading-prompts.md) is about that cost.

## 12.7 Bigger windows

The preset's window is 65,536 tokens, and a real attempt writes more than 40,000 tokens on average
([chapter 11](11-after-v3.md)). So I measured how far one rig can go.

| Window | Fits? | Writing speed deep in the window | The 4 GB host at its lowest |
|---|---|---|---|
| 131,072 tokens | Yes | at 120,000 tokens: 15.24 and 12.71 tok/s with the draft, 10.96 and 10.90 without | 175 MiB available |
| 196,608 tokens | Yes, the largest that does | at 196,000 tokens: 10.56 tok/s with the draft, 8.80 without | 109 MiB available |
| 262,144 tokens | No | two cards end with less than 150 MiB free | |

Memory on the cards decides it: the cache grows with the window, and two of the nine cards carry both more layers
and more attention layers than the rest.

One caution about the draft this deep. Those runs wrote with the simplest sampling. With the makers' sampling
settings the draft's guesses are accepted less often far into a conversation: a step yielded 2.93 tokens at 65,536
tokens and 2.69 at 120,000. At v3's step times that puts the draft at break-even there, est. +1.4 % and −1.3 %, and
below it in any turn that also reads a long prompt. Those step times are v3's, and the step at depth has halved
since, so this is measured again before anything is built on it.

## 12.8 Where depth stands

The sparse attention with its threshold at 1.0, the in-place index read and the index tables on the cards, together
with the check-pass kernels of [chapter 13](13-the-check-pass.md) and the prompt changes of
[chapter 14](14-reading-prompts.md), in one run against v3 on the same rig the same day. The two smallest fixes of
section 12.5 were timed while that run was under way and are not in it:

| Depth (tokens) | v3, ms a step | The v4 line, ms a step | Change |
|---:|---|---|---:|
| 0 | 123.66 and 123.70 | 108.42 and 108.80 | −12.2 % |
| 8,192 | 158.48 and 152.65 | 127.22 and 126.80 | −18.4 % |
| 16,384 | 184.07 and 185.79 | 130.06 and 130.33 | −29.6 % |
| 32,768 | 244.09 and 244.11 | 140.05 and 140.58 | −42.5 % |
| 49,152 | 302.25 and 302.37 | 149.81 and 150.10 | −50.4 % |

The slope is nearly gone: from the start to 49,152 tokens a step now grows 41.4 ms, where it grew 178.6 (the first
load of each).

What is left has a clear shape. Between the start and 8,192 tokens the step still grows 18.80 and 18.00 ms in the two
loads, and that stretch is exactly where an agent on small tasks works. It has not been profiled with all of these changes on. That map is
the next step for depth.

## What to remember

- A cost that grows with the conversation is worth more than its size at the start suggests: real agent work
  happens deep.
- The largest fix read fewer bytes without changing a single result. Look for bytes that are read and thrown away.
- A fast path and a simple path have a crossover. Measure where it is; for a four-token check it sits near 8,200
  cache cells.
- Removing the biggest cost makes the next ones visible. After the attention came a copy, then uploads of tables
  that had barely changed.
- Deep into a conversation, comparing written text is a blunt test. Rounding alone changes the text.
- The draft keeps paying at depth in writing. Its cost is in reading prompts.

## Where the details are

- The deep lane's runs, with every table of this chapter's first seven sections:
  [2026-11-10_deep_lane](../data/2026-11-10_deep_lane/README.md): the sparse attention off and on (Q1143), the draft
  on and off (Q1153), the 131,072-token window (Q1154, Q1218), the 196,608-token window (Q1219), the threshold
  (Q1212).
- The sparse attention kernel: Q230 ([design](../data/2026-11-04_qsa_sparse_attn/README.md)); the whole model on
  newer cards and what rounding does, Q1171 ([data](../data/2026-11-09_sparse_attn_whole_model/README.md)); the
  replay on a P106 ([data](../data/2026-11-10_replay_checks_rig2/README.md)); pass times by size at depth, Q873
  ([data](../data/2026-11-10_tk_depth/README.md)). Switch `GGML_CUDA_FATTN_SPARSE_PASCAL=1` with
  `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0`; the kernel for a check's four tokens at once is Q1315
  ([data](../data/2026-11-11_c103_union_gather/README.md)), the crossover Q1252.
- What still grew: Q1213 and Q1243 ([data](../data/2026-11-10_step_parts_depth/README.md)).
- The four fixes: the in-place index read, Q1244 and Q1282
  ([code](../data/2026-11-11_c704_indexer_gather/README.md), [rig](../data/2026-11-11_r77_c704_p106/README.md)),
  its slower scoring Q1342; the tables on the cards, Q640 and Q1273
  ([code](../data/2026-11-11_c100_kpool_on_cards/README.md), [rig](../data/2026-11-11_r74_c100_p106/README.md)),
  what still grows outside the GPU Q1356; the small inputs, Q1293
  ([code](../data/2026-11-11_c720_kpool_small_inputs/README.md), [rig](../data/2026-11-11_r80_c720_p106/README.md));
  the mask, Q1284 ([code](../data/2026-11-11_c719_pool_mask_on_cards/README.md),
  [rig](../data/2026-11-11_r78_c719_p106/README.md)). Switches `LLAMA_QSA_INDEXER_ROWS=1`,
  `LLAMA_QSA_KPOOL_DEV=1`, `LLAMA_QSA_KPOOL_STAGE=1`, `LLAMA_QSA_POOL_MASK_DEV=1`.
- The draft's acceptance by depth: Q1172 ([data](../data/2026-11-10_draft_acceptance_by_depth/README.md)); with the
  makers' sampling at 65,536 and 120,000 tokens, Q1251 ([data](../data/2026-11-11_fn65_draft_yield_deep/README.md));
  the same question on the v4 line, Q1403. The other engines' runs are in
  [the second review](../research/notes/flash_next_review2_2026-11.md).
- v3 against the v4 line by depth: Q1322 ([data](../data/2026-11-11_r90_v4_vs_v3/README.md)); the map between the
  start and 8,192 tokens, Q1393.
- Registry: [quants/experiments.md](../../quants/experiments.md).
