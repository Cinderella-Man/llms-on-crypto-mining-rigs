# 13. The check pass: making four tokens cheaper

> **In short**
> - With the draft on, the model's main work is a *check pass*: one pass over the newest token and three guesses. On
>   v3 it took 102.10 ms of GPU time, where a pass over one token takes 45.05.
> - Two kernels rewritten for two to four tokens shortened the step, each alone: from 122.11 to 116.37 ms with the
>   output layer's (8.105 to 3.149 ms for four tokens), and from 122.10 to 114.01 ms with the one for two weight
>   formats (16.212 to 8.291 ms a check).
> - Both add each token's numbers in exactly the order plain one-token decoding does. On v3 a check over several
>   tokens was a slightly different arithmetic, and this model turns that into different text.
> - Accepting guesses by probability instead of by exact match made writing 2.97 % faster on the rig.
> - A longer draft yields more tokens per step, 4.50 at six guesses against 3.24 at three, but pays only if each
>   extra checked token is cheap enough. Token trees lose to a plain longer draft.

Facts in this chapter are from 2026-11-11. The model is Qwen3.8-Flash-Next on the release preset with its draft of
three tokens ([chapter 8](08-guessing-ahead.md)). Step times are from the second of two identical requests at the
start of a conversation, on the rebased build of [chapter 11](11-after-v3.md) with one switch set at a time.

## 13.1 The ledger of a step

[Chapter 11](11-after-v3.md) mapped one step on v3: 133.18 ms, of which the check pass's GPU time is 102.10, the
draft's own passes 18.29, the server between two steps 7.82 and the rest 4.98. It also showed that a check pass
costs 15.68 ms more than a plain pass over four tokens.

So there are four places to look, and this chapter takes them in turn:

| Where | On v3 | What this chapter does about it |
|---|---|---|
| The second, third and fourth token of a pass | 14.30 ms each | new kernels for two to four tokens |
| The output layer, computed for all four tokens | 8.05 ms | a kernel that reads it once |
| Fewer steps for the same text | 3.043, 3.072 and 3.090 tokens a step in three loads | another accept rule; a longer draft |
| The server between two steps | 7.82 ms | traced; five small changes written |

The copies of the recurrent state that let a wrong guess be undone, 5.37 ms a pass, are not attacked here. Upstream
llama.cpp is working on replaying the rejected tokens instead of copying the state, and I wait for that.

## 13.2 Story: the output layer, four times

**The puzzle.** The output layer is one big matrix, 437 MB in the file the rig runs. For one token its product takes
2.86 ms, which is simply the time to read 437 MB. For four tokens it took 8.05 ms. The weights are read once either
way. Why 2.8 times the time?

**What I found.** The kernel that handles several tokens fetched each token's input again for every row of the
matrix, from the card's small scratch memory. I read its machine code: of the est. 452 clock cycles that one
round of its inner loop spends on memory operations, est. 320 go to those fetches. The weights were waiting for
the inputs.

**What I did.** I wrote a kernel for two to four tokens that lays the work out the way the one-token kernel does:
each group of threads reads and unpacks a block of weights once and applies it to every token's inputs, which it
holds ready.

**What happened.** On a P106 the four-token product takes 3,340.78 and 3,346.73 µs in two runs, against 8,103.16 and
8,100.11 before: 1.10 times one token. In real generation the output layer went from 8.105 to 3.149 ms a check
pass (the first load of each), and the step from 122.11 and 121.81 ms to 116.37 and 116.25. Every result is the one-token kernel's, bit for
bit: of 21 compared products none differed, and a control that should differ did, 14 times.

Two settings were wrong at first. The design let two rows share one fetch of the inputs and read ahead in the
weights; my estimate made both the default. The rig measured one row at a time, without the read-ahead, faster.
The default follows the measurement now.

**Why.** "Read each weight once" is not enough on a small chip. What the kernel does between the reads decides
whether the memory is kept busy.

## 13.3 Story: the same method for two more formats

**The idea.** The attention and recurrent layers store part of their matrices in two 4-bit formats. In a check pass
their products took 16.21 ms. The kernel behind them had the same habit as the output layer's.

**What I did.** The same redesign: the one-token kernel's layout, each weight read and unpacked once for all the
tokens.

**What happened.** Those products fell from 16.212 to 8.291 ms a check pass (the first load of each), and the step
from 122.10 and 122.03 ms to 114.01 and 113.62. Bit for bit again: 0 of 60 compared products differed, the control 50 times.

**Why.** A fix for one hot spot is often a method. Once the output layer's problem had a name, the profile showed
the same pattern in the next-largest group of products, and the second kernel was written the same night.

## 13.4 Story: a check was a different arithmetic

**The puzzle.** Guessing ahead is supposed to change nothing: the model keeps a guess only when it is the token it
would have written itself ([chapter 8](08-guessing-ahead.md)). So with random sampling off, the release preset
should write the same text with the draft and without it. On v3 it did not, at any of five depths, although each
setting repeated its own text exactly.

**What I ruled out.** Three suspects, one experiment each, on newer cards with the rig's own GPU code.

- *The split over the cards.* Both settings got the same split, layer for layer. The texts still parted.
- *A near-tie.* Where the texts part, the two candidate tokens are 0.07 to 1.78 apart in log-probability. Rounding
  does not bridge that in one step.
- *A fault in accepting or undoing guesses.* A draft of one token and a draft of three accept at different places,
  and they wrote the same 256 tokens with the same probabilities to the last digit.

**What it was.** A pass over two or more tokens ran different kernels than a pass over one, and those kernels add
the same numbers in another order. The results differ in the last bits. This model amplifies such a difference to
tenths of a log-probability within a few tokens, and then the text goes another way. The plain setting does it to
itself: when it reads its own reply back as a prompt, and from 8,192 tokens of context on, from one load of the
model to the next.

**What the new kernels change.** Both kernels of the last two sections are built to give each token exactly the
sums of the one-token kernel. With them on, a check pass and plain decoding do the same arithmetic in those
products. The text at the start of a conversation changes once, because the check's numbers are now plain
decoding's. A few products still add in another order; their kernels are written and wait for a rig.

**One thing left open.** In the stacked run of [chapter 15](15-toward-v4.md), the new texts at the two shallowest
depths accepted fewer guesses than v3's: 85 steps against 75 for 256 tokens at the start, 84 against 67 and 71 at
8,192 tokens. At the three deeper points they did not. One text per depth cannot tell bad luck from a real effect. A count over 66 recorded requests can, and it is queued.

**Why.** "Exact" has levels. The draft was exact in its logic. The engine under it was not bit-identical between
pass sizes, and nobody had checked, because on most models a difference in the last bits stays there.

## 13.5 Sampling four results in one go

After the output layer the card samples each of the four results ([chapter 2](02-how-an-llm-makes-text.md)). It did
that with four separate launches of a top-k filter and a padding step. One filter over the four rows together does
the same job: 0.323 ms a check pass against 0.650, the same output bit for bit, and a step of 121.05 and 121.26 ms
against 121.78 and 121.98. It is 0.7 ms. It is kept.

## 13.6 Story: accept by probability

**The idea.** The preset accepts a guess only if it equals the token the model then draws. There is a more generous
rule that keeps the output's statistics exactly the same: accept a guess with a probability that depends on how
much the model and the draft agree on it, and if it is rejected, draw from what the model wanted beyond the draft
([chapter 8](08-guessing-ahead.md), section 8.9). A guess the model finds likely is then often kept even when its
own draw would have landed elsewhere.

**What I did.** Four measurements, cheapest first. A count of tokens per step on newer cards. A timed pair on the
rig, replaying the same 66 recorded requests to both settings. And two checks for a side effect I feared: that the
new rule makes replies longer.

**What happened.**

| Measurement | The preset | Accept by probability |
|---|---|---|
| Tokens a step, counted off the rig (two loads each) | 3.043 and 3.072 | 3.283 and 3.304 |
| Writing speed on the rig, tok/s (two loads each) | 20.60 and 20.70 | 21.31 and 21.22 |
| Agent check off the rig: tasks passed, turns (two runs each) | 12 and 12; 92 and 67 | 12 and 12; 64 and 60 |
| Tokens written for 60 replies to fixed prompts (two loads each) | 66,802 and 91,163 | 95,670 and 72,564 |

On the rig that is 2.97 % faster writing. Replies are not longer: the two loads of one setting differ from each
other far more than the settings do.

**A cost found on the way.** On the rig the rule yielded 4.04 % more tokens a step, half of the count's 7.74 %, and
a step cost 1.10 % more. The step itself was not slower. The extra time sat at the end of each turn: after the
model had accepted its end-of-turn token, the engine still tested the guesses behind it, rejected one, and undid
it, at est. 389.65 ms each. A switch that stops testing after an accepted
end-of-turn token took those rejections from 29 and 24 per 60 requests to 0 and 0, with every reply unchanged.

**Why.** An exact rule can still be a wasteful one. The match rule throws away guesses the model would have been
content with.

## 13.7 Story: a draft that was not what it claimed

**The idea.** The rig runs Swift-1.5, a fine-tune of Flash-Next, with a draft made for the original model. Swift's
own published files carry a draft layer too. A draft trained with the fine-tune should guess it better.

**What I did.** I extracted that layer and compared it with the draft in use, tensor by tensor, then counted tokens
per step with each.

**What happened.** 25 of its 34 tensors are bit for bit the draft already in use, and the other nine are as close as
two roundings of the same numbers. It is the original model's draft layer, carried along by the fine-tune. As a
draft it yielded 3.086 and 3.073 tokens a step against 3.043, 3.072 and 3.090: nothing.

**Why.** Compare the bytes before the benchmark. A diff of the two files answered what rig time would only have
hinted at.

## 13.8 Story: a longer draft, and trees

**The idea.** If a fourth checked token costs 14 ms and a fifth 20 ([chapter 11](11-after-v3.md)), a longer draft
cannot pay on v3. The new kernels change those prices. So first count what a longer draft would yield.

**What I did.** A count on newer cards, no timing: the same 66 recorded requests with drafts of three to seven
tokens, two loads each. Then, from the same recordings, the price of a *token tree*: instead of one chain of
guesses, check the second-best guess at a position too.

**What happened.**

| Guesses per step | Tokens a step | Share of guesses accepted | Pays against three only if a checked token costs less than (est.) |
|---:|---:|---:|---:|
| 3 | 3.2401 | 0.7578 | |
| 4 | 3.7028 | 0.6877 | 14.77 ms |
| 5 | 4.1257 | 0.6359 | 13.97 ms |
| 6 | 4.5037 | 0.5975 | 13.11 ms |
| 7 | 4.5024 | 0.5138 | 8.91 ms |

Both loads of each length gave the same counts. Seven guesses yield no more than six.

Every tree lost to the plain chain with as many checked tokens. A chain of six guesses plus one second-best guess
at the first position, seven checked tokens, yielded 4.4025 tokens a step; a plain chain of seven yielded 4.6025 on
the same steps. A second-best guess at the first position adds 0.0798 tokens a step; one more token on the chain
adds 0.2798 even at the seventh position.

**Why.** This draft is right 0.8481 of the time at the first position, so its second choice rarely matters. Its
weakness is how far it can see, and only a longer chain uses that. Whether the longer chain pays is now a question
about one number, the cost of a checked token with the new kernels, and that is measured next. A rule that picks
the length step by step from the draft's own confidence is in the build since v3; it will be priced on the same
recordings.

## 13.9 The 7.5 ms between two steps

**The puzzle.** Between the draft's last pass and the next check, the server spent 7.82 ms doing nothing a card could
see. On a 133 ms step that is 6 %.

**What I did.** I added a trace to the server that timestamps each stage of a step on the host, and a second one for
each batch the engine prepares.

**What happened.** Of 7.512 ms, 5.610 sat between "the check's inputs are set" and "the first card starts". Inside
that, the largest piece was the engine preparing the work for the first card: 4.379 ms. In three later requests
that had already seen the same text, the same piece took 0.076, 0.077 and 0.076 ms.

The likely cause is the lookup table of [chapter 5](05-nine-cards.md). Every token needs 16 rows of it from the
SSD. For tokens the model has just guessed, those rows are not in memory yet, and the check waits for the disk.
[Chapter 5](05-nine-cards.md) had priced those reads at est. 0.2 to 1.3 % of a token. Against a 136 ms token that
was right. Against today's step they are est. 3 %, and they sit on the one path where nothing else can run.

Five changes for this gap are written and checked on a CPU. None is timed yet:

| Change | Est. saving a step |
|---|---:|
| Ask for a guessed token's table rows the moment it is guessed | up to 4.3 ms |
| Prepare the check pass while the draft's last pass still runs | 0.89 to 1.75 ms |
| Keep the draft's two work plans instead of rebuilding both every step | 0.95 ms |
| Limit a cache clean-up to the cells it can change | 0.67 ms |
| Yield to the server's request queue once a step instead of three times | 0.23 to 0.37 ms |

**Why.** A cost of 1 % stops being 1 % when everything around it gets faster. Old "too small to matter" verdicts
have a date on them.

## 13.10 Written, checked, not yet timed

Eight more kernels follow the method of sections 13.2 and 13.3. Each is exact on a CPU, in the build behind its own
switch, and waits for its turn on a rig:

| Kernel | What it covers |
|---|---|
| 3-bit products at two to eight tokens | the largest group of products left in a check pass: 14.535 ms at four tokens against 8.69 at one, on v3 |
| 6-bit products at two to four tokens | two wide matrices per layer |
| A tile-by-tile version for wide rows | keeps six groups of threads on a chip where the whole-row version fits three to five; est. 0.90 to 1.10 ms a pass |
| Five to eight tokens for three formats | what a longer draft needs |
| The 16-bit mixing matrices at three and four tokens, in the one-token order | exactness, not speed |
| The shared expert's small 32-bit products at four tokens and more | they fall to a slow general routine at four tokens: 0.448 ms |
| The shared expert's three products as one launch | fewer launches in most layers |
| The output layer at one token | the same kernel run for a single token; est. 2.58 ms against 3.02 |

## What to remember

- Map what a pass really does in production. A check pass computes four outputs and keeps rollback copies; a
  benchmark pass does neither.
- Reading each weight once is necessary and not sufficient. On a 5-SM chip, how the inputs are fetched between the
  reads set most of the old kernel's time (est.).
- A fix for one hot spot is often a method. Look for the same pattern in the next-largest part.
- Bit-identical logic on top of non-identical arithmetic is not bit-identical. If two pass sizes use two kernels,
  compare their sums.
- An exact accept rule can still waste guesses. Accepting by probability keeps the same statistics and more tokens.
- Count before timing. Token counts on any machine told which draft lengths and which trees were worth rig time.
- Small costs grow as the rest shrinks. Re-read old verdicts against the current total.

## Where the details are

- The step's ledger: [chapter 11](11-after-v3.md); Q961 and Q1211. The recurrent state's copies and upstream's
  replay: Q1159.
- The output layer's kernel: Q1222 ([design](../data/2026-11-11_c690_head_four_rows/README.md),
  [rig](../data/2026-11-11_r75_c690_p106/README.md)), switch `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1`; its default, Q1347.
- The two 4-bit formats (IQ4_XS and Q4_K): Q1277 ([design](../data/2026-11-11_c722_short_batch_products/README.md),
  [rig](../data/2026-11-11_r79_c722_p106/README.md)), switch `GGML_CUDA_MMVQ_PASCAL_DMT=1`; its default, Q1335.
- Why the draft's text differed: Q1175 ([data](../data/2026-11-10_draft_line_exactness/README.md)); the count
  queued for the new kernels, Q1404.
- One top-k filter: Q1300 ([code](../data/2026-11-11_c691_sampling_launches/README.md),
  [rig](../data/2026-11-11_r83_c691_p106/README.md)), switch `LLAMA_SAMPLING_TOP_K_ROWS=1`.
- Accepting by probability (`--spec-draft-sampling probabilistic --spec-draft-p-min 0`): tokens a step, Q1173
  ([data](../data/2026-11-10_prob_verify_acceptance/README.md)); on the rig, Q731
  ([data](../data/2026-11-10_prob_verify_rig/README.md)); the agent check, Q1183
  ([data](../data/2026-11-10_prob_verify_agent_check/README.md)); reply length, Q1184
  ([data](../data/2026-11-10_prob_verify_reply_length/README.md)). The end-of-turn stop: Q1187 and Q1262
  ([code](../data/2026-11-11_c675_end_of_turn_stop/README.md), [count](../data/2026-11-11_fn69_fn66_stops/README.md)),
  switch `LLAMA_SPEC_REJECT_STOP=1`; its time on a rig, Q1384.
- Swift's own draft layer: Q1168 ([data](../data/2026-11-10_swift_own_mtp_head/README.md)).
- Longer drafts and trees: Q1396 and Q1397 ([data](../data/2026-11-11_fn80_fn81_draft_yield/README.md)), the tools
  Q1395 ([data](../data/2026-11-11_c798_draft_count_tools/README.md)); the rig arms Q1394; the length chosen per
  step, Q687 and Q1402.
- Between two steps: the step trace, Q960 and Q1313 ([code](../data/2026-11-11_c405_server_gap/README.md),
  [rig](../data/2026-11-11_r86_step_trace/README.md)); the batch trace, Q1295
  ([data](../data/2026-11-11_r81_ubatch_trace/README.md)). The five changes: Q1354
  ([data](../data/2026-11-11_c615_draft_prefetch/README.md)), Q1345
  ([data](../data/2026-11-11_c738_verify_prepare/README.md)), Q1334
  ([data](../data/2026-11-11_c619_draft_graph_reuse/README.md)), Q1344
  ([data](../data/2026-11-11_c740_seq_rm_bound/README.md)), Q1350
  ([data](../data/2026-11-11_c741_server_yields/README.md)); timed together as Q1400.
- The eight kernels: Q1285 ([3-bit](../data/2026-11-11_c726_q3k_multitoken/README.md)), Q1286
  ([6-bit](../data/2026-11-11_c727_q6k_readonce/README.md)), Q1287
  ([tiles](../data/2026-11-11_c728_readonce_k6144/README.md)), Q1288
  ([five to eight tokens](../data/2026-11-11_c729_readonce_5to8/README.md)), Q1325
  ([16-bit, exact](../data/2026-11-11_c750_bf16_exact/README.md)), Q1330
  ([32-bit](../data/2026-11-11_c754_f32_mmvf/README.md)), Q1333
  ([one launch](../data/2026-11-11_c757_shexp_fused/README.md)), Q1348
  ([one token](../data/2026-11-11_c767_q5k_one_row/README.md)); their rig runs, Q1391.
- Registry: [quants/experiments.md](../../quants/experiments.md).
