# 14. Reading prompts: the wait before the first word

> **In short**
> - A third of an agent task is reading. Every turn the model must read its new input before the first word comes
>   back, and that wait is what a user feels.
> - On v3, 30.72 % of the reading time went to saving checkpoints: each one stopped the nine-card pipeline and copied
>   112.6 MiB to the host.
> - Keeping the checkpoints on the cards, and then saving them without stopping the pipeline, took most of that
>   away: a replay of the agent check's prompts went from 509.16 s to 338.96 s a pass. While reading a long prompt
>   only 1.42 of the nine cards were busy on average; recording each batch as one launch took 10.2 % off it.
> - The draft must read every prompt too, which costs 28.4 % of reading speed on an empty cache and 41.3 % deep into
>   a conversation. A slimmer catch-up pass took a quarter of that cost away.
> - One loss is still open: on the stacked line, turns with 1,024 or more new tokens read slower than on v3.

Facts in this chapter are from 2026-11-11. The model is Qwen3.8-Flash-Next on the release preset with its draft.
Reading a prompt is the *prompt pass* of [chapter 2](02-how-an-llm-makes-text.md): many tokens share one read of
the weights. A *micro-batch* is the number of prompt tokens the engine sends through the cards in one go, 512 on the
preset.

## 14.1 Why reading matters to an agent

A coding agent's turn is short to write and long to read. It writes a tool call, a few dozen tokens. It gets back a
file or a test log, hundreds or thousands of tokens, and has to read all of it before it can write again.

On the agent check with the stacked line of [chapter 15](15-toward-v4.md), a task takes 143 s on average: 74 s of
writing, 47 s of reading, 22 s of everything else. Reading is a third. And it sits in few turns:

| New tokens to read in a turn | Turns | Reading time, s |
|---|---:|---:|
| under 512 | 42 | 119.7 |
| 512 to 1,023 | 9 | 94.9 |
| 1,024 to 1,999 | 8 | 135.7 |
| 2,000 and more | 7 | 216.7 |

Fifteen of the 66 turns hold 62 % of the reading time. Those are the turns where the user stares at an empty line:
the median wait for a first word was 4.55 s, and one turn in ten waited 24.48 s or more.

## 14.2 A pipeline that is fast only when full

When the model writes, one card works at a time ([chapter 5](05-nine-cards.md)). Reading is different. The prompt is
cut into micro-batches, and several can be in flight at once: while card 3 works on the first, card 1 can already
start the third. It is an assembly line, and it is the only time many cards work together.

An assembly line has a fill time and a drain time. One 512-token micro-batch needs seconds to cross all nine cards
(est. 7.6 s when I first measured it, on 2026-11-04). A long prompt keeps the line full and reads fast. A short prompt is one micro-batch that crosses alone, with
eight cards idle around it. And anything that makes the engine wait for the line to empty pays the drain time
again, then the fill time again.

Most of this chapter is about things that emptied the line without need.

## 14.3 Story: the checkpoint tax

**The setup.** Most of Flash-Next's layers keep a running state that cannot be rewound
([chapter 2](02-how-an-llm-makes-text.md)). So the server saves a copy of that state, a *checkpoint*, twice in
every prompt: 516 and 4 tokens before its end. If the next turn's text differs near the end, the server restores a
checkpoint and re-reads a little, not everything.

**What it cost.** To save a checkpoint the server cut the prompt at that point, waited for all nine cards to finish,
and copied 112.6 MiB from the cards to host memory over the slow links. One copy took 641.5 ms. The wait for the
line to empty cost as much again.

**What I did.** No code. I replayed the agent check's 62 prompts, 36,793 new tokens, against settings the release
build already had: the checkpoints as shipped, none at all, and the newest four kept in card memory.

**What happened.**

| Checkpoints | Reading the 62 prompts, s a pass | Against the release |
|---|---|---:|
| As released: four, in host memory | 509.16 and 499.93 | |
| None (a diagnostic: any edit then re-reads the whole prompt) | 349.57 | −30.72 % |
| Four, in card memory | 437.54 and 416.17 | −15.40 % |

Checkpoints cost 30.72 % of all reading time. Keeping them on the cards gave half of that back: a copy there takes
17.6 ms. It also takes the largest buffers out of the 4 GB host: on the stacked line of
[chapter 15](15-toward-v4.md), which keeps them on the cards, the host's free memory at its lowest was 415 MiB in the
agent check, against 66 MiB on v3.

**Why.** The switch had been in the build since v3 and had never been timed. The expensive part of a checkpoint was
never the data. It was moving it over a 189 MB/s link, and stopping an assembly line to do so.

## 14.4 Story: a checkpoint without stopping the line

**The idea.** With the copies on the cards, the stops were left: 18.10 % of reading time. But nothing forces a stop.
A checkpoint only needs each card's state as it is right after the checkpoint's token. Each card can save that
itself, at the right moment, while the micro-batches behind keep moving.

**What I did.** A switch that takes checkpoints inside the running call. Each card copies its state into its own
store as the checkpoint's token passes. On a CPU the stored state is byte for byte what the stop-and-copy way
stores, on three models, and the server's replies are the same.

**What happened.** On the rig, with the checkpoints already on the cards: 338.96 and 331.09 s a pass against 416.56
and 410.60, 18.99 % less. Every reply, every checkpoint and every restore matched. 85 of the 124 checkpoints in a
pass were taken without a stop; the others fall where a call ends anyway.

From the release's 509.16 s to 338.96 s, the same prompts read in two thirds of the time (two runs a day apart, the
second on the rebased build).

**Why.** "Stop the world, then snapshot" is the simple design, and on one GPU it costs nothing. On a nine-stage
pipeline over slow links, the stop is the cost. A snapshot taken stage by stage, as the marked item passes, is the
same data.

## 14.5 Story: small batches at the edges

**The idea.** A prompt's last micro-batch often runs alone: the engine keeps the final few tokens apart for the
checkpoints, so a full 512-token batch crosses nine cards with nothing behind it. If batches near the start and end
of a call are small, the line fills and drains in short steps. I call it a *taper*.

**What happened.** On the same replay the taper took the release from 509.16 and 499.93 s to 434.46 and 430.50,
14.28 % less, and with the checkpoints on the cards to 358.94 and 349.97. With the in-call checkpoints it still
helped, 321.40 and 321.09 s against 338.96 and 331.09.

**The catch.** Those are totals over 62 prompts. By size the picture is mixed: for turns of 1,024 to 1,999 new
tokens the taper was slower, 126.93 s against 112.12 over a pass. Section 14.9 comes back to this.

**Why.** Small batches fill a line quickly and read slowly. Which effect wins depends on the size of the prompt, and
a rule that is right for the very short and the very long can be wrong in the middle.

## 14.6 Story: where the cards idle

**The idea.** If the line were full while reading, several cards would be busy at any moment. Count them.

**What I did.** One fresh prompt of 4,097 tokens on the release preset, with the profiler recording each card's busy
time and the host's waits.

**What happened.**

| A 4,097-token prompt, 29,983.0 ms | ms | Share |
|---|---:|---:|
| The busiest single card's GPU time | 5,631.0 | 18.8 % |
| The host waiting for the cards to empty, after each of the prompt's four calls | 13,760.4 | 45.9 % |
| The host blocked inside a call, handing over its third micro-batch | 8,851.0 | 29.5 % |

On average 1.42 of the nine cards were busy. The line was almost never full.

The second wait had a recognizable shape. Each micro-batch reached a card as 109 to 327 separate small launches.
A card's driver holds only so many launches in its queue. Most likely the queue was full: the host stood still
until the card had worked some launches off, and so could not feed the other cards either. The next story
confirmed it.

**Why.** A producer blocked on one full queue starves every other consumer. The cards were idle because the host
was stuck talking to one of them.

## 14.7 Story: read the other fork first

**The idea.** Someone had met this before. A public fork of llama.cpp for a six-card machine with starved links
reports prompt reading going from 859 to 2,500 tok/s. Before writing anything, read it.

**What I did.** I read its 47 commits against my tree. Of the causes it removes, one applied here: the full launch
queue. Its fix records a prompt micro-batch as a CUDA graph, a recorded launch sequence
([chapter 6](06-gpu-code.md)), so a whole batch reaches a card as a few launches instead of hundreds.

I ported that, and tested the port's decision logic in a small model of the whole capture machinery before it saw
a GPU. The port could, in rare orders of events, replay a recording made for other inputs: in 2,000 random runs it
did so in 711. My version replays only recordings made from the current inputs, and did so in none.

**What happened.** On the rig the 4,097-token prompt took 24,932.3 and 24,305.5 ms against 27,783.8 and 27,050.6,
10.2 % less. The 15 blocked hand-overs were gone, and the replies were the same.

**What is left in it.** Every micro-batch is recorded and none is replayed: 621 recordings and 0 replays in the
run, because something about each batch differs from the last (the cache has grown, the inputs sit in another
buffer). Recording costs host time. Making batches replayable is on the list.

**Why.** Reading the other fork's commits found the cause, the fix and a flaw in the fix, without a minute of rig
time. The alternative was to rediscover all three on the rig.

## 14.8 Story: the draft reads too

**The setup.** The draft that guesses tokens ahead ([chapter 8](08-guessing-ahead.md)) is a small model with its own
cache. To guess after a prompt, it must have seen the prompt. So for every prompt token the main model reads, the
draft runs a *catch-up* pass on the last card.

**What it cost.** With the draft on, the rig read prompts 28.4 % slower on an empty cache and 41.3 % slower on
32,188 cached tokens. Over a load's 50,124 prompt tokens, 217.6 of 567.6 s were the draft's.

**Three attempts.**

| Change | Reading speed, tok/s | What it took off the draft's cost |
|---|---|---:|
| The draft reads in batches of 512 instead of 128 | 91.8 and 91.5, against 88.0 and 88.7 | 9.3 % |
| A catch-up pass that only writes the draft's caches | 101.4 and 101.8, against 92.2 and 92.6 | 24.9 % |
| No draft at all, for scale | 145.4 and 145.2 | |

The second is the interesting one. A catch-up pass does not need the draft's guess. It only needs to leave the
draft's cache in the right state. Cutting the pass down to that left 531 of its 1,458 operations, with the cache
and every later guess byte for byte the same.

The first row came with a surprise. The preset had kept the draft's batches small to protect the 4 GB host. The
larger batch left more host memory free, 263 and 238 MiB at the lowest against 71 and 81. The slimmer pass went the
other way: 93 and 66 MiB at the lowest, against 154 and 249. That is close to the floor at which the watchdog stops
the server, so it is watched in every run that uses it.

**Why.** Ask what a step must leave behind, not what it computes. A pass kept only for its side effect can drop
everything else.

## 14.9 The loss that is still open

Put together on the agent check, the changes of this chapter and the last two read short turns much faster than v3
and long turns slower:

| New tokens in a turn | v3, tok/s | The stacked line, tok/s |
|---|---:|---:|
| under 512 | 39.4 | 64.4 |
| 512 to 1,023 | 61.3 | 71.2 |
| 1,024 to 1,999 | 90.5 | 80.1 |
| 2,000 and more | 93.3 | 76.4 |

The median wait for a first word fell from 6.52 s to 4.55. The wait that one turn in ten exceeds rose from 16.17 s
to 24.48.

The suspect was the taper of section 14.5, which loses in the same size range in the replay. On this line it is
not the cause. With the taper off, the replay reads turns of 1,024 to 1,999 new tokens at 79.3 and 81.1 tok/s in two
passes, the agent check's speed. The full taper loses from 512 new tokens up, because of its small batches at the
start of a call. Without that head rule, `short=600:64` reads a pass in 469.17 s against 492.68 and 480.70 with the
taper off, and loses in no size range. So the line takes the short rule only.

The loss sits somewhere else. The night's build with only the checkpoint switches read the same turns in 112.12 s a
pass against 192.99 s, and a fresh 2,049-token prompt in 12.27, 12.38, 12.41 and 12.14 s against 23.36, 22.94, 23.32
and 22.97. The server logs show the difference. On the stacked line every request rebuilds its graph's memory plan
partway through, 1,547 times in one load of 373 requests, and the night's build never did. Each rebuild makes the
host wait for all nine cards, so the pipeline of section 14.2 empties there. Which switch causes it is the next thing
to find (Q1405).

One more gap shows in these numbers. The switches of this chapter were timed on a fresh 4,097-token prompt or on
8,128-token chunks, never on the mix of sizes an agent really sends. The next runs replay an agent session's own
prompts and report by size.

## 14.10 Written, checked, not yet timed

| Change | The problem it attacks | Estimate |
|---|---|---|
| The draft catches up on one call while the main model already reads the next | the wait for the cards to empty after each call, 45.9 % of the long prompt | not priced |
| The lookup table's rows fetched ahead by a helper thread | a fresh 512-token batch held the host 291.3 ms before the first card started, 198.3 ms of it fetching inputs; 11.4 ms on a prompt read before | up to 991.5 ms of a 4,097-token prompt |
| The draft's own token table on the last card | each catch-up pass moved 1.31 MB from card 4 through the host to card 8, where 128 token numbers would do | 0.27 s of a 4,097-token prompt |
| Prompt batches through the sparse attention kernel, with their own rule | deep in a conversation, prompt reading slows like writing did: 70.2 and 70.0 tok/s on 32,000 cached tokens | 131.3 and 131.9 tok/s in a first run |

The last row has a measurement behind it, with a flaw. Sending prompt batches through the sparse kernel of
[chapter 12](12-long-conversations.md) read a load's three chunks at 131.1 and 131.3 tok/s against 89.5 and 89.4,
and no longer slowed with depth. But the setting that did it also sent writing through that kernel, and a step at
the start of a conversation took 282.35 and 282.53 ms against 113.12 and 113.14. So prompts got their own rule,
which leaves writing alone. On newer cards its output is as close to the reference as the standard path's: a KL
divergence of 0.026896 at 32,768 tokens against 0.035516 for the standard path at its usual batch size. Its rig
timing is queued.

## What to remember

- For an agent, reading is a third of the time and most of the waiting. Measure it by the size of the turn.
- A pipeline over several cards is fast only while it is full. Count how many cards are busy; here it was 1.42 of 9.
- Saving state does not require stopping. Snapshot each stage as the marked item passes.
- Keep big buffers off a small host. Checkpoints in card memory were faster, and the line that uses them left the
  host 349 MiB more at its lowest.
- A producer blocked on one full queue starves the rest. One launch per batch, not hundreds, unblocked it.
- Read what others built for a similar machine before writing. Then test the port: this one had a flaw.
- A helper that must follow the main model pays for everything the main model reads. Cut its pass to what it must
  leave behind.
- A total over many prompts can hide a loss in one size range.

## Where the details are

- The time split of an agent task and reading by turn size: Q1322
  ([data](../data/2026-11-11_r90_v4_vs_v3/README.md)); the earlier note on prompt reading,
  [flash_next_prefill.md](../research/notes/flash_next_prefill.md).
- Checkpoints and the taper on the replay: Q1155 ([data](../data/2026-11-10_prompt_lane/README.md)); card-memory
  checkpoints (`--ctx-checkpoints-vram 4`), Q440; the taper (`--ubatch-taper on`), Q771
  ([data](../data/2026-11-04_prompt_taper/README.md)); its loss by size, Q1235.
- Checkpoints inside the call (`--ctx-checkpoints-incall`): Q770 and Q1306
  ([code](../data/2026-11-11_c217_incall_checkpoints/README.md), [rig](../data/2026-11-11_r85_c217_p106/README.md)),
  on newer cards ([data](../data/2026-11-11_fn72_c217_ptx/README.md)).
- The busy cards: Q1156 ([data](../data/2026-11-10_prompt_timeline/README.md)); one prompt batch by batch, Q1295
  ([data](../data/2026-11-11_r81_ubatch_trace/README.md)).
- The other fork: [multigpu_fork_read.md](../research/notes/multigpu_fork_read.md). Prompt batches as CUDA graphs
  (`GGML_CUDA_GRAPHS_FORCE=1`): Q1240 and Q1297 ([code](../data/2026-11-11_c702_prompt_graphs/README.md),
  [rig](../data/2026-11-11_r82_c702_p106/README.md)), on newer cards
  ([data](../data/2026-11-11_fn71_c702_ptx/README.md), [data](../data/2026-11-11_fn78_c702_ptx48/README.md));
  replaying instead of recording, Q1368.
- The draft's cost in reading: Q1153 and Q1215 in [2026-11-10_deep_lane](../data/2026-11-10_deep_lane/README.md);
  the slimmer catch-up pass (`LLAMA_MTP_KV_ONLY=1`), Q1105 and Q1278
  ([code](../data/2026-11-11_c620_draft_kv_catchup/README.md), [rig](../data/2026-11-11_r76_c620_p106/README.md)).
- Not yet timed: the catch-up during the next call, Q1304
  ([data](../data/2026-11-11_c703_catchup_overlap/README.md)); the table rows ahead, Q1351
  ([data](../data/2026-11-11_c761_ngram_prefetch/README.md)); the draft's table on the last card, Q1349
  ([data](../data/2026-11-11_c762_draft_rows/README.md)); prompts through the sparse kernel, Q1214 in the deep lane's
  folder, the rule Q1272 ([data](../data/2026-11-11_c705_sparse_prompt_rule/README.md)), its quality Q1248
  ([data](../data/2026-11-11_fn64_sparse_prompt_ptx/README.md)). The prompt switches on an agent session's own
  prompts: Q1401.
- Registry: [quants/experiments.md](../../quants/experiments.md).
