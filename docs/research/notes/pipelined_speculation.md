# Pipelined speculation in the fork: executor and correctness (Q87)

On a layer split only one card works at a time: token t passes card 1, then card 2, and so on, and the last card
samples token t+1. Pipelined speculation feeds a guess for token t+1 into card 1 as soon as card 1 has finished token
t, a guess for t+2 one stage later, and so on. Every guess is an ordinary one-token step, so the fork's single-token
Pascal kernels and CUDA graphs run unchanged, and there is no multi-token verify step (which costs 2.2–3.6 plain tokens
on these cards, Q19). When the last card samples the real token t+1 it is compared with the guess already in flight:
same token, the guess's work is the right work and keeps going; different token, the guesses are dropped, their KV
cells are released and the sampled token enters card 1.

This note describes the executor built into the project's llama.cpp fork (branch `pascal-p106-v2-pipespec`), why it is
correct, how that was checked, and how to time it on the rig. The gain is estimated in
[breakthrough candidates](breakthrough_candidates.md) (candidate 1) and simulated on Flint's own transcripts in
[ngram_replay](../../../bench/ngram_replay/README.md#pipelined-speculation-q87).

## Summary

- **What exists.** A pipelined decoding API in the fork (`llama_pipe_init`, `llama_pipe_submit`,
  `llama_pipe_get_logits`, `llama_pipe_get_sampled`, `llama_pipe_drain`; branch `pascal-p106-v2-pipespec`, commit
  a217f42c8) and a tool, `llama-pipespec`, that runs plain decoding or pipelined decoding with up to D guesses in flight
  for one sequence. Guessers: the copy lookup, llama.cpp's `ngram-simple`, any llama.cpp drafter (`ngram-map-k`,
  `ngram-mod`, …), and an oracle that replays a reference run with a chosen share of wrong guesses. A job list runs many
  configurations with one model load.
- **Correctness.** 89 pipelined runs of Flint (UD-Q4_K_XL), 21,688 generated tokens, 18,390 guesses of which 2,401 were
  wrong and 5,208 tokens were dropped: every run gave plain decoding's tokens, bit-identical logits for every token, the
  same KV state after generation and the same logits for probe tokens decoded afterwards. Layouts: the CPU alone, two
  RTX 3090s (two stages), four virtual devices on them (four stages, every hop between the two GPUs), and a sliding-
  window cache small enough to wrap; greedy with host or GPU sampling, and temperature 1 with a fixed seed. Every
  process finished within its timeout. Speed was not measured (the machine used for these checks is not the rig).
- **Wrong guesses are dropped, not withdrawn.** All stages of a token are queued when it is submitted, and a wrong
  guess simply runs to the end while its result is ignored. Because every guess runs ahead of the corrected token on
  every card, spaced by at least one stage time, this costs nothing while D ≤ S_eff − 1 (S_eff = sum of stage times /
  slowest stage time: 4.2 for six cards with the output head on the last one, 5.9 and 8.5 for the two nine-card splits
  below). An event model shows dropping and ideal per-stage cancellation give the same speed up to that depth; beyond
  it, dropping loses (six cards, every guess wrong: 0.90× at D = 4, 0.75× at D = 5).
- **The output is plain decoding's output.** Each token is sampled once, in order, from logits computed by the same
  graph, kernels and KV cell layout as plain decoding, with the same sampler state, so greedy and seeded sampling give
  the same tokens, with bit-identical logits.
- **Several samples of one request share the same cards.** A race of k samples (Q93: keep the first to finish its
  reasoning) runs one token of each sample in flight; the cards hold about S_eff tokens, so guesses and race samples
  share one budget: race samples while Flint reasons, where guesses are rarely right, and all of it as guesses for the
  winner after `</think>`. 31 pipelined races of 3–4 samples (8,164 sample tokens; winners 0, 1 and 2) were
  bit-identical to the same races decoded plainly, round robin.
- **Several conversations run through the same executor (streams mode, Q104).** One KV stream and one slot range per
  conversation, prompts entering as pipeline chunks, sampling per sequence on the device with position-keyed draws, and
  guesses only in what the conversations leave of the budget. 48 conversation runs (three conversations at a time; all at
  once, staggered arrivals, guesses in the slack, a budget of one item) matched a plain run of each conversation in tokens, logits
  or sampled ids, KV state and probe, with host sampling and with the temperature and greedy chains on the device.
- **Next:** time it on the rig ([plan](#measuring-it-on-the-rig)): the all-right oracle shows the
  pipeline's ceiling (event model: 2×/3×/4×/4.5× at D = 1/2/3/4+ on six cards), the all-wrong oracle the cost of wrong
  guesses, and the copy lookup the real gain per kind of prompt.

## How llama.cpp runs a layer split

- `llama_decode` builds (or reuses) one graph for the ubatch; `ggml_backend_sched` cuts it into splits, one per run of
  consecutive nodes on one device. `ggml_backend_sched_graph_compute_async` → `compute_splits` queues every split at
  once: for each split it copies the split's inputs to its device (activations from the previous card with
  `cudaMemcpyPeerAsync` on the source stream plus an event the destination stream waits on; user inputs such as
  positions, KQ masks and KV indices with a synchronous host-to-device copy), then launches the split's graph (a
  captured CUDA graph per split, keyed by the split's first node). The logits come back with an asynchronous copy into
  pinned memory, and `llama_get_logits` synchronizes.
- **Pipeline parallelism** (`n_copies` = 4 input copies per split, events per device and copy) lets the ubatches of one
  prompt batch overlap across cards. It does nothing for generation: with graph reuse, `process_ubatch` synchronizes the
  scheduler before writing inputs (the copy index only rotates when a graph is allocated, so a reused graph would
  overwrite inputs that the previous token still reads), and `ggml_backend_sched_synchronize` resets the next copy to 0
  on purpose, so that the CUDA graphs keep seeing the same input addresses.
- With a single copy (no events), `compute_splits` blocks the host on `ggml_backend_synchronize(split device)` before
  copying any split input. Two tokens can therefore never be in flight through one scheduler without changes.
- **Speculative decoding** verifies a draft as one multi-token ubatch, samples each row with
  `common_sampler_sample_and_accept_n` (sample-and-match), and removes the rejected positions with
  `llama_memory_seq_rm(mem, seq, n_past, -1)`.
- **KV cells.** `find_slot` takes the first usable cell from `head`: an empty cell, or one whose position is outside the
  sliding window of every token still to come (judged against the sequence's highest position + 1). Overwriting a
  sliding-window cell purges every older position of the sequence. `seq_rm` frees cells and moves `head` back to the
  lowest freed cell index. The sliding-window cache holds n_swa + n_ubatch cells, padded to 256.
- **Virtual devices.** `GGML_CUDA_DEVICES=N` (upstream) splits one GPU into N devices with their own streams. A layer
  split over them has real hops and real concurrency, which makes a single GPU usable for functional tests of
  multi-stage code.

## Prior work

- **PipeInfer** ([paper](https://arxiv.org/abs/2407.11798), [code](https://github.com/AutonomicPerfectionist/PipeInfer)),
  built on llama.cpp with MPI between nodes. The head node keeps launching speculative micro-batches (drafts of 1–4
  tokens) while no logits are waiting, plus one non-speculative one-token run per accepted token, so the worst case stays
  plain pipelined decoding. Verification is sample-and-match at the frontier. Cancellation travels backwards through the
  ranks (`GGML_MPI_CANCEL_RUN`); a stage skips a cancelled run it has not started, but still allocates its KV cells and
  forwards an empty activation, and the head still receives and discards its logits. KV consistency comes from "KV
  multibuffering": every run gets its own block of sequence ids, accepted ranges are copied back to sequence 0 with
  `seq_cp`, and every KV edit is itself a transaction sent down the rank chain, so it reaches each stage in order with
  the runs.
- **SPD** ([paper](https://arxiv.org/abs/2605.30852), [code](https://github.com/yuyijiong/speculative_pipeline_decoding)):
  one token per stage, in lock-step cycles broadcast by rank 0. Attention is exact because each stage's KV shard already
  holds every older token. Rank 0 verifies one token per cycle (argmax match, or rejection sampling against a trained
  drafter's distribution); a miss sends a DISCARD cycle in which every stage crops its KV to the rejected position and
  rewinds recurrent layers from a ring of saved states, then the pipeline refills. Fixed per-cycle costs (verify,
  transfers, control) are why 16 stages fall back to ~1.5×.
- **SpecPipe** ([paper](https://arxiv.org/abs/2504.04104)) feeds one level of a 64-wide draft tree per pipeline step and
  prunes dead branches at every stage after each verification; **FlowSpec** ([paper](https://arxiv.org/abs/2507.02620))
  streams sorted EAGLE-2 tree segments and keeps going while the accepted prefix stays in the tree. Both need cheap
  multi-token steps, which these cards don't have.

What carries over to a one-token chain on one machine: each card runs its queue strictly in order, so a token reaches
card j only after every older token has written its KV there (exact attention without masks or sequence copies); each
in-flight token needs private inputs and activation buffers; a single sequence truncated by position replaces
multibuffering; sampling happens only at the frontier and advances the sampler only for accepted tokens.

## Design

### Slots

`llama_pipe_init(ctx, n)` creates n slots. A slot is one in-flight one-token evaluation with its own
`ggml_backend_sched` (one input copy), its own graph (built once and reused while its shapes hold, so it has its own
CUDA graphs: the first-node keys differ between slots), compute buffers reserved for a one-token graph over the whole
cache, an event and a pinned buffer for its logits. Slots share only the weights and the KV cache. A slot is reused only
after its previous token's output event has fired, and that event follows the token's last split, which follows every
earlier split of that token (each hop is queued after the whole split on the source card). So a slot's inputs and
buffers are never overwritten while in use, and the scheduler can skip its host-side waits: the new scheduler flag
`ggml_backend_sched_set_caller_sync(sched, true)` does exactly that. The hops between cards stay asynchronous.

`llama_pipe_submit(ctx, slot, token, pos, seq)` does what `llama_decode` does for one token (memory slot, graph,
inputs, queued splits), then queues the logits copy and records the slot's event; it returns without waiting.
`llama_pipe_get_logits(ctx, slot)` waits for that event. With a backend sampler (`llama_set_sampler`, as llama-server's
`-bs` uses) the graph ends in the sampler and only the token id comes back: `llama_pipe_get_sampled(ctx, slot)`.
`llama_decode` drains all slots first.

### The loop (one sequence)

The window holds the newest committed token, in flight, and up to D guesses behind it.
1. Top up: while fewer than D guesses are in flight and the guesser has one, submit the next guess at the next position.
2. Wait for the oldest token's logits, sample the next token (the same sampler chain plain decoding uses), commit it.
3. If the next entry in the window is a guess for that same token, it becomes the newest committed token. Otherwise
   remove every position from the first guess on (`llama_memory_seq_rm(mem, seq, p, -1)`), forget the window and
   submit the sampled token at p.

### Dropping instead of withdrawing

CUDA work cannot be taken back once queued, and the executor queues all stages of a token at submit time. A wrong guess
therefore runs to the end, and only its result is ignored. On a one-token chain this costs no time as long as the guesses
ahead of the corrected token never hold a card it needs: guesses enter card 1 at most one per stage time, so they move
through the cards ahead of the corrected token, and the corrected token waits only if more guesses are in flight than
the pipeline can hold, D > S_eff − 1. PipeInfer's per-stage skipping, or a host that launches stage j of a token only
once stage j − 1 has finished and the token is still valid, would save card time (power) but no wall time below that
depth.

Event model of the cards ([drop_vs_cancel.py](../../../bench/pipespec/drop_vs_cancel.py); stage times of
[ngram_replay](../../../bench/ngram_replay/README.md#model) with 0.3 ms per hop and 1.5 ms of host time per sample;
i.i.d. guesses right with probability p; speed-up over plain decoding):

| Cards (layers per card) | S_eff | D | Every guess right | p = 0.7 dropped (cancelled) | Every guess wrong |
|---|---|---|---|---|---|
| 6 (9/8/8/8/8/8) | 4.2 | 1 | 2.00× | 1.62× (1.62×) | 1.00× |
| | | 2 | 3.00× | 1.95× (1.95×) | 1.00× |
| | | 3 | 4.00× | 2.13× (2.13×) | 1.00× |
| | | 4 | 4.48× | 2.02× (2.18×) | 0.90× |
| | | 5 | 4.48× | 1.78× (2.18×) | 0.75× |
| 9 (6/6/6/6/5/5/5/5/5) | 5.9 | 5 | 5.99× | 2.40× (2.40×) | 1.00× |
| | | 8 | 6.20× | 1.81× (2.43×) | 0.69× |
| 9 (6 × 7/5/2) | 8.5 | 5 | 5.99× | 2.53× (2.54×) | 1.00× |
| | | 8 | 8.87× | 2.59× (2.68×) | 0.99× |

The model has no host contention and no power limit; the rig measures both.

### KV cache

- **Order.** Every layer's KV cache lives on one card, and every card runs its queue in submission order. A token
  therefore writes its K/V on card j after every older token has written there and before any newer one reads it,
  whatever the other cards do. A dropped guess's writes land before those of the corrected token at the same position,
  which overwrites them.
- **Release.** `seq_rm` frees the dropped positions on the host at once. Masks and KV indices are host data copied at
  submit time, so tokens submitted later never see the dropped cells, while tokens already in flight keep their own
  copies.
- **Layout.** For bit-identical attention the corrected token must take the cell the dropped guess had, as plain
  decoding would. Two things in llama.cpp broke that, and the fork fixes both:
  - `seq_rm` moved `head` to the lowest freed index, which is wrong when the dropped positions wrap around the end of
    the cache. The fork moves `head` to the freed cell with the lowest position when the tail of a sequence is removed;
    the cell is the same as before in every other case.
  - The slot search restarted from cell 0 whenever `head` > used cells + 2 × tokens (commit `fbb1b24f3` fixes this).
    The dropped guesses' cells count as unused, so with D ≥ 4 of them and `head` on one of the last cells of a full
    cache, the corrected token went to cell 0. The search now keeps `head` when it points at a free cell.
- **Sliding window.** A guess judges sliding-window reuse against its own position, D positions ahead of the committed
  one. That is safe as long as the cell it overwrites is older than every window a re-submitted token needs. With the
  layout kept as plain decoding's ring, the cell at `head` always holds the oldest position. The window cache has
  n_ubatch spare cells, so `llama_pipe_init` requires n_ubatch ≥ number of slots for a sliding-window model; the default
  512 is far above it.
- **Recurrent layers** (the Gated DeltaNet layers of Qwen3.5/3.6, Ornith and Qwen3.8; fork branch
  `pascal-p106-v2-hybrid`, Q162) can't be truncated by position: a sequence has one state, and every token rewrites it.
  - **One state group per slot.** `llama_pipe_init` gives the recurrent memory one group of state rows per slot plus a
    stash group; plain decoding writes group 0. Each slot writes its tokens' states to its own group, at an offset
    that is a constant of the slot's graph, so graph reuse, CUDA graphs and the fused gated-delta-net state write stay
    as in plain decoding. A token reads its source state from whichever group holds it, through the `s_copy`
    indirection that llama.cpp already uses to pick each step's source state.
  - **Rollback.** Per sequence the memory records which position's state each group holds. Dropping a wrong guess
    (`seq_rm(seq, p, -1)`) moves the sequence back to the group that holds position p − 1; nothing is copied.
  - **What stays.** A slot is reused only after the executor has read its token, so the states of tokens in flight
    are never overwritten. The one older state a rollback may still need is the one before the oldest unread token (a
    wrong guess behind it, llama-server's slot leaving pipelined decoding, an executor's cleanup at the end of a
    request). If its slot is about to be reused, the context copies that state to the stash group first, on the same
    devices and ahead of the new evaluation. The executors' slot rotation never does that in steady state; the first
    pipelined token after a prompt does it once (slot 0 reuses group 0). `llama-pipe-recurrent-check` forces these
    copies and the rollbacks that read them.
  - **Memory.** A group is one sequence's state: ~63 MiB for Qwen3.6 (30 layers × 2 MiB of delta-net state and 96 KiB
    of convolution history). llama-server's 2 × (D + 1) slots plus the stash are 9 groups at D = 3, ~570 MiB spread
    over the cards that hold the recurrent layers (~63 MiB per card on nine cards).
  - Only the delta-net models (`qwen35`, `qwen35moe`, `qwen3next`) write their states this way; `llama_pipe_init`
    refuses other recurrent architectures, and upstream's rollback snapshots (`n_rs_seq` > 0) can't be combined with it.

### Guessers

| `--pipe-guess` | Source | History it sees |
|---|---|---|
| `copy` | the token after the longest earlier match of the history's suffix (up to 12 tokens, newest on ties), if at least `--pipe-copy-min` long | committed + in flight |
| `cont` | Q96: `copy` if it finds a match of at least `--pipe-copy-min` (4), else the most frequent continuation of the longest 1–8-token context in the request so far, plus the counts of a table of other outputs (`--pipe-table`) if given. With `--pipe-store FILE` (Q161, below): after the copy lookup, the request's continuations of 2-token or longer contexts, then the store, then the request's continuations of any context | committed + in flight (counts from committed tokens) |
| `ngram-simple` | llama.cpp's `common_ngram_simple_draft` (`--spec-ngram-simple-size-n/-m`) | committed + in flight |
| `spec` | any llama.cpp drafter (`--pipe-spec ngram-map-k`, `ngram-mod`, …) through `common_speculative`; one draft per committed state, consumed one guess at a time | committed only (stateful drafters must not see tokens that may be dropped) |
| `oracle` | a reference run's tokens, a share `--pipe-miss` made wrong on purpose | – |

### A datastore of Flint's own answers (Q161, simulated)

A third stage for `cont` that learns from Flint's past output: a suffix array over the generated tokens of many Flint
answers ([datastore.cpp](../../../bench/ngram_replay/datastore.cpp)). Its guess is the most frequent next token after
the longest context, up to 32 tokens, that occurs in the store, as SuffixDecoding (arXiv 2411.04975) scores
continuations by frequency. It runs on the host and takes no card time.

- **Data.** The answers generated for the staircase drafter (Q20): 11,949 answers, 14.5 M tokens, from Flint Q8_0 and
  G3 at T 1, top-k 20, top-p 0.95. An answer is left out when its prompt shares any 12-word span with a Q51 prompt's
  user or tool messages ([datastore_corpus.py](../../../bench/ngram_replay/datastore_corpus.py)). That removes the 352
  edit answers whose instructions repeat Q51's word for word and leaves 11,597 answers, 13.8 M tokens. The agent prompts
  of both sets define the same four tools. Stores of 3, 7 and 13.8 M tokens are nested shuffled subsets. Two more stores
  are for comparison: every answer (`all`), and the clean set without the agent and tool-call answers (`noagent`).
- **Replay.** Q51's 38 held-out transcripts go through pipeline_sim.py's guessers and pipeline model
  ([datastore_sim.py](../../../bench/ngram_replay/datastore_sim.py); a brute-force scan of a 600k-token store confirms
  every query). A plain longest-match guess everywhere loses to the request's own counts. Two rules make the store
  pay: use it only when the context has at least 2 tokens and its top continuation holds at least 30 % of its
  occurrences, and ask it after the request's own counts of contexts of 2 tokens or more.
- **The chain:** copy lookup ≥ 4, else the request's counts (context ≥ 2 tokens), else the store, else the request's
  counts (any context).

One token ahead on the true prefix (right guesses, %):

| Guesser | Chat | Code | Edits | Agent | All | Fresh text | Tool calls |
|---|---|---|---|---|---|---|---|
| `cont` today (copy ≥ 4, else request counts) | 19.6 | 33.3 | 60.2 | 29.4 | 39.2 | 19.0 | 20.3 |
| the chain, 3 M clean tokens | 23.4 | 36.7 | 62.4 | 37.5 | 42.5 | 23.7 | 53.8 |
| the chain, 13.8 M clean tokens | 24.5 | 38.0 | 62.9 | 38.7 | 43.5 | 25.0 | 54.7 |
| copy ≥ 4, else the store, else request counts (13.8 M) | 23.0 | 36.4 | 60.9 | 37.4 | 41.8 | 23.6 | 57.8 |
| bound: the better of the store's and the request's guess at every token (13.8 M) | 30.3 | 43.4 | 66.5 | 44.9 | 48.4 | 31.3 | 63.7 |

Speed-up over plain decoding, simulated, nine cards with 6/6/6/6/5/5/5/5/5 layers:

| Guesser | Depth | Chat | Code | Edits | Agent | All | File rewrites | Tool calls |
|---|---|---|---|---|---|---|---|---|
| `cont` today | 1 | 1.14 | 1.24 | 1.49 | 1.22 | 1.29 | 2.05 | 1.15 |
| the chain, 13.8 M | 1 | 1.19 | 1.30 | 1.53 | 1.31 | 1.34 | 2.06 | 1.46 |
| `cont` today | 2 | 1.18 | 1.32 | 1.73 | 1.28 | 1.40 | 3.03 | 1.18 |
| the chain, 13.8 M | 2 | 1.23 | 1.39 | 1.80 | 1.41 | 1.46 | 3.06 | 1.67 |
| `cont` today | 3 | 1.19 | 1.36 | 1.87 | 1.30 | 1.44 | 3.96 | 1.18 |
| the chain, 3 M | 3 | 1.24 | 1.41 | 1.94 | 1.42 | 1.50 | 4.00 | 1.72 |
| the chain, 7 M | 3 | 1.24 | 1.42 | 1.94 | 1.44 | 1.51 | 4.01 | 1.76 |
| the chain, 13.8 M | 3 | 1.25 | 1.43 | 1.95 | 1.45 | 1.52 | 4.02 | 1.74 |
| the chain, 13.0 M without agent and tool-call answers | 3 | 1.25 | 1.43 | 1.95 | 1.36 | 1.52 | 4.02 | 1.24 |
| `cont` today | 4 | 1.19 | 1.37 | 1.96 | 1.32 | 1.47 | 4.87 | 1.18 |
| the chain, 13.8 M | 4 | 1.25 | 1.45 | 2.05 | 1.46 | 1.55 | 4.97 | 1.78 |

With the head and two layers on the last card the picture is the same: at depth 3, chat goes from 1.20 to 1.26 and
all classes from 1.46 to 1.54.

- **It adds ~5 %, not 15–30 %.** Chat, code and edits gain 4–6 % at depths 2–4 and 3–5 % at depth 1, all classes
  1.44 → 1.52 at depth 3. File rewrites stay where the copy lookup puts them. On fresh text the chain is right 25 % of
  the time against 19 % for `cont`; the staircase drafter's coupled acceptance on fresh text was 0.46 at 8.6 M tokens
  of the same data (Q20).
- **More data barely helps.** At depth 3, all classes go 1.50, 1.51, 1.52 at 3, 7 and 13.8 M tokens; one-token
  accuracy rises ~0.5 points per doubling. Doubling to ~28 M tokens overnight would add an estimated ~0.01.
- **Tool calls gain from answers with the same tools.** 1.18 → 1.74 at depth 3, but 1.24 when the store holds no agent
  or tool-call answers. Q51's agent turns and the generated agent prompts share their four tools. A store of one agent
  harness's own past answers would see that gain; for a new tool set expect the smaller one.
- **Leaving Q51's instructions in changes little:** with every answer (`all`, 14.5 M), edits rise by 0.01–0.02.
- **Cost.** 8 bytes per stored token: 110 MB at 13.8 M tokens, 220 MB at 28 M. The rig's 4 GB of RAM holds it, and
  the store can be read with mmap. A lookup at 13.8 M tokens takes 84 suffix-array probes and scans 360 entries to
  count next tokens; a cache of 643 frequent contexts covers the large ranges. The chain asks the store for half the
  generated tokens. At about 90 ns per random memory access on the Celeron's single DDR4-2133 channel (an assumed
  figure), that is est. 30–50 µs per lookup and 15–40 µs per generated token, under 0.3 % of a token's time. The store
  is built offline on any PC (`datastore build`, prefix doubling) and copied to the rig.

**In the engine** (fork branch `pascal-p106-v2-q161`): `llama-server --pipe-store FILE` and `llama-pipespec --pipe-store
FILE` (job key `store=0` leaves it out of a job) add the store to `cont` in the chain above. `common_pipe_store`
(`common/pipe-store.cpp`) memory-maps the file: the 8 bytes `PIPESTO1`, the token count, the reversed tokens and the
suffix array (int32 each), as `datastore build` writes it. Pages come in as lookups touch them, so a 110 MB store costs
the rig's 4 GB of RAM at most its size, in page cache the kernel can drop. All of a server's slots share one store, and
the cache of large ranges takes a lock. The server's log line per request counts the store's guesses ("pipelined = …,
S from the store"). The store holds token ids, so it fits only models with Flint's vocabulary; with another model it
guesses wrong and changes nothing else. On the CPU the outputs stay identical (Checks, below).

### Several samples of one request (race)

Q93 found that on Flint, keeping whichever of k sampled attempts closes its reasoning (`</think>`) first loses no task
accuracy (+3.2 points at k = 5, with 56 % of a single attempt's reasoning tokens; k = 3 at 64 %). On a layer split those
k samples cost almost nothing extra, because the idle cards that guesses would use can carry the other samples instead:

- The prompt is decoded once in sequence 0 and given to sequences 1..k−1 with `llama_memory_seq_cp` (a unified KV cache:
  the prompt's cells get k sequence ids, nothing is copied). Each sample has its own sampler and seed.
- Each sample keeps one token in flight. Tokens are submitted round robin (the next token of a sample as soon as its
  previous one is sampled), so the cards work on different samples' tokens at once. While k ≤ S_eff each sample runs at
  the speed of plain decoding, and the race ends after min over the samples of their reasoning length, not the average.
- When a sample emits the stop text, the other samples' tokens still in flight finish and are ignored, their sequences
  are removed (`llama_memory_seq_rm(mem, s, -1, -1)`; the shared prompt cells keep the winner's id), and the winner
  continues alone, now with guesses.
- Guesses and race samples share one budget: about S_eff tokens in flight, one per slowest stage, of which the race takes
  one per sample and the guesses the rest (summed over the samples, guesses + samples ≤ S_eff keeps wrong guesses free).
  In fresh text, most of Flint's reasoning, a 6-token match precedes only 1–6 % of tokens
  ([ngram_replay](../../../bench/ngram_replay/README.md)), so the budget goes to samples there: k = 4 on six cards (S_eff 4.2), up to k = 5–8 on nine. After `</think>` the
  winner takes it all as guesses (D = S_eff − 1), where answers often copy code from the reasoning.
- The submission order of the pipelined race is the order of a plain round-robin race (one `llama_decode` per token,
  sample 0, 1, …, k−1, 0, …), so both put every token in the same KV cell and give bit-identical results; the extra
  in-flight tokens of the losers at the end are removed with their sequences. That is how the race is checked.
- `llama-pipespec --pipe-race k --pipe-race-stop '</think>'` (job keys `race`, `race_stop`) runs it; host sampling only,
  since sampling on the GPU would need a random draw per sequence and position.

### Several conversations (streams mode, Q104)

[Multi-conversation serving](multi_conversation_serving.md) settles how a rig serves several conversations at once: stagger
them through the cards, one token of each in flight, instead of batching them into one step. `--pipe-streams N` (job key
`streams`) does that:

- **One KV stream per conversation.** The context is split into N streams (`kv_unified = false`, `-np N`, `-c` = N × one
  conversation's context); conversation c is sequence c, so a token's attention reads only its own conversation's cells.
  On a unified cache (the race's mode) every one-token step reads every conversation's KV.
- **Slots bound to conversations.** Conversation c has its own range of one-token slots, so a slot's graph and CUDA
  graphs are reused for the same sequence (a graph is reusable only for the same sequence ids); prompt chunks share
  `--pipe-chunks` chunk slots (default 2).
- **Prompts as pipeline work.** A new conversation's prompt is submitted in chunks of `--pipe-chunk` tokens
  (`llama_pipe_submit_tokens` on a slot reserved for a chunk) and runs between the other conversations' tokens.
  `llama_decode` would first drain the pipeline and stop every conversation for the whole prefill. A plain run with the
  same chunk size computes the same numbers, which is what the check compares against.
- **Several chunks in flight (Q155).** Up to `--pipe-chunks` K chunks run at once (job key `chunks`), prompts in order
  of arrival, several of one prompt: each card runs a prompt's chunks in submission order, so a chunk's attention on a
  card finds the KV the previous chunk wrote there. With one chunk per prompt in flight, a prompt occupies one card at a
  time. `--pipe-chunks-gen` G (job key `chunks_gen`) caps the chunks while any conversation generates.
- **Chunks below the tokens' priority (Q155).** On one stream per card a token queued behind a chunk waits for it on
  every card, so each chunk ahead of a token costs that token the chunk's whole pass through the cards (a 64-token
  chunk holds a P106 for ~0.1 s, est.). `LLAMA_PIPE_CHUNK_STREAM=1` gives the chunk slots a second backend per GPU: the
  context creates its own GPU backends with the device's greatest stream priority (the CUDA backend's init parameter
  `priority=high`) and the chunk backends with the default, the lowest. A card then schedules a waiting token's kernels
  ahead of a chunk's, so tokens overtake the chunks queued before them at kernel boundaries and the chunks fill the
  time the tokens leave. The two streams share only the weights and the KV cache, where a chunk writes only its own
  conversation's cells. The loop no longer waits for the oldest item: `llama_pipe_ready` (an event query,
  `ggml_backend_event_query` in the CUDA backend's procedure table) tells it which items are complete, and it handles
  the oldest complete item among each conversation's oldest (a conversation's own items still complete in order), else
  waits for the oldest token or guess, else the oldest chunk.
- **Sampling per sequence, on the device if wanted.** Each conversation has its own sampler and seed (seed + c). With
  `--pipe-backend-sample` the temperature chain runs on the last card: every slot's graph carries only its own
  sequence's chain, each token's random draw is keyed by (seed, position) with `llama_sampler_set_pos`, and each graph
  keeps its own sampler input tensors (the fork's `dist` sampler kept only those of the graph built last, so a second
  graph using the same sampler read a stale draw).
- **One budget.** A conversation's next token is submitted the moment its previous one is sampled; then prompt chunks;
  then guesses, copy-lookup guesses of any conversation before continuation guesses, only while fewer than
  `--pipe-budget` items are in flight. With the budget at S_eff, N ≥ S_eff conversations leave no room for guesses, and
  a conversation left alone gets the slack back. Section 4 of the serving note ranks the candidates for a free slot.
- **Not built:** the attention mask on the card (it is still copied from the host to every card for every token),
  races inside the streams mode (they need several sequences in one stream, or a stream copy per sample), and
  several conversations in llama-server (its `--pipe-depth` pipelines a slot only while it generates alone).

## Correctness

Method ([check_identity.sh](../../../bench/pipespec/check_identity.sh)): one `llama-pipespec` process decodes the prompt
and generates with plain decoding (one `llama_decode` per token), then repeats the same prompt pipelined: depth 0 (the
API without guesses), an oracle that replays the plain output with 0 %, 10 % or 50 % of its guesses made wrong on
purpose (wrong at the same positions in every run), and the copy lookup (match ≥ 2), `ngram-simple` (4/16) and
`ngram-map-k` (4/8) guessers, at depths 1–3. Each run is compared with the plain run on:
- **tokens**: the generated token ids;
- **logits**: a hash of the full logits row each token was sampled from (a hash of the token id with GPU sampling);
- **state**: the sequence's KV state after generation (`llama_state_seq_get_data`: cell positions and K/V rows);
- **probe**: eight more tokens decoded with `llama_decode` after generation; their logits read every KV cell that is
  still visible, so a cell left behind or overwritten by a dropped guess would change them.

Model: Flint UD-Q4_K_XL (49 layers, 35 of them with a 513-token sliding window, 150,144-token vocabulary), the
`rewrite.txt` prompt in Flint's chat template (750 tokens), flash attention, 4,096-token context.

| Layout | Sampling | Pipelined runs | Tokens | Guesses | Wrong | Dropped tokens | Result |
|---|---|---|---|---|---|---|---|
| CPU only (no overlap: the executor's logic) | greedy, host | 11 | 1,408 | 880 | 168 | 336 | all same |
| CPU only | T 1, top-k 20, top-p 0.95, seed 42 | 11 | 1,408 | 990 | 154 | 306 | all same |
| CPU only, `-ub 16`: 768-cell window cache, wraps during generation | greedy, host | 5 | 3,000 | 2,489 | 427 | 1,275 | all same |
| two RTX 3090s, two stages | greedy, host | 19 | 4,864 | 4,331 | 510 | 1,016 | all same |
| four virtual devices on the two GPUs, four stages | greedy, host | 19 | 4,864 | 4,331 | 510 | 1,016 | all same |
| four stages | greedy on the GPU | 11 | 2,816 | 2,333 | 296 | 589 | all same |
| four stages | T 1, top-k 20, top-p 0.95, seed 42 | 13 | 3,328 | 3,036 | 336 | 670 | all same |

"Guesses" counts submitted guesses, including those dropped before they were checked; "wrong" counts checked guesses
that did not match. With T 1 the oracle replays the sampled plain run and was right on 254 of 255 tokens at 0 % injected
misses: the random draws line up position by position. Every process finished within its timeout (no deadlock).

**Hybrid models (Q162, fork branch `pascal-p106-v2-hybrid`).** The same checks on Qwen3.6-35B-A3B Q4_0 (40 layers, 30 of
them Gated DeltaNet, 248,320-token vocabulary), the `rewrite.txt` prompt in its chat template (804 tokens) unless the row
says the chat prompt; the state hash then covers the recurrent states too. The CPU runs generate 64 tokens in a
2,048-token context, the GPU runs 128 tokens in 4,096, on a V100 through the compute_61 PTX with the layers split over virtual devices
(`GGML_CUDA_DEVICES`). "Device (checked)" compares the mask built on the device with the host mask at every step
(`LLAMA_GPU_KQ_MASK_CHECK=1`, which aborts on any difference).

| Layout | Mask | Sampling | Depths | Pipelined runs | Tokens | Guesses | Wrong | Dropped tokens | Result |
|---|---|---|---|---|---|---|---|---|---|
| CPU only | host | greedy | 1–4 | 47 | 3,008 | 2,495 | 399 | 899 | all same |
| CPU only | host | T 1, top-k 20, top-p 0.95, seed 42 | 1, 3 | 11 | 704 | 510 | 104 | 206 | all same |
| CPU only | device (checked) | greedy | 1–3 | 22 | 1,408 | 1,192 | 204 | 406 | all same |
| CPU only | device | greedy | 2, 4 | 11 | 704 | 611 | 104 | 307 | all same |
| CPU only | device | T 1, seed 42, chat prompt | 3 | 6 | 384 | 215 | 45 | 133 | all same |
| two virtual devices | host | greedy | 3 | 6 | 768 | 647 | 139 | 416 | all same |
| two virtual devices | device (checked) | greedy | 3 | 6 | 768 | 647 | 139 | 416 | all same |
| two virtual devices | device | greedy | 1–4 | 29 | 3,712 | 3,393 | 600 | 1,497 | all same |
| four virtual devices | device | greedy | 2, 4 | 15 | 1,920 | 1,846 | 300 | 898 | all same |
| two virtual devices | device | T 1, seed 42, chat prompt | 1, 3 | 15 | 1,920 | 1,244 | 254 | 506 | all same |
| two virtual devices | device | greedy on the GPU, fed tokens | 0 (fed), 1, 3 | 14 | 1,792 | 1,271 | 278 | 555 | all same but the fed run's state (below) |

- On the rewrite prompt the first 64 tokens are a stereotyped reasoning preamble, and T 1 sampled the greedy tokens; the
  chat-prompt rows sample other tokens.
- Plain decoding gives the same tokens, logits, state and probe with the host mask and with the device mask, on the
  CPU and on two virtual devices. The device-mask graphs hold one mask node per attention layer and no host mask input.
- The fed run (`pipe-d0-fed`) has the plain run's tokens, sampled ids and probe, and the same 85 MB of KV and recurrent
  data; its state differs only in 216 bytes of cell metadata. A fed token's batch carries a placeholder id (the real one
  is copied between the cards), the KV cache keeps each cell's token id, and the state stores it for M-RoPE models.
  Qwen3.6's graph never reads it, but models that do (n-gram embeddings) would compute with the placeholder, so
  `llama_pipe_submit_fed` now refuses every model whose KV cells keep a token (M-RoPE or PLE): fed tokens stay a
  plain-attention feature.
- `llama-pipe-recurrent-check` (three slots, every third token submitted to its predecessor's slot, wrong guesses, the
  token in flight dropped and submitted again or decoded with `llama_decode`), on the CPU (64 tokens: 21 states to the
  stash, 21 drops) and on two and four virtual devices (96 tokens: 32 and 32): tokens, logits, state and probe identical.

Races ([above](#several-samples-of-one-request-race)), checked with `RACE=K` against the same race decoded plainly,
round robin: four stages on the two GPUs, temperature 1, the winner continuing with the oracle, copy lookup,
`ngram-simple` and `ngram-map-k` guessers at depths 1–3:

| Race | Prompt, stop | Seeds | Pipelined runs | Winners | Sample tokens | Result |
|---|---|---|---|---|---|---|
| 3 samples | short question, `</think>` | 42 | 11 | 0 | 3,509 | all same |
| 4 samples | `rewrite.txt`, the first "." | 42 | 5 | 0 | 105 | all same |
| 4 samples | short question, `</think>` | 1, 2, 3 | 15 | 1, 0, 2 | 4,550 | all same |

"Same" covers the winner, every sample's tokens and logits, and the winner's tokens, logits, KV state and probe.

The `cont` guesser was checked the same way on the two GPUs (greedy, depths 1 and 3, without and with a continuation
table of Q51's transcripts), and a single sample through the race path (`race=1`) on four stages: all same.

Streams mode ([check_streams.sh](../../../bench/pipespec/check_streams.sh)), checked on the CPU of the 7-GPU machine
(its GPUs were busy) against one plain run per conversation, in that conversation's sequence, with its prompt, seed +
c and the same 64-token prompt chunks. Three conversations (`chat.txt`, 64 prompt tokens; `rewrite.txt`, 941 tokens in 15
chunks; `code.txt`, 67 tokens) in four jobs: all joining at once; joining 16 generated tokens apart, so that prompt
chunks run between the others' tokens; guesses (copy ≥ 4, else continuation counts, depth 2) in a budget of five items in
flight; and a budget of one item, which must allow no guess:

| Model | Sampling | Conversation runs | Tokens | Prompt chunks | Guesses (wrong) | Dropped tokens | Result |
|---|---|---|---|---|---|---|---|
| granite-3.1-3b-a800m (MoE, 2 GB) | host, T 1, seeds 42–44 | 12 | 1,536 | 72 | 266 (67) | 133 | all same |
| granite-3.1-3b-a800m | device, T 1 | 12 | 1,536 | 72 | 262 (60) | 117 | all same |
| granite-3.1-3b-a800m | device, greedy | 12 | 1,536 | 72 | 276 (70) | 135 | all same |
| Flint UD-Q4_K_XL (sliding-window layers) | device, T 1 | 12 | 1,152 | 56 | 188 (53) | 104 | all same |

The budget of one item allowed no guess in any run. Two bugs surfaced on the way and are fixed: the `dist` sampler's
input tensors (above), and the graph callback that pinned each layer's `norm` and last op to its card on the context's
scheduler instead of the slot's, so slot graphs missed that placement and filled the context scheduler's hash table.
Multi-GPU runs of the streams mode come with the rig.

**Several chunks in flight, items handled out of order (Q155).** Three more jobs in check_streams.sh: `multi` (all
conversations at once, four 32-token chunks in flight), `multi-arrive` (staggered arrivals) and `multi-gen` (four chunks
in flight, one while any conversation generates, guesses in the slack). On granite on the CPU: three conversations with
the temperature chain on the device, where the 941-token `rewrite.txt` prompt takes 30 chunks and has up to four of
them outstanding at once, first with the default order and then with `PIPESPEC_PICK=newest`, which handles the newest
complete item first. On the CPU every item is complete when it is submitted, so that order handles the conversations'
items in the reverse of submission order. Then four conversations with host sampling (`edit.txt` in 78 chunks), newest
first, with `LLAMA_PIPE_CHUNK_STREAM=1` set. All 70 conversation runs of the seven jobs equal their plain runs in tokens,
logits, KV state and probe logits, and `check_identity.sh` at depths 1 and 3 and `check_server.sh` at depth 3 still
match. On one V100 of the 7-GPU machine (the fork's v2 release candidate), the same check with four 32-token chunks in
flight gave 84 of 84 conversation runs equal to their plain runs on granite, with and without `LLAMA_PIPE_CHUNK_STREAM=1`
and with sampling on the device and on the host. `pipe_init` reported the second stream ("chunk streams 1"), and Flint G3
with the chunk streams gave 21 of 21. The timing side (decode priority on the rig's nine cards) is step 4d.

**The wrapped window cache at depth ≥ 4.** On the rig, K2h-r16 with the edit prompt (2,076 tokens) at depths 4–8
differed from plain decoding from about the 490th generated token, on six and on nine cards alike. By then the prompt and
the output fill the window cache (1,280 cells at the default `-ub 512`). Every other row matched. The cause was the slot
search's restart from cell 0 (KV cache, above): it fires only with at least four freed cells and `head` on one of the
last cells of a full cache, so depths 1–3 never trigger it.

[check_swa_wrap.sh](../../../bench/pipespec/check_swa_wrap.sh) makes it happen within the first tokens. `-ub 32` shrinks
the window cache to 768 cells, the 750-token prompt plus 128 generated tokens wrap it, and an oracle that is always wrong
drops D tokens at every step. On K2h-r16 on the CPU (12 pipelined runs at depths 4 and 8: oracle with all and half of
the guesses wrong, the copy lookup, `cont`, `ngram-simple`, `ngram-map-k`):

| Build | Runs equal to plain decoding (tokens, logits, KV state, probe) |
|---|---|
| before the fix (`f249104ab`) | 3 of 12 (oracle half wrong at depth 4, `ngram-map-k` at both depths) |
| with the fix (`fbb1b24f3`) | 12 of 12 |

The checks ran on two RTX 3090s of a 7-GPU machine used for quality work, where no speed is measured; the Pascal
kernels are exercised on the rig ([plan](#measuring-it-on-the-rig), step 1).

## llama-server (`--pipe-depth`)

The executor's loop runs inside llama-server on the fork's branch `pascal-p106-v2-pipeserve` (commit `cd49e507c`, off
`pascal-p106-v2-pipespec2`, so it includes the packed input uploads and the window-cache fix):

```
llama-server -m <gguf> -ngl 99 -fa on -c 262144 -bs --pipe-depth 3 [--pipe-guess cont|copy|none] [--pipe-copy-min 4] ...
```

- **When it runs.** A slot that generates alone runs pipelined: its newest sampled token and up to `--pipe-depth`
  guesses of the tokens after it are in flight in the context's pipe slots, one token is sampled per turn of the server
  loop, and the next guesses go in before the loop turns. As soon as another slot has work (a prompt, a second
  conversation), the slot drops what it has in flight, the newest sampled token included, and returns to batched
  decoding, which decodes that token again; when it generates alone again, it starts a new window.
- **Which requests.** The request's sampler chain must run entirely on the GPU (`-bs`, and nothing that needs the host:
  no grammar, so no tool calls, JSON schemas or `response_format`; no `n_probs`; no DRY, XTC, typical-p, top-n-sigma
  or mirostat). Penalties must be off, because they read the tokens before the one sampled and a guess's evaluation does
  not have those yet. The makers' settings (T 1, top-p 0.95, top-k 20, min-p 0, penalties off) qualify; every other
  request decodes batched as before. Speculative decoding with a draft model and multimodal models are not combined
  with it.
- **Exactness.** With `--pipe-depth` on, every sampled token's random draw is keyed by its position
  (`llama_sampler_set_pos`), on the pipelined path and on the batched one alike. The output of a request is therefore
  the same whatever the depth, the guesser, or how often the slot switches between the two paths. At temperature 0 it
  is the output of a server without `--pipe-depth`; at temperature 1 it has the same distribution, with other draws.
- **Stops.** Stop strings, the end-of-generation token, `n_predict` and the time limit are checked on sampled tokens in
  order, as for batched decoding; a stop drops the guesses in flight. A slot that is released while tokens are in
  flight (a stop, a cancelled request) keeps its newest sampled token, whose KV cells are being written, and drops
  the guesses, so the prompt cache stays consistent.
- **Checkpoints.** llama-server takes context checkpoints while it processes a prompt, never during generation, so
  they hold committed positions only.
- **Guesses** (`common_pipe_guesser`): the copy lookup (the token after the longest earlier match of the text's end, up
  to 12 tokens, at least `--pipe-copy-min`), and with `cont` the most frequent continuation of the last 1–8 tokens in
  the request when the copy lookup finds nothing. The copy lookup searches the newest 256 occurrences of the last token
  through an index, and the continuation counts cover the last 32,768 tokens, so an agent's long prompt costs little
  host time per guess.
- **Pipe slots:** 2 × (depth + 1). Each holds the compute buffers of a one-token graph on every card. A model with
  sliding-window layers needs `-ub` of at least that many (the default 512 is enough).
- **Floor** (`--pipe-floor N`, Q154): when the guesser has nothing, a filler token (the last one in the window) keeps N
  guesses in flight. It is almost always wrong, but idle cards drop to their 1,354 MHz base clock: with every guess
  wrong at depth 3, nine cards still ran 1.24× plain decoding, from the clocks alone. `llama-pipespec` takes
  `--pipe-floor` (job key `floor`) too.
- **KV steps** (Q153): each time n_kv grows by a step (256 cells), every pipe slot's graph is rebuilt and its CUDA
  graphs re-captured once. On the rig at depth 8 that is 18 tokens at about +9.7 ms each. The packed upload buffers
  now grow to twice their size, so they are rarely replaced (replacing them synchronized every card; fork commit
  `e3f497d6f`). `LLAMA_KV_PAD=1024` makes the steps four times rarer, at the cost of attention over up to 768 masked
  cells in the full-attention layers. Plain and pipelined decoding see the same n_kv, so the output stays identical.
- The server log prints, per request, the guesses and how many were right and wrong ("pipelined = …").

- **Tool requests** (with Q67): on `pascal-p106-v2-pipeserve-tools` (commit `ba83d3a1f`, the server branch plus Q67's
  commits) a request with a grammar keeps backend sampling, and its slot runs pipelined while the grammar cannot reject
  a token: before a lazy grammar's trigger, and inside the reasoning. The token that switches the grammar on (Flint's
  `<tool_call>`) ends the window: the guesses behind it go, and batched decoding takes over from that token, checking
  every token against the grammar on the host. The reasoning and the text before a call, most of an agent turn's tokens
  (§5 of [tool-request sampling](tool_request_sampling.md)), run pipelined; the call's arguments run batched. A finite
  reasoning budget keeps the request batched, since its forced tokens are applied only by `common_sampler_sample`.
  Q67's per-request counters cover the tokens sampled on the batched path.

### Checks

[check_server.sh](../../../bench/pipespec/check_server.sh) starts one server per configuration (`-bs`, `-np 1`). It
sends every prompt as one user message through `/v1/chat/completions`, at temperature 0 and at the makers' sampling with
fixed seeds, plus one streamed request and one with a stop string taken from the middle of an earlier output, so that
the stop arrives with guesses in flight. Each configuration must return, for every request, the same reasoning,
content and token count as the reference, `--pipe-depth 1 --pipe-guess none` (one token in flight, no guess). At
temperature 0 a server without `--pipe-depth` must agree too. It ran on the CPU of the 7-GPU machine, where the backend
sampler runs on the CPU backend:

| Model | Configurations | Requests each | Guesses per configuration (right, wrong) | Result |
|---|---|---|---|---|
| granite-3.1-3b-a800m, 256 tokens | depths 3, 5, 8 × `cont`, `copy` | 11 (chat, rewrite and edit prompts at T 0 and T 1 with two seeds; streamed; stop string) | `cont` 1,658 / 596, `copy` 1,470 / 35 | all identical |
| K2h-r16 (sliding-window layers, `-ub 32` so that the window cache wraps), 192 tokens | depths 4, 8, `cont` | 5 (chat and rewrite at T 0 and T 1; streamed) | 246 / 309 | all identical |

At temperature 0 the requests also equal a server without `--pipe-depth` (3 for granite, 2 for K2h-r16). On CUDA (one
V100, the fork's v2 release candidate, Flint UD-Q4_K_XL): one BFCL tool request per category at T 0 and T 1 with seed 1,
sent to a server at depth 3 with `cont` and the continuation store. Its replies matched those of a server with one
token in flight and no guess for 8 of 8 tasks at each temperature. The requests ran pipelined until their tool calls
started (4,551 guesses: 724 right, 1,276 wrong, 1,873 of them from the store).

Tool requests on `pascal-p106-v2-pipeserve-tools`, with
[grammar_sampling_check.py](../../../bench/tool_sampling/grammar_sampling_check.py): one BFCL task per category with its
tools (Flint's lazy tool grammar), K2h-r16 on the CPU, T 0 and T 1 with seed 1. The server at depth 4 with `cont`
returned what the server with one token in flight and no guess returned for 8 of 8 tasks at each temperature
(reasoning, content and calls). 1,816 of the 2,444 tokens ran pipelined (3,759 guesses: 415 right, 830 wrong); the
628 tokens of the calls ran batched, with 4 grammar rejections resampled from the fetched row.

The continuation store (`--pipe-store`, branch `pascal-p106-v2-q161`) on granite on the CPU, with a store built the same
way from granite's own answers to the four prompts at six other seeds (6,168 tokens; the Flint store's tokens mean
nothing in granite's vocabulary). The store answered often:

| Check | Runs | Store guesses | Result |
|---|---|---|---|
| check_server.sh, `cont` at depths 3 and 8 (chat, rewrite, edit at T 0 and T 1; streamed; stop string) | 22 requests | up to 179 per request | all identical to one token in flight without guesses; T 0 also to a server without `--pipe-depth` |
| check_identity.sh, chat and code prompts, T 1 seed 42, depths 1, 3, 8 (host sampling), and depth 3 with the temperature chain on the device | 41 runs (34 + 7) | 65–187 per `cont` run | all equal to plain decoding (tokens, logits, KV state, probe) |
| check_identity.sh, rewrite prompt, T 0 | 14 runs | 0 (the copy lookup covers it) | all equal |
| check_streams.sh, N = 3 with the temperature chain on the device | 21 conversation runs | 43–63 per chat or code conversation | all equal to their plain runs |

The Flint store file itself, replayed through datastore_sim.py on Q51's transcripts, gives the accuracy of the
simulation above to the last digit (43.5 % right, the store supplying 26.0 % of the guesses at 27.9 % right).

## Measuring it on the rig

The executor is exact; what the rig has to show is its speed. Everything below runs `llama-pipespec` from the build of
the fork's commit a217f42c8, `llama.cpp-20261024-a217f42c8-pascal-p106-v2-pipespec.tar.gz` (made with
`TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v2-pipespec`), and the scripts in
[bench/pipespec](../../../bench/pipespec/README.md). Commands run from a checkout of this repository on the machine that
reaches the rig as `ssh rig`; `$B` is expanded there. One run at a time, nothing else on the cards; each run loads its
model once (minutes from a USB stick) and prints a summary table when it ends.

**Builds and the window-cache fix.** Builds without fork commit `fbb1b24f3` stop being bit-identical at depth ≥ 4 on a
model with sliding-window layers once its window cache wraps ([Correctness](#correctness)): a217f42c8 here, and
`f249104ab`, the Q120/Q121 build of [host decode loop](host_decode_loop.md) §6. Their speeds stand, but their identity
column is wrong in those rows. `llama.cpp-20261024-74f7c7182-pascal-p106-v2-pipespec2` is the `f249104ab` build with
the fix. These are the rows that differed on the rig (K2h-r16, edit prompt), run again on it:

```
B=/opt/llama-builds/llama.cpp-20261024-74f7c7182-pascal-p106-v2-pipespec2/bin
ssh rig "setsid nohup env BIN=$B MODEL=<K2h-r16> TAG=k2h-6c-swafix GPUS=0,1,2,3,4,5 PROMPTS=edit DEPTHS='3 4 5' \
    ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/k2h-6c-swafix.out 2>&1 < /dev/null &"
# nine cards: GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 DEPTHS='3 4 5 6 8' TAG=k2h-9c-swafix
```

Every `same_as_plain` cell in `summary.tsv` must read `yes`.

**0. Install** (copy the tarball to the rig's `/opt/llama-builds/` first):

```
ssh rig 'cd /opt/llama-builds && tar xzf llama.cpp-20261024-a217f42c8-pascal-p106-v2-pipespec.tar.gz && ls llama.cpp-20261024-a217f42c8-pascal-p106-v2-pipespec/bin/llama-pipespec'
scp -r bench/pipespec rig:/tmp/
B=/opt/llama-builds/llama.cpp-20261024-a217f42c8-pascal-p106-v2-pipespec/bin
```

**1. Identity on the P106 kernels** (six cards, S4-L0, 128 tokens, ~5 min each): sampling on the host, then on the last
card. Every row must say `same`; stop and keep `run.log` if one does not.

```
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf \
    ARGS='-ngl 99 -fa on -c 4096' N=128 DEPTHS='1 3 5' MISSES='0 0.3' OUT=/tmp/pipespec/check-s4l0 \
    bash /tmp/pipespec/check_identity.sh"
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf \
    ARGS='-ngl 99 -fa on -c 4096 --pipe-backend-greedy' N=128 DEPTHS='1 3 5' MISSES='0 0.3' \
    OUT=/tmp/pipespec/check-s4l0-bg bash /tmp/pipespec/check_identity.sh"
```

**2. Plain against pipelined** ([rig_run.sh](../../../bench/pipespec/rig_run.sh), ~25 min per run plus the load). Per
run, 52 jobs of 512 tokens, greedy on the last card (`BG=1`, as llama-server's `-bs`):
- four prompts ([prompts/](../../../bench/pipespec/prompts/), in Flint's chat template): `chat` (an explanation, Q51's
  first chat prompt), `code` (an LRU cache with tests, Q51's first code prompt), `edit` (Q51's first edit instruction,
  type hints and docstrings, on a 90-line script of this repository) and `rewrite` (rename two variables in a 53-line
  script);
- per prompt: plain decoding (the reference, also written to `plain.jsonl`), the pipelined API without guesses (`d0`),
  the Q96 guesser (copy lookup ≥ 4, else the most frequent continuation of the longest 1–8-token context in the request
  so far; `cont`) at D = 1–5, and at D = 3 the copy lookup alone (≥ 2, ≥ 4) and `ngram-map-k` 4/8;
- on the chat prompt: the oracle with every guess right (the ceiling) and every guess wrong (the cost of wrong guesses)
  at D = 1–5, and 30 % wrong at D = 3.

| Run | `TAG` | `MODEL` | `GPUS` | extra |
|---|---|---|---|---|
| S4-L0, six cards | `s4l0-6c` | `/mnt/models2/flint/flint-S4L0.gguf` | `0,1,2,3,4,5` | |
| S4-L0, nine cards | `s4l0-9c` | same | `0,1,2,3,4,5,6,7,8` | `DEPTHS='1 2 3 4 5 6 8'` |
| S4-L0, nine cards, light last card | `s4l0-9c-light` | same | all nine | `TS=6,6,6,6,6,6,6,5,2 DEPTHS='1 2 3 4 5 6 8'` |
| G3, six cards | `g3-6c` | G3 on a stick | `0,1,2,3,4,5` | |
| G3, nine cards | `g3-9c` | same | all nine | `DEPTHS='1 2 3 4 5 6 8'` |

All with `ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1` (both files are scored with the rescale off). The first row in full; the
others change only the variables of the table:

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-6c GPUS=0,1,2,3,4,5 \
    ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/s4l0-6c.out 2>&1 < /dev/null &"
ssh rig 'tail -3 /tmp/pipespec/s4l0-6c.out'
```

Optional, the continuation table of Q96 on top (`TABLE=/tmp/pipespec/table.txt`, adds `conttab-d3-*` rows): make it
from Q51's transcripts without the transcripts of the same prompts, `python3 -I bench/pipespec/make_table.py
<Q51 gen.jsonl> table.txt --exclude '^(chat_0|code_0|edit_)'`, and copy it to `/tmp/pipespec/` on the rig.

**3. Host sampling once** (logits to the host for every token, the path without `-bs`):

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-6c-host BG=0 GPUS=0,1,2,3,4,5 \
    PROMPTS=rewrite DEPTHS='1 3 5' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh \
    > /tmp/pipespec/s4l0-6c-host.out 2>&1 < /dev/null &"
```

**4. Races, K = 2 and 3** ([rig_race.sh](../../../bench/pipespec/rig_race.sh); temperature 1, top-k 20, top-p 0.95;
seeds 1–3; the short question in `prompts/short.txt`; ~15 min per seed): each seed runs one sample alone (`race1`,
timed to its `</think>` the same way) and races of 2 and 3, all plainly and pipelined.

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-6c-race GPUS=0,1,2,3,4,5 \
    RACES='2 3' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_race.sh > /tmp/pipespec/s4l0-6c-race.out 2>&1 < /dev/null &"
```

**Several conversations (Q104, [multi-conversation serving](multi_conversation_serving.md) section 5.1; ~3 h).**
Experiments A–C need no new code; D times the streams mode. Nine cards with the light last card unless noted.

*A. Step cost of N tokens in one batch* (llama-server's way, ~1 h): `llama-batched-bench` from the v2 build and from the
multi-token build (`llama.cpp-20261023-032b130f4-pascal-p106-v2` and `llama.cpp-20261023-dca843432-pascal-p106-v2-multicol`);
`S_TG` per row gives v(N).

```
for b in llama.cpp-20261023-032b130f4-pascal-p106-v2 llama.cpp-20261023-dca843432-pascal-p106-v2-multicol; do
  ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /opt/llama-builds/$b/bin/llama-batched-bench \
      -m /mnt/models2/flint/flint-S4L0.gguf -ngl 99 -fa on -c 16384 -b 512 -ub 128 -npp 512 -ntg 128 -npl 1,2,3,4,6,8" \
      > batched-$b-6c.txt
  ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 /opt/llama-builds/$b/bin/llama-batched-bench -m /mnt/models2/flint/flint-S4L0.gguf \
      -ngl 99 -fa on -ts 6,6,6,6,6,6,6,5,2 -c 16384 -b 512 -ub 128 -npp 512 -ntg 128 -npl 1,2,4" > batched-$b-9c.txt
done
```

*B. The staggered ceiling* (~1 h): with every guess right, depth D keeps D + 1 one-token steps in flight, the card load
of D + 1 conversations. Greedy on the last card; `host.csv` is the submit thread's CPU.

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q104-ceiling-6c GPUS=0,1,2,3,4,5 \
    PROMPTS=chat DEPTHS='1 2 3 4 5' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh \
    > /tmp/pipespec/q104-ceiling-6c.out 2>&1 < /dev/null &"
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q104-ceiling-9c GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 PROMPTS=chat DEPTHS='1 2 3 4 5 6 7 8' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 \
    bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/q104-ceiling-9c.out 2>&1 < /dev/null &"
```

Read `oracle-dD-m0-chat` (and `d0-chat` for D = 0). Expected (est.): six cards 75 / 112 / 149 / 159–167 tok/s for 2 / 3 /
4 / ≥ 5 in flight; nine cards up to 235–256.

*C. N real sequences, host sampling* (~1 h): races with a stop text that never appears run N sequences of their own
(one shared prompt, a unified cache), one token each in flight, until the first reaches 256 tokens; `race_tok_s` is the
staggered rate of N sequences.

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q104-seqs-9c GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 RACES='2 3 4 6 8' SEEDS=1 PROMPT=chat N=256 STOP=@@never@@ ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 \
    bash /tmp/pipespec/rig_race.sh > /tmp/pipespec/q104-seqs-9c.out 2>&1 < /dev/null &"
```

Expected below B by the logits copy of host sampling (~110 tok/s ceiling on six cards in the note's estimate); the
difference prices sampling on the device.

*D. Streams mode* ([rig_streams.sh](../../../bench/pipespec/rig_streams.sh), ~45 min per context size): N = 1, 2, 3, 4, 6,
8 conversations (the four prompts cycled), one KV stream each, sampling on the last card at temperature 1, prompts as
64-token chunks; guesses in the slack at N = 2 and 4; staggered arrivals at N = 8; every conversation checked against its
own plain run.

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q104-streams-9c-8k GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 CTX=8192 ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_streams.sh \
    > /tmp/pipespec/q104-streams-9c-8k.out 2>&1 < /dev/null &"
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=q104-streams-9c-32k GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 CTX=32768 NS='1 2 4 8' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_streams.sh \
    > /tmp/pipespec/q104-streams-9c-32k.out 2>&1 < /dev/null &"
```

`summary.tsv`: total tok/s, slowest and median conversation, `same_as_plain`. Expected (est., serving note): 64 tok/s at
N = 2, 118–128 at 4, 235–256 at 8, each conversation at 29–32. Gates: B at ≥ 3.3× one stream with every card busy on six
cards, and `host.csv` below one core at the ceiling; A's 2-token step on the multi-token build ≤ 1.5 T before groups of
two are built.

**4b. Served: llama-server with `--pipe-depth`** ([above](#llama-server---pipe-depth); the build of
`pascal-p106-v2-pipeserve-tools`, `llama.cpp-20261024-ba83d3a1f-pascal-p106-v2-pipeserve-tools`). It runs the five-turn chat of
[server_bench.sh](../../../scripts/server_bench.sh) (the model's default sampling: T 1, top-p 0.95, top-k 20, min-p 0)
and the coding-agent tool request of [tool_request_ab.sh](../../../bench/tool_sampling/tool_request_ab.sh), each
batched and pipelined, on nine cards, then on six (`GPUS=0,1,2,3,4,5`, TAG `…-6c-…`); ~1 h in all:

```
B=/opt/llama-builds/llama.cpp-20261024-ba83d3a1f-pascal-p106-v2-pipeserve-tools/bin; M=<K2h-r16>
for d in 0 3 5; do
  P=$([ $d -gt 0 ] && echo "--pipe-depth $d")
  ssh rig "env BIN=$B MODEL=$M TAG=serve-9c-d$d PROMPTS=chat MAX_TOKENS=2048 LLAMA_NO_MOE_DOWN_RESCALE=1 \
      SERVER_ARGS='-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4 $P' bash -s" < scripts/server_bench.sh
done
for d in 0 3; do
  P=$([ $d -gt 0 ] && echo "--pipe-depth $d")
  ssh rig "env BIN=$B MODEL=$M GPUS=0,1,2,3,4,5,6,7,8 TAG=tools-9c-d$d SERVER_ARGS='$P' LLAMA_NO_MOE_DOWN_RESCALE=1 \
      bash -s" < bench/tool_sampling/tool_request_ab.sh
done
ssh rig 'grep -h "pipelined =" /tmp/server_bench/serve-9c-d*/*.log /tmp/tool_request_ab/tools-9c-d*/server.log | head -20'
```

What to read:
- **Chat:** tokens per second per turn at depth 0 against 3 and 5. rig_run.sh's `cont` gains at the best depth (from
  the traces: chat 1.38×, code 1.35–1.43×, edits 1.41–1.44×, rewrites 1.72–1.84× on K2h-r16, six and nine cards) are the
  expectation for the reasoning-heavy turns.
- **Identity:** the transcripts at depths 3 and 5 (`chat.json`) must be identical, since both use position-keyed draws.
  Depth 0 uses the server's ordinary draws, so its output differs at temperature 1, with the same distribution.
- **The tool request:** "auto" with backend sampling at depth 3 against depth 0. The log's "pipelined" line counts the
  guesses before the call; the call itself runs batched.
- **Fallback:** a request that does not qualify (penalties on, `n_probs`) shows no guesses.

**4c. The fixes after the first rig runs** (the build of `pascal-p106-v2-pipeserve` with Q153, Q154 and Q156,
`llama.cpp-20261024-2650855b2-pascal-p106-v2-pipeserve`; ~1 h). From this step on the commands set
`LLAMA_NO_MOE_DOWN_RESCALE=tg`: the MoE down-projection rescale is dropped only where that product has one token,
which the fork's Pascal kernels compute with FP32 block sums. The earlier steps ran with `=1`, which drops it in prompt
batches too, where MMQ reads FP16 block sums (Q4_1, Q5_1, Q4_K, Q5_K, and Q4_0 without tensor cores). That gave NaN on a
long-context prompt chunk of K2h-r16 (found on the V100s).

```
B=/opt/llama-builds/llama.cpp-20261024-2650855b2-pascal-p106-v2-pipeserve/bin; M=<K2h-r16>
# Q156: fed tokens in streams mode (n4-fed and n8-fed must now be at least n4 and n8)
ssh rig "setsid nohup env BIN=$B MODEL=$M TAG=fedfix-9c GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 CONFIGS=u2f NS='4 8' \
    ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg' bash /tmp/pipespec/rig_hostloop.sh > /tmp/pipespec/fedfix-9c.out 2>&1 < /dev/null &"
# Q153: depth 8 on the edit prompt, KV steps of 256 against 1024 (traces.py: the slow tokens after each step)
for pad in 256 1024; do
  ssh rig "env BIN=$B MODEL=$M TAG=kvpad$pad-9c GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 PROMPTS=edit DEPTHS='8' \
      ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg LLAMA_KV_PAD=$pad' bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/kvpad$pad-9c.out 2>&1"
done
# Q154: the floor in the server, fillers only (no guesser) at depth 3, against depth 0
ssh rig "env BIN=$B MODEL=$M TAG=floor-9c GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 DEPTHS='0 3' GUESS=none \
    EXTRA_ARGS='--pipe-floor 3' ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg' bash /tmp/pipespec/rig_server.sh > /tmp/pipespec/floor-9c.out 2>&1"
```

Every identity column must read `yes`. The fed runs report tok/s per configuration in `summary.tsv`. The KV-pad pair
compares `tok_s` of `cont-d8-edit`, and `traces.py` shows the gap to the next token after the 256-token steps. The floor
run gives the speed-up of fillers alone over depth 0.

**4d. Prompt chunks next to decoding conversations (Q155)** (the build of `pascal-p106-v2-q155`, which sits on
`pascal-p106-v2-pipeserve-tools` and so also serves the tool-request A/B of the tool sampling note,
`llama.cpp-20261024-98e79d8c9-pascal-p106-v2-q155`; ~20 min):

```
B=/opt/llama-builds/llama.cpp-20261024-98e79d8c9-pascal-p106-v2-q155/bin; M=<K2h-r16>
ssh rig "setsid nohup env BIN=$B MODEL=$M TAG=prefill-9c GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 \
    CONFIGS='k2 k8 k8g1 k4s k8s' ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg' bash /tmp/pipespec/rig_prefill.sh \
    > /tmp/pipespec/prefill-9c.out 2>&1 < /dev/null &"
```

`k2` is the executor before Q155 (at most one chunk per prompt, two in flight): its `all` job is Q120's
eight-conversation mix. `all` measures prefill throughput when nothing else is decoding: the first-token times of the
long prompts and the job's tokens per second. `arrive` brings each prompt in while the others decode, so the comparison
between `k8`, `k8g1` and `k8s` there is decode priority. Every `same_as_plain` must read `yes`. A configuration whose
chunk slots do not fit next to the model stops at its start (`run-<config>.log`), and the next one runs.

**4e. The continuation store in llama-server (Q161)** (the build of `pascal-p106-v2-q161`, which also carries Q155 and
the tool-request pipelining, `llama.cpp-20261024-9d50872e0-pascal-p106-v2-q161`, and a store of 13.8 M tokens of Flint
answers, `flint-answers-clean-13.8M.pstore`, 110 MB, copied next to the builds; ~30 min):

```
B=/opt/llama-builds/llama.cpp-20261024-9d50872e0-pascal-p106-v2-q161/bin; M=<K2h-r16>
S=/opt/llama-builds/stores/flint-answers-clean-13.8M.pstore
ssh rig "setsid nohup env BIN=$B MODEL=$M TAG=store-9c GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 DEPTHS='0 3 3s' \\
    STORE=$S ENVS='LLAMA_NO_MOE_DOWN_RESCALE=tg' bash /tmp/pipespec/rig_server.sh > /tmp/pipespec/store-9c.out 2>&1 < /dev/null &"
```

`d3s x` against `d3 x` is the store's gain per request: chat, code and edit prompts and the coding-agent tool request,
at T 0 and T 1. The simulation expects ~5 % on chat, code and edits and more on the tool request. The guesses column
counts the store's guesses (right / wrong / from the store). Every `identical` must read `yes`.

**5. Collect and clean up:**

```
mkdir -p <results> && scp -r rig:/tmp/pipespec/out rig:/tmp/pipespec/*.out <results>/
ssh rig 'rm -rf /tmp/pipespec'
```

### What to log

Per run, in `out/<TAG>/`: `summary.tsv` (speed-up against the plain job of the same prompt, right and wrong guesses,
dropped tokens, and `same_as_plain`), `results.txt` (one `PIPESPEC` line per job: tokens/s, hashes, race fields),
`run.log`, `plain.jsonl` (the plain outputs, for the simulator), one `.trace` per timed job (time of every sampled token
and what was in flight behind it) and `gpu.csv` (clock, power, temperature and load of every card each second). Keep
with them the build's `BUILDINFO.txt`, the model file names and the card set; summarize the traces with
`python3 -I bench/pipespec/traces.py out/<TAG>`.

### Reading the results

`tok_s` and `speedup` in summary.tsv come from the traces: generated tokens per second from a job's first sampled token
to its last. llama-pipespec's own rate (`tok_s_tgen`) also counted the wait for the first token until fork commit
`99c3a7c00`: `llama_decode` returns before the cards finish the prompt, so that wait held 0.1–3.3 s of the 750- and
2,076-token prompts, a different share in every job. On those prompts it put plain decoding 1–19 % low and the
pipelined jobs up to a third low. Edit speed-ups read up to 0.14 too high, rewrite speed-ups up to 0.8 too low. On the
short chat and code prompts the wait was 0.3–0.5 s: 2–6 % at plain decoding's speed, up to ~20 % for the fast oracle
jobs with every guess right.

| Rows | Question | Expected (event model) | If not |
|---|---|---|---|
| `same_as_plain` | Is every pipelined job plain decoding's output? | `yes` | a bug: keep `run.log` and the jobs |
| `d0-*` | Does the pipelined API cost anything by itself? | 0.98–1.02 | host time per submit: the traces' gaps after `n` |
| `oracle-dD-m0-chat` | The ceiling, one token per slowest stage | six cards 2.0 / 3.0 / 4.0 / 4.5 / 4.5 at D = 1–5; nine cards up to 6.2 (even split) and 8.9 (light last card) | per-stage overheads (launches per slot, mask copies, hops) or the clock under load (`gpu.csv`) |
| `oracle-dD-m1-chat` | Do wrong guesses cost nothing at D ≤ S_eff − 1? | six cards ≥ 0.98 up to D = 3, then 0.90 and 0.75 at D = 4, 5 | the host or the last card is the bottleneck |
| `cont-dD-*` | The real gain per kind of prompt | simulator (Q96, six cards, D = 5): chat 1.18, code 1.35, edits 1.85, rewrites ~4 | compare with the simulator on the same outputs (below) |
| `copy2/copy4/mapk-d3-*` | Does the continuation fallback pay over the copy lookup alone and over `ngram-map-k`? | cont ≥ copy4 ≥ mapk | – |
| `race1/raceK-pipe` | Does a race of K run at the speed of one sample, and how much earlier does it end? | `race_tok_s` of race K ≈ K × that of race 1; `t_race` of race K below race 1 (Q93: 64 % of the reasoning at K = 3) | K > S_eff, or the host |

Then choose D per card count (the event model says D ≤ 3 on six cards, where the output head makes the last card the
slowest stage) and whether the next step is a per-stage launch (if `oracle-d4/d5-m1-chat` lose), sampling on the GPU for
temperature sampling, or the staircase drafter of [pipeline_drafter.md](pipeline_drafter.md).

### Comparing with the simulator

[pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) replays a transcript through the event model with the
same guessers. Fed with the rig's own outputs it predicts the right and wrong guesses exactly (the executor's copy lookup
and continuation counts are the simulator's: on a 256-token rewrite output both gave 179 right and 38 wrong for `cont`,
167 and 14 for copy ≥ 2), so any difference in speed comes from timing alone:

```
python3 -I bench/ngram_replay/pipeline_sim.py <results>/out/s4l0-6c/plain.jsonl \
    --setup "s4l0-6c=<ms per plain token>:9,8,8,8,8,8" --depths 1,2,3,4,5 \
    --only 'request counts \(no table\)|copy, match >= (2|4) \|' --out sim-s4l0-6c.json
```

- `<ms per plain token>`: 1000 / the mean tokens/s of the `plain-*` rows. Layers per card: 9/8/8/8/8/8 on six cards,
  6/6/6/6/5/5/5/5/5 on nine, 6/6/6/6/6/6/6/5/2 with the light last card.
- Compare per class (chat, code, edit; `rewrite` is filed as edit) and per depth: the simulator's right and wrong guess
  shares against `hits`/`misses` over `n_gen` (they should agree), then its speed-up against `cont-dD-*`.
- The simulator assumes 0.3 ms per hop and 1.5 ms of host time per sample (`--hop`, `--gap`). If the measured ceiling
  (`oracle-dD-m0-chat` at the largest depth) is below the simulator's "one token per slowest stage + hop" (its setup
  table), raise `--hop` until the two match, then compare the guessers again. The simulator withdraws wrong guesses per
  card, the executor drops them; the two agree only while D ≤ S_eff − 1.

## What a production version needs

- **Seeded sampling on the GPU (built).** Host sampling copies each token's full logits to the host (Flint: 150,144
  floats, 0.6 MB, ~3 ms on a PCIe 1.0 x1 link), queued on the last card behind every token, guesses included. Greedy
  sampling and the temperature chain run on the GPU. A guess's random draw is set when it is submitted, before the
  tokens ahead of it are sampled, so each draw is keyed by the token's position (`llama_sampler_set_pos`): pipelined
  decoding at T 1 then equals plain decoding with the same keyed draws (`--pipe-backend-sample` in `llama-pipespec`,
  every eligible request in llama-server).
- **llama-server integration (built):** [above](#llama-server---pipe-depth), for a slot that generates alone. Still
  missing: grammars once they can reject (the window stops at a tool call's start), penalties, several slots at once.
- **Several sequences:** independent conversations staggered by one stage keep every card busy with no guessing at all;
  the slots and the race loop already do this for samples of one request, a server would do it across its slots.
- **Per-stage launch** if depths beyond S_eff − 1 are wanted (the host would launch stage j of a token only after stage
  j − 1 finished and the token is still valid).
- **Recurrent models:** built for the delta-net hybrids ([KV cache](#kv-cache), Q162); other recurrent architectures
  (Mamba, RWKV, …) would need their state writes moved to the slot's group the same way.
- **Input copies:** on `pascal-p106-v2-pipespec` every in-flight token copies its inputs (masks, positions, cell
  indices, the embedding row) to every card synchronously at submit time, so host time per token grows with the number
  of cards and the context. `pascal-p106-v2-pipespec2` uploads each card's inputs in one asynchronous copy and can feed
  the next token from card to card without the host ([host decode loop](host_decode_loop.md) §6); the masks are still
  filled on the host until Q116 builds them on the card.
