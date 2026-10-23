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

- **What exists.** A pipelined decoding API in the fork (`llama_pipe_init`, `llama_pipe_submit`, `llama_pipe_get_logits`,
  `llama_pipe_drain`) and a tool, `llama-pipespec`, that runs plain decoding or pipelined decoding with up to D guesses
  in flight for one sequence. Guessers: the copy lookup, llama.cpp's `ngram-simple`, any llama.cpp drafter
  (`ngram-map-k`, `ngram-mod`, …), and an oracle that replays a reference run with a chosen share of wrong guesses.
  A job list runs many configurations with one model load.
- **Correctness.** RESULTS_PLACEHOLDER
- **Wrong guesses are dropped, not withdrawn.** All stages of a token are queued when it is submitted, and a wrong
  guess simply runs to the end while its result is ignored. Because every guess runs ahead of the corrected token on
  every card, spaced by at least one stage time, this costs nothing while D ≤ S_eff − 1 (S_eff = sum of stage times /
  slowest stage time: 4.2 for six cards with the output head on the last one, 5.9 and 8.5 for the two nine-card splits
  below). An event model shows dropping and ideal per-stage cancellation give the same speed up to that depth; beyond
  it, dropping loses (six cards, every guess wrong: 0.90× at D = 4, 0.75× at D = 5).
- **The output is plain decoding's output.** Each token is sampled once, in order, from logits computed by the same
  graph, kernels and KV cell layout as plain decoding, with the same sampler state, so greedy and seeded sampling give
  the same tokens, with bit-identical logits.
- **Next:** time it on the rig ([plan](../../../bench/pipespec/README.md#rig-plan)): the all-right oracle shows the
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
`llama_pipe_get_logits(ctx, slot)` waits for that event. `llama_decode` drains all slots first.

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

Event model of the cards (stage times of [ngram_replay](../../../bench/ngram_replay/README.md#model) with 0.3 ms per hop
and 1.5 ms of host time per sample; i.i.d. guesses right with probability p; speed-up over plain decoding):

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
  decoding would. `seq_rm` moved `head` to the lowest freed index, which is wrong when the dropped positions wrap around
  the end of the cache; the fork now moves `head` to the freed cell with the lowest position when the tail of a sequence
  is removed (same cell as before in every other case).
- **Sliding window.** A guess judges sliding-window reuse against its own position, D positions ahead of the committed
  one. That is safe as long as the cell it overwrites is older than every window a re-submitted token needs: cells are
  reused oldest first and the window cache has n_ubatch spare cells, so `llama_pipe_init` requires n_ubatch ≥ number of
  slots for a sliding-window model (the default 512 is far above it). The purge that comes with such an overwrite can
  remove up to D old, already masked positions earlier than plain decoding would; attention is unaffected, but a dump of
  the sequence state can differ in those masked cells.
- **Recurrent layers** (Qwen3.6's Gated DeltaNet) can't be truncated by position; they need a state snapshot per
  in-flight token (SPD's ring of states). Not implemented.

### Guessers

| `--pipe-guess` | Source | History it sees |
|---|---|---|
| `copy` | the token after the longest earlier match of the history's suffix, if at least `--pipe-copy-min` long | committed + in flight |
| `ngram-simple` | llama.cpp's `common_ngram_simple_draft` (`--spec-ngram-simple-size-n/-m`) | committed + in flight |
| `spec` | any llama.cpp drafter (`--pipe-spec ngram-map-k`, `ngram-mod`, …) through `common_speculative`; one draft per committed state, consumed one guess at a time | committed only (stateful drafters must not see tokens that may be dropped) |
| `oracle` | a reference run's tokens, a share `--pipe-miss` made wrong on purpose | – |

## Correctness

CORRECTNESS_PLACEHOLDER

## What a production version needs

- **Sampling on the GPU.** The executor copies each token's full logits to the host (Flint: 151,936 floats, 0.6 MB, ~3
  ms on a PCIe 1.0 x1 link) and samples there; the last card's stream carries that copy for every token, guesses
  included. llama-server's backend sampling returns one token id instead. Greedy needs nothing more; seeded sampling needs
  a random draw tied to the position (or the transactional draws backend sampling already has for speculative decoding)
  because guesses are submitted, and their inputs set, before earlier tokens are sampled.
- **llama-server integration:** a window per slot, guesses from the slot's prompt + output, the grammar and stop checks
  applied to sampled tokens only (sample-and-match keeps them exact), and `--ctx-checkpoints` taken at committed
  positions only.
- **Several sequences:** independent conversations staggered by one stage keep every card busy with no guessing at all;
  the slots already allow it, the loop doesn't.
- **Per-stage launch** if depths beyond S_eff − 1 are wanted (the host would launch stage j of a token only after stage
  j − 1 finished and the token is still valid).
- **Recurrent models:** per-token state snapshots and rollback.
- **Mask copies:** every in-flight token copies its masks to every card synchronously at submit time (host time per
  token grows with the number of cards and the context).
