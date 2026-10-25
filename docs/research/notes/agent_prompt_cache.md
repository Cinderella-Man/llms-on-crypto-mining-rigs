# Agent prompt cache: multi-turn sessions that stop re-reading their prompts (Q111, Q132)

An agent sends its whole conversation with every request. Replaying 200 of Flint's BFCL v4 multi-turn sessions (2,364
requests), the prompts re-send 39 tokens per generated token ([prefill on Pascal](prefill_pascal.md) §4.3). Reading a
prompt is the rig's slow direction (714 tok/s at 8k tokens on nine cards, ~68 tok/s at 64k), so what the server can
reuse from earlier requests decides how long an agent waits. This note covers what llama-server reuses for Flint, the
fork's changes (branch `pascal-p106-v2-agentcache`, on `pascal-p106-v2-next` `d379da309`), how they are checked, and
the rig run that measures them. Tools: [bench/agent_cache](../../../bench/agent_cache/README.md).

## 1. What llama-server reuses for Flint

Flint has 35 sliding-window layers (513-token window) and 14 full-attention layers. The full-attention cache keeps every
position; the sliding-window cache is a ring of n_swa + n_ubatch cells rounded up to 256 (1,280 at `-ub 512`) that keeps
overwriting its oldest cell. A new request reuses the longest token prefix it shares with the slot's cached tokens, if
the cells before that point are still there:
- **The live ring** serves a divergence up to ~770 tokens before the end of the previous request (1,280 cells minus the
  513-token window, at `-ub 512`; ~255 at `-ub 128`). A reply sent back exactly reuses everything, and a tool call that
  the chat template re-renders differently near the end of a short reply costs only the tool call.
- **Context checkpoints** serve divergences further back. A checkpoint holds only the last 513 positions before its point
  (the state writer leaves out the ring's cells outside the window, whatever the checkpoint's recorded range says), so it
  serves only a divergence at or after its own point. Hence stock's two checkpoints at each prompt's end, 4 + n_ubatch
  and 4 tokens before it, plus one at user messages. None are taken while generating.
- **Nothing else crosses conversations.** A new session with the same system and tool prompt as the slot's last one
  restores the checkpoint at the first user message, if it survived; a session that follows another conversation
  (another agent, a sub-agent, another project's prompt) or a restart reads its whole system and tool prompt (BFCL: a
  median 5,033 tokens, 54 % of what a session reads after llama-server's own reuse when 8 slots serve interleaved
  tasks). `--cache-ram` would copy whole states into the 4 GB host and stays off.
- **Fixed costs per request:** with backend sampling (`-bs`, the rig's setting) every request that samples on the device
  attaches and detaches its sampler chain (with Q67, tool requests too), and each change marks the scheduler for a full
  reserve: a new scheduler and three worst-case graph builds before the request's first micro-batch. Each of the two end-of-prompt checkpoints waits for the pipeline to
  drain and copies ~35 MiB of window cells to the host.
- **Restores move cells.** A restored checkpoint landed in cells 0.. in saved order, not where the cells had been, and a
  cache stream emptied by a new prompt kept its head wherever the ring had wrapped. Attention adds keys in cell order,
  so a cached answer differed from a cold server's in the last bits even when every value was the same.

**Q132 (checkpoint eviction):** for a linear agent session the eviction rule is not what costs. A request's checkpoints
are protected while that request runs, and the next request restores before it creates any, so the previous turn's
checkpoints are always there when they are needed. What re-reads a lot is a divergence more than ~770 tokens before the
end of the previous reply: a client or template that changes the reasoning (whitespace, a dropped block) or a long tool
call (a file written in its arguments). Stock then falls back to the checkpoint before the previous prompt's end and
reads the whole reply again, up to 7,900 reasoning tokens (est. 60–100 s on v1, ~11 s on v2). Edits in the middle of
the history (pruning old tool output, compaction) fall back to the newest checkpoint before them, as before.

## 2. The changes

| Change | What it does | Cost |
|---|---|---|
| `--prefix-cache DIR` | The state of a prompt's shared prefix (everything before the first user message: the system and tool prompt) is saved to a file when a slot processes it from its first token, and restored into any slot whose prompt starts with it and that cannot reuse that far itself (another conversation in between, a restart). Files carry a fingerprint of the model file and the settings that shape the cells (batch sizes, flash attention, cache types), outlive the server, and are trimmed to `--prefix-cache-max` (default 16) least recently used first; `--prefix-cache-min` (default 1,024 tokens) | Flint: 28 KiB per prefix token plus 35 MiB of window cells, ~170 MiB for a 5k-token prefix on disk; the save waits once per distinct prefix |
| One ring checkpoint per prompt end | Sliding-window models take only the checkpoint 4 tokens before the end, with the whole ring (`LLAMA_STATE_SEQ_FLAGS_SWA_RING`): it covers every position from the ring's oldest, so after cutting it back to the divergence it also serves what the one 4 + n_ubatch tokens earlier served. `--checkpoint-tail2` keeps stock's two | One pipeline drain per request instead of two; 87.5 MiB per checkpoint at `-ub 512` instead of 2 × 35 MiB |
| `--checkpoint-gen` | Ring checkpoints while generating, at `</think>` and `<tool_call>` (`--checkpoint-gen-at`), once 512 tokens (`--checkpoint-gen-min`) passed since the slot's last checkpoint: a re-rendered reply that differs after the reasoning re-reads only from there, not the reasoning | A drain and an 87.5 MiB copy per checkpoint, only after long reasoning; `--ctx-checkpoints 4` holds at most ~350 MiB |
| No reserve per request | The scheduler reserve remembers the attached sampler chains by their samplers' names; attaching a covered set or detaching needs no new reserve (a larger top-k grows the allocator's buffers on first use). `LLAMA_SAMPLER_RESERVE=1` restores stock | None |
| Canonical cell layout | An emptied cache stream restarts at cell 0, and a restored state goes where processing the sequence from its first token puts each cell (position p in cell p, p mod size in the ring), with the head after the newest | None; it also makes stock's checkpoint restores exact |
| Restores resume after the checkpoint | A sliding-window checkpoint restore resumes at its last position + 1 (stock: at its last position, which a recurrent state needs), so the batch after it is the one that followed the checkpoint when it was taken, and a GPU rounds it the same way | One token less to evaluate per restore |
| Cache counters | Timings report where the reused cells came from (`cache_src`: none, live, checkpoint, prefix), the restored checkpoint (`cache_ckpt_n`), checkpoints created (`ckpt_created_n`) and restore time (`cache_restore_ms`) | None |

The pipelined generation path (`--pipe-depth`) takes no generation checkpoints; tool requests run batched in this build
anyway (their grammar keeps them off the pipeline).

## 3. Exactness

`bench/agent_cache/check_exact.py` sends every request to the server under test and to a cold server (cache_prompt
off, so a fresh slot processes the whole prompt) and compares the top-8 log-probabilities of every generated token.
On a CPU build without llamafile GEMM and with `-fa off`, a query's sum over the keys does not depend on the batch it
is in, so every cache path must match bit for bit; the model is a 6-layer Flint with random weights and Flint's
tokenizer, template, window and layer pattern (`make_tiny_flint.py`).

| Path (tiny Flint, CPU, `-fa off`) | Result |
|---|---|
| Prompts from scratch, one after another in one slot | identical |
| Saved prefix restored after another conversation in the slot; after a server restart, from its file | identical |
| A chat whose replies go back as history (live ring, 6 turns) | identical |
| Divergence 50 tokens before the end (live ring) | identical |
| Divergence 500 tokens before a ring checkpoint, cut back (stock would read from 16 tokens earlier) | identical |
| Divergence 400 tokens into a 1,500-token reply, from a generation checkpoint | identical |
| Stock's two tail checkpoints, restored | identical (they differed before the canonical layout) |
| `-bs`, three tool requests: scheduler reserves 1 instead of 2 on `v2-next`; 2 instead of 4 on the release candidate, whose on-device sampling (Q67) attaches a chain to tool requests too | the same replies |

On a V100 with a real Flint file (Q4_0, 23 GB; one server, a test slot and a cold slot; the V100's own kernels), the
saved prefix restored into the test slot gives the cold slot's log-probabilities bit for bit (2,869 tokens, 113.5 MiB,
~400 ms), as do a first chat turn and a repeated request restored from its tail checkpoint. Paths that evaluate tokens
in other batches than a cold server does differ in the last bits and then in sampled tokens, as stock llama.cpp's own
cache does: generated tokens (one at a time, against one batch for the cold server; largest first-token difference among
the top candidates 0.16 to 2.35 log-probability units, same top token), and a restore cut back into the middle of an
earlier batch. So do two configurations that split prompts differently: with stock's two tail checkpoints, the one
4 + n_ubatch tokens before the end falls inside the shared prefix when the user message is short, so the prefix's cells
depend on the message's length and a saved prefix cannot equal them; the single tail checkpoint never splits the prefix.
The replay's exactness control (`stock-tail1`) is therefore stock behaviour with the single tail checkpoint.

Before the canonical layout, every restore (stock checkpoints too) differed from the cold server from the first token,
and the first cut-back design used checkpoints that hold only the window: a restore then lacked most of the window,
which the strict comparison caught. With flash attention on, the CPU's attention splits the keys between threads by
batch size, and GPU kernels differ between one-token and multi-token batches, so there only paths whose reused cells
come from the same prompt evaluations as a cold server's (a saved prefix) are bit-identical; the others match as
llama.cpp's own prompt cache does.

## 4. On the rig

`bench/agent_cache/rig_replay.sh` replays eight BFCL sessions (two tool sets × four, alternating, 4–11 requests each)
per configuration on nine cards with K2h-r16: stock behaviour (a reserve per request, two tail checkpoints, no prefix
files or generation checkpoints), stock behaviour with the single tail checkpoint (the exactness control: the agent
cache's micro-batches, so the agent cache must give its replies), the agent cache, and stock and the agent cache with the client dropping
earlier replies' reasoning (what most OpenAI-compatible clients do). Each request reuses the server's own last reply, so
the replay measures what an agent gets: time to first token per request, prompt tokens read, reuse classes and
identical replies.

Expected (est., from §1 and the rig's prompt speeds): with the tool sets alternating, stock reads the whole first prompt
of all eight sessions and the agent cache of two, restoring the other six (~1 s of state instead of
3.5–4.8k tokens, 5–7 s on v2); every request saves one pipeline drain, and every request that samples on the device one
reserve (0.1–0.5 s). On a V100 (counts only), one tool set after the other, stock's own checkpoint at the first user
message already served each next session's prompt: 57.7k against 55.6k prompt tokens read over six sessions.
