# Hyperkernel / Hyperserve: what it is, and what I can take from it (2026-11-07)

Repository: https://github.com/Hyperstitioncc/Hyperkernel (MIT), read at commit `7048c58` (2026-11-07, "hyperserve
demos"). Read as text only: shallow clone with no hooks, no submodules, no LFS; nothing built or run.

## What it is

- **Not the Hyperkernel itself.** The generator that writes the "megakernels" is not public. The repository is
  **Hyperserve**: about 12,000 lines of Rust host code plus **precompiled megakernels for NVIDIA B300 only**
  (`kernels/b300/*/entry.cubin.zst`, `loader*.cubin.zst`, SASS listings built with `-arch sm_103a`), one bundle per
  model and per concurrency preset.
- Models: DeepSeek-V4.1-Flash on 4 × B300 (OpenAI-compatible server, presets c1–c8, 1M context, ~203 GB of host RAM
  for the Engram tables), openpilot, Evo 2 7B, Matrix-Game 3.0, ACT, π₀-FAST, ESMFold, and a Qwen 30B training binary.
- **Nothing runs on Pascal.** The kernels are sm_103a binaries; there is no kernel source to port.

## Safety check (what was looked at)

- No `build.rs` and no `.cargo/` config, so `cargo build` runs no repository code at build time. Dependencies are
  ordinary crates (`cudarc` 0.19.10, serde, httparse, fancy-regex, clap, ruzstd, libc).
- Network: the server listens on `127.0.0.1` by default; the only outbound URLs are a localhost test script and one
  Hugging Face dataset URL in `scripts/qwen30b/fetch.py`. No other HTTP clients, sockets or downloads in `src/`.
- Process spawns re-run its own binary (one process per GPU rank, `src/gpu/ranks.rs`; the world model's VAE,
  `src/mg3/cli.rs`). Temporary files go under the system temp directory with the process id.
- The real unknown is the **binary GPU code** (cubins): it cannot be reviewed as source. It only loads on B300s, so it
  never touches my machines. Verdict: the host code looks clean; I don't build or run any of it.

## What is worth taking (ideas, not code)

1. **A prefix cache shaped as a tree of pages, with a state snapshot per page** (`src/engine/cache.rs`): prompts are
   cut into fixed pages held in a radix tree keyed by token runs; pages are reference-counted and shared between
   requests, evicted least recently used, and each node also keeps a "window" page, the sliding-window state at that
   point, so a request resumes from the deepest shared page, not only from the single slot's last prompt.
   For an agent this means a retry, an edited turn, a subagent or a compaction branch reuses the shared start instead
   of prefilling it again. llama-server already reuses one slot's prefix with checkpoints (Q111); the step beyond is
   many checkpoints in a tree, which suits Swift Flash-Next's recurrent layers (their state cannot be cut back, so it
   has to be snapshotted). Queued as Q1135 / C649.
2. **One persistent kernel per GPU fed through a command ring** (`src/gpu/feed.rs`, `src/gpu/load.rs`): the host
   writes commands into ring blocks the running kernel reads, reads the sampled tokens back, and orders several
   kernels on one GPU with stream events so "the host never waits". On the rig every decode step is many small
   launches on 9 cards; a persistent decode kernel would remove launch and sync gaps that CUDA graphs still leave.
   Large research item; queued as Q1136 / C650.
3. **Chunked prefill paced against decoding** (`src/engine/pacing.rs`): a prefill token budget minus
   `decoding × block` per step, prefill every N decode steps, a headroom of 2,048 tokens. Useful only when several
   conversations decode at once; I run one agent, so it is parked. Q1137 / C651.

Not worth taking: the Engram host tables (203 GB of host RAM, against the rig's 4 GB), the B300 bundles and SASS,
the OpenAI API layer, grammar and detokenizer (llama-server has its own).
