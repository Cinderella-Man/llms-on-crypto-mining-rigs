# Q228: how many reads of Flash-Next's n-gram table a token costs, from replayed agent sessions (2026-11-03)

Qwen3.8-Flash-Next keeps a 28.8 GB n-gram table on disk, memory-mapped by llama.cpp, and every token it processes
gathers 16 rows of it. The rig has 4 GB of RAM, so the page cache keeps only a little of the table, and the rest of
each token's rows come from disk. This run replays 3,210 recorded coding-agent requests (5.0 million processed tokens)
through Flash-Next's own tokenizer, chat template and n-gram hash, and runs the rows they read through simulated page
caches and row caches of 16 MiB to 32 GiB. It answers three questions: how many reads a token costs, how much of the
table a rig's RAM or a bigger host keeps hot, and therefore how fast the table's disk has to be. The tool is
[bench/ngram_table](../../../bench/ngram_table/README.md); the registry row is Q228 in
[quants/experiments.md](../../../quants/experiments.md), and the consequences are in
[flash_next_variants.md](../../research/notes/flash_next_variants.md), section 5.

## Result: 3.6–5.7 reads per generated token; the rig's SATA SSD costs about 1 % of decode

With the rig's page cache of about 2 GiB (R11: ~2.5 GB of the host is free before the server starts, and the server's
buffers take 0.05–0.4 GB at 64k):

| | Agent check (31 runs of 12 tasks, cold at each run) | 86 sessions on one host (other tasks and harness) |
|---|---:|---:|
| Page reads per generated token | **5.70** | **3.59** |
| Page reads per prompt token | 3.33 | 1.44 |
| Generated tokens that read nothing | 52 % | 68 % |
| The first request after the server starts | 2,780 prompt tokens, 32,793 reads | 1,034 tokens, 12,037 reads |
| A later session's first request | 2,324 tokens, 4,464 reads (median) | 1,072 tokens, 289 reads |
| A later turn | 256 prompt tokens, 536 reads (median) | 156 tokens, 47 reads |

With no cache at all a token reads 16.3 pages (16 rows, 2 % of them across a page boundary).

- **The misses are new n-grams, and more RAM barely removes them.** The 8 heads of an n-gram share one hash, so a
  bigram or trigram seen before is a hit in all 8 rows, and a new one misses all 8. Within a 12-task run, page reads
  per generated token fall from 8.66 at 64 MiB to 6.23 at 512 MiB and stop at 5.70 from 2 GiB on: past that size every
  miss is an n-gram the run has not used yet. Over 86 different tasks on one host the cache keeps paying, but slowly:
  3.59 at 2 GiB, 3.00 at 4 GiB, 2.39 at 8 GiB, 2.10 at 16 GiB.
- **A row cache reaches the same floor in 30–45 times less memory.** Caching the 90-byte rows themselves (direct reads,
  as Strata's `ple_reader` and agentionai's `llama-ple-disk.cpp` do) gives the agent check's floor at 32–64 MiB
  (5.82 reads per generated token) and the 86 sessions' 3.09 at 256 MiB, fewer than a 2 GiB page cache.
- **Strata's "up to 82 %" fits agent work.** A 95 MB row cache serves 72 % of all row reads in the agent check (79 %
  of prompt tokens' rows, 64 % of generated tokens') and 84 % over the 86 sessions (91 %, 78 %). ninfer-all's
  33.8 % at 256 MiB is lower than anything in these sessions.
- **No smarter eviction helps at the rig's size.** At 2 GiB, Belady's optimal policy equals LRU in the agent check;
  over the 86 sessions it would read 2.53 pages per generated token against LRU's 3.59. CLOCK is within 0.05 reads of
  LRU for row caches.
- **A hot-row sidecar learned on other sessions saves 16–23 %.** The 2 million rows used most in the other set of
  sessions, packed 45 to a page (189 MB, kept resident), cut generated-token reads from 5.70 to 4.41 in the agent check
  and from 3.59 to 3.03 over the 86 sessions, with the same 2 GiB in all. Prompt-token reads fall 21 % and 11 %.
- **The token embeddings add almost nothing:** 0.03 reads per generated token (273 MB of `token_embd`, one row per
  token, in the same page cache).

### What it costs per disk (est.)

The engine announces a token's rows to the kernel together (`MADV_WILLNEED`), so its 8 or 16 reads overlap. The read
times are R10's measurements ([table storage](../2026-11-03_table_storage/README.md)); the stick's are ssd-tester's
figures, not measured on the rig. The per-token times are my arithmetic.

| Disk | Per generated token | Per prompt token | First request after start | Share of decode (est. 66–86 ms a token) |
|---|---:|---:|---:|---:|
| Internal SATA SSD (152 µs at depth 1, 23,792 reads/s at 32) | 0.15–0.87 ms | 0.06–0.14 ms | 1.4 s | 0.2–1.3 % |
| USB 3 flash stick (367–588 µs, ~2,700 reads/s) | 1.3–3.4 ms | 0.5–1.2 ms | 12 s | 1.5–5 % |
| USB 3 portable hard disk (10.4 ms, 101 reads/s at 32) | 37–59 ms | 14–33 ms | 5.4 min | +43–90 % |

### What it means for where the table lives

- **On the rig, the internal SSD is the place, and it is enough.** The table costs it about 1 % of decode, 0.1 ms
  per prompt token and 1.4 s once per server start. The table moved there on 2026-11-03 (Q435).
- **The rig's RAM is not the lever.** Its ~2 GiB of page cache already reaches the floor of a run; with a quarter of
  it (512 MiB), reads per generated token rise only from 5.70 to 6.23. Context and the MTP draft can take host memory
  from the page cache at little cost.
- **The USB hard disk is ruled out:** it would add 37–59 ms to every generated token and 5 minutes to the first
  request. A USB flash stick would work at a few percent of decode, but is not measured on the rig.
- **A bigger host gains only by keeping the whole table.** Partial caches level off because agent sessions keep
  producing new n-grams: even 16 GiB leaves ~2 reads per generated token. A host with ~29 GB free for the page cache
  holds all of it and reads nothing after the first pass.
- **Engine work on the table's reads (Q229) is not a speed lever while the table sits on an SSD:** the sidecar would
  save ~0.05–0.2 ms a token. A row cache is a memory lever instead: 64–256 MiB of it does what ~2 GiB of page cache
  does.

## How it was measured

- **Sessions.** Two sets, re-rendered for Flash-Next; their text was written by other models:
  - **Agent check:** the 31 runs of the release check with full request bodies (`runs.txt`), from 2026-10-26 to
    2026-10-28: the 12 bugfix tasks of [bench/agent_check](../../../bench/agent_check/README.md) driven by the pi agent,
    written by Flint files (23 runs) and Ornith-1.5-122B files (8 runs). 331 sessions, 1,963 requests.
  - **86 sessions:** the dataset export of the agentic data pipeline's run 1 ([bench/agentic_data](../../../bench/agentic_data/README.md)),
    written by Flint Q8_0 with that pipeline's own harness and tools on the CPython 3.12 and Elixir 1.18 standard
    libraries (49 bug fixes, 29 test-writing tasks, 8 implementations). 1,247 requests, all treated as one run on one
    host, in file order.
- **Rendering and tokens.** Each request was rendered twice with the GGUF's chat template (the prompt, and the prompt
  plus the recorded reply) and tokenized with the model's tokenizer. In every request of both sets, the prompt text is
  a prefix of the prompt-plus-reply text, and the next request's prompt extends it, because the template keeps earlier
  reasoning: every request after a session's first reuses the whole slot, and only its new tokens are processed.
  Agent check: 1,560,763 prompt tokens processed and 1,308,683 tokens generated; 86 sessions: 906,274 and 1,237,636.
- **The hash.** The `qwen4exp` hash of llama.cpp (Miner Llama v3 at `37e717ff6`), with the constants of the file's
  header (`params.txt`). A second implementation in Python gave the same rows for the first 20,000 tokens of each set.
- **The file.** Swift-1.5 GSQ-RCO IQ3_XXS in the three-shard layout of Q435, whose table shard is byte-identical to
  Q2_0's: the table's rows start 192 bytes into their shard, which sets which rows cross a page boundary.
- **Caches.** LRU exactly at every size (stack distances), Belady's optimal policy and CLOCK at six sizes, all cold at
  the start of each run unless the scope says otherwise; pages of 4 KiB, rows of 90 bytes. Linux's page cache is not
  exactly LRU, so the optimal policy's column bounds what any eviction policy could do.
- **Scopes.** "run" starts cold at each run's first session, as one server process would after a reboot; "session"
  starts cold at every session; "all" chains the agent check's 31 runs on one cache. The 31 runs repeat the same 12
  tasks, so "all" overstates what a long-lived host would see across different tasks; the 86 sessions show that case.
- **Sidecar splits.** For the agent check, also learned on the sessions of six tasks and tested on the other six
  (`even->odd`, `odd->even`): generated-token reads fall from 5.89 to 3.65 and from 6.09 to 4.84 with 2 million rows.
  In the 86-session set, neighbouring sessions work on the same module, so its task-parity splits put the same code on
  both sides; only `other->all` (learned on the other set) is a clean test there. `all->all` (learned on the test
  sessions themselves) is an upper bound.
- **Not covered.** MTP: a verify step reads the rows of every drafted token, and a rejected draft's rows are extra
  reads. Flash-Next's own output: its reasoning and replies differ from Flint's and Ornith's, while the prompts' tool
  results and code are the same text.

## Files

Each of `agent/` (the agent check) and `sessions/` (the 86 sessions) holds:

| File | What |
|---|---|
| `tables.md` | the main tables: page and row reads per token by cache size and scope, and per request at 2 GiB |
| `summary.tsv` | tokens processed by kind, page reads per token with no cache, the share of rows across a page boundary, distinct pages per 512-token prompt micro-batch (5,293 and 5,032 of 8,192 row reads, mean) |
| `working_set.tsv` | distinct rows and pages per session, per run and in all |
| `lru_curve.tsv` | LRU hit rate and reads per token for pages, rows (also by bigram and trigram heads) and pages with the token embeddings, by scope and kind, at 16 MiB to 32 GiB |
| `policies.tsv` | LRU, OPT and CLOCK at 95 MB to 4 GiB |
| `token_reads.tsv` | the distribution of page reads per token at 256 MiB to 4 GiB |
| `request_reads.tsv` | per request: prompt tokens processed, tokens generated, page reads at 256 MiB to 4 GiB |
| `sidecar.tsv` | the hot-row sidecar by size (1–16 million rows), budget (512 MiB to 4 GiB) and split |
| `requests.tsv` | per request: the session, turn, prompt tokens, tokens reused from the slot, processed and generated, and the source run's own counts (agent check) |
| `traces.tsv` | the sessions: run, task and tag |
| `params.txt` | the hash constants and the layout of the table and the token embeddings in the file |

`agent/runs.txt` lists the 31 agent-check runs. `run/replay.sh` is the run.

## Commands

On the quant machine, CPU only, at `nice -n 19` and a 256 CPU share for the build (about 5 minutes in all, 5 GB of
RAM):

```bash
SRC=<Miner Llama checkout> REF=37e717ff6 OUT=$WORK/bin JOBS=8 \
  DOCKER_ARGS="--cpu-shares=256 --label pascal-quants=1" bench/ngram_table/build.sh
MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00003.gguf \
TABLE=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00003-of-00003.gguf \
RUNS="$(sed "s#^#$AGENT_RUNS/#" docs/data/2026-11-03_ngram_replay/agent/runs.txt)" \
SESSIONS=$AGENTIC/export/hf/data/sessions.jsonl TOOL=bench/ngram_table BIN=$WORK/bin WORK=$WORK/replay \
OUT=$WORK/tables bash docs/data/2026-11-03_ngram_replay/run/replay.sh
```

## Provenance and licence

The tables are counts computed from the sessions; no session text is published here. The agent-check tasks come from
Exercism's MIT-licensed exercises; the 86 sessions are on Python-2.0 and Apache-2.0 code. The scripts are under this
repository's licence.
