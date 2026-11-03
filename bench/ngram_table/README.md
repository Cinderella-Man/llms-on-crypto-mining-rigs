# Flash-Next's n-gram table: reads per token from replayed agent sessions

Qwen3.8-Flash-Next keeps a 28.8 GB n-gram embedding table (`per_layer_token_embd`, 320,001,536 rows of 160 weights)
that llama.cpp leaves memory-mapped on disk. Every token the model processes, prompt or generated, gathers 16 rows of
it on the CPU. On a host with little RAM most of those rows come from disk, so how many reads a token costs decides how
fast the table's disk has to be. This folder counts them: it replays recorded coding-agent sessions through the
model's own tokenizer, chat template and n-gram hash, and runs the rows each token reads through simulated caches.
The first results are in [docs/data/2026-11-03_ngram_replay](../../docs/data/2026-11-03_ngram_replay/README.md)
(registry row Q228 in [quants/experiments.md](../../quants/experiments.md)).

## What a token reads

The hash is the one in llama.cpp's `qwen4exp` graph (`llm_graph_input_qwen4exp_ple::set_input`), with its constants
from the GGUF header (`qwen4exp.ple.*`):

- **16 heads:** 8 hash the bigram (the token and the one before it), 8 the trigram (one more token back). Each head owns
  a contiguous range of about 20 million rows, so a token's 16 rows are always distinct and far apart.
- **One hash per n-gram:** the 8 heads of an n-gram share one mixed value and differ only in the prime they take it
  modulo. An n-gram seen before finds all 8 of its rows in a row cache; a new one misses all 8. A token's rows therefore
  miss in blocks of 8: none, one n-gram's or both. In a page cache, some rows of a new n-gram land on pages that other
  rows already brought in.
- **Row size:** 90 bytes in IQ4_NL (5 blocks of 32 weights). A 4 KiB page holds 45.5 rows, but the hash scatters them,
  so a page read for one row rarely serves another. About 2 % of rows cross a page boundary and need two pages.
- **How llama.cpp reads them:** the table is mapped with `MADV_RANDOM` (no readahead: a fault reads one page), and the
  rows of each micro-batch are announced with `MADV_WILLNEED` before the gather, so a token's 8 or 16 reads go to the
  disk together, and a 512-token prompt micro-batch issues thousands at once.
- **Token embeddings:** each token also reads one row of `token_embd` (1,100 bytes, 273 MB in all), through the same
  page cache.

## The replay

1. **`replay.py prepare`** renders every recorded request twice with the GGUF's own chat template (jinja2 with the
   filters Hugging Face and llama.cpp use): the prompt as llama-server builds it, and the prompt followed by the
   recorded reply. Tool-call arguments become objects, as llama-server passes them. Two sources:
   - agent-check runs ([bench/agent_check](../agent_check/README.md)): `proxy.jsonl` and `proxy.jsonl.bodies.gz`, one
     session per `<arm>/<task>/<attempt>/<try>` tag, sessions in the order they ran;
   - a sessions JSONL with `messages` and `tools` per line, as [bench/agentic_data](../agentic_data/README.md)
     exports: every assistant message is the reply to the messages before it.
2. **`ple_tok`** tokenizes both texts with the model's tokenizer through libllama (vocabulary only), as llama-server
   tokenizes a prompt.
3. **`replay.py stream`** writes the tokens the server processes, with its prompt cache modelled for a hybrid
   model (Flash-Next's 36 Gated DeltaNet layers can only resume where a context checkpoint was kept):
   - every session starts on a cold slot;
   - a request whose tokens extend the slot (the previous prompt and reply, tokenized as one text) processes only the
     new tokens;
   - a request that diverges resumes from the last kept checkpoint at or before the divergence (one per earlier
     prompt end, the last 4 kept, as `--ctx-checkpoints 4`) or from zero;
   - the reply's tokens are decoded up to the stop token, which is sampled but never processed.

   Flash-Next's template keeps the reasoning of earlier turns, so in the recorded sessions every request extends the
   slot.
4. **`replay.py check`** recomputes the first 20,000 tokens' rows with a second implementation of the hash (Python
   integers masked to 64 bits); `ple_sim` stops if any row differs.
5. **`ple_sim`** hashes every processed token into its rows and builds three access streams: the table's 4 KiB pages
   (the page cache), its rows (a row cache with direct reads), and the table pages with the token-embedding pages
   (one shared page cache). It writes:

   | File | What |
   |---|---|
   | `summary.tsv` | tokens by kind, page reads per token with no cache, the share of rows that cross a page, distinct pages per full 512-token prompt micro-batch |
   | `working_set.tsv` | distinct rows and pages per session, per run and in all, and the largest share of the table's pages touched |
   | `lru_curve.tsv` | LRU hit rate and reads per token for every stream, scope and kind, at 16 MiB to 32 GiB; for rows also by bigram and trigram heads |
   | `policies.tsv` | LRU, Belady's optimal policy (OPT) and CLOCK side by side at 95 MB to 4 GiB, cold at each run |
   | `token_reads.tsv` | the distribution of page reads per token (mean, percentiles, share of tokens with 0–20 reads) at 256 MiB to 4 GiB |
   | `request_reads.tsv` | per request: prompt tokens processed, tokens decoded, and their page reads at 256 MiB to 4 GiB |
   | `sidecar.tsv` | a static hot-row sidecar: the K rows used most in one half of the tasks (or in another set of sessions) packed densely and kept resident, the rest through an LRU page cache of the remaining budget, tested on the other sessions |

   LRU is exact for every size at once: each access's stack distance (how many distinct items were used since the
   same item's previous use) comes from a Fenwick tree over access times, and a cache of C items hits when it is
   below C. Three scopes: cold at every session, at every run (a run's sessions in the order they ran, as one server
   would see them), or once for all the sessions given.
6. **`replay.py report`** writes the main tables as Markdown (`tables.md`).

## Running it

CPU only; the whole replay of ~5 million tokens takes about 5 minutes and 5 GB of RAM.

```bash
SRC=<llama.cpp checkout that knows qwen4exp> REF=<commit> OUT=<bin dir> bench/ngram_table/build.sh
MODEL=<shard 1 of a Flash-Next GGUF> TABLE=<the shard with per_layer_token_embd> \
RUNS="<agent-check run folders>" SESSIONS=<sessions.jsonl> TOOL=bench/ngram_table BIN=<bin dir> \
WORK=<scratch> OUT=<tables> bash docs/data/2026-11-03_ngram_replay/run/replay.sh
```

`build.sh` builds a CPU-only libllama from `SRC` at `REF` and links `ple_tok` against it, in the
`nvidia/cuda:12.9.2-devel-ubuntu24.04` image the fork's builds use (`IMAGE` overrides it). Only the GGUF headers are
read: the table shard gives the table's offset and row size, shard 1 the hash constants, the template, the tokenizer
and the token embeddings' layout.

## Files

| File | What |
|---|---|
| [replay.py](replay.py) | rendering, the prompt-cache model and token stream, the hash check, the Markdown report |
| [ple_tok.cpp](ple_tok.cpp) | the tokenizer, multithreaded, on libllama with `vocab_only` |
| [ple_sim.cpp](ple_sim.cpp) | the hash and the caches (LRU stack distances, OPT, CLOCK, the sidecar); standard library only |
| [build.sh](build.sh) | the Docker build of both |
