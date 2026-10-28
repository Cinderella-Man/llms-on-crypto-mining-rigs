# The answer store on agentic work: Flint's own past answers as guesses

*Measured 2026-10-27; the paired replay on 2026-10-28.*

Miner Llama's pipelined decoding feeds a guessed next token into the first card while the later cards still finish the
current one, and keeps the guess only if the last card samples the same token, so the output is exactly plain
decoding's ([pipelined speculation](../research/notes/pipelined_speculation.md)). The guesses cost no card time when
they come from the host. The built-in guesser (`cont`) copies from the request's own text and its own continuation
counts. The **answer store** (`--pipe-store FILE`, registry row Q161) adds a host-side lookup of Flint's own past
continuations: a suffix array over the tokens Flint generated in earlier sessions, asked after the copy lookup and the
request's own counts. It is in the v2.0 engine already; until now no store existed that could be published.

This page measures a store built from 1,177 agent sessions that Flint wrote on code that may be published, first
offline on held-out sessions, then on the rig with the agent check, the release metric, and finally with the agent
check's own requests replayed to both setups, so that both generate the same tokens. **Result: +1.4 % per generated
token (95 % interval 0.4–1.9 %) on the agent check, at no cost in card or host memory; it ships with v2.1.**

## Setup

| Item | Value |
|---|---|
| Rig | the [reference rig](../mining_rig.md): nine P106-090 6 GB cards, each on a PCIe 1.0 x1 link; a 2-core Celeron with 4 GB of RAM |
| Engine | Miner Llama [v2.0](https://github.com/Cinderella-Man/llama.cpp/releases/tag/miner-llama-v2.0) (`37e717ff6`), unchanged |
| Model | Flint Q8_0 (Unsloth), nine cards, 262,144 tokens of context |
| Release line | `-c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4` |
| Store arm | the release line plus `--pipe-store agentic-run1.pstore` |
| Measurement | [bench/agent_check](../../bench/agent_check/README.md): the pi coding agent fixes 12 small injected bugs (6 Python, 6 JavaScript Exercism exercises) with real tool runs; time per task, turns, time to first token, generation tokens per second over every request |

## The store

| Item | Value |
|---|---|
| Sessions | 1,177 coding-agent sessions of [agentic data run 1](../research/notes/agentic_data_pipeline.md) (2026-10-26/27): Flint writes unit tests for, repairs injected bugs in, and re-implements removed functions of single modules of the Python 3.12 standard library (PSF-2.0) and Elixir 1.18.4's library (Apache-2.0), in network-less sandboxes, with both tool profiles (`agx` and the pi agent's `read`/`write`/`edit`/`bash`) |
| Writers | Flint only: the makers' NVFP4 GGUF in vLLM (1,091 sessions) and Unsloth's Q8_0 in llama-server (86) |
| Content | the generated tokens of each of the 18,885 turns (reasoning, answer, tool calls) as Flint's token ids, each turn followed by a separator; no prompts, no tool results |
| Size | 15,445,634 tokens with separators, 123.6 MB (8 bytes per token: the tokens and a suffix array over them), memory-mapped by the engine |
| Built by | [datastore_corpus.py](../../bench/ngram_replay/datastore_corpus.py) and `datastore build` ([datastore.cpp](../../bench/ngram_replay/datastore.cpp)), as [bench/agentic_data/run.sh](../../bench/agentic_data/run.sh)'s checks step does |
| Gated | each step by [check_sources.py](../../bench/staircase_drafter/check_sources.py); [chain.py](../../bench/agentic_data/chain.py) walks the store's 19 gate records back to the workspaces' licence scans: every record passed, every parent's hash matches, every file unchanged, one `sources.json` |
| Held-out tests | the sessions gate excluded any session sharing a 12-word span with Q51's prompts or with any file of the agent check's tasks. [heldout_check.py](../../bench/agentic_data/heldout_check.py) re-checks the store's sessions with 8-word spans and the exercise names: no 12-word span, no bugfix exercise name. The only 8-word matches are counting sequences (`2, 3, 4, 5, 6, 7, 8, 9,`), "greater than or equal to the length of" and a Python `__enter__`/`__exit__` idiom, all against the polyglot task set, which the agent check below does not run |

## Offline: held-out agent sessions

The drafter's held-out split ([heldout.py](../../bench/agentic_data/heldout.py)) leaves out every session of every 20th
module; 40 of those sessions (38 Python, 2 Elixir; 648 turns, 680,600 guessed positions, contexts up to 65,536 tokens)
are replayed here against a second store built the same way from the other 1,108 sessions only (14.3 M tokens).
[store_agentic.py](../../bench/ngram_replay/store_agentic.py) turns each generated span into one request, as
llama-server sees an agent loop, and follows the engine's guess sources: the copy lookup (a match of 4 to 12 tokens),
the request's own continuation counts over the prompt's last 32,768 tokens and the output (contexts of 2 to 8 tokens),
the store (longest context up to 32 tokens; at least 2 tokens, top continuation at least 30 % of its occurrences), and
the request's counts of any context.

One token ahead on the true prefix:

| Guesser | Right | Fresh text right (52.2 % of positions) | Copied text right | Reasoning (64.7 %) | Tool calls (31.8 %) | Answer text (3.5 %) |
|---|---|---|---|---|---|---|
| `cont` (copy lookup, request counts) | 56.9 % | 27.5 % | 89.1 % | 44.4 % | 84.3 % | 39.8 % |
| `cont` + store | **59.6 %** | **32.4 %** | 89.3 % | 47.2 % | 86.1 % | 47.9 % |

Which source guessed, and how often it was right (share of all positions / right):

| Source | `cont` | `cont` + store |
|---|---|---|
| Copy lookup | 52.9 % / 81.6 % | 52.9 % / 81.6 % |
| Request counts, contexts of 2+ tokens | – | 23.6 % / 42.7 % |
| Store | – | 13.5 % / 34.9 % |
| Request counts, any context | 42.5 % / 32.2 % | 8.2 % / 19.3 % |
| No guess | 4.5 % | 1.8 % |

- **The store adds 2.7 points of right guesses** (4.9 on fresh text). It answers where neither the copy lookup nor the
  request's own 2-token contexts do (23.5 % of positions): there `cont` guesses from 1-token contexts and is right 15 %
  of the time, the store first and those contexts after it 27 %.
- **The engine's order is the right one:** the store before the request's 2-token contexts gives 57.8 %, below both.
- **Thresholds:** the engine's (2 tokens, 30 %) ties for best. Right guesses over the grid:

  | Minimum context | share ≥ 0 | ≥ 0.2 | ≥ 0.3 | ≥ 0.5 | ≥ 0.7 |
  |---|---|---|---|---|---|
  | 1 token | 59.1 % | 59.4 % | 59.4 % | 59.3 % | 58.9 % |
  | 2 tokens | 59.6 % | 59.6 % | **59.6 %** | 59.4 % | 59.0 % |
  | 3 tokens | 59.4 % | 59.4 % | 59.4 % | 59.2 % | 58.8 % |
  | 4 tokens | 58.9 % | 58.9 % | 58.9 % | 58.8 % | 58.5 % |
  | 6 tokens | 58.1 % | 58.1 % | 58.1 % | 58.1 % | 57.9 % |

Pipelined speed in [pipeline_sim.py](../../bench/ngram_replay/pipeline_sim.py)'s model of the release split (layers
6/6/5/5/6/5/5/6/5 and the output head on the last card; 34.1 ms per token without guessing, the agent check's 29.3 tok/s
with no guess at all on 2026-10-26):

| Depth | `cont` | `cont` + store | Gain |
|---|---|---|---|
| 3 | 1.80× (52.8 tok/s) | 1.88× (55.0 tok/s) | +4.1 % |
| 4 (release) | 1.87× (54.8 tok/s) | 1.95× (57.2 tok/s) | **+4.3 %** |

The model is optimistic in absolute terms: the agent check measures 1.54× for `cont` at depth 4 (45.0 against 29.3
tok/s), so the gain to expect on the rig is the relative one, about 4 %. Per segment at depth 4: reasoning +3.7 %,
tool calls +5.2 %, answer text +11.6 %.

**Store size** (prefixes of the shuffled training turns, depth 4):

| Store | 1 M tokens | 2 M | 4 M | 8 M | 14.3 M |
|---|---|---|---|---|---|
| Right guesses | 58.7 % | 59.0 % | 59.2 % | 59.4 % | 59.6 % |
| Speed in the model | +2.9 % | +3.3 % | +3.7 % | +4.0 % | +4.3 % |

Each doubling adds about 0.4 %, so more sessions help a little and steadily.

## Offline: Q51

Q51 (38 transcripts of hand-written chat, code, edit and agent requests, short prompts) with the full store, release
split, depth 4: 1.47× → 1.49× overall (+1.2 %). Agent turns gain 9.3 % (1.32× → 1.44×; their tool calls 1.72× →
4.02×, since the store's sessions used the same four tools), chat 1.0 %, code 1.8 %, edits 0.3 %. A store of agent
sessions pays on agent work; the earlier store of mixed answers (Q161, not published) gained ~5 % on chat, code and
edits as well.

## On the rig: the agent check

The store was copied to the rig's internal disk and checked by SHA-256. Both arms ran in one slot, the store arm first,
then the release line alone; the release line had also run alone at 14:25 the same day.

| Arm | Tasks passed | Minutes, all tasks | Minutes per task (median) | Turns (median) | TTFT per turn (median, p90) | Generation tok/s | Tool calls / failed / malformed |
|---|---|---|---|---|---|---|---|
| Release line (14:25) | 12 / 12 | 22.7 | 1.2 | 6.0 | 2.30 s, 6.52 s | 45.0 | 66 / 0 / 0 |
| **Release line + store** (17:18) | 12 / 12 | 21.5 | 1.0 | 5.0 | 2.51 s, 6.46 s | **46.9** | 61 / 0 / 0 |
| Release line (17:50, control) | 12 / 12 | 23.7 | 1.1 | 6.0 | 2.77 s, 6.13 s | 46.5 | 65 / 0 / 0 |

Generation tok/s is the server's generated tokens over its generation time, summed over every request of the arm.

- **Against the same-slot control the store arm is 0.9 % faster, against the earlier run 4.2 %.** The offline estimate
  is +4.3 %. One live run per arm cannot separate a gain of this size: the agent writes different sessions in every
  run (sampling at temperature 1), and how much of a session the guessers can predict moves the speed as much as the
  store does.
- **Guesses** (the server's counters over all requests, [guesses.tsv](../data/2026-10-27_answer_store/agent/guesses.tsv)):
  with the store 118,533 guesses were launched, 19,471 of them (16 %) from the store, and 40.3 % of those whose sample
  arrived were right. Without the store 39.1 % were right in the earlier run and 41.6 % in the control: the control's
  sessions were easier to guess, which is the content effect above.
- **Host memory:** the store is memory-mapped (124 MB of page cache, no card memory). The lowest MemAvailable was
  691 MB and swap grew by 267 MB, against 687 MB and 263 MB in the control, inside the release line's range on the same
  day and the day before (581–675 MB, +254–454 MB).
- **Same output rules:** pipelined decoding keeps every sampled token, so the store changes speed only; the agent fixed
  all 12 bugs in every arm, with no failed or malformed tool calls.

### Paired replay: the same tokens with and without the store (2026-10-28)

Six traces of the control arm (picked across its range of task times: space-age and phone-number in JavaScript,
complex-numbers, affine-cipher, list-ops and scale-generator) were re-sent request by request to the release line with
the store and then without it ([replay.py](../../bench/agent_check/replay.py) through `release_check.sh`'s replay
mode): the same messages and tools, a fixed seed per request, and the recorded reply length as the limit. Pipelined
decoding is exact, so both setups generated the same tokens; all 36 requests agree on every token count, cache hit,
finish reason and tool call (20,544 generated tokens each). The difference in speed is the store's alone.

| Setup | ms per generated token | tok/s | ms per prompt token | Guesses launched | from the store | right | wrong |
|---|---|---|---|---|---|---|---|
| Release line | 22.87 | 43.7 | 5.13 | 51,655 | – | 7,128 | 11,131 |
| Release line + store | 22.55 | 44.3 | 5.12 | 53,180 | 8,168 | 7,504 | 11,449 |
| **Ratio, store / release line** (95 % interval over traces) | **0.986 (0.981–0.996)** | | 0.998 (0.994–1.001) | | | | |

- **The store makes generation 1.4 % faster** (0.4–1.9 %) on the agent check's own requests, and leaves prompt reading
  unchanged. It adds 5.3 % right guesses (376 more over the six traces), and 2.9 % more wrong ones.
- **The model is too optimistic by about three times:** it gave +4.3 % on the held-out sessions. It also gives `cont`
  1.87× where the rig measures 1.54×, and it does not count the host's time per store lookup on the 2-core Celeron.
- **This is why the live arms disagreed:** +1.4 % is below the spread between single live runs (45.0 and 46.5 tok/s
  for the same setup on the same day).

## Verdict

The store ships with Miner Llama v2.1 as an optional download: **+1.4 % (0.4–1.9 %) on the agent check**, measured on
identical tokens, with no card memory, no change to the host's memory, the same output, and one flag on the v2.0 engine.
The engine's compiled thresholds are the right ones, so v2.1 needs no new build for it. It is a small gain, and it is
not a general-purpose one: offline on chat, code and edit requests (Q51) it gains 0.3–1.8 %.

## Publishing the store

- **What it holds:** the token ids of the text Flint generated in the 1,177 sessions (reasoning, answers, tool calls),
  and a suffix array over them. No prompts, no tool results, no harness texts except where Flint repeated them.
- **Licence:** Flint's weights are Apache-2.0 and its card sets no terms on outputs; every session's code base, writer
  and harness text is approved in [sources.json](../../bench/staircase_drafter/sources.json), and every stage is gated
  ([chain](../data/2026-10-27_answer_store/chain.tsv)). Where Flint's edits copy library code, that code keeps its
  licence: the Python standard library (PSF, SPDX `Python-2.0`) and Elixir's library (Apache-2.0). So the store ships
  with the same licence files as the session dataset's export: `LICENSE` (Apache-2.0, for the store's own arrangement),
  `LICENSES/Python-2.0.txt`, `LICENSES/Apache-2.0.txt` and a `NOTICE` with CPython's and Elixir's attribution (Elixir's
  NOTICE file included) and Flint's.
- **Held-out tests:** no text of the agent check's tasks or of Q51 is in the store's sessions (above), so the numbers
  on this page are not inflated by the store having seen the test.
- **It works with every Flint file:** it holds Flint's token ids, so Q8_0, Flint-Pascal-Q5 and Unsloth's quants share
  it.

## Using it (Miner Llama v2.0, no new engine)

1. Download the store (124 MB) and check its SHA-256: `86c5dedb2bc692c14999dc68c1a837624729ca982486286fceeab88a3ed8c13c`.
2. Keep it on the rig's internal disk, as in this test: a lookup reads scattered entries of the memory-mapped file
   (~84 suffix-array probes and a few hundred entries scanned, Q161), and a store on the USB stick was not tried.
3. Add one flag to the release line:

   ```bash
   llama-server -m flint-Q8_0.gguf -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs \
     --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4 --pipe-store /path/to/agentic-run1.pstore
   ```

   The server's log says `continuation store <file>: 15445634 tokens` at load, and each request's timing line counts
   the guesses that came from the store. The store is used with the default guesser (`--pipe-guess cont`) and needs
   `--pipe-depth` (pipelined decoding) on.

## Where this stopped (2026-10-28)

**Done.** All three measurements: the offline estimate (registry Q200), the two live arms (Q201) and the paired replay
(Q202). The replay's decision rule was set before it ran, and the result picked the first row:

| Paired ratio, store / release line | Decision |
|---|---|
| **interval entirely below 1.00 (measured: 0.986, 0.981–0.996)** | **ship the store with v2.1 and report the replay's gain as its agentic number** |
| interval includes 1.00 | the store is not shown to help on the agent check: it does not go into the release line |
| interval entirely above 1.00 | do not ship it: the store's lookups cost host time the model does not count |

The tools are in the repository: `datastore_corpus.py --sessions`, `store_agentic.py`, `chain.py`,
`heldout_check.py`, and `release_check.sh`'s replay mode with `replay.py pick --task-set`. The data and scripts are in
[docs/data/2026-10-27_answer_store](../data/2026-10-27_answer_store/README.md).

**Half-done.** Nothing measured is half-done. Not done yet: the store is not published, and `--pipe-store` is not in
v2.1's documented command line. Both wait on the owner's decision.

**Next steps, in order:**

1. **Publish the store** (on the owner's yes). Put a folder together holding `agentic-run1.pstore`, a `SHA256SUMS`
   line, `LICENSE` (Apache-2.0), `LICENSES/Python-2.0.txt`, `LICENSES/Apache-2.0.txt` and a `NOTICE`. The last three
   are copied from the session dataset's export (`export/hf/` of [bench/agentic_data](../../bench/agentic_data/README.md)),
   with the NOTICE's first paragraph saying the file holds the token ids of Flint's generated text. Add a section to
   the model card with [Using it](#using-it-miner-llama-v20-no-new-engine), then upload:

   ```bash
   sha256sum agentic-run1.pstore > SHA256SUMS
   hf upload CinderellaMan/Flint-Pascal-Q5-GGUF <that folder> answer-store
   ```

2. **Add `--pipe-store` to v2.1's documented command line** (README, the Flint model page, the release notes), with
   +1.4 % on the agent check as its number.
3. **Measure the host's time per store lookup** on the rig's Celeron, which the model leaves out. This needs a timing
   mode in [datastore.cpp](../../bench/ngram_replay/datastore.cpp) (it counts probes today): time `ds_longest` on the
   contexts of the six replay traces, on the rig with nothing else running. If the lookups take a real share of a
   token's 22.6 ms, a cheaper lookup (fewer suffix-array probes per guess) raises the gain without new data.
4. **A store from more sessions**, once later data runs export (the model gives ~+0.4 % per doubling, so expect less on
   the rig). Build it with [build_stores.sh](../data/2026-10-27_answer_store/run/build_stores.sh) (`STORES=full`, `RUN` =
   the new export), then rerun the paired replay with the new file in place of the store arm's:

   ```bash
   ENV_FILE=<release.env> STORE_DIR=<the new store's folder on the rig> \
     REPLAY_LOG=<client folder>/out-store-base/proxy.jsonl CLIENT_ROOT=<client folder> OUT=replay2 \
     bash docs/data/2026-10-27_answer_store/run/rig_replay.sh
   ```

   (`rig_replay.sh` checks the store's SHA-256 against this store's; change the hash in the script for a new file.)
5. **The store on the USB disk** instead of the internal disk: the same replay with `STORE_DIR` on the model disk.
6. **Flint's MTP head behind the store**, once one guess costs less (the cheaper guess, [drafter.md](../drafter.md)
   section 8, item 0). The same six traces with `--pipe-guess staircase --pipe-draft <head> --pipe-store <store>` as
   one more arm against the store arm: add the arm to `rig_replay.sh`'s `case`.

**Open questions:**

- Why the model is ~3× too optimistic (+4.3 % against +1.4 %): host time per lookup (step 3), or the model's general
  optimism (it gives `cont` 1.87× where the rig measures 1.54×).
- Whether the gain holds for Flint-Pascal-Q5 and the 4-bit files. They decode faster, so a fixed host cost per lookup
  weighs more. Not measured.
- Whether the writer matters: 93 % of the store's sessions were written by the makers' NVFP4 file in vLLM, and the rig
  runs Q8_0.
- Card counts other than nine: not measured.

**Kill criteria:**

- A new store, setting or placement replaces this one only if its paired replay against this store's arm has an
  interval entirely below 1.00 on the same six traces.
- The store leaves the documented command line if a later release line's paired replay with and without it gives an
  interval that includes 1.00.
- Step 3 stops if the lookups take under ~1 % of a token's time: the gap is then the model's, and only data or a
  different guesser can add more.

## Reproduce

Data, tables and the exact commands: [docs/data/2026-10-27_answer_store](../data/2026-10-27_answer_store/README.md).
