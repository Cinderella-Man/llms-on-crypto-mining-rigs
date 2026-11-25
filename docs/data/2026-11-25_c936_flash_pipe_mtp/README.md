# Flash-Next with its MTP head as the pipe's guesser, on a P106 (C936, Q1675; 2026-11-25)

**Passed, but slow.** Pipelined decoding guesses tokens ahead and runs them through the cards as a pipeline. Here its
guesses came from Qwen3.8-Flash-Next's own MTP head (`--pipe-guess mtp`), added to Flash-Next's v5 preset. It loads at
65,536 tokens at depths 1, 2 and 3, and its replies are plain decoding's token for token at every depth, with the guess
chain kept on a hit (`--pipe-mtp-walk 1000`) and with the pipe kept on under a grammar (`LLAMA_PIPE_GRAMMAR=1`). The
JSON-schema request that hung the server before pull request 3 finished on every load. **But every depth is slower
than plain decoding without any draft, and slower again than the preset's MTP draft**:
- **Speed:** on the 256-token reply, 12.65 tok/s at depth 1, 9.66 at depth 2 and 7.00 at depth 3, against 18.07 for
  plain decoding and 20.28 for the preset.
- **The guesses are not the problem:** the head was right on 181 of 254 checked guesses at depth 1 (71.3 %), about the
  70 % the pipe's estimate assumed (Q554, est. 2.30× at depth 3).
- **Where the time likely goes:** the pipe step. It runs no CUDA graph, re-plans a pipe slot's graph about 1.5 times a
  token at depth 3, and adds about two MTP-head rows a token at depth 1 (8.950 ms each) and about four at depth 3
  (6.893 ms each).

So on today's engine the MTP-guessed pipe cannot beat the MTP draft on Flash-Next, and the agent check that was to
follow this row (Q1676) does not run.

Registry [Q1675](../../../quants/experiments.md) (this check). The guesser is pull request 7's part 2 (Q1610); the fault
it had to wait for is Q1670 ([2026-11-21_c932_pipe_logits](../2026-11-21_c932_pipe_logits/README.md), checked on a P106
in [2026-11-24_c934_pipe_fix_p106](../2026-11-24_c934_pipe_fix_p106/README.md)); the hang is pull request 3's
(Q1610's row). New rows from this check: Q1867 (where the pipe step's time goes) and Q1868 (a schema request under the
MTP guesser takes about ten times the preset's time a token).

## What was run

| | |
|---|---|
| Rig | the second rig: nine P106-090 6 GB cards (PCIe 1.0 x1 each), a 4 GB host. Its clock offsets were set back to 0 on every card and its watchdog stopped at 19:40 before the first load, so every card ran at stock; nothing here is timed |
| Build | Miner Llama `pascal-p106-v5-cand-b12674` at `eee615cdb` (the v5 candidate tree, which holds the Flash-Next pipe fix `0c83f4a86` and pull requests 3 and 7); `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67` |
| Model | Swift GSQ-RCO Q2_0: UkisAI's Swift-1.5 file of Qwen3.8-Flash-Next, with its n-gram table as the third shard, and the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf`, the v5 preset's files |
| Settings | the v5 preset [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it was on `v5` at `2c01a6f0` (copied into [run/settings.env](run/settings.env), md5 `b8e950f2de8c7906c156d582416731bf`): 65,536-token context, `-bs`, the MTP draft three tokens a step, probabilistic verification |
| Driver | [run/pipe_mtp.sh](run/pipe_mtp.sh): one server load per configuration, the requests of [run/client.py](run/client.py), each card's MiB after the load and at the end; tables by [run/tables.py](run/tables.py) |

The loads, in order (19:42–20:20, and the extra plain load 20:21–20:26):

| Config | Added to the preset |
|---|---|
| `preset` | nothing: the MTP draft, three tokens a step |
| `plain` | the draft off (no `-md`, no `--spec-*` flags): plain one-token decoding |
| `d1`, `d2`, `d3` | `--pipe-depth 1`, `2` or `3` with `--pipe-guess mtp` |
| `walk` | `--pipe-depth 3 --pipe-guess mtp --pipe-mtp-walk 1000` (depth 3 was the deepest that loaded) |
| `grammar` | `--pipe-depth 3 --pipe-guess mtp` with `LLAMA_PIPE_GRAMMAR=1` |
| `plain_probs` | plain again, asking for the top two tokens on the 512-token and JSON-schema replies too, to price the points where replies part |

The requests, through `/completion` on each load, one after another, each with prompt caching off and at
temperature 0 (greedy). Each prompt is a single user message put through the model's chat template (`/apply-template`):
- **`reply256`:** 256 tokens of "Write a Python class implementing an LRU cache with a per-entry time-to-live, with
  unit tests." ([bench/pipespec/prompts/code.txt](../../../bench/pipespec/prompts/code.txt)).
- **`reply256_probs`:** the same request with the top two tokens at every position (`n_probs` 2), on the reference
  loads only, since token probabilities take a slot out of the pipe.
- **`reply512`:** 512 tokens of "Explain how a Bloom filter works, when it is the right tool, and when it is not."
  ([chat.txt](../../../bench/pipespec/prompts/chat.txt)). Its generation tok/s is the diagnostic.
- **`schema`:** up to 256 tokens under a JSON schema (an object with a name and a list of one to six file names) for a
  question about the same project. This is the request that hung the server under `--pipe-guess mtp` before pull
  request 3. The client gave it 300 s.

## Results

### Every depth loads

| Config | Loaded in (s) | The server's pipe set-up |
|---|---:|---|
| `preset` | 229 | - |
| `plain` | 213 | - |
| `d1` | 229 | "the MTP head guesses (rows of 10240)", "up to 1 guesses (mtp, copy >= 4, floor 0), 4 pipe slots" |
| `d2` | 229 | the same, up to 2 guesses, 6 pipe slots |
| `d3` | 229 | the same, up to 3 guesses, 8 pipe slots |
| `walk`, `grammar` | 229 each | as `d3` |

Each card's MiB after the load (of 6,144), and the host's available MiB after the load and after the requests
([mib.tsv](mib.tsv) has both points for every card):

| Config | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | host, load | host, end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `preset` | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | 840 | 259 |
| `plain` | 4,961 | 5,025 | 4,865 | 5,041 | 4,139 | 4,881 | 4,319 | 4,893 | 2,359 | 1,021 | 637 |
| `d1` | 5,193 | 5,245 | 5,095 | 5,261 | 4,357 | 5,111 | 4,527 | 5,123 | 5,715 | 761 | 175 |
| `d2` | 5,309 | 5,353 | 5,211 | 5,369 | 4,465 | 5,227 | 4,629 | 5,239 | 5,805 | 756 | 63 |
| `d3` | 5,425 | 5,463 | 5,327 | 5,479 | 4,575 | 5,343 | 4,733 | 5,355 | 5,897 | 729 | 379 |

On card 0, depth 1 adds 186 MiB over the preset and each further depth 116 MiB (cards 1 to 7 within 14 MiB of
card 0's step, [mib.tsv](mib.tsv)); card 8, which holds the draft, adds 170, 90 and 92 MiB. At the end of `d3`'s requests card 8 held 5,947 MiB, 197 MiB below its 6,144. The host
fell to 63 MiB available at the end of `d2`'s requests: a depth-3 pipe fits the cards at 65,536 tokens, with little to
spare on card 8 and the host.

### The replies are plain decoding's

| Config | `reply256` against plain | `reply512` against plain | `schema` against plain | the same three against the preset |
|---|---|---|---|---|
| `preset` | parts at token 136 | parts at token 43 | parts at token 13 | - |
| `d1`, `d2`, `d3`, `walk` | **the same** | **the same** | **the same** | part where plain parts |
| `grammar` | **the same** | **the same** | parts at token 23 | part where plain parts |
| `plain_probs` | the same | the same | the same | as plain |

Every place where two replies part is a near tie, and the other token is always the second choice there
([parts.tsv](parts.tsv)). The log-probabilities below are read on the path of the reference named:

| Who parts | Request, token | Plain decoding's top two there | The other path's top two |
|---|---|---|---|
| the preset (and so every pipe config against it) | `reply256`, 136 | " expiration" -1.0007, " expire" -1.0324 | the preset's: " expire" -0.9994, " expiration" -1.0161 |
| the preset | `reply512`, 43 | " membership" -1.3120, " insert" -1.4555 | not asked |
| the preset | `schema`, 13 | "\_\_" -1.2729, "l" -1.3369 | not asked |
| `grammar` | `schema`, 23 | "tests" -1.4265, "test" -1.4787 | not asked (token probabilities take a slot out of the pipe) |

The preset itself parts from plain decoding at such ties: its verify passes read several tokens at once, so its sums
can differ from a one-token pass in the last bits. [C964](../2026-11-25_c964_flash_mtp_greedy_p106/README.md) found the
same that day on a P106 on three other requests. The pipe evaluates every token alone, as plain decoding does, and
gives plain's tokens on plain text at every depth. Under the grammar switch the schema request mixes pipe steps with
batched passes (two grammar rejections, see below), and its one parting is at a tie of 0.052. Every JSON reply is
valid and fits the schema: plain's `{"name": "lru_cache_ttl", "files": ["__init__.py", "cache.py",
"tests/test_cache.py", "README.md", "setup.py"]}`; `grammar`'s writes "test_cache.py" for "tests/test_cache.py" and
stops after "README.md".

### The guesses, and the speed (a diagnostic)

From the server's stats line per request ([guesses.tsv](guesses.tsv), [replies.tsv](replies.tsv)). "Guesses" counts
every token guessed; "right" and "wrong" are the guesses checked against the sampled token (one per generated token
after the first two); the rest were dropped unchecked behind a wrong one.

| Config | `reply256`: guesses, right, wrong | MTP rows, ms a row | `reply256` ms a token (tok/s) | `reply512`: right, wrong | `reply512` ms a token (tok/s) |
|---|---|---|---:|---|---:|
| `preset` | (draft: 144 of 333 drafted tokens accepted) | - | 49.31 (20.28) | (312 of 596) | 41.92 (23.85) |
| `plain` | - | - | 55.35 (18.07) | - | 54.78 (18.26) |
| `d1` | 254, 181, 73 | 509, 8.950 | 79.06 (12.65) | 330, 180 | 68.27 (14.65) |
| `d2` | 342, 166, 88 | 762, 7.083 | 103.47 (9.66) | 311, 199 | 90.67 (11.03) |
| `d3` | 431, 165, 89 | 1,012, 6.893 | 142.84 (7.00) | 308, 202 | 119.43 (8.37) |
| `walk` | 445, 158, 96 | 542, 7.052 | 137.98 (7.25) | 299, 211 | 118.11 (8.47) |
| `grammar` | 431, 165, 89 | 1,012, 6.932 | 128.16 (7.80) | 308, 202 | 108.36 (9.23) |

- **The guesser is good:** 71.3 % of checked guesses right at depth 1 on the code reply (64.7 % on the Bloom-filter
  reply), 65.0 % at depth 3 (60.4 %). The walk halves the MTP rows (542 against 1,012) and is right a little less
  often (158 against 165).
- **The pipe step is slow:** a pipelined token costs more than a plain one at every depth, and more at each depth
  added. The `d3` and `grammar` loads did the same work (the same guesses, right and wrong, and MTP rows; the grammar
  switch changes nothing in plain text) at 7.00 and 7.80 tok/s, so one load's figure moves by about that much; every
  depth's gap to plain decoding is far larger.
- **What the server shows about the step** (a lead, not a measurement of its parts): in every pipe load no generation
  step ran as a CUDA graph (1 graph reused in the 256-token reply, against 253 for plain decoding and 108 for the
  preset). The pipe slots re-planned their graphs 221 times in that reply at depth 1, 312 at depth 2 and 397 at depth 3
  (with 4, 6 and 8 more after a change of card). An MTP-head row took 8.950 ms at depth 1 and 6.893 ms at depth 3 in
  that reply, of which 4.616 and 5.270 ms were the wait for its result (the sync figure of the server's `pipe mtp`
  line). Registry row Q1867 (C1134) asks where the step's time goes.

### No hang, but a slow schema request under the MTP guesser

| Config | `schema`: tokens | generation ms | ms a token | the client's seconds | how it ran |
|---|---:|---:|---:|---:|---|
| `preset` | 34 | 1,557.66 | 47.20 | 3.4 | the MTP draft: 20 of 39 drafted tokens accepted |
| `plain` | 37 | 2,076.69 | 57.69 | 3.9 | one token at a time |
| `d1` | 37 | 17,151.23 | 476.42 | 19.0 | out of the pipe, the batched MTP draft: 18 of 33 accepted, 3 grammar rejections of a device draw |
| `d2` | 37 | 15,637.78 | 434.38 | 17.4 | the same counts |
| `d3` | 37 | 17,604.07 | 489.00 | 19.6 | the same counts |
| `walk` | 37 | 18,342.30 | 509.51 | 20.4 | the same counts |
| `grammar` | 32 | 9,896.06 | 319.23 | 11.7 | the pipe kept on: 50 guesses, 20 right, 8 wrong, 2 grammar rejections, 120 MTP rows; 0 of 6 drafted tokens accepted |

Every schema request finished with a valid reply (pull request 3's fix holds on the P106). But under `--pipe-guess mtp`
a token under the grammar took 476.42, 434.38, 489.00 and 509.51 ms at depths 1, 2 and 3 and with the walk, against
47.20 with the preset. The guesser turns off the main model's
rollback snapshots (`n_rs_seq` 0, as pull request 7's review found), so the batched draft that runs outside the pipe
undoes a rejected draft by restoring a context checkpoint. The server's default log level counts no restores, so their
number here is not known. With `LLAMA_PIPE_GRAMMAR=1` the pipe stays on under the grammar: 319.23 ms a token, still
far above the preset. In an agent session every tool call would pay this; registry row Q1868 (C1135) keeps the idea.

## Against the row's pass rule

The rule: a depth loads, the reply is the preset's or plain decoding's (or parts only at a near tie, as the preset
itself can), and nothing hangs.
- **Loads:** depths 1, 2 and 3 all load at 65,536 tokens.
- **Replies:** the 256-token reply is plain decoding's at every depth, with the walk and with the grammar switch; so
  is the 512-token reply.
- **Hang:** none; every JSON-schema request finished.

**Passed.** For a pipe line on Flash-Next this means correctness no longer blocks it: C932's fix holds on the cards,
and the MTP head guesses well. Speed does. The agent check of row 11c (Q1676) needed this row's diagnostic to be no
slower than the preset's, and it is slower at every depth: 12.65 against 20.28 tok/s at best. Q1676 is parked with
these numbers, until a Flash-Next pipe step costs less than the preset's step (Q1867).

## Files

| File | What |
|---|---|
| [loads.tsv](loads.tsv) | per load: what was added to the preset, whether it loaded, the seconds to the first healthy reply (polled every 5 s), the server's pipe set-up lines |
| [mib.tsv](mib.tsv) | each card's MiB after the load and after the requests, and the host's available MiB |
| [replies.tsv](replies.tsv) | per request: tokens, stop reason, the client's seconds, whether it matches the preset's and plain decoding's tokens (or where it parts), the server's generation tok/s, the reply's first 60 characters |
| [parts.tsv](parts.tsv) | each point where two replies part: the two token ids and the top two tokens with their log-probabilities on each path where they were asked for |
| [guesses.tsv](guesses.tsv) | per request: the pipe's guesses (right, wrong, from the store, grammar rejections), MTP rows and their ms, the draft's accepted and drafted tokens, graphs reused and pipe re-plans |
| [run/pipe_mtp.sh](run/pipe_mtp.sh) | the driver: one load per configuration, the requests, the cards' MiB, a stop at `STOP_AT` |
| [run/client.py](run/client.py) | the requests and their prompts |
| [run/settings.env](run/settings.env) | the copy of the v5 preset that ran |
| [run/tables.py](run/tables.py) | the tables from the driver's output folder |

## Commands

On the rig, from a folder holding `pipe_mtp.sh`, `client.py` and `settings.env` (`BIN`, `MODEL`, `DRAFT` and `OUT` are
the build's `bin` folder, the first shard, the draft and an output folder):

```bash
setsid -f nohup env BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT OUT=$OUT STOP_AT=21:50 bash pipe_mtp.sh > driver.log 2>&1 < /dev/null
setsid -f nohup env BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT OUT=$OUT CONFIGS=plain_probs STOP_AT=21:50 bash pipe_mtp.sh > driver2.log 2>&1 < /dev/null
```

The second line was added after the first run, once the schema reply under the grammar switch parted from plain
decoding, to price that point. Then, with the output folder copied back:

```bash
python3 -I run/tables.py $OUT docs/data/2026-11-25_c936_flash_pipe_mtp
```

Every model reply here is generated text; the prompts are this repository's. Licence: the repository's.
