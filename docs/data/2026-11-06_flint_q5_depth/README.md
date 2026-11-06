# Flint Pascal Q5 GGUF: generation speed at fixed context depth, v2.0 against v3.0 (2026-11-06)

On the agent check, Flint Pascal Q5 GGUF ran 49.4 tok/s on the v3.0 release build against v2.0's 53.8
([release regression](../2026-11-05_release_regression/README.md)). The difference was one task, complex-numbers.
Its session generated 14,024 tokens at 38.0 tok/s as its context grew to ~30k tokens, against 2,508 tokens at 50.5 on
v2.0. The other eleven tasks ran 53.64 against 54.04. That leaves one question: is v3 slower than v2.0 at the same
context depth on this file? This test holds the content fixed and steps the depth, on the agent check's own server
line, on one rig of nine P106-090 cards with 4 GB of host RAM.

## Result

- **At the same depth, v3 generates the same tokens as v2.0, and the gap does not grow with depth.** Both builds
  generate the same 256 ids at every depth, and the same pipelined guesses come out right. v3 is 1.1 % slower at
  depth 0 and 0.1 % at 32,768. Each load's speed below is its two passes together, v3's two loads first, then v2.0's
  two:

  | Depth (tokens) | v3 | v2.0 | v3 against v2.0 | Class |
  |---:|---|---|---:|---|
  | 0 | 50.79, 50.89 | 51.59, 51.18 | −1.1 % | loss |
  | 8,192 | 45.03, 45.46 | 45.37, 45.51 | −0.4 % | negligible |
  | 16,384 | 43.77, 43.81 | 43.92, 43.89 | −0.3 % | loss |
  | 24,576 | 35.58, 35.52 | 35.59, 35.53 | −0.0 % | negligible |
  | 32,768 | 34.60, 34.55 | 34.59, 34.64 | −0.1 % | negligible |
  | all five | 41.07, 41.13 | 41.25, 41.21 | −0.3 % | loss |

  The losses are beyond the spread between loads (0.41 tok/s at depth 0, 0.04 at 16,384, 0.06 over all five) but
  small: 0.55, 0.11 and 0.14 tok/s. At depth 0 most of the loss is in each load's first timed request. Pass 1 ran
  47.15 and 47.36 on v3 against 48.40 and 47.82 on v2.0. Pass 2 ran 54.56 and 54.53 against 54.77 and 54.59.
- **Depth costs both builds the same.** On each build's first load, generation falls from 51.59 tok/s at depth 0 to
  34.59 at 32,768 on v2.0, and from 50.79 to 34.60 on v3. A session whose context grows to ~30k tokens generates at
  ~35 tok/s on either build (est., from this text). So complex-numbers' 38.0 tok/s on v3 comes from its long session,
  not from v3 being slower at depth. Sampling at temperature 1.0 gave v3 a run of 14,024 tokens and v2.0 a run of
  2,508.
- **Reading a whole prompt fresh, v3 is slightly slower at every depth**, beyond the spread. It is 1.6 % slower at
  8,192 (528.2 and 527.8 tok/s against 535.2 and 538.5), 1.3 % at 16,384 (499.8 and 499.8 against 505.9 and 506.9),
  1.0 % at 24,576 (453.4 and 453.4 against 457.2 and 458.9) and 0.7 % at 32,768 (402.5 and 401.7 against 404.3 and
  405.2). In the agent check, where each turn reads a short new chunk on top of the cache, v3 read prompts faster:
  182.3 against 175.2 tok/s on the eleven tasks.

The full tables, with each pass by value, the guesses and the checks, are in [table.md](table.md). Every request is
listed in [requests.tsv](requests.tsv).

## What was measured

- **The engine path of the agent check:** llama-server, started and stopped by
  [bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) with its RAM watchdog. It ran on the
  agent check's line for the 5-bit files, layers split by count across the nine cards:
  `-ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0
  --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui -b 512 -ub 128 --pipe-depth 4`.
- **Each request was one chat turn in Flint's template** (`<|user|>…<|assistant|>`, from the server's
  `/apply-template`). The user message is Python standard-library source followed by "Explain what the code above
  does, file by file."
  - The code is cut so that the whole prompt is exactly 8,192, 16,384, 24,576 or 32,768 tokens. Depth 0 is the
    template and the question alone, 14 tokens. Every load read the same ids (an md5 per depth, in the client's meta
    line).
  - Each request goes to `/completion` with those ids, read fresh (`cache_prompt` false). It generates 256 tokens at
    temperature 0 and top-k 1, with end of text ignored.
  - Flint answers in its reasoning ("We need answer user. Need parse request. They provided …"), the kind of text an
    agent's turn generates.
  - Each request ran twice per load (pass 1, pass 2), after one untimed warm-up of 64 tokens in and 32 out.
- **Four loads, alternated:** v2.0, v3, v2.0, v3, from 16:27:10 to 17:24:47. Each was a fresh server start that read the
  file from the rig's USB disk, up after 424, 422, 423 and 422 s. Nothing else ran on the rig. Before the first load
  it held no llama process and 0 MiB on every card, and the same after the last.
- **Speeds are the server's own timings** (`timings.predicted_per_second`, `prompt_per_second`). The guesses come from
  each request's `pipelined = … guesses, … right` line in the server log. A load's speed at a depth is its two passes'
  tokens over their time.
- **The class:** the spread is the larger of the two builds' differences between their two loads, and the delta is
  v3's mean minus v2.0's. A delta above the spread is a gain, one below minus the spread is a loss, and anything else
  is negligible.

### The first start: raw code continuation, stopped

The first start at 16:17 sent the code alone, as a continuation of the source, and its pass 2 continued from the
cache. Greedy continuation of source code repeats itself, and `--pipe-depth 4`'s guesses were right 208 of 248 times
at depth 0 and 254 of 254 on pass 2. v2.0 generated 108.28 tok/s at depth 0, 189.32 on pass 2 and 86.89 at 8,192, twice
the agent check's speed. That measured the guesser on repetitive text, not generation at depth. So it was stopped
during v2.0's first load (16:26), before any v3 request, and restarted with the chat turn. On the chat turn the
guesses are right 50 of 447 to 77 of 627 times a request, and depth 0 generates at 48.40 and 54.77 tok/s.

## Builds and file

| Name here | Build | What |
|---|---|---|
| v2.0 | fork `37e717ff6` (tag `miner-llama-v2.0`), rig build `llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4` | the build of v2.0's agent check (53.8 tok/s on this file) |
| v3 | fork `1551f5fe1` (`pascal-p106-v3-b12076-ladder5`), rig build `llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5`, tarball md5 `288b57025e6ff64e76b90674a0509228` | the v3.0 release build (49.4 tok/s) |

The file is Flint Pascal Q5 GGUF (`flint-pascal-Q5.gguf`, K2h-r16 with the release header, 29,390,812,736 bytes, md5
`54e0f28553d0c18ba940d1e8d894dfcf`, sha256 `0163ec54…df6003`).

## Files

| File | What |
|---|---|
| [table.md](table.md) | generation tok/s by depth for each load and pass, the spread and the class; the guesses; prompt reading by depth with its class; the checks |
| [requests.tsv](requests.tsv) | every request of the four loads: the server's prompt and generation timings, the guesses, wall time, the md5 of the generated ids |
| [run/depth_sweep.sh](run/depth_sweep.sh) | the driver: per load, the server on the rig, the client, the stop; the quiet checks; the logs back |
| [run/depth_client.py](run/depth_client.py) | the client, run on the rig: the prompts, the warm-up, two passes per depth |
| [run/depth_table.py](run/depth_table.py) | the tables from the client's JSON lines and the server logs |

## Commands

From the repository root on a machine that reaches the rig over ssh, with nothing else on the rig:

```bash
RIG=rig MODEL=<flint-pascal-Q5.gguf on the rig> \
  V2_BIN=/opt/llama-builds/llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4/bin/llama-server \
  V3_BIN=/opt/llama-builds/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin/llama-server \
  OUT=sweep bash docs/data/2026-11-06_flint_q5_depth/run/depth_sweep.sh
python3 -I docs/data/2026-11-06_flint_q5_depth/run/depth_table.py sweep v2a,v2b v3a,v3b v2.0 v3 tables
```

The defaults are the run's: loads `v2a:V2 v3a:V3 v2b:V2 v3b:V3`, depths `0 8192 16384 24576 32768`, 256 tokens and
port 8091. The tables were made on another machine from the copied JSON lines and server logs.

## Provenance and licence

The prompt text is Python 3.12's standard library on the rig (Ubuntu 24.04's `/usr/lib/python3.12`, the first 11
`*.py` files in sorted order, PSF licence). It is not republished: the client rebuilds it from the same folder. The
tables and scripts are under this repository's licence.
