# Pipelined speculation: llama-pipespec, checks and rig scripts (Q87)

`llama-pipespec` (in the fork, branch `pascal-p106-v2-pipespec`, `examples/pipespec`) decodes one sequence either plainly
(one `llama_decode` per token) or pipelined: up to D guessed tokens run on the first cards while the later cards still
work on the current token, and each guess is checked by sampling the real token once and comparing (sample-and-match).
Design, prior work and results: [pipelined_speculation.md](../../docs/research/notes/pipelined_speculation.md).

| File | What it does |
|---|---|
| [check_identity.sh](check_identity.sh) | Plain run plus ~25 pipelined runs (oracle with 0/10/50 % wrong guesses, copy lookup, copy ≥ 4 else continuation counts, ngram-simple, ngram-map-k at depths 1–3, depth 0; or a race with `RACE=K`) in one process; compares tokens, per-token logits hashes, the KV state after generation and probe logits with the plain run |
| [rig_run.sh](rig_run.sh) | The timing matrix for the rig: on four prompts plain, depth 0, the Q96 guesser at depths 1–5 and three other guessers at depth 3; oracle ceilings and the cost of wrong guesses; writes `summary.tsv` (speed-ups, guesses, identity per job), the plain outputs for the simulator, traces and a GPU clock/power log |
| [check_streams.sh](check_streams.sh) | The streams mode against separate plain runs: every conversation of N (all at once, staggered arrivals, guesses in the slack, a budget of one item) must match a plain run of its prompt, seed and chunk size in its own sequence |
| [rig_streams.sh](rig_streams.sh) | Several conversations on the rig (Q104): N = 1–8 staggered through the cards, total and per-conversation tokens/s, identity per conversation, GPU and host CPU logs |
| [check_hostloop.sh](check_hostloop.sh) | The host-side changes of Q120/Q121 against plain decoding: `check_identity.sh` in the three input upload modes, with the embedding lookup on card 1, and with fed tokens (greedy and seeded temperature sampling on the device); `check_streams.sh` at N = 1/4/8 with fed conversations and at N = 4 with host sampling; a fed run that must refuse when the CPU reads the token ids |
| [rig_hostloop.sh](rig_hostloop.sh) | The host's share of a token on the rig (Q120/Q121): synchronous against packed asynchronous uploads, and host-fed against device-fed tokens, single stream and N = 1/4/8 conversations; per job tokens/s and the host time per token inside the submit calls, inside the waits and the rest, plus identity with plain runs |
| [rig_race.sh](rig_race.sh) | Races of K samples of one request on the rig (Q93): time until the first sample closes its reasoning, against one sample alone, with the plain round-robin race as the identity check |
| [drop_vs_cancel.py](drop_vs_cancel.py) | Event model of the cards: speed-up with every guess right, 70 % right and every guess wrong, when wrong guesses run to the end (this executor) or are skipped per card (ideal cancellation) |
| [make_table.py](make_table.py) | A continuation table for the `cont` guesser (`--pipe-table`) from transcripts in bench/ngram_replay's format, leaving out chosen ones |
| [traces.py](traces.py) | Per job: tokens/s and the median time from one sampled token to the next, split by whether a right guess (one pipeline cycle), a wrong one (a full pass plus the dropped work) or none (a full pass) was in flight behind it |
| [prompts/](prompts/) | `chat.txt` and `code.txt` (Q51's first chat and code prompts: little to copy), `edit.txt` (Q51's first edit, type hints for a 90-line script: partial copy), `rewrite.txt` (rename two variables in a 53-line script: almost all copy), `short.txt` (a one-line question with short reasoning, for races) |

## llama-pipespec in short

```
llama-pipespec -m <gguf> -ngl 99 -fa on -c 8192 [--temp 0 | --temp 1 --seed 42 ...] --pipe-chat --pipe-prompt prompts/rewrite.txt \
    -n 512 --pipe-mode pipe --pipe-depth 3 --pipe-guess copy --pipe-copy-min 2 [--pipe-out tokens.txt] [--pipe-trace t.tsv]
```

- `--pipe-mode plain|pipe`, `--pipe-depth D` (guesses in flight), `--pipe-guess none|copy|cont|ngram-simple|spec|oracle`,
  `--pipe-copy-min` for `copy` and `cont`, `--pipe-table <file>` for `cont`, `--pipe-spec <llama.cpp drafter>` for
  `spec`, `--pipe-oracle <tokens file>` and `--pipe-miss <share>` for `oracle`.
- `cont` (Q96): the copy lookup (longest earlier match of the suffix, up to 12 tokens, newest on ties) if its match is at
  least `--pipe-copy-min` long, else the most frequent continuation of the longest 1–8-token context in the request so
  far (prompt and committed output), plus the counts of `--pipe-table` if given. The same lookups as
  [bench/ngram_replay/pipeline_sim.py](../ngram_replay/pipeline_sim.py), which therefore predicts the same guesses.
- `--pipe-jobs FILE`: one job per line as `key=value` pairs (`name mode guess depth copy_min miss oracle prompt n spec
  probe chat out state_out trace race race_stop jsonl cat`), all with one model load; `oracle=@<name>` replays an
  earlier job's output. `jsonl=FILE cat=CLASS` appends the job's prompt and output in the simulator's transcript format.
- Every job prints a `PIPESPEC name=… tok_s=… hits=… misses=… wasted=… tokens=<hash> logits=<hash> state=<hash>
  probe=<hash>` line. `hits`/`misses` count guesses that were checked; `wasted` counts dropped tokens.
- `--pipe-trace` writes one line per sampled token: microseconds since the start, guesses in flight, and `h` (the guess
  behind it was right), `m` (wrong), `n` (no guess), `p` (plain), `r` (a raced sample's token), `f` (the next token was
  already in flight, fed on the devices).
- `--pipe-race K` (job key `race`): K samples of the request (seeds seed … seed + K − 1, sharing the prompt's KV cells)
  until one emits `--pipe-race-stop` (default `</think>`), then the winner continues alone with the job's guesser
  (`K = 1`: one sample, timed to the stop text the same way). Plain mode decodes the samples round robin; pipe mode
  keeps one token of each in flight. `check_identity.sh` checks it with `RACE=K`. Host sampling only.
- `--pipe-backend-greedy`: greedy sampling on the last card in both modes (llama.cpp's backend sampling), so only a token
  id comes back instead of the logits (Flint: 150,144 floats, 0.6 MB per token, ~3 ms on a PCIe 1.0 x1 link, queued on
  the last card behind every token's compute). `rig_run.sh` uses it unless `BG=0`. `--pipe-backend-sample`: the
  temperature chain (`--temp`, `--top-k`, `--top-p`, `--min-p`) on the last card, one per sequence, each token's draw keyed
  by (seed, position).
- `--pipe-streams N` (job key `streams`): N conversations staggered through the cards, one KV stream each (`-np N`, `-c` =
  N × one conversation's context), prompts from `--pipe-prompts a.txt,b.txt,...` (cycled), seeds seed + c, prompts
  submitted as `--pipe-chunk` token chunks between the other conversations' tokens, one token of every generating
  conversation in flight, guesses (`--pipe-depth`, `--pipe-guess copy|cont`) only while fewer than `--pipe-budget` items
  are in flight; conversation c joins after `--pipe-arrive` × c generated tokens. One `PIPESPEC` line per conversation
  (`<name>-c<c>`) and a `PIPESPEC_STREAMS` line with the total rate. Job keys `chunk`, `seed` and `seq` make the matching
  plain run.
- `--pipe-feed` (job key `feed`, branch `pascal-p106-v2-pipespec2`): pipe mode at depth 0 and streams mode without
  guesses, with a sampler on the device and `LLAMA_INPUT_ON_GPU=1` (the embedding lookup on card 1). A sequence's next
  token is submitted before its current token is sampled, its id copied from the last card to card 1
  (`llama_pipe_submit_fed`), so the host stays one token ahead. Trace outcome `f`.
- Pipe and streams jobs also print the host time per generated token: `host_submit_us` (inside the submit calls),
  `host_wait_us` (inside the waits for results) and `host_rest_us` (the rest). `LLAMA_PIPE_UPLOAD=0|1|2` sets how the slots
  upload their host inputs: 0 one synchronous copy each (as `llama_decode`), 1 one asynchronous copy each, 2 (default on
  `pascal-p106-v2-pipespec2`) each card's inputs in one asynchronous copy. Design and checks:
  [host_decode_loop.md §6](../../docs/research/notes/host_decode_loop.md#6-built-in-the-executor-packed-uploads-and-fed-tokens-q120-q121).

## Rig plan

The commands for the rig, what to log and how to compare with the simulator are in the note:
[Measuring it on the rig](../../docs/research/notes/pipelined_speculation.md#measuring-it-on-the-rig).
