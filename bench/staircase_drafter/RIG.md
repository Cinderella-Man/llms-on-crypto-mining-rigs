# The staircase guesser in the pipelined executor (Q20/D3)

A head trained here ([README](README.md)) becomes a guesser of the pipelined executor (Q87,
[pipelined speculation](../../docs/research/notes/pipelined_speculation.md)): `llama-pipespec --pipe-guess staircase` and
`llama-server --pipe-guess staircase`, in the fork branch `pascal-p106-v2-drafter`. The copy lookup goes first; the head
guesses whenever the lookup has no match of at least `--pipe-copy-min` tokens, as
[pipeline drafter](../../docs/research/notes/pipeline_drafter.md) recommends. The executor still samples every token and
compares it with the guess, so the output is plain decoding's whatever the head guesses.

## Export

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py python export_gguf.py --head heads/A --vocab vocab.npy --out drafter-A.gguf \
    --type q8_0 --alpha $(cat heads/A/alpha.txt 2>/dev/null || echo 3)
```

alpha is the sharpening chosen on validation answers at scoring (`heads/<X>/alpha.txt`; 3 for every head so far).

[export_gguf.py](export_gguf.py) writes one GGUF (architecture `staircase-drafter`): the head's matrices with every RMS
norm's weight folded into the projection after it, the anchor embeddings, the output head and its bias, the draft
vocabulary (draft id to Flint token id), and the metadata the executor needs: sizes, the anchor layers (inputs of layers
9, 17, 25, 33, 41, then the final state after Flint's output norm), whether the head is fused, the anchor of committed
rows, the staircase of the tokens in flight (from the head's profile, e.g. `L9,L17,L33,L41` for head A) and alpha (the
guess is drawn from q^alpha). A two-layer head with 40,000 draft ids is 0.20 GB at Q8_0.

## What runs where

- **Taps.** Every one-token pipe evaluation copies the inputs of the head's anchor layers and its final hidden state
  (8 KB each, float32) into pinned host memory, each on the card that computes it, right after that card's work for the
  token, and records an event. On six cards split 9/8/8/8/8/8 each card copies one row per token.
- **Prompt.** A prompt micro-batch also runs the last layer's feed-forward half for every position (normally only the
  output rows go through it) and copies the final hidden state of every position to the host: the head's committed
  rows need it. The output rows keep their own path, so the logits do not change.
- **The head** runs on card 1 (or `--pipe-draft-dev N`) in its own CUDA stream, so it never blocks Flint's work there. A
  guess is launched right after each submission whose next position the copy lookup cannot guess. It waits on its own
  stream for the events of the anchors it reads (the newest token's card-1 output, the older tokens' deeper anchors
  along the staircase), uploads each anchor once into a store on its card, runs the committed rows since the last guess
  and the tokens in flight through its two layers (attention over the newest `--pipe-draft-ctx` committed positions, 4096
  by default), and copies back the top 16 of its distribution.
- **The draw.** With the sampler on the last card (`--pipe-backend-sample`, and always in llama-server), every token's
  random draw is keyed by its position; the executor reads the uniform the sampler will use for the guessed position
  (`llama_sampler_pos_uniform`) and the head picks the first of its candidates, in descending order of q^alpha, whose
  cumulative probability exceeds it, the rule the sampler itself applies to Flint's distribution. This is the
  coupling the served sampler allows; score.py's Gumbel coupling (argmax of alpha·logit plus Gumbel noise keyed by
  seed, position and token) would need a Gumbel-max sampler on the last card. With a sampler on the host, and at
  temperature 0, the guess is the head's most likely token.
- **Commits.** When a token's output has been sampled and the token belongs to the output, its row becomes a committed
  row (its final state) at the next guess. Wrong guesses leave nothing behind in the head.

## Options

| llama-pipespec | llama-server | Meaning |
|---|---|---|
| `--pipe-guess staircase` (job key `guess`) | `--pipe-guess staircase` | the copy lookup, else the head |
| `--pipe-draft FILE` | `--pipe-draft FILE` | the head (export_gguf.py) |
| `--pipe-draft-dev N` | `--pipe-draft-dev N` | its GPU (default: the first layer's card) |
| `--pipe-draft-ctx N` | `--pipe-draft-ctx N` | committed positions it attends to (default 4096) |
| `--pipe-draft-stair L9,L17,...` | `--pipe-draft-stair ...` | anchors of the tokens in flight, newest first (default: the file's); older tokens in flight read the deepest anchor layer |
| `--pipe-draft-alpha A` | `--pipe-draft-alpha A` | q^A before the draw (default: the file's) |
| `--pipe-draft-rows N`, `--pipe-draft-topk K`, `--pipe-draft-top1` | | rows per guess (default 8, or depth + 2 rounded up to 8), candidates, most likely token instead of the draw |

llama-pipespec's `PIPESPEC` line adds `copy_guesses`, `copy_hits`, `draft_guesses` and `draft_hits`; llama-server's timings
print the head's guesses and right guesses per request.

Limits: one sequence (no `--pipe-streams`, no `--pipe-race`, no fed tokens); llama-server guesses with the head for the
slot that started its prompt last; the embedding table must be on the head's card (`LLAMA_INPUT_ON_GPU`, the default on
compute 6.1, puts it on card 1) or in host memory (`LLAMA_INPUT_ON_GPU=0`, needed for the head on a spare card). The
head reads the anchors of its staircase and waits for them on its own stream; a pipeline slower than the staircase
assumes makes the guess late, never wrong.

## Checks (any CUDA GPU or the CPU)

```bash
# the C++ head against drafter.Drafter.forward on random weights: bulk prompt commits, then guesses with commits and
# windows of 1-6 tokens in flight; the table on the GPU and in host memory
python check_export.py case /tmp/case-A --fuse
llama-pipe-drafter-check /tmp/case-A/drafter.gguf /tmp/case-A/case.bin [--host-table] [--cpu]
# a random-weight head of the real shape: every guess wrong, every output still plain decoding's
python check_export.py head /tmp/rand-A.gguf
DRAFT=/tmp/rand-A.gguf BIN=... MODEL=... ARGS='-ngl 99 -fa on -c 4096 -ub 64 --pipe-backend-greedy' ../pipespec/check_identity.sh
BIN=... MODEL=... ARGS='-ngl 99 -fa on -c 8192 -ub 64 --pipe-draft /tmp/rand-A.gguf' GUESSES=staircase FLOORS= ../pipespec/check_server.sh
```

## On the rig

1. **Cost of a guess on one P106** (nothing else on the card): `llama-pipe-drafter-check drafter.gguf --bench 300
   --window W --ctx 2048` prints the median time from the launch to the guess on the host for W tokens in flight.
2. **Pipelined decoding**: [rig_run.sh](../pipespec/rig_run.sh) with `DRAFT=drafter.gguf ONLY_STAIR=1` adds staircase
   jobs next to plain and `cont` on chat, code, edit and rewrite; `BG=2 SAMPLING='--temp 1 --top-k 20 --top-p 0.95
   --min-p 0 --seed 42'` samples on the last card with position-keyed draws, so the head's guess is drawn with the
   sampler's own uniform. Six cards: `TS=9,8,8,8,8,8` (the head's anchors are the card boundaries). Nine cards
   (6/6/6/6/5/5/5/5/5): `DRAFT_ARGS='--pipe-draft-stair emb,L9,L17,L33,L41'`, drafter.py's `NINE` staircase, since card
   1 holds layers 0-5 and the newest token has no anchor yet when the head runs after card 1's stage.
3. **A spare card**: Flint on six cards (`GPUS=0,...,6 TS=9,8,8,8,8,8,0`), the head on card 7 (`DRAFT_ARGS='--pipe-draft-dev 6
   --pipe-draft-stair emb,L9,L25,L33,L41'` for the pre-step staircase), `LLAMA_INPUT_ON_GPU=0`.
4. **Served**: [check_server.sh](../pipespec/check_server.sh) with `GUESSES='cont staircase' KEEP_SPEED=1` and
   `--pipe-draft` in `ARGS`: identity and tokens per second per request.
