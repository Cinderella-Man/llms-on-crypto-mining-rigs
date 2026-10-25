# The staircase guesser in the pipelined executor (Q20/D3)

A head trained here ([README](README.md)) becomes a guesser of the pipelined executor (Q87,
[pipelined speculation](../../docs/research/notes/pipelined_speculation.md)): `llama-pipespec --pipe-guess staircase` and
`llama-server --pipe-guess staircase`, in the fork branch `pascal-p106-v2-drafter`. The copy lookup goes first; the head
guesses whenever the lookup has no match of at least `--pipe-copy-min` tokens, as
[pipeline drafter](../../docs/research/notes/pipeline_drafter.md) recommends. The executor still samples every token and
compares it with the guess, so the output is plain decoding's whatever the head guesses. How the drafter is built and
where it stands: [docs/drafter.md](../../docs/drafter.md).

## Export

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py python export_gguf.py --head heads/A --vocab vocab.npy --out drafter-A.gguf \
    --type q8_0 --alpha $(cat heads/A/alpha.txt 2>/dev/null || echo 3)
```

alpha is the sharpening chosen on validation answers at scoring (`heads/<X>/alpha.txt`; 3 for every head so far).

[export_gguf.py](export_gguf.py) writes one GGUF (architecture `staircase-drafter`): the head's matrices with every RMS
norm's weight folded into the projection after it, the depth embedding, the output head and its bias, the draft
vocabulary (draft id to Flint token id), and per-depth metadata, so a new head shape needs no engine change:
- `depth_proj` and `depth_emb_row`: for each depth (0 = the embedding alone, n = the input of layer n, 49 = the final
  state after Flint's output norm), its projection and its embedding row. A six-anchor head has one projection per
  anchor (L9, L17, L25, L33, L41, final); a depth-general head (`train.py --general`) one per depth bucket and an
  embedding per depth;
- `recorded`: the depths the head was trained on (for a general head, the layers every training dump stored; or
  `--recorded`);
- `timing`: the stage model the engine derives the window from (`drafter.split_depths`);
- sizes, `fuse`, `committed` (the depth of committed rows), `alpha`.

The default stores the matrices as Q8_0 and the output head as F16 (`--f16 head`): a two-layer head with 40,000 draft
ids is 0.27 GB (0.29 GB fused). On a trained head and Flint's real hidden states, a Q8_0 output head moves the logits by
~0.16 rms (the most likely id unchanged, its probability by up to 0.07), an F16 one by ~0.035; the other matrices in
F16 change nothing measurable. All Q8_0 (`--f16 ''`) is 0.22 GB.

## What runs where

- **The window.** The engine reads Flint's layers per card from the device assignment of the run. With every card busy a
  token enters card 1 once per cycle, so the token m cycles older has finished every card that ends within m cycles
  (`split_depths` with the file's stage times); each such depth becomes the deepest recorded depth at or below it. A
  head on a card that holds layers starts right after that card's stage (`stair`: the newest token at card 1's
  output); a head on a spare card starts with the cycle (`pre`: the newest token has only its embedding). Tokens in
  flight older than the window read the last boundary before the output card. Six cards 9/8/8/8/8/8 give L9, L17, L33,
  L41; nine cards 6/6/6/6/5/5/5/5/5 give L6, L12, L24, L34, L39, L44 for a head trained on those layers.
- **Taps.** Every one-token pipe evaluation copies the inputs of the window's layers and its final hidden state (8 KB
  each, float32) into pinned host memory, each on the card that computes it, right after that card's work for the
  token, and records an event: about one row per card per token.
- **Prompt.** A prompt micro-batch also runs the last layer's feed-forward half for every position (normally only the
  output rows go through it) and copies the final hidden state of every position to the host: the head's committed
  rows need it. The output rows keep their own path, so the logits do not change.
- **The head** runs on card 1 (or `--pipe-draft-dev N`) in its own CUDA stream, so it never blocks Flint's work there. A
  guess is launched right after each submission whose next position the copy lookup cannot guess. It waits on its own
  stream for the events of the hidden states it reads, uploads each once into a store on its card, runs the committed
  rows since the last guess and the tokens in flight through its layers (attention over the newest `--pipe-draft-ctx`
  committed positions, 4096 by default), and copies back the top 16 of its distribution. A late hidden state makes the
  guess late, never wrong.
- **The draw.** With the sampler on the last card (`--pipe-backend-sample`, and always in llama-server), every token's
  random draw is keyed by its position; the executor reads the uniform the sampler will use for the guessed position
  (`llama_sampler_pos_uniform`) and the head picks the first of its candidates, in descending order of q^alpha, whose
  cumulative probability exceeds it, the rule the sampler itself applies to Flint's distribution. On Q51 this coupling
  scores 0.007-0.010 below score.py's Gumbel coupling, which would need a Gumbel-max sampler on the last card. With a
  sampler on the host, and at temperature 0, the guess is the head's most likely token.
- **Commits.** When a token's output has been sampled and the token belongs to the output, its row becomes a committed
  row (its final state) at the next guess. Wrong guesses leave nothing behind in the head.

## Options

| llama-pipespec | llama-server | Meaning |
|---|---|---|
| `--pipe-guess staircase` (job key `guess`) | `--pipe-guess staircase` | the copy lookup, else the head |
| `--pipe-draft FILE` | `--pipe-draft FILE` | the head (export_gguf.py) |
| `--pipe-draft-dev N` | `--pipe-draft-dev N` | its GPU (default: the first layer's card) |
| `--pipe-draft-ctx N` | `--pipe-draft-ctx N` | committed positions it attends to (default 4096) |
| `--pipe-draft-kind stair\|pre` | `--pipe-draft-kind ...` | when the head starts (default: stair on a card that holds layers, else pre) |
| `--pipe-draft-stair L9,L17,...` | `--pipe-draft-stair ...` | the window, newest first, instead of the split's |
| `--pipe-draft-alpha A` | `--pipe-draft-alpha A` | q^A before the draw (default: the file's) |
| `--pipe-draft-rows N`, `--pipe-draft-topk K`, `--pipe-draft-top1` | | rows per guess (default 8, or depth + 2 rounded up to 8), candidates, most likely token instead of the draw |

The drafter's load line prints the split, the window and the taps. llama-pipespec's `PIPESPEC` line adds
`copy_guesses`, `copy_hits`, `draft_guesses` and `draft_hits`; llama-server's timings print the head's guesses and right
guesses per request.

Limits: one sequence (no `--pipe-streams`, no `--pipe-race`, no fed tokens); llama-server guesses with the head for the
slot that started its prompt last; the embedding table must be on the head's card (`LLAMA_INPUT_ON_GPU`, the default on
compute 6.1, puts it on card 1) or in host memory (`LLAMA_INPUT_ON_GPU=0`, needed for the head on a spare card).

## Checks (any CUDA GPU or the CPU)

```bash
# the C++ head against drafter.Drafter.forward on random weights: bulk prompt commits, then guesses with commits and
# windows of tokens in flight; the window derived from a split; the table on the GPU and in host memory
python check_export.py case /tmp/case-A --fuse
llama-pipe-drafter-check /tmp/case-A/drafter.gguf /tmp/case-A/case.bin --split 9,8,8,8,8,8 [--host-table] [--cpu]
python check_export.py case /tmp/case-G9 --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6
llama-pipe-drafter-check /tmp/case-G9/drafter.gguf /tmp/case-G9/case.bin --split 6,6,6,6,5,5,5,5,5
# a random-weight head of the real shape: every guess wrong, every output still plain decoding's
python check_export.py head /tmp/rand-G.gguf --general --f16 head
DRAFT=/tmp/rand-G.gguf BIN=... MODEL=... ARGS='-ngl 99 -fa on -c 4096 -ub 64 --pipe-backend-greedy' ../pipespec/check_identity.sh
BIN=... MODEL=... ARGS='-ngl 99 -fa on -c 8192 -ub 64 --pipe-draft /tmp/rand-G.gguf' GUESSES=staircase FLOORS= ../pipespec/check_server.sh
```

## On the rig

1. **Cost of a guess on one P106** (nothing else on the card): `llama-pipe-drafter-check drafter.gguf --bench 300
   --split 9,8,8,8,8,8 --ctx 2048` prints the median time from the launch to the guess on the host for the split's
   window (`--window W` for another count of tokens in flight).
2. **Pipelined decoding**: [rig_run.sh](../pipespec/rig_run.sh) with `DRAFT=drafter.gguf ONLY_STAIR=1` adds staircase
   jobs next to plain and `cont` on chat, code, edit and rewrite; `BG=2 SAMPLING='--temp 1 --top-k 20 --top-p 0.95
   --min-p 0 --seed 42'` samples on the last card with position-keyed draws, so the head's guess is drawn with the
   sampler's own uniform. The window follows the run's split (`TS=9,8,8,8,8,8` puts six cards' boundaries at the
   six-card anchors).
3. **A spare card**: Flint on six cards (`GPUS=0,...,6 TS=9,8,8,8,8,8,0`), the head on card 7 (`DRAFT_ARGS='--pipe-draft-dev
   6'`, the `pre` window), `LLAMA_INPUT_ON_GPU=0`.
4. **Served**: [check_server.sh](../pipespec/check_server.sh) with `GUESSES='cont staircase' KEEP_SPEED=1` and
   `--pipe-draft` in `ARGS`: identity and tokens per second per request.
