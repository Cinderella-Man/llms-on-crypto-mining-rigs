# Staircase guesser for pipelined speculation (Q20/D1)

On a layer split only one card works at a time. Pipelined speculation (Q87,
[pipelined speculation](../../docs/research/notes/pipelined_speculation.md)) feeds guessed next tokens into card 1 while
later cards still finish the current token, and keeps a guess only when it equals the token Flint samples, so the output
is plain decoding's. The idle cards are worth as much as the guesses are right. This folder trains the guesser that
[pipeline drafter](../../docs/research/notes/pipeline_drafter.md) recommends and measures its acceptance offline,
before any rig run: a small transformer that, for every earlier token, reads the deepest Flint hidden state that already
exists when the guess is made (a "staircase": the newest token has only passed card 1, the one before it card 2 or 3,
and so on; older tokens are complete).

Results are in [pipeline drafter](../../docs/research/notes/pipeline_drafter.md), section "D1 results".

## Pipeline

All heavy steps run on a GPU machine (here a 7-GPU quant machine; never on the rig). Each step is a plain command;
paths are placeholders.

1. **Prompts** ([prompts.py](prompts.py)): 7,150 first turns in the mix of the
   [EAGLE-3 plan](../eagle3_flint/TRAINING_PLAN.md) (ShareGPT, UltraChat, Magicoder Evol and OSS, Hermes function
   calling, NuminaMath, OpenR1-Math), plus 1,100 whole-file edits and 1,100 coding-agent turns built from source files
   of a Python standard library and a llama.cpp tree. None of Q51's 38 test prompts or their source files are used.
   `--extend` writes more prompts (other rows, other files and tasks), deduplicated against an existing file and
   numbered after it, so their answers and dumps join the first ones.

   ```bash
   pip install huggingface_hub pyarrow
   python prompts.py prompts.jsonl hf-cache /usr/lib/python3.12 /path/to/llama.cpp
   python prompts.py prompts-ext.jsonl hf-cache /usr/lib/python3.12 /path/to/llama.cpp --scale 3 --seed 21 \
       --extend prompts.jsonl
   ```

2. **Answers** ([generate.py](generate.py)): Flint in llama-server (`--jinja`), the makers' sampling (temperature 1.0,
   top-k 20, top-p 0.95, min-p 0), reasoning kept, 2,048 tokens at most, exact token ids, seed 20000 + the prompt's
   number. Several server copies share one prompt file (each prompt is claimed by an exclusive file create), so faster
   copies simply take more prompts.

   ```bash
   llama-server -m flint-Q8_0.gguf -ngl 99 -fa on --jinja -np 56 -c 458752 -no-kvu -bs -b 2048 -ub 512 \
       --cache-ram 0 --port 8001
   python generate.py http://127.0.0.1:8001 prompts.jsonl gen-1.jsonl claims --conc 56
   ```

   - Give every slot its own KV cells (`-no-kvu`, 8,192 per slot here, enough for the longest prompt plus the answer).
     With the unified KV cache (`-kvu`, llama-server's default) every decode token attends over all slots' cells, and
     64 slots ran several times slower than 24.
   - `-bs` samples on the GPU, so 150,144 logits per slot don't cross the PCIe link every step.
   - Measured: Flint Q8_0 on 2 × RTX 3080 + 2 × RTX 3090 with 56 slots ~470 response tokens/s; on two RTX 3090s with
     24 slots ~370; on two V100s with 48 slots ~150–200 (Volta is slow at batched Q8_0 MoE products). G3 (the
     29.6 GB release-candidate mix) on RTX 3080 + 3080 + 3090 with 28 slots ~420–475.

3. **Features** ([staircase_dump/](staircase_dump/), built inside a llama.cpp tree like `bench/ngram_replay/logit_dump`):
   replays each answer from an empty cache with every position as output and saves, per position, layer inputs and
   the final hidden state in FP8 E4M3 with one scale per row, plus the top-20 next-token log-probabilities. The run
   here stores the inputs of layers 6, 9, 12, 17, 18, 24, 25, 29, 30, 33, 34, 36, 39, 41, 42, 44 and 47, every card
   boundary of the six-card split (9/8/8/8/8/8) and both nine-card splits (6/6/6/6/5/5/5/5/5 and
   6/6/6/6/6/6/6/5/2), so a different staircase never needs a new dump: ~37 KB per answer token. Layers 6 and 9 (card
   1's output) are kept for every position, the rest for the answer. A second output (`DUMP_OUT2`) writes the six-card
   subset (layers 9, 17, 25, 33, 41, ~12 KB per answer token) for training from a fast disk; `feats.py pack` does the
   same afterwards. [check_dump.py](check_dump.py) verifies a dump: Flint's output head applied to the dumped FP8 final
   state must give the dumped top-1 token (98–99 % of positions; the rest are near-ties moved by llama.cpp's 8-bit
   activation quantization in batched products) and log-probabilities within a few thousandths.

   ```bash
   python feats.py seqs seqs.txt map.jsonl gen-*.jsonl
   DUMP_IN=seqs.txt DUMP_OUT=big/train-1.bin DUMP_LAYERS=6,9,12,17,18,24,25,29,30,33,34,36,39,41,42,44,47 \
   DUMP_ALLPOS=6,9 DUMP_OUT2=fast/train-1.bin DUMP_LAYERS2=9,17,25,33,41 \
       llama-staircase-dump -m flint-Q8_0.gguf -ngl 99 -fa on -c 8192 -b 2048 -ub 512
   python flint_weights.py flint-Q8_0.gguf flint/        # embedding table, output head, output norm (needs gguf-py)
   python check_dump.py fast/train-1.bin flint/
   ```

   The dump copies 150,144 logits per position to the host (600 KB per token): put the card holding the output head on
   the widest PCIe link. With the last card on x8 it ran at ~1,650 tokens/s (prompt and answer); with the last card on
   x4, ~850; with V100s on x1 links, ~280.

4. **Vocabulary** ([vocab.py](vocab.py)): the most frequent ids in the answers, Flint's added and control tokens (ids
   150000+) forced in. 40,000 ids here: at 32,000, Q51's code coverage was 94 %.

   ```bash
   python vocab.py vocab.npy gen-*.jsonl --size 40000 --test q51/gen.jsonl
   ```

5. **Training** ([train.py](train.py), model in [drafter.py](drafter.py)), one head per GPU, PyTorch. Checkpoints with
   the optimizer state every 30 minutes; `--resume` continues a run, also on more `--feats` files.

   ```bash
   python train.py --head A --feats fast/train-*.bin --vocab vocab.npy --flint flint/ --out heads/A --minutes 300
   python train.py --head A --layers 4 ...       # a larger head
   python train.py --head T ... --fp16           # on a V100 (no bfloat16)
   ```

6. **Scoring** ([score.py](score.py)) on Q51's held-out transcripts (dumped the same way), then speed per workload
   through Q87's simulator ([sim.py](sim.py) on top of [pipeline_sim.py](../ngram_replay/pipeline_sim.py)), and the
   gate table ([report.py](report.py)):

   ```bash
   python score.py --heads heads/A --feats q51.bin --gen q51/gen.jsonl --vocab vocab.npy --flint flint/ \
       --out scores/A --alpha 3
   python sim.py --gen q51/gen.jsonl --scores scores/sim --out sim.json     # scores-<head>.npz files
   python report.py scores/A/summary.json sim.json
   ```

## The guesser

- **Rows.** Every position is one row: Flint's frozen input embedding of its token, plus one projected Flint hidden
  state (the deepest one available: an "anchor"), plus a learned embedding of which anchor it is. Each anchor has its
  own RMS norm and 2,048 × 2,048 projection.
- **Body.** Two transformer layers at Flint's width (16 query and 4 KV heads of 128, SwiGLU 4,096, RoPE), Flint's
  output norm, and a head over the 40,000-id vocabulary plus one "other" class: 182.7 M trainable parameters (the
  embedding table is Flint's and stays frozen).
- **Initialization.** With the blocks' output projections at zero, a row's prediction starts as Flint's output head on
  its anchor: a layer input with an identity projection gives that layer's logit lens (layer 41: 0.41 coupled on Q51
  before training), and the final hidden state, dumped after Flint's output norm, gets the inverse of that norm's
  weight so that a row holding it reproduces Flint (0.87 coupled; the rest is vocabulary, FP8 and a small embedding
  term). The output head starts from Flint's output-head rows.
- **Profiles.** Committed positions (every token older than the guesses in flight) use one anchor; the newest W
  positions (a window) use a staircase. Training computes every committed row once with causal attention and, for
  every target, its window rows, which attend to the committed rows older than the window and causally to each other.
  The newest window row predicts the next token.

| Head | Committed | Windows trained (newest first) | Stands for |
|---|---|---|---|
| A | final state | staircase L9, L17, L33, L41; after a miss L9 | six cards, head on card 1 after card 1's stage |
| B | final state | pre-step emb, L9, L25, L33, L41; late pre-step emb, L17, L33, L41; after a miss emb | a spare card, before or late in the cycle |
| C | L9 | L9 | card 1's output only, no cross-card features |
| U | final state | L9 | every older token complete (A's inputs after a miss) |
| T | embedding | embedding | tokens only |

  The windows come from the stage times of the six-card split (section 3.1 of the note). Heads A and B are also scored
  on nine-card staircases, each card boundary mapped to the nearest lower six-card layer: emb, L9, L17, L33, L41 for
  6/6/6/6/5/5/5/5/5 layers and emb, L9, L17, L25, L33, L41, L41 for 6/6/6/6/6/6/6/5/2.
- **Loss.** Forward KL divergence from Flint's sampler distribution (its top-20 log-probabilities through top-k 20,
  top-p 0.95, min-p 0 at temperature 1; mass outside the vocabulary goes to the "other" class) to the guesser, plus 0.2
  × cross-entropy on the sampled token. AdamW (0.9, 0.95), learning rate 2e-4 with warm-up and a cosine decay over a
  fixed time budget, gradient clipping 1.0, ~8k targets per step, BF16 (FP16 with loss scaling on V100s). An
  EAGLE-style regression of the newest row onto Flint's final state of that token (`--reg`) gave no gain.
- **Acceptance.** Per position, the Gumbel-coupled match probability Σ_i 1 / Σ_j max(p_j/p_i, q_j/q_i) between Flint's
  kept distribution p and the guesser's q (mass on tokens Flint drops counts against the guess), and the probability
  that the guesser's top token is the sample. The guesser's distribution is sharpened to q^α before the coupled draw,
  with α chosen on validation answers (3 % of the training answers held out); the top token is unchanged. Copied text
  is text inside a ≥ 8-token copy from the prompt or the earlier output.

## Files

| File | What it does |
|---|---|
| [prompts.py](prompts.py) | training prompt set, and extensions of it |
| [generate.py](generate.py) | answers with exact token ids from llama-server, several copies sharing one prompt file |
| [staircase_dump/](staircase_dump/) | `llama-staircase-dump`: layer inputs, final state (FP8 + row scale) and top-20 per position |
| [feats.py](feats.py) | dump reader, `seqs`/`info`/`pack` commands, Flint's sampler chain on top-k log-probabilities |
| [check_dump.py](check_dump.py) | dump checks (FP8 table, output head on the final state, against an earlier logit dump) |
| [flint_weights.py](flint_weights.py) | Flint's embedding table, output head and output norm from the GGUF |
| [vocab.py](vocab.py) | the guesser's vocabulary and its coverage |
| [drafter.py](drafter.py) | the model, profiles, losses and coupled acceptance |
| [train.py](train.py) | training one head, resumable |
| [score.py](score.py) | per-position acceptance on held-out transcripts, intervals by transcript |
| [sim.py](sim.py) | speed per workload through Q87's simulator |
| [report.py](report.py) | tables and gate verdicts from score.py and sim.py |
