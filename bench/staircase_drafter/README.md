# Staircase guesser for pipelined speculation (Q20/D1)

On a layer split only one card works at a time. Pipelined speculation (Q87,
[pipelined speculation](../../docs/research/notes/pipelined_speculation.md)) feeds guessed next tokens into card 1 while
later cards still finish the current token, and keeps a guess only when it equals the token Flint samples, so the output
is plain decoding's. The idle cards are worth as much as the guesses are right. This folder trains the guesser that
[pipeline drafter](../../docs/research/notes/pipeline_drafter.md) recommends and measures its acceptance offline,
before any rig run: a small transformer that, for every earlier token, reads the deepest Flint hidden state that already
exists when the guess is made (a "staircase": the newest token has only passed card 1, the one before it card 2 or 3,
and so on; older tokens are complete).

The process, the methods tried, results per card count and open questions are kept in one living document:
[Building a drafter for pipelined speculation](../../docs/drafter.md).

## Pipeline

All heavy steps run on a GPU machine (here a 7-GPU quant machine; never on the rig). Each step is a plain command;
paths are placeholders.

1. **Prompts**: only sources that can be published.
   [clean_prompts.py](clean_prompts.py) builds code-first prompts that are clean by construction: "implement this
   function" stubs (signature, docstring and imports of documented functions), whole-file edits and coding-agent turns
   on the Python standard library, llama.cpp's own code (not `vendor/` or build folders) and permissively licensed
   packages (MIT, BSD, Apache-2.0, PSF). It drops every prompt that shares a 12-word span with a Q51 test prompt.
   [prompts.py](prompts.py) holds the task texts and, as data sets, OpenR1-Math's crawled competition problems; its
   `EXCLUDED_SETS` list the first runs' mix, which is not publishable and not used: ShareGPT, UltraChat, both
   Magicoder sets, unfiltered NuminaMath-CoT (NuminaMath only with its crawled `source` rows) and Hermes function
   calling. [tool_prompts.py](tool_prompts.py) (ToolACE, glaive-function-calling-v2) waits for a decision on those
   LLM-written sets. `--extend` writes more prompts, deduplicated against an existing file and numbered after it, so
   their answers and dumps join the first ones.

   ```bash
   pip install huggingface_hub pyarrow
   python clean_prompts.py prompts-clean.jsonl hf-cache --stdlib /usr/local/lib/python3.12 --llama /path/to/llama.cpp \
       --site /usr/local/lib/python3.12/site-packages --extend used.jsonl --q51-prompts q51/prompts.jsonl
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

   **vLLM as the writer** ([vllm/](vllm/)), on cards of compute 7.5 to 8.9 (RTX 20xx/30xx; not V100s, which vLLM's
   CUDA 13 images don't support): the makers' NVFP4 GGUF (RecursiveSI/flint-nvfp4-gguf, 22.9 GB) runs as W4A16 on
   Marlin, pipeline-parallel over two 24 GB cards (~12 GB of weights and ~10 GB of KV cache each). On two RTX 3090s
   with 160 requests in flight it writes ~2.4–2.7k answer tokens/s (vLLM's own log), ~7× the G3 llama-server copy
   on the same cards (~350). Text from a 4-bit writer is fine here: the Q8_0 dump labels every target.

   ```bash
   GPUS=2,5 MODEL_DIR=/models/flint-nvfp4 HF_DIR=/models/flint WORK=/work/vllm DATA=/data/q20 OUT=gen-V1 \
       PROMPTS="prompts-ext.jsonl prompts-ext2.jsonl" bash vllm/run_generation.sh
   DATA=/data/q20 OUT=gen-V1 bash vllm/stop_generation.sh
   ```

   - vLLM's Flint model (merged in vLLM PR #62938, nightly images from commit `899a1689`) asserts FlashInfer
     attention, and vLLM's FlashInfer serves attention sinks only on compute 9.0 and newer.
     [patch_flint_attention.py](vllm/patch_flint_attention.py) makes it use the standard attention layer whenever the
     backend is not FlashInfer: Flint's own Triton kernel still applies the q/k norms and RoPE, then
     `--attention-backend TRITON_ATTN` serves the sinks and the 513-token window. [serve_vllm.sh](vllm/serve_vllm.sh)
     rebuilds the patched file from the pinned image's copy at every start and mounts it read-only over it.
   - `HF_DIR` is the BF16 repository's small files (config, tokenizer, chat template); the GGUF carries neither a
     config vLLM reads nor the tokenizer.
   - `generate.py --api vllm` builds prompt ids with the Hugging Face chat template; [check_prompts.py](vllm/check_prompts.py)
     confirms they equal llama-server's (900 of 900 stored rows, 459 of them with tool lists, once string tool-call
     arguments are parsed into objects as llama-server does). Rows keep llama-server's conventions: the end token stays
     the last id of an `eos` answer.
   - [sanity_greedy.py](vllm/sanity_greedy.py) compares greedy answers with a llama-server copy before a run.
   - Startup takes a few minutes plus the time to read 22.9 GB twice (once per pipeline stage): ~14 min from a busy
     hard disk. Decode steps run as CUDA graphs; steps with a prompt chunk run eagerly (the model has no
     `torch.compile` support).

3. **Features** ([staircase_dump/](staircase_dump/), built inside a llama.cpp tree like `bench/ngram_replay/logit_dump`):
   replays each answer from an empty cache with every position as output and saves, per position, layer inputs and
   the final hidden state in FP8 E4M3 with one scale per row, plus the top-20 next-token log-probabilities. The run
   here stores the inputs of layers 6, 9, 12, 17, 18, 24, 25, 29, 30, 33, 34, 36, 39, 41, 42, 44 and 47, every card
   boundary of the six-card split (9/8/8/8/8/8) and both nine-card splits (6/6/6/6/5/5/5/5/5 and
   6/6/6/6/6/6/6/5/2), so a different staircase never needs a new dump: ~37 KB per answer token. Layers 6 and 9 (card
   1's output) are kept for every position, the rest for the answer. A second output (`DUMP_OUT2`) writes the six-card
   subset (layers 9, 17, 25, 33, 41, ~12 KB per answer token) for training from a fast disk; `feats.py pack` does the
   same afterwards. For heads that serve any split from 3 to 12 cards, the dump records a layer grid instead:
   layers 4, 6, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22, 24, 25, 27, 29, 30, 33, 34, 36, 37, 39, 41, 42, 44 and 47, with
   4, 6 and 9 kept for every position. That puts a recorded layer at or 1–2 layers below every card boundary of any
   such split (card 1 ends near layer 4 with 12 cards and near layer 16–17 with 3), ~58 KB per answer token. [check_dump.py](check_dump.py) verifies a dump: Flint's output head applied to the dumped FP8 final
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

   Several dump instances can share one pool of answers while generation still runs: [feed_split.py](feed_split.py)
   appends the next answers to whichever instance's queue runs low, so faster card sets get more. Each instance
   follows its own list with `DUMP_IN=<(tail -n +1 -F seqs-x.txt)` and appends its log, which tells the feeder what
   is done. Tool-call answers go last: ~60 % of their tokens are prompt, which the dump replays without training on it.

   The dump copies 150,144 logits per position to the host (600 KB per token): put the card holding the output head on
   the widest PCIe link, or set `DUMP_GPU_TOPK=1` (with `-b` equal to `-ub`), which computes the top-20
   log-probabilities on that card through llama.cpp's backend-sampling path and copies ~160 B per position instead.
   [compare_dumps.py](compare_dumps.py) compares two dumps of the same sequences (on one transcript the GPU top-k path
   and the default path agree to 2e-6 in log-probability). With the last card on x8 it ran at ~1,650 tokens/s (prompt and answer); with the last card on
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
   # fine-tune from a trained head: one window per sequence (nominal staircase, after-miss, random staircase),
   # every reached anchor fused, the acceptance-matched loss, noise and dropout against overfitting
   python train.py --head A ... --init-from heads/A/head.pt --lr 1e-4 --window-sample 0.3,0.2,0.5 \
       --fuse --loss lk --eta 3 --noise 0.2 --dropout 0.1 --val-seqs 300
   python check_train.py --feats q51.bin --vocab vocab.npy --flint flint/ --head heads/A/head.pt
   # one head for every split from 3 to 12 cards, on files with many recorded layers (the superset or the grid)
   python train.py --head A --feats feats/train-*.bin ... --general --init-from heads/A/head.pt --lr 1e-4
   python train.py ... --general --cards 6,9     # the same head trained on the six- and nine-card splits only
   python check_general.py --feats q51.bin --vocab vocab.npy --flint flint/ --head heads/A/head.pt
   ```

   - `--init-from` loads weights only (new parameters keep their zero initialization, so the warm-started head starts
     as the checkpoint), with a fresh optimizer and schedule; `--resume` continues the same run.
   - `--window-sample` trains each sequence on one window instead of all of them: cheaper steps, and a head that reads
     any staircase (a head trained on fixed windows scores ~0.06 on window shapes it never saw).
   - `--fuse` gives every row each shallower anchor its token has reached, as SPD does (committed rows get layer 9 next
     to the final state, the only shallower anchor stored for every position).
   - `--loss lk` is the hybrid of the LK-losses paper: λ·KL + (1 − λ)·(1 − coupled match), λ = exp(−η·match).
   - [check_train.py](check_train.py) checks the options: key slicing in the window attention changes nothing, a fused
     head warm-started from an unfused one reproduces it exactly, the LK loss has finite gradients.
   - `--general` trains the depth-general head (see "The guesser"): every sequence gets one window drawn from a random
     split of 3 to 12 cards, using the layers its file recorded. [check_general.py](check_general.py) checks the
     sampler's windows and that a general head warm-started from a six-card head reproduces it on six-card windows.
   - **Agent sessions** ([bench/agentic_data](../agentic_data/README.md)): one sequence per session, the tool results
     included, so the dump replays the session once. `--spans` (the export's `sessions_seq.jsonl`, or a map from
     `feats.py seqs` or `feed_split.py`) trains only on the tokens Flint generated. `--ctx 4096` lets the head attend
     to the last 4,096 committed positions only, the ring the engine keeps (`--pipe-draft-ctx`, default 4096); sessions
     run to 64k tokens, and full attention would train a head on context it never sees when served. `--chunk-targets
     16384` splits a session into pieces of at most that many target positions, one step each; with `--ctx` a piece
     holds only the positions its targets can reach through the head's blocks, so memory stays flat however long the
     session. [check_ctx.py](check_ctx.py) checks on synthetic data that the windowed attention equals a dense masked
     reference and that a session in pieces gives the same logits as the whole. [agentic_head.sh](agentic_head.sh)
     runs the whole path: one export of several writers' runs, a held-out set of whole modules
     ([heldout.py](../agentic_data/heldout.py)), the dump fed to several GPU sets, the vocabulary, the heads, scores
     on Q51 and on the held-out sessions, and the GGUF side files, each gated.

6. **Scoring** ([score.py](score.py)) on Q51's held-out transcripts (dumped the same way), then speed per workload
   through Q87's simulator ([sim.py](sim.py) on top of [pipeline_sim.py](../ngram_replay/pipeline_sim.py)), and the
   gate table ([report.py](report.py)):

   ```bash
   python score.py --heads heads/A --feats q51.bin --gen q51/gen.jsonl --vocab vocab.npy --flint flint/ \
       --out scores/A --alpha 3
   python sim.py --gen q51/gen.jsonl --scores scores/sim --out sim.json     # scores-<head>.npz files
   python report.py scores/A/summary.json sim.json
   ```

   `--window-group` (default 4) sets how many windows share one forward pass. Fewer windows need less memory: a
   general head's 14 windows at once do not fit a 10 GB card.

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

  The windows come from the stage times of the six-card split (section 3.1 of the note). `drafter.staircase(split)`
  derives the card-1 window of any split the same way, each card boundary mapped to the nearest lower six-card layer
  (the embedding below layer 9), and heads are also scored on the splits an owner may run:

  | Split (layers per card) | Card-1 window, newest token first |
  |---|---|
  | 5 cards, 10/10/10/10/9 | L9, L17, L25, L33 |
  | 6 cards, 9/8/8/8/8/8 | L9, L17, L33, L41 |
  | 7 cards, 7 each | emb, L9, L25, L41 |
  | 8 cards, 7/6/6/6/6/6/6/6 | emb, L9, L25, L33, L41 |
  | 9 cards, 6/6/6/6/5/5/5/5/5 | emb, L9, L17, L33, L33, L41 |
  | 9 cards, 6/6/6/6/6/6/6/5/2 | emb, L9, L17, L17, L25, L33, L41, L41 |

  With `--window-sample`, the random share of the windows covers all of them: the newest token at L9 or as its
  embedding, then up to seven older tokens in flight at non-decreasing depths.
- **Depth-general head** (`--general`). One head serves any split from 3 to 12 cards. Each row's anchor can be any
  recorded layer input (1–48) or the final state. Anchors share a projection within six depth buckets (layers 1–12,
  13–20, 21–28, 29–36 and 37–48, and the final state), which start from a six-card head's projections, and every layer
  has a learned depth embedding that starts at zero. Training draws a split per sequence, with card counts weighted
  toward what owners run: 6 and 9 cards 25 % each, 3, 4 and 5 cards 8 % each, 7 and 8 cards 6 % each, 10 and 12 cards
  5 % each, 11 cards 4 %. About 30 % of the draws have a lighter last card. From the split it takes the card-1
  staircase (60 %), the pre-step window (15 %) or the after-miss window (25 %), with stage times from the per-layer
  times measured at six and nine cards. Each depth maps to the deepest recorded layer at or below it. `--cards`
  restricts training to a few card counts and their standard splits (the even split; for nine cards also
  6/6/6/6/6/6/6/5/2), which measures what specialization buys over the general head. score.py scores
  such a head per card count on even splits: `g3` … `g12`, `g9light`, and the pre-step and after-miss windows at 6 and
  9 cards.
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
| [tool_prompts.py](tool_prompts.py) | tool-call prompts from two public tool-calling sets |
| [generate.py](generate.py) | answers with exact token ids from llama-server or vLLM (`--api vllm`), several copies sharing one prompt file |
| [vllm/](vllm/) | vLLM as the writer: attention patch for compute < 9.0, server and client scripts, prompt-id and greedy checks |
| [staircase_dump/](staircase_dump/) | `llama-staircase-dump`: layer inputs, final state (FP8 + row scale) and top-20 per position |
| [feats.py](feats.py) | dump reader, `seqs`/`info`/`pack` commands, Flint's sampler chain on top-k log-probabilities |
| [check_dump.py](check_dump.py) | dump checks (FP8 table, output head on the final state, against an earlier logit dump) |
| [feed_split.py](feed_split.py) | one pool of answers fed to several dump instances by queue length, tool calls last |
| [compare_dumps.py](compare_dumps.py) | record-by-record comparison of two dumps of the same sequences |
| [flint_weights.py](flint_weights.py) | Flint's embedding table, output head and output norm from the GGUF |
| [vocab.py](vocab.py) | the guesser's vocabulary and its coverage |
| [drafter.py](drafter.py) | the model, profiles, losses and coupled acceptance |
| [train.py](train.py) | training one head, resumable, warm-startable; agent sessions with `--spans`, `--ctx`, `--chunk-targets` |
| [check_train.py](check_train.py) | checks of the training options |
| [check_ctx.py](check_ctx.py) | checks of the agent-session path on synthetic data: the head's attention window, pieces, spans |
| [agentic_head.sh](agentic_head.sh) | the head from agent sessions in one command: export, held-out set, dump, vocabulary, training, scores, GGUF |
| [check_general.py](check_general.py) | checks of the depth-general head (window sampler, warm start from a six-card head) |
| [score.py](score.py) | per-position acceptance on held-out transcripts, intervals by transcript |
| [sim.py](sim.py) | speed per workload through Q87's simulator |
| [report.py](report.py) | tables and gate verdicts from score.py and sim.py |
