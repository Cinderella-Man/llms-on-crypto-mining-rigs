# Training the staircase drafter from scratch

The drafter is a small transformer (two layers, 40,000 draft ids) that guesses Flint's next token from the hidden
states Flint has already computed for the tokens in flight on a layer split. This page rebuilds the release head end to
end: prompts, Flint's answers, the hidden-state dump, training, scoring, export, and the checks on any CUDA GPU. Why the
design is what it is, every run so far and the lessons: [docs/drafter.md](../../../docs/drafter.md). The tools and their
options: [bench/staircase_drafter](../../../bench/staircase_drafter/README.md).

`drafter_pipeline.sh` runs the steps below on one machine, one after another. Here they ran on a 7-GPU machine
(3 × RTX 3090, 2 × V100 32 GB, 2 × RTX 3080) with several copies in parallel; the commands are the same.

**The release heads:** R2-9 for nine cards and R2-gen for six cards and any other count from 3 to 12 (step 5), the
winner and the runner-up of the 2026-10-25 morning round. Both are depth-general, unfused, two layers, 40,000 draft ids.
Everything up to step 5 is shared by both. RIG-TBD: whether the 15:00 round's continuation of R2-9 replaces it.

## What you need

| | |
|---|---|
| Flint for the answers | any faithful Flint GGUF in llama-server (we used Unsloth's Q8_0, our G3 mix and Unsloth's UD-Q4_K_XL), or the makers' NVFP4 GGUF in vLLM on RTX 20xx/30xx cards |
| Flint for the dump | Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF) (42.3 GB): the dump's hidden states and targets come from it |
| llama.cpp | a tree with `bench/staircase_drafter/staircase_dump/` built as an example (CUDA), and `bench/ngram_replay/datastore.cpp` for the store |
| The engine | the release engine (`reproduce/engine.md`) for export checks and serving |
| Python | PyTorch with CUDA, numpy, huggingface_hub, pyarrow, gguf-py |
| GPUs | 12–32 GB cards for training (one head per GPU; general heads score with `--window-group 2` on 12 GB), any CUDA cards that hold Flint Q8_0 for the dump |
| Disk | ~58 GB per million answer tokens for the dump (the 26-layer grid); the release round's 27.6 M answer tokens took ~1.3 TB, on hard disks |

## 1. Prompts

All in `bench/staircase_drafter/`; `$HF` is a Hugging Face cache, `$STD` a Python standard library folder (source
files for edit and agent prompts), `$LLAMA` a llama.cpp tree (more source files).

| File | Prompts | Command |
|---|---|---|
| prompts.jsonl | 7,150 | `python prompts.py prompts.jsonl $HF $STD $LLAMA --seed 20 --scale 1` |
| prompts-ext.jsonl | 19,809 | `python prompts.py prompts-ext.jsonl $HF $STD $LLAMA --scale 3 --seed 21 --extend prompts.jsonl` |
| prompts-ext2.jsonl | 34,142 | `cat prompts.jsonl prompts-ext.jsonl > prompts-all12.jsonl; python prompts.py prompts-ext2.jsonl $HF $STD $LLAMA --scale 6 --seed 22 --extend prompts-all12.jsonl` |
| prompts-tools.jsonl | 15,330 | `cat prompts.jsonl prompts-ext.jsonl prompts-ext2.jsonl > used.jsonl; python tool_prompts.py prompts-tools.jsonl $HF --extend used.jsonl` |

**Sources and licences:**
- `prompts.py`: ShareGPT, UltraChat, Magicoder Evol and OSS, Hermes function calling, NuminaMath and OpenR1-Math (the
  datasets' own licences apply to the prompts; only Flint's answers to them are trained on), plus whole-file edits and
  coding-agent turns built from the Python standard library and the llama.cpp sources. The published dataset lists
  each source with its licence and leaves out the ShareGPT rows (Aeala/ShareGPT_Vicuna_unfiltered, unclear terms);
  the release heads were trained with them.
- `tool_prompts.py`: Team-ACE/ToolACE and glaiveai/glaive-function-calling-v2 (Apache-2.0), with their tool lists passed
  as `tools` and their schema types mapped to JSON Schema (without that, 44 % fail to render).
- The 38 test prompts and their source files ([bench/expert_overlap](../../../bench/expert_overlap/README.md), "Q51")
  are excluded. Prompt numbers are global: `--extend` numbers after the file it extends.

## 2. Answers

The makers' sampling for every writer: temperature 1.0, top-k 20, top-p 0.95, min-p 0, reasoning kept, at most 2,048
answer tokens, seed 20000 + the prompt's number, exact token ids kept. Copies share a prompt file through a `claims/`
folder (one exclusive file per prompt).

**llama.cpp:**

```bash
llama-server -m flint-Q8_0.gguf -ngl 99 -fa on --jinja -np 96 -c 589824 -no-kvu -bs -b 2048 -ub 512 --cache-ram 0 --port 8001
python generate.py http://127.0.0.1:8001 prompts-ext.jsonl gen-x.jsonl claims --conc 96 --n-predict 2048 --max-prompt 4096
```

`-c` is slots × 6,144; `-no-kvu` gives every slot its own cells (the unified cache made 64 slots several times slower
than 24).

**vLLM** (RTX 20xx/30xx; ~7× llama.cpp on the same cards): the makers' NVFP4 GGUF
(RecursiveSI/flint-nvfp4-gguf, 22.9 GB), pipeline-parallel over two 24 GB cards, image
`vllm/vllm-openai:nightly-c50aca14416ef29f6ed0ed256302f30646726dcd`, the Flint attention patch
`vllm/patch_flint_attention.py`, and `--pipeline-parallel-size 2 --attention-backend TRITON_ATTN --max-model-len 8192
--max-num-seqs 256 --gpu-memory-utilization 0.92 --no-enable-prefix-caching`:

```bash
GPUS=0,1 MODEL_DIR=<flint-nvfp4-gguf> HF_DIR=<RecursiveSI/flint small files> WORK=<work> DATA=<data> OUT=gen-V1 \
    PROMPTS="prompts-ext.jsonl prompts-ext2.jsonl" bash vllm/run_generation.sh
```

`vllm/check_prompts.py` confirms the prompt ids equal llama-server's, and `vllm/sanity_greedy.py` compares greedy
answers before a run.

**What the release data holds** (answers / answer tokens): Flint Q8_0 7,148 / 8.61 M; G3 14,615 / 9.58 M; UD-Q4_K_XL
7,763 / 3.73 M; NVFP4 in vLLM 42,185 / 52.74 M; in total 71,711 / 74.66 M. Rates: Q8_0 on two RTX 3090s ~370 answer
tokens/s, vLLM on two RTX 3090s ~2.4–2.7k. A 4-bit writer is fine: the Q8_0 dump labels every target. The
[published answers](https://huggingface.co/datasets/CinderellaMan/flint-drafter-data) are these without the COUNT-TBD
answers to ShareGPT prompts.

## 3. The dump

Each answer is replayed through Flint Q8_0 from an empty cache; per position the dump stores the inputs of 26 layers
and the final hidden state (FP8 E4M3, one scale per row) and the top-20 next-token log-probabilities.

```bash
python feats.py seqs seqs.txt map.jsonl gen-*.jsonl
DUMP_IN=seqs.txt DUMP_OUT=feats/train-x.bin \
DUMP_LAYERS=4,6,8,9,10,12,14,16,17,18,20,22,24,25,27,29,30,33,34,36,37,39,41,42,44,47 DUMP_ALLPOS=4,6,9 \
DUMP_OUT2=pack/train-x.bin DUMP_LAYERS2=9,17,25,33,41 DUMP_THREADS=8 \
    llama-staircase-dump -m flint-Q8_0.gguf -ngl 99 -fa on -c 8192 -b 2048 -ub 512
python flint_weights.py flint-Q8_0.gguf flint/
python check_dump.py feats/train-x.bin flint/
```

- The grid puts a recorded layer at or 1–2 layers below every card boundary of any split from 3 to 12 cards.
- On narrow PCIe links add `DUMP_GPU_TOPK=1` with `-b 512` (top-20 on the card, exact).
- The first 13.94 M answer tokens (rounds d1–d3) used a 17-layer superset
  (`6,9,12,17,18,24,25,29,30,33,34,36,39,41,42,44,47`, `DUMP_ALLPOS=6,9`): every boundary of the six- and nine-card
  splits.
- `feed_split.py` shares one pool between several dump instances; tool answers go last (~60 % of their tokens are
  prompt, replayed but not trained on).
- **Size and rate:** ~58 GB per million answer tokens on the grid (~37 GB on the superset); ~1.0–1.2k tokens/s (prompt
  and answer) on 2 × RTX 3080 + RTX 3090.
- **The test set:** Q51's 38 transcripts (`bench/expert_overlap/prompts.py` and `generate.py`, Unsloth Q8_0, a fixed
  seed per prompt), dumped with the same command (`DUMP_IN=q51/seqs.txt`) into `q51-grid.bin`.

## 4. Vocabulary

```bash
python vocab.py vocab.npy gen-*.jsonl --size 40000 --test q51/gen.jsonl
```

40,000 ids from the first 11.6 M answer tokens plus Flint's added and control tokens; anything else is "other".

## 5. Training

`train.py` with `--flint flint/ --vocab vocab.npy`; one head per GPU, checkpoints every 30 minutes (`--resume`
continues).

| Run | From | Arguments | Data | GPU and time |
|---|---|---|---|---|
| U | scratch | `--head U --feats pack/train-d1..d3b.bin --minutes 300 --lr 2e-4 --warmup 100` | six-card packs, 13.5 M | RTX 3080 12 GB, best at 180 min (validation 0.615) |
| R1 | U | `--head A --init-from U/head.pt --fuse --window-sample 0.6,0.25,0.15 --lr 1e-4 --warmup 100 --minutes 480 --val-seqs 300` | six-card packs d1–d3b, 13.5 M | RTX 3090, best at 180 min (0.580); Q51 0.502 fresh / 0.562 overall |
| R2-gen | R1 | `--head A --general --init-from R1/head.pt --lr 1e-4 --warmup 100 --eval-every 20 --val-seqs 300 --minutes 134 --fp16` | superset and grid files d1–d4, 27.6 M | V100 (FP16), ~1.2 passes |
| R2-69 | R1 | the same with `--cards 6,9` | the same | V100 (FP16) |
| R2-9 | R1 | the same with `--cards 9` (no `--fp16`) | the same | RTX 3090 (BF16) |
| Final round | R2-9 (and R2-gen) | R3-9: `--init-from R2-9/head.pt --cards 9e,9l,9l`; variants R3-9L4 (`--layers 4`), R3-9lk (`--loss lk --eta 3`), R3-9l (light split only) and R3-gen (R2-gen continued); RIG-TBD: the exact arguments of the winner | ~50 M answer tokens | RIG-TBD |

- `--general` draws one window per sequence from a random split of 3–12 cards (weights in `train.py`'s
  `CARD_WEIGHTS`: a quarter each on six and nine cards); `--cards` restricts it to the standard splits of those counts.
  The general head drops R1's fused-anchor terms when warm-started (`init_general`).
- The R2 runs were restarted once at 30 minutes from their checkpoint with `--resume` (same budget).
- ~27 steps per minute (~4 answers/s) per head on an RTX 3090 or V100.
- Checks before a long run: `check_train.py`, `check_general.py`.

## 6. Scoring and simulation

```bash
python score.py --heads heads/R2-x --feats q51-grid.bin --gen q51/gen.jsonl --vocab vocab.npy --flint flint/ \
    --out scores/R2-x --alpha 3 --window-group 2
python report.py scores/R2-x/summary.json
python sim.py --gen q51/gen.jsonl --scores scores/sim --jobs 8 --out sim.json
```

| Head | Card count | Q51 fresh text | Q51 overall | Simulated speed |
|---|---|---|---|---|
| R1 | 6 | 0.502 | 0.562 | 1.90× plain decoding (70.9 tok/s) |
| **R2-9, the nine-card release head** | 9 (6/…/5/2) | 0.476 | 0.538 | RIG-TBD |
| R2-9 | 9 (even) | 0.492 | 0.553 | — |
| R2-9 | 6 | 0.507 | 0.567 | — |
| R2-gen | 9 (6/…/5/2) / 9 (even) | 0.469 / 0.488 | 0.531 / 0.549 | — |
| **R2-gen, the release head for six and other counts** | 6 | 0.523 | 0.583 | RIG-TBD |
| R2-gen | 3 / 12 | 0.556 / 0.449 | 0.624 / 0.508 | — |

Acceptance is the share of guesses that equal Flint's sample at the makers' sampling (coupled, α 3); "fresh text" is
what the copy lookup in front of the drafter cannot guess. The pick is made on the nine-card split with a light last
card (`-ts 6,6,6,6,6,6,6,5,2`).

## 7. Export

```bash
PYTHONPATH=<llama.cpp>/gguf-py python export_gguf.py --head heads/R2-9 --vocab vocab.npy --out flint-staircase-drafter-9cards-v2.0.gguf --alpha 3
PYTHONPATH=<llama.cpp>/gguf-py python export_gguf.py --head heads/R2-gen --vocab vocab.npy --out flint-staircase-drafter-v2.0.gguf --alpha 3
```

`--alpha` is the head's own value (`alpha.txt` in its folder when validation chose one; 3 for every head so far).

Q8_0 matrices with the output head in F16 (~0.035 rms logit error on real hidden states), ~0.27–0.29 GB. A
depth-general head is detected from its config; the side file records the 26-layer grid and the stage timing the
engine uses to build the window from the run's own split. The format: [docs/drafter.md](../../../docs/drafter.md),
step 7.

## 8. Checks on any CUDA GPU

- **The C++ head against PyTorch** (random weights, windows from a split):
  `python check_export.py case case-G9 --general --split 6,6,6,6,5,5,5,5,5 --windows 6,6,6,7,8,1,2,6`, then
  `llama-pipe-drafter-check case-G9/drafter.gguf case-G9/case.bin --split 6,6,6,6,5,5,5,5,5`. Pass: `bad=0`, logits
  within 0.3 % of their standard deviation. The set of 14 cases passes on a V100 with every top id equal; on the CPU
  one near-tie top id differs.
- **The output never depends on the head** (a random head of the real shape, every guess wrong):
  `python check_export.py head rand-G.gguf --general --f16 head`, then `bench/pipespec/check_identity.sh` with
  `DRAFT=rand-G.gguf DEPTHS='1 3 5'` (tokens, per-token logit hashes, KV state and a probe equal to plain decoding),
  and `bench/pipespec/check_server.sh` with `--pipe-draft rand-G.gguf --pipe-store <store> GUESSES='cont staircase'`.

## 9. The continuation store

The release guesser consults, after the conversation itself, a store of Flint's earlier answers (`--pipe-store`): 8
bytes per token, memory-mapped, built with `bench/ngram_replay/datastore.cpp` from the generated answers; ours holds
13.8 M tokens (110 MB). How it is built and what it adds (~5 %): [bench/ngram_replay](../../../bench/ngram_replay/README.md).

## Compute and data, in total

| Step | Rate here | Data |
|---|---|---|
| Prompts | minutes | 76,431 prompts |
| Answers | vLLM ~2.4–2.7k answer tokens/s on two RTX 3090s; llama.cpp copies ~110–470 | 74.66 M answer tokens |
| Dump | ~1.0–1.2k tokens/s (prompt and answer) per instance; three instances shared one pool | ~1.3 TB for 27.6 M answer tokens (superset and grid) |
| Training | ~27 steps/min per head; R1 180 min, the R2 heads 134 min each | one head per GPU |
| Scoring, export | minutes | a 0.27–0.29 GB side file |

Neither the generated answers nor the dumps are in this repository: they are tens of gigabytes to terabytes. They are
defined by the prompt files, the writers' settings and the sampling seeds above. A manifest of the answer files
(prompt numbers, writer, token counts, SHA-256 per file) is listed under "Gaps" in [the index](../README.md).
