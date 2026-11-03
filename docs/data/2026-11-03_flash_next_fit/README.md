# Q213: which Flash-Next files and settings fit on nine 6 GB cards? (2026-11-03)

Qwen3.8-Flash-Next's rig-sized files are 37.7 GB (Swift-1.5 GSQ-RCO Q2_0) and 47.2 GB (IQ3_XXS) once the 28.8 GB
n-gram table, which llama.cpp keeps in host memory, is left out. Nine P106-090 cards hold 54.0 GB between them, but
the KV cache, the recurrent state and the compute buffers take their share, and a layer cannot be split between cards.
This run asks llama.cpp itself how much each card would hold, for both files, at 32k to 262k of context, with F16 and
q8_0 cache, micro-batches (`-ub`) of 64 to 512 and with and without an MTP draft, and finds the split that leaves the
most room. Nothing was loaded and nothing was timed: the answer is memory only. The research behind the files is in
[flash_next_variants.md](../../research/notes/flash_next_variants.md); the registry row is Q213 in
[quants/experiments.md](../../../quants/experiments.md).

## Result: Q2_0 fits with room to spare; IQ3_XXS fits up to 48k

Largest context with at least 150 MiB left on every card, F16 cache, Miner Llama v2.0 (= v3's base), each row with its
best `-ts` from [scripts/split_plan.py](../../../scripts/split_plan.py). Contexts probed: 32k and 48k (IQ3_XXS), 64k,
128k, 262k. In brackets: free MiB on the fullest card.

| File | `-ub 64` | `-ub 128` | `-ub 256` | `-ub 512` |
|---|---|---|---|---|
| Q2_0 | 262k (224) | 128k (925) | 128k (664) | 64k (735) |
| Q2_0 + MTP draft | 128k (678) | 128k (570) | 128k (309) | 64k (519) |
| IQ3_XXS | 48k (247) | 48k (187) | 32k (157) | none (64k: −414) |
| IQ3_XXS + MTP draft | 32k (218) | none (32k: 149) | none | — |
| IQ3_XXS, 5–10 expert tensors moved (`-ot`) | 128k (301) | 128k (300) | 64k (529) | — |
| IQ3_XXS + MTP draft, 5–6 moved | 64k (186) | 64k (201) | — | — |

The settings for R13 (Q215), Swift Q2_0 with and without the draft, stock against v3:

| Line | Settings (`-fa on -fit off`, F16 cache) | Fullest card free: v2.0 / stock |
|---|---|---|
| Baseline | `-c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6` | 735 / 736 MiB |
| Long context | `-c 131072 -b 2048 -ub 256 -ts 5,6,5,6,5,5,6,5,6` | 664 / 666 MiB |
| MTP draft (R14) | `-c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -md <draft> --spec-type draft-mtp --spec-draft-n-max 3` | 519 / 520 MiB |
| MTP draft, long context | the same with `-c 131072 -ub 128` | 570 / 572 MiB |

- **Q2_0 has room.** Its 48 layers are 725–757 MiB each, so whole layers balance well: at 64k every micro-batch
  fits, at 128k up to `-ub 256`, and 262k only at `-ub 64`. It needs 7 cards at 64k (`-ub 128`: 328 MiB left) and 8
  at 128k (217 MiB), so a rig with fewer cards still runs it at shorter context.
- **IQ3_XXS is tight because its layers are uneven.** They run from 701 to 1,214 MiB (`layers.tsv`), so the best
  whole-layer split still leaves one card short while others keep ~1 GB: at 64k and `-ub 128`, 5,680 MiB is free
  across the cards but only 79 MiB on card 2. Moving a few routed-expert tensors to the next card (`split_plan.py
  --split-experts`, written as `-ot 'blk\.18\.ffn_down_exps=CUDA3,…'`) balances it, but **any `-ot` turns off
  llama.cpp's pipeline parallelism** (the overlap of micro-batches across the cards in prompt processing), which is
  also why those rows' compute buffers are smaller (77 instead of 180 MiB per card at 64k, `-ub 128`). What that
  costs in prompt speed is not measured (Q510).
- **The q8_0 cache buys no context on these cards.** It halves the cache (1.34 instead of 2.52 KiB per token per
  attention layer), but each card's compute buffer grows by about as much (128k, `-ub 128`: 557 instead of 304 MiB).
  Q2_0 fits the same contexts in both; only IQ3_XXS at `-ub 64` gains a step (64k, 192 MiB). Why the buffer grows is
  Q508.
- **Compute buffers grow with micro-batch × context**, by 13–16 bytes per (micro-batch token × context token) on
  every card: 180 MiB at 64k and `-ub 128`, 1,961 MiB at 262k and `-ub 512` (F16). That, not the cache, is what limits
  262k. Upstream's change that drops the sparse-attention mask (#30815, Q463) targets this buffer.
- **The MTP draft costs the last card 2,243–4,462 MiB** (64k at `-ub 128` to 262k at `-ub 512`): its 1,917 MiB of
  weights (its 644 MiB of token embeddings stay in host memory), one attention layer's cache and its own compute
  buffer. The main model also keeps a copy of the recurrent state per drafted token: 12.3 instead of 3 MiB per Gated
  DeltaNet layer at `--spec-draft-n-max 3`. The last card then holds only two or three of the model's layers.
- **The default split wastes room:** llama.cpp's equal shares leave Q2_0 583 MiB at 64k and `-ub 512` (planned:
  735), and put IQ3_XXS 828 MiB over on card 6 at 64k and `-ub 128`.
- **Stock llama.cpp projects the same** as v2.0, within 4 MiB per card on every line above, so R13's two engines can
  share settings.
- **Host memory:** the n-gram table and token embeddings stay mapped (27,726 MiB, read through the page cache), and
  the buffers llama.cpp allocates in RAM take 50–380 MiB at 64k (`-ub 64` to `512`) and up to 1,350 MiB at 262k; the
  draft adds 50–1,400 MiB. Of the host's 4 GB, about 2.5 GB is free before the server starts.

These are llama.cpp's projections, not measurements of a loaded model: Flash-Next has not been loaded on the rig yet.
Prompt batches also take some memory outside the projection, which is what the 150 MiB margin is for (it held for
Ornith-1.5, [Ornith data](../2026-10-27_ornith/README.md)). R13's first load records the free memory per card.

## Files

| File | What |
|---|---|
| `fit_projection.csv` | one row per probe (184): file, engine, context, micro-batch, cache type, `-ts`, `-ot` moves, MTP draft; llama.cpp's projected MiB per card (with a draft, the draft's share included), the smallest free MiB and its card, the total on the cards, host memory mapped (the n-gram table and token embeddings, read through the page cache) and allocated (buffers in RAM), and whether every card keeps 150 MiB |
| `fit_parts.csv` | the memory breakdown behind each row: per probe, model (main or draft) and card, MiB of weights, context (KV cache plus recurrent state) and compute buffer |
| `split_plans.txt` | [scripts/split_plan.py](../../../scripts/split_plan.py)'s best split per configuration, file and draft, with every card's layers, attention layers, MiB and free MiB; the `-ot` plans move the fewest expert tensors that leave 150 or 300 MiB on every card |
| `layers.tsv` | per layer of both files: attention or Gated DeltaNet, MiB on the card, of which routed experts |
| `run/fit_grid.sh` | the probes: every combination of context, cache type, micro-batch and draft for one file, through [scripts/fit_probe.sh](../../../scripts/fit_probe.sh), with llama.cpp's default split, one `-ts`, or each configuration's planned split |
| `run/plan_splits.sh` | the planning step: the per-layer costs measured with the default split, and `split_plan.py` per configuration |
| `run/fit_table.py` | the probe logs turned into `fit_projection.csv` and `fit_parts.csv` |

## How it was made

On 2026-11-03, on nine P106-090 cards (6,068 MiB each; 6,001–6,007 MiB free once llama.cpp's own CUDA context is up,
the desktop holding 5 MiB per card), with Miner Llama v2.0 (`37e717ff6`, which is also v3's base) and stock llama.cpp
`d77256999` (upstream build 11820), both built for compute 6.1. Both know the `qwen4exp` architecture and Q2_0.

| File | Hugging Face repository, revision | Shard bytes | Header ends at byte | SHA-256 (recorded at download) |
|---|---|---|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf` | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` | 39,799,117,984 + 26,750,834,816 | 10,967,968; 56,192 | `79e2a3873f1b05df835a1a191b20f2e74b9e12df949cd89861d17d98b6ef9678`, `ef3bb04fb4e14f04624abd8acf25faf1603594a034050b048df85ec152ad5ae0` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` | the same | 39,785,790,560 + 36,180,282,560 | 10,966,240; 57,920 | `3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23`, `b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160` |
| `mtp-Qwen3.8-Flash-Next-draft.gguf` (MTP draft: one layer, 49 blocks declared) | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) `273edc1d2ffae5241fe3cab69568d682dad4c718` | 2,697,439,648 | 10,946,976 | `008183784ac5aa2cdcd3d7777e4e6febb0645ae4b8d1da1c065b2149c1d6391b` |

**The rig held header-only copies.** llama.cpp's projection reads only a file's header, so the rig got the first
12 MiB of each first shard and of the draft and the first 1 MiB of each second shard, each extended to its full size
with `truncate -s`: the five files read as complete and take 55 MB of disk. After its projection the server goes on to
load the weights (zeros here); `fit_probe.sh` stops it two seconds after the verdict. The projections agree with
`split_plan.py`'s sums from the real files' headers within 4 MiB per card (8 MiB for IQ3_XXS). With a draft they come
out ~36 MiB lower, since the planner models the draft only roughly, and with `-ot` 90–140 MiB higher, since the
planner counts the compute buffers measured with pipeline parallelism on.

From the repository root, with `RIG` the rig's ssh alias (default `rig`):

```bash
# header-only copies on the rig, from a machine that holds the files
for f in <first shard> <draft>; do ssh <source> "head -c 12582912 $f" | ssh $RIG "cat > <dir>/$(basename $f) && truncate -s <bytes> <dir>/$(basename $f)"; done
for f in <second shard>; do ssh <source> "head -c 1048576 $f" | ssh $RIG "cat > <dir>/$(basename $f) && truncate -s <bytes> <dir>/$(basename $f)"; done
D=docs/data/2026-11-03_flash_next_fit/run
export BUILDS=<rig builds folder> BUILD=<v2.0 build folder> DRAFT=<dir>/mtp-Qwen3.8-Flash-Next-draft.gguf

# 1. llama.cpp's default split: the per-layer costs (cache, state, compute) for every context, micro-batch, cache type
MODEL=<dir>/<Q2_0 first shard> TAG=q2-def OUT=probe_q2_default.md bash $D/fit_grid.sh
MODEL=<dir>/<Q2_0 first shard> TAG=q2-def UBS=64 CTXS="32768 49152 65536 131072 262144" OUT=probe_q2_default_more.md bash $D/fit_grid.sh
MODEL=<dir>/<Q2_0 first shard> TAG=q2-def UBS="128 256" CTXS="32768 49152" OUT=probe_q2_default_more.md bash $D/fit_grid.sh
MODEL=<dir>/<Q2_0 first shard> TAG=q2-def MTP=1 OUT=probe_q2_default_mtp.md bash $D/fit_grid.sh
MODEL=<dir>/<IQ3_XXS first shard> TAG=iq3-def OUT=probe_iq3_default.md bash $D/fit_grid.sh

# 2. the best split per configuration (where the full shards are; header reads only)
SHARDS="<Q2_0 shard 1> <Q2_0 shard 2>" TAG=q2 bash $D/plan_splits.sh > plan_q2.txt            # and MTP=1 > plan_q2_mtp.txt
SHARDS="<IQ3_XXS shard 1> <IQ3_XXS shard 2>" TAG=iq3 bash $D/plan_splits.sh > plan_iq3.txt     # and MTP=1
SHARDS="<IQ3_XXS shards>" TAG=iq3 PLAN_ARGS="--split-experts --target 300" bash $D/plan_splits.sh > plan_iq3_ot300.txt  # also 150, MTP=1

# 3. each plan probed
MODEL=<dir>/<Q2_0 first shard> TAG=q2-plan PLANS=plan_q2.txt UBS="64 128 256 512" CTXS="65536 131072 262144" OUT=probe_q2_plan.md bash $D/fit_grid.sh
MODEL=<dir>/<Q2_0 first shard> TAG=q2-plan PLANS=plan_q2_mtp.txt MTP=1 UBS="64 128 256 512" CTXS="65536 131072 262144" OUT=probe_q2_plan_mtp.md bash $D/fit_grid.sh
MODEL=<dir>/<IQ3_XXS first shard> TAG=iq3-plan PLANS=plan_iq3.txt UBS="64 128 256" CTXS="32768 49152 65536 131072" OUT=probe_iq3_plan.md bash $D/fit_grid.sh
MODEL=<dir>/<IQ3_XXS first shard> TAG=iq3-plan PLANS=plan_iq3_mtp.txt MTP=1 UBS="64 128 256" CTXS="32768 49152 65536" OUT=probe_iq3_plan_mtp.md bash $D/fit_grid.sh
# one configuration at a time, e.g. an -ot plan, the stock build, or fewer cards
MODEL=<dir>/<IQ3_XXS first shard> TAG=iq3-ot300 PLANS=plan_iq3_ot300.txt KVS=f16 UBS=128 CTXS=65536 OUT=probe_iq3_ot.md bash $D/fit_grid.sh
BUILD=<stock build folder> MODEL=<dir>/<Q2_0 first shard> TAG=q2-stock PLANS=plan_q2.txt KVS=f16 UBS=512 CTXS=65536 OUT=probe_stock.md bash $D/fit_grid.sh
MODEL=<dir>/<Q2_0 first shard> BUILDS=<rig builds folder> OUT=probe_cards.md \
  CONFIGS='q2-7cards-f16-ub128-64k|<v2.0 build folder>|65536|-b 2048 -ub 128 -dev CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6 -ts 7,7,7,7,7,7,7' \
  bash scripts/fit_probe.sh

# 4. the tables
python3 -I $D/fit_table.py fit_projection.csv fit_parts.csv probe_*.md
```

Every probe runs `llama-server -m <file> -ngl 99 -fa on -c <ctx> -b 2048 -ub <ub> [-ts …] [-ot …] [-ctk q8_0 -ctv q8_0]
[-md <draft> --spec-type draft-mtp --spec-draft-n-max 3 [-ctkd q8_0 -ctvd q8_0]] -np 1 -fit on -lv 5
--ctx-checkpoints 4 --cache-ram 0`. With `-ngl` given, `--fit` prints its projection and then leaves the split as it
is. Planning and the tables ran on the 7-GPU quality machine; the probes ran on the rig with nothing else running.

The files' weights are under the Swift Open License 1.0 and qwen-community-1.0 (the Swift files) and
qwen-community-1.0 (the draft); the tables here hold only memory figures and tensor sizes read from their headers.
