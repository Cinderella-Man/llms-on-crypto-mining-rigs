# Fewer card boundaries: the fewest cards each Flash-Next file needs (2026-11-04)

What this measured: for Qwen3.8-Flash-Next split over the rig's nine 6 GB P106-090 cards, the fewest cards that hold
every layer at a 64k context with at least 300 MiB free on each, from the GGUF headers alone. Registry row Q964 in
[quants/experiments.md](../../../quants/experiments.md); the rig run that checks it is [run/fh_rig.sh](run/fh_rig.sh).

A token crosses a card boundary for every card after the first, and each crossing is a round trip through the host
over PCIe 1.0 x1 (no peer-to-peer). The round trip costs the same whatever the residual's size. On S35's line the Q8_0
hop cast took plain decode from 14.98 to 17.06 tok/s, i.e. 8.1 ms a token over eight boundaries
([2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md)); what each boundary still costs with it is what the rig
run's T(k) step measures. A file that fits on 8 cards runs 7 hops a pass instead of 8. The freed card can then hold the
MTP draft's whole pass. Today the draft's token embeddings sit on card 4 and its layer and head on card 8, so each of a
step's draft passes crosses cards once (Q961).

The planner is [scripts/split_plan.py](../../../scripts/split_plan.py)'s `--fewest` mode, tested by
[scripts/split_plan_test.py](../../../scripts/split_plan_test.py). It scores every card count, keeps the first k cards
and leaves the rest free, and prints the fewest that keep the margin, with `-dev`, `-ts` and any `-ot` moves. Three
lines per file:
- **nodraft:** the main model alone;
- **draft-own:** the MTP draft's whole pass (2,781 MiB) on the first free card, through `-devd` and
  `-otd token_embd=<that card>`;
- **draft-last:** the draft's layer and head on the last card of the split, and its token embeddings on whichever card
  leaves the most free. This is S35's placement on fewer cards.

Costs are llama.cpp's own projection on the rig for Miner Llama's `-kshape` build at `-b 2048 -ub 512`, f16 cache
(S35's fit probe), per card:
- free memory: 6,023 MiB;
- KV cache: 161 MiB per attention layer;
- recurrent state: 3 MiB per Gated DeltaNet layer, 12.3 MiB with a draft;
- compute buffers: 344 MiB on the first card and 419 MiB on the others;
- the draft: 2,136 MiB for its layer, head, cache and compute, plus 645 MiB of token embeddings.

## Result

| File | Line | Cards (hops) | Free card(s) | Fullest card keeps | `-ts` (with `-dev` over the first cards) |
|---|---|---|---|---:|---|
| **pmix-q2-fast** (S35's file) | nodraft | **8 (7)** | CUDA8 | **416 MiB** | `6,6,6,6,6,6,6,7` |
| | **draft-own** | **8 (7)** | CUDA8 holds the draft (3,242 MiB free) | **379 MiB** | `6,6,6,6,6,6,6,7` |
| | draft-last | 8 (7) | CUDA8 idle | 101 whole; 301 with 3 `-ot` moves | `7,7,6,6,6,7,6,4`; `6,7,6,7,6,7,6,4` + 3 moves |
| | nodraft at `-ub 256` (est.) | 7 (6) | CUDA7, CUDA8 | 317 (est.) | `7,7,7,7,7,7,7` |
| pmix-q2-spine6 | nodraft | 8 (7) | CUDA8 | 335 MiB | `6,6,6,6,6,6,6,7` |
| | draft-own | 8 (7) | CUDA8 holds the draft | 297 whole; 375 with 1 `-ot` move | `6,6,6,6,6,6,6,7` |
| t256c1 (cold tier) | nodraft | **7 (6)** | CUDA7, CUDA8 | 462 MiB | `7,7,7,7,7,7,7` |
| | draft-own | **7 (6)** | CUDA7 holds the draft, CUDA8 idle | 415 MiB | `7,7,7,7,7,7,7` |
| t256iq3c1 (cold tier) | nodraft / draft-own | 8 (7) | CUDA8 | 386 / 330 MiB | `7,7,6,6,6,5,6,6` |
| t256iq3, t320iq3 (cold tier) | nodraft | 9 (8) | — | 366 / 244 MiB | as today; no line fits fewer cards, and the draft cannot have a card of its own |

- **pmix-q2-fast fits on 8 cards with whole layers, with or without the draft**, at S35's micro-batch. The draft-own
  line is S35's arm with two changes: the main model gets `-dev CUDA0,…,CUDA7 -ts 6,6,6,6,6,6,6,7`, and the draft gets
  `-devd CUDA8 -otd token_embd=CUDA8` in place of `-otd token_embd=CUDA4`. A pass then crosses seven boundaries
  instead of eight, and each draft pass runs on one card.
- **Keeping the draft on the split's last card does not fit on 8 cards with whole layers** (101 MiB). Three routed-expert
  tensors moved by `-ot` would leave 301 MiB, but any tensor override turns off llama.cpp's pipelined prompt
  (`llama_context` runs no pipeline parallelism with tensor overrides). That line is parked (Q967); Q968 is the fork
  change that would lift the limit for overrides that keep every tensor on a GPU.
- **Seven cards (6 hops) need less memory per card.** For pmix-q2-fast that means `-ub 256`: an estimated 317 MiB
  with no draft and 270 MiB with the draft on its own card. The estimate takes `-ub 256` compute buffers of 176/214 MiB,
  S35's `-ub 512` values scaled by v2.0's measured ratio; the rig probe measures them (Q965). The cold-tier file t256c1
  fits on 7 cards at `-ub 512` (Q966), once its KLD and the tiered kernels' rig timing are in.
- The planner's sums match llama.cpp's projection on the rig within 2 MiB a card on S34's no-draft and S35's MTP
  nine-card splits ([calibration.tsv](calibration.tsv)).
- No N2P or trellis-format Flash-Next file had headers yet. `run/plan.sh` plans them the same way once they do.

## Files

| File | What it is |
|---|---|
| [plans/plans.tsv](plans/plans.tsv) | every card count (9 down to 1) of every line and file: the free cards, the draft card's free MiB, the best whole-layer split's smallest free and `-ts`, and the best split with `-ot` moves (smallest free, tensors moved) |
| [plans/plans-pmix-q2-fast.txt](plans/plans-pmix-q2-fast.txt) and one per file | the planner's output: the calibration splits (pmix-q2-fast), then per line the table and the fewest-card plan with each card's layers, MiB and free MiB |
| [plans/plans-pmix-q2-fast-ub256est.txt](plans/plans-pmix-q2-fast-ub256est.txt) | the same at `-ub 256` with estimated compute buffers (176/214 MiB) |
| [calibration.tsv](calibration.tsv) | per card, llama.cpp's projection on the rig against the planner's sum for S34's no-draft and S35's MTP splits |
| [run/plan.sh](run/plan.sh) | the planning run: the calibration and the three lines per file |
| [run/header_copies.sh](run/header_copies.sh) | header-only copies of GGUF shards (sparse, a few MB packed) for `scripts/fit_probe.sh` on a rig that lacks the files |
| [run/fh_rig.sh](run/fh_rig.sh) | the rig run: llama.cpp's projection (no weights loaded), one real load per line, T(k) on real text for 9 and 8 cards, and the agent check against S35's arm |

## Commands

From the repository root, on any machine that holds the files (only their headers are read):

```bash
D=docs/data/2026-11-04_fewer_hops
FILES="pmix-q2-fast=<shard1>,<shard2>,<shard3> pmix-q2-spine6=<shard1>,<shard2>,<shard3>" OUT=$D/plans bash $D/run/plan.sh
FILES="pmix-q2-fast=<shard1>,<shard2>,<shard3>" OUT=$D/plans TAG=-ub256est COMPUTE=176,214 LINES="nodraft draft-own" \
  CALIBRATE=0 bash $D/run/plan.sh
# one fewest-card plan by hand: any card count, any free memory per card (e.g. 12 cards of 8 GB)
python3 -I scripts/split_plan.py --cards 12 --free 8100 --kv-mib 161 --state-mib 12.3 --compute 344,419 --fewest \
  --margin 300 --own-card 2781 --split-experts <shard1> <shard2> <shard3>
# the planner's self-test (brute force over every split, six planted bugs)
python3 -I scripts/split_plan_test.py /tmp/split_plan_test
# header copies for a rig's projection
FILES="pmix-q2-spine6=<shard1>,<shard2>,<shard3>" OUT=headers bash $D/run/header_copies.sh
```

On the rig, from the repository root with S12's machine file filled in
([s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example), with `DRAFT`, `RIG` and S35's ports and client
folder), the rig idle:

```bash
ENVFILE=s12.env MODEL=<pmix-q2-fast shard 1 on the rig> BUILD=llama.cpp-20261104-39c3ab5f1-pascal-p106-v3-b12076-kshape \
  OUT=/tmp/fh RDIR=<rig work folder> TEXT=<agent text on the rig> PROBE_FILES="spine6=<its header copy's shard 1>" \
  bash docs/data/2026-11-04_fewer_hops/run/fh_rig.sh
```

## Provenance

The files are pascal-mix's ([2026-11-04_pascal_mix](../2026-11-04_pascal_mix/README.md): pmix-q2-fast, pmix-q2-spine6)
and the cold tier's donors ([2026-11-04_cold_tier](../2026-11-04_cold_tier/README.md): t256c1, t256iq3c1, t256iq3,
t320iq3). The MTP draft is the Gyro draft of [2026-11-03_flash_next_fit](../2026-11-03_flash_next_fit/README.md): one
layer (`blk.48`, 1,421 MiB), a 497 MiB head and 644 MiB of token embeddings. The costs come from S34's and S35's fit
probes ([2026-11-04_stacked_gains](../2026-11-04_stacked_gains/fit_probe.txt),
[2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md)). The tables are this project's work; the models keep their
makers' licences.
