# Card clocks during the agent checks of 2026-10-27

How often the rig's nine P106-090 cards ran at their boost clock while a coding agent worked against them, for Flint
and for Ornith-1.5-122B. Idle cards fall to the 1,354 MHz base clock, and Flint's plain decoding loop ran 13 % faster
at boost than at base (38.5 against 34.0 tok/s, [idle cards](../../research/notes/idle_cards.md)); this checks how much
of the agent check runs at base. Registry row: Q154 in [quants/experiments.md](../../../quants/experiments.md).

## Results

Mean over the nine cards of the share of samples at each clock (`clock_share.tsv` has every card):

| Run | Engine | Boost ≥ 1,700 MHz | Base 1,300–1,699 MHz | Low < 1,300 MHz |
|---|---|---|---|---|
| Flint Q8_0, release line (`--pipe-depth 4`) | v2.0 | 81 % | 4 % | 15 % |
| Flint Q8_0, release line + answer store | v2.0 | 80 % | 4 % | 16 % |
| Flint Q8_0, release line + nine-card MTP head | v2.0 | 76 % | 11 % | 13 % |
| Ornith-1.5-122B UD-Q2_K_XL | stock llama.cpp | 33 % | 61 % | 6 % |
| Ornith-1.5-122B UD-Q2_K_XL | v2.0 | 43 % | 50 % | 7 % |

- **Flint's release line already holds the cards at boost.** Its pipelined guesses keep every card busy enough for the
  driver's clock policy, so holding the clock by other means adds little there.
- **Ornith runs at base clock half the time.** Pipelined guessing does not yet run on this hybrid model, so it decodes
  one token at a time and each card is busy ~1/9 of a token. With half the samples at base and 13 % between base and
  boost, a held boost clock is worth est. +6 % on Ornith's agent check, up to ~+13 % if the base-clock samples are the
  decoding ones. The ways to hold it: pipelined guessing on the hybrid, or a clock setting the driver accepts on these
  cards (PowerMizer mode 1 does not hold it).
- **Limits:** one instantaneous sample per card every 10 s; a sample counts when the model is loaded and any card shows
  utilization above 0 %, so pauses while the agent runs tools fall partly into "low". A 100 ms log during decoding gives
  the decoding-only split.

## Files

| File | What |
|---|---|
| `clock_share.tsv` | per run and card: samples, share at boost, base and low clock |
| `run/clock_share.sh` | the tally, from the `gpu.csv` that `bench/agent_check/rig_server.sh` writes on the rig |

## Made with

```bash
bash run/clock_share.sh flint-release=<run>/rig/gpu.csv ornith-ud-v2=<run>/rig/gpu.csv ... > clock_share.tsv
```

The `gpu.csv` files come from the agent checks of 2026-10-27 ([MTP head](../2026-10-27_mtp_head/README.md),
[answer store](../2026-10-27_answer_store/README.md), [Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md));
`rig_server.sh` logs `nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu`
every 10 s. The raw logs stay outside the repository.

## Provenance and licence

Measurements on this project's rig; the table and script are under this repository's licence.
