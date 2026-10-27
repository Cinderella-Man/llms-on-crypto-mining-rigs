# E0.8: power, clocks and the power cap per card (2026-10-22)

The single-card calibration ran Llama-2-7B on each of the nine P106-090 cards, then on 2 and 9 cards, with an
`nvidia-smi` sample of every card every 500 ms. The speed results are in
[2026-10-22-single-card-calibration.md](../../benchmarks/2026-10-22-single-card-calibration.md). This folder holds
what the power and clock traces show for each test: power draw, SM clock, utilisation and how often the driver
reports the software power cap, per card.

| File | What |
|---|---|
| `smi_summary.csv` | one row per run, GPU set, test (pp512 or tg128, flash attention off or on) and card: llama-bench tok/s, samples in the test's window, mean utilisation, mean and peak power (W), mean, min and max SM clock (MHz), memory clock, P-states seen, share of samples with the software power cap flag (`0x4`), with a hardware slowdown or power-brake flag (`0x8`, `0x80`), with a thermal flag (`0x20`, `0x40`), peak temperature (°C) |
| `run/smi_summary.py` | builds the CSV from a run folder's `smi.csv` and llama-bench JSONL files |

What the summary shows:

| Setup | Test | Power per card | SM clock | Samples with the power cap flag |
|---|---|---|---|---|
| Q4_0, one card (each of the nine) | tg128 | 68–72 W | 1,637–1,779 MHz | 46–86 % |
| Q4_K_M, one card | tg128 | 69 W | 1,671–1,707 MHz | 84–93 % |
| Q4_0, two cards (0+1, 0+4) | tg128, FA on | 55–59 W | 1,809–1,862 MHz | 10–25 % |
| Q8_0, two cards (0+1) | tg128 | 55–57 W | 1,785–1,811 MHz | 22–26 % |
| Q4_0, all nine cards | tg128, FA on | 29–37 W | 1,360–1,733 MHz (mean 1,598) | 0 % |

A single card runs into the 75 W cap while it generates, two cards each stay under it, and with nine cards each one is
busy only ~10 % of the time and its clock drops between bursts. No card showed a thermal or hardware slowdown flag.

Made on 2026-10-22 on the rig with [scripts/bench.sh](../../../scripts/bench.sh) and the rig's llama.cpp build at that
time (upstream `85d5703a3`), TheBloke's Llama-2-7B GGUFs from `/mnt/models`:

```bash
RIG=${RIG:-rig}
M=/mnt/models/TheBloke/Llama-2-7B-GGUF
ssh "$RIG" "MODEL=$M/llama-2-7b.Q4_0.gguf TAG=q4_0 GPU_SETS='0 1 2 3 4 5 6 7 8' BENCH_ARGS='-ngl 99 -fa 0,1 -p 512 -n 128 -r 5' bash -s" < scripts/bench.sh
ssh "$RIG" "MODEL=$M/llama-2-7b.Q4_K_M.gguf TAG=q4_k_m GPU_SETS=0 BENCH_ARGS='-ngl 99 -fa 0,1 -p 512 -n 128 -r 5' bash -s" < scripts/bench.sh
ssh "$RIG" "MODEL=$M/llama-2-7b.Q8_0.gguf TAG=q8_0_2gpu GPU_SETS=0,1 BENCH_ARGS='-ngl 99 -fa 0,1 -p 512 -n 128 -r 5' bash -s" < scripts/bench.sh
ssh "$RIG" "MODEL=$M/llama-2-7b.Q4_0.gguf TAG=q4_0_split GPU_SETS='0,1 0,4 0,1,2,3,4,5,6,7,8' BENCH_ARGS='-ngl 99 -fa 1 -p 512 -n 128 -r 5' bash -s" < scripts/bench.sh
```

Each run writes `/tmp/bench/<TAG>/` on the rig: `smi.csv` and one `gpus_<set>.jsonl` per GPU set. The summary was
built from those four folders with `python3 run/smi_summary.py q4_0 q4_k_m q8_0_2gpu q4_0_split > smi_summary.csv`
(Python 3 standard library only). A test's window starts at its llama-bench `test_time` and covers the warm-up and
the five repetitions; the raw traces (2.8 MB) are not published.
