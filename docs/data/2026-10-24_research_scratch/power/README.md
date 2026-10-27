# Power and clocks per benchmark test, from the rig's nvidia-smi traces (research round 16, 2026-10-24)

The derivation behind [power_and_tuning.md](../../../research/notes/power_and_tuning.md) section 1 (draw per card and
per mode, the clock-event reasons, the cap share, joules per token): the rig's benchmark runs of 2026-10-22 and
2026-10-23 logged `nvidia-smi` next to their `llama-bench` output, and these scripts cut those traces into one row per
`llama-bench` test.

| File | What |
|---|---|
| `phases.txt` | 220 rows, one per `llama-bench` test of every run folder with a trace: run folder, model, test (`pp512`, `tg128`, …), depth, micro-batch, cards in use, tok/s, mean utilization, mean power over all and over busy samples, peak, the range of per-card means, busy SM clock (mean and minimum), power-cap share (bit `0x4` in samples at ≥ 5 % utilization), hardware and thermal slowdown sample counts, highest temperature, the nine cards' summed power, joules per token, cards busy at once (mean and maximum), the bitmasks seen |
| `smi_phases.py` | makes `phases.txt`: test windows from each jsonl row's `test_time` and `samples_ns` (plus the warm-up), samples of the cards that hold memory in that window |
| `busy_samples.py` | idle power by P-state, and per run the samples at ≥ 80 % utilization: power, SM clock, cap share, cards busy at once |
| `capclk.py` | bitmask counts; SM clock and power in capped against uncapped busy samples (≥ 90 %); memory clock under the cap; samples above 75 W; uncapped clock by temperature |

The run folders are named as in [docs/data](../../) (date and experiment): the E0.8 calibration, the Flint and
speed-up benchmarks (`bench/`; `bench_all/` repeats the same runs), the Qwen baselines, Q38, Q46 and step 1. Each
held an `smi.csv` next to its `llama-bench -o jsonl` output, written during the run by
[scripts/bench.sh](../../../../scripts/bench.sh)'s query (or the same columns in the experiment's own script):

```bash
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -lms 500
```

The traces themselves are raw output and are not kept here.

Made on 2026-10-24 (CPU only, standard library):

```bash
python3 -I smi_phases.py <results>/<run folder> ... > phases.txt
python3 -I busy_samples.py <results>/*/smi.csv
python3 -I capclk.py <results>/*/smi.csv
```

`smi_phases.py` reads the trace's timestamps as local time at UTC+2 (the rig's clock in October 2026, before the
change to winter time) and the jsonl's `test_time` as UTC. Measured data and scripts of this repository (MIT).
