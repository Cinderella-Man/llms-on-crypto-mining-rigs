# Q141: PowerMizer "Prefer Maximum Performance" against the cold-clock decode (2026-10-24)

Q141 asked whether the cards' default clock policy costs decode speed when a request starts on idle cards, and whether
NV-CONTROL's PowerMizer mode 1 ("Prefer Maximum Performance") recovers it. It does not: the same three measurements in
mode 2 (Auto, the default) and mode 1 give the same speed, and mode 1 only keeps idle cards in P0 at ~26 W instead of
P8 at ~6 W. The result is registry row Q141 ([quants/experiments.md](../../../quants/experiments.md)) and section 3.1
of [power_and_tuning.md](../../research/notes/power_and_tuning.md).

| Measurement (UD-Q4_K_XL, six cards) | Mode 2 (Auto) | Mode 1 (Prefer Maximum Performance) |
|---|---|---|
| tg128 from idle (`llama-bench -p 0 -n 128 -r 3`) | 29.79 ± 1.61 tok/s | 29.75 ± 1.66 |
| pp2048, then tg128 in the same process (`-p 2048 -n 128`) | 510.22 ± 1.73 / 35.07 ± 0.96 | 508.33 ± 0.39 / 34.31 ± 1.12 |
| llama-server, 28-token prompt, 500-token reply after 60 s idle | 33.30 tok/s (prompt 20.6) | 33.25 (prompt 20.6) |
| Busy cards' SM clock, median per card | 1,860–1,911 MHz | 1,847–1,911 MHz |
| SM clock of a card's first busy sample in each step | 1,354–1,911 MHz | 1,354–1,493 MHz |
| Idle unused card (6–8) | P8, 139 MHz SM, 405 MHz memory, 5.7–6.0 W | P0, 1,354 MHz SM, 4,006 MHz memory, 26.0–26.3 W |

Mode 1 neither raises the busy clock nor makes a card start a step at a higher clock; its only effect is ~+20 W per
idle card (~+180 W for nine). The cold start is a `llama-bench` tg-only effect: a served reply after a minute idle
already runs within ~5 % of the warm tg128.

| File | What |
|---|---|
| `clocks_summary.csv` | per mode, step, card (0–8) and state (`active`: utilization above 0, `idle`): samples, seconds, SM clock p10 / median / max and at the first active sample, median memory clock, mean and maximum power, P-state counts |
| `run/q141.sh` | the A/B: phase A in the current mode, phase B with mode 1 on all nine cards, the mode restored to 2 on exit; clocks and power logged every 200 ms |

Steps in `clocks_summary.csv`, each from its start time in the run's log to the next one: (1) 60 s idle, then the
tg128 from idle (model load included); (2) the pp2048 + tg128 test; (3) llama-server's load, 60 s idle and the
500-token reply. Cards 6–8 hold no layers and show each mode's idle state.

Made on 2026-10-24 (14:12–14:38) on the rig with the combined v2 build of the fork (`b3052a4f8`), Unsloth's Flint
UD-Q4_K_XL on cards 0–5, and NVIDIA driver 580.178.04 with Xorg running:

```bash
RIG=rig BIN=<build>/bin MODEL=<path>/flint-UD-Q4_K_XL.gguf XAUTH=<desktop session Xauthority> OUT=<results> \
  bash docs/data/2026-10-24_rig_q141/run/q141.sh
```

`clocks_summary.csv` is computed from the two 200 ms traces (`raw/clk_A_mode2.csv`, `raw/clk_B_mode1.csv`, 4.2 MB,
not kept) with the step start times above; a sample took ~0.23 s per card. Measured data of this repository (MIT).
