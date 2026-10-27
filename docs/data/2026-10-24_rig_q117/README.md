# Q117: stock, v1, the v2 base and the combined v2 build on UD-Q4_K_XL (2026-10-24)

llama-bench on the rig with Unsloth's Flint UD-Q4_K_XL: stock llama.cpp (`d77256999`), Miner Llama v1 (`eeaa59660`),
the v2 base (`032b130f4`) and the combined v2 test build (`b3052a4f8`: the v2 base plus Q108/Q109 prefill, the Q110
card-built prompt mask, the Q100 norm and Q8_1 fusion, the Q92 screened head and the Q78 multi-token kernels, each with
its own off switch) on six and nine cards; then the combined build with one of its
switches off at a time; then a tg-only check of whether the combined build's lower decode speed was real. Result:
registry row Q117 in [quants/experiments.md](../../../quants/experiments.md); the order-effect reading is in
[2026-10-24_q108_split](../2026-10-24_q108_split/README.md). These are single-request diagnostics, not the release's
agentic numbers.

| File | What |
|---|---|
| `llama_bench.csv` | 65 rows: `part` (headline, ablation, tg-only check), `arm` (build, or the switch turned off), cards, round, start time, the card temperatures (°C) read just before the run, cards 0–8 (headline and ablations) or 0–5 with their SM clocks (tg-only check), test (pp512, pp2048, pp8192, tg128), tok/s and standard deviation |
| `run/q117.sh` | the headline (`-p 512,2048,8192 -n 128 -r 3`, four builds, 6 then 9 cards) and the ablations (combined build, six cards) |
| `run/q117_tgcheck.sh` | tg128 only (`-p 0 -n 128 -r 5`): combined with Q108 on, with Q108 off (`GGML_CUDA_MMQ_MOE_LEGACY=1`) and the v2 base, alternating, three rounds, six cards |

## What it shows

| Question | Result (tok/s) |
|---|---|
| Prompts, six cards (stock → combined) | pp512 193 → 250, pp2048 410 → 511, pp8192 507 → 617; v1 and the v2 base equal stock (193 / 410 / 507) |
| Prompts, nine cards | pp2048 493 → 613, pp8192 593 → 723 |
| tg128 after three pp runs in the same process, six / nine cards | stock 15.2 / 16.5, v1 31.2 / 32.6, v2 base 34.7 / 36.5, combined 33.5 / 32.9 |
| Ablations, six cards, tg128 | all on 33.40 and 33.49 (start, end); Q100 off 32.81; **Q108 off 36.80** (with prompts back to 193 / 411 / 508); Q109 off 34.70; Q110 off 34.52 |
| tg-only check (no prompt run before tg128) | combined with Q108 30.67 / 30.61 / 30.62, Q108 off 30.62 / 30.65 / 30.60, v2 base 29.86 / 29.86 / 29.86: no gap, the combined build's lower tg128 was an order effect |

The decode numbers of the tg-only check are lower than those after a prompt run: the cards start at base clock
(1,354–1,670 MHz just before each run) instead of warm. The release ships Q108 at every batch size.

## How it was made

On 2026-10-24 10:07–12:19, nothing else on the rig, the four builds unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)) and the model at
`/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_q117/run/q117.sh            # headline.md, ablations.md
RIG=rig OUT=. docs/data/2026-10-24_rig_q117/run/q117_tgcheck.sh    # tgcheck.md
```

The v2 builds run with `LLAMA_NO_MOE_DOWN_RESCALE=tg`. The llama-bench tables of the three logs were turned into
`llama_bench.csv` one row per test. Flint UD-Q4_K_XL is Unsloth's quant of Flint (Apache-2.0).
