# Q125: output heads and the screened head on the combined build (2026-10-24)

The head comparison that picked the release file: Flint UD-Q4_K_XL with and without the screened output head (Q92),
K2h-r16 (GPTQ Q5_0 head, no screen), K2-r16 without and with the screen (K2sr), all on the combined v2 build
`b3052a4f8`, six then nine cards, llama-bench pp512 then tg128 in one process (warm clocks, Q141). Before the timings,
the screen's exactness on the P106s with `llama-screen-check`. Result: registry row Q125 in
[quants/experiments.md](../../../quants/experiments.md); the screen itself is in [quants/gptq_head](../../../quants/gptq_head/README.md).
Single-request diagnostics, not the release's agentic numbers.

| File | What |
|---|---|
| `llama_bench.csv` | 20 rows: arm, cards, start time, the file's type, size (GiB) and parameters as llama-bench reports them, test (pp512, tg128), tok/s, standard deviation |
| `run/q125.sh` | the driver: the screen check on UD and on K2-r16, then the ten llama-bench runs |

## What it shows

| Arm | tg128, six cards | tg128, nine cards |
|---|---|---|
| UD-Q4_K_XL | 35.92 ± 0.81 | 37.48 ± 0.10 |
| UD-Q4_K_XL + screen | 36.45 ± 0.66 | 38.33 ± 0.10 |
| K2h-r16 | **37.13 ± 0.99** | **39.31 ± 0.39** |
| K2-r16 | 36.03 ± 0.90 | 38.49 ± 0.24 |
| K2sr (K2-r16 + screen) | 36.78 ± 0.90 | 39.28 ± 0.24 |

pp512 is the same for every arm (249–252 tok/s on six cards, 228–231 on nine). The screen check passed on both files:
two prompts, 64 steps each, greedy and sampled output identical with and without the screen, 32,768 candidate logits
compared, none different. K2h-r16 is as fast as K2sr without a side file and became the release's Flint-Pascal-Q5.

## How it was made

On 2026-10-24 13:14–14:08, nothing else on the rig, the combined build unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)), the three model files and the screen
side file `flint-output-screen-gptq-q4_0.gguf` ([quants/gptq_head](../../../quants/gptq_head/README.md)) on the model
sticks:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_q125/run/q125.sh    # q125.md
```

Every run sets `LLAMA_NO_MOE_DOWN_RESCALE=tg LLAMA_SCREEN_HEAD_FILE=<screen file>`; `LLAMA_SCREEN_HEAD=1` turns the screen
on. The llama-bench rows of `q125.md` were turned into `llama_bench.csv`. Recipes K2h-r16 and K2-r16:
[quants/experiments.md](../../../quants/experiments.md) (Q113, Q126); UD-Q4_K_XL is Unsloth's. Flint is Apache-2.0.
