# E0.9: the full tg64 grid of the small-MoE launch-overhead probe (2026-10-22)

Five small MoE models and a Llama-2-7B control generated 64 tokens on one P106-090 with 1 or 2 CPU threads, flash
attention off or on, at an empty context and at 4,096 tokens (1,024 for Llama-2-7B), and then with one or both CPU
cores kept busy. The write-up, with graph node counts and the flash-attention-on table, is
[2026-10-22-small-moe-overhead.md](../../benchmarks/2026-10-22-small-moe-overhead.md). This folder holds every
llama-bench row, including the flash-attention-off and 1-thread rows the page only summarizes.

| File | What |
|---|---|
| `tg64_grid.csv` | 60 rows: model, llama-bench model type and file size in bytes, CPU load (`idle`, `busy1` = a busy loop on core 1, `busy2` = on both cores), busy cores, threads, flash attention, depth, tokens generated, tok/s ± standard deviation, repetitions, the three per-repetition tok/s, llama.cpp build, GPU |
| `run/tg64_grid.py` | builds the CSV from the probe's JSONL files |

What the extra rows add:

- **Flash attention matters at depth, not at an empty context.** At 4,096 tokens, turning it on takes granite-3.1-3b
  from 24.71 to 35.51 tok/s, OLMoE from 29.91 to 46.44 and Ling-3.0-tiny from 18.05 to 27.57; LFM2.5 gains 16 %,
  granite-4.0-h-tiny (4 attention layers) 5 %, Llama-2-7B at 1,024 tokens 30 %. At an empty context it adds
  0.2–3.6 tok/s.
- **Threads make no difference** once the run is warm: 1 and 2 threads agree within 0.1 tok/s in every row except the
  first test of each run (1 thread, flash attention off, empty context), which is slower and noisier for every model
  (e.g. LFM2.5 35.56 ± 2.59 against 38.63 ± 0.02 with 2 threads, OLMoE 47.6 ± 3.04 against 50.46). That is the
  run's warm-up, not the thread count.

Made on 2026-10-22 on the rig with [scripts/launch_overhead_probe.sh](../../../scripts/launch_overhead_probe.sh) and
the rig's llama.cpp build at that time (upstream `85d5703a3`), GPU 0, models from `/mnt/models`:

```bash
RIG=${RIG:-rig}
M=/mnt/models
MOES="$M/bartowski/granite-3.1-3b-a800m-instruct-GGUF/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf
      $M/ibm-granite/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q4_K_M.gguf
      $M/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf
      $M/bartowski/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf
      $M/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf"
ssh "$RIG" "MODELS=$M/TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_0.gguf DEPTHS=0,1024 GPU=0 bash -s" < scripts/launch_overhead_probe.sh
ssh "$RIG" "MODELS='$MOES' GPU=0 bash -s" < scripts/launch_overhead_probe.sh
python3 run/tg64_grid.py /tmp/launch_probe > tg64_grid.csv     # on a copy of the rig's /tmp/launch_probe
```

The models' Hugging Face repositories are linked on the result page. The verbose logs and the `nvidia-smi` trace are not
published; the node counts taken from the verbose logs are on the result page.
