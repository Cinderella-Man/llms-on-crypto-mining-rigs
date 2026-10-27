# Ornith-1.5-122B on nine P106 cards: agent check, memory projections, probes and power (2026-10-27)

The first rig run of Ornith-1.5-122B (registry row Q158): llama.cpp's own memory projection for 16 configurations,
single-request probes of eight server lines, the agent check on six lines (with AtomicChat's AD-IQ3_XXS-IQ2_S file,
stock llama.cpp and Miner Llama v2.0, each with and without the model's MTP head; with Unsloth's UD-Q2_K_XL, stock
and v2.0), GPU power and energy per task (AD lines), the two GGUF files' tensor layouts and v2.0's decode profile. The results are in [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md) and on
[the model page](../../../models/ornith-1.5-122b/README.md).

## More results

- **The harness rates every agent line "Overall: NO-GO", on host memory.** Its memory check asks for at least 600 MB
  available and no swap growth; the six lines kept 421–595 MB available at their lowest and swapped ~300 MB more
  during the run. Four lines also fail the proxy-row check, each on a task that ran into the 12-minute cap (the
  request in flight at the cap has no reply).

| Line | Lowest MemAvailable | Swap used | Proxy rows |
|---|---|---|---|
| stock | 507 MB | 349 → 653 MB | NO-GO (list-ops, cut at the cap) |
| stock + MTP | 482 MB | 369 → 670 MB | GO |
| v2.0 | 595 MB | 369 → 696 MB | NO-GO (list-ops, scale-generator) |
| v2.0 + MTP | 421 MB | 354 → 652 MB | NO-GO (scale-generator) |
| UD-Q2_K_XL, stock (`ud-stock`) | 564 MB | 356 → 659 MB | NO-GO (scale-generator) |
| UD-Q2_K_XL, v2.0 (`ud-v2`) | 495 MB | 359 → 660 MB | GO |

- **GPU energy (the nine cards, from `nvidia-smi` every 10 s):** 270–295 W while tasks run, ~90 W while the server
  loads. On the six tasks both passed, stock used 109.0 Wh and v2.0 80.6 Wh (1.35× less); per generated token on the
  tasks that ended on their own, stock 68.2 J, stock + MTP 56.9, v2.0 45.3, v2.0 + MTP 50.1. Card power only: the
  host, the PSU's losses and the fans are not in it.
- **Identity hashes** (`serve_probe.csv`): v2.0 gives the same chat output at `-ub 512` with the profiler on and at
  `-ub 128` with 122,880 tokens of context (`86dbbb9bb2b8`); stock's differs (`194b3c3239bb`), since its kernels round
  differently.
- **Load times:** 360–366 s for the AD file and 329–331 s for UD-Q2_K_XL through llama.cpp's memory-mapped loader.

## Files

| File | What |
|---|---|
| `agent/<line>/compare.md` | per task: result, minutes, turns, tool calls (all / failed / malformed); per line: tasks passed, minutes, minutes per task, time to first token per turn (median, p90), prompt and generation tok/s, tool calls, runaways; time to first token per turn number. The harness names the arm `stock` or `rel` (Miner Llama) whatever the line |
| `agent/<line>/table.md` | per line: attempts, pass@1, minutes per task (median, p90), mean seconds of prefill / decode / fixed cost / other, turns, prompt tokens processed per task, cache hit, TTFT p50, malformed calls, runaways |
| `agent/<line>/gonogo.md` | the harness's checks (infrastructure, proxy rows, first token against prefill, tool calls, cache, memory), the server log's counters and the overall verdict |
| `agent/<line>/validation.txt` | the 12 tasks validated in the sandbox image before the run (the reference solution passes all tests, the empty file fails some, the visible tests run) |
| `agent/<line>/attempts.csv` | one row per attempt (`bench/agent_check/report.py`'s input): result, tokens, turns, wall, prefill, decode and fixed seconds, cache, tool-call counts, test counts |
| `agent/<line>/turns.csv` | one row per request: time to first token, end time, new and cached prompt tokens, prompt ms, generated tokens, decode ms, finish reason, tool calls |
| `agent_power.csv` | the four AD lines, per line and task: minutes, samples, the nine cards' mean and peak summed power, the highest card, median SM clock of busy cards, energy (Wh), output tokens and joules per output token (left out for runaways, whose last reply never finished); rows for the server load and for all tasks |
| `fit_projection.csv` | per configuration (file, engine, context, micro-batch, `-ts`, MTP drafts): projected MiB per card, the smallest free MiB and its card, the total, host buffers, and whether every card keeps 150 MiB |
| `serve_probe.csv` | per server line: status, load seconds, MiB per card after loading, host MemAvailable after loading and after the requests, fresh prompt tok/s, next-turn first token, decode tok/s at 16k with MTP drafts and acceptances, and the two identity prompts' output hash (first 12 hex digits of MD5) and tok/s |
| `header-Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.txt`, `header-Ornith-1.5-122B-UD-Q2_K_XL.txt` | [scripts/gguf_summary.py](../../../scripts/gguf_summary.py) on each file: metadata, tensor bytes per role and type, per block |
| `decode_profile_v2.txt` | [scripts/prof_summary.py](../../../scripts/prof_summary.py) on v2.0's `GGML_CUDA_PROFILE` trace of the `v2-plain-prof` probe: ~591 one-token steps, GPU time per token per card, per first op, per weight type and per launch group |
| `run/agent_ornith.sh` | the agent-check lines (`stock`, `v2`, `stock-mtp`, `v2-mtp` with the AD file; `ud-stock`, `ud-v2` and `ud-v3`, the IQ2_S/IQ3_XXS kernels' build, with UD-Q2_K_XL), with the harness's machine values as variables |

`fit_projection.csv`: per card the main model's memory (weights, context, compute) plus, with the MTP head, the extra
model's context and compute (it shares the weights). Where llama.cpp printed its own "projected memory use" lines they
agree to the MiB and their free figures are used; with the MTP head the log kept only cards 0–2's lines
(`checked_against_llama_cpp`). The `ud-v2-fit-48k` row is the default split before `--fit`'s own search, which
overfills card 5.

## How it was made

On 2026-10-27, nine P106-090 cards, stock llama.cpp `d77256999` and Miner Llama v2.0 (`37e717ff6`) built for compute
6.1, the files on a USB 3 stick:

| File | Hugging Face repository, revision | Bytes | SHA-256 |
|---|---|---|---|
| `Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf` | [AtomicChat/Ornith-1.5-122B-GGUF](https://huggingface.co/AtomicChat/Ornith-1.5-122B-GGUF) `6c2d63ef5523` | 47,643,074,624 | `f6285e9a98a7cebcbafdbd4cc89d05f47999e7fce3cf9be74a32b7da6aace4ea` |
| `Ornith-1.5-122B-UD-Q2_K_XL.gguf` | [unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) `745bd742473a` | 44,533,952,768 | `93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66` |

From the repository root, with `RIG` the rig's ssh alias (default `rig`):

```bash
# the files, pinned and checked (size and SHA-256)
export DEST=<download folder>
python3 -I scripts/hf_manifest.py "$DEST/manifest.tsv" 'AtomicChat/Ornith-1.5-122B-GGUF=AD-IQ3_XXS-IQ2_S' \
  'unsloth/Ornith-1.5-122B-GGUF=UD-Q2_K_XL'
bash scripts/hf_get.sh
python3 -I scripts/gguf_summary.py Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf > header-Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.txt

# memory projections, e.g. the v2.0 line at 122,880 tokens (the 16 configurations: the rows of fit_projection.csv)
MODEL=<rig path>/Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf BUILDS=<rig builds folder> OUT=fit_probe.md \
  CONFIGS='v2-ub128-120k|<v2.0 build>|122880|-b 2048 -ub 128 -ts 5,6,5,6,5,5,6,5,7' bash scripts/fit_probe.sh

# single-request probes (args and env per line: serve_probe.csv); the profiled one
MODEL=... BUILDS=... OUT=serve_probe.md \
  CONFIGS='v2-plain-prof|<v2.0 build>|49152|-b 2048 -ub 512 -ts 5,6,5,6,5,5,6,5,7 -bs -fit off -lv 4|GGML_CUDA_PROFILE=/tmp/ornith-prof.csv GGML_CUDA_NORM_Q8_DEBUG=1' \
  bash scripts/serve_probe.sh

# the agent check, four lines (35-minute budget each)
CLIENT=<client> CLIENT_WORK=<folder on the client> AD=<rig path> STOCK=<stock llama-server> V2=<v2.0 llama-server> \
  OUT=agent CANDS="stock v2 stock-mtp v2-mtp" bash docs/data/2026-10-27_ornith/run/agent_ornith.sh

# the same with UD-Q2_K_XL, split by bytes
CLIENT=<client> CLIENT_WORK=<folder on the client> AD=<rig path> UD=<rig path of the UD file> STOCK=... V2=... \
  UD_STOCK_LINE="-b 2048 -ub 512 -ts 8,6,5,5,5,4,5,6,6 -fit off" UD_V2_LINE="-b 2048 -ub 512 -ts 8,6,5,5,5,4,5,6,6 -fit off" \
  OUT=agent CANDS="ud-stock ud-v2" bash docs/data/2026-10-27_ornith/run/agent_ornith.sh
```

`fit_projection.csv` and `serve_probe.csv` are those two scripts' logs turned into tables; `agent_power.csv` comes
from each line's `rig_server.sh` trace (`nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,
memory.used,utilization.gpu -l 10`) cut at each attempt's start and end. The server logs, request bodies and agent
transcripts are not published.

Ornith-1.5-122B is MIT-licensed by its makers; the headers list the metadata of AtomicChat's and Unsloth's files.
The probes' prompts are CPython standard-library source (PSF); the agent tasks are Exercism exercises (MIT). Tables
and scripts of this repository (MIT).
