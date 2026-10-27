# Q67 and Q66: tool requests sampled on the device, checked against the host (2026-10-24)

The correctness runs of Q67 (sample on the device, check the sampled token against the tool-call grammar on the host,
fetch the full row only on a rejection) and the first count for Q66 (how many tokens are sampled while the grammar
applies, and how often it rejects the sampled token). A CPU-only llama-server build of the fork (the backend sampler on
the CPU backend), BFCL v4 tasks through llama-server's chat endpoint with the tools and `tool_choice` auto. Every request
runs once per variant: `device` (backend sampling on: the Q67 path), `host` (backend sampling off), and `device-none` /
`host-none` (the same prompt without tools, so without a grammar). Result:
[tool_request_sampling.md](../../research/notes/tool_request_sampling.md) section 5; the GPU and rig measurements are in
[2026-10-25_release_rig](../2026-10-25_release_rig/README.md) (`tool_request_q67.tsv`).

| File | What |
|---|---|
| `flint-runs.jsonl` | Flint UD-Q4_K_XL, 8 BFCL tasks (one per category) × 4 variants × T 0 and T 1: 64 requests. Per request: task, category, variant, temperature, seed, finish reason, reasoning, content, tool calls, and the server's token counts (prompt, cached, generated, sampled on the device, sampled while the grammar applied, rejected by it) |
| `flint-forced-reject-runs.jsonl` | the same 8 tasks, `device` and `host`, T 0 and T 1, with the test switch `LLAMA_GRAMMAR_TEST_REJECT=3` (every third token checked against the grammar counts as rejected, forcing the resample path): 32 requests |
| `granite-runs.jsonl` | granite-3.1-3b-a800m-instruct, 40 BFCL tasks (five per category) × 4 variants × T 0 and T 1: 320 requests; its chat template gets no tool grammar here, so it checks the device path without one |
| `verdicts.txt` | `grammar_sampling_check.py compare` and `rejects` for granite and Flint: identical (task, seed) pairs per variant pair and temperature, and per category the tokens sampled, sampled while the grammar applied and rejected |
| `verdicts-forced-reject.txt` | the same for the forced rejections |

## What it shows

| Question | Result |
|---|---|
| Device against host sampling, Flint, with the tool grammar | T 0: 8 of 8 (task, seed) pairs identical in reasoning, content and calls; T 1: 8 of 8 |
| The same without the grammar | 8 of 8 at T 0 and at T 1 |
| Forced rejections | 8 of 8 identical at T 0 and at T 1; 337 rejections over the 16 requests of each variant, every one resampled |
| granite, 40 tasks | 40 of 40 identical at T 0 and at T 1, with and without tools |
| Q66: how much of a tool request the grammar touches | 601 of 2,925 sampled tokens (20.5 %), from 13.9 % (simple) to 53.0 % (parallel_multiple); 4 of those 601 rejected (0.67 %) |
| The old switch-off warning ("backend sampling is not compatible with grammar") | 0 times |

## How it was made

On 2026-10-24 on a CPU-only build of the fork branch `pascal-p106-v2-toolsample` (commits `e2d587398` and the test
switch `f11ac1589`), `llama-server -m <model> -ngl 0 -t 8 -c 16384 -np 1 -bs --cache-ram 0 --ctx-checkpoints 0` on Flint
UD-Q4_K_XL and on granite, with
[bench/tool_sampling/grammar_sampling_check.py](../../../bench/tool_sampling/grammar_sampling_check.py) and the task
file of [bench/bfcl_flint/prepare.py](../../../bench/bfcl_flint/README.md):

```bash
python3 -I bench/tool_sampling/grammar_sampling_check.py run --url http://127.0.0.1:8080 --tasks tasks.jsonl \
    --out flint-runs.jsonl --per-category 1 --temps 0,1 --seeds 1 --max-tokens 1536 \
    --variants device:backend_sampling=true host:backend_sampling=false \
               device-none:backend_sampling=true\;tool_choice=none host-none:backend_sampling=false\;tool_choice=none
python3 -I bench/tool_sampling/grammar_sampling_check.py compare flint-runs.jsonl --a device --b host
python3 -I bench/tool_sampling/grammar_sampling_check.py compare flint-runs.jsonl --a device-none --b host-none
python3 -I bench/tool_sampling/grammar_sampling_check.py rejects flint-runs.jsonl --variant device
```

granite (granite-3.1-3b-a800m-instruct, Q4_K_M): the same with `--per-category 5 --max-tokens 1024`; the forced
rejections: a server started with `LLAMA_GRAMMAR_TEST_REJECT=3` and the `device` and `host` variants only. No speeds are
kept (`--keep-speed` not given).

The tasks are from the Berkeley Function-Calling Leaderboard (BFCL v4, Apache-2.0); the example addresses and names in
them, and the example network addresses in the answers (a generated proxy setting), are fictional. The
answers are Flint's (Apache-2.0) and granite's (Apache-2.0) outputs.
