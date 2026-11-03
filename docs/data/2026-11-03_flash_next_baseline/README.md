# Q215: Qwen3.8-Flash-Next's baseline on nine P106 cards, stock llama.cpp against Miner Llama v3 (2026-11-03)

The first runs of Flash-Next on the rig: Swift-1.5 GSQ-RCO Q2_0 (37.7 GB on the cards, the 28.8 GB n-gram table on the
internal SSD, [Q435](../2026-11-03_ssd_room/README.md)) with R11's settings ([Q213](../2026-11-03_flash_next_fit/README.md)),
on stock llama.cpp `d77256999` and on Miner Llama v3's base `37e717ff6` (the v2.0 commit, rebuilt on 2026-11-02).
First the requests a single user makes (prompt speed at 1k, 8k and 32k tokens, decode at each, the table's reads,
context checkpoints, memory), then the agent check: a coding agent fixing 12 small bugs on each engine. The registry
row is Q215 in [quants/experiments.md](../../../quants/experiments.md).

## Result: a coding agent passes 7 of 8 tasks on stock and 9 of 9 on v3, at 3.0–3.3 minutes per task

The [agent check](../../../bench/agent_check/README.md): the pi coding agent fixes 12 small bugs (six Python, six
JavaScript), one attempt per task, at most 12 minutes each, and no attempt starts after an arm's 40 minutes. Both
engines with the same settings as below; the client ran pi, the proxy and the tests in containers on another machine,
its requests relayed to the rig's loopback.

| Engine | Tasks done (passed) | Minutes, all tasks | Minutes per task, median | First token per turn, median (p90) | Prompt tok/s | Generation tok/s | Tool calls (malformed) | Runaways |
|---|---|---|---|---|---|---|---|---|
| stock llama.cpp `d77256999` | 8 (7) in 39.6 min | 39.6 | 3.3 | 9.2 s (23.2) | 53 | 6.4 | 33 (0) | 1 |
| **Miner Llama v3 `37e717ff6`** | **9 (9)** in 44.2 min | 44.2 | **3.0** | **7.7 s** (19.8) | **61** | **7.8** | 47 (0) | 1 |

- **v3 is 1.12× faster per task** (geometric mean over the 8 tasks both engines scored, range 0.75–1.55×; 38.6
  against 39.6 minutes in all, since one task ran to its time cap on each) and passed one more task. It generates
  22 % faster over the whole sessions (7.8 against 6.4 tok/s) and reads prompts 15 % faster.
- **Every tool call parsed:** 80 calls, none malformed or inside the reasoning.
- **The cache works for agent turns:** from the second turn on, the median request reused 96.6 % of its prompt
  (the turns append, so the checkpoint at the end of the previous prompt is the one restored), and a turn's first
  token came after 3–5 s from the fourth turn on. The first two turns, which carry the tools, the task and the first
  file, took 16–22 s.
- **The rig's host is the limit to watch:** MemAvailable fell to 228 MB and swap grew from 356 to 974 MB during the
  check (Q549). The harness's watchdog was set to 150 MiB instead of its default 400 for this run.
- **The report's three NO-GO flags** ([gonogo.md](agent/gonogo.md)) are the memory above, stock's runaway attempt (its
  last reply never finished, so the proxy has one row more than pi has replies), and the gap between the client's
  first token and the server's prefill (p90 1.88 s against a 0.8 s limit set for Flint): this model's fixed cost per
  request includes the 0.63 s checkpoint restore.
- One v3 attempt was re-run after the relay dropped its connection (infrastructure, not scored).
- Load on the client's machine: 38 on its 24 threads at the start (other jobs at a low CPU share; the client ran at
  normal priority), 4 at the end.

### One request at a time

Both engines, `-c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fa on -fit off -lm dio`, F16 cache, one slot, the
GGUF's sampling (temperature 1.0, top-p 0.95, top-k 20), requests sent on the rig itself:

| Request | Stock: prompt tok/s | first token | decode tok/s | v3: prompt tok/s | first token | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 65 tokens | 9.2 | 7.1 s | 6.56 | 10.5 | 6.5 s | 6.25 |
| 1,382 tokens | 59.4 | 23.3 s | 6.78 | 71.8 | 20.2 s | 5.43 |
| 7,183 tokens | 116.4 | 62.4 s | 7.15 | 139.3 | 52.2 s | 8.30 |
| 32,674 tokens | 128.0 | 256.7 s | 5.66 | 148.6 | 220.8 s | 6.24 |
| identity prompts (30 and 735 tokens, T 0) | 8.2, 44.2 | 3.7, 16.7 s | 6.73, 6.86 | 8.5, 52.4 | 3.6, 14.1 s | 8.45, 8.90 |

- **v3 reads prompts 15–21 % faster** (1k–32k); decode is the same within the noise of these single requests:
  5.4–8.9 tok/s on both, with no trend between them (each request generates a different text, so its expert and
  table reads differ). The two engines' identity outputs at T 0 differ after the first sentence (different kernels
  round differently): `dfc124c1e214`/`4b5233be357b` for stock, `c746031ccec1`/`5800edf22d04` for v3.
- **Short prompts are slow:** 65 tokens take 6.5–7.1 s to the first token. Every request first restores the
  context checkpoint after the chat template's 42-token prefix (0.63 s, below), and a micro-batch passes all nine
  cards in turn over x1 links.
- **Where a decoded token's time goes** (v3, `GGML_CUDA_PROFILE`, [decode_profile_v3.txt](decode_profile_v3.txt)): 136 ms
  of GPU time per token summed over the cards (7.4 tok/s if the cards ran back to back), so the measured decode is
  GPU-bound. The **hyper-connection projections in BF16 take ~41 ms (30 %)**: `hc_attn_up` and `hc_ffn_up`
  (320 × 10,240, 6.5 MB each) 16 ms apiece, ~0.33 ms a call, about 8× slower than their bytes should allow. The routed
  experts (Q2_0) take ~26 ms (19 %), the attention projections (Q3_K, IQ4_XS) ~15 ms, flash attention 3.5 ms, the
  recurrent state update 3.7 ms, the F32 router 2.5 ms and the output layer 3.0 ms.
- **The n-gram table on the SSD:** decode makes 11–21 reads of the SSD per token (46–119 KiB), close to the research
  note's estimate of 11–14 rows; prompt processing 8–12 reads per prompt token (32k: 270,000–290,000 reads,
  1.1–1.2 GB). With 136 ms of GPU time per token, the reads do not show in the decode speed; the rig's root file
  system is LUKS-encrypted, so each read is also decrypted on the Celeron.
- **Context checkpoints** ([checkpoints.csv](checkpoints.csv)): v3 saves a 112.6 MiB recurrent-state checkpoint in
  0.67–0.69 s and restores one in 0.63–0.64 s. It saves them after the 42-token template prefix and in the last
  ~512 tokens of each prompt (at 6,709 and 7,221 of 7,225), so a request that diverges further back than that
  re-processes the whole prompt: a 31k-token follow-up that changed its last 1,500 tokens took 209 s (v3) and 240 s
  (stock) to its first token. Stock logs no checkpoints at its default log level and re-processed the same way.
- **Memory:** the fullest card holds 5,325–5,333 MiB of 6,144 after the load (card 6; R11 projected 735 MiB free of
  the 6,001 llama.cpp sees). The host keeps 735 MiB of pinned buffers; MemAvailable fell to 230–460 MiB during the
  requests and 350–760 MB went to swap.
- **Load:** 320–326 s with direct I/O (`-lm dio`), the USB disk at ~127 MB/s. With the default memory map, llama.cpp
  first populates the mapping of the two shards that hold no lazy tensor (37.7 GB), which a 4 GB host cannot keep,
  and then reads them again for the upload: that load was stopped after 30 GB had been read and nothing had reached
  the cards.

## Files

| File | What |
|---|---|
| `single_request.csv` | per engine line and request: load seconds, MiB per card after the load, prompt and cached tokens, prompt tok/s, time to first token, generated tokens, decode tok/s, SSD reads and MiB while reading the prompt, SSD reads, KiB and major faults per generated token, MemAvailable after the request, the identity outputs' MD5 |
| `checkpoints.csv` | every context-checkpoint save and restore in the v3 and diagnostic servers' logs: tokens, MiB, ms |
| `decode_profile_v3.txt` | [scripts/prof_summary.py](../../../scripts/prof_summary.py) on v3's `GGML_CUDA_PROFILE` trace (the diagnostic line): GPU time per token per card, per first op, per weight type and per launch group |
| `run/fn_line.sh` | one server line on the rig, detached: load, memory, the requests, the checkpoint lines, the profile |
| `run/fn_requests.py` | the requests, streamed, with the page-fault and disk counters read before, at the first token and at the end |
| `run/single_table.py`, `run/checkpoint_table.sh` | the two tables from the lines' outputs and the server logs |
| `agent/compare.md` | the agent check per task (result, minutes, turns, tool calls all / failed / malformed, the time ratio) and per engine, and the time to first token per turn |
| `agent/table.md` | per engine: attempts, pass@1, minutes per task, mean seconds of prefill / decode / fixed cost / other, turns, prompt tokens processed per task, cache hit, malformed calls, runaways |
| `agent/gonogo.md` | the harness's checks (infrastructure, proxy rows, first token against prefill, tool calls, cache, memory) |
| `agent/attempts.csv`, `agent/turns.csv` | one row per attempt and per request (time to first token, new and cached prompt tokens, prompt ms, generated tokens, decode ms, finish reason, tool calls) |
| `agent/validation.txt` | the 12 tasks validated in the sandbox image before the run |
| `run/agent.env.example` | the agent check's settings for [release_check.sh](../../../bench/agent_check/release_check.sh), machine values left out |

## How it was made

On 2026-11-03, nine P106-090 cards, nothing else running on the rig. Stock llama.cpp `d77256999` (upstream build
11820) and Miner Llama v3's base `37e717ff6` (build 11911, the tarball rebuilt on 2026-11-02 that the second rig also
runs), both built for compute 6.1 in `nvidia/cuda:12.9.2-devel-ubuntu24.04`. The model: Swift-1.5 GSQ-RCO Q2_0 as
three shards (the table alone in the third, on the internal SSD; Q435), from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (Swift Open License 1.0 and qwen-community-1.0).

From the repository root, with `RIG` the rig's ssh alias:

```bash
# the single-request lines: the two scripts copied to a work folder on the rig, each line started detached
scp docs/data/2026-11-03_flash_next_baseline/run/fn_{line.sh,requests.py} $RIG:<work>/
M=<disk>/models/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf
A='-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio'
ssh $RIG "cd <work> && LABEL=stock BUILD=<stock build> BUILDS=<builds> MODEL=$M CTX=65536 ARGS='$A' MODE=speed \
  OUT=<work> setsid -f nohup bash fn_line.sh > line-stock.out 2>&1 < /dev/null"
#   then LABEL=v3 with the v3 build, and the diagnostic line:
#   LABEL=v3diag ARGS='$A -lv 5' ENVS=GGML_CUDA_PROFILE=<work>/prof.csv MODE=diag
python3 -I scripts/prof_summary.py prof-v3diag.csv > decode_profile_v3.txt
python3 -I docs/data/2026-11-03_flash_next_baseline/run/single_table.py single_request.csv line-*.out
bash docs/data/2026-11-03_flash_next_baseline/run/checkpoint_table.sh server-v3.log server-v3diag.log > checkpoints.csv
```

The agent check, after the client's self-test passed in its containers (`AC_SANDBOX=docker bench/agent_check/selftest.sh`):

```bash
set -a; . my-agent.env; set +a    # run/agent.env.example with the machine values filled in
STEPS=prep bench/agent_check/release_check.sh agent
STEPS="run report collect" bench/agent_check/release_check.sh agent
```

The client is pi 0.87.1 in the `agent-check:0.87.1` image ([docker/Dockerfile](../../../bench/agent_check/docker/Dockerfile));
the tasks are Exercism exercises (MIT) with one bug each ([the bugfix set](../../../bench/agent_check/README.md#the-bugfix-task-set)).

`fn_requests.py` sends its requests from the rig itself to 127.0.0.1, so no prompt crosses the network. Its prompts
are CPython standard-library source (PSF) read on the rig. The diagnostic line's speeds are not results: the kernel
profiler and trace logging slow it down.
