# Q218 on the rig: the retyped Swift Q2_0 on the agent check (2026-11-03)

The first speed step of the Flash-Next ladder ([Q551](../../../quants/experiments.md)): Swift-1.5 GSQ-RCO Q2_0 with
its BF16 hyper-connection projections rewritten to Q8_0 and its F32 router to F16
([Q218](../2026-11-03_retype/README.md), `scripts/gguf_retype.py`), served on the rig's nine P106 cards with Q215's v3
line, against the unmodified file on the same line. Q215's profile put 43 of the 136 ms of GPU time per decoded token
in those BF16 products ([decode_profile_v3.txt](../2026-11-03_flash_next_baseline/decode_profile_v3.txt)). This folder
measures what the rewrite removes from that profile and what it changes in a coding agent's sessions.

## Result: generation 15 % faster, sessions 1.10× at equal work, 12 / 12 passed

The [agent check](../../../bench/agent_check/README.md): the pi coding agent fixes 12 small bugs (six Python, six
JavaScript), one attempt per task, at most 12 minutes each, no attempt started after an arm's 40 minutes. The client
ran pi, the proxy and the tasks' tests in containers on another machine, its requests relayed to the rig's loopback.
The two reference arms are the unmodified file on the same line and build: Q215's v3 arm (`rel`) and Q551 step 1's
arm without pipelining (`pipe0`, the pipelining branch with pipelining off, which matched Q215's arm).

| Arm | Tasks done (passed) | Minutes, all tasks | Minutes per task, median | First token per turn, median (p90) | Prompt tok/s | Generation tok/s | Tool calls (malformed) | Runaways |
|---|---|---|---|---|---|---|---|---|
| unmodified, Q215's v3 arm (`rel`) | 9 (9) | 44.2 | 3.0 | 7.7 s (19.8) | 61 | 7.8 | 47 (0) | 1 |
| unmodified, Q551 step 1 (`pipe0`) | 8 (7) | 42.3 | 3.2 | 7.6 s (18.2) | 61 | 7.9 | 36 (0) | 1 |
| **retyped (`hcq8`)** | **12 (12)** | 42.6 | 3.7 | 8.3 s (18.2) | 64 | **9.0** | 57 (0) | 0 |

On the 8 tasks every arm ran ([shared_tasks.md](agent/shared_tasks.md)):

| Arm | Passed | Minutes | Per task against `rel` (geomean) | Generated tokens | Gen tok/s | New prompt tokens | Prompt tok/s | `rel`'s work at this arm's rates |
|---|---|---|---|---|---|---|---|---|
| `rel` | 8 / 8 | 38.6 | 1.00× | 12,153 | 7.85 | 25,135 | 59.6 | 38.6 min, 1.00× |
| `pipe0` | 7 / 8 | 42.3 | 0.89× | 10,142 | 7.88 | 23,543 | 60.8 | 38.3 min, 1.01× |
| `hcq8` | 8 / 8 | 25.6 | 1.35× | 9,652 | **9.02** | 22,682 | 60.7 | **35.1 min, 1.10×** |

- **Generation is 15 % faster:** 16,685 tokens in 1,856 s of decoding over the session's 62 requests (8.99 tok/s,
  from the server's timings), against 7.8 and 7.9; on the shared tasks 9.02 against 7.85 and 7.88 tok/s. The
  profile's 11.2 % less GPU time per token predicts +12.6 %.
- **Prompts run as before:** 60.7 against 59.6 and 60.8 tok/s on the shared tasks (the 64 over all tasks includes
  the four the references did not reach). Prompt processing at `-ub 512` runs matrix-matrix kernels, on which the
  retyped tensors' format made no measurable difference.
- **The sessions are 1.10× faster at equal work, the retype's own gain.** The raw 1.35× per task (25.6 against 38.6
  minutes; 1.52× against `pipe0`) mixes in sampling: at temperature 1.0 this arm generated 21 % fewer tokens on the
  same tasks, and neither reference's runaway recurred. Putting `rel`'s new prompt tokens and generated tokens to this
  arm's rates, with `rel`'s own tool and fixed time, gives 35.1 against 38.6 minutes (1.10×; 1.09× without the tasks
  any arm ran away on).
- **12 / 12 passed,** all twelve tasks inside the 40 minutes (the references reached 9 and 8); no runaways, no
  malformed tool calls, a median cache hit of 96.9 % of the prompt from the second turn on. Twelve tasks cannot show a quality
  difference of the size Q218's KLD measured (1.8–2.1× a change of summation order), but they show no loss.
- **The fixed cost per request was 0.10 s, against 1.29 s (`rel`) and 0.62 s (`pipe0`),** and host memory had more
  room: MemAvailable at least 231 MiB, MemAvailable minus CmaFree 236 MiB on average (lowest 120), swap at most
  305 MiB, memory pressure (PSI full avg10) at most 4.4 % ([host_memory.csv](host_memory.csv)), against `pipe0`'s 227,
  205, 845 MiB and 13 %. This arm ran headless and the references with the desktop; the retype frees memory on the
  cards only, so the host's extra room is the desktop's (~280 MB, Q83). The fixed cost is under a minute per session
  and is not in the equal-work figure. `mem_guard.sh` never fired.
- The report's NO-GO flag ([gonogo.md](agent/gonogo.md)) is memory (lowest MemAvailable 231 MB, below the harness's
  limit), as for both references.

**Where it leaves the release line:** the retype is a measured 15 % on generation and 1.10× on sessions, for a KLD of
1.8–2.1× a change of summation order. Q546's exact BF16 kernel aims at the same 43 ms (est. −29 to −32 ms per
token against the retype's −15.3) without that cost; its P106 timing (R16) decides which one the line takes, and a
Pascal kernel for short Q8_0 rows (Q604) would close the rest of the gap for the retyped files.

## Where a decoded token's time goes now

`GGML_CUDA_PROFILE` on the same three requests as Q215's diagnostic line (a 65-token prompt with 64 tokens decoded,
an ~8k prompt, and a follow-up that diverges inside it; ~96 decode steps), summarized by
[scripts/prof_summary.py](../../../scripts/prof_summary.py) ([decode_profile_hcq8.txt](decode_profile_hcq8.txt)):

| GPU time per decoded token, summed over the cards (ms) | Unmodified (Q215) | Retyped | Change |
|---|---:|---:|---:|
| `hc_attn_up`, `hc_ffn_up` (320 × 10,240) | 32.11 (BF16) | 18.61 (Q8_0) | −13.50 |
| `hc_attn_down`, `hc_ffn_down` (10,240 × 320) | 7.00 (BF16) | 4.00 (Q8_0) | −3.00 |
| `hc_attn_inject`, `hc_ffn_inject` (10,240 × 4) | 1.34 (BF16) | 1.90 (Q8_0) | +0.56 |
| `ffn_gate_inp` (2,560 × 512), `ffn_gate_inp_shexp` (2,560 × 1) | 3.14 (F32) | 3.02 (F16) | −0.12 |
| every other launch group | 92.37 | 93.17 | +0.80 |
| **all** | **135.96** | **120.70** | **−15.26 (−11.2 %)** |

- **The rewrite removes 15.3 ms of the 136, not the 25–35 ms estimated.** At 120.7 ms the cards run back to back at
  8.3 tok/s, against 7.4 before. The decode is GPU-bound on this line, so the profile's 11 % is the ceiling of what
  generation speed can gain from it.
- **The 320-wide products still run far below their bytes.** In Q8_0 the two up projections read 334 MB per token in
  18.6 ms (~18 GB/s, the BF16 version's ~20 GB/s): their time fell with their bytes, so the kernel's cost per byte
  did not change. The 10,240-wide down projections read 167 MB in 4.0 ms (~42 GB/s). A row of 320 weights is ten Q8_0
  blocks, too short for llama.cpp's matrix-vector kernel to keep a P106's memory busy; a Pascal kernel for short Q8_0
  rows is Q604, the BF16 equivalent Q546.
- **The injects got slower** (+0.56 ms): 10,240 × 4 products are launch cost, and Q8_0 adds a quantization of the
  input vector to each. Q557 fuses them into the down projections.
- **The router's F16 changes nothing in time** (−0.12 ms), as it changed nothing in KLD.

The profiled requests' own speeds are not results (the kernel profiler and `-lv 5`'s logging slow them down). Loaded on the same
split as before, the cards hold 4,489–5,243 MiB each (the fullest, card 6, 5,243 MiB).

## Files

| File | What |
|---|---|
| `decode_profile_hcq8.txt` | `prof_summary.py` on the retyped file's `GGML_CUDA_PROFILE` trace: GPU time per token per card, per first op, per weight type and per launch group |
| `diag_requests.txt` | the profiled line's load, memory per card and the three requests' timings, beside Q215's diagnostic line |
| `agent/compare.md` | the report over the three arms: per task result, minutes, turns and tool calls; per arm the summary; first token per turn |
| `agent/shared_tasks.md` | the 8 tasks every arm ran (and the 6 without a runaway): tokens, seconds and rates per arm, and the reference's work at each arm's rates |
| `agent/table.md`, `agent/gonogo.md` | the harness's per-arm table (prefill, decode, fixed and other seconds per task) and this arm's go/no-go checks |
| `agent/attempts.csv`, `agent/turns.csv` | per attempt and per turn of the three arms: tokens, cache, times, tool calls (bench/agent_check's columns) |
| `agent/validation.txt` | the 12 tasks validated in the sandbox image before the run |
| `requests.csv` | this arm's requests from the rig's server log: prompt tokens and ms, generated tokens and ms, ms per token, tokens sampled with the tool grammar active |
| `host_memory.csv` | this arm's host memory while the server ran, from `mem_guard.sh`'s log: MemAvailable minus CmaFree (mean, lowest), MemAvailable (lowest), swap (highest), the most swapped out in 10 s, memory pressure |
| `run/compare_refs.sh` | one report over arms from separate agent checks (here the two reference arms and this one) |
| `run/shared_tasks.py` | `agent/shared_tasks.md` from the combined `attempts.csv` |

## How it was made

On 2026-11-03, 20:01–21:05, on the first rig (nine P106-090 6 GB cards on PCIe 1.0 x1, a two-core Celeron, 4 GB of
host RAM), nothing else running on it (a quiet check before the arm, logged; an idle `tail` left from watching the
profiled line used no measurable CPU), with the desktop stopped
(`multi-user.target`). The two reference arms ran on the same rig with the desktop running; Q83 measured no speed
change from stopping it (it frees ~280 MB of host RAM and 5 MiB per card), so that difference is a caveat here, not a
cause. The engine is Miner Llama v3's base `37e717ff6` (the rig tarball `llama.cpp-20261102-37e717ff6-pascal-p106-v3`,
built for compute 6.1 in `nvidia/cuda:12.9.2-devel-ubuntu24.04`), Q215's line unchanged: `-ngl 99 -fa on -c 65536 -np 1
-bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio` (F16 cache).

The model is Swift-1.5 GSQ-RCO Q2_0 from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (Swift Open License 1.0 and qwen-community-1.0), rewritten by
[gguf_retype.py](../../../scripts/gguf_retype.py) as [Q218's folder](../2026-11-03_retype/README.md) describes:
`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-hcq8-0000{1,2,3}-of-00003.gguf`, md5 `d8118668cbb7f8e24e7ae28281c6ccb7` and
`9157bcfa939a7a1a52c7762b381f90ae` for shards 1–2 (checked on the rig after the copy), shard 3 the unchanged n-gram
table (`98ce140e641bc5d68af9f31df0f1c048`) on the rig's internal SSD under the new name. The copy ran over the LAN with
`nc` at ~109 MB/s (37.0 GB in 339 s).

From the repository root:

```bash
# 1. the profiled line (Q215's script, the retyped first shard), detached on the rig, then the summary
scp docs/data/2026-11-03_flash_next_baseline/run/fn_{line.sh,requests.py} $RIG:<work>/
ssh $RIG "cd <work> && LABEL=hcq8diag BUILD=<v3 build> BUILDS=<builds> MODEL=<retyped shard 1> CTX=65536 \
  ARGS='-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio -lv 5' ENVS=GGML_CUDA_PROFILE=<work>/prof.csv \
  MODE=diag OUT=<work> setsid -f nohup bash fn_line.sh > line-hcq8diag.out 2>&1 < /dev/null"
python3 -I scripts/prof_summary.py prof-hcq8diag.csv > decode_profile_hcq8.txt    # on the client

# 2. the agent check: the machine values in a private env file (RIG, CLIENT, ports, CLIENT_DIR, CLIENT_OUT,
#    RIG_LOGDIR, REL_BIN, MODEL=<retyped shard 1>, CTX=65536, TIME_CAP_MIN=12, BUDGET_MIN_hcq8=40, CPU_SHARES=1024,
#    MIN_AVAIL_MB=150, BENCH=bench/agent_check, GUARD_LOG=/dev/shm/...) and the arm's line:
#      ARGS_hcq8="-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio"
set -a; . <env>; set +a
STEPS=prep bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>
ARMS=hcq8 STEPS=arms bash docs/data/2026-11-03_flash_next_pipelined_rig/run/agent_check.sh <out>
ARMS=hcq8 STEPS=report bash docs/data/2026-11-03_flash_next_pipelined_rig/run/agent_check.sh <out>

# 3. on the client: one report over Q215's v3 arm, Q551 step 1's arm without pipelining and this arm, and the
#    per-request and host-memory tables from the rig's server log and mem_guard's log
BENCH=<client dir>/agent_check bash docs/data/2026-11-03_retype_agent/run/compare_refs.sh <cmp> \
  rel=<Q215's results> pipe0=<Q551 step 1's results> hcq8=<this check's results>
python3 -I docs/data/2026-11-03_flash_next_pipelined_rig/run/rig_tables.py <out>/rig <guard logs> <tables> hcq8
python3 -I docs/data/2026-11-03_retype_agent/run/shared_tasks.py <cmp>/res/attempts.csv rel pipe0 hcq8 \
  > agent/shared_tasks.md
```

`agent_check.sh` runs a quiet check of the rig and starts `mem_guard.sh`
([2026-11-03_flash_next_mtp/run/](../2026-11-03_flash_next_mtp/run/)) before the arm, then the arm through
`release_check_s12.sh`. The tasks are bench/agent_check's bugfix set; the client image `agent-check:0.87.1`.
