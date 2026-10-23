# Flint's decode cliff between 32k and 64k of context on nine P106 cards: what the code does at depth, ranked causes and the rig runs that separate them (2026-10-23)

Scope: experiment Q79 in [quants/experiments.md](../../../quants/experiments.md), raised by Q54. On nine cards, S4-L0
decoded 23.3 tok/s with 32k tokens in the cache and 3.6 tok/s with 64k. This note reads the code that changes with
context length (flash-attention dispatch for compute 6.1, the attention mask, CUDA graphs, buffers and memory, and how
llama-bench builds a depth), ranks the causes, and gives for each one the cheapest rig run that confirms or kills it and
what to change if it holds. The attention kernel itself is analysed in
[pascal_attention_long_context.md](pascal_attention_long_context.md) and not repeated here.

Conventions:
- "(est.)" marks this note's own arithmetic or inference.
- "Claim to test" marks anything not measured on the rig.

Code references: the fork's branch `pascal-p106-v2` at `032b130f4`. That is the measured build `eeaa59660` plus two
commits that touch only the Pascal matrix-vector kernels, a gate/up fusion and the `LLAMA_NO_MOE_DOWN_RESCALE` switch.
Every file cited below has the same content at `eeaa59660`, except `ggml-cuda.cu`, whose line numbers are those of
`032b130f4`. The flash-attention files are unchanged from upstream `d77256999`.

## The measurement

S4-L0 (32.5 GB), nine cards, fork `eeaa59660`, f16 KV, `LLAMA_NO_MOE_DOWN_RESCALE=1`,
`llama-bench -ngl 99 -fa on -ub 128 -p 0 -n 64 -r 1 -d 0,0,32768,65536`, one sample per depth:

| Depth | tok/s | ms per token |
|---|---|---|
| 0 (first test, warm-up effects) | 23.1 | 43.3 |
| 0 | 27.8 | 36.0 |
| 32,768 | 23.3 | 42.9 |
| 65,536 | **3.6** | **275** |

- **The slope from 0 to 32k predicts ~20 tok/s at 64k.** The first 32k tokens add 6.8 ms per token, and doubling the
  KV adds as much again: ~50 ms, 20 tok/s. If attention instead reads at the 57–87 GB/s of the attention note, the
  prediction is 17–19 tok/s. The 64 tokens took 17.6 s instead of 3.2–3.8 s, so **~14 s of extra time** (est.).
- **Prefill to 65,536 shows no cliff.** From llama-bench's test start times and the job's end time:
  - the 32k test (fill, state copy, 64 tokens) took 4 min 6 s;
  - the 64k test took 11 min 46 s;
  - so the fill from 32k to 64k took ~1.8× as long as the fill to 32k, in line with attention growing with depth. A
    6× slowdown above 32k would have made that part alone take over half an hour (est.).

  So the cliff hits the decode that follows the fill. It does not slow attention at 64k keys in general.
- **Qwen3.6-35B-A3B fell further than expected at the same point** (nine cards, stock `85d5703a3`, CUDA graphs off on
  Pascal, `-ub 512`, the same `-d 0,8192,32768,65536`;
  [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)):
  - 16.6 → 14.0 tok/s from 8k to 32k (60.2 → 71.4 ms per token), which predicts ~86 ms (11.6 tok/s) at 64k;
  - measured 8.1 tok/s (123.5 ms), i.e. ~2.4 s of extra time over 64 tokens (est.).

## Verdict

- **Most likely, llama-bench ran the 4 GB host out of RAM; attention isn't the cause.**
  - After filling a depth, llama-bench copies the whole sequence state into a `std::vector` in host memory, for reuse
    by later repetitions, and only then starts the timer (`llama-bench.cpp:2444-2484`). The buffer stays allocated for
    the rest of the run.
  - Flint's state is 28 KiB per context token: 0.91 GiB at 32k, 1.78 GiB at 64k. Moving from the 32k test to the
    64k test, `resize` briefly holds both buffers: 2.69 GiB of anonymous memory.
  - The rig has 3.75 GiB of RAM, of which 2.59 GiB is available at idle with the desktop running. The 2.69 GiB peak
    exceeds that before counting llama-bench's own CUDA runtime for nine devices and its pinned buffers.
  - The kernel has to push ~1 GiB out to swap or drop it from the page cache (est.), and the timed decode starts right
    after the copy. Refaulting a few hundred MiB from an encrypted swap file on SATA is ~14 s (est.).
  - Qwen3.6's smaller excess fits the same mechanism: its state is smaller (20 KiB per token: 1.25 GiB at 64k,
    1.9 GiB during the resize).
- **Nothing on the attention path changes between the two depths.**
  - Compute 6.1 runs the same TILE kernel, `<128,128,1,8>`, with 5 parallel blocks per KV head, at 33,024 and at
    65,792 keys.
  - The selector's only threshold on context length (8,192) sits in the Ada Lovelace branch.
  - The partial-result buffers (80 KiB) don't depend on the context.
  - The F16 mask is one unpadded row of n_kv × 2 B (128.5 KiB at 64k).
  - CUDA graphs see one unchanged graph for all 64 tokens.
  - Device memory never falls back to host memory: a failed allocation aborts, or fails at context creation.
- **Two weaker hypotheses remain.**
  - The 64k decode is the run's first point with more than 65,536 keys: the fill ended at exactly 65,536, and the
    decode pads n_kv to 65,792. No 16-bit index or n_kv-sized grid dimension was found on Flint's decode path.
  - GPU clocks may have dropped after the 11-minute fill. Sustained runs measured 46–57 °C against an 83 °C target,
    and the fill itself ran at a normal pace.
- **First rig run: E1.** It is the same command with host memory, swap and a per-token host/GPU timeline logged
  (~25 min). Then E2: llama-batched-bench at 32,768 and 65,536, which fills and decodes without any state copy
  (~23 min). If E2 decodes 64k at 17–20 tok/s and E1 shows swap-ins during the 64k decode, the cliff belongs to the
  benchmark and Q54's 64k point is void.
- **If it holds:**
  - don't keep llama-bench's depth state on this host (a fork switch, or one depth per run);
  - measure deep contexts with llama-batched-bench;
  - log available memory and swap in `bench.sh`;
  - keep `--cache-ram 0` in llama-server, whose host prompt cache copies the same state;
  - a second SO-DIMM (the board has a free slot) removes the problem in hardware.

## 1. What changes between the 32k and the 64k decode

| | 32k test | 64k test | Code |
|---|---|---|---|
| Context (`n_prompt + n_gen + n_depth`, padded to 256) | 33,024 | 65,792 | `llama-bench.cpp:1308`, `llama-context.cpp:387` |
| n_kv while decoding: used cells padded to 256, capped at the cache size | 33,024 | 65,792 | `llama-kv-cache.cpp:1285-1299` |
| Largest n_kv during the fill | 32,768 | 65,536 | same |
| Kernel at one token | TILE `<128,128,1,8>`, grid (1, 5, 4), then the 5-way combine | same | `fattn.cu:704-718`, `fattn-tile.cuh:1252-1253`, `fattn-common.cuh:1137-1224` |
| Full-attention KV read per token | 0.95 GB | 1.89 GB | 28 KiB × n_kv |
| Mask per token and card (F16, one row) | 64.5 KiB | 128.5 KiB | `llama-graph.cpp:30-48` |
| llama-bench's depth-state buffer in host RAM | 0.91 GiB | 1.78 GiB, **2.69 GiB during the resize** | `llama-bench.cpp:2472-2476` |

### How llama-bench builds a depth

For each test (`llama-bench.cpp:2323-2532`):
1. Load the model only if it changed, then create a new context of `n_prompt + n_gen + n_depth` tokens.
2. Warm-up: one decoded token on the empty cache (`:2415-2442`). The decode graph at the test's real depth is
   therefore first built inside the timed part, together with its CUDA graph warm-up and capture, at every depth.
3. For each repetition (`:2444`), clear the cache, then either restore the cached state, if the previous test had
   the same depth (`:2448-2456`), or fill:
   - calls of 2,048 random tokens (32 at 64k), each split into 16 micro-batches of 128;
   - one output row per call;
   - then `llama_synchronize`.
4. Save the state: `cstate.buf.resize(llama_state_seq_get_size(ctx, 0))`, then `llama_state_seq_get_data` into it
   (`:2472-2476`). The `cstate` buffer is declared once, outside the test loop (`:2317`), so it lives until the
   program ends.
5. Start the timer (`:2484`), and decode 64 tokens one at a time, each followed by `llama_synchronize` (`:2198-2222`).

After the fill:
- Cells 0–65,535 are filled contiguously.
- The slot search continues at the head (`llama-kv-cache.cpp:1024-1056`).
- This tree has no defragmentation, and Flint's full-attention layers use no RoPE, so there is no K-shift.
- The cache's last 256 cells (65,536–65,791) hold the 64 new tokens; the rest stay masked with −inf.

Nothing is left in a state that slows the decode, except the host buffer.

Without `-v`, llama-bench sends every llama.cpp and ggml log line through a callback that keeps only errors
(`llama-bench.cpp:2224-2230`, `:2286`; `llama_log_set` also sets ggml's logger, `llama-impl.cpp:54-58`). The CUDA pool's
flush-and-retry message, the CUDA graph debug lines and every warning are invisible, so "no warning in the log" says
nothing either way.

## 2. Ranked hypotheses

### H1. llama-bench's depth-state copy pushes the host into swap before the timed decode (most likely; est. ~80 %)

**Mechanism in the code**
- The state holds every K and V row of the sequence (`llama-kv-cache.cpp:2076-2144`):
  - the 14 full-attention layers × 65,536 cells × 2 KiB = 1,792 MiB;
  - plus the ~512 live cells of the 35 sliding-window layers, ~35 MiB.

  That is 1.78 GiB at 64k and 0.91 GiB at 32k (est.).
- `std::vector::resize` from 0.91 to 1.78 GiB proceeds in three steps:
  1. allocate the new block;
  2. zero-fill the new part and move the old contents in (both commit pages);
  3. free the old block.

  Peak: 2.69 GiB of anonymous memory (est.).
- The copy itself is one `ggml_backend_tensor_get` per layer into pageable memory, queued and run when the writer is
  destroyed (`llama-context.cpp:2919-2970`, `:3431-3448`). At ~213 MB/s GPU→host that takes ~9 s (est.), outside the
  timed part.

**The host** (rig inventory, `scripts/rig_inventory.sh`, 2026-10-22)
- 3,836 MiB of RAM; at idle with the desktop, 1,186 MiB used and 2,650 MiB available.
- A 3.7 GiB swap file on the encrypted SSD.

**Host memory llama-bench itself holds**
- The pinned CPU compute buffer, which holds the inputs, among them the attention mask (`llama-context.cpp:508-514`).
  With pipeline parallelism every input has 4 copies (`ggml-backend.cpp:1973`): 4 × 65,792 × 128 × 2 B = 64 MiB plus
  ~11 MiB at 64k (est.). At 262,144 tokens the same buffer measured 267 MiB ([Flint benchmark](../../benchmarks/2026-10-22-flint.md)).
- The CUDA runtime for nine devices (not measured).
- The mapped embedding rows (one random row per token).

**What it predicts**
- **At 32k** the 0.91 GiB buffer fits, and the decode matches the slope.
- **At 64k** ~0.5–1.2 GiB has to leave RAM while the state is copied (est.). The kernel drops clean page cache (shared
  libraries, the CUDA driver and ggml-cuda code among them) and swaps out anonymous pages, including pages the decode
  loop touches every token: graph metadata and the CUDA driver's host structures.
- **During the timed decode**, these pages fault back in, while kswapd and dm-crypt compete with the decode loop for
  the two cores. The stall falls on the first tokens; if the working set no longer fits, it falls on every token.
- **The ~14 s of extra time** matches ~300–600 MiB of 4-KiB swap-ins at 20–40 MB/s (est.).

**Evidence**
- **For:**
  - The arithmetic above: the 64k test is the only one whose buffer, during the resize, exceeds the memory available
    at idle.
  - The prefill to 65,536 ran at a normal pace.
  - Qwen3.6 shows a smaller excess at a smaller state size, on a build without CUDA graphs.
- **Against, or open:**
  - The host footprint of the CUDA runtime for nine devices was never measured.
  - One sample, so it is unknown whether the time went to a few stalled tokens or to all 64.

**Experiments:** E1 (mechanism), E2 (control without any copy), E4 (the same mechanism at 32k, optional).

### H2. Something breaks above 65,536 keys (est. ~10 %)

- **Why it is a candidate.** The 64k decode is the first point in the run where n_kv exceeds 2^16:
  - n_ctx 65,600 is padded to 65,792 (`llama-context.cpp:387`), and the decode pads n_kv to 65,792
    (`llama-kv-cache.cpp:1290-1295`);
  - the fill's last micro-batch had exactly 65,536.

  Any 16-bit index, or grid y/z dimension sized by n_kv, would act only in the timed decode. No decode beyond 65,536
  tokens has been measured on this rig before; Q58's chats at a 262,144-token context stayed far below 65,536 tokens.
- **Code read: no such path on Flint's decode graph.**
  - Only `FLASH_ATTN_EXT` reads the K/V views and the mask. f16 K/V need no conversion, and the mask scan kernel runs
    only for ≥ 1,024 queries (`fattn-common.cuh:1117`).
  - The kernel takes n_kv and the strides as 32-bit integers (`fattn-common.cuh:20-39`, `:1251-1266`). The largest
    offset at 65,792 keys is 1,024 B × 65,792 = 64 MiB, and the mask stride is 131,584 B.
  - The decode grid is (1, parallel blocks, 4 KV heads) (`fattn-common.cuh:1217-1219`): n_kv sets no grid dimension.
  - The 65,535 caps in `convert.cu`, `getrows.cu`, `cpy.cu`, `rope.cu` and `norm.cu` sit on ops that don't see n_kv in
    this graph.
- **Qwen3.6 crossed 2^16 at its 64k point too,** so its data doesn't separate H2 from H1.
- **Experiments:** E2 kills it if 65,536 decodes normally without the state copy. If it doesn't, E3 compares 65,000
  with 65,536 keys and finds the op.

### H3. GPU clocks drop after the long fill (est. ~5 %)

- **Why it is a candidate.** The 64k fill is the run's longest uninterrupted GPU load (~11 min, against ~4 min for 32k).
  In a layer split every token waits for the slowest card, so one card in a power brake or HW slowdown slows the whole
  token.
- **Against:**
  - Sustained inference measured 46–57 °C per card, against an 83 °C target and 98 °C slowdown
    ([hardware spec](../../mining_rig.md)).
  - The fill above 32k ran at a normal pace.
  - The 32k decode, right after its own 4-minute fill, was normal.
- **Experiment:** the `nvidia-smi` log in E1 (SM clock and `clocks_event_reasons.active` per card during the 64k
  decode).

### Ruled out by the code, or too small to matter

- **Host work per token that grows with n_kv:** together ≤ ~4 ms per token at 64k (est.), against ~225 ms observed.
  E1's host timeline measures it.
  - The mask fill visits every cell (`llama-kv-cache.cpp:1650-1720`): ~0.3–0.7 ms at 64k (est.).
  - The mask is a user input, so the scheduler copies it synchronously to each card before that card's split
    (`ggml-backend.cpp:1829-1837`). That is 128.5 KiB per card, ~0.7 ms each over PCIe 1.0 x1. The copies run while
    the previous card computes, so mostly only the first is on the critical path.
- **CUDA graphs:**
  - All 64 tokens have the same n_kv, so llama.cpp reuses its graph (`llama-graph.cpp:49-66`, `can_reuse_kq_mask`).
    The CUDA backend then sees the same graph id and returns before comparing nodes (`ggml-cuda.cu:2764-2776`).
  - The first token runs eagerly, the second is captured, and the rest replay the capture (`ggml-cuda.cu:5043-5080`),
    at both depths.
  - The legacy pool's flush-and-retry, which drops captured graphs (`ggml-cuda.cu:553-566`), isn't in use: the cards
    report VMM, so the VMM pool serves allocations (`ggml-cuda.cu:595-742`, `:744-753`).
  - Qwen3.6's excess appeared on a build with CUDA graphs off on Pascal.
  - Control: `GGML_CUDA_DISABLE_GRAPHS=1` in E3.
- **GPU memory and fallbacks:**
  - Device buffers come from `cudaMalloc` or `cuMemCreate`. A failure aborts (VMM pool, `ggml-cuda.cu:648`) or fails
    context creation (KV cache, compute buffers), as the 6-card logs of Q54 show. Managed memory is used only with
    `GGML_CUDA_ENABLE_UNIFIED_MEMORY` (`ggml-cuda.cu:159`, `:188-216`).
  - On nine cards, S4-L0 loads with a 262,144-token context (Q58), four times the 64k KV. At 64k each card therefore
    had at least the difference free: ~0.55 GiB on a card with one full-attention layer, ~0.95 GiB with two (KV plus
    mask copies; est.). Card 3's shortage on six cards doesn't carry over.
- **Flash-attention heuristics:**
  - With 2 blocks per SM allowed (`fattn-common.cuh:1137`) the wave holds 10 blocks. For 4 KV-head tiles the search
    picks 5 parallel blocks (100 %, two waves) and stops at 6; that holds for any n_kv ≥ 320 (`:1193-1224`).
  - The partials (5 × 32 heads × 128 × 4 B = 80 KiB) and their metadata (1.25 KiB) come from the VMM pool
    (`:1221-1223`).
  - Sinks are applied by the first split only (`fattn-tile.cuh:1053`), and the combine kernel's cost doesn't depend on
    n_kv.

## 3. What grows with context, per card and on the host

Nine cards, f16 KV, `-ub 128` (est. from the code; the Qwen3.6 table in the [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)
fits the same formulas):

| Item | Per context token | At 33,024 | At 65,792 | Where |
|---|---|---|---|---|
| KV of each full-attention layer on the card | 2 KiB | 64.5 MiB | 128.5 MiB | `llama-kv-cache.cpp`; 1–2 such layers per card on nine cards |
| Sliding-window KV (768 cells) | — | 1.5 MiB per layer | same | `llama-kv-cache-iswa.cpp:73` |
| GPU compute buffer: 4 pipeline copies of the 128-row mask | 1 KiB | 32 MiB | 64 MiB | `ggml-backend.cpp:1973`; the Qwen table grows 4 KiB per token at `-ub 512` |
| Pool allocations during decode (partials) | — | 81 KiB | 81 KiB | `fattn-common.cuh:1221-1223` |
| Host, pinned: the same mask copies plus inputs | 1 KiB | ~43 MiB | ~75 MiB | `llama-context.cpp:508-514` |
| Host, pageable: llama-bench's depth state | 28 KiB | 0.91 GiB | 1.78 GiB (2.69 GiB peak) | `llama-bench.cpp:2472-2476` |

The f16 scratch that a quantized cache needs (`fattn-common.cuh:53-82`) isn't allocated for f16 K/V.

## 4. Experiments on the rig

`BIN` is the measured build's `bin/` (`eeaa59660`), `MODEL` the S4-L0 file, `OUT` a scratch directory under `/tmp`
(remove it afterwards). Loading S4-L0 from the slower stick takes ~7 min; a 32k fill ~4 min and a 64k fill ~11 min.

### E1. Reproduce with the host, swap and clocks logged (first; ~25 min; tests H1, H3 and the host costs)

```bash
OUT=/tmp/cliff; mkdir -p $OUT
vmstat -t -w 1 > $OUT/vmstat.txt & VM=$!
while sleep 1; do awk -v t="$(date +%T)" '/MemAvailable|SwapFree/ {s = s " " $1 " " int($2/1024)} END {print t s}' /proc/meminfo; done \
    > $OUT/mem.txt & MEM=$!
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 1000 > $OUT/smi.csv & SMI=$!
grep -E 'pswpin|pswpout|pgmajfault' /proc/vmstat > $OUT/vmstat-before.txt
LLAMA_NO_MOE_DOWN_RESCALE=1 GGML_CUDA_PROFILE_HOST=$OUT/phost.csv \
    $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 64 -r 1 -d 32768,65536 -o jsonl --progress \
    > $OUT/bench.jsonl 2> >(while IFS= read -r l; do printf '%(%T)T %s\n' -1 "$l"; done > $OUT/progress.log)
grep -E 'pswpin|pswpout|pgmajfault' /proc/vmstat > $OUT/vmstat-after.txt
kill $VM $MEM $SMI
```

`--progress` timestamps the "depth run" and "generation run" lines, so the memory, swap and clock logs can be cut at
the start of each phase. The timestamps come from a shell builtin, so logging forks no process per line. Per-token
timeline of the 64k decode (on any machine with the CSV):

```python
import csv, collections, sys
ev = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    ev[int(r["eval"])][int(r["device"])] = r
last = sorted(e for e in ev if len(ev[e]) == 9)[-63:]  # tokens 1-63: a row is written at the next evaluation
for a, b in zip(last, last[1:]):
    wall = float(ev[b][0]["host_enter_ms"]) - float(ev[a][0]["host_enter_ms"])
    chain = sum(float(r["gpu_us"]) for r in ev[a].values()) / 1000
    print(f"{a:6d}  wall {wall:7.1f} ms  GPU chain {chain:6.1f} ms")
```

`scripts/cuda_profile_summary.py $OUT/phost.csv --tokens 63` gives the averages. Every device evaluates once per token
and once per micro-batch, so evaluation numbers line up across devices.

| Reading | H1 holds | H2 or a GPU-side cause | H3 holds |
|---|---|---|---|
| tok/s at 64k | ~3.6 again | ~3.6 again | ~3.6 again |
| MemAvailable during the 64k state copy | falls to ~0, SwapFree falls | stays above ~0.5 GiB | — |
| `si` (vmstat) during the 64k generation run | > 0 | ~0 | ~0 |
| Wall time against GPU chain per token | wall ≫ chain on the slow tokens; chain ~45–60 ms (est.) | chain ≈ wall ≈ 275 ms | chain ≈ wall |
| SM clocks during the 64k decode | ≥ ~1,700 MHz | ≥ ~1,700 MHz | one or more cards far lower, event reasons ≠ 0 |

The same run gives the host's baseline: MemAvailable after the model load and after the 32k copy. That is the
CUDA-runtime footprint this note could not measure.

### E2. The same depths without any state copy (~23 min; confirms H1, kills H2)

```bash
LLAMA_NO_MOE_DOWN_RESCALE=1 $BIN/llama-batched-bench -m $MODEL -ngl 99 -fa on -fit off -c 65792 -b 2048 -ub 128 \
    -npp 32768,65536 -ntg 64 -npl 1 --output-format jsonl > $OUT/bb.jsonl 2> $OUT/bb.log
```

- llama-batched-bench fills each depth with the same 2,048-token calls and 128-token micro-batches, then decodes
  one token at a time. It keeps no copy of the state (`tools/batched-bench/batched-bench.cpp:122-222`).
- `-fit off` keeps the default layer split, as llama-bench does (the common tools fit by default).
- Its decode n_kv is the same as llama-bench's: 33,024 and 65,792.
- Keep the `vmstat`/`mem.txt` loggers of E1 running.

Expected:
- **H1:** S_TG ~23 tok/s at 32k and 17–20 at 64k (est.), with no swap activity.
- **H2:** ~3.6 at 64k without swapping.

The run also records prefill speed (S_PP) at both depths.

### E3. Only if E2 is still slow at 65,536: 2^16 or not, and which op (~1 h)

```bash
for pp in 65000 65536; do   # decode n_kv 65,280 (below 2^16) and 65,792
    LLAMA_NO_MOE_DOWN_RESCALE=1 GGML_CUDA_PROFILE=$OUT/prof-$pp.csv $BIN/llama-batched-bench -m $MODEL -ngl 99 -fa on \
        -fit off -c 65792 -b 2048 -ub 128 -npp $pp -ntg 32 -npl 1
    python3 scripts/cuda_profile_summary.py $OUT/prof-$pp.csv --tokens 24 --by op
done
LLAMA_NO_MOE_DOWN_RESCALE=1 GGML_CUDA_DISABLE_GRAPHS=1 $BIN/llama-batched-bench -m $MODEL -ngl 99 -fa on -fit off \
    -c 65792 -b 2048 -ub 128 -npp 65536 -ntg 64 -npl 1
```

- **65,000 fast and 65,536 slow:** H2. The op whose time jumps is the culprit.
- **Both slow:** compare `FLASH_ATTN_EXT` per call with 2 × its 32k value. If it is much more, look at the kernel; if
  attention is normal and other ops grew, look at those ops.
- **The run without CUDA graphs fast:** the graph path re-captures at that depth. `GGML_CUDA_GRAPH_DEBUG=1` with
  `-v` then shows "warmup reset" or "update failed" lines per token.

### E4. Optional: the same mechanism at 32k (~15 min)

Hold 1 GiB of anonymous memory during a 32k run:

```bash
python3 -c 'import time; b = bytearray(b"\x01") * (1 << 30); time.sleep(3600)' & HOG=$!
LLAMA_NO_MOE_DOWN_RESCALE=1 $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 64 -r 1 -d 32768 -o jsonl
kill $HOG
```

- H1 predicts that the 32k decode falls well below 23 tok/s, with swap-ins.
- The hog is swappable, so the kernel swaps rather than killing processes. Still, watch `mem.txt` and kill the hog if
  MemAvailable stays under ~300 MiB.

### E5. Optional: where the cliff starts, and the clean curve (~1 h for each part)

- **Clean speed curve:** `llama-batched-bench ... -npp 40960,49152,57344,65536 -ntg 64 -npl 1` (with `-c 65792`), for
  the public numbers.
- **Onset under H1:** llama-bench with one depth per invocation (40,960 / 49,152 / 57,344 / 65,536), with
  `mem.txt` running.
  - The states are 1.12, 1.34, 1.56 and 1.78 GiB.
  - H1 puts the onset at the first depth whose state no longer fits in MemAvailable after the model load.
  - One depth per run also removes the resize peak, so the onset is later than in the original sequence.

### E6. Only if E1's `memory.used` shows a card within ~200 MiB of 6,068 MiB

Move part of that card's share elsewhere with `-ts`, e.g. `-ts 1/1/1/0.85/1/1/1/1/1` for card 3 (llama-bench separates
split values with `/`). Under H1 this changes nothing.

## 5. What to change if each hypothesis holds

**H1 (host RAM):**
1. **Q54:** drop the 64k point, and take 64k (and 131k) from llama-batched-bench (E2, E5).
   - Expected (est.): 64k 17–20 tok/s; 131k 10–16 tok/s, depending on whether attention reads at ~135 GB/s (the 0→32k
     slope) or at the 57–87 GB/s of the attention note.
   - Q54's original nine-card line, `-d 32768,65536,131072`, would need a 3.5 GiB state buffer at 131k, plus the 1.8
     GiB one during the resize. That is beyond RAM and most of the swap (est.); don't run it as it stands.
2. **llama-bench in the fork** (`llama-bench.cpp:2444-2476`, ~20 lines; a candidate for upstream, since any small-RAM
   host hits it):
   - keep the depth state only when a later repetition or test uses the same depth;
   - skip it when `llama_state_seq_get_size` exceeds half of MemAvailable, and say so with a direct `fprintf(stderr)`
     (the logger is silenced);
   - an environment switch to turn the cache off.
3. **`scripts/bench.sh`:** log MemAvailable, SwapFree and vmstat `si`/`so` next to `smi.csv`, and print a warning when
   swap-ins occurred during a run.
4. **Serving:**
   - llama-server's host prompt cache copies a slot's whole state with the same call (`server-context.cpp:341-364`).
     Its default limit is 8,192 MiB (`common/common.h:642`), above the host's RAM, and at 64k one saved prompt is
     1.78 GiB. Keep `--cache-ram 0`, as `server_bench.sh` and the chat benchmark already do.
   - Context checkpoints copy only the sliding-window cache, ~35 MiB each for Flint at `-ub 128` (est.). A few (the
     chat benchmark uses 4) are affordable; the default 32 are not.
5. **Hardware:** the board has a second SO-DIMM slot (32 GB maximum; [hardware spec](../../mining_rig.md)). A second
   DDR4 module removes this class of problem and makes the memory dual-channel. Without it, running benchmarks with
   the desktop stopped frees a few hundred MiB (est.).

**H2 (above 65,536 keys):** E3's per-op profile names the kernel. Fix its index type or launch dimensions there, and
add a `test-backend-ops` case at n_kv = 65,792. Until it is fixed, decode beyond 65,536 tokens of context is broken
for every long-context use of the rig, so report it upstream with the shapes.

**H3 (clocks):** read the event reasons in `smi.csv`. HW slowdown or a power brake points to cooling or power
delivery; a software power cap at 75 W is the card's maximum and can't be raised. This needs no code change.

**Host per-token costs** (if E1's timeline shows ≥ 3 ms per token at 64k): the GPU-built mask of Q65, or copying
only the cells that changed. That would pay off at 131k–262k, not for the cliff.

## Not verified

- The host memory taken by the CUDA runtime for nine devices, MemAvailable after the model load, and whether the rig
  swapped during the Q54 run (no memory or swap log was kept).
- Whether the ~14 s went to a few stalled tokens or was spread over all 64 (one sample, no per-token timing).
- GPU clocks and per-card memory during the 64k decode. Free VRAM at 64k is inferred from the 262,144-token load of
  Q58.
- That no op other than `FLASH_ATTN_EXT` sees n_kv in Flint's decode graph. This was read from the graph builder and
  the CUDA dispatch, not from a dumped graph.
- The 0→32k slope (6.8 ms for 0.95 GB, ~138 GB/s) is above the ~87 GB/s ceiling estimated for the TILE kernel in the
  attention note. Both depth points are single samples; E2 measures the slope without the state copy.
- The swap-in rate on the encrypted SSD, and the size of the sliding-window part of the state (~512 live cells
  assumed).

## Sources

llama.cpp fork `pascal-p106-v2` at `032b130f4` (P):
- `tools/llama-bench/llama-bench.cpp`:
  - context size `:1308`;
  - `ctx_state` `:2162-2166`;
  - `test_gen` `:2198-2222`;
  - log filter `:2224-2230`, `:2286`;
  - test loop `:2317-2532`, with the warm-up `:2415-2442`, the depth and state cache `:2444-2476` and the timer
    `:2484`.
- `tools/batched-bench/batched-bench.cpp`: warm-up and the fill, decode and timing loop `:100-222`.
- `src/llama-context.cpp`:
  - n_ctx padding `:387`;
  - pinned host buffer for the CPU backend `:508-514`;
  - pipeline parallelism `:527-557`;
  - host state writer `:2919-2970`;
  - `state_seq_get_size` / `state_seq_get_data` `:3418-3448`.
- `src/llama-kv-cache.cpp`:
  - slot search `:1024-1056`;
  - `get_n_kv` `:1285-1299`;
  - mask fill `:1575-1724`, `:1765-1806`;
  - `state_write` `:2076-2144`.
- `src/llama-kv-cache-iswa.cpp`: sliding-window cache size `:73`.
- `src/llama-graph.cpp`: mask tensor `:30-48`, mask reuse check `:49-66`.
- `src/llama-impl.cpp`: `llama_log_set` `:54-58`.
- `ggml/src/ggml-backend.cpp`:
  - input copies `:1823-1863`;
  - split loop `:1865-1945`;
  - `n_copies` `:1973`;
  - `GGML_SCHED_MAX_COPIES` `:911-912`.
- `ggml/src/ggml-cuda/fattn.cu`:
  - kernel selection `:540-719`, with the GQA condition `:560-571`, the Ada-only 8,192 threshold `:645` and the
    no-tensor-core branch `:704-718`;
  - scratch size `:721-755`;
  - dispatch `:757-772`.
- `ggml/src/ggml-cuda/fattn-common.cuh`:
  - kernel signature `:20-39`;
  - f16 scratch `:47-82`;
  - `launch_fattn` `:980-1314`, with the mask scan condition `:1117`, occupancy `:1137`, parallel-block search
    `:1193-1224`, partials `:1221-1223` and grid `:1217-1219` and launch arguments `:1251-1266`.
- `ggml/src/ggml-cuda/fattn-tile.cuh`: KV loop `:953-981`, sinks `:1053`, launch switches `:1150-1320`, GQA limit
  `:1252-1253`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - `ggml_cuda_device_malloc` `:188-216`, unified memory switch `:159`;
  - legacy pool with flush-and-retry `:460-591` (retry `:553-566`);
  - VMM pool `:595-742`, pool choice `:744-753`;
  - CUDA graph checks `:2721-2835`;
  - host profiler `:4941-5010`;
  - graph enable and warm-up `:5013-5080`.
- `ggml/src/ggml-cuda/common.cuh`: `ggml_cuda_graph` and `GGML_CUDA_DISABLE_GRAPHS` `:1271-1302`; graph cache and
  eviction `:1474-1495`.
- `tools/server/server-context.cpp`: prompt cache save `:341-364`, setup `:1574-1585`; `common/common.h:639-643`
  (checkpoints and cache-ram defaults).
- Upstream `85d5703a3`: CUDA graphs disabled below Volta (`ggml-cuda.cu:4403-4407`); the same llama-bench depth cache
  (`llama-bench.cpp:2444-2470`).

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): Q54 (the measurement, the 6-card out-of-memory failures),
  Q58, Q65, Q79.
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md): host memory at 262,144 tokens (267 MiB pinned at `-ub 128`).
- [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md): S4-L0 at a 262,144-token context on nine cards,
  `--cache-ram 0`, `--ctx-checkpoints 4`.
- [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md): depth speeds on nine cards, pinned host buffer and
  GPU compute buffer by context and micro-batch, the out-of-memory incident.
- [hardware spec](../../mining_rig.md): RAM, SO-DIMM slots, PCIe links, thermal limits and measured temperatures.
- [pascal_attention_long_context.md](pascal_attention_long_context.md): the TILE kernel's read rate at depth.
- `scripts/rig_inventory.sh` (host RAM and swap, 2026-10-22), `scripts/bench.sh`, `scripts/server_bench.sh`,
  `scripts/cuda_profile_summary.py`.
