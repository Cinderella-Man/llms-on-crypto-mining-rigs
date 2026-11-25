# Ornith with a Q8_0 cache and pull request 9's attention kernels: four alternated agent checks on the second rig (R232 (b), Q1706; 2026-11-24)

**In plain words.** Ornith-1.5-122B keeps its attention cache in 16-bit numbers (F16). An 8-bit cache (Q8_0) is half
the size, and a contributor's pull request 9 added a decode kernel for it on Pascal cards. On one card at Ornith's
exact shape that kernel made a one-token attention call 1.85 times faster than the F16 cache at 8,192 tokens, but
prompts were read slower ([R232 (a)](../2026-11-24_r232a_ornith_q8_kernels/README.md)). This folder decides whether a
whole coding-agent session gets faster: four agent checks on one boot of the second rig, every card at its soaked clock
profile, alternating Ornith's v5 preset as it is (F16 cache, the control) and the same with a Q8_0 cache and the two
switches that route its one-token calls to the new kernel.

**Run 1, the control (`o5f16a`, 2026-11-24 16:14–16:45): 12 of 12 tasks passed, 1.88 minutes a task at equal work,
19.42 tok/s generation, and no reply was read again.** As context only, R17's two runs of the same preset on the same
rig and clock profile, on another boot and an older v5 engine (`602691153`): 1.89 and 1.90 minutes a task, 19.41 and
19.01 tok/s.

**Run 2, the Q8_0 cache (`o5q8a`, 16:52–17:26): 12 of 12 tasks passed, 1.84 minutes a task at equal work, 20.39 tok/s
generation, and no reply was read again.** Like for like against run 1, a decode step took 48.75 ms against 51.58 at
the five runs' shared mix of contexts (5.5 % less), and the 19 requests read on the same prompt took 184.21 s against
179.86 (2.4 % more): faster generation, slower prompts, as the kernel times said.

**Run 3, the control again (`o5f16b`, 18:36–19:13): 12 of 12 tasks passed, 1.93 minutes a task at equal work, 18.80
tok/s generation, and no reply was read again.** Like for like it repeats run 1: a decode step at the five runs' shared
mix took 51.63 ms against run 1's 51.58, and the 19 same-prompt requests 180.09 s against 179.86. Its session numbers
are lower because of its work: Python `scale-generator` generated 9,522 tokens over 8 turns (9.7 minutes), so more of
its decoding ran at long contexts. Against the two F16 runs, the Q8_0 run's decode step took 48.75 ms against 51.58 and
51.63, and its same-prompt requests 184.21 s against 179.86 and 180.09 (at the five runs' mix of run 3's record; the
table below takes the mix over six runs).

**Run 4, the Q8_0 cache again (`o5q8b`, 19:29–20:06): 12 of 12 tasks passed, 1.84 minutes a task at equal work, 20.45
tok/s generation, and no reply was read again.** Like for like it repeats run 2: a decode step at the six runs' shared
mix took 48.66 ms against run 2's 48.79, and the 19 same-prompt requests 184.23 s against 184.21.

**The verdict: the Q8_0 cache with the two switches is a gain on agentic speed.** Two runs against two
([class.tsv](class.tsv)): 1.84 and 1.84 minutes a task at equal work against the F16 runs' 1.88 and 1.93 (3.4 % less,
beyond both arms' spreads), 20.39 and 20.45 tok/s generation against 19.42 and 18.80 (6.9 % more). Like for like, a
decode step took 48.79 and 48.66 ms against 51.65 and 51.72 (5.7 % less), faster in every context bucket and more so
at longer contexts. The price is prompt reading: the 19 same-prompt requests took 184.21 and 184.23 s against 179.86
and 180.09 (2.4 % more), so prompt tok/s and the first token on a task's first turn (8.16 and 8.12 s against 8.01 and
7.96) are losses. Decoding is about three quarters of a session's prompt and decode time here, so the gain outweighs
the price: on run 1's work, est. 46.4 s less of its 1,286.9 s (3.6 %). All 12 tasks passed in each of the four runs, and the Q8_0
cache's quality was within noise of F16 on Ornith ([C976](../2026-11-23_c976_ornith_q8_cache_kld/README.md); a
perplexity rise of a few tenths of a percent beyond 8,192 tokens is not ruled out). Whether Ornith's v5 preset takes
it is the owner's decision; this record recommends it.

## The runs

The release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at the preset's 49,152 tokens on nine P106-090 cards, on the
v5 engine (`pascal-p106-v5` at `af00bb13b`, which holds pull request 9's kernels) with Ornith's v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)). The arms differ only in the cache and the
two switches ([run/ornith_q8.sh](run/ornith_q8.sh)):

| Run | Arm | Added to the v5 preset | Status |
|---|---|---|---|
| 1, F16 a | `o5f16a` | nothing | done 16:14–16:45, 12/12 |
| 2, Q8_0 a | `o5q8a` | `-ctk q8_0 -ctv q8_0`; `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` | done 16:52–17:26, 12/12 |
| 3, F16 b | `o5f16b` | nothing | done 18:36–19:13, 12/12 |
| 4, Q8_0 b | `o5q8b` | as run 2 | done 19:29–20:06, 12/12 |

The class is two runs against two by [class_arms.py](../../../bench/agent_check/class_arms.py): a measure is a
gain or a loss only where the difference of the two arms' means is beyond the larger of their spreads. Beside them, as
context, R17's text-reuse runs `o5t1a` and `o5t1b`
([R17](../2026-11-19_r17_ornith_textreuse/README.md#text-reuse-run-b-o5t1b-r17f-2026-11-19)): the same preset,
rig and clock profile and the same started command but for the engine's path, on 2026-11-19's boot with the older v5
engine `602691153`. They are not a control: the engine and the boot differ.

## Runs 1 to 4: `o5f16a` (F16 a), `o5q8a` (Q8_0 a), `o5f16b` (F16 b) and `o5q8b` (Q8_0 b), 2026-11-24

The shared mix of contexts and the requests read on the same prompt are taken over every run in the table, so they are
recomputed as runs are added.

| | R17d `o5t1a` (context) | R17f `o5t1b` (context) | **`o5f16a`** | **`o5q8a`** | **`o5f16b`** | **`o5q8b`** |
|---|---|---|---|---|---|---|
| passed | 12/12 | 11/12 (Python `scale-generator` at the 12-minute cap) | **12/12** | **12/12** | **12/12** | **12/12** |
| minutes a task at equal work (`v3ka`'s work) | 1.89 | 1.90 | **1.88** | **1.84** | **1.93** | **1.84** |
| minutes, all 12 tasks; a task's median; turns | 21.4; 1.33; 64 | 34.8; 1.52; 55 | **21.9; 1.68; 59** | **25.0; 1.76; 65** | **27.2; 1.44; 59** | **26.6; 1.80; 67** |
| generation tok/s, session | 19.41 | 19.01 | **19.42** | **20.39** | **18.80** | **20.45** |
| ms a decode step at the six runs' shared mix of contexts; tok/s at it | 51.43; 19.44 | 51.35; 19.47 | **51.65; 19.36** | **48.79; 20.50** | **51.72; 19.33** | **48.66; 20.55** |
| prompt tok/s, session | 102.5 | 106.0 | **104.3** | **100.5** | **103.1** | **99.2** |
| the 19 requests read on the same prompt in all six, prompt s | 180.29 | 180.80 | **179.86** | **184.21** | **180.09** | **184.23** |
| of them the 12 first turns, prompt s | 110.39 | 110.77 | **110.01** | **111.86** | **109.98** | **112.32** |
| first token median; p90 s | 6.13; 9.58 | 6.94; 10.31 | **6.59; 10.27** | **6.16; 9.95** | **6.40; 10.32** | **5.71; 9.93** |
| first token on a task's first turn, median s | 8.00 | 7.98 | **8.01** | **8.16** | **7.96** | **8.12** |
| longest first-token wait s (task, turn) | 26.70 (Python `affine-cipher`, 1) | 26.89 (the same) | **26.28 (the same)** | **26.48 (the same)** | **26.49 (the same)** | **27.18 (the same)** |
| prompt tokens read (a task) | 36,956 (3,080) | 34,628 (2,886) | **35,247 (2,937)** | **36,712 (3,059)** | **35,517 (2,960)** | **35,970 (2,998)** |
| tokens generated (a task) | 17,458 (1,455) | 21,478 (1,790) | **18,433 (1,536)** | **22,559 (1,880)** | **23,708 (1,976)** | **24,664 (2,055)** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 360.4) | 0 (0; 0.0 of 326.8) | **0 (0; 0.0 of 337.8)** | **0 (0; 0.0 of 365.5)** | **0 (0; 0.0 of 344.6)** | **0 (0; 0.0 of 362.5)** |
| tool calls / failed / malformed | 64 / 2 / 0 | 55 / 1 / 0 | **59 / 1 / 0** | **65 / 1 / 0** | **59 / 0 / 0** | **67 / 2 / 0** |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 11 / 0 / 0; 0; 0 in 64 | 11 / 0 / 0; 0; 0 in 54 | **11 / 0 / 0; 0; 0 in 59** | **11 / 0 / 0; 0; 0 in 65** | **11 / 0 / 0; 0; 0 in 59** | **11 / 0 / 0; 0; 0 in 67** |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8) | 2113 2138 2138 2088 2100 2088 2151 2151 2113 (2,120.0) | **2113 2138 2138 2088 2100 2100 2151 2138 2126 (2,121.3)** | **2113 2138 2138 2088 2100 2088 2151 2138 2126 (2,120.0)** | **2100 2138 2138 2088 2100 2088 2138 2138 2113 (2,115.7)** | **2113 2138 2138 2088 2100 2088 2151 2138 2113 (2,118.6)** |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | **0; 0** | **0; 0** | **0; 0** | **0; 0** |
| lowest MemAvailable MiB; swap MiB during the load | 1,189; 53 → 363 | 1,227; 53 → 363 | **1,223; 50 → 362** | **1,240; 52 → 362** | **1,220; 54 → 365** | **1,196; 54 → 364** |

- **The class** ([class.tsv](class.tsv), [class_arms.py](../../../bench/agent_check/class_arms.py)): the Q8_0 arm's two
  runs against the F16 arm's two, a gain or a loss only where the difference of the means is beyond the larger of the
  two arms' spreads (the highest run minus the lowest):
  - **gain:** minutes a task at equal work 1.84 and 1.84 against 1.88 and 1.93 (3.4 % less; spreads 0 and 0.05);
    generation 20.39 and 20.45 tok/s against 19.42 and 18.80 (6.9 % more; spreads 0.06 and 0.62); first token median
    6.16 and 5.71 s against 6.59 and 6.40 (8.6 % less), p90 9.95 and 9.93 s against 10.27 and 10.32 (3.4 % less);
  - **loss:** prompt tok/s 100.5 and 99.2 against 104.3 and 103.1 (3.7 % less; spreads 1.3 and 1.2); first token on a
    task's first turn 8.16 and 8.12 s against 8.01 and 7.96 (1.9 % more); prompt tokens read a task 3,059 and 2,998
    against 2,937 and 2,960 (2.7 % more);
  - **negligible:** the longest first-token wait, 26.48 and 27.18 s against 26.28 and 26.49 (spreads 0.70 and 0.21);
    no reply was read again in any run.
- **What the first-token gain and the work loss are made of:** the median and p90 are taken over all of a run's turns,
  and the Q8_0 runs took more turns (65 and 67 against 59 and 59), most of them short late turns that read a few dozen
  new tokens (17 and 19 turns at turn 5 or later, against 11 and 11). Over the 55 task-turns all four runs have, the
  median first token is 7.05 and 6.85 s with the Q8_0 cache against 6.75 and 6.40 with F16, and on the same prompts
  the Q8_0 cache reads slower (below). So that gain comes from the agent's turns, not from faster first tokens, and
  the extra prompt tokens read come from the same extra turns (no reply was read again). A first-token measure on the
  same prompts for the class is code row C1075 (Q1808).
- **Run 4 repeats run 2 like for like** ([decode.md](decode.md), [same_prompts.md](same_prompts.md)): a decode step at
  the six runs' shared mix 48.66 ms against 48.79; the 19 same-prompt requests 184.23 s against 184.21, the 12 first
  turns 112.32 against 111.86. Run 3 repeats run 1 the same way: 51.72 ms against 51.65 (within 0.61 ms in each
  context bucket: 48.31 against 48.34 ms under 3,000 tokens, 49.81 against 49.98 at 3,000 to 4,500, 51.34 against
  51.54 at 4,500 to 6,000, 53.79 against 53.18 at 6,000 to 8,000), 180.09 s against 179.86, 109.98 against 110.01.
- **Generation:** a decode step at the shared mix took 48.79 and 48.66 ms with the Q8_0 cache against 51.65 and 51.72
  with F16: 2.96 ms (5.7 %) less between the means, beyond both arms' spreads (0.13 and 0.07 ms); 20.50 and 20.55
  tok/s against 19.36 and 19.33. It was faster in every context bucket, and the gap grows with the context
  ([decode.md](decode.md)): under 3,000 tokens 47.45 and 47.47 ms against 48.34 and 48.31; 3,000 to 4,500 47.91 and
  47.83 against 49.98 and 49.81; 4,500 to 6,000 48.64 and 48.63 against 51.54 and 51.34; 6,000 to 8,000 49.68 and
  49.33 against 53.18 and 53.79; at 8,000 tokens and over (3 and 6 requests against 1 and 5) 52.41 and 50.23 against
  54.91 and 59.08. R232 (a)'s kernel times put the gap at est. 3.42 ms a token at 8,192 tokens and est. 21.00 ms at
  49,152 ([R232 (a)](../2026-11-24_r232a_ornith_q8_kernels/README.md)).
- **Prompts:** the 19 same-prompt requests took 184.21 and 184.23 s against 179.86 and 180.09: 4.25 s (2.4 %) more
  between the means (spreads 0.02 and 0.23 s); 16 of the 19 took longer in both Q8_0 runs than in both F16 runs, the
  other three are small first turns (JavaScript `grade-school`, `phone-number` and `triangle`) on which the two F16
  runs themselves differ by 189.9, 501.9 and 14.3 ms. The 12 first turns took 111.86 and 112.32 s against 110.01 and
  109.98 (1.9 % more). R232 (a) measured the Q8_0 cache's 512-token prompt calls 23.4 % slower than F16's at depth 256.
- **What it adds up to** (est.): decoding took 73.8 %, 75.2 %, 78.5 % and 76.9 % of the four runs' prompt and decode
  seconds (runs 1 to 4). Priced at the like-for-like rates above (a decode step 5.7 % shorter, prompts 2.4 % longer),
  run 1's work would take est. 54.4 s less decoding and est. 8.0 s more prompt reading with the Q8_0 cache: est. 46.4 s
  less of its 1,286.9 s (3.6 %); run 3's work est. 64.1 s less of its 1,605.4 s (4.0 %). The minutes at equal work say
  the same (3.4 % less).
- **The work** ([work.md](work.md)): in all four runs every later turn found the previous prompt and reply in the cache
  (no reply read again; median cache hit from turn 2 95.0 % in `o5f16a`, 98.5 % in `o5q8a`, 96.0 % in `o5f16b` and
  99.0 % in `o5q8b`, [o5f16a_gonogo.md](o5f16a_gonogo.md), [o5q8a_gonogo.md](o5q8a_gonogo.md),
  [o5f16b_gonogo.md](o5f16b_gonogo.md), [o5q8b_gonogo.md](o5q8b_gonogo.md)). The longest wait is Python
  `affine-cipher`'s first turn, a 2,742-token prompt with nothing cached: 26.28 s, 26.48 s, 26.49 s and 27.18 s.
  The tokens generated differ with what the agent did: run 1 18,433, run 2 22,559 (Python `scale-generator` 7,557
  against run 1's 3,462, `affine-cipher` 12 turns against 6), run 3 23,708 (`scale-generator` 9,522 in 8 turns, 9.7
  minutes; five requests decoded at a middle context of 8,000 tokens or more), run 4 24,664 (`list-ops` 4,901 in 10
  turns, `affine-cipher` 3,022 in 10 turns, `scale-generator` 4,455). So the minutes for all 12 tasks (21.9, 25.0,
  27.2 and 26.6) follow the work; the minutes at equal work take each run's own session speeds and so carry some of
  it too (run 3's 1.93 against run 1's 1.88, from its long contexts; code row C1064, Q1797).
- **Run 1 runs at R17's speed:** a decode step at the six runs' shared mix 51.65 ms against 51.43 and 51.35
  ([decode.md](decode.md)); the 19 requests every run read on the same prompt 179.86 s against 180.29 and 180.80
  ([same_prompts.md](same_prompts.md)). These compare two engines on two boots, so they are context, not a class.
- **Answers sound in all four runs:** every task passed, also its hidden tests, test files untouched; 0 of 59, 0 of
  65, 0 of 59 and 0 of 67 calls malformed, none inside the reasoning. The failed calls are edits in Python
  `affine-cipher` whose old text did not match the file, which the tool answered with an error, and the agent's next
  edit went through: one in run 1, one in run 2, none in run 3, two in run 4. No NaN or inf, no watchdog kill. The
  report's gates: all GO except memory (NO-GO on the swap growth during the load, as for every Ornith run of this
  preset).
- **Power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): above 80 W in 0 4 4 0 1 0 0 1 4 single
  samples on cards 0 to 8 in `o5f16a` (highest 87.81 W on card 1), 2 3 3 2 1 1 1 1 3 in `o5q8a` (highest 84.50 W on
  card 1), 1 5 3 4 0 3 5 4 4 in `o5f16b` (highest 91.48 W on card 7) and 2 2 3 6 2 3 0 2 3 in `o5q8b` (highest 85.77 W
  on card 1); the clock watchdog (80 W for 5 s) never tripped and the boot logged no GPU Xid.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,223 MiB (16:42:47) in `o5f16a`, 1,240 MiB
  (17:24:40) in `o5q8a`, 1,220 MiB (19:13:18, eight seconds after the last task) in `o5f16b` and 1,196 MiB (20:03:15)
  in `o5q8b`. Each card's peak memory on cards 0 to 8: 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB in
  `o5f16a`, 5,784 5,756 5,730 5,756 5,408 5,516 5,678 5,706 5,156 MiB in `o5q8a`, 5,830 5,846 5,776 5,846 5,454 5,562
  5,724 5,796 5,202 MiB in `o5f16b` and 5,762 5,738 5,708 5,738 5,390 5,498 5,660 5,692 5,142 MiB in `o5q8b`. `o5q8b`
  is lower than `o5f16a` by 46 90 46 90 46 46 46 90 46 MiB: one attention layer's cache at 49,152 tokens is est.
  96 MiB in F16 and est. 51 MiB in Q8_0 (2 KV heads of 256 values, keys and values), 45 MiB less, so cards 1, 3 and 7
  hold two of Ornith's 12 attention layers and the others one. The whole cache is est. 1,152 MiB in F16 against
  612 MiB in Q8_0.
- **The longer context the half-size cache allows** (est., from the sizes above): a Q8_0 cache at 98,304 tokens holds
  est. 102 MiB a layer, 6 MiB more than today's F16 cache at 49,152 on a card with one attention layer and 12 MiB on a
  card with two, before any growth of the other buffers with the context; at 65,536 tokens est. 68 MiB a layer, 28 MiB
  less than today's F16. The cache's quality at those lengths is LLMRig's C990 (Q1723); the fit and speed on the rig
  is rig 2 row 19e, C1074 (Q1807).

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 13:43 and set to its soaked clock profile between 14:02 and
  14:19 by the offset kit's `apply` ([R234](../2026-11-19_r209_oc_v5/r234_apply.md)): memory +1,300 +1,000 +1,000 +1,100 +1,100
  +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8 (R17's values),
  the 75 W power limit, the kit's watchdog running. Before and after each run every card read back at those values, no
  load process, 0 Xid, 0 step-downs. Nothing was set or reset during or between the runs. R232 (a) ran on card 0 of the
  same boot before them (14:41–16:01).
- **The build:** `pascal-p106-v5` `af00bb13b`, the rig's tarball `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2`
  (`llama-server` md5 `f5cca34ea6bd4f3712cd6193bf0605be`, checked before and after).
- **The started command** of `o5f16a` (575 bytes with its newline, md5 `1da8fb3fcf808af67ebcf41f008df06d`) is the
  launcher's dry run byte for byte (`cmp`), and R17's text-reuse runs' line byte for byte but for the engine's path; the
  server logged `switches: 8 set`, the v5 preset's eight:

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  ```

  The Q8_0 arm's line is the same with `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` after
  `LLAMA_SERVER_TEXT_REUSE=1` and `-ctk q8_0 -ctv q8_0` at the end: `o5q8a` started it as 658 bytes with its newline
  (md5 `070c3b31b29cd8289fcd3f048c616b8a`), `cmp`-equal to `o5f16a`'s line with those two edits, and its server logged
  `switches: 10 set`, the preset's eight and the two attention switches. `o5f16b` started `o5f16a`'s line byte for
  byte (575 bytes, md5 `1da8fb3fcf808af67ebcf41f008df06d`, `cmp`), and its server logged the same `switches: 8 set`.
  `o5q8b` started `o5q8a`'s line byte for byte (658 bytes, md5 `070c3b31b29cd8289fcd3f048c616b8a`, `cmp`), and its
  server logged the same `switches: 10 set`.
- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the bench files R17's runs used. Each run has its own client folder, which also names its
  containers, so another check on the same client at the same time cannot remove them. The work tables come from v5's
  [report.py](../../../bench/agent_check/report.py) `work` on the published rows.
- **Timeline:** run 1's server started at 16:14:08 and was up after 542 s (R17's runs: 658, 653, 660 and 662 s); the
  client reached it at 16:23:11, the 12 tasks ran 16:23:11–16:45:14, and the rig's logs went to the client at 16:45:35.
  Run 2's server started at 16:52:14 and was up after 539 s; the client reached it at 17:01:14, the 12 tasks ran
  17:01:14–17:26:27, and the rig's logs went to the client at 17:26:46. Run 3's server started at 18:36:40 and was up
  after 543 s; the client reached it at 18:45:44, the 12 tasks ran 18:45:44–19:13:10, and the rig's logs went to the
  client at 19:13:23. Run 4's server started at 19:29:35 and was up after 590 s, the slowest load of the four (the load
  is not part of the check's times); the client reached it at 19:39:26, the 12 tasks ran 19:39:26–20:06:13, and the
  rig's logs went to the client at 20:06:20. Between runs 2 and 3 the same preset ran on the same rig and boot on
  another engine (the upstream b12674 candidate, [C912's Ornith run](../2026-11-24_c912_ornith_agent/README.md),
  17:52–18:22). Nothing else ran on the rig.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local ARM=f16 RUN=a \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2/bin/llama-server \
  RIG_LOGDIR=<rig folder of this run> RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r232b-o5f16a OUT=<out>/agent-o5f16a \
  bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# the other runs: ARM=q8 RUN=a, ARM=f16 RUN=b, ARM=q8 RUN=b, each with its own rig folder and client folder
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` (the equal-work reference of every v5 Ornith table, as
[R17](../2026-11-19_r17_ornith_textreuse/README.md#commands) used them), `o5t1a` and `o5t1b`
([R17's agent/](../2026-11-19_r17_ornith_textreuse/agent/)) and each run's rows ([agent/o5f16a/](agent/o5f16a/),
[agent/o5q8a/](agent/o5q8a/), [agent/o5f16b/](agent/o5f16b/), [agent/o5q8b/](agent/o5q8b/)), each run's server log as
`rig/server-<arm>.log`, and in `rig_<arm>/` its `meminfo.log`, `cards.csv` and `window` (from the driver's log; `o5f16a`'s
`2026-11-24 16:23:11|2026-11-24 16:45:14`, `o5q8a`'s `2026-11-24 17:01:14|2026-11-24 17:26:27`, `o5f16b`'s
`2026-11-24 18:45:44|2026-11-24 19:13:10`, `o5q8b`'s `2026-11-24 19:39:26|2026-11-24 20:06:13`):

```bash
bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/tables.sh <tables folder> o5f16a o5q8a o5f16b o5q8b
python3 -I bench/agent_check/class_arms.py <tables folder>/agent.tsv o5q8=o5q8a,o5q8b o5f16=o5f16a,o5f16b > class.tsv
```

## Files

| File | Content |
|---|---|
| [class.tsv](class.tsv) | `class_arms.py`: the Q8_0 arm's two runs against the F16 arm's two, per measure (runs by value, means, spreads, the difference, the class) |
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5t1a`, `o5t1b`, `o5f16a`, `o5q8a`, `o5f16b`, `o5q8b`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `o5t1a`, `o5t1b` and each run: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [reallocs.tsv](reallocs.tsv), [o5f16a_requests_reallocs.tsv](o5f16a_requests_reallocs.tsv), [o5q8a_requests_reallocs.tsv](o5q8a_requests_reallocs.tsv), [o5f16b_requests_reallocs.tsv](o5f16b_requests_reallocs.tsv), [o5q8b_requests_reallocs.tsv](o5q8b_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5f16a_compare.md](o5f16a_compare.md), [o5f16a_report_table.md](o5f16a_report_table.md), [o5f16a_gonogo.md](o5f16a_gonogo.md); [o5q8a_compare.md](o5q8a_compare.md), [o5q8a_report_table.md](o5q8a_report_table.md), [o5q8a_gonogo.md](o5q8a_gonogo.md); [o5f16b_compare.md](o5f16b_compare.md), [o5f16b_report_table.md](o5f16b_report_table.md), [o5f16b_gonogo.md](o5f16b_gonogo.md); [o5q8b_compare.md](o5q8b_compare.md), [o5q8b_report_table.md](o5q8b_report_table.md), [o5q8b_gonogo.md](o5q8b_gonogo.md) | the harness's report of each run: per task, per turn and the go/no-go |
| [agent/o5f16a/](agent/o5f16a/), [agent/o5q8a/](agent/o5q8a/), [agent/o5f16b/](agent/o5f16b/), [agent/o5q8b/](agent/o5q8b/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of each run |
| [run/ornith_q8.sh](run/ornith_q8.sh), [run/tables.sh](run/tables.sh) | the launcher of the four runs (`ARM=f16\|q8`, `RUN=a\|b`; its `ARM=kdp` is C1102's, [2026-11-25_c1102_kdp_agent](../2026-11-25_c1102_kdp_agent/README.md)) and the tables |

## Provenance and licence

Measured on the project's own rig on 2026-11-24 with the repository's own scripts and Miner Llama's v5 engine (MIT, like
llama.cpp, with pull request 9's kernels under the same licence); the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<arm>/attempts.csv`); the
tables hold times, counts and task ids, no text. The tables and scripts here are under the repository's licence.
