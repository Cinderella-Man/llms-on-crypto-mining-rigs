# Flash-Next's four-token verify on the per-token sparse kernel in 5 splits, with today's switches (C1175, Q1908, 2026-11-26)

What this measured: Qwen3.8-Flash-Next's decode step on nine P106-090 cards when its per-token sparse attention kernel
splits a four-token verify pass into 5 pieces instead of 3, using two switches the engine already has, first on the
deep lane and then on the coding-agent check. Registry row [Q1908](../../../quants/experiments.md), from C839
([Q1473](../2026-11-26_c839_union_attn_rig2/README.md)).

C839 timed the attention call alone on one card: the engine's split rule (`fasp_pick_splits`, about 4 blocks a SM)
gives a four-token verify 3 splits a list, 24 blocks where a P106's 5 SMs hold 20 at once, so a second wave of 4
blocks. With 5 splits (two full waves) the call took 590.4, 589.9 and 590.2 µs instead of 968.3, 968.7 and 967.2 at
4,096 cells, and 625.7, 625.8 and 625.9 instead of 1,003.4, 1,003.7 and 1,002.1 at 8,192: est. 4.53 ms less a verify
pass of 12 attention calls. At 5 splits the per-token kernel also beat the dense kernels from est. 3,935 cells.

**Result: a gain on the decode step, and slower on no figure the switches reach.** On the deep lane both arms make the
step shorter wherever they run the sparse kernel: 4.55 and 4.51 ms at 8,192 tokens (C839's est. 4.53), 3.88 and 4.02
at 6,144, and `_MIN_RATIO=0.5` also 1.40 at 4,096; at depth 0 the step is the same. On the coding-agent check the
better arm, `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10` with `_MIN_RATIO=0.5`, against the v5 preset, two runs each,
alternated, on one rig, engine and boot:
- **The decode step:** 104.05 and 104.26 ms over all requests against 109.42 and 107.10 (a gain), and shorter in every
  bucket of context where the switches act (4,102 to 6,153 cells: 103.62 and 104.05 against 107.75 and 105.72; 6,153
  to 8,204: 105.64 and 103.48 against 109.80 and 108.61); below 4,102 cells, where every arm runs the same kernels,
  the same (104.59 and 103.93 against 105.93 and 103.85).
- **The task:** 2.23 and 2.21 minutes at equal work against 2.29 and 2.23, negligible because one control run had a
  task at its 12-minute cap; without that task 2.31 and 2.30 against 2.34 and 2.33, a gain. 12/12 and 12/12 tasks
  against 11/12 and 12/12. Generation 31.54 and 32.04 tok/s against 29.05 and 31.56 (negligible).
- **The first token:** the median over all turns is a loss by the class rule (2.77 and 2.58 s against 2.36 and 2.18),
  but it comes from the turn mix, not from slower prompts: the arm's runs had fewer later turns, so the median lands on
  the first turns' floor; the first turns' median and the later turns' median are both negligible, and the 10 requests
  that read the same prompt on the same cache took 67,137.5 and 66,460.8 ms against 66,906.0 and 67,616.0.

By the project's rule (a gain joins the stacked line), `_NSM=10 _MIN_RATIO=0.5` qualifies for Flash-Next's v5 preset;
[release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) keeps `_NSM=5 _MIN_RATIO=0.75` until
the owner's word. The code fix C1176 ([Q1909](../../../quants/experiments.md)) would give the same 5 splits at 4 tokens
without `_NSM=10`'s 20 splits at 1 token (+43.3 µs a one-token call).

## The switches

In the engine's sparse attention ([fattn-sparse-pascal.cuh](https://github.com/Cinderella-Man/miner-llama/blob/eee615cdb/ggml/src/ggml-cuda/fattn-sparse-pascal.cuh) and `fattn-sparse-pascal.cu` at `eee615cdb`),
with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (Flash-Next's v5 preset):
- a decode or verify call of `n_q` query tokens (up to 8) goes to the per-token kernel once the cache holds at least
  `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO` × 2,051 (the cells a QSA layer keeps) × `n_q` cells; calls of more tokens
  (prompt micro-batches) follow their own rule (`_PROMPT_MIN_RATIO`), which no arm changes;
- its splits per list are `ceil(4 × GGML_CUDA_FATTN_SPARSE_PASCAL_NSM / columns)`, where the columns are the query
  tokens × 2 KV heads (at least 64 cells a split). `_NSM` is the SM count the rule assumes; the preset sets 5, a P106's.

| Arm | `_NSM` | `_MIN_RATIO` | Splits at 4 tokens; at 1 token | A four-token verify on the sparse kernel from |
|---|---:|---:|---|---:|
| `c`, the v5 preset (control) | 5 | 0.75 | 3; 10 | 6,153 cells |
| `n10` | 10 | 0.75 | 5; 20 | 6,153 cells |
| `r05` | 10 | 0.5 | 5; 20 | 4,102 cells |

At 1 token `_NSM=10` gives 20 splits instead of 10, which C839 timed at 258.5, 258.5 and 258.6 µs a call against
215.2, 215.2 and 215.2 (+43.3 µs). A request at depth D generates 256 tokens, so its verify passes read D to D + 256
cells plus the cache's padding: at depth 0 every arm runs the dense kernels; at 4,096 only `r05` runs the sparse one;
at 6,144 and 8,192 all three do, `n10` and `r05` in 5 splits.

## The deep lane

### Result

**Both arms make the step faster wherever they run the sparse kernel, by what C839 estimated, and neither is slower
where they do not.** On the repeated request (pass 2, which meets no token new to the draft's shortlist), ms a step,
each value one load:

| Depth (tokens) | `c` (the preset) | `n10` | `n10` − `c`, class | `r05` | `r05` − `c`, class |
|---:|---|---|---|---|---|
| 0 | 90.31, 90.29 | 90.44, 90.28 | +0.06, negligible | 90.19, 90.20 | −0.10 (see below) |
| 4,096 | 99.75, 99.60 | 99.69, 99.89 | +0.11, negligible | 98.23, 98.32 | **−1.40, gain** |
| 6,144 | 102.83, 102.91 | 98.92, 99.05 | **−3.88, gain** | 98.80, 98.90 | **−4.02, gain** |
| 8,192 | 103.99, 103.89 | 99.32, 99.46 | **−4.55, gain** | 99.39, 99.47 | **−4.51, gain** |

The class is the lane's rule: a gain or a loss only beyond the larger of the two arms' spreads (the difference between
an arm's two loads). [lane_steps.md](lane_steps.md) has pass 1, the sampled request and both class rules.

- **At 8,192 tokens the step is 4.55 and 4.51 ms shorter, C839's est. 4.53 ms a verify pass** (12 calls × 377.8 µs).
  At 6,144, 3.88 and 4.02 ms.
- **At 4,096 tokens only `r05` acts: −1.40 ms a step on pass 2 and −1.69 on pass 1.** C839 estimated 0.26 ms a pass
  from the kernels alone at 4,096 cells (the dense call 611.9 µs, the per-token one at 5 splits 590.2); the requests
  here read 4,096 to 4,352 cells, where the dense call grows by 0.1437 µs a cell and the sparse one hardly at all.
  `r05`'s reply there differs from the preset's after 114 equal tokens, so the two arms' steps cover other text (81
  steps against 79).
- **At depth 0 all three arms run the same kernels** (the cache holds at most 320 cells, under every switch point).
  `r05`'s −0.10 ms on pass 2 is a "gain" only against the control's 0.02 ms spread; `n10`'s two loads differ by
  0.16 ms there, so it is not read as the switch's.
- **Pass 1** (the first request at a new depth): `n10` −3.34 and −9.32 ms at 6,144 and 8,192, `r05` −1.69, −4.13 and
  −8.99 at 4,096, 6,144 and 8,192, all gains; depth 0 negligible for both (the first timed load `ca` ran a slow first
  request there, 103.08 against `cb`'s 94.93, as in R175). At 8,192 pass 1 compares other replies: the arms' reply
  accepts more drafts (185 of 210 against 173 of 245) and, over the whole load, meets fewer tokens new to the draft's
  shortlist (21 for `n10` and 17 for `r05` against the preset's 27), so part of its 9 ms is the text.
- **The sampled request** (the preset's sampler, a fixed seed): `n10` −2.25 and −4.81 ms at 6,144 and 8,192, `r05`
  −4.02 and −4.84, gains; at 0 and 4,096 negligible for both against the control's spreads of 10.09 and 11.91 ms
  (`ca` and `cb` each ran one slow request).
- **Generation over all depths** (the lane's own table, [table.md](table.md)): `n10` 31.67 and 31.33 tok/s, `r05`
  31.69 and 31.70, against the preset's 29.62 and 30.06 (+5.6 % and +6.2 %, gains). At 4,096 tokens `r05` is a
  −1.0 % "loss" in tok/s (31.46 and 31.47 against 31.76 and 31.79) though its step is shorter: its other reply accepts
  175 of 237 drafts against 177 of 231, so it needs two more steps.

**The best arm is `r05`**: a gain at 4,096, 6,144 and 8,192 tokens, where `n10` gains only from 6,144, and not slower
at depth 0. It went to the agent check.

### Checks

- **Replies** ([replies.tsv](replies.tsv), R69's [replies.py](../2026-11-20_r69_sparse_switch/run/replies.py)):
  every reply is the same token ids in both loads of each arm, and every prime the same in all six. Where an arm
  changes the kernel, its greedy reply differs from the preset's: at 8,192 both 5-split arms after 2 equal tokens
  (" response" for " list"), the same reply in both; at 4,096 `r05` after 114. At 6,144 all three arms
  write the same greedy reply. 5 splits merge each list's partial sums in another order than 3, and the dense and
  sparse kernels sum in different orders too, so a near-tie can turn; R69 saw the same when the switch point moved.
  The sampled replies differ between the arms after 0 to 19 equal tokens at 4,096 to 8,192.
- **Prompt reading** ([table.md](table.md)): negligible at 4,096 for both arms; at 6,144 `n10` 280.8 and 281.4 tok/s
  against 278.0 and 271.8 (+2.3 %), `r05` 218.6 and 271.5 (negligible, its spread 53.0); at 8,192 `n10` 233.1 and
  241.6 against 234.2 and 237.5 (negligible), `r05` 226.2 and 231.3 (−3.0 %, a loss by the lane's rule). A prime
  reads 4,612 tokens on 3,516 cached in two calls, of 4,096 tokens (micro-batches of 512) and 516 (micro-batches of
  64: `--ubatch-taper short=1400:64`, which makes no micro-batch under 9 tokens). `n10` and `r05` differ only in
  `_MIN_RATIO`, which acts on calls of 8 tokens or fewer, so they run every call of that prime on the same kernels
  with the same splits, and `r05`'s slower primes are not read as the switch's; R69 and R175 saw primes scatter the
  same way. The agent
  check reads prompt time per task.
- **Memory and health:** every load's lowest host MemAvailable 176 to 423 MiB, no RAM watchdog stop, 0 graph
  reallocations in all 100 requests; every prime drafted nothing. The card log ([cards.md](cards.md)): median busy SM
  clock 2,100 to 2,151 MHz, memory 4,455 to 4,657 MHz; the kit's watchdog never tripped.

### What ran

- **Rig and settings:** rig 2, nine P106-090 cards with that rig's soaked clock profile set on the morning's boot
  (memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206
  +282 +256 +231 MHz on cards 0 to 8, the 75 W default power limit, the kit's watchdog running:
  [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), "The profile again on the boot of 2026-11-26"). The
  offsets read the same before and after, with no watchdog trip and no GPU Xid; every run of this folder was on that
  one boot.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it (md5 `b8e950f2de8c7906c156d582416731bf`: the sparse ratio 0.75, `--ubatch-taper short=1400:64`): UkisAI's
  Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`; the third shard is its
  n-gram table) and the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a step. Context 65,536.
- **Build:** one for every arm, fork `pascal-p106-v5-cand-b12674` at `eee615cdb` (upstream b12674 merged), the build
  C839, C912 and R229 ran; `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`.
- **The arms:** the run script drops the preset's `_NSM` and `_MIN_RATIO` and each arm sets both, so the three server
  lines differ in those values only; each server's switch line printed them.
- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 11:04:01–11:47:55.
  - **Loads:** a warm-up load of `c` (untimed), then `ca n10a r05a r05b n10b cb`, each a fresh server start; the
    second round runs the arms in reverse order (Q1253). Every load read the files in 224 to 227 s.
  - **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
  - **Requests:** at 0, 4,096, 6,144 and 8,192 tokens each load read the prompt (a `prime` request, from 4,096 on),
    ran 256 tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed
    seed. The prompts are the same ids on every load, cut from this project's recorded agent text.

## The agent check

### Result

Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and
JavaScript alternating, a 12-minute cap a task, a 40-minute budget), the preset (`c`) against `r05`, four runs
alternated. Each value is one run; the class is the release check's rule (a gain or a loss only beyond the larger of
the two arms' spreads; [class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv), [class_cells.tsv](class_cells.tsv)).

| | `ca` (preset) | `r05a` | `cb` (preset) | `r05b` | `r05` against `c` |
|---|---|---|---|---|---|
| passed | 11 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | |
| minutes a task at equal work (R90 `v3`'s work) | 2.29 | 2.23 | 2.23 | 2.21 | −1.8 %, negligible |
| the same without JavaScript `complex-numbers` | 2.34 | 2.31 | 2.33 | 2.30 | −1.3 %, **gain** |
| generation tok/s | 29.05 | 31.54 | 31.56 | 32.04 | +4.9 %, negligible |
| the same without `complex-numbers` | 30.22 | 31.48 | 31.54 | 32.05 | +2.9 %, negligible |
| prompt tok/s, the whole session | 161.6 | 158.9 | 154.5 | 158.6 | +0.4 %, negligible |
| ms a decode step, all requests | 109.42 | 104.05 | 107.10 | 104.26 | −3.8 %, **gain** |
| the same, middle context under 4,102 cells (the same kernels in both arms) | 105.93 | 104.59 | 103.85 | 103.93 | −0.6 %, negligible |
| the same, 4,102 to 6,153 cells (only `r05` on the sparse kernel) | 107.75 | 103.62 | 105.72 | 104.05 | −2.7 %, **gain** |
| the same, 6,153 to 8,204 cells (both on it, 3 splits against 5) | 109.80 | 105.64 | 108.61 | 103.48 | −4.3 %, **gain** |
| the same, 8,204 to 16,000 cells | 108.97 | 105.39 | 108.49 | 106.19 | −2.7 %, **gain** |
| requests in those three buckets | 25 / 9 / 12 | 29 / 9 / 3 | 30 / 14 / 1 | 31 / 10 / 10 | |
| first token median s, all turns | 2.36 | 2.77 | 2.18 | 2.58 | +17.8 %, loss (the turn mix, below) |
| first token median s, the tasks' first turns | 4.94 | 4.68 | 4.65 | 4.81 | −1.1 %, negligible |
| first token median s; p90 s, later turns | 2.12; 8.15 | 1.90; 7.97 | 1.90; 7.94 | 2.04; 7.90 | −2.0 %; −1.4 %, negligible |
| the 10 first turns that read the same prompt on the same cache, prompt ms | 66,906.0 | 67,137.5 | 67,616.0 | 66,460.8 | |
| draft acceptance; tokens a decode step | 0.725; 3.17 | 0.759; 3.28 | 0.792; 3.38 | 0.779; 3.34 | |
| turns; tool calls malformed | 85; 0 | 63; 0 | 71; 0 | 73; 0 | |
| minutes, all 12 tasks | 35.3 | 14.8 | 22.1 | 26.1 | |
| graph reallocations | 0 in 84 requests | 0 in 63 | 0 in 71 | 0 in 73 | |
| lowest host MemAvailable MiB; highest swap MiB | 294; 997 | 316; 398 | 228; 52 | 224; 52 | |

The step buckets place each request by the context at its middle ([steps_cells.md](steps_cells.md),
[run/steps_cells.py](run/steps_cells.py)); a request that crosses a cut counts in one bucket, and above 8,204 cells
two runs hold 1 and 3 requests. [steps.md](steps.md) has the same in R203's buckets of 4,000 tokens.

- **Where the switches act the step is 2.9 to 4.6 ms shorter**, the size the deep lane measured, and below 4,102
  cells, where both arms run the dense kernels, it is the same: the change does what the lane said inside real tasks.
- **The task time moves less than the step** because the controls differ: run 1's JavaScript `complex-numbers` ran to
  its 12-minute cap in 25 turns (the other three runs passed it in 5 to 7), as R220's control did on 2026-11-20, so
  the preset's two runs are 0.06 minutes a task apart over all tasks. Without that task they are 0.01 apart and the
  arm is 0.03 minutes (1.8 s) a task faster, a gain.
- **The first-token median over all turns is the turn mix.** Every run has 12 first turns, the shortest 2.77 to 2.83 s
  (they read 403 to 2,780 new tokens), and the arm's runs had 51 and 61 later turns against 73 and 59. With fewer later
  turns the median of all turns lands on the first turns' floor: in `r05a` it is 2.77 s, the shortest first turn.
  Each part alone is negligible (first turns 4.68 and 4.81 s against 4.94 and 4.65; later turns 1.90 and 2.04 against
  2.12 and 1.90), the prompt reading per new token is the same in every size bucket
  ([prompt_buckets.tsv](prompt_buckets.tsv)), and the matched first turns took the same time. `_MIN_RATIO` acts only on
  calls of 8 tokens or fewer, and the taper makes no prompt micro-batch under 9 tokens.
- **Answers sound:** 12 of 12 in both arm runs, 0 malformed calls, no reply read again, 0 graph reallocations; the
  replies differ from the preset's, as the lane's did, so each run's work is priced against R90 `v3`'s
  ([work.md](work.md)). The report's gates are all GO except memory (NO-GO on the low MemAvailable), as in every
  Flash-Next run of this preset; the RAM watchdog stopped nothing.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy SM medians 2,088 to 2,151 MHz per card, the same profile in all four
  runs; the kit's watchdog never tripped and the boot logged no GPU error.

### What ran

| # | Run | Arm | Launched | Server up after | Tasks | Driver |
|---|---|---|---|---|---|---|
| 1 | `ca` | the v5 preset (`_NSM=5 _MIN_RATIO=0.75`) | 11:48:53 | 227 s | 11:52:53–12:28:22 | exit 0 at 12:28:34 |
| 2 | `r05a` | `_NSM=10 _MIN_RATIO=0.5` | 12:28:50 | 226 s | 12:32:49–12:47:45 | exit 0 at 12:47:56 |
| 3 | `cb` | the v5 preset | 12:48:59 | 226 s | 12:52:59–13:15:12 | exit 0 at 13:15:20 |
| 4 | `r05b` | `_NSM=10 _MIN_RATIO=0.5` | 13:16:01 | 226 s | 13:20:01–13:46:15 | exit 0 at 13:46:22 |

The same rig, clock profile, boot and build as the deep lane, which ran right before. The machine that drives the rig
is the agent's client.

**How the preset is started.** [run/flash_nsm.sh](run/flash_nsm.sh) starts it as C912 and R229 did: C854's launcher
([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh)), R156's driver
([r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh)), the harness's arm `v4r`, then the report, with R156's `v4r`
line made into the v5 preset (text reuse added to `V4R_ENV`, `GAINS_ENV`'s ratio 1.0 set to 0.75, the taper
`short=1400:64`). Both switches live in `GAINS_ENV`, so the arm replaces their two values there and nothing else.
- The preset's two runs started a 1,717-byte command, md5 `e88afbaa6d856ae489378b407852abc3`: C912's candidate run and
  R229's control byte for byte.
- The arm's two runs started a 1,717-byte command, md5 `b65c090f6b98f8446cf19f2ece272fba`, which differs from the
  preset's in `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10` and `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5` only.
- Each started command is its dry line ([run/dry_line.sh](run/dry_line.sh)) byte for byte and passes
  [run/preset_check.sh](run/preset_check.sh) against `settings.env` with its two values (40 switches, 30 flags).

## Files

| File | What |
|---|---|
| [lane_steps.md](lane_steps.md), [lane_steps.tsv](lane_steps.tsv) | the deep lane's verdict table: the step of every load per depth and pass type, both class rules, per arm |
| [table.md](table.md), [classes.tsv](classes.tsv), [classes_pass1.tsv](classes_pass1.tsv), [classes_pass2.tsv](classes_pass2.tsv) | the lane's own tables and classes (on the load, pass 1 and pass 2); `table.md` is the rig's byte for byte |
| [replies.tsv](replies.tsv) | per lane request and load, the reply's token-id and text md5 and the first token where it differs from `ca`'s |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the lane: one line per request (deep_client.py) and per load (its start, load time, full server line) |
| [cards.md](cards.md) | the lane's card log summary (`scripts/card_table.py`) |
| [agent.tsv](agent.tsv), [agent_nocap.tsv](agent_nocap.tsv), [capped.tsv](capped.tsv) | the agent check per run, with and without the task that ran to its cap in a run |
| [class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv), [class_cells.tsv](class_cells.tsv) | the arm's class against the preset: the session's measures, without the capped task, and the step per bucket and the first-token wait per part |
| [steps_cells.md](steps_cells.md), [steps.md](steps.md) | ms a decode step per bucket of context: where the switches act, and R203's buckets |
| [same_prompts.md](same_prompts.md), [prompt_tasks.tsv](prompt_tasks.tsv), [prompt_buckets.tsv](prompt_buckets.tsv), [first_token.tsv](first_token.tsv) | prompt reading: the requests that read the same prompt on the same cache in every run, seconds per task, per size of the new tokens, and the first-token waits |
| [work.md](work.md), [work.tsv](work.tsv) | the work each run read and generated, priced against R90 `v3`'s |
| [clocks.tsv](clocks.tsv), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv), [slot_gap.tsv](slot_gap.tsv), [grammar.tsv](grammar.tsv), [taper_lines.tsv](taper_lines.tsv) | busy clocks, host memory and swap, graph reallocations, the slot-to-processing gap, the tool-call grammar's samples and the taper each server read |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv), [text_reuse_points.tsv](text_reuse_points.tsv) | reply re-reads (none) and where text reuse kept the cache |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report for each run |
| [agent/](agent/) | each run's `turns.csv` and `attempts.csv` (the harness's arm renamed to the run) |
| [run/c1175.sh](run/c1175.sh) | the deep lane's run script (R175's `run/r175.sh` with the split count and the switch point as the arms) |
| [run/flash_nsm.sh](run/flash_nsm.sh), [run/dry_line.sh](run/dry_line.sh), [run/preset_check.sh](run/preset_check.sh) | the agent runs' script, the server line it would start, and its check against `settings.env` |
| [run/tables_agent.sh](run/tables_agent.sh), [run/steps_cells.py](run/steps_cells.py), [run/class_cells.py](run/class_cells.py) | the agent tables (R229's `run/tables.sh` without its shortlist tables), the step per bucket, and its class |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours
RIG=rig WORK=/tmp/c1175 BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-26_c1175_nsm10_rig2/run/c1175.sh
# the same with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
python3 -I bench/deep_lane/deep_table.py OUT --control c --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass1 --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass2 --tsv classes_pass2.tsv
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm n10 --tsv steps_n10.tsv > steps_n10.md
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm r05 --tsv steps_r05.tsv > steps_r05.md
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py OUT --control-load ca --loads ca,n10a,r05a,r05b,n10b,cb > replies.tsv
python3 -I scripts/card_table.py OUT/cards.csv > cards.md

# the agent check, one run at a time, in the order ca, r05a, cb, r05b, each with its own OUT and CLIENT_OUT
ARM=ca RIG=rig CLIENT=local MODEL=<first shard on the rig> DRAFT=<MTP draft on the rig> NEW=<build folder on the rig> \
  REF=<v4.0 build folder on the rig> RIGDIR=<folder on the rig> OUT=<folder> CLIENT_DIR=<bench on the client> \
  CLIENT_OUT=<results on the client> PREFIX=<container prefix> \
  bash docs/data/2026-11-26_c1175_nsm10_rig2/run/flash_nsm.sh
ARM=ca WORK=<scratch> NEW=<build folder on the rig> bash docs/data/2026-11-26_c1175_nsm10_rig2/run/dry_line.sh > line_ca.txt
bash docs/data/2026-11-26_c1175_nsm10_rig2/run/preset_check.sh <started command of ca>            # r05: ... 10 0.5
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <CLIENT_OUT>/res agent/ca v4r ca
bash docs/data/2026-11-26_c1175_nsm10_rig2/run/tables_agent.sh TABLES ca r05a cb r05b
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here.
The agent check's tasks are the bench's own ([bench/agent_check](../../../bench/agent_check/README.md)). The model is
UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
