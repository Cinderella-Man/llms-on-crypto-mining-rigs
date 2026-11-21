# The first request at a new depth with the keep-alive's new key (R195; Q1562; 2026-11-17)

**In plain words:** on Flash-Next's released preset, an agent turn's first request at a new context length runs slower
than the same request repeated, because the cards drop their clock before a few passes. C866's switch value
`GGML_CUDA_KEEPALIVE_EAGER=3` cuts most of that: at 4,096 and 6,144 tokens the first request now generates 97.88 and
107.15 ms faster than with the released setting, the replies are the same, and nothing waits a second. One pass at
the end of each reply still drops the clock: the 256-token limit leaves room for only part of a draft, so the main
model's last pass is two tokens long at 4,096 and one at 6,144, and each of those widths runs there for the first time
since the warm-up, with new kernel picks ([C878](../2026-11-21_c878_keepalive_why_new/README.md)). The first request is
still 227.23, 301.06 and 306.84 ms slower than its repeat at 4,096, 6,144 and 8,192 tokens.

What this measured: R192 ([2026-11-17_r192_first_request_graphs](../2026-11-17_r192_first_request_graphs/README.md))
found that a first request at a new depth runs three passes kernel by kernel at about twice the repeat's GPU time,
because the keep-alive switch `GGML_CUDA_KEEPALIVE_EAGER=1` (C811) ends each card's clock-holding block before an
evaluation at a shape new to the card, and the cache's length is part of the shape. R191
([2026-11-17_r191_keepalive_stop](../2026-11-17_r191_keepalive_stop/README.md)) showed that without the stop those
passes cost what the repeat's cost, but the MTP draft's first evaluation at the new depth then waits about 1 s. C866
([2026-11-17_c866_keepalive_key](../2026-11-17_c866_keepalive_key/README.md)) added `=3`: the key without the cache's
length but with what the length picks (which attention and top-k kernels run), so the draft's case still stops. This is
its rig check, on the v5 build that holds it, with both values in two loads on the same build. Nine P106-090 cards,
UkisAI's Swift-1.5 GSQ-RCO Q2_0 file with its MTP draft, the v4 preset otherwise unchanged. Registry row Q1562 in
[quants/experiments.md](../../../quants/experiments.md).

Every value below is one measured request; each arm is one load.

## Result

**`=3` keeps the clock where only the cache's length is new, and still stops it where a kernel pick changes.** At
6,144 tokens, where no pick changes after the 4,096 pair, the first request's three slow passes now cost what the
repeat's cost (15.23 ms more GPU time in all, against 238.63 with `=1`). At 4,096 and 8,192 one pass a depth still
stops, as C866 planned: the prompt's last 4 tokens at 4,096 (the indexer's top-k for 4 rows changes path) and the
first verify pass at 8,192 (the sparse attention starts taking 4-token passes). The draft's first evaluations still
stop, and none waited: the slowest took 10.06 ms.

**The last pass:** the main model's last pass of each reply (kernel by kernel) still stops on all nine cards at 4,096
and 6,144: 187.51 and 187.11 ms of GPU time (with `=1` 186.94 and 186.17; R191 at 4,096 without any stop on v4.0,
70.26). The 256-token limit makes it shorter than a verify pass: two tokens at 4,096 (one token and one draft), one at
6,144 (no room for a draft). Each of those two graphs had run only at the end of the warm-up request, at 93 and 94
cells, where the attention and the indexer's top-k take other kernels, so at `=3` both keys are new by design
([C878](../2026-11-21_c878_keepalive_why_new/README.md), Q1585). That pass, and the draft's own stops, are most of what
is left of the gap.

### Generation

The server's generation time of each request (256 tokens, temperature 0), the first request against its repeat
([summary.md](summary.md)):

| Arm | Depth | First, ms | Repeat, ms | First − repeat | Steps, first / repeat | Paired steps: first − repeat | of which steps 0 to 2 | steps 3 on | Block stops, first / repeat |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `=1` (released) | 4,096 | 8,965.94 | 8,635.74 | +330.21 | 77 / 77 | +330.26 | +138.81 | +191.45 | 30 / 0 |
| `=1` | 6,144 | 8,461.49 | 8,018.89 | +442.60 | 69 / 68 | +238.82 | +143.41 | +95.41 | 31 / 0 |
| `=1` | 8,192 | 8,255.25 | 7,948.18 | +307.08 | 69 / 68 | +189.68 | +132.54 | +57.15 | 21 / 0 |
| `=3` (C866) | 4,096 | 8,868.06 | 8,640.83 | +227.23 | 77 / 77 | +228.78 | +29.56 | +199.22 | 22 / 0 |
| `=3` | 6,144 | 8,354.34 | 8,053.28 | +301.06 | 69 / 68 | +96.44 | +33.79 | +62.65 | 13 / 0 |
| `=3` | 8,192 | 8,263.42 | 7,956.58 | +306.84 | 69 / 68 | +188.42 | +135.73 | +52.69 | 11 / 0 |

- **The first request with `=3` against `=1`:** 4,096 −97.88 ms (8,868.06 against 8,965.94), 6,144 −107.15
  (8,354.34 against 8,461.49), 8,192 +8.17 (8,263.42 against 8,255.25). The repeats, where nothing stops, moved by
  +5.09, +34.39 and +8.40 ms between the two loads. So `=3` is a **gain** at 4,096 and 6,144 and **negligible** at
  8,192, where the first verify pass stops in both arms.
- **The replies are the same:** each depth's first request and repeat give the same text in both arms (the same md5,
  [requests.csv](requests.csv)).
- **At 6,144 and 8,192 the first request takes one step more than its repeat** (69 against 68: 204 tokens drafted and
  186 accepted, against 201 and 187), in both arms, so the server's difference holds one unpaired step. The paired steps
  (R121's [first_request.py](../2026-11-17_r121_flash_step_map/run/first_request.py)) compare like with like: at
  6,144, +96.44 ms with `=3` against +238.82 with `=1`.
- **Block stops per request**, from the server's own count ([tasks-e1.md](tasks-e1.md), [tasks-k1.md](tasks-k1.md)):
  with `=3`, 22 at 4,096, 13 at 6,144 and 11 at 8,192 (est. from the pass tables: the prompt's tail, the last pass
  and the first verify pass on 9 cards each, as marked below, and 4, 4 and 2 of the draft's evaluations); none in any repeat.

### The passes

How each pass ran on cards 0 to 8 (E kernel by kernel, C captured, R replayed) and its GPU time summed over the nine
cards, ms. The passes are paired by what they read ([run/summary.py](run/summary.py)):

| Arm | Depth | Prompt's last 4 tokens, first | repeat | First verify, first | repeat | Second verify, first | repeat | The three, first − repeat | First's last pass (2 tokens at 4,096, 1 at 6,144) |
|---|---:|---|---|---|---|---|---|---:|---|
| `=1` | 4,096 | E 237.57 | E 97.75 | E 221.89 | E 101.53 | C 99.00 | C 93.65 | +265.53 | E 186.94 |
| `=1` | 6,144 | E 220.00 | E 103.57 | E 224.90 | E 107.78 | C 104.99 | C 99.91 | +238.63 | E 186.17 |
| `=1` | 8,192 | E 223.17 | E 106.71 | E 227.96 | E 105.30 | C 102.53 | C 97.11 | +244.54 | none kernel by kernel |
| `=3` | 4,096 | E 240.17 | E 97.97 | **E 114.64** | E 101.64 | C 99.40 | C 93.64 | +160.96 | E 187.51 |
| `=3` | 6,144 | **E 107.68** | E 104.97 | **E 117.33** | E 108.54 | C 105.35 | C 101.62 | **+15.23** | E 187.11 |
| `=3` | 8,192 | **E 113.22** | E 106.77 | E 227.31 | E 105.30 | C 102.77 | C 97.17 | +134.06 | none kernel by kernel |

E and C stand for all nine cards. Bold: the passes `=3` no longer stops.

- **Where `=3` keeps the block,** the pass costs what the repeat's costs: the first verify pass at 4,096 114.64 ms
  against 101.64, the prompt's tail at 6,144 107.68 against 104.97.
- **Where a pick changes,** it stops as `=1` does: the prompt's tail at 4,096 (240.17 against `=1`'s 237.57) and the
  first verify pass at 8,192 (227.31 against 227.96). Each is once a server: the next pass of the same kind keeps its
  block (the first verify pass at 4,096 after the tail's stop).
- **The last pass:** at 4,096 and 6,144 the first request ends with a pass shorter than a verify pass, run kernel by
  kernel: two tokens at 4,096 (one draft: the 256-token limit leaves two tokens) and one token at 6,144 (it leaves
  one), at 186.94 and 186.17 ms with `=1` and 187.51 and 187.11 with `=3` (which pass ran on which graph:
  [C878's tables](../2026-11-21_c878_keepalive_why_new/README.md)). The repeat's is never written by the profiler (it
  writes an evaluation when its graph is next evaluated); the repeat ends with a pass of the same width and ran it
  kernel by kernel too, without a stop (est. from its 27 kernel-by-kernel evaluations and 0 stops). Both stops are by
  design: the two-token and the one-token graphs had run only at the end of the warm-up request, at 93 and 94 cells
  (64 pools), where the indexer's top-k runs its bitonic kernel and the sparse attention does not take them; at 4,349
  and 6,398 cells (1,088 and 1,600 pools) the top-k runs another kernel for one or two rows and the sparse attention
  takes one token (from 2,051 cells), so their picks are new on every card. At 8,192 the reply ends with a full verify pass, and no shorter pass runs.
- **No wait of about 1 s:** the slowest of the draft's evaluations that ran kernel by kernel or were captured took
  10.06 ms (`=3`, 4,096), and no paired main pass of a first request took more than 142.21 ms over its repeat's
  ([passes-k1-d4096.md](passes-k1-d4096.md) and the other pass tables).

### Also seen, not traced

- **The first request's 60-token prompt micro-batch at 4,096 is slow:** captured on all nine cards, 1,962.81 ms of GPU
  time with `=1` and 1,637.05 with `=3`, against 1,176.78 and 1,167.62 at 6,144. The first request's prompt took
  2,282.73 and 1,950.50 ms at 4,096, against 1,303.22 and 1,289.43 at 6,144. Est. the first launch of the sparse
  attention's prompt kernels (9 to 128 tokens from 2,051 cells): the prime's micro-batches were larger, so this is the
  first such batch past the threshold in the server. That is what C868 (Q1567) would load at the server's start.
- **The prompt's tail at 8,192 took 109.95 ms less GPU time with `=3`,** but the first request's prompt took the same
  time in both arms (1,351.44 against 1,351.89 ms), and so did its steps 0 to 2 (+135.73 against +132.54 over the
  repeat). At 6,144 the tail's and the first verify pass's 219.89 ms less GPU time show as 109.62 ms less in steps 0
  to 2. The GPU times are summed over the nine cards, which run one after another; how much of a card's evaluation
  time is waiting for the card before it is not separated.

## What ran

- **The engine:** Miner Llama v5 (`pascal-p106-v5` at `1f1110c67`: v4.0 with C861, C865 and C866 merged, each off by
  default, and upstream llama.cpp b12410 merged), the build whose P106 check passed the same day
  ([2026-11-17_v5_cand_p106_check](../2026-11-17_v5_cand_p106_check/README.md)); `llama-server` md5
  `5b134f43f3a3e885bfba02538ffebcbe`, checked on the rig before the run.
- **The line:** C866's runner with a second timed load ([run/r195.sh](run/r195.sh)): the deep lane at three depths in
  one server, in order 4,096, 6,144 and 8,192 tokens. Per depth: the prime (64 tokens short of the depth: it restores
  the load's 60-token checkpoint and reads the rest), then the first request (the whole prompt; it reads the last 64 tokens on the live cache, as
  60 and 4 with `--ctx-checkpoints-incall`) and its repeat (it restores a checkpoint four tokens before the end and
  reads 4), each 256 tokens at temperature 0 with the MTP draft, four idle seconds before every request.
- **Why 6,144:** on this file two picks change right at 4,096 and 8,192 (the indexer's top-k for 4 and 5 rows past
  1,024 pools; the sparse attention for 4 tokens past 4 × 2,051 = 8,204 cells), so a first request there is new on its
  cards at `=3` too, by design. At 6,144, after the 4,096 pair, no pick changes.
- **Loads:** `warm` (the preset, depth 0, untimed), then `e1` (`=1`) and `k1` (`=3`), each with R192's logging
  (`GGML_CUDA_GRAPH_DEBUG=1` with `-lv 5`, the graph-mode profile `GGML_CUDA_PROFILE` with
  `GGML_CUDA_PROFILE_MODE=graph`, R121's host profile and step trace) and the arm's value after the preset's `=1` (env
  keeps the last value). The two timed server lines differ in that one word ([started_lines.diff](started_lines.diff),
  the log files' names made equal); the server logged "GGML_CUDA_KEEPALIVE_EAGER=1: …" and "GGML_CUDA_KEEPALIVE_EAGER=3:
  …" at its start ([loads.tsv](loads.tsv)).
- **When:** 2026-11-17, `warm` 21:03:40 to 21:09:35, `e1` 21:09:35 to 21:18:28, `k1` 21:18:28 to 21:27:21 (the server
  up after 309, 307 and 307 s). The rig's cards at their stock clocks (no offset, no power setting). Nothing else ran on the
  rig, and the only contact during the loads was the lane's own `wait` call (a test for its end marker every 20 s).
  Every card held 0 MiB before each load and after the last; the host's MemAvailable stayed at 352 MiB or more, swap
  50 MiB or less, no stop by the RAM watchdog.

## How it was read

- R192's scripts, unchanged: [graph_passes.py](../2026-11-17_r192_first_request_graphs/run/graph_passes.py) (passes
  from the graph-mode profile) and [server_graph_log.py](../2026-11-17_r192_first_request_graphs/run/server_graph_log.py)
  (debug lines and keep-alive counts per request); R121's
  [first_request.py](../2026-11-17_r121_flash_step_map/run/first_request.py) with R86's
  [align_t0.py](../2026-11-11_r86_step_trace/run/align_t0.py) (steps and the host profile).
- [run/split_gp.py](run/split_gp.py), new: cuts each load's profile per depth (the first request and its repeat), as
  `graph_passes.py` compares the first two generation requests it finds ([split-e1.txt](split-e1.txt),
  [split-k1.txt](split-k1.txt)). Checked on R192's published profile: the same pass table.
- [run/summary.py](run/summary.py), new: the tables above. With several depths in one load the profile also holds the
  first request's 60-token prompt micro-batch (its graph is evaluated again at the next depth; in R192's one-depth load
  it never was), so the first request has one main pass more at its start: the passes are paired by graph, not by
  order.
- [run/tables.sh](run/tables.sh) runs them all. The tables ran on the operator's machine after the run, with
  `python3 -I`.

## Commands

From the repository's root, on the machine that drives the rig (it runs only ssh and tar):

```bash
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v5 llama-server on the rig> \
  MODEL=<Swift Q2_0 shard 1 on the rig> MODEL_DRAFT=<the MTP draft on the rig> \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  bash docs/data/2026-11-17_r195_first_request_fix/run/r195.sh            # STEP=start; DRY=1 prints the lines
STEP=wait RIG=<rig alias> WORK=<run folder> bash docs/data/2026-11-17_r195_first_request_fix/run/r195.sh
STEP=collect OUT=<folder here> RIG=<rig alias> WORK=<run folder> \
  bash docs/data/2026-11-17_r195_first_request_fix/run/r195.sh
```

The tables (on any machine with Python 3):

```bash
S=<this repository> OUT=<the collected results> RES=<tables folder> \
  bash docs/data/2026-11-17_r195_first_request_fix/run/tables.sh
```

## Files

| File | What |
|---|---|
| [summary.md](summary.md), [summary.tsv](summary.tsv) | the answer: per arm and depth, generation, steps, stops, the paired passes, the draft's slowest evaluation, the replies |
| `passes-<load>-d<depth>.md`, `.tsv` (e.g. [passes-k1-d6144.md](passes-k1-d6144.md)) | every main pass of the depth's first request and repeat, in profile order: how it ran on each card, its GPU time; the draft's kernel-by-kernel and captured evaluations (`e1` = `=1`, `k1` = `=3`) |
| [tasks-e1.md](tasks-e1.md), [tasks-k1.md](tasks-k1.md) and `.tsv` | per request: the backend's debug lines by kind, llama.cpp's graph reuses, the keep-alive's counts |
| [first-e1.md](first-e1.md), [first-k1.md](first-k1.md), `steps-*.tsv`, `pairs-*.tsv`, `align-*.md` | R121's comparison of each pair step by step, from the step trace and the host profile |
| [split-e1.txt](split-e1.txt), [split-k1.txt](split-k1.txt) | the profile's pieces and which are the generation requests |
| [gp-e1.csv.gz](gp-e1.csv.gz), [gp-k1.csv.gz](gp-k1.csv.gz) | the graph-mode profiles as written: one line per evaluation and card (`how`, `graph`, `eval_us`) |
| [requests.csv](requests.csv) | every request of the three loads: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | each load's environment and server line as it ran (paths as `<OUT>`, `<BIN>`) |
| [started_lines.diff](started_lines.diff) | the two timed server lines, one word a line |
| [run/](run/) | `r195.sh` (the run), `tables.sh`, `split_gp.py`, `summary.py` |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v5 at `1f1110c67`. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0) with its MTP draft file; the prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)), R192's and R121's. The tables hold times, counts, graph
decisions and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
