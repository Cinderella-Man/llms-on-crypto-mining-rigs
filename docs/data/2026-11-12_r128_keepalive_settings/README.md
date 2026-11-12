# The keep-alive's first-call waits without code, and rung `f`'s switches apart (R128, Q1419, 2026-11-12)

What this is: six served loads on nine P106-090 cards with Miner Llama `pascal-p106-v4` `d097d6d88` (the `-kern`
tarball), on [R122](../2026-11-12_r122_host_ladder/README.md)'s driver, preset and two requests. They test whether a
setting removes the waits that [FN85](../2026-11-12_fn85_server_gap/README.md) found. There, a decode or verify graph's
first call waited about one second for its card's keep-alive block (`GGML_CUDA_KEEPALIVE=decode`) to time out. The loads
also tell which of rung `f`'s three switches removes the first request's two card-8 waits, and whether the waits come
from the host profiler. The code fix is [C811](../2026-11-12_c811_keepalive_wait/README.md)'s
`GGML_CUDA_KEEPALIVE_EAGER=1`. Registry [Q1419](../../../quants/experiments.md).

## The answer

- **`GGML_CUDA_KEEPALIVE_MS=250` shortens every wait to the shorter timeout, and costs no memory.** The waits left in
  the replies take 283.8 ms (card 0, first reply) and 265.7 ms (card 8, second reply) instead of about 1,000. C811
  leaves none.
  - **First request:** prompt 4,519.7 ms and reply 10,094.1 ms, against rung `f`'s 7,522.2 and 7,545.3, and 10,856.6
    and 10,824.7. That is about what C811 gave in R131 on its own build (4,726.8 and 4,771.3; 10,091.0 and 10,108.4).
  - **2,760-token prompt:** 24,722.0 ms, against `f`'s 26,436.8 and 26,304.4, and C811's 28,311.1 and 28,267.0.
  - **Memory:** each card's memory is `f`'s to the MiB. The lowest MemAvailable was 529 MiB, against 519 and 510.
  - **The step:** 112.28 ms against `f`'s 111.89 and 111.99. That is +0.34 against a spread of 0.10, so a loss by the
    mean of the medians. Its greedy steps paired with `f`'s are −0.29 ± 0.13 ms, and the greedy reply is the same text.
- **`CUDA_MODULE_LOADING=EAGER` removes every wait, but a 4 GB host cannot carry it.**
  - **Card memory:** each card holds 174 to 176 MiB more (all nine together 46,179 MiB against 44,595 and 44,597). The
    fullest card has 811 MiB free instead of 987.
  - **Host memory:** 20 s into the load, MemAvailable was 217 MiB against 1,506 in `fka250`. The lowest was 79 MiB.
    Swap went from 20 to 1,825 MiB, with 424,074 pages swapped out and 26,069 major faults.
  - **Speed:** the first request's prompt took 29,483.2 ms and its reply 19,088.3 ms, with a median step of 214.93 ms.
- **C771 (`LLAMA_MTP_EMBD_FIRST=1`) is the switch that removes the first request's two card-8 waits.** With C771 added
  to rung `e`, the first reply has 1,702.0 ms outside the profiled steps, as `f` has (1,692.4 and 1,668.3), against
  `e`'s 3,622.2 and 3,634.4. C744 alone and C742 alone keep both waits: 1,007.5 + 1,003.4 and 1,007.4 + 1,003.4 ms.
- **The generation's waits are not the profiler's.** Without `GGML_CUDA_PROFILE_HOST` the first reply took 10,777.3 ms.
  That is 79.3 and 47.4 ms less than profiled `f`, and 2,034.6 and 2,053.4 ms less than `e` (the row expected about
  1.9 s if the waits are real). If the profiler made them, card 0's ~1 s wait would be gone too, est. ~9.8 s. The greedy
  reply is `f`'s text.
- **One finding the row did not ask for: the long prompt is about 4 s shorter when either the profiler or lazy
  loading is off.**
  - 22,074.1 ms without the profiler (4,362.8 and 4,230.4 ms less than `f`'s 26,436.8 and 26,304.4) and 21,836.7 with
    `EAGER` (4,600.1 and 4,467.7 less).
  - The first 756 tokens take 7.06 s and 6.80 s, against 7.17 and 7.17, so the difference comes after the first chunk.
  - In `f` the 244-token second pass of that chunk costs card 0 1,390.2 and 1,389.7 ms of GPU time, against 631.7 at
    the 250 ms timeout and 425.8 with `EAGER`. That is est. a lazy load inside the pass waiting for card 0's block.
    `fn85.py` does not count it, because the pass's own work follows the wait.
  - Profiled loads with lazy loading (all of R122's, R131's and this row's but `feager` and `fnoprof`) therefore carry
    est. 4 s more in a long prompt. FN87 (Q1428) places it. R131's +1,851.1 and +1,819.5 ms for C811 on a long prompt
    (Q1426) were measured that way too.

## The loads

Every load is R122's preset ([run/preset.env](../2026-11-12_r122_host_ladder/run/preset.env)) through R108's
driver: the n-gram table dropped from the page cache, S37's greedy request (79 prompt tokens, 256 generated,
temperature 0), then the first-turn request (2,760 prompt tokens, 256 generated, temperature 1.0). Rung `e` is the
preset with C615, C738, C619, C740 and C741; rung `f` adds C771, C744 and C742 (R122's ladder). The reference loads
are R122's own `e`, `e-2`, `f` and `f-2`: the same rig, build, preset and requests, run at 12:26 to 13:35 that day.
This row ran 19:22:01 to 19:52:04.

| Load | Switches | First prompt, ms | First reply, ms | Outside its steps, ms | Second prompt, ms | Second reply, ms | Outside, ms | Step, ms | Lowest MemAvailable, MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e` (R122) | rung `e` | 8,516.7 | 12,811.9 | 3,622.2 | 26,477.0 | 12,043.5 | 2,266.1 | 112.44 | 505 |
| `e-2` (R122) | rung `e` | 8,518.5 | 12,830.8 | 3,634.4 | 26,499.0 | 12,717.2 | 2,266.5 | 112.64 | 513 |
| `f` (R122) | rung `f` | 7,522.2 | 10,856.6 | 1,692.4 | 26,436.8 | 10,624.5 | 1,244.2 | 111.89 | 519 |
| `f-2` (R122) | rung `f` | 7,545.3 | 10,824.7 | 1,668.3 | 26,304.4 | 12,863.7 | 2,260.9 | 111.99 | 510 |
| `fka250` | `f` + `GGML_CUDA_KEEPALIVE_MS=250` | 4,519.7 | 10,094.1 | 954.0 | 24,722.0 | 9,535.1 | 509.2 | 112.28 | 529 |
| `feager` | `f` + `CUDA_MODULE_LOADING=EAGER` | 29,483.2 | 19,088.3 | 2,282.6 | 21,836.7 | 10,259.1 | 320.8 | 165.70 | 79 |
| `e771` | `e` + C771 `LLAMA_MTP_EMBD_FIRST=1` | 7,549.3 | 10,869.2 | 1,702.0 | 26,306.0 | 11,217.2 | 1,261.0 | 112.31 | 520 |
| `e744` | `e` + C744 `LLAMA_MTP_DRAFT_CAP=1` | 8,494.2 | 12,835.2 | 3,673.3 | 26,388.2 | 10,985.0 | 1,251.8 | 112.20 | 523 |
| `e742` | `e` + C742 `LLAMA_MTP_SKIP_LAST_ROW=1` | 8,537.6 | 12,801.8 | 3,657.0 | 26,302.9 | 10,946.2 | 1,274.1 | 112.05 | 506 |
| `fnoprof` | `f`, no host profile | 7,464.5 | 10,777.3 | (no profile) | 22,074.1 | 11,867.0 | (no profile) | (no profile) | 510 |

- **Prompt and reply** are the server's own timings (`timings.tsv`). **Outside its steps** is FN85's: the reply's
  generation time less its profiled steps after the first (`summary.md`; `fka250` read at a 250 ms timeout,
  `ka250/summary.md`). **Step** is S37's mean of the two requests' median typical step (`arms_f.md`, `arms_e.md`).
- **The second reply** samples at temperature 1.0, so its text, length in steps and time differ from load to load. Its
  time outside the steps is the fair comparison there.
- **The greedy reply** is the same text in all ten loads: 256 generated, 254 drafted, 170 accepted.
- **No graph was reallocated** in any request (`reallocs.tsv`). C615's and C738's closing lines are in every load
  (`switch_lines.tsv`).

## The waits per load

FN85's `fn85.py` finds each evaluation that ended 1,000 to 1,060 ms after its card's last beat, i.e. one that waited for
the block to end by its timeout. For `fka250` it was run with `--timeout 250 --min-wait 100` (250 to 310 ms).

| Load | First request: card 8, first drafting pass | Card 0, last verify pass | Card 8, last catch-up | Second request: card 8, first drafting passes | Card 4, last step |
|---|---:|---:|---:|---:|---:|
| `e` | 1,007.6 | 1,034.6 | 1,003.5 | 1,007.9 | 1,022.3 |
| `e-2` | 1,007.6 | 1,033.6 | 1,003.4 | 1,007.7 | 1,019.0 |
| `f` | — | 1,034.0 | — | 1,004.1 | — |
| `f-2` | — | 1,032.9 | — | 1,003.5 | 1,022.3 |
| `fka250` | — | 283.8 | — | 265.7 | — |
| `feager` | — | — | — | — | — |
| `e771` | — | 1,033.1 | — | 1,012.9 | — |
| `e744` | 1,007.5 | 1,033.1 | 1,003.4 | 1,007.9 | — |
| `e742` | 1,007.4 | 1,033.4 | 1,003.4 | 1,008.1 | — |

- **The card-4 wait** comes only when the draft's one heavier evaluation in the second reply runs at another size
  (FN85). That depends on the sampled reply, not on the load (`card4.md`).
- **In the prompts:** every load but `feager` has one card-8 draft wait in each prompt (1,020.4 to 1,025.3 ms; 270.5
  and 275.2 in `fka250`). `e`, `e744` and `e742` have a second one in the first prompt (1,003.3 to 1,004.2 ms), which
  C771 removes.
- **At a 250 ms timeout two of the long prompt's ordinary passes also end in the window** (card 6's second and card 0's
  fifth, with 552.1 and 655.1 ms of GPU time). The same passes show in R122's `f` read the same way, so they are not
  counted here.

## Memory with `CUDA_MODULE_LOADING=EAGER`

Each card's largest `memory.used` in the load, MiB, from `rig_server.sh`'s `gpu.csv` (every 10 s); each card has
6,144 MiB.

| Load | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | All nine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f` (R122) | 5,095 | 5,141 | 4,999 | 5,157 | 4,907 | 5,015 | 4,415 | 5,027 | 4,839 | 44,595 |
| `fka250` | 5,095 | 5,141 | 4,999 | 5,157 | 4,907 | 5,015 | 4,415 | 5,027 | 4,839 | 44,595 |
| `feager` | 5,271 | 5,317 | 5,175 | 5,333 | 5,083 | 5,191 | 4,591 | 5,203 | 5,015 | 46,179 |

- **Card memory:** `EAGER` loads every kernel of the build onto every card when its context is made: 176 MiB more on
  each card (174 on card 8 against `f-2`).
- **Host memory:** the driver also holds more of it while it loads. 20 s into the load, MemAvailable was 217 MiB, against
  1,506 MiB at the same point of `fka250`. From about 220 s in, near the end of the load (the server was up at 236 s),
  the host swapped: swap rose from 20 MiB to a peak of 1,825 MiB, and the lowest MemAvailable was 79 MiB, at 244.1 s,
  during the first request. The RAM watchdog (`MIN_AVAIL_MB=60`) did not act.
- **The first request** then ran from swap: prompt 29,483.2 ms and reply 19,088.3 ms, with the draft's span at a
  121.01 ms median against `f`'s 16.00. **The second request** ran at `f`'s speed: step median 116.48 against 115.68
  and 116.06.

## The long prompt, pass by pass

`prompt_passes.md` lists the second request's six prompt passes, each card's GPU time in each pass. The 2,760 tokens
run as a 756-token chunk (a 512-token pass and a 244-token pass), then 2,004 tokens.

| Load | Card 0, second pass, ms | All cards, second pass, ms | All cards, all six passes, ms | First 756 tokens (server), s | Whole prompt, ms |
|---|---:|---:|---:|---:|---:|
| `e` (R122) | 1,389.9 | 5,882.3 | 32,218.1 | 7.17 | 26,477.0 |
| `f` (R122) | 1,390.2 | 5,884.7 | 32,172.4 | 7.17 | 26,436.8 |
| `f-2` (R122) | 1,389.7 | 5,888.6 | 32,032.1 | 7.17 | 26,304.4 |
| `fka250` | 631.7 | 4,630.8 | 30,500.9 | 6.41 | 24,722.0 |
| `feager` | 425.8 | 3,209.7 | 29,104.6 | 6.80 | 21,836.7 |
| `fnoprof` | (no profile) | | | 7.06 | 22,074.1 |

- **Card 0's second pass:** in `f` it takes 758.5 ms more than in `fka250`, about the difference between the two
  timeouts (1,000 − 250 ms). With `EAGER` no kernel is loaded at run time and the pass takes 425.8 ms. So the pass, the
  first at 244 tokens on a card, est. waits there for card 0's block, inside its own GPU time, where `fn85.py` cannot see
  it.
- **`fnoprof` and `feager`:** both are about 4 s shorter than `f` (4,362.8 and 4,230.4 ms; 4,600.1 and 4,467.7), and
  both spend about `f`'s time on the first 756 tokens. Est. the cost comes where lazy loading meets the profiler: at each
  evaluation's entry the profiler synchronizes on the card's previous evaluation, so the host stops queueing ahead
  while a lazy load waits. FN87 (Q1428) reads where that time goes.

## What it changes

- **C811 stays the fix for the waits.** `GGML_CUDA_KEEPALIVE_EAGER=1` is already in `v4c2` (R131). `EAGER` loading is
  not an option on the 4 GB host.
- **`GGML_CUDA_KEEPALIVE_MS=250` is a candidate setting of its own** (R137, Q1429): alone or together with C811, on the
  `-c811` build, with same-day reference loads and unprofiled loads. It shortened the long prompt where C811 lengthened
  it (C817, Q1426), and costs no memory. Its +0.34 ms step against this morning's references needs a same-day reference
  before it is classed.
- **C771 is the switch for the first request's card-8 waits.** C744 and C742 are not, though each alone on rung `e`
  reads as a gain here: e744 −0.34 and e742 −0.48 ms against `e`'s 112.44 and 112.64 (spread 0.20). `e771` reads −0.23.
- **Served timings measured with the host profile include est. 4 s of the long prompt that unprofiled servers do not
  pay** (FN87). The agent check (`bench/agent_check`) does not set the profile, so its numbers do not carry it.

## Files

| File | What |
|---|---|
| `tables/summary.md`, `tables/ka250/summary.md` | FN85's two summary tables: each request's time outside the steps, wait by wait |
| `tables/fn85.md`, `tables/requests.tsv`, `tables/waits.tsv`, `tables/ka250/` | `fn85.py`'s full output: every request, every wait |
| `tables/timings.tsv` | the server's timings of every reply |
| `tables/arms_f.md`, `tables/arms_e.md` (+ `.tsv`, `_class.tsv`, `_paired.tsv`) | S37's `arms.py`: `fka250` and `feager` against `f` and `f-2`; `e771`, `e744` and `e742` against `e` and `e-2` |
| `tables/split_<load>.md`, `tables/steps_<load>.tsv` | S37's `step_split.py` per profiled load |
| `tables/gpu_mem.tsv` | each card's largest memory.used per load |
| `tables/vm.tsv` | the lowest MemAvailable, swap, major faults, pages swapped in and out per load |
| `tables/prompt_passes.md`, `tables/prompt_progress.tsv` | the second request's prompt, pass by pass; the server's progress line |
| `tables/card4.md`, `tables/timeline/` | the draft's heavier evaluations on card 4; the first request of each new load, evaluation by evaluation |
| `tables/switch_lines.tsv`, `tables/reallocs.tsv` | C615's and C738's closing lines; graph reallocations per request |
| `run/r128_loads.sh` | the driver's loads (runs R108's `rig_loads.sh` unchanged) |
| `run/r128_tables.sh` | every table above, on any CPU |
| `run/prompt_passes.py` | the long prompt pass by pass |

## Commands

On the rig, detached, the rig idle (paths through the environment, as R122):

```bash
cd $D && BIN=$BUILD/bin MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf TABLE=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf \
  PRESET=$D/docs/data/2026-11-12_r122_host_ladder/run/preset.env RIG_SERVER=$D/bench/agent_check/rig_server.sh \
  VM_SAMPLER=$D/docs/data/2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
  RIG_LOADS=$D/docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh \
  REQ_GREEDY=$S37/docs/data/2026-11-04_draft_one_card/run/greedy_request.json REQ_LONG=$S37/long_request.json \
  OUT=$D/out setsid nohup bash docs/data/2026-11-12_r128_keepalive_settings/run/r128_loads.sh > $D/console.out 2>&1 &
```

The tables, on a build machine's CPU, with R122's `out/` as the reference:

```bash
REPO=<this repository> REF_OUT=<R122's out/> nice -n 19 bash docs/data/2026-11-12_r128_keepalive_settings/run/r128_tables.sh out tables
```

`r128_tables.sh` calls FN85's `fn85.py` with `--timeout 250 --min-wait 100` for `fka250`. The two options were added to
`fn85.py` for this row; their defaults (1,000 and 300 ms) are FN85's.

## Provenance and limits

- **Build:** `pascal-p106-v4` `d097d6d88`, the rig tarball `-kern` (R122's), CUDA 12.9, driver 580. Model: Swift
  Qwen3.8-Flash-Next GSQ-RCO Q2_0 with the MTP draft. Requests: S37's (`docs/data/2026-11-04_draft_one_card/run/`).
- **The reference loads are R122's,** run about six and a half hours earlier on the same rig. No same-day `f` load ran,
  so the step classes against them are approximate (the row had six loads).
- **`fnoprof` has no profile,** so its waits are inferred from the server's timings only.
- **One run per load.**
- **Licence:** the tables are this repository's; the requests are S37's.
