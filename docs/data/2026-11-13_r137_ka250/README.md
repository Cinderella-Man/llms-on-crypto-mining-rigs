# The keep-alive's 250 ms timeout as a setting, alone and with C811's switch (R137, Q1429, 2026-11-13)

What this is: nine served loads on nine P106-090 cards with Miner Llama `pascal-p106-v4-b12246-c817` `5f8dd491b` (the
`-c817` tarball: [C811](../2026-11-12_c811_keepalive_wait/README.md)'s `GGML_CUDA_KEEPALIVE_EAGER` plus
[C817](../2026-11-12_c817_keepalive_held/README.md)'s fix). They use [R122](../2026-11-12_r122_host_ladder/README.md)'s
rung `f`, preset and two requests. [R128](../2026-11-12_r128_keepalive_settings/README.md) found that
`GGML_CUDA_KEEPALIVE_MS=250` shortens the keep-alive block's first-call waits to about 250 ms. This run asks two things:
does the shorter timeout still help once C811's switch removes the waits, and which timeout goes into the stacked line
with the switch on? Registry [Q1429](../../../quants/experiments.md).

## The answer

- **With `GGML_CUDA_KEEPALIVE_EAGER=1`, keep the default timeout (1,000 ms, `_MS` unset).** At 250 ms nothing
  measured gets faster: both prompts, the greedy reply and the memory are the same. Of the two loads, the step is a
  loss in one and negligible in the other. Neither timeout leaves a wait for a keep-alive block anywhere.
- **The 250 ms timeout without the switch** (`_EAGER=0`, R128's setting) keeps short waits: 283.6 and 253.8 ms in the first load's replies, and 283.3, 253.9 and 264.4 in the second's. Its
  first prompt is 4,468.3 and 4,430.4 ms, against 4,287.7 and 4,022.4 with the switch. **Its 2,760-token prompt is the
  shortest:** 24,429.8 and 24,486.4 ms against 24,793.3 and 24,751.8. Unprofiled, 20,422.2 against 20,652.7.
  - Each full 512-token chunk takes less GPU time without the switch: pass 3 adds up to 5,013.6 ms over the nine
    cards, against 5,238.3 with it (`tables/prompt_passes.md`). The timeout makes no difference there: `f1ka250` has
    5,222.1.
  - So the switch costs the long prompt time in its full chunks: est. 314.4 ms profiled (the mean of `f1` and `f1-2` against the mean of `f0ka250` and `f0ka250-2`) and 230.6 ms unprofiled. This is new: Q1461 (C829).

## The loads

All on rung `f`: R122's `preset.env` plus `LLAMA_DRAFT_PREFETCH=1`, `LLAMA_VERIFY_PREPARE=2`,
`LLAMA_MTP_GRAPH_KEEP=1`, `LLAMA_KV_SEQ_RM_BOUND=1`, `LLAMA_SERVER_YIELD=2`, `LLAMA_MTP_EMBD_FIRST=1`,
`LLAMA_MTP_DRAFT_CAP=1`, `LLAMA_MTP_SKIP_LAST_ROW=1` (the preset sets `GGML_CUDA_KEEPALIVE=decode`). Run order: `f1`,
`f1ka250`, `f0ka250`, the three `np` loads, then the three `-2` loads, 09:44:12 to 10:27:02.

| load | `GGML_CUDA_KEEPALIVE_EAGER` | `GGML_CUDA_KEEPALIVE_MS` | host profile |
|---|---:|---:|---|
| `f1`, `f1-2` | 1 | 1000 (the default; the reference) | on |
| `f1ka250`, `f1ka250-2` | 1 | 250 | on |
| `f0ka250`, `f0ka250-2` | 0 | 250 | on |
| `f1np`, `f1ka250np`, `f0ka250np` | as above | as above | off (`GGML_CUDA_PROFILE_HOST` set empty; FN87) |

## The server's timings (ms; `tables/timings.tsv`)

| load | first prompt (79 tokens) | first reply's generation (256 tokens, greedy) | 2,760-token prompt | second reply's generation (tokens) |
|---|---:|---:|---:|---:|
| `f1` | 4,287.7 | 10,044.5 | 24,793.3 | 9,859.0 (256) |
| `f1-2` | 4,022.4 | 10,095.4 | 24,751.8 | 10,650.8 (256) |
| `f1np` | 3,923.7 | 10,035.3 | 20,652.7 | 10,482.6 (256) |
| `f1ka250` | 3,980.7 | 10,111.1 | 24,763.4 | 9,925.8 (256) |
| `f1ka250-2` | 4,030.6 | 10,104.1 | 24,782.8 | 10,654.2 (256) |
| `f1ka250np` | 3,952.0 | 10,053.0 | 20,657.3 | 10,451.3 (256) |
| `f0ka250` | 4,468.3 | 10,064.2 | 24,429.8 | 2,407.5 (61) |
| `f0ka250-2` | 4,430.4 | 10,064.6 | 24,486.4 | 10,715.2 (256) |
| `f0ka250np` | 4,454.4 | 9,948.0 | 20,422.2 | 11,091.1 (256) |

- The first reply is the same text in all nine loads, with 170 of 254 drafts accepted. The second request samples at
  temperature 1.0, so its replies differ; `f0ka250`'s ended after 61 tokens.
- No graph was reallocated in any load (`tables/realloc_marks.tsv`).
- Card memory was the same in every load, to the MiB: 44,595 MiB over the nine cards (`tables/gpu_mem.tsv`). The
  lowest MemAvailable was 521, 490 and 494 MiB in `f1`, `f1-2` and `f1np`. In `f1ka250`, `f1ka250-2` and
  `f1ka250np` it was 516, 493 and 493; in `f0ka250`, `f0ka250-2` and `f0ka250np`, 503, 466 and 498. Swap stayed at
  5 MiB throughout (`tables/vm.tsv`).
- R128's values for comparison, on the `-kern` build. Its references `f` and `f-2`: first prompt 7,522.2 and 7,545.3 ms,
  2,760-token prompt 26,436.8 and 26,304.4. Its `fka250` (`_EAGER` not in that build): 4,519.7 and 24,722.0. On this
  build at `_EAGER=0` and 1,000 ms, R140's `f0` and `f0-2` had 7,742.09 and 7,467.70, and 26,637.95 and 26,065.24.
- The host profile lengthens the 2,760-token prompt (FN87). `f1`: 24,793.3 and 24,751.8 profiled, 20,652.7 not;
  `f1ka250`: 24,763.4 and 24,782.8 against 20,657.3; `f0ka250`: 24,429.8 and 24,486.4 against 20,422.2.

## The step (S37's `arms.py`, against `f1` and `f1-2`; `tables/arms.md`, `tables/arms_class.tsv`)

The step is in ms: the median per request, then the mean of the two medians.

| load | greedy | long | mean of the medians | minus the references' mean | class (spread 0.072) |
|---|---:|---:|---:|---:|---|
| `f1` | 107.66 | 115.30 | 111.48 | (reference) | |
| `f1-2` | 107.38 | 115.73 | 111.55 | (reference) | |
| `f1ka250` | 107.61 | 116.46 | 112.04 | +0.517 | loss |
| `f1ka250-2` | 107.31 | 115.64 | 111.48 | −0.039 | negligible |
| `f0ka250` | 107.48 | 114.75 (14 steps) | 111.12 | −0.402 | gain |
| `f0ka250-2` | 107.41 | 115.95 | 111.68 | +0.160 | loss |

The greedy steps, paired step by step with `f1`'s (same text, same draft counts, 78 steps): `f1ka250` −0.183 ± 0.216 ms,
`f1ka250-2` −0.308 ± 0.190, and `f1-2` itself −0.375 ± 0.187 (`tables/arms_paired.tsv`). The greedy medians are
negligible in every load. The loads differ only in the long request, whose reply is sampled. **`_MS=250` with the
switch is not a gain.** By the mean of its two loads (111.76) it reads as a loss, +0.239 against a spread of 0.072.

## The waits for a keep-alive block

FN85's `fn85.py` ran at the 1,000 ms timeout over `f1` and `f1-2` (`tables/ka1000/`). Over the four `_MS=250` loads it
ran with `--timeout 250 --min-wait 100`, as R128 did (`tables/ka250/`). The server's own count of blocks that ended by
the timeout comes from the `keep-alive =` line after each reply, in the raw server logs.

| load | waits `fn85.py` lists (prompt / generation) | waits that are a block's | blocks ended by the timeout after the first and second reply |
|---|---|---|---|
| `f1`, `f1-2` | 0 / 0, 0 / 0 | none | 0 and 18, 0 and 18 (`f1np` 0 and 9) |
| `f1ka250`, `f1ka250-2` | 3 / 1, 2 / 1 | none | 0 and 19, 0 and 21 (`f1ka250np` 0 and 10) |
| `f0ka250`, `f0ka250-2` | 5 / 2, 5 / 3 | in generation 283.6 (card 0, the first reply's last verify pass) and 253.8 (card 8, the second reply's first drafting pass); 283.3, 253.9 and 264.4 (card 4, the last step's draft) | 40 and 109, 40 and 114 (`f0ka250np` 40 and 101) |

**At a 250 ms timeout, `fn85.py`'s rule also catches evaluations and idles that are not waits.** The rule counts an
evaluation that ends 250 to 310 ms after its card's last beat. That fits any long evaluation and any host idle of that
length. Checked one by one (`tables/ka250/waits.tsv`, `tables/prompt_passes.md`):

- **With the switch:** the one wait listed in generation (card 4, 229.1 and 226.9 ms) is a 0.04 ms evaluation after a
  229.1 ms host idle. `f1` has the same idle after its last verify pass, 228.4 and 231.4 ms (`tables/ka1000/summary.md`).
- The prompt entries are main passes whose GPU time matches `f1`'s:
  - card 0, pass 5: 651.7 and 649.7 ms, against 652.8 and 652.7;
  - card 8, pass 3: 275.9 and 283.7, against 286.9 and 285.4;
  - card 4, pass 2: 344.3, against 343.3 and 345.7.
- **Without the switch,** these are a block's waits:
  - card 8's draft passes in both prompts, 270.4 and 270.3 ms in the first, 275.2 and 275.2 in the second (R140's
    `f0`: about 1,025 each);
  - pass 2 of the long prompt: 629.9 GPU ms on card 0 against `f1`'s 445.2, and 551.0 and 552.5 on card 6 against
    349.8;
  - the generation waits in the table.

## Files

| file | what |
|---|---|
| `run/r137_loads.sh` | the nine loads (R128's driver with R137's `LOAD_ENV`; runs R108's `rig_loads.sh` unchanged) |
| `run/r137_tables.sh` | every table below, from the rig's `out/` folder, on any CPU |
| `tables/timings.tsv` | the server's timings per reply, with an md5 of each reply's text |
| `tables/arms.md`, `arms.tsv`, `arms_class.tsv`, `arms_paired.tsv` | S37's step per load, its class against `f1` and `f1-2`, the greedy steps paired |
| `tables/split_<load>.md`, `steps_<load>.tsv` | S37's step split per profiled load |
| `tables/ka1000/`, `tables/ka250/` | FN85's `fn85.py` (`fn85.md`, `requests.tsv`, `waits.tsv`) and `summary.py` (`summary.md`) |
| `tables/prompt_passes.md`, `prompt_progress.tsv` | the 2,760-token prompt pass by pass (R128's `prompt_passes.py`), and the server's progress lines |
| `tables/timeline/<load>_greedy.md` | FN85's timeline of the first reply per profiled load |
| `tables/vm.tsv`, `gpu_mem.tsv` | the lowest MemAvailable, swap and paging per load; each card's largest memory.used |
| `tables/reallocs.tsv`, `realloc_marks.tsv`, `switch_lines.tsv`, `status.txt` | graph reallocations, the switches' closing lines, the driver's log |

`fn85.py` crashed when no load in its set had a wait, as here with `f1` and `f1-2`, because it took `waits.tsv`'s
header from the first wait. It now writes the header alone (`docs/data/2026-11-12_fn85_server_gap/run/fn85.py`,
`WAIT_KEYS`). Nothing else changed in its output.

## Commands

On the rig, from a folder with this repository's files (paths are the rig's):

```bash
BIN=<the -c817 build's bin/> MODEL=<Swift Q2_0 first shard> DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> \
  TABLE=<the shard with the n-gram table> PRESET=docs/data/2026-11-12_r122_host_ladder/run/preset.env \
  RIG_SERVER=bench/agent_check/rig_server.sh VM_SAMPLER=docs/data/2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
  RIG_LOADS=docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh REQ_GREEDY=<S37's greedy_request.json> \
  REQ_LONG=<S37's long_request.json> OUT=out setsid nohup bash docs/data/2026-11-13_r137_ka250/run/r137_loads.sh &
```

Then, on any CPU (here the quant machine at `nice -n 19`, no container):

```bash
REPO=<a checkout of this repository> nice -n 19 bash docs/data/2026-11-13_r137_ka250/run/r137_tables.sh out tables
```

Provenance: the requests are S37's ([2026-11-04_draft_one_card](../2026-11-04_draft_one_card/README.md) for the greedy
one). The model is Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 with its MTP draft. The tables are this run's own output.
Licence as the repository.
