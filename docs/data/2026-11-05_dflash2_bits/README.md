# C528 / Q1017: the DFlash2 drafter at 4 and 5 bits on the rig (2026-11-05)

Flash-Next's DFlash2 line drafts with Inco AI's DFlash2 drafter in Q8_0 (652 MB), on the last of nine P106-090 cards.
A drafter only proposes tokens, so a coarser copy of it keeps the output exact and is judged by tokens per step against
the time and memory it saves ([SpecPilot note](../../research/notes/specpilot.md) §4.4, registry
[Q1017](../../../quants/experiments.md)). This folder puts the publisher's Q4_K_M (362 MB) and Q5_K_M (432 MB), both
made from the BF16 original, against the Q8_0 on the line [C134r](../2026-11-04_adaptive_draft/README.md) ran on the
second rig: Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0, fork branch `pascal-p106-v3-b12076-ladder3` (`7bc1552f4`),
`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, `-b 512`, `--spec-draft-n-max 2`.

One llama-server load per drafter, in the order Q8_0, Q4_K_M, Q5_K_M, Q8_0 (the two Q8_0 loads give the spread), and
the same five greedy requests to each, sent from the rig itself:
- the agent check's first-turn request (the first bugfix task's, 2,760 tokens, 512 generated);
- Q561's greedy smoke request (79 tokens, 384 generated);
- three windows of ~6,000 bytes of held-out coding-agent text ([C271](../2026-11-04_kreal_p106/README.md)'s), each
  continued for 384 tokens (2,363–2,728 prompt tokens).

This is the single-request part: the speed this project reports is the agent check's, and that arm runs when its client
is available (the command is under Commands).

## Result

**Q5_K_M is a gain: 0.8 % less time a token, the same tokens per step, 210 MiB free on the drafter's card. Q4_K_M is
negligible: its draft pass is the shortest, but it accepts 1.3 % fewer tokens a step, which spends the saving.** Every
drafter gave the same greedy text on all five requests.

| Drafter | File | Tokens per step | Acceptance | Draft pass on the agent text, base / boost clock | Time a token (base steps) | Against Q8_0 | Drafter's card once loaded | Host low point, load / requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 (load 1) | 652 MB | 2.600 | 0.800 | 23.1 / 19.0 ms | 65.69 ms | | 3,307 MiB | 729 / 319 MiB |
| Q8_0 (load 4) | | 2.600 | 0.800 | 23.1 / 18.9 ms | 65.80 ms | spread 0.11 ms (0.17 %) | 3,307 MiB | 708 / 303 MiB |
| **Q5_K_M** | 432 MB | 2.597 | 0.798 | **20.9 / 17.8 ms** | **65.22 ms** | **−0.53 ms (−0.8 %): gain** | **3,097 MiB (−210)** | 673 / 301 MiB |
| Q4_K_M | 362 MB | 2.567 | 0.784 | 19.7 / 16.7 ms | 65.71 ms | −0.04 ms (−0.05 %): negligible | 3,031 MiB (−276) | 710 / 293 MiB |

- **The draft pass** (the drafter's GPU time a step, from the host profile, on the agent-text requests): Q5_K_M saves
  2.2 ms at base clock and 1.1 ms at boost, Q4_K_M 3.4 and 2.2 ms; the two Q8_0 loads agree to 0.1 ms. The step
  shortens by about as much (base-clock steps on the agent text: Q8_0 178.0–180.9 ms, Q5_K_M 175.8–178.8, Q4_K_M
  175.2–178.1), and the verify pass is the same on every load (128–129 ms on the agent text).
- **Tokens per step** are deterministic here (greedy requests, the same text): both Q8_0 loads accepted 1,256 of 1,570
  drafted tokens. Q5_K_M accepted 1,255 of 1,572, Q4_K_M 1,246 of 1,590: Q4_K_M's misses fall mostly on the smoke
  request (2.379 against 2.439 tokens a step) and the first agent-text window (2.329 against 2.388).
- **Time a token** is each request's median base-clock step over its tokens per step, summed over the five requests by
  their steps. Per request the Q8_0 loads are 0.05–0.54 ms apart. Q5_K_M is below both of them on four requests (by
  0.1–1.3 ms) and 0.16 ms above their mean on the first agent-text window. Q4_K_M is below both on two requests, above
  both on two and between them on the first turn.
- **Memory:** the drafter's card holds 210 MiB (Q5_K_M) and 276 MiB (Q4_K_M) less, once loaded and at its highest
  during the requests (3,463 MiB with Q8_0). The host does not gain: its lowest MemAvailable (CMA is 0 on this kernel
  line, so it is also the usable memory) comes in the load's last ~5 s, after the drafter's load starts, and is
  673–729 MiB whatever the drafter. During the requests it is 293–319 MiB, around the first-turn request (66–106 s
  after the load). The RAM watchdog (floor 60 MiB) never acted, and every load took 213–215 s.
- **Does it change anything?** Yes. The DFlash2 line takes Q5_K_M: est. +0.8 % decode on these requests, and 210 MiB
  on the last card toward a re-fit of the split (Q1062). Its agent-check arm `dfq5` runs when the client is available.
  Q4_K_M is the fallback when the last card's memory is the limit. A drafter that keeps Q8_0's acceptance at
  Q4_K_M's draft time (a mixed quant of the BF16 original, Q1060) is the next step on the drafter's bits.
  SpecPilot's estimate was at most 1.7–2.4 ms a draft step from the bytes alone; at base clock the saving is larger
  (3.4 ms for Q4_K_M), at boost it fits (2.2).

### The decode's clock, and why the raw tok/s is not the measure

The replies' own decode speeds (`decode_tok_s` in [requests.tsv](requests.tsv): Q8_0 14.09 and 14.32 tok/s over all
five requests, Q4_K_M 14.83, Q5_K_M 14.67) differ by up to 11 % between the two Q8_0 loads on a single request. Two
things that have nothing to do with the drafter set that spread:
- **Boost steps:** after each agent-text prompt the first 8–57 decoding steps run at ~147 ms instead of ~178 ms
  (verify 99–101 against 128 ms, draft 18.9 against 23.1 ms with Q8_0). The cards still hold the prompt's boost
  clock and drop to base a few seconds into the decode, as [C532](../2026-11-05_card_clocks/README.md) found. How many
  steps run at boost varies between loads (the two Q8_0 loads: 21/44/15 and 50/32/8 on the three windows).
- **Stalls:** 0–3 steps a request run 0.2–2.3 s longer than the median (11–13 s a Q8_0 load in total). The replies
  also report 1.3–2.5 s of checkpoint saves a request, the same on every load (within 0.07 s).

So the comparison uses steps at the same clock: the base-clock steps (between 0.9 and 1.5 times the request's
median), which are 64–100 % of each request's steps. The boost steps are also a measurement for
[Q1021](../../../quants/experiments.md) (holding the boost clock): on this line a step at boost is ~31 ms shorter,
~12 ms a token at 2.6 tokens a step.

## Per request

Time a token on base-clock steps (ms) and tokens per step:

| Request | Prompt / generated | Q8_0 (1) | Q8_0 (4) | Q5_K_M | Q4_K_M | Tokens per step Q8_0 / Q5_K_M / Q4_K_M |
|---|---:|---:|---:|---:|---:|---:|
| first turn | 2,760 / 512 | 64.86 | 65.40 | 64.75 | 65.08 | 2.642 / 2.642 / 2.629 |
| smoke (greedy) | 79 / 384 | 59.83 | 60.04 | 59.50 | 60.52 | 2.439 / 2.439 / 2.379 |
| agent text a | 2,363 / 384 | 75.79 | 75.56 | 75.83 | 76.46 | 2.388 / 2.358 / 2.329 |
| agent text b | 2,550 / 384 | 65.50 | 65.41 | 64.23 | 64.49 | 2.736 / 2.755 / 2.736 |
| agent text c | 2,728 / 384 | 62.79 | 62.74 | 61.97 | 62.22 | 2.837 / 2.837 / 2.816 |

## Files

| File | What it is |
|---|---|
| [arms.tsv](arms.tsv) | per load: the load's time, generated tokens and raw decode and prompt speeds, drafted and accepted tokens, tokens per step, the draft pass's GPU and host time and the verify pass's GPU time (medians over the steps), the step (median and mean), boost and slow steps, the draft pass at base and boost clock, the time a token (from the median, mean and base-clock step), texts equal to the first load's, the host's lowest MemAvailable and usable memory during the load and the requests, swap, watchdog kills, the drafter's card's memory, the share of 10 s clock samples at boost |
| [requests.tsv](requests.tsv) | the same per load and request, with the md5 of each reply's text |
| [requests_manifest.tsv](requests_manifest.tsv) | the five request bodies: endpoint, size, the agent-text byte range, md5 (the bodies are not published: see Provenance) |
| [run/launch.sh](run/launch.sh) | the run from the driving machine: the scripts and requests to the rig, then `rig_bits.sh` detached |
| [run/rig_bits.sh](run/rig_bits.sh) | on the rig: per drafter one load through [bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) (its monitors and RAM watchdog), the requests with 20 s idle before each, memory and VRAM after each, then the tables |
| [run/make_requests.py](run/make_requests.py) | the requests: the first-turn body made greedy, the smoke request, the agent-text windows |
| [run/bits_tables.py](run/bits_tables.py) | the two tables from the replies, the adapt log (one line per drafted step), the host profile (per card and graph evaluation: host entry and exit, GPU time) and the monitors |

## Commands

From the repository root, the drafters on the rig in one folder, the agent text on the rig:

```bash
RIG=rig W=<a work folder on the rig> BIN=<-ladder3 build>/bin/llama-server \
  MODEL=<Swift Q2_0 first shard> DRAFT_DIR=<the three drafter files> TEXT=<the agent text on the rig> \
  bash docs/data/2026-11-05_dflash2_bits/run/launch.sh
```

It builds the first-turn request on the rig (`bench/agent_check/first_turn_request.sh` with `BUILD_SSH="ssh $RIG"` and
`LONG_MAX_TOKENS=512`), writes the requests with `make_requests.py`, and starts `rig_bits.sh` detached; ~44 minutes for
four loads. When `W/out/DONE` exists, `W/out/tables/` holds the two tables. Each load runs

```
llama-server -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui -b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off \
  -lm dio --no-host --spec-type draft-dflash --spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99 \
  -md <drafter> -m <Swift Q2_0 first shard>
```

with `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 LLAMA_SPEC_ADAPT_LOG=<file> GGML_CUDA_PROFILE_HOST=<file>` (the adapt log
with its switch off only records the steps; the host profile adds two CUDA events per graph evaluation and no
synchronize). The requests ask for temperature 0 and top-k 1.

**The agent-check arm (next):** C134r's `fix2` line with Q5_K_M, through
[release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) with S18's machine values
([s18.env.example](../2026-11-03_flash_next_dflash2/run/s18.env.example)), reported against C134r's `fix2`:

```bash
L="-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host --spec-type draft-dflash --spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99"
MIN_AVAIL_MB=60 REL_BIN=<-ladder3 build>/bin/llama-server REL_ENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768" \
  ARGS_dfq5="$L -md <Q5_K_M drafter>" BUDGET_MIN_dfq5=40 STEPS=run ARMS=dfq5 \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>/agent-dfq5
```

## Provenance and licence

- The drafters: Inco AI's `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, revision `4f5a39be8f0a9e9a573ac9824251830a8f4096d9`,
  Apache-2.0: `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` (652,101,024 bytes, sha256 `541f4d6e…`), `-Q5_K_M.gguf`
  (431,920,544 bytes, sha256 `26693801…`), `-Q4_K_M.gguf` (362,124,704 bytes, sha256 `2f174e59…`); each checked on the
  rig against the publisher's sha256.
- The main model and the line: [2026-11-03_flash_next_dflash2](../2026-11-03_flash_next_dflash2/README.md) and
  [2026-11-04_adaptive_draft](../2026-11-04_adaptive_draft/README.md).
- The first-turn request is built from the agent check's polyglot bugfix tasks (task `bugfix-python-affine-cipher`) by
  `first_turn_request.sh`; the task text is not republished. The agent text is C271's file (the first 20,000 bytes of
  the held-out agent text the quality runs score on, md5 `312fd205dec2fba4879d4bf421f1e80e`), windows at bytes 0–5,992,
  6,045–12,040 and 12,003–17,995 (b and c share 37 bytes); it is not republished either.
- The second rig: 4 GB host, the `kho=off` kernel line, nine P106-090 at PCIe 1.0 x1, idle before each load (checked by
  the script: no other llama process, every card at 0 MiB).
