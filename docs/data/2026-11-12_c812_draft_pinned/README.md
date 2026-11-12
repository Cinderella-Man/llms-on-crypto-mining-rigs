# The MTP draft's table on its card without four copies of its inputs (C812, Q1420, 2026-11-12)

What this is: the cause of the 64.81 MiB of pinned host memory that the MTP draft's context took once its token table
moved to its layer's card ([R123b](../2026-11-12_r123b_prompt_ladder/README.md), Q1401 (b)), a change that gives it
back behind a new switch, `LLAMA_PP_MULTI_SPLIT` (off by default), and its checks on a build machine's CPU. It also
names what `-b 4096` adds on card 8, and the check on nine P106 cards, the rig row R135 (Q1421, below). Registry
[Q1420](../../../quants/experiments.md); switch list [docs/switches.md](../../switches.md).

## Result

**The 64.81 MiB is pipeline parallelism. With `-otd token_embd=CUDA4` out of the line, llama.cpp planned the MTP
draft's context with four scheduler copies of every graph input instead of one. The draft's graph is one split on
card 8, so the copies buy nothing. With `LLAMA_PP_MULTI_SPLIT=1` a one-split context gets one copy. On a CPU with two
devices laid out as the rig's cards, the draft's host compute buffer is then 0.63 MiB, as on the `-otd` line, against
2.52 MiB with the table on its layer's device. The tokens, probabilities, drafts and 406 verify steps are the same in
every arm.**

- **On the rig, est.:** 64.81 MiB of pinned host memory back (the draft's `CUDA_Host` buffer 86.41 MiB → 21.60), so the
  host's lowest MemAvailable on R43's replay est. 453 MiB against `embd`'s 388 (`base`: 470). Card 8's draft buffer
  est. 213.59 MiB against 278.39 (three copies of the 21.60 MiB of inputs fewer). The table stays on card 8.
- **`-b 4096`'s +160 MiB on card 8** is the second half of the target's hidden-row buffer, sized by `n_batch`: 160.08 MiB
  at `-b 2048`, 320.08 at `-b 4096`. With `-b 4096` R43's replay never used that half. New idea C813 (Q1422).
- **The checks pass on the build machine's CPU:** every switch off gives the base's logits on 146 of 146 random models,
  and so do the switch alone and the switch with C762's. ctest is 70 of 70, static and shared. `llama-server`'s GPU code
  is the base's. The rig tarball is built. The branch is merged into `pascal-p106-v4` as `f91533730`, and the merged
  tree's "every switch off" check passes (below).
- **On nine P106 cards (R135, Q1421) the switch does what it should, and it joins the stacked line beside
  `LLAMA_DRAFT_EMBD_DEV=layer`.** In one load of R43's replay with R123b's `embd` line, the draft's reserves have one
  copy (`sched copies = 1`, one split, the switch's line once). Its pinned host buffer is 21.60 MiB against `embd`'s
  86.41, and its card-8 buffer 195.49 MiB against 278.39. After loading, card 8 holds 5,461 MiB against 5,545 and
  the host has 929 MiB available against 844. The greedy replies are `base`'s on 62 of 62 turns in both passes. The
  host swapped 496 MiB of the server's memory from 289 s on, which `embd` did not (below), so the lowest
  MemAvailable, 494 MiB against 388, is not a like-for-like number. At the same moment of the replay, before the swap,
  the host had 69 MiB more available (494 against 425 MiB at 281 s).

## What R123b saw

In R123b's `embd` rung, `LLAMA_DRAFT_EMBD_DEV=layer` (C762) put the MTP draft's Q8_0 token table on card 8 beside its
layer, in place of the preset's `-otd token_embd=CUDA4`. The draft context's compute buffers then grew
([buffers.tsv](../2026-11-12_r123b_prompt_ladder/buffers.tsv)):

| Load | Draft: card 4 | Draft: card 8 | Draft: pinned host (`CUDA_Host`) | Host's lowest MemAvailable |
|---|---:|---:|---:|---:|
| `base` (`-otd token_embd=CUDA4`) | 5.00 MiB | 195.48 MiB | 21.60 MiB | 470 MiB |
| `embd` (`LLAMA_DRAFT_EMBD_DEV=layer`, no `-otd`) | — | 278.39 MiB | 86.41 MiB | 388 MiB |

## The cause: pipeline parallelism, which the `-otd` rule used to turn off

The reserve's own lines name it ([r123b_reserves.tsv](r123b_reserves.tsv), read from R123b's six server logs at
`-lv 5`). Every draft reserve of `base` and `base2` ends with `sched copies = 1`; every draft reserve of `embd`,
`sparse`, `queues` and `b4096` ends with `sched copies = 4`, after `llama_context: pipeline parallelism enabled`.

- **llama.cpp turns pipeline parallelism on** for a context whose model spans more than one device, with every layer
  offloaded, the layer split and the KV cache on the devices, and **no tensor override** (`-ot`, `-otd`). Under it the
  scheduler keeps `n_copies` = 4 copies of every graph input, so that the next micro-batch can start on the first
  devices while the last ones still compute. The copies sit in the input's own buffer (the host inputs in the pinned
  `CUDA_Host` buffer) and, for each split that reads the input, on that split's device.
- **The v4 line's draft had a tensor override**, `-otd token_embd=CUDA4`, so its context ran without pipeline
  parallelism: one copy of each input. C762's switch made the override unnecessary, and R123b took it out of the line.
  With no override left, the draft model (loaded with all nine cards in its device list, its one MTP layer on card 8)
  met every condition, and its context turned pipeline parallelism on.
- **The draft's graph is one split.** With the table on card 8, every node of the draft's graph runs there
  (`splits = 1` against 2 in `base`). There is no second device to overlap, so the copies buy nothing.
- **The numbers:** the draft's pinned host buffer holds the graph's inputs, 21.60 MiB with one copy. Their largest is the
  target's hidden rows, `mtp_h_input` [10,240 × 512] F32 = 20.00 MiB at the draft's micro-batch of 512
  (`--spec-draft-n-ubatch 512`). Four copies are 4 × 21.60 = 86.40 MiB; the log says 86.41 (21.60 is rounded). Card 8's
  draft buffer holds the copies of those inputs its split reads, four instead of one, and the gathered rows that card 4
  held before: 278.39 MiB against 195.48 and 5.00 on card 4.

So the 64.81 MiB are the scheduler's copies of the draft graph's inputs, sized by its micro-batch of 512. The `get_rows`
on card 8 adds none of them; the override's removal turned on pipeline parallelism, which multiplied them by four.

## What `-b 4096` adds on card 8: the second half of the hidden-row buffer

Card 8 held 5,705 MiB after loading with `-b 4096` against 5,545 with `-b 2048`, with no scheduler buffer changed. The
160 MiB is the target context's hidden-row buffer on its output card (`h_nextn_dev`, C693 and C703), which the MTP
draft reads its `mtp_h_input` rows from. It holds 10,240 F32 per row and two halves under
`LLAMA_MTP_CATCHUP_OVERLAP=1` (one for the call the draft is catching up on, one for the next call), each of
`n_batch` + `n_seq_max` rows:

| `-b` | Rows (2 halves × (`n_batch` + 1)) | Bytes | The log (`h_nextn_dev_alloc: CUDA8 h_nextn buffer size`) |
|---:|---:|---:|---:|
| 2048 | 4,098 | 167,854,080 | 160.08 MiB (two halves) |
| 4096 | 8,194 | 335,626,240 | 320.08 MiB (two halves) |

The difference is 167,772,160 B, exactly 160.00 MiB. With `-b 4096` none of R43's 62 prompts is cut into calls (R123b:
"0 catch-ups ran after the next call was queued"), so the second half held nothing in that replay. Making it only when a
prompt needs it is the new idea C813 (Q1422, below).

## The change

Fork branch `pascal-p106-v4-b12246-c812`, one commit `d679a6d9d` on `pascal-p106-v4` `e27947287`, host code only:
`src/llama-context.cpp`, 33 lines.

| `LLAMA_PP_MULTI_SPLIT` | What it does |
|---|---|
| unset or `0` | As before: a context turns pipeline parallelism on by llama.cpp's own conditions. |
| `1` | When a context would turn pipeline parallelism on, `sched_reserve` first splits its prompt-sized and its one-token graph without allocating anything. If each is one split, it makes the scheduler without pipeline parallelism (one copy of each input) and logs `sched_reserve: LLAMA_PP_MULTI_SPLIT: the graph runs as one split (pp 1, tg 1): planned without pipeline parallelism, one copy of each input`. A context whose graph has two splits or more (the target, ten on the rig) is planned as before. Once off, it stays off for that context: its weights do not move. |

- It follows the layout, not a card's name: on any card count the draft whose graph sits on one card gets one copy, and
  one that spans two cards (the `-otd` line) keeps whatever llama.cpp chose.
- No copy is allocated and then freed: the split runs before the first reserve.
- The draft's per-size schedulers (`LLAMA_MTP_GRAPH_KEEP`, batches of up to 8 tokens) never used pipeline parallelism;
  the switch changes the context's main scheduler, which runs the catch-up passes of 9 or more rows.
- With the switch the draft's catch-up passes copy their inputs as they did on the line before R123b (one copy, a wait
  for the previous micro-batch before the next one's inputs are copied). Est. no change in time: `base` ran its draft so.

### Ways out not taken

- **Name the draft's card instead** (`-devd CUDA8`): a draft model with one device in its list never meets llama.cpp's
  condition, and its context has one GPU backend instead of nine. It needs no code, but each card count needs its own
  line, while the switch follows the layout. It is kept as the new idea C814 (Q1423), for the nine backends' host cost.
- **Size the copies by a smaller micro-batch** (`--spec-draft-n-ubatch 128`, the value `r43_run.sh`'s own
  line holds before the line's 512): four copies of a 128-row batch are est. 21.6 MiB, but the catch-up then runs four
  passes where it ran one.

## Results on a CPU

The checks ran on a build machine's CPU (no GPU), each build and test in a container at a low CPU share, with one
build at a time ([checks.tsv](checks.tsv)).

### The draft's scheduler on two devices

[run/server_check.sh](run/server_check.sh) uses a random-weight `qwen4exp` model (Flash-Next's architecture) with one MTP
block, and that block alone as a separate draft file with its token table in Q8_0, as Flash-Next's MTP draft
(`test-llama-archs -o --mtp-draft`). Two `ggml-rpc-server`s on the same machine act as two devices. They advertise
asynchronous compute and events, so llama.cpp turns pipeline parallelism on for them as it does for the rig's cards.
Every layer is offloaded, so the draft's MTP layer is on the second device, as Flash-Next's is on card 8. Each arm
serves four requests: two greedy, and two at temperature 1.0 with the preset's top-k 20 and top-p 0.95. Every verify
step is dumped (`LLAMA_SPEC_DUMP`). The draft's reserves, from [server_reserves.tsv](server_reserves.tsv):

| Arm | The line | Draft splits | Draft `sched copies` | Draft host (CPU) buffer | Draft, second device | Draft, first device |
|---|---|---:|---:|---:|---:|---:|
| `otd` | `-otd token_embd=<first>` (the v4 line before R123b) | 2 | 1 | 0.63 MiB | 6.00 MiB | 0.06 MiB |
| `embd` | `LLAMA_DRAFT_EMBD_DEV=layer`, no `-otd` (R123b's `embd`) | 1 | 4 | 2.52 MiB | 8.40 MiB | — |
| `fix` | `embd` + `LLAMA_PP_MULTI_SPLIT=1` | 1 | 1 | 0.63 MiB | 6.26 MiB | — |
| `otdfix` | `otd` + `LLAMA_PP_MULTI_SPLIT=1` | 2 | 1 | 0.63 MiB | 6.00 MiB | 0.06 MiB |
| `bembd` | `embd` with `-bs` (the preset's backend sampler) | 1 | 4 | 2.52 MiB | 8.40 MiB | — |
| `bfix` | `fix` with `-bs` | 1 | 1 | 0.63 MiB | 6.26 MiB | — |

- **`embd` repeats R123b on the CPU:** the draft has one split and four copies, and its host buffer is 4 × 0.63 = 2.52 MiB.
  On the rig it was 4 × 21.60 = 86.40 MiB (logged as 86.41).
- **`fix` gives it back:** one copy, a host buffer of 0.63 MiB (no larger than `otd`'s, the line without C762's
  switch). Its second device holds 6.26 MiB, as with C762's switch in C762's own check (the gathered rows that the first device held in
  `otd`).
- **The switch acts only where it should.** The draft's first reserve logs `LLAMA_PP_MULTI_SPLIT: the graph runs as one
  split (pp 1, tg 1)`. Its second reserve has one copy from the start, with no line. The target (three splits on two
  devices here) keeps four copies in every arm. With the `-otd` rule (`otdfix`) the draft already had one copy, so
  nothing changes there.
- **The same output:** `embd`, `fix` and `otdfix` give `otd`'s tokens, top-3 probabilities, draft counts and all 406
  verify steps on every request, and `bfix` gives `bembd`'s ([server_requests.tsv](server_requests.tsv),
  [server_verdict.txt](server_verdict.txt)).
- **The first run of this check said FAIL on a rule of the check, not of the change**
  ([server_verdict_try1.txt](server_verdict_try1.txt)). The rule wanted the switch's line in each of the draft's two
  reserves, but by design only the first logs it. Every number there was the same as in the second run. The rule was
  corrected and the step run again; the second run is the one above.

### Every switch off, the tests, the GPU code and the rig tarball

| Check | Result |
|---|---|
| `test-llama-archs -b CPU -s 1` against the base's logits (`d097d6d88`'s static CPU build, the identity base of the v4 line's checks) | 146 of 146 the same with the switch unset, with `LLAMA_PP_MULTI_SPLIT=1`, and with it and `LLAMA_DRAFT_EMBD_DEV=layer` (no file there is an MTP-only draft, and the CPU never turns pipeline parallelism on) |
| ctest, static and shared builds ([C565's set](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh)) | 70 of 70 each, the network test 1 of 1, archs 151 of 151 and 146 files the same written and read back, the hand tests (hop-cast, route-once ×1 and ×30, bytes-switches, moe-ids, thread-safety, context, eval-callback) all passing |
| The switch in the build's list (C770) | 863 names, `LLAMA_PP_MULTI_SPLIT` among them and in `bin/llama-server` |
| `llama-server`'s GPU code against the `-kern` build's (`d097d6d88`) | the same 8,540 functions, SASS and PTX, the same dumps: host code only |
| The rig tarball (`TESTS=1`, sm_61) | `llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812.tar.gz`, 678,437,620 B, md5 `7b73ac4c778c5c9d03bf4163120e2dd8`; 0 AVX instructions; `test-backend-ops` in `bin/` |
| The merged `pascal-p106-v4` (`f91533730`) | a static CPU build of every target; `test-llama-archs` 146 of 146 the same as the base with every switch off and with `LLAMA_PP_MULTI_SPLIT=1` |

## The rig check (R135, Q1421)

**On nine P106 cards the draft gets one copy of each input, as on the `-otd` line, with the table still on card 8. Card 8
gets 84 MiB back and the host 85 MiB after loading, and the replies are `base`'s on all 62 turns.
`LLAMA_PP_MULTI_SPLIT=1` joins the stacked line beside `LLAMA_DRAFT_EMBD_DEV=layer`.**

One load of R43's replay (the agent check's 62 prompts, two passes) on nine P106 cards. The line is R123b's `embd`
rung: R123a's kept line, `LLAMA_DRAFT_EMBD_DEV=layer`, and `r43_run.sh` without `-otd token_embd=CUDA4`. To that the
load adds `LLAMA_PP_MULTI_SPLIT=1`. It ran on this tarball (`bin/llama-server` md5 `b8365b25864cf1aab0b81b47ec2f40a7`),
launched by [run/r135_load.sh](run/r135_load.sh) with `RUNG=pp1`. The tables in [rig/](rig/) come from three readers:
- R123b's reader, [r123b_logs.sh](../2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh), run on this load and R123b's
  `embd` folder;
- R123a's ladder, [r123a_ladder.py](../2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py), with `base`, `embd`, this
  load and `base2`;
- [run/r135_extra.sh](run/r135_extra.sh), for the reserves, the swap and the MemAvailable over time.

Read against R123b's `embd`:

| What | R123b `embd` | R135 `pp1` | Est. before the run |
|---|---:|---:|---:|
| The draft's reserves ([reserves.tsv](rig/reserves.tsv)) | `sched copies = 4`, 1 split | `sched copies = 1`, 1 split, the switch's line in the first reserve | `sched copies = 1`, the line once |
| The draft's `CUDA_Host` compute buffer | 86.41 MiB | 21.60 MiB | 21.60 MiB |
| The draft's card-8 compute buffer | 278.39 MiB | 195.49 MiB | 213.59 MiB |
| Card 8 after loading ([cards.tsv](rig/cards.tsv)) | 5,545 MiB | 5,461 MiB | 5,480 MiB |
| Card 8 after the replay | 5,627 MiB | 5,543 MiB | — |
| The host after loading: used, shared, available ([host.tsv](rig/host.tsv)) | 2,992, 264, 844 MiB | 2,906, 199, 929 MiB | available 909 MiB |
| The host's swap used, after loading and after the replay ([swap.tsv](rig/swap.tsv)) | 20, 20 MiB | 22, 518 MiB | — |
| The host's lowest MemAvailable ([memory.tsv](rig/memory.tsv)) | 388 MiB at 801 s | 494 MiB at 281 s (swap from 289 s on) | 453 MiB |
| MemAvailable at 281 s, both loads ([memavail.tsv](rig/memavail.tsv)) | 425 MiB | 494 MiB | — |
| Greedy replies ([replies.tsv](rig/replies.tsv)) | 62 of 62 as `base` | 62 of 62 as `base`, both passes | the same |
| Seconds a pass ([ladder.tsv](rig/ladder.tsv)) | 246.53, 238.34 | 244.89, 233.83 | no change |
| The target graph's reallocations ([realloc_marks.tsv](rig/realloc_marks.tsv)) | 0 in 146 requests | 0 in 146 requests | — |

- **The draft's buffers are the `-otd` line's.** The pinned host buffer is `base`'s 21.60 MiB. The card-8 buffer is
  195.49 MiB, against `base`'s 195.48 on card 8 plus 5.00 on card 4. It is 82.90 MiB below `embd`'s, more than the
  64.80 the estimate took off (three copies of the host inputs). The target's reserves are `embd`'s to the hundredth
  of a MiB, with four copies and ten splits. The switch's line, in the draft's first reserve:
  `sched_reserve: LLAMA_PP_MULTI_SPLIT: the graph runs as one split (pp 1, tg 1): planned without pipeline parallelism,
  one copy of each input`.
- **Card 8: −84 MiB** after loading and after the replay (5,461 and 5,543 against 5,545 and 5,627). Cards 0 to 7 are
  the same to the MiB.
- **The host after loading: 85 MiB more available** (929 against 844). Shared memory, where the driver's pinned
  buffers count, is 65 MiB lower (199 against 264), the draft's 64.81 MiB.
- **The swap.** During the replay the host moved 496 MiB of the server's memory to swap: 518 MiB used after the replay
  against 22 after loading. It was freed when the server stopped (22 MiB a minute later). Swap is read only after
  loading and after the replay; it began at about 289 s into the load, a minute into the replay, where `pp1`'s
  MemAvailable starts to climb while `embd`'s stays flat ([memavail.tsv](rig/memavail.tsv)). In
  R123b's `base`, `embd`, `sparse` and `base2` loads swap stayed at 20 MiB; its `queues` and `b4096` loads, with the
  doubled launch queue's 222 MiB more host memory, swapped 867 and 929 MiB. The server grew as much as in `embd`
  (used plus swap after the replay 3,412 against 3,445 MiB). The host kept its page cache (1,350 MiB after the replay
  against 893) and swapped instead.
  - **This load started with a different host.** It began with 1,622 MiB of page cache, the `-c812` tree unpacked onto
    the build drive 0 to 6 minutes before, at a load average of 3.05. `embd` began with 687 MiB of page cache at 0.67.
    The switch only removes pinned memory, so est. the swap is that host state and not the switch. The new row Q1430
    tells the two apart. It did ([R138](../2026-11-12_r138_swap_cause/README.md)): on the `-c812` build neither
    `embd` nor `pp1` swapped, `embd` with 1,817 MiB of page cache at its start.
  - **The lowest MemAvailable compares only up to 288 s.** Until then both loads track each other: `pp1` is 61.2 MiB
    above `embd` on average, from 21 (the servers up 2 s apart) to 75. Both reach their low of that stretch at the
    same sample: 494 against 425 MiB at 281 s, +69. After that, `pp1`'s MemAvailable climbs as the swap fills (938 MiB
    at 801 s, where `embd` reached its lowest, 388). The whole-run lowest, 494 against 388, holds the swap and is not a
    like-for-like number. The est. 453 MiB against 388 is neither confirmed nor refuted.
- **The time is a diagnostic across builds.** `embd` ran on the `-kern` build (`d097d6d88`), this load on `-c812`
  (`d679a6d9d`, on `e27947287`). The ladder's rule calls it negligible: 244.89 and 233.83 s a pass against
  246.53 and 238.34, means 239.36 against 242.43, −3.07 s (−1.27 %), within R123b's spread of 3.77. By new tokens:
  −0.19, −0.46, −1.83 and −0.59 s. The swap did not slow it.
- **Every count as `embd`:** 315 checkpoint lines, 150 taper lines, "4 catch-ups ran after the next call was queued",
  0 graph reallocations, the table on card 8 ("token_embd.weight (644.14 MiB q8_0) in CUDA8"), 28 switches set (27 and
  the new one). The server was up after 224 s (`embd`: 226). The load ran 20:11:36–20:25:39.
- **No same-build `embd` load was run**, so the time against `embd` mixes the two builds. Q1430's `embd` load on this
  tarball also gives that reference.

## New ideas

- **C813 (Q1422): the hidden-row buffer's second half on card 8 only when a prompt is cut into calls.** With
  `-b 4096` the second half is 160.04 MiB of card 8 that R43's replay never used; with `-b 2048`, 80.04 MiB used only on
  prompts of more than 2,048 new tokens. Making it on the first such prompt (and falling back to one half, the catch-up
  before the next call, if the card has no room then) would give card 8 that memory back on agent prompts. It matters
  at the long-context lines, where card 8 is the fullest card with C762's table on it (est. 51 MiB over at 196,608
  tokens, [C762](../2026-11-11_c762_draft_rows/README.md)).
- **C814 (Q1423): the draft's context on its layer's card alone** (`-devd`), above: the same one copy without code,
  one GPU backend instead of nine in the draft's scheduler. Planned: a CPU check with two RPC devices
  (`-devd <second>`) first.
- **Q1430: why R135's host swapped 496 MiB.** The candidates are the switch, the `-c812` build, or a page cache full of
  the tree unpacked minutes before. Run two loads on this tarball, with the tree unpacked well before the first:
  `RUNG=embd` (the switch unset, which also gives a same-build time) and `RUNG=pp1` again. Each logs swap every 2 s
  beside MemAvailable. Each reads swap used, the lowest MemAvailable and the time. Est. no swap in either load, and
  `pp1`'s lowest MemAvailable about 65 MiB above `embd`'s.

## Commands

The checks, in three chains of [run/checks.sh](run/checks.sh) (each step in a container, under one build lock):

```bash
export M=<work root> REPO=<this repository> LOCK=<flock file> \
  BASE_COMMIT=e279472876de7a693782f21fd8f5113128c2e1b6 \
  BASE_TREE=<d097d6d88's unpacked rig tarball, the -kern build> BASE_CPU=<bin/ of d097d6d88's static CPU build> \
  SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> JOBS=8 THREADS=8
STEPS="cpu server" bash docs/data/2026-11-12_c812_draft_pinned/run/checks.sh                    # the first server run
STEPS="rig names sass server off c565 trim" bash docs/data/2026-11-12_c812_draft_pinned/run/checks.sh
STEPS=merged MERGED_STEPS="build off" bash docs/data/2026-11-12_c812_draft_pinned/run/checks.sh  # after the merge
```

The server check on its own, with a GGML_RPC build's `llama-server` and `ggml-rpc-server`, and the models written by
`test-llama-archs -a '^qwen4exp$' -s 1 -o <dir> --mtp-draft`:

```bash
RPC_BIN=<bin/ of the RPC build> MODEL=<dir>/qwen4exp-moe-mtp.gguf DRAFT=<dir>/qwen4exp-moe-mtp-draft.gguf OUT=<dir> \
  SARGS='--override-kv qwen4exp.context_length=int:16384' bash docs/data/2026-11-12_c812_draft_pinned/run/server_check.sh
```

R123b's draft reserves ([r123b_reserves.tsv](r123b_reserves.tsv)) were read from its six server logs (`-lv 5`), one
row per `sched_reserve: reserving ...` block; a block of fewer than 1,000 nodes is the draft's:

```bash
for r in base embd sparse queues b4096 base2; do
  awk -v load=$r '/sched_reserve: reserving \.\.\./ { open = 1; bufs = ""; n++ }
    open && /compute buffer size = / { b = $0; sub(/.*sched_reserve: +/, "", b); sub(/ compute buffer size = +/, "=", b); sub(/ MiB.*/, "", b); bufs = bufs (bufs == "" ? "" : ", ") b }
    open && /graph: nodes = / { x = $0; sub(/.*nodes = /, "", x); sub(/,.*/, "", x); nodes = x + 0; y = $0; sub(/.*splits = /, "", y); sub(/,.*/, "", y); splits = y }
    open && /sched copies = / { z = $0; sub(/.*sched copies = /, "", z); ctx = (nodes < 1000) ? "draft" : "target"
      print load "\t" ctx "\t" n "\t" nodes "\t" splits "\t" z+0 "\t" bufs; open = 0 }' r123b_$r/load-A/server.log
done
grep -o 'h_nextn_dev_alloc: .*' r123b_*/load-A/server.log   # the hidden-row buffer: 160.08 MiB, 320.08 with -b 4096
```

R135 on a rig (an ssh alias in `RIG`; the tarball unpacked in `BUILDS`; R43's folder in `WORK`), then its tables on
any machine with R123b's load folders and this one in `data/` (R123b's six as `r123b_<rung>`, this one as `r135_pp1`):

```bash
RIG=<alias> WORK=<R43's folder> BUILDS=/opt/llama-builds BUILD=llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812 \
  MODEL=<Swift Q2_0's first shard> DRAFT=<the MTP draft> RUNG=pp1 bash docs/data/2026-11-12_c812_draft_pinned/run/r135_load.sh
ssh "$RIG" "while [ ! -e $WORK/r135_pp1/DONE ]; do sleep 30; done"
ssh "$RIG" "tar cf - -C $WORK r135_pp1 r135_pp1.out r135_pp1.launch" | tar xf - -C data/

# R123b's reader and R123a's ladder (md5 cd13ac0ade46bafb95f3f8cc22fd0503), then the reserves, swap and MemAvailable
P=docs/data/2026-11-12_r123b_prompt_ladder/run
OUT=rig REPO=<this repository> bash $P/r123b_logs.sh embd=data/r123b_embd pp1=data/r135_pp1
python3 -I docs/data/2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py --load base=data/r123b_base/load-A \
  --load embd=data/r123b_embd/load-A --load pp1=data/r135_pp1/load-A --load base2=data/r123b_base2/load-A \
  --base base --base2 base2 --out rig
DATA=data OUT=rig bash docs/data/2026-11-12_c812_draft_pinned/run/r135_extra.sh embd=r123b_embd pp1=r135_pp1 \
  base=r123b_base base2=r123b_base2
```

## Files

| File | What it is |
|---|---|
| [r123b_reserves.tsv](r123b_reserves.tsv) | R123b's reserves per load: context, nodes, splits, scheduler copies, every compute buffer |
| [server_reserves.tsv](server_reserves.tsv) | the CPU check's reserves per arm, the same columns, and whether the switch's line is in the block |
| [server_requests.tsv](server_requests.tsv) | the CPU check's requests per arm: prompt and generated counts, draft counts, the tokens |
| [server_verdict.txt](server_verdict.txt), [server_verdict_try1.txt](server_verdict_try1.txt) | the CPU check's verdict, and the first run's (the check's own rule wrong, as above) |
| [checks.tsv](checks.tsv) | every check's verdict and detail, in order of the three chains (the work root and the base build's folder in angle brackets) |
| [run/checks.sh](run/checks.sh) | the build machine's steps: the rig tarball, the switch list, the GPU code, the CPU and RPC builds, the server check, the logits identity, ctest static and shared, the merged tree |
| [run/server_check.sh](run/server_check.sh) | the two-device check's arms, its reserve reader and its verdict |
| [run/r135_load.sh](run/r135_load.sh) | R135's launcher on a rig |
| [run/r135_extra.sh](run/r135_extra.sh) | R135's reserves, swap and side-by-side MemAvailable tables |
| [rig/reserves.tsv](rig/reserves.tsv) | R135: every reserve of `embd`, `pp1`, `base` and `base2`: context, nodes, splits, scheduler copies, every compute buffer, whether the switch's line is in the block |
| [rig/swap.tsv](rig/swap.tsv) | R135: swap used after loading and after the replay, per load |
| [rig/memavail.tsv](rig/memavail.tsv) | R135: `embd`'s and `pp1`'s MemAvailable every 2 s side by side, and the difference |
| [rig/host.tsv](rig/host.tsv), [rig/cards.tsv](rig/cards.tsv), [rig/buffers.tsv](rig/buffers.tsv), [rig/loads.tsv](rig/loads.tsv), [rig/switch_lines.txt](rig/switch_lines.txt), [rig/realloc_marks.tsv](rig/realloc_marks.tsv) | R135: R123b's reader on `embd` and `pp1` (its columns are in [R123b's README](../2026-11-12_r123b_prompt_ladder/README.md)) |
| [rig/ladder.tsv](rig/ladder.tsv), [rig/replies.tsv](rig/replies.tsv), [rig/memory.tsv](rig/memory.tsv), [rig/first.tsv](rig/first.tsv), [rig/decision.md](rig/decision.md), [rig/turns.csv](rig/turns.csv) | R135: R123a's ladder over `base`, `embd`, `pp1` and `base2` (`pp1` classed against `embd`, the rung before) |

## Limits

- The CPU check shows the scheduler's copies and the buffers' sizes in proportion, not the rig's own numbers: R135
  measures those.
- R135 is one load against R123b's on another build, with the host swapping in R135 and not in R123b. The buffers,
  cards and host after loading are exact; the lowest MemAvailable and the time are not like for like (Q1430).
- The switch decides from the reserved graphs (the prompt-sized and the one-token one). A context whose later graphs had
  more splits than its reserve's would keep one copy; no context of the Flash-Next line does that (the weights decide
  the splits, and they do not move).

## Provenance and licence

Everything here was made for this repository: the change, the scripts and the tables. The R123b numbers come from that
row's own published data and server logs. The random-weight test models are written by the fork's own
`test-llama-archs`. The scripts are under this repository's licence.
