# Flash-Next's 131,072-token line: its prompt effects split between the context and the micro-batch (R244, Q1924, 2026-11-26)

What this measured: R243 ([its data](../2026-11-26_r243_flash_131k_agent_rig1/README.md)) ran Qwen3.8-Flash-Next's
v5 preset against the same preset with `-c 131072 -ub 256`, two values changed at once. It found two prompt effects
in opposite directions. A fresh server's first request (2,780 tokens on an empty cache) was read 8.3 s faster, and
four short first turns (under 600 new tokens) were 7.9 % slower. This run tells the two values apart. Three lines of
the v5 preset (`release/flash-next-v5/settings.env` as it stands, the taper `short=1400:64,cuts=1`), nothing else
changed:
- `p512`: the preset, `-c 65536 -ub 512`;
- `u256`: the same with `-ub 256`;
- `l131`: the same with `-c 131072 -ub 256` (R124's and R243's line).

The three lines were loaded twice each, alternated, on one rig and boot (nine P106-090 cards). Each load replayed the
same eight requests recorded in R243's agent check. Registry row [Q1924](../../../quants/experiments.md), from Q1921
(R243).

**Result: the micro-batch makes both effects; the larger context costs nothing measurable on prompts, only card
memory.**
- **A fresh server's first request is the micro-batch's gain:** at `-ub 256` 18,322.5 and 18,718.7 ms against the
  preset's 27,222.1 and 26,848.4 (−8,514.6 ms, −31.5 %, gain). The 131,072-token context adds 88.8 ms (18,591.3 and
  18,627.4; +0.5 %, negligible).
- **Most of that cost is paid once a load.** The same 2,780 tokens read again on an empty cache, after the first
  reply, took 12,444.7 and 12,228.9 ms on the preset and 10,821.3 and 10,890.6 at `-ub 256`. So the first request
  carries 14,777.3 and 14,619.5 ms extra at `-ub 512` and 7,501.2 and 7,828.1 at `-ub 256` (7,545.5 and 7,596.0 on
  the 131,072-token line): `-ub 256` halves the once-a-load cost (−47.9 %, gain). An empty-cache read later in a load
  does not pay it again, and `-ub 256` reads it 12.0 % faster too (−1,480.9 ms, gain). There the 131,072-token context
  costs 182.7 ms (+1.7 %, a loss by the rule: the two lines' spreads are 14.5 and 69.3 ms).
- **The short calls' loss is also the micro-batch's,** although every call of them is read in micro-batches of 64 in
  all three lines (the taper). The four first turns under 600 new tokens took 13,482.8 and 13,473.8 ms at `-ub 256`
  against 12,475.4 and 12,410.7 on the preset (+1,035.2 ms, +8.3 %, loss). The context adds 4.2 ms (negligible). Most
  of it is the first short call after the long reads (JavaScript `triangle`, 430 tokens): 4,296.3 and 4,300.9 ms
  against 3,498.1 and 3,495.9 (+801.6 ms, +22.9 %). The other three together took 9,186.5 and 9,172.9 against 8,977.2
  and 8,914.9 (+233.7 ms, +2.6 %, loss).
- **The two first turns of 600 to 1,400 tokens are faster at `-ub 256`:** 10,005.9 and 9,990.4 ms against 10,315.4
  and 10,359.6 (−339.3 ms, −3.3 %, gain). All of it is JavaScript `phone-number` (4,166.2 and 4,165.4 against
  4,478.8 and 4,524.8); Python `scale-generator` does not move (5,839.7 and 5,825.0 against 5,836.6 and 5,834.9).
- **Over all eight requests of a load:** 52,632.5 and 53,073.4 ms at `-ub 256` against 62,457.6 and 61,847.7 on the
  preset (−15.0 %, gain). The seven after the first took 34,310.0 and 34,354.7 against 35,235.5 and 34,999.3
  (−785.0 ms, −2.2 %, gain). The 131,072-token line: 53,111.4 and 53,228.2 for all eight (negligible against `-ub 256`), and
  34,520.1 and 34,600.8 for the seven (+228.0 ms against `-ub 256`, +0.7 %, a loss by the rule).
- **Card 8, which holds the output head, sets the room.** At the highest it held 5,619 MiB on the preset (449 left of
  6,068), 5,495 at `-ub 256` (573 left) and 5,991 on the 131,072-token line (77 left), the same in both loads of each.
  After loading it held 5,545, 5,433 and 5,931. `-ub 256` frees 124 MiB on card 8 at 65,536 tokens; the doubled cache
  takes 496.
- **The host:** its lowest MemAvailable was 290, 305, 300, 291, 320 and 293 MiB in run order; swap stayed at the
  51 MiB in use before the runs.

What it means for the preset (the owner's call; nothing changed here):
- Within the prompts measured here, `-ub 256` is the better micro-batch for the 65,536-token preset as well: 8.5 s off
  each load's first request and 12.0 % off later empty-cache reads, for +8.3 % on short calls (mostly the first short
  call after long reads). The preset line would be `PRESET_ARGS="$PRESET_ARGS -b 4096 -ub 256"`. The agent check has
  measured `-ub 256` only together with the 131,072-token context (R243: a task not measurably slower). Since this run
  finds the context neutral on prompts, that result should carry over, but `-ub 256` at 65,536 tokens has no agent
  check of its own.
- The 131,072-token line keeps all of `-ub 256`'s prompt behaviour. Its costs are card 8's room (77 MiB left on this
  rig, 69 and 71 on rig 1 in R243) and 182.7 ms (+1.7 %) on a warm empty-cache read of 2,780 tokens.

## How it ran

- **The rig:** nine P106-090 cards on one boot, every arm with the same engine build (`pascal-p106-v5` candidate at
  `eee615cdb`, llama-server md5 `2727faf28b5b34dfe42399a9d431da67`). The cards ran the rig's memory-offset profile
  from the clock kit ([bench/mem_offset](../../../bench/mem_offset/README.md)) with its watchdog on, the same in all
  six loads. The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (Swift Qwen3.8-Flash-Next Pascal Q2
  GGUF's maker file) with its MTP draft. The servers came up after 227, 224, 225, 224, 224 and 224 s in run order.
- **Loads:** `p512a` 18:21, `u256a` 18:26, `l131a` 18:31, `p512b` 18:36, `u256b` 18:41, `l131b` 18:46, each ended by
  stopping its server. A load checked that the rig was quiet first (no llama process, no memory on any card).
- **The started command:** [run/r244.sh](run/r244.sh) builds each line from `settings.env`'s `PRESET_ENV` and
  `PRESET_ARGS`, in the order the agent check's harness writes them. It then replaces `-c` and `-ub` word for word
  (each found exactly once) and starts the line with the agent check's own rig script
  ([bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh), with its card log, host log and RAM
  watchdog). Once the server was up, the script read the running process's command line and environment back from
  `/proc` and compared them with the line it meant to start: equal in all six loads. The preset's line is R243's
  started preset command byte for byte, except for the taper's `cuts=1` (the preset since 2026-11-26). `u256` differs
  from it in `-ub` only, `l131` in `-c` and `-ub`. Each line passes [run/preset_check.sh](run/preset_check.sh)
  (R243's check with the arm `u256` added); `p512` and `l131` pass R243's own check too
  ([checks.txt](checks.txt); the lines with the build and draft paths replaced by placeholders are in
  [lines/](lines/)).
- **The eight requests,** in this order, sent one at a time by
  [bench/agent_check/replay.py](../../../bench/agent_check/replay.py) with the recorded body, the recorded reply
  length and a fixed seed. They are R243's own recorded requests from its first preset run, so each reads the same
  prompt tokens on the same cached prefix as in R243 (the `r243_prompt_n` and `r243_cache_n` columns of
  [requests.tsv](requests.tsv)):
  1. `first`: the first request of every R243 run, the first turn of Python `affine-cipher` (the system prompt and
     the task, 2,780 tokens), on the fresh server's empty cache;
  2. `erased`: the same request with `"cache_prompt": false` added to its body. The server then clears the slot's
     cache and reads all 2,780 tokens again. [run/no_cache_log.py](run/no_cache_log.py) writes it as a one-request
     log. The slot-erase endpoint answers 501 on a server started without `--slot-save-path`, which the preset does
     not set;
  3. `small`: the four first turns that read under 600 new tokens on the cached prefix of 1,549 tokens (the start
     every task's first turn shares), in R243's run order: JavaScript `triangle` (430), Python `list-ops` (441),
     JavaScript `grade-school` (403), JavaScript `variable-length-quantity` (577);
  4. `mid`: two first turns of 600 to 1,400 new tokens on the same prefix: JavaScript `phone-number` (610) and Python
     `scale-generator` (1,086).
- **Classes:** the release check's rule as R243 applies it. A difference is a gain or a loss only beyond the larger
  of the two lines' spreads (the larger load's value less the smaller), else negligible; prompt ms and MiB are better
  lower. Each pair is compared: `u256` against `p512` (the micro-batch alone), `l131` against `u256` (the context
  alone), `l131` against `p512` (R243's comparison).
- **A pilot load first** (`p512`, 18:14, the same boot) used replay.py's `--erase-slot` for the second request. The
  erase answered 501, so the second request read 4 tokens on the cached prompt. Its last request (Python
  `phone-number`) also reused 1,643 tokens of the JavaScript `phone-number` turn before it, not 1,549. The pilot is
  not in the tables. The steps above replace those two requests. Its first request took 27,217.2 ms, and its four
  short turns 3,502.9, 2,837.6, 2,718.1 and 3,349.0.

## Per load

| | `p512a` | `u256a` | `l131a` | `p512b` | `u256b` | `l131b` |
|---|---:|---:|---:|---:|---:|---:|
| first request, empty cache on a fresh server, prompt ms | 27,222.1 | 18,322.5 | 18,591.3 | 26,848.4 | 18,718.7 | 18,627.4 |
| the same after clearing the cache (`erased`), prompt ms | 12,444.7 | 10,821.3 | 11,045.8 | 12,228.9 | 10,890.6 | 11,031.4 |
| the first's extra, paid once a load, ms | 14,777.3 | 7,501.2 | 7,545.5 | 14,619.5 | 7,828.1 | 7,596.0 |
| 4 first turns under 600 tokens (1,851), prompt ms | 12,475.4 | 13,482.8 | 13,483.4 | 12,410.7 | 13,473.8 | 13,481.6 |
| of them JavaScript `triangle`, the first short call | 3,498.1 | 4,296.3 | 4,307.9 | 3,495.9 | 4,300.9 | 4,295.8 |
| of them the other three | 8,977.2 | 9,186.5 | 9,175.5 | 8,914.9 | 9,172.9 | 9,185.7 |
| 2 first turns of 600 to 1,400 tokens (1,696), prompt ms | 10,315.4 | 10,005.9 | 9,990.8 | 10,359.6 | 9,990.4 | 10,087.9 |
| all eight, prompt ms | 62,457.6 | 52,632.5 | 53,111.4 | 61,847.7 | 53,073.4 | 53,228.2 |
| card 8 MiB after loading; at the highest; left of 6,068 | 5,545; 5,619; 449 | 5,433; 5,495; 573 | 5,931; 5,991; 77 | 5,545; 5,619; 449 | 5,433; 5,495; 573 | 5,931; 5,991; 77 |
| host lowest MemAvailable MiB; swap in use MiB | 290; 51 | 305; 51 | 300; 51 | 291; 51 | 320; 51 | 293; 51 |

Every request in [requests.tsv](requests.tsv); every measure and pair, classed, in [classes.tsv](classes.tsv).

## Files

| File | What |
|---|---|
| [requests.tsv](requests.tsv) | the 48 replayed requests: load, line, group, task, new tokens read, tokens from the cache, prompt ms, ms a prompt token, tokens generated, and R243's tokens read and from the cache for the same request |
| [loads.tsv](loads.tsv) | per load: the groups' prompt ms, the first request's extra over the erased one, the first short call and the other three apart, all eight, card 8 and the host |
| [classes.tsv](classes.tsv) | each measure for the three pairs of lines: both loads, means, spreads, difference and class |
| [cards.tsv](cards.tsv) | every card's MiB after loading and at its highest per load, on nvidia-smi's scale, and what the highest leaves of 6,068 |
| [lines/](lines/) | the three started lines (`p512.txt`, `u256.txt`, `l131.txt`), the build folder and draft path replaced by `<build>` and `<draft>` |
| [checks.txt](checks.txt) | the started commands' and dry lines' MD5s and the preset checks on both |
| [run/r244.sh](run/r244.sh) | one load of one line: the line built and checked, the server started, the started command read back, the eight requests replayed, the rig's logs collected |
| [run/no_cache_log.py](run/no_cache_log.py) | the one-request log for the `erased` request (`cache_prompt` false) |
| [run/preset_check.sh](run/preset_check.sh) | a started or dry line against `settings.env`, for `p512`, `u256` and `l131` |
| [run/tables.py](run/tables.py) | the tables from the six loads' replay rows, card logs and host logs |

## Commands

```bash
# one load at a time, alternated: p512a u256a l131a p512b u256b l131b (ARM is the load's name without its last letter);
# from the repository root on the machine that drives the rig; RIG, the paths and the ports are yours
ARM=p512 LOAD=p512a OUT=<folder> RIG=rig BIN=<build>/bin/llama-server MODEL=<first shard on the rig> \
  DRAFT=<MTP draft on the rig> RIG_PORT=8291 LOCAL_PORT=<free local port> RIGDIR=<folder on the rig> \
  LOG=<R243's first preset run's proxy.jsonl, with its .bodies.gz> \
  bash docs/data/2026-11-26_r244_flash_ctx_ub/run/r244.sh
# the line alone, without touching the rig, and its check
DRY=1 ARM=u256 LOAD=u256a OUT=<folder> BIN=<build>/bin/llama-server MODEL=x DRAFT=<draft> \
  bash docs/data/2026-11-26_r244_flash_ctx_ub/run/r244.sh
bash docs/data/2026-11-26_r244_flash_ctx_ub/run/preset_check.sh <folder>/u256a/started_command.txt u256
python3 -I docs/data/2026-11-26_r244_flash_ctx_ub/run/tables.py <folder> <tables folder>
```

The replayed bodies are the agent check's own requests as its proxy logged them in R243
([bench/agent_check](../../../bench/agent_check/README.md)); any run of the check with the same task set records the
same first turns (trace tags `v4r/bugfix-<task>/a1/t1`, request 0).

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The replayed requests are the
agent check's own task prompts ([bench/agent_check](../../../bench/agent_check/README.md)). The model is UkisAI's
Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
