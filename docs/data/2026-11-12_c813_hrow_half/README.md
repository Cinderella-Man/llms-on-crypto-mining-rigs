# C813: the MTP hidden-row buffer's second half only when a prompt needs it

Registry: [Q1422](../../../quants/experiments.md) (the change) and Q1432 (its rig check, R139). The switch:
`LLAMA_MTP_HROW_LAZY` in [docs/switches.md](../../switches.md). Fork branch `pascal-p106-v4-b12246-c813`
(`cb1ae3145`), merged into `pascal-p106-v4` as `65ac9a47d`.

## What it fixes

With `LLAMA_MTP_CATCHUP_OVERLAP=1` ([C703](../2026-11-11_c703_catchup_overlap/README.md)) the MTP draft catches up on a
prompt call only once the target has its next call queued. Meanwhile the target keeps that call's hidden rows in one
half of its row buffer and writes the next call's rows into the other. The buffer (`h_nextn_dev`) sits on the target's
output card, card 8 in the Flash-Next preset, and holds two halves of `n_batch` + 1 rows of 10,240 F32 values: 160.08
MiB at `-b 2048`, 320.08 at `-b 4096` ([C812](../2026-11-12_c812_draft_pinned/README.md)).

Only a call without an output row waits like that: a prompt cut into several calls. An agent prompt that fits one call
never writes the second half. With `-b 4096`, none of R43's 62 recorded agent prompts was cut into calls (R123b's
`b4096` load logged `0 catch-ups ran after the next call was queued`), so the second half's 160.04 MiB held nothing.
Card 8 is the fullest card once the MTP draft's token table sits there (C762: est. 51 MiB over at 196,608 tokens).

## The change

`LLAMA_MTP_HROW_LAZY=1` (off by default; read when a context is made; `src/llama-context.cpp`, host code only):
- When the overlap asks for two halves, the buffer keeps one.
- The first call whose catch-up waits (the first prompt cut into calls) makes the second half. It allocates a buffer of
  two halves while the old one still exists, copies the old half's rows into the new first half on the card, frees the
  old buffer and drops the target's reusable graphs. The new tensor has a new address, so the draft's graphs, which
  compare it, are built again too.
- If the card has no room for the new buffer then, the hold is refused. The driver runs that call's catch-up before the
  next call, as with the overlap off, and the buffer keeps one half for the context's life (one warning line).
- `2` never makes the second half: the check of that path.
- Alone, without the overlap, the switch changes nothing.

The server logs the buffer at load as `h_nextn buffer size = 160.04 MiB (one half, the second at the first held call:
LLAMA_MTP_HROW_LAZY)` at `-b 4096`, and the growth as `h_nextn_grow: ... 320.08 MiB (two halves, made at the first held
call: LLAMA_MTP_HROW_LAZY)`.

Costs:
- Making the second half needs 320.08 MiB free on the card for a moment at `-b 4096` (the new buffer beside the old
  one); 160.04 MiB comes back when the old one is freed.
- That one call waits for its own rows to be written and frees a device buffer, so the next call is queued after the
  call has finished, once per server. Every later call overlaps as before.

`llama_get_embeddings_nextn_dev_halves()` (new, in `llama-ext.h`) returns the halves the buffer holds now.

## Checks

On the build machine's CPU, no GPU, at a low CPU share under one build lock ([run/checks.sh](run/checks.sh)); every
verdict and detail is in [checks.tsv](checks.tsv).

| Step | What | Verdict |
|---|---|---|
| rig tarball | `TESTS=1 EXTRA_TARGETS=test-llama-archs`, sm_61 | OK: `llama.cpp-20261112-cb1ae3145-pascal-p106-v4-b12246-c813.tar.gz`, 744,359,212 B, md5 `db7f414713ea4948bc67b7938bc122ed`; 0 AVX instructions, GLIBC_2.38; `bin/llama-server` md5 `799ca051b3f2a923aff0b3f1a138b0d1` |
| switch list | C770's list in the build | 868 names, `LLAMA_MTP_HROW_LAZY` in it and in the binaries |
| GPU code | `llama-server`'s SASS against the `-kern` build (`d097d6d88`; every merge since is host or test code) | the same file, 8,540 functions |
| | its PTX | 10 `flash_attn_tile` instances differ in 70 lines, all the number in a function-local `__shared__` array's name (nvcc counts the declarations before it, and C779's `common.cuh` has more than `-kern`'s). With that number masked ([C779's ptx_names.sh](../2026-11-12_c779_graph_replay/run/ptx_names.sh)) the whole dumps are the same file |
| CPU build | static, native, every target | OK |
| `--mtp-hrow-lazy` | the rig tarball's CPU code and the CPU build, seeds 1, 2, 3 | 25 of 25 in each of the six runs ([hrow.tsv](hrow.tsv)) |
| `--mtp-catchup-overlap` | the switch unset and at `1`, both builds, seeds 1, 2, 3 | 5 of 5 in each of the twelve runs, 0 and 4 catch-ups waiting ([overlap.tsv](overlap.tsv)) |
| `--mtp-incall-overlap`, `--mtp-h-dev` | the switch at `1`, the CPU build, seed 1 | 5 of 5 each |
| every switch off | 146 architectures' logits against `b9a1c1fa7`'s static CPU build | 146 of 146 the same with the switch unset, at `1` and at `2` |
| ctest | static and shared builds | 71 of 71 each; the network test, the archs write and compare (146 files), the hand tests: all OK |
| merged tree | `pascal-p106-v4` `65ac9a47d` (this branch merged on C815's and C817's merges): static CPU build, `test-llama-archs` against `b9a1c1fa7`'s logits, `--mtp-hrow-lazy` seed 1 | build OK; 146 of 146 the same with the switch unset and at `1`; 25 of 25 |

The first run of the tests, on the branch's first commit `48b597171` (`STEPS="cpu lazy"`), gave the same verdicts; the
second commit only changed the separator the test prints in its halves column (`|` to `;`, the table's own cell mark).

### The CPU test: `test-llama-archs --mtp-hrow-lazy`

It drives a target context and an MTP draft context of a random `qwen4exp` (QSA, MoE, one MTP block) through the
real driver in llama-server's order, with `n_batch` 32:
1. a first prompt of 20 tokens in one call, two draft steps;
2. a second prompt of 73 tokens in calls of 32, 32 and 9, with a checkpoint's draft part after the first call, two
   draft steps;
3. a third prompt of 12 tokens in one call, one draft step.

Five arms, each compared with the overlap off: the draft's sequence state and the driver's after the long prompt and at
the end, the checkpoint's draft part, every draft and every drafted token's probability, byte for byte. Each arm also
has an expected count of waiting catch-ups and the halves after each call ([hrow.tsv](hrow.tsv); `;` before a draft
step's count):

| Arm | Overlap | `LLAMA_MTP_HROW_LAZY` | Waited | Halves after each call |
|---|---|---|---|---|
| off | off | unset | 0 | `0;0;0 0,0,0;0;0 0;0` (the CPU has no events: the host rows, C703's reference) |
| on | on | unset | 2 | `2;2;2 2,2,2;2;2 2;2` (C703: two halves from the start) |
| lazy | on | `1` | 2 | `1;1;1 2,2,2;2;2 2;2` (one half until the long prompt's first call) |
| noroom | on | `2` | 0 | `1;1;1 1,1,1;1;1 1;1` (never two: every catch-up before the next call) |
| off+lazy | off | `1` | 0 | `0;0;0 0,0,0;0;0 0;0` (the switch alone changes nothing) |

Every arm was identical to the reference in all five configurations (F16 KV with and without flash attention, Q8_0 KV,
C620's KV-only draft pass, the driver's top-k on the host), with seeds 1, 2 and 3, on the CPU build and on the rig
tarball's CPU code.

C703's own test (`--mtp-catchup-overlap`) passes with the switch unset and at `1`: its prompt is longer than `n_batch`
from the first call, so the second half is made at once, and its four catch-ups still wait. C735's
(`--mtp-incall-overlap`) and C693's (`--mtp-h-dev`) pass with the switch at `1`.

## The rig check (R139, Q1432)

**On nine P106 cards card 8 gets the whole 160 MiB back: 5,545 MiB after loading and 5,627 after the replay, against
5,705 and 5,787 without the switch, and 5,627 MiB is also its peak. The buffer keeps one half all session, the replies
are `base`'s on all 62 turns, and the time is the same. `LLAMA_MTP_HROW_LAZY=1` joins the stacked line beside
`-b 4096`.**

One load of R43's replay (the agent check's 62 prompts, two passes) on nine P106 cards, 2026-11-13. The line is R123b's
`b4096` rung: R123a's kept line, `LLAMA_DRAFT_EMBD_DEV=layer`, `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1`,
`CUDA_SCALE_LAUNCH_QUEUES=2x`, `-b 4096` and `r43_run.sh` without `-otd token_embd=CUDA4`. The load adds
`LLAMA_MTP_HROW_LAZY=1`; the two loads' launch files differ in that one line. It ran on this tarball
(`bin/llama-server` md5 `799ca051b3f2a923aff0b3f1a138b0d1`), launched by [run/r139_load.sh](run/r139_load.sh) with
`RUNG=lazy`, which also sampled card 8's memory every 2 s ([card8.csv](rig/card8.csv)). R123b's `b4096` ran on the
`-kern` build (`d097d6d88`), so the time is a diagnostic across builds. The tables in [rig/](rig/) come from three
readers:
- R123b's reader, [r123b_logs.sh](../2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh), on this load and R123b's
  `b4096` folder;
- R123a's ladder, [r123a_ladder.py](../2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py), with `base`, `b4096`,
  this load and `base2`;
- R138's [r138_swap.sh](../2026-11-12_r138_swap_cause/run/r138_swap.sh), for the host's start, its swap and its
  MemAvailable over time.

Read against R123b's `b4096`:

| What | R123b `b4096` | R139 `lazy` | Est. before the run |
|---|---:|---:|---:|
| The `h_nextn` buffer on card 8 (server log) | `320.08 MiB (two halves)` | `160.04 MiB (one half, the second at the first held call: LLAMA_MTP_HROW_LAZY)` | 160.04 MiB |
| `h_nextn_grow` lines | — | none | none |
| Catch-ups queued behind the next call (server log at exit) | 0 | 0 | 0 |
| Card 8 after loading ([cards.tsv](rig/cards.tsv)) | 5,705 MiB | 5,545 MiB | 5,545 MiB |
| Card 8 after the replay | 5,787 MiB | 5,627 MiB | 5,627 MiB |
| Card 8's peak, every 2 s ([card8.csv](rig/card8.csv)) | not sampled | 5,627 MiB (from 09:20:27 on; 205 of 425 samples) | 5,627 MiB |
| Cards 0 to 7, after loading and after the replay | [cards.tsv](rig/cards.tsv) | the same to the MiB | the same |
| Every model, KV and compute buffer line ([buffers.tsv](rig/buffers.tsv)) | 113 lines with the other loads | the same | the same |
| The host after loading: used, shared, available ([host.tsv](rig/host.tsv)) | 3,289, 264, 547 MiB | 3,281, 265, 555 MiB | the same |
| The host after the replay: used, buff/cache, available | 2,950, 1,445, 886 MiB | 3,205, 1,187, 631 MiB | the same |
| Swap used after loading and after the replay ([swap_summary.tsv](rig/swap_summary.tsv)) | 20, 949 MiB | 0, 770 MiB | — |
| The host's lowest MemAvailable ([memory.tsv](rig/memory.tsv)) | 183 MiB at 247 s | 120 MiB at 281 s | the same |
| Greedy replies ([replies.tsv](rig/replies.tsv)) | 62 of 62 as `base` | 62 of 62 as `base`, both passes | the same |
| Seconds a pass ([ladder.tsv](rig/ladder.tsv)) | 243.95, 233.12 | 242.46, 231.43 | within 3.77 s |
| The target graph's reallocations ([realloc_marks.tsv](rig/realloc_marks.tsv)) | 0 in 146 requests | 0 in 146 requests | — |

- **The buffer does what C813 said.** It is one half at load, and no prompt of the replay was cut into calls at
  `-b 4096`, so no call held its rows and the second half was never made. Card 8 is 160 MiB lower after loading and
  after the replay, the buffer's 160.04 MiB. The sample shows card 8 at 5,545 MiB when the server came up (09:17:09),
  growing during the replay's first pass to 5,627 at 09:20:27, and never above it.
- **The time:** 242.46 and 231.43 s a pass against 243.95 and 233.12, −1.59 s on the means (−0.67 %), negligible
  against R123b's spread of 3.77 s. Every size bucket is within its spread or faster.
- **The host held more of the server in RAM during the replay, and swapped less of it.**
  - After loading the host is the same (555 against 547 MiB available). While the model loaded, the lowest
    MemAvailable was 431 against 438 MiB.
  - In both loads the host moved part of the server to swap during the replay (R123b's `b4096` and `queues` loads were
    the ones that swapped). This load swapped 179 MiB less (770 against 949), so more of the server stayed in RAM:
    lined up at the second the server came up (247 and 226 s), its MemAvailable was 215.7 MiB lower on average over
    the next 600 s, 292 MiB at the widest (440 s). Used plus swap after the replay is 3,975 MiB against 3,899, +76.
  - The lowest MemAvailable came in the replay's first minute in both: 120 MiB, 34 s after the server came up,
    against 183, 21 s after.
  - The switch only changes a buffer on card 8, and every host-side buffer line is `b4096`'s. Two other things differ
    from R123b's load: this one started 11 s after the `-c813` tree was unpacked, at a load average of 3.86 with
    1,667 MiB of page cache (R123b's: 0.82 and 1,128 MiB), the start [R138](../2026-11-12_r138_swap_cause/README.md)
    links to R135's swap; and the card sampler (`nvidia-smi -l 2`) ran beside it. Est. the host difference is those
    two and not the switch. Q1457 (R151) separates them with a `b4096` load on this build.
- **No swap over time:** the rig's copy of `rig_taper.sh` in R43's folder predates R138's SwapFree column, so swap is
  read only after loading and after the replay ([swap_time.tsv](rig/swap_time.tsv) has MemAvailable alone).

## Commands

The checks, in two chains of [run/checks.sh](run/checks.sh) (each step in a container, under one build lock), then the
merged tree:

```bash
export M=<work root> REPO=<this repository> LOCK=<flock file> \
  BASE_COMMIT=b9a1c1fa7b9b98dfe919374897afe6fde60b7219 \
  BASE_TREE=<d097d6d88's unpacked rig tarball, the -kern build> BASE_CPU=<bin/ of b9a1c1fa7's static CPU build> \
  SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> JOBS=8 THREADS=8
STEPS="cpu lazy" bash docs/data/2026-11-12_c813_hrow_half/run/checks.sh       # the first test run, on 48b597171
bash docs/data/2026-11-12_c813_hrow_half/run/checks.sh                        # every step, on cb1ae3145
STEPS=merged bash docs/data/2026-11-12_c813_hrow_half/run/checks.sh           # after the merge

# the PTX with the numbers in internal names masked (C779's script)
NEW=<this build's bin/llama-server> BASE=<the -kern build's bin/llama-server> OUT=<dir> \
  bash docs/data/2026-11-12_c779_graph_replay/run/ptx_names.sh
```

The CPU test on its own, with a build of the branch:

```bash
test-llama-archs --mtp-hrow-lazy -s 1
LLAMA_MTP_HROW_LAZY=1 test-llama-archs --mtp-catchup-overlap -s 1
```

R139 on a rig (an ssh alias in `RIG`; the tarball unpacked in `BUILDS`; R43's folder in `WORK`), then its tables on any
machine with R123b's load folders and this one in `data/` (R123b's as `r123b_<rung>`, this one as `r139_lazy`):

```bash
RIG=<alias> WORK=<R43's folder> BUILDS=/opt/llama-builds BUILD=llama.cpp-20261112-cb1ae3145-pascal-p106-v4-b12246-c813 \
  MODEL=<Swift Q2_0's first shard> DRAFT=<the MTP draft> RUNG=lazy bash docs/data/2026-11-12_c813_hrow_half/run/r139_load.sh
ssh "$RIG" "while [ ! -e $WORK/r139_lazy/DONE ]; do sleep 30; done"
ssh "$RIG" "tar cf - -C $WORK r139_lazy r139_lazy.out r139_lazy.launch r139_lazy.card.csv" | tar xf - -C data/

P=docs/data/2026-11-12_r123b_prompt_ladder/run
OUT=rig REPO=<this repository> bash $P/r123b_logs.sh b4096=data/r123b_b4096 lazy=data/r139_lazy
python3 -I docs/data/2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py --load base=data/r123b_base/load-A \
  --load b4096=data/r123b_b4096/load-A --load lazy=data/r139_lazy/load-A --load base2=data/r123b_base2/load-A \
  --base base --base2 base2 --out rig
grep -h 'h_nextn buffer size\|h_nextn_grow\|catch-ups ran' data/r139_lazy/load-A/server.log
awk -F', ' '$2 > m { m = $2; t = $1 } END { print "card 8 peak", m, "MiB at", t }' data/r139_lazy.card.csv
DATA=data OUT=rig bash docs/data/2026-11-12_r138_swap_cause/run/r138_swap.sh b4096=r123b_b4096 lazy=r139_lazy
cp data/r139_lazy.card.csv rig/card8.csv
```

As run on 2026-11-13: the tarball relayed to the rig's model drive, its md5 checked, unpacked into `/opt/llama-builds`
and the tarball deleted (09:09:40–09:12:50); the load started 09:13:01, the server up after 247 s, the replay's end
09:27:16, `DONE` 09:27:25; the tables read on a CPU at `nice -n 19`. The MemAvailable comparison lines each load up at
its server's start (226 s for `b4096`, 247 s for `lazy`) and pairs the 2 s samples over the next 600 s, from
[rig/swap_time.tsv](rig/swap_time.tsv):

```bash
awk -F'\t' 'NR > 1 { if ($1 == "b4096") { t = $2 - 226; if (t >= 0 && t <= 600) b[int(t / 2)] = $3 }
  if ($1 == "lazy") { t = $2 - 247; if (t >= 0 && t <= 600) l[int(t / 2)] = $3 } }
  END { for (k in l) if (k in b) { d = l[k] - b[k]; n++; s += d; if (n == 1 || d < mn) { mn = d; at = k * 2 } }
  printf "pairs %d, mean %.1f, widest %d at %d s\n", n, s / n, mn, at }' rig/swap_time.tsv
```

## Files

| File | What it is |
|---|---|
| [checks.tsv](checks.tsv) | every check's verdict and detail, in the order of the chains (the work root and the base build's folder in angle brackets) |
| [hrow.tsv](hrow.tsv) | `--mtp-hrow-lazy`'s rows, per build and seed: config, arm, the five comparisons, the waits, the halves |
| [overlap.tsv](overlap.tsv) | `--mtp-catchup-overlap`'s rows, per build, seed and switch value |
| [run/checks.sh](run/checks.sh) | the build machine's steps: the rig tarball, the switch list, the GPU code, the CPU build, the tests, the logits identity, ctest static and shared, the merged tree |
| [ptx_names.txt](ptx_names.txt) | the PTX of `llama-server` against the `-kern` build's with the numbers in internal names masked: the differing lines' names and the masked dumps' md5s |
| [run/r139_load.sh](run/r139_load.sh) | R139's launcher on a rig |
| [rig/](rig/) | R139's tables: R123b's reader (`loads.tsv`, `cards.tsv`, `host.tsv`, `buffers.tsv`, `switch_lines.txt`, `realloc_marks.tsv`) on `b4096` and `lazy`; R123a's ladder (`ladder.tsv`, `decision.md`, `replies.tsv`, `first.tsv`, `memory.tsv`, `turns.csv`) on `base`, `b4096`, `lazy` and `base2`; R138's swap reader (`swap_summary.tsv`, `swap_time.tsv`) on `b4096` and `lazy`; `card8.csv`, card 8's used MiB every 2 s (nvidia-smi's timestamp, MiB) |

## Limits

- The CPU check shows the buffer's halves, its growth and the bytes the draft sees, not card 8's own numbers: R139
  measures those.
- R139's replay never cut a prompt into calls at `-b 4096`, so the growth (`h_nextn_grow`) and the no-room path have
  run on the CPU only. On a card they run the first time a prompt is longer than `n_batch`.
- Card 8's peak is sampled in R139's load only; R123b's loads read the cards after loading and after the replay.
- On the CPU the growth's copy is a memory copy; on a card it is a device-to-device copy of the ggml CUDA buffer. Both
  are ggml's `ggml_backend_tensor_copy`, and the rows a draft reads after it are the ones the test compares.
- The no-room path is checked with `2`, which refuses the growth before allocating. A refused CUDA allocation takes the
  same path after ggml logs its failed `cudaMalloc`.

## Provenance and licence

Everything here was made for this repository: the change, the test, the scripts and the tables. R123b's numbers come
from that row's own published data. The random-weight test models are written by the fork's own `test-llama-archs`.
The scripts are under this repository's licence.
