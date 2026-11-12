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

## The rig check (R139, Q1432, not run)

One load of R43's replay with R123b's `b4096` line and the switch, on the `-c813` tarball, against R123b's `b4096` load
([R123b](../2026-11-12_r123b_prompt_ladder/README.md)). Rig commands: [run/r139_load.sh](run/r139_load.sh). It samples
card 8's memory every 2 s beside the load, which R123b's loads did not.

What to read:
- the server log's `h_nextn buffer size = 160.04 MiB (one half ...)` line, no `h_nextn_grow` line, and
  `0 catch-ups ran after the next call was queued`;
- card 8 after loading against R123b's 5,705 MiB (est. 5,545) and after the replay against 5,787 (est. 5,627); the other
  cards and the host as R123b's;
- card 8's peak from the sample (est. the after-replay value);
- 62 of 62 replies as R123b's `base`;
- the time a pass against `b4096`'s 243.95 and 233.12 s. It is a diagnostic across builds: the code path is the same
  call for call, so est. within R123b's spread of 3.77 s.

A pass puts `LLAMA_MTP_HROW_LAZY=1` into the stacked line beside `-b 4096`.

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

## Limits

- The CPU check shows the buffer's halves, its growth and the bytes the draft sees, not card 8's own numbers: R139
  measures those.
- On the CPU the growth's copy is a memory copy; on a card it is a device-to-device copy of the ggml CUDA buffer. Both
  are ggml's `ggml_backend_tensor_copy`, and the rows a draft reads after it are the ones the test compares.
- The no-room path is checked with `2`, which refuses the growth before allocating. A refused CUDA allocation takes the
  same path after ggml logs its failed `cudaMalloc`.

## Provenance and licence

Everything here was made for this repository: the change, the test, the scripts and the tables. R123b's numbers come
from that row's own published data. The random-weight test models are written by the fork's own `test-llama-archs`.
The scripts are under this repository's licence.
