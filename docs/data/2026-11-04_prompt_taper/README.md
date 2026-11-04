# C218: a taper of prompt micro-batches (Q771, Q363), 2026-11-04

Flash-Next reads prompts at ~60 tok/s on the nine P106 cards, and [the prefill note](../../research/notes/flash_next_prefill.md)
traces most of it to pipeline fill and drain, not to a per-token cost: a 512-token micro-batch crosses the nine cards one
after another in ~7.6 s, every prompt above 516 tokens ends with a 512-token call that runs alone after a checkpoint, and
a short prompt is a single pass. The note's idea I2 (registry [Q771](../../../quants/experiments.md), with the older Q363)
is a taper: micro-batches of 64–128 tokens where a call fills or drains the pipeline, and `-ub` (512) in between. This
folder holds that switch, off by default, the checks that ran without a GPU, and the rig script that measures it on the
agent check's prompts.

**Status:** written, built for the rig and checked on the CPU (the tables below). Not yet run on the rig:
[run/rig_taper.sh](run/rig_taper.sh) is next (C218r), together with Q770's no-code test.

## The build

Fork branch `pascal-p106-v3-b12076-taper`, one commit on `pascal-p106-v3-b12076-ladder2` (`a330672c1`,
[the merged engine](../2026-11-04_ladder2_merge/README.md)):

| Commit | What it does |
|---|---|
| `7b1632aee` | `llama_ubatch_taper` and `llama_ubatch_taper_plan()` (`src/llama-batch.*`), the batch splitter following a per-call plan; `LLAMA_UBATCH_TAPER`, `llama_set_ubatch_taper()`, `llama_get_ubatch_taper()`, `llama_set_ubatch_taper_next()` (`llama.h`, `src/llama-context.*`); llama-server's `--ubatch-taper`, a request's `"ubatch_taper"` and the per-call tail hint; `tests/test-ubatch-taper.cpp` |

The rig tarball and its md5 are under [Build and md5](#build-and-md5).

## The switch

Off unless one of these sets it. With it off, every call is split exactly as before (the plan is empty and the splitter
takes `n_ubatch` tokens at a time).

| Where | How |
|---|---|
| any llama.cpp tool | `LLAMA_UBATCH_TAPER=<spec>`, read when a context is created |
| llama-server | `--ubatch-taper <spec>` (env `LLAMA_ARG_UBATCH_TAPER`) for the target model's context; a request's `"ubatch_taper": "<spec>"` for the calls of its own prompt (a bad spec there falls back to the server's, with a warning) |
| the C API | `llama_set_ubatch_taper(ctx, spec)` (takes effect at the next call), `llama_get_ubatch_taper(ctx)`, and `llama_set_ubatch_taper_next(ctx, head, tail)` for what the caller knows about the next call |

`<spec>` is `off` (or empty, `0`), `on` (or `1`), or any of these, comma-separated:

| Rule | Meaning | In `on` |
|---|---|---|
| `head=N:S` | the first N tokens of a call that starts with nothing in flight, in micro-batches of S tokens (the remainder first) | `head=256:64` |
| `tail=N:S` | the last N tokens of a call that the caller waits for, in micro-batches of S tokens (the remainder last) | `tail=512:128` |
| `short=N:S` | a call that is both and holds N tokens or fewer: all of it in micro-batches of S tokens | `short=600:64` |
| `min=N` | no micro-batch under N tokens: a shorter remainder joins its neighbour, a pair above `n_ubatch` is halved, and sizes under N are raised to N | `min=9` (the default) |

- **`n_ubatch` stays the largest size**, so the compute buffers reserved for it hold every tapered micro-batch, and the
  CUDA graphs keep one entry per split as before. The sizes repeat (64, 128, 512), so consecutive micro-batches of a
  call have the same shapes.
- **Why `min=9`:** batches of 1–8 tokens take the matrix-vector products (MMVQ and the Pascal k-token kernels on CUDA,
  the one-column path on the CPU), which cost more per token and round differently from the matrix-matrix kernels of a
  prompt (see "One-token batches" below). A plan never makes one; the plain split still does where `n_ubatch` leaves a
  small remainder (Q776).
- **Which calls:** token batches of one sequence, as llama-server's prompts with one slot. Batches of embeddings,
  several sequences, states or embedding rows (draft contexts), and non-causal or all-output (embedding) contexts keep
  the plain split. A call that no rule touches (no head, no tail, not short) keeps it too.
- **Head and tail.** By default a call is a head when nothing has been submitted since the last synchronize (the
  context's own in-flight flag, cleared by every logits read and state read), and every call is a tail. llama-server
  sets the tail per call: a call is a tail when it has outputs or when the prompt batch stopped short of `n_batch` (for a
  checkpoint, a user message or the prompt's end); a full batch is followed at once by the next one.
- **Log:** every call above 8 tokens logs its micro-batch sizes while the taper is on, e.g.
  `decode: ubatch taper: 2048 tokens, head 1, tail 0, 8 ubatches: 64x4 512x3 256`.

## Where the split happens, and why there

A prompt reaches the context as llama-server's calls (`-b 2048` tokens at most, cut 4 + `n_ubatch` and 4 tokens before
the end, where it reads a checkpoint), and each call is split into micro-batches by the memory module through the batch
splitter (`llama_batch_allocr::split_simple`, `split_equal`, `split_seq`). The taper changes only the second step:

- **The server's calls stay as they are**, so the checkpoints land where llama-server expects them (the same
  `pos_min`, `pos_max`, `n_tokens`, checked below) and restores resume where they did. Splitting a call does not add a
  host wait: the micro-batches of a call are issued back to back and overlap across the cards, up to the scheduler's
  four input copies. Splitting the server's calls instead (the note's first idea) would make each a new call, and would
  move the checkpoint cuts.
- **One place for every memory type.** The plan is a list of micro-batch ends that the splitter respects: each split
  takes at most the tokens up to the next planned end. The recurrent rule that keeps a sequence's last
  `n_rs_seq + 1` tokens in one micro-batch still applies on top of it.

What `on` does to the agent check's prompts, by llama-server's calls (`-b 2048 -ub 512`, checkpoints on):

| New tokens | The server's calls | Micro-batches without the taper | With `on` |
|---|---|---|---|
| 30 | 26 + 4 | 26; 4 | 26; 4 (unchanged) |
| 105 | 101 + 4 | 101; 4 | 64, 37; 4 |
| 450 | 446 + 4 | 446; 4 | 64 × 6, 62; 4 |
| 1,500 | 984 + 512 + 4 | 512, 472; 512; 4 | 64 × 4, 216, 128 × 4; 64 × 8; 4 |
| 3,000 | 2,048 + 436 + 512 + 4 | 512 × 4, 436; 512; 4 | 64 × 4, 512 × 3, 256, 128 × 3, 52; 64 × 8; 4 |

## What the pipeline arithmetic says (est.)

Take each card's time for a micro-batch of m tokens as p(m), the same on every card, and the note's fit for a pass of m
tokens, 0.5 s + 13.8 ms × m, plus the experts' extra cost at small m (+3.75 ms a token at 64, +1.75 at 128, the note's
section 2.3): p(64) ≈ 0.18 s, p(128) ≈ 0.28 s, p(512) ≈ 0.84 s.

- **With no limit on micro-batches in flight**, nine cards in series are a proportionate flow shop: a call of
  micro-batches m₁ … mₖ ends after Σ p(mᵢ) + 8 × max p(mᵢ), whatever their order. A head or a tail taper alone does not
  shorten a call whose body keeps 512-token micro-batches; what shortens it is a smaller largest micro-batch.
- **A lone pass is where the taper pays.** The agent check's calls that start and end with a wait (a prompt of up to
  ~520 tokens, and the 512-token call after the first checkpoint of every longer prompt) take the `short` rule: 446
  tokens as 64 × 6 + 62 cost est. 7 × 0.18 + 8 × 0.18 ≈ 2.7 s against 6.7 s; the 512-token call as 64 × 8 est. 2.9 s
  against 7.6 s. With the scheduler's four copies (at most four or five micro-batches in flight) the bound is about two
  64-token pass latencies, est. 3.3 s.
- **Long pipelined calls run at the copy limit**, a period of 2.0–3.3 s per 512 tokens where a stage is 0.84 s (the
  note's section 1.2). There the head and tail rules shorten the ragged start and end (est. 0–2 s a prompt); a call in
  small micro-batches throughout (`short=2048:128`, Q775) would shorten the 8 × p(512) term too, for +13 % of card time
  per token.
- **Over the agent check's 62 requests (est.):** the 18 of 28–36 tokens gain nothing, the 8 of 70–175 ~0.2 s each, the
  12 of 257–470 ~3.5 s each, the 24 above 516 ~4.5 s on their 512-token call and 0–2 s on the rest: ~150–200 s of
  586 s, 1.3–1.5× on prompt time, in line with the note's 1.3–1.6×.

## CPU checks

On the build machine, in containers without a GPU, at `--cpu-shares 256` ([run/cpu_checks.sh](run/cpu_checks.sh)):

| Check | Verdict | What it shows |
|---|---|---|
| build | OK | the rig tarball (`TESTS=1`, sm_61 + compute_61 PTX, 0 AVX instructions, GLIBC_2.38) and the extra tests |
| cpubuild, basebuild | OK | the branch and `-ladder2` without CUDA |
| unit | OK | the parser (good and bad specs), 13 fixed plans (llama-server's calls of 450-, 1,500- and 3,000-token prompts among them) and the clamp, raise and `min` cases, 4,000 random schedules: every plan covers the call in order, no micro-batch empty, above `n_ubatch` or under `min`, and it is empty exactly when it equals the plain split |
| batchalloc | OK | `test-batch-alloc`, the splitter's own tests, unchanged |
| taper | OK | `test-ubatch-taper` on a random qwen4exp, prompts of 100, 600, 3,000 and 9,000 tokens in llama-server's calls: 56 comparisons against off, 0 failed ([taper.tsv](taper.tsv), [taper.txt](taper.txt); below) |
| identity | OK | `test-llama-archs` on 136 architectures: with the taper off every architecture's logits equal the base's bit for bit; with `short=4096:9` every one within the base's own split tolerance ([identity_runs.tsv](identity_runs.tsv); below) |
| ctx | OK | `test-llama-context` on the random qwen4exp (state save, load, copy, rollback): 19 passed, 1 skipped, 0 failed, off and with `short=4096:9` ([ctx.txt](ctx.txt)) |
| server | PASS | llama-server, five arms against the base build: the same tokens, probabilities, checkpoints and restores ([server_verdict.txt](server_verdict.txt); below) |
| replay | OK | the rig replay against the CPU server: 24 requests, both arms, the summary written ([replay_dry_run.txt](replay_dry_run.txt)) |

### Which comparison, and what it shows

`test-ubatch-taper` decodes each prompt as llama-server does (calls of at most 2,048 tokens, cut 516 and 4 tokens before
the end, a recurrent-state checkpoint read before each of those two calls, `-ub 512`), once with the taper off and once
per arm, with every position's logits as output ("all") or the last one only, as the server asks ("last"), then 8 greedy
tokens. Arms: `on`; `small` (`head=512:16,tail=512:16,short=600:16`); `odd` (`head=100:7,tail=50:13,short=300:5`, sizes
raised to 9 by `min`); `uniform` (`short=1000000:32` on every call); `tiny1` (every micro-batch one token, `min=1`);
`on` and `odd` again with recurrent rollback snapshots (`n_rs_seq` 3, so the last 4 tokens of a call stay together).

- **Bit for bit, in every arm that keeps `min` at 9 (48 comparisons):** the logits of the prompt's last token (24 of 24
  with no value different), both checkpoints and the full sequence state after the prompt (byte for byte, 48 of 48), and
  the 8 greedy tokens. The context's logged micro-batch sizes equal the plan for every call, and the head and tail it
  used equal llama-server's rule (head after the start or a checkpoint read, tail at a cut or an output).
- **Logits of every position** differ in the last bits: nmse at most 1.6e-11, largest absolute difference 9.4e-7. The
  states are byte-identical, so nothing that feeds later tokens changed: the difference is in the work done per output
  row after the last state write (the last layer for each output row and the output head), which the CPU kernels round
  by how many rows they take at once. The base shows the same between one batch and steps of 1–8 tokens (qwen4exp:
  nmse 9.5e-13 in `test-llama-archs`).
- **The pass line is test-llama-archs' own:** nmse ≤ 1e-8 for logits (its tolerance between two builds), byte equality
  for checkpoints and states, the same greedy tokens. `tiny1` takes the one-token path (next section): its states are
  compared as floats (largest difference 1.8e-7, bar 1e-5), its last-token logits within nmse 3.1e-11, its tokens the
  same.

### One-token batches

The first draft of the plan split a 96-token call at `short=300:5` into 19 micro-batches of 5 and a last one of 1 token,
and that arm's checkpoint differed from off's by up to 1.8e-7 while the 5-token arms with rollback snapshots (whose last
micro-batch holds 4 tokens or more) matched byte for byte. The base engine does the same with the taper off: 96 tokens
in one call against 95 and then 1 leave recurrent states that differ by up to 8.9e-8, and 512 against 511 + 1 by
9.3e-8 ([taper.txt](taper.txt), the `base:` lines). A one-token batch runs a different product (the CPU's one-column
path; on CUDA every 1–8-token batch takes the matrix-vector kernels), whose rounding differs from the matrix-matrix
kernels of a prompt. Hence `min=9`: a plan never makes a micro-batch of 1–8 tokens, so a tapered prompt stays on the
kernels a plain prompt uses.

### llama-server

[run/server_check.sh](run/server_check.sh): the random qwen4exp at `-c 16384 -b 2048 -ub 512 --ctx-checkpoints 4`,
12 requests per arm (for each of 100, 600, 3,000 and 9,000 tokens: a fresh prompt, the agent's next turn on top of it,
and an edit 10 tokens before its end that restores a checkpoint), greedy, top-3 probabilities.

| Arm | Taper | Result |
|---|---|---|
| base | the `-ladder2` build | the reference: 22 checkpoints created and 3 restored at the server's cuts (516 and 4 before each prompt's end), one full re-processing where no checkpoint lies before the edit ([server_checkpoints.txt](server_checkpoints.txt)) |
| off | none | = base: tokens, top-3 probabilities, `prompt_n`, `cache_n` of all 12 requests, and the checkpoint lines (`pos_min`, `pos_max`, `n_tokens`, `n_past`) line for line; no taper line |
| on | `LLAMA_UBATCH_TAPER=on` | = base, as above; 23 taper lines, e.g. the 9,000-token prompt as `64x4 512x3 256`, then three plain calls, `128x2 36` before the first cut, `64x8` after it ([server_taper_lines.txt](server_taper_lines.txt)) |
| flag | `--ubatch-taper head=512:16,tail=512:16,short=600:16` | = base |
| req | each request's `"ubatch_taper": "head=100:7,tail=50:13,short=300:5"` | = base |
| badreq | each request's `"ubatch_taper": "bogus"` | = base, with the warning and the server's own (off) taper |

The checkpoints land where llama-server put them without the taper, and the restores resume from the same positions.

### Every architecture

`test-llama-archs -b CPU` with `LLAMA_ARCHS_LOGITS`: each architecture's logits for one batch of 128 tokens (every
position an output) and for the same tokens in steps of 1–8, written by the base build and compared by the new one
([identity.tsv](identity.tsv), [identity_runs.tsv](identity_runs.tsv), the base's own one-batch-against-steps nmse per
architecture in [base_steps.tsv](base_steps.tsv)):

| Run | Architectures | With any value different | Over nmse 1e-8 | Above the base's own steps nmse | Largest nmse |
|---|---|---|---|---|---|
| taper off | 136 | 0 | 0 | 0 | 0 |
| `short=4096:9` (every call in micro-batches of 9) | 136 | 133 | 71 | 0 | 8.0e-7 (gemma4; its own steps 1.6e-6) |
| `short=4096:1,min=1` (one token at a time, for information) | 136 | 136 | 71 | 1 (minimax-m3, 5.9e-8 against 4.9e-8) | 9.8e-7 (gemma4) |

- **Off is the old path, bit for bit**, on every architecture.
- **On, the one-batch logits move as far as a different split moves them in the base**: no architecture moves further
  than between the base's own one batch and its steps. Only the one-batch half changes (the steps, 1–8 tokens, are below
  `min` and keep the plain split). 71 architectures pass nmse 1e-8, the line test-llama-archs uses between two builds,
  because every position is an output and the per-row work rounds by the rows taken at once (the qwen4exp section
  above); test-llama-archs' own line for chunked decoding against one batch is 1e-4.
- `test-llama-archs` exits 1 in every run, the base's included: the 7 second models it builds (llama, granite, …) fail
  their comparison against the first model's file by design ([identity_exit.txt](identity_exit.txt)); the chain
  recognizes their lines and leaves them out.

## On the rig next (C218r)

[run/rig_taper.sh](run/rig_taper.sh), on the rig and detached, with [run/taper_replay.py](run/taper_replay.py) and the
agent check's [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) beside it: two loads of the taper build with the
agent check's line (`-c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio -fa on -bs`, F16 cache). The
taper is off on the server and set per request, so each load runs both arms without a second load (5.4 min each):

| Load | Server | What runs |
|---|---|---|
| A | `--ctx-checkpoints 4`, as the agent check | the replay of the check's 62 prompts in four passes, taper off, on, on, off; then 4 fresh prompts of 2,049 tokens, each with the taper off and on (the order alternating) |
| B | `--ctx-checkpoints 0`: no checkpoint drains (Q770's no-code test) | the replay in two passes, off and on, and the same fresh prompts |

- **The replay** follows [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (Q218's arm `hcq8`): per task, the
  shared 1,549-token prefix is evaluated by an unmeasured request (the check had it in the slot), then each turn sends
  its `prompt_n` new tokens of Python standard-library source on top of the task's history and asks for 8 greedy tokens.
  The context is shallower than in the check by the replies' length (prompt time showed no depth effect below ~6.8k
  tokens; `PAD=1` adds the missing tokens with unmeasured requests). The same tokens go out in every pass.
- **Reading it:** `load-A/summary.txt` gives prompt time and tok/s per arm, on/off per turn paired by task and turn, by
  the note's size bins, and the fresh prompts; A's fresh prompts against B's give the drains' cost (Q770, predicted
  ≥ 5 s at 2k tokens), and B with the taper on approximates I1 + I2. `requests.tsv` has every request,
  `checkpoints.txt` and `taper_lines.txt` the server's own lines, `memavail.txt` the host's free memory every 2 s
  (a watchdog stops the server below 150 MiB).
- **More schedules** go in the same load as passes: `PASSES_A="off on short=2048:128 off short=2048:128 on"` adds Q775.
- Est. 1.5 h (load A ~1 h, load B ~0.5 h).

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | the chain's verdict per check |
| [taper.tsv](taper.tsv) | `test-ubatch-taper`'s model part: per arm, prompt length and output mode, the comparison against off |
| [taper.txt](taper.txt) | its console lines (the unit summary, the `base:` lines, one line per comparison) |
| [identity.tsv](identity.tsv), [identity_runs.tsv](identity_runs.tsv), [base_steps.tsv](base_steps.tsv), [identity_exit.txt](identity_exit.txt) | `test-llama-archs` per architecture and run (the base, the new build off, `short=4096:9`, `short=4096:1,min=1`) against the base's logits; per run the counts; the base's own one-batch-against-steps nmse per architecture; each run's summary line |
| [server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt), [server_checkpoints.txt](server_checkpoints.txt), [server_taper_lines.txt](server_taper_lines.txt) | llama-server per arm and request (prompt_n, cache_n, tokens), the verdict, the base arm's checkpoint lines and the `on` arm's taper lines |
| [ctx.txt](ctx.txt) | `test-llama-context` off and with `short=4096:9` |
| [replay_dry_run.txt](replay_dry_run.txt), [replay_dry_run.tsv](replay_dry_run.tsv) | the rig replay's dry run against the CPU server, two tasks (that it runs and reports; the random model on a CPU has no pipeline, so its speeds mean nothing) |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the build-and-check chain |
| [run/server_check.sh](run/server_check.sh) | the llama-server check |
| [run/rig_taper.sh](run/rig_taper.sh), [run/taper_replay.py](run/taper_replay.py) | the rig run and its replay |

## Commands

The chain on a build machine (a clone of llama.cpp with the fork as a remote in `$WORK/src`, this repository in
`$REPO`), CPU only:

```bash
WORK=<work dir> REPO=<this repository> BRANCH=pascal-p106-v3-b12076-taper REMOTE=fork BASE=a330672c1 \
  JOBS=6 THREADS=8 LENS=100,600,3000,9000 bash docs/data/2026-11-04_prompt_taper/run/cpu_checks.sh
# summary.tsv: one line per check; checks/<check>/ the logs
```

The model check alone, with a random qwen4exp from `test-llama-archs -a '^qwen4exp$' -s 1 -o <dir>`:

```bash
test-ubatch-taper -m <dir>/qwen4exp-moe.gguf -t 8 --lens 100,600,3000,9000 --tsv taper.tsv
```

On the rig (C218r), from a folder with `rig_taper.sh`, `taper_replay.py` and `turns.csv`:

```bash
BUILDS=<builds> BUILD=<the taper build> MODEL=<the hcq8 file's first shard> OUT=<work>/out \
  setsid nohup bash rig_taper.sh > <work>/rig_taper.out 2>&1 < /dev/null &
# done when <work>/out/DONE exists; <work>/out/load-A/summary.txt, load-B/summary.txt
```

## Build and md5

| File | Bytes | md5 |
|---|---|---|
| `llama.cpp-20261104-7b1632aee-pascal-p106-v3-b12076-taper.tar.gz` (`scripts/build_llamacpp_docker.sh`, `TESTS=1`) | 346,502,579 | `785e0b9595e396823b0d22d44c951cdd` |
| `extra-7b1632aee.tar.gz` (`test-ubatch-taper`, `test-llama-archs`, `test-llama-context` from the same tree) | — | `9e450cb38dea56c34826d39da149ae0e` |

## Provenance and licence

The engine change is llama.cpp's licence (MIT). The tables come from random-weight models that `test-llama-archs`
generates and from token ids the checks generate; the rig replay's text is the Python standard library's source on the
rig (PSF licence), tokenized there and not stored. Request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv).
