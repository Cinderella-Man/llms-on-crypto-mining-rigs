# The prompt taper's 9-token floor kept after the in-call checkpoint cuts (C905, Q1638; 2026-11-21)

What this is: a change to the v5 engine's prompt taper (`--ubatch-taper`), built and checked on a build machine's CPU.
The taper reads a prompt call in smaller micro-batches; Flash-Next's v5 preset reads calls of up to 1,400 tokens in
micro-batches of 64 (`short=1400:64`). It never plans a micro-batch under 9 tokens, because 1 to 8 tokens run the
matrix-vector kernels, which cost more per token. But with `--ctx-checkpoints-incall` (also in the preset) the server
cuts each call at its two checkpoints, 516 and 4 tokens before the call's end, after the taper has planned it. Where
the first cut lands a few tokens short of a planned micro-batch end, a piece of 4 to 6 tokens is left
([C222](../2026-11-20_c222_flash_microbatch/README.md) found it). The new key `cuts=1` in the taper's setting plans
each stretch between the cuts on its own. Fork branch `pascal-p106-v5-taper-cuts` (`1720f26d3`, two commits on
`pascal-p106-v5` `3dfad876f`). Registry row Q1638 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **With `cuts=1` no call keeps a piece under 9 tokens beside a cut.** On the 52 call sizes of R225's replay under
  the v5 preset's `short=1400:64`, five calls had one (577, 583, 775, 896 and 1,086 tokens: a piece of 4 to 6 tokens);
  with `cuts=1` none has. The same holds for the four other settings C222 and R225 ran. The only pieces under 9 left
  are the call's last 4 tokens, which the server's own checkpoint cut makes a stretch of 4.
- **Each call it changes reads in one micro-batch fewer.** Under `short=1400:64` it changes 14 of the 52 call sizes
  (14 of the 73 requests of a replay pass), every one by one micro-batch fewer, and the stretch between the two cuts
  is exactly eight micro-batches of 64. For example, a 577-token call goes from `61 5 62 64x6 61 4` to `61 64x8 4`.
- **The check reproduces the rig.** Before the change, the CPU check gives the exact micro-batches the rig's server
  logged in C222's and R225's replays, for every one of the 261 setting and call-size pairs.
- **Nothing changes unless `cuts=1` is set.** Without it every check gives the base's result: the logits of 148
  random models bit for bit, ctest test by test, the server's replies and its micro-batches line for line, and the GPU
  code, all 8,836 kernels byte for byte.
- **The answers stay the same.** With `cuts=1` the server gave the same tokens and the same top-3 probabilities to the
  digit on 16 requests sized to hit the cuts, one of them restored from a checkpoint the server took inside a call.
  Checkpoints and states were byte for byte the same in every model-level comparison.
- **Not yet on a P106.** Whether one micro-batch fewer on 14 calls a pass is measurably faster is for the rig to say:
  R225's replay with `short=1400:64` against `short=1400:64,cuts=1`, a separate row (C930).

## What changed in the code

| File | Change |
|---|---|
| `src/llama-batch.h`, `.cpp` | `llama_ubatch_taper` gets `cuts` (spec key `cuts=0` or `cuts=1`, default 0, printed as `,cuts=1`). `llama_ubatch_taper_plan` takes the call's cuts: with `cuts=1` it lays out each stretch between them with the rules' sizes (the head's remainder first, the tail's and a short call's last), then applies the `min` floor inside each stretch; a stretch shorter than `min` is one piece. A plan equal to the split without one (n_ubatch at a time, per stretch) is dropped, as before. Without `cuts=1`, or with no cut inside the call, the code path is the old one, unchanged |
| `src/llama-context.cpp` | the decode passes the cuts it already sets for the call (the caller's `llama_set_ubatch_cuts` and the in-call checkpoint fills) to the plan |
| `include/llama.h`, `common/arg.cpp` | the new key in the API comment and in `--ubatch-taper`'s help |
| `tests/test-ubatch-taper.cpp` | unit cases and a 20,000-call sweep for `cuts=1`; model arms that decode each call whole with the server's cuts inside it and its checkpoints filled there; `--plan-check`, which logs the splitter's micro-batches for given call sizes |

The setting goes where the taper's always went: `--ubatch-taper short=1400:64,cuts=1`, `LLAMA_UBATCH_TAPER`, or a
request's `"ubatch_taper"`. Host code only.

## The plan check

`test-ubatch-taper --plan-check` decodes one call of each size on a random model as llama-server's prompt call with
in-call checkpoints (nothing in flight before it, waited for, the cuts 516 and 4 tokens before its end, `-b 4096 -ub
512`), with the recurrent rule of Flash-Next's MTP draft of three (`n_rs_seq` 3: a stretch's last four tokens stay
together), and logs the micro-batches the context ran. The sizes are every call size of C222's and R225's replays plus
520 and 1,400. [run/plan_table.py](run/plan_table.py) sets them beside the rig's own logged lines
([plans.tsv](plans.tsv)):

| Setting | Call sizes | CPU plan = rig's | Calls with a piece under 9 tokens beside a cut, before (size@offset) | After (cuts=1) | Calls whose plan changes | Micro-batches over the call sizes, before and after |
|---|---:|---:|---|---:|---:|---:|
| `short=600:64` (v4.0's) | 53 | 53 of 53 | 2: 577 (5@61), 583 (4@63) | 0 | 4 | 269, 265 |
| `short=1023:64` | 52 | 52 of 52 | 4: 577 (5@61), 583 (4@63), 775 (4@255), 896 (5@380) | 0 | 10 | 335, 325 |
| `short=1400:64` (v5's) | 52 | 52 of 52 | 5: 577 (5@61), 583 (4@63), 775 (4@255), 896 (5@380), 1,086 (6@570) | 0 | 14 | 419, 405 |
| `short=2048:64` | 52 | 52 of 52 | 7: 577 (5@61), 583 (4@63), 775 (4@255), 896 (5@380), 1,086 (6@570), 1,534 (6@1018), 1,985 (5@1469) | 0 | 21 | 591, 570 |
| `short=2048:128` | 52 | 52 of 52 | 3: 775 (4@255), 896 (5@380), 1,534 (6@1018) | 0 | 21 | 353, 332 |

Every one of the 70 changed calls reads in exactly one micro-batch fewer. The v5 preset's 14:

| Call | Before (the rig's log, R225) | After (`cuts=1`) |
|---:|---|---|
| 569 | `53 11 64x7 53 4` | `53 64x8 4` |
| 577 | `61 5 62 64x6 61 4` | `61 64x8 4` |
| 583 | `63 4 61 64x6 67 4` | `67 64x8 4` |
| 599 | `64 19 45 64x7 19 4` | `64 19 64x8 4` |
| 610 | `64 30 34 64x7 30 4` | `64 30 64x8 4` |
| 775 | `64x3 63 4 61 64x6 67 4` | `64x3 67 64x8 4` |
| 803 | `64x4 31 33 64x7 31 4` | `64x4 31 64x8 4` |
| 815 | `64x4 43 21 64x7 43 4` | `64x4 43 64x8 4` |
| 896 | `64x5 60 5 63 64x6 60 4` | `64x5 60 64x8 4` |
| 936 | `64x6 36 28 64x7 36 4` | `64x6 36 64x8 4` |
| 1,086 | `64x8 58 6 64x7 58 4` | `64x8 58 64x8 4` |
| 1,130 | `64x9 38 26 64x7 38 4` | `64x9 38 64x8 4` |
| 1,300 | `64x12 16 48 64x7 16 4` | `64x12 16 64x8 4` |
| 1,359 | `64x13 11 53 64x7 11 4` | `64x13 11 64x8 4` |

Where the old plan's 64-token boundary sat a few tokens past the first cut, the recurrent rule stretched the first
piece after the cut to 5 tokens (`61 5 62`), or pulled the cut's last 4 tokens into a piece of their own (`63 4 61`).
Without that rule (`n_rs_seq` 0, [plans-rs0.tsv](plans-rs0.tsv)) the same calls had a piece of 3 to 6 tokens. With
`cuts=1` the context ran exactly the planned micro-batches on every call, with the rule and without it.

## The CPU checks

All on a build machine's CPU, no GPU work; the CUDA build's binaries run their CPU backend against the driver stub.

| Check (head `1720f26d3` against base `3dfad876f`) | Result |
|---|---|
| Compiler warnings | rig build 38 distinct on both, native CPU build 26 on both: the same sets, none new |
| ctest, 75 tests, no network | 71 pass on both, test by test the same (`test-ubatch-taper`'s unit part among them). Fail on both: `test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales`, the base's known four ([C908](../2026-11-21_c908_v5_merged_build/README.md)) |
| `test-llama-archs -b CPU -s 1` | native: 153 of 153 on both, 148 of 148 logit files the base's bit for bit; rig build against the base's rig build: 153 of 153, 148 of 148 the same |
| Hand-run tests (hop-cast, route-once ×2, bytes-switches ×2, moe-ids, thread-safety, context-model, eval-callback) | all pass on both |
| `test-ubatch-taper`, unit part | the old part as before (13 fixed plans, 4,000 random schedules); `cuts=1`: the parser, 9 fixed calls with the server's cuts (C222's 577-, 583-, 610- and 1,534-token calls among them, before and after), 20,000 random calls with cuts: every plan covers the call in order, holds each cut as a micro-batch end, no piece above n_ubatch, none under `min` unless the cuts make a whole stretch shorter, the splits take it as it is; without `cuts=1` the cuts are not read. 1,178 of those calls had a piece under `min` beside a cut before |
| `test-ubatch-taper`, model part (random qwen4exp MoE, prompts of 100, 577, 583, 600, 1,534, 3,000 and 9,000 tokens, all-position and last-token logits; [taper.tsv](taper.tsv)) | 238 comparisons against the taper off, 0 failed. The 12 new in-call arms (each call whole with the server's cuts inside it and its two checkpoints filled there, `n_rs_seq` 0 and 3; `short=1400:64` with and without `cuts=1`, `on` and two odd schedules with it): logits within nmse 1e-8, checkpoints and states byte for byte, the same 8 greedy tokens, the micro-batches as planned, and with `cuts=1` none under 9 but whole stretches. With every logit kept and Flash-Next's recurrent rule, `short=1400:64,cuts=1` gave the taper-off logits bit for bit on the 577- and 583-token prompts, where `short=1400:64` alone differed in the last bits on 23,678 and 38,757 values (its 5- and 4-token pieces) |
| llama-server on the CPU ([server.tsv](server.tsv), [calls.tsv](calls.tsv)) | below |
| `cuobjdump -sass` of `llama-server` (base `5b330ecc…`, head `0cdda341…`) | all 8,836 kernels identical, and the 176 kernels built a second time with `kq_fp32` identical: no card code changed |

**The server check.** Both rig builds' llama-server on a random qwen4exp MoE model with one MTP block and that block
alone as its draft, with the v5 preset's prompt line: `-b 4096 -ub 512`, 4 checkpoints in device stores taken inside
the calls (`--ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall`), the MTP draft of three. Sixteen
requests ([run/client.py](run/client.py)), prompts of token ids so each call has a known size: fresh prompts of 30 to
3,313 tokens (C222's sizes where the cuts fall, and around them), a continuation that reads a reply and 600 new tokens
on top of a cached 577-token prompt, and an edit 1,100 tokens into a 1,534-token prompt, which restored the checkpoint
the server took inside the call 516 tokens before its end.

| Run | Against | Result |
|---|---|---|
| head, taper unset | base, taper unset | the same tokens, top-3 probabilities, `prompt_n`, `cache_n` and restored checkpoint on all 16; no taper line in either |
| head, `short=1400:64` | base, `short=1400:64` | the same, and all 16 calls' micro-batches line for line |
| head, `short=1400:64,cuts=1` | head, `short=1400:64` | the same tokens, top-3 probabilities to the digit, `prompt_n`, `cache_n` and restored checkpoint |
| head, server at `short=1400:64`, each request sending `"ubatch_taper": "short=1400:64,cuts=1"` | head, `short=1400:64,cuts=1` set on the server | everything the same, the micro-batches line for line |

Calls with a piece under 9 beside a cut: 5 of 16 with `short=1400:64` on both builds (577, 583, 775, 896, 1,086),
0 of 16 with `cuts=1` either way. Every run: 28 checkpoints taken inside calls, 1 restored, no error line.

## Next: the rig (C930)

R225's replay ([C222's run/c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh), unchanged) on this branch's rig
build with Flash-Next's v5 preset: one load, a warm-up pass, then `short=1400:64` and `short=1400:64,cuts=1`
alternated per request (4 and 3 passes, the control first and last), est. 40 minutes. It is QUEUES row C930 with its
exact commands; the change goes into the preset only on a rig verdict and the owner's word.

## Files

| File | What |
|---|---|
| [plans.tsv](plans.tsv) | per setting and call size: the rig's logged micro-batches, the CPU check's before and after, whether before is the rig's, the micro-batch counts, the pieces under 9 beside a cut |
| [plans_summary.md](plans_summary.md) | the summary table above, as `plan_table.py` prints it |
| [plans-rs3.tsv](plans-rs3.tsv), [plans-rs0.tsv](plans-rs0.tsv) | the plan check's own output, with Flash-Next's recurrent rule (`n_rs_seq` 3) and without (0): per spec and call size the cuts, the logged and the planned micro-batches, the pieces under `min`, the floor |
| [taper.tsv](taper.tsv) | `test-ubatch-taper`'s model part: per arm, prompt size and output mode, the comparison against the taper off |
| [server.tsv](server.tsv), [calls.tsv](calls.tsv) | the server check's verdicts, and every call's micro-batches per run |
| [run/run.sh](run/run.sh) | the build machine's driver: sources, a fail-fast build of the taper test, the rig tarball, the CPU test set, test models, the taper test, the plan check, the rig build's logits, the server check, the SASS comparison |
| [run/client.py](run/client.py), [run/compare.py](run/compare.py) | the server check's requests and its comparison |
| [run/plan_table.py](run/plan_table.py) | the plan table from the rig's lines and the plan check |

The CPU test set's driver is [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), the build script
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh), the SASS hashes
[sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py).

## Commands

On the build machine (CPU only), with the base's work folder from C908 as the seed (its clone, rig build, build log
and compiler caches), the scripts above in `$W/repo`:

```bash
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> \
  BASE_RIG=<3dfad876f's rig build folder> BASE_RIG_LOG=<its build log> REPO=$W/repo \
  RIG_CCACHE=<rig compiler cache> CPU_CCACHE=<CPU compiler cache> LOCK=<flock file> JOBS=8 HF_UI_VERSION=b12483 \
  bash $W/repo/run.sh > $W/logs/run.out 2>&1
```

The run took 03:20–03:39 on 2026-11-21 (the rig build 2 min 44 s from warm caches, the CPU test set 5 min 47 s). The
first start stopped at the fail-fast step on a wrong expectation in a unit case (the 1,534-token case was under
`short=1400:64`, which leaves it to the plain split; C222 logged its 64s under `short=2048:64`), fixed in the second
commit. The SASS step ran again on its own (`STEPS=sass`) with the published comparison.

The tables, from this folder:

```bash
python3 -I run/plan_table.py --rig ../2026-11-20_c222_flash_microbatch ../2026-11-20_r225_flash_taper1400 \
  --cpu plans-rs3.tsv --out plans.tsv > plans_summary.md
```

Rig tarball `llama.cpp-20261121-1720f26d3-pascal-p106-v5-taper-cuts.tar.gz` (`TESTS=1`, `HF_UI_VERSION=b12483`):
909,394,215 bytes, md5 `f4102df5e469492ea57fa7a0517461d8`; `llama-server` md5 `0cdda3417d39cea7886bc9b0555346fd`,
`test-ubatch-taper` `007df0fd302d207d8fda928387fba241`, `test-backend-ops` `9dce1d9aba3f64348d65b7ae39d8cee9`.

## Provenance and licence

The engine is Miner Llama (llama.cpp's MIT licence). The test models are random weights written by the engine's own
`test-llama-archs`. The rig lines compared are the published `taper_lines.txt` and request lists of
[C222](../2026-11-20_c222_flash_microbatch/README.md) and [R225](../2026-11-20_r225_flash_taper1400/README.md). Tables
and scripts here: the repository's licence.
