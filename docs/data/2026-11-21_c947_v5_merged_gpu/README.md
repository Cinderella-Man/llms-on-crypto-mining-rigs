# The merged v5 engine on three newer cards: the GPU half of its check (C947; 2026-11-21)

Registry row [Q1687](../../../quants/experiments.md) (the GPU half of the night's branches; this is its first point,
C947). The processor half is [2026-11-21_c908_v5_merged_build](../2026-11-21_c908_v5_merged_build/README.md).

**Passed.** The card suite passes 26,644 of its 26,645 cases (every case but the repeated-expert-id ones, the same
cases as pull request 4's run on these cards); the one failure is the known random rounding tie in a copy to Q8_0.
On these cards the suite and the loads ran llama.cpp's own matrix-vector kernels for products of 1 to 8 tokens: the
fork's Pascal kernels for them turn themselves on only on compute 6.1 cards (the P106). Forced on, with a P106's launch
shapes, they pass the suite's `MUL_MAT`, `MUL_MAT_ID` and fused cases too; one fused case there fails on a router
tie with them off as well ([2026-11-23_c968_pascal_mmvq_suite](../2026-11-23_c968_pascal_mmvq_suite/README.md)). On
three cards
the merged engine runs the three release files exactly as v5 before the contributions does: every greedy reply is the
same token for token, Flash-Next's draft proposes and lands the same tokens, and every card holds the same memory to
the MiB. The card-to-card mailbox test, which holds the one piece of new card code (pull request 4's gate), puts every
byte through in all 45 of its settings with no timeout, with the same per-setting result as pull request 4's own run.
Nothing here is timed: the speed half of the check is a P106 rig row.

## What ran where

| | |
|---|---|
| New build | `pascal-p106-v5` `3dfad876f`: v5 with the five contributed pull requests and the `LLAMA_PIPE_GRAMMAR` fix ([its build](../2026-11-21_c908_v5_merged_build/README.md)). Rig tarball md5 `0b5b983e22d7296ff44fe363d55f61bb`; `llama-server` `5b330eccef892b7dd9f4faf1027319c3`, `test-backend-ops` `526d0231d933867a14e51f11050c1b79` |
| Base build | `602691153`: v5 before the contributions ([its build](../2026-11-17_c854b_upstream_newest/README.md)); `llama-server` `86607661d5a19d1a90d19ee4db10c90b` |
| `test-hop-mailbox` | not in the rig tarball: built from the same tree (`3dfad876f`, clean) with `cmake --build build --target test-hop-mailbox`, nothing else recompiled; md5 `fbb5a351a5e2416ca0b881a0a472ff3e`. `tests/` is the same in both builds |
| Cards | three RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT (the way [pull request 4's GPU tests](https://github.com/Cinderella-Man/miner-llama/pull/4) were run). A CUDA preflight passed on all three first |
| Settings | every switch a preset does not set at its default; no clock, power or other card setting changed. At these defaults the fork's Pascal matrix-vector kernels stay off on these cards: `GGML_CUDA_MMVQ_PASCAL` and `GGML_CUDA_MMVF_PASCAL` at 1 turn them on only on compute 6.1 cards (the P106), so here llama.cpp's own matrix-vector kernels computed the products of 1 to 8 tokens, in the card suite and in the loads |

Between the two builds 13 files changed. The only card code among them is `ggml/src/ggml-cuda/hop-mailbox.cu` (pull
request 4), which runs only with `GGML_CUDA_HOP_MAILBOX=1`, a switch no preset sets. Pull requests 3, 5, 6 and 7 and
the fix change server and model code: the temperature-0 replay after a speculation checkpoint, `--spec-ckpt-vram`,
the draft shortlist in the `qwen35moe` MTP head (Ornith's kind; its preset sets no draft), `--mtp-device`,
`--pipe-guess mtp`, `--pipe-mtp-walk` and `LLAMA_PIPE_GRAMMAR`, all off unless asked for.

## The release files on both builds

Each file with its v5 preset ([release/flash-next-v5](../../../release/flash-next-v5/settings.env),
[flint-v5](../../../release/flint-v5/settings.env), [ornith-v5](../../../release/ornith-v5/settings.env)), the layer
split changed from the rig's nine cards to the three (`-ts 1,1,1`), everything else as the preset has it: Flash-Next's
40 switches and its MTP draft, Flint's `--pipe-depth 4` pipeline, Ornith's unfused expert products, the staged card
hops and the Q8_0 residual between cards where the preset sets them. Per file, the new build first, then the base:
one llama-server, its memory per card 20 s after it answered, then C854b's two greedy requests (`/completion`,
temperature 0, top-k 1, 256 tokens, no prompt cache, token ids returned): a coding request and a prose paragraph
([2026-11-18_c854b_p106_check](../2026-11-18_c854b_p106_check/README.md#part-3-the-release-files-on-both-builds)).

The files: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (UkisAI's Swift-1.5 GSQ-RCO Q2_0 with its n-gram table as the third
shard), Flint Pascal Q8 GGUF (`p065`) and Ornith-1.5-122B Pascal N2P GGUF (`valid1`), the published files with the
release header. **Flash-Next's draft here is Swift-1.5's own MTP layer**
([2026-11-10_swift_own_mtp_head](../2026-11-10_swift_own_mtp_head/README.md)), not the preset's Gyro draft: that file
was on the rigs only, which were switched off. Swift's own head is written with the Gyro draft's tensor names, order,
shapes and types, and 24 of its 34 tensors are the Gyro draft's bytes, so it takes the same code path; both builds
used the same file.

**Every reply is the same on both builds** ([reply-agree.tsv](reply-agree.tsv), [load-replies.tsv](load-replies.tsv)):

| File | Coding request | Prose | Drafted and accepted (coding; prose) |
|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | the same 256 token ids, 748 characters | the same 256, 1,206 characters | 236 and 176; 276 and 162, on both builds |
| Flint Pascal Q8 GGUF | the same 256, 1,064 | the same 256, 1,244 | — |
| Ornith-1.5-122B Pascal N2P GGUF | the same 256, 953 | the same 256, 1,233 | — |

On these cards the replies' small products ran on llama.cpp's own matrix-vector kernels on both builds, not on the
fork's Pascal kernels (Flash-Next's preset switches act inside those kernels, so they changed nothing here); the two
builds' matrix-vector kernel files are the same. All six replies ran to the 256-token limit. Five of them open with
the same 60 characters as the P106 rig's replies to the same requests in C854b; Ornith's prose parts from the rig's after "A hash map handles collisions ". The texts are
not compared with the rig's beyond that, because these cards add up their sums in another order.

**Every card holds the same memory on both builds** ([load-memory.tsv](load-memory.tsv)):

| File | MiB per card 20 s after it answered, cards 0–2 (both builds) | After the two requests (both builds) |
|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 13,917 13,191 16,009 | 13,999 13,269 16,097 |
| Flint Pascal Q8 GGUF | 15,953 17,585 13,851 | 16,015 17,655 13,927 |
| Ornith-1.5-122B Pascal N2P GGUF | 15,425 16,641 15,605 | 15,517 16,733 15,681 |

The server took 46 and 40 s to come up with Flash-Next (direct I/O, so no page cache), 397 and 10 s with Flint and
447 and 10 s with Ornith (the new build read each from a hard disk first, the base then found it in the page cache).
These are load diagnostics, not results.

## The mailbox test

`test-hop-mailbox` on all three cards: three card boundaries (0 → 1, 1 → 0, 1 → 2), long matrix products holding one
card's stream, the scheduler's guard event before each copy, and every round's bytes compared with the source. The
test's nine settings ran with pull request 4's gate unset, then with the gate set four ways, the arms of
[pull request 4's review](https://github.com/Cinderella-Man/miner-llama/pull/4) on the same cards
([mbox.tsv](mbox.tsv), [mbox-arms.tsv](mbox-arms.tsv)):

| Arm (exported) | Rounds | Lost bytes, timeouts, CUDA errors | The test's hop-count checks | Pull request 4's run |
|---|---:|---|---|---|
| gate unset | 96 | none | **27 of 27 pass** | the same, setting by setting |
| `GGML_CUDA_HOP_MAILBOX_GATE_KB=16 GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB=8192` | 96 | none | 26 of 27 (`max16k-staged`: 96 hops by mailbox, the test expects 42) | the same |
| `GGML_CUDA_HOP_MAILBOX_GATE_KB=16` (buffer 4,096 KiB) | 96 | none | 19 of 27 (90 against 96 in the eight mailbox settings) | the same |
| `GGML_CUDA_HOP_MAILBOX_GATE_KB=1 GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB=64` | 96 | none | 19 of 27 (66 against 96) | the same |
| `GGML_CUDA_HOP_MAILBOX_GATE_KB=64` (buffer 4,096 KiB) | 480 | none | 19 of 27 (450 against 480; `max16k-staged` 330) | the same |

The count misses in the gate arms are the test's own expectations, which do not know about the gate: gated hops count
as mailbox hops, and hops too big for the gate's buffer fall back to the peer copy. They are the same numbers in every
setting as in pull request 4's run.

## The card suite

`test-backend-ops test -b CUDA0 -p "^(?!.*dup=1)"`: every operator case the CUDA backend supports, computed on one
card and judged against the processor, every case but the repeated-expert-id ones (`dup=1`). It runs one card at a
time, so it cannot reach the hop code. It checks every kernel the new build runs on these cards, which leaves out the
fork's Pascal matrix-vector kernels (`mul_mat_vec_q_pascal` with its multi-token, k-column and expert kernels, and the
BF16 kernels): they turn themselves on only on compute 6.1 cards, so here llama.cpp's own matrix-vector kernels
computed those products. The fork's kernels are checked forced on, with a P106's launch shapes, in
[2026-11-23_c968_pascal_mmvq_suite](../2026-11-23_c968_pascal_mmvq_suite/README.md). Split by operator over the three
cards, 10:01–10:40 ([bops.tsv](bops.tsv)):

| Share | Card | Cases | Passed | Failed |
|---|---|---:|---:|---|
| `MUL_MAT` | 0 | 4,963 | 4,963 | — |
| `MUL_MAT_ID` | 1 | 2,254 | 2,254 | — |
| every other operator the support listing names (121) | 2 | 14,154 | 14,153 | `CPY` f32 → q8_0 at `[2560,4,512,1]`, ERR 0.000000000 > 0.000000000 |
| the fused cases test mode adds (43 operators: `MUL_MAT_VEC_FUSION`, `TOPK_MOE`, `MUL_MAT_GLU`, `RMS_NORM_MUL_ROPE` and others) | 2, beside the third share | 5,274 | 5,274 | — |
| **all** | | **26,645** | **26,644** | |

- **The one failure is a random rounding tie.** Converting a float to Q8_0 or another 4–8-bit type can land exactly on
  a rounding boundary, where the card and the processor may round apart; which cases of `CPY` at `[2560,4,512,1]` fail
  changes from run to run, with v5's own binary as with any other (pull request 4's run: v5 failed 4, 3 and 3 of 360
  `CPY` cases in three repeats, its own build 3, 4 and 3; its full run 3, all in this shape). Here it is 1 of 360.
- **Every case ran.** Per operator, the four shares ran exactly as many cases as pull request 4's full run of the same
  test file (`tests/` is the same in both builds): 166 operators, 26,645 cases ([bops-coverage.tsv](bops-coverage.tsv)).
  The support listing names 123 operators; the fourth share holds the 43 fused-case operators that only test mode
  runs ([run/ops-testmode-only.txt](run/ops-testmode-only.txt)), started by hand at 10:09 once the first three shares
  showed the gap.
- No illegal memory access, no crash, every share's exit code 0 except the third's (1, from that failure).
- **Not run: 278 of the fork's own cases**, the 162 `MUL_MAT` and 116 `MUL_MAT_ID` products with activations large
  enough that a block's sum passes FP16's range (`test_mul_mat_large_act`). They declare themselves supported only
  where the fork's Pascal kernels are on, so here they were skipped as not supported; forced on, all 278 pass (C968).
- **Not run: the repeated-expert-id cases.** The first of them stops with an illegal memory access in llama.cpp's own
  expert kernel on these cards, with v5's own binary too (pull request 4's run); upstream disables them for CUDA, and
  they exercise single-card expert kernels the two builds share.

## Files

| File | What it holds |
|---|---|
| [reply-agree.tsv](reply-agree.tsv) | per file and request: the same text, the same token ids, the same drafted and accepted counts on both builds |
| [load-replies.tsv](load-replies.tsv) | every reply: tokens, characters, md5 of the text and of the token ids, drafted and accepted tokens, stop reason, opening |
| [load-memory.tsv](load-memory.tsv) | MiB per card after each load and after its two requests |
| [mbox.tsv](mbox.tsv), [mbox-arms.tsv](mbox-arms.tsv) | the mailbox test per arm and setting, beside pull request 4's run |
| [bops.tsv](bops.tsv) | the card suite per share: exit code, cases passed of run, failing cases |
| [bops-coverage.tsv](bops-coverage.tsv) | cases per operator in pull request 4's full run and in the four shares together |
| [run/run.sh](run/run.sh) | the driver: the mailbox test, the release files on both builds, the card suite in three shares |
| [run/bops_testmode_only.sh](run/bops_testmode_only.sh), [run/ops-testmode-only.txt](run/ops-testmode-only.txt) | the card suite's fourth share: the 43 fused-case operators test mode runs but the support listing does not name |
| [run/tables.py](run/tables.py) | the tables from a copy of the driver's work folder (standard library, `python3 -I`) |

## Commands

On the build machine, with the two rig builds unpacked, the release files and Swift's own MTP head on its disks, and
copies of `run/` and `release/*-v5/settings.env` in a work folder (the three cards free, a CUDA preflight first):

```bash
WORK=<work folder> NEW=<3dfad876f build>/bin REF=<602691153 build>/bin HOPTEST=<work folder>/bin/test-hop-mailbox \
REPO=<work folder>/repo FN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> \
DRAFT=<mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf> FLINT_Q8=<flint-pascal-Q8.gguf> \
ORNITH=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> MOUNTS='<folders holding the builds and models>' \
DOCKER_ARGS='--cpu-shares=256' setsid nohup bash run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the fourth share of the card suite, once the support listing has run (here on the third card, beside the last share)
WORK=<work folder> NEW=<3dfad876f build>/bin MOUNTS='...' CARD=2 setsid nohup bash run/bops_testmode_only.sh &
# the tables, from a copy of the work folder (pull request 4's mailbox logs and full test log as references)
python3 -I run/tables.py <work copy> <this folder> <pr4 mbox folder> <pr4 full test log>
```

Pull request 4's run behind the reference columns: the same arms of `test-hop-mailbox` and the full
`test-backend-ops test -b CUDA0 -p "^(?!.*dup=1)"` on these cards, 2026-11-20, with its own build (`602691153` plus
pull request 4).
