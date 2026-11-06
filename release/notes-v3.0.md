# Miner Llama {{VERSION}}: llama.cpp for P106 mining rigs and other Pascal GPUs

A llama.cpp build for NVIDIA GPUs of compute capability 6.1: the P106 and P104 mining cards, the GTX 10 series and the
Tesla P4/P40. Built from commit `{{COMMIT}}` of this fork (branch `pascal-p106-v3-b12076-ladder5`, on upstream
llama.cpp b12076), for Linux on x86-64 without AVX, so it runs on mining rigs' Celeron hosts. This release is tuned and
measured on three models on nine P106-090 cards; other compute-6.1 cards run the same code but were not measured.
Measurements, settings and the scripts behind every number:
[llms-on-crypto-mining-rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

## Speed

**Qwen3.8-Flash-Next, a 125B-parameter model, at 20.40 tok/s for a coding agent on nine 6 GB mining cards and a 4 GB
host.** On 2026-11-03 it ran at 6.4 tok/s on stock llama.cpp and 7.8 on v2.0's code. Flint and Ornith-1.5-122B run on
the same build at their v2-era speed or better.

A coding agent ([pi](https://github.com/earendil-works/pi)) fixing twelve small bugs, Python and JavaScript, real test
runs, the context growing every turn, on nine P106-090 cards (one PCIe 1.0 lane each, a 2-core Celeron with 4 GB of
RAM). Generation in tokens per second over every request:

| Model, file | Stock llama.cpp b12076 | Miner Llama v2.0 | **{{VERSION}}**, the model's preset |
|---|---|---|---|
| **Qwen3.8-Flash-Next**, UkisAI's Swift-1.5 GSQ-RCO Q2_0, 65,536-token context | 8.20, 9 of 9 (3 not reached) | 7.8, 9 of 9 (no draft) | **20.40, 12 of 12** |
| **Flint**, Unsloth Q8_0, 262,144-token context | 14.4, 10 of 12 (65,536-token context) | 45.9 and 44.6 (two runs), 12 of 12 | **45.9, 12 of 12** |
| **Flint**, Flint Pascal Q8 GGUF (`p065`, 40.4 GB), 262,144-token context | TODO(flintq5) | 47.1, 12 of 12 | **47.4, 12 of 12** |
| **Ornith-1.5-122B**, Ornith-1.5-122B Pascal N2P GGUF (`valid1`, 48.2 GB), 49,152-token context | 6.0, 11 of 12 (UD-Q2_K_XL: stock cannot load the N2P/N3P types) | 9.7 (UD-Q2_K_XL) | **16.3, 12 of 12** |

- **Flash-Next:** the preset on this build generates 20.40 tok/s, 12 of 12, with prompts at 69.4 tok/s and the first
  token of a turn at 6.89 s (median). Stock ran the same server line with the MTP draft and without the fork's switches.
  My Pascal type mix of the file (`pmix-q2-fast`, not published) ran the preset at 20.75 tok/s, 2.45 against 2.47
  minutes a task at equal work: within the spread of two runs, so the preset uses UkisAI's file as published.
  [Data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_flashnext/README.md).
  The keep-alive alone: 20.75 against 19.19 tok/s for the same line without it, −8.4 % time a task at equal work
  ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_keepalive/README.md)).
  v2.0's 7.8 and an older stock commit's 6.4 (7 of 8 tasks tried):
  [data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-03_flash_next_baseline/README.md).
- **Flint:** 45.9 tok/s on this build (`1f7215948`) with v2's command line, against v2.0's 45.9 and 44.6 in two runs;
  first token per turn 2.67 s (v2.0: 2.6). Flint Pascal Q8 GGUF, my 40.4 GB file of Q8_0's quality, on `1551f5fe1`:
  47.4 tok/s, first token 2.66 s.
  [Data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md);
  v2.0's numbers: [results.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/results.md).
  Stock b12076 runs at 65,536 tokens of context (at 262,144 its host-side attention mask does not fit in 4 GB), tuned
  to `-b 2048 -ub 512` with flash attention: 14.4 tok/s, 10 of 12, first token 4.13 s (median).
- **Ornith-1.5-122B:** 16.3 tok/s, 12 of 12, first token per turn 8.39 s, with the keep-alive
  (`GGML_CUDA_KEEPALIVE=decode`), which its preset sets. Without it 14.9 tok/s on this build (`1f7215948`), against 14.5
  for the same file on 2026-10-28, first token 9.41 s against 9.6. [Data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md);
  the earlier run and v2.0's 9.7 (stock 6.2, both UD-Q2_K_XL):
  [data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-27_ornith/README.md).

Flint's Q8_0 and Ornith's runs used `1f7215948`; this release is `1551f5fe1`, the same code plus one row-guard fix (below)
that cannot change these models' tokens. On the rigs both commits give the same greedy tokens for Flint (8 of 8
requests) and Flash-Next (both prompts); Ornith's: TODO(identity).

Speed-up against stock llama.cpp b12076, the upstream release this engine is built on (`20a51f24a`, no fork commit),
each engine on the same file:
- **Flash-Next:** 20.40 against 8.20 tok/s (2.49×), 12 of 12 against 9 of 9 reached; on the nine tasks stock reached,
  a task's work takes 1.90 against 3.74 minutes.
- **Flint:** 45.9 against 14.4 tok/s (3.2×), 12 of 12 against 10 of 12, 1.2 against 2.9 minutes a task (median).
- **Ornith-1.5-122B:** on Unsloth's UD-Q2_K_XL, 12.6 against 6.0 tok/s (2.1×), 12 of 12 against 11 of 12, 1.8 against
  3.1 minutes a task (median). Stock cannot load Ornith-1.5-122B Pascal N2P GGUF, on which the preset runs at 16.3.

## What is new since v2

**Every feature behind a switch.** Every behaviour the fork adds to upstream b12076 can be turned off on its own at run
time, in one binary, and every experiment stays off unless set. With every switch at its default the build gives the
same logits, bit for bit, as the build the Flash-Next runs before it used, on every architecture llama.cpp's
`test-llama-archs` builds ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_ladder5/README.md)). The list
with defaults, what each does and which preset sets it:
[docs/switches.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/switches.md).

**One preset per model.** Each model has a `settings.env` with the switches and the command line it was measured with;
one command starts it (below).

**Pascal kernels for Flash-Next's shapes** (on by default on compute 6.1):
- A Pascal kernel for BF16 matrices at 1–8 tokens, and for 2-bit Q2_0 experts.
- k-column kernels for dense products at 1–8 tokens, and a multi-token expert kernel that reads each distinct expert
  once for 2–8 tokens.
- Together with upstream b12076, a Flash-Next token on nine cards went from ≈136 ms to ≈75 ms, and a four-token verify
  from 305 to 163 ms ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_kcurve_p106/README.md)).
- Pascal kernels for my own 2- and 3-bit types N2P and N3P (Ornith-1.5-122B Pascal N2P GGUF).

**The hop cast** (`LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1`, off by default; Flash-Next's preset sets it): the
model's state crosses each of the eight card boundaries in Q8_0 through a pinned host ring instead of F32. Without a
draft, 14.98 → 17.06 tok/s on Flash-Next, with no measurable change in KL divergence against the F32 hops
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_s35_stack/README.md)).
Flint and Ornith keep F32 hops until their own quality check.

**The MTP draft with a shortlist head** (Flash-Next): the published Gyro draft guesses three tokens ahead, and
`LLAMA_DRAFT_SHORTLIST=32768,8192,2` runs its output head over a shortlist of tokens with an exact fallback. The draft
took Flash-Next from 7.8 to 12.2 tok/s on v3's base
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_mtp_redo/README.md));
the shortlist adds 10 % on the newer kernels, 14.6 → 16.0 tok/s
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_stacked_gains/README.md)).
With the hop cast and the newest kernels on top: 18.8 tok/s
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_s35_stack/README.md)).

**The keep-alive** (`GGML_CUDA_KEEPALIVE=decode`, off by default; Flash-Next's preset sets it): on a layer split only
one card works at a time, and the driver drops the waiting cards from their boost clock. One block of one thread per
card, on the card's lowest-priority stream, polls a word in host memory while the split decodes, so every card stays at
1,873–1,911 MHz instead of 1,354. Plain decoding −14 to −19 % a token; on the agent check 19.19 → 20.75 tok/s and
prompts +12.5 %, at 5–11 % more energy a token
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_keepalive/README.md)).
`request` also wakes every card when a request arrives (no better than `decode`); don't use `idle` (known issues).

**The re-read fix** (`LLAMA_SERVER_TEXT_REUSE=1`, off by default): llama-server keeps its cached tokens when a client
sends a reply back as text that tokenizes differently, instead of reading the conversation again from there
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-28_reply_rereads/README.md)).
On Flint's agent check, with the tail checkpoint rule and the keep-alive, 44.7 tok/s against 45.9 without them, one run
each. On its own, with the tail checkpoint rule, it is negligible: 2.59 against 2.60 minutes a task at equal work, the
first token at 2.25 against 2.64 s (median), one run each on one rig
([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-06_flint_side_rig2/README.md));
the keep-alive, the staged hop, the answer store and the negligible group on their own: TODO(flintside).

**Also in the build, off by default:** the expert skip (`LLAMA_MOE_TAIL`: the router's low-weight experts dropped),
the adaptive draft length (`LLAMA_SPEC_ADAPT`), the staircase drafter's cheaper guess, checkpoints spread through long
prompts and kept on the cards, the decode-only card-built mask, and the diagnostics. Their measurements are in the
switch list's rows.

**Upstream base:** llama.cpp b12076 (v2.0 was on an older base): Flash-Next's `qwen4exp` architecture, Q2_0, and
upstream's own changes since v2.0.

## Two bugs in upstream llama.cpp

Testing this release turned up two bugs that are in upstream llama.cpp's own code, still on its `master` on
2026-11-05. Each is patched on upstream `master` and reproduced there, before and after:
- **`mul_mat_vec_q` bounds a one-token MUL_MAT_ID's rows by the column stride**, so when the rows are not a multiple of a
  block's, each expert's last block writes into the next expert's output. New `test-backend-ops` cases fail 30 of 30 and
  29 of 30 runs on master, 0 of 80 with the fix:
  [Cinderella-Man/llama.cpp#1](https://github.com/Cinderella-Man/llama.cpp/pull/1). This release carries the fix
  (`1551f5fe1`); no release model has such a row count.
- **A checkpoint restore in speculative decoding hands the sampler back draws it already used**, which skews rejection
  sampling on recurrent and hybrid models: p = 6.05e-9 on master, 0.23 with the fix:
  [Cinderella-Man/llama.cpp#2](https://github.com/Cinderella-Man/llama.cpp/pull/2). In this build the fix is the switch
  `LLAMA_SPEC_KEEP_DRAWS=1`.

The write-up, with every test and number:
[docs/upstream_fixes.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/upstream_fixes.md).

## Known issues

- **Greedy tokens differ from v2.0's.** On Flint Q8_0 at temperature 0, this build's tokens leave v2.0's after 123, 60,
  99 and 133 tokens on four test prompts, each time at a near tie: v2.0's two best tokens there were 0.011, 0.024, 0.056
  and 0.343 nats apart. Prompt processing is unchanged: on wikitext-2 this build's KL divergence against v2.0 is
  −0.000000 ± 0.000000 (0.000056 at most), with the same top token 100.000 % of the time and the same perplexity,
  24.4293. In one-token decoding the chosen token's log-probability moves by 0.0001 nats (median over 415 tokens),
  0.438 at most, and such steps compound over a generation. The likeliest cause is the one-token kernel's summation
  order (two warps a block on compute 6.1, `GGML_CUDA_MMVQ_2WARPS=1`); upstream b12076's own changes are under the build
  too ([data](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md#how-large-the-difference-is)).
- **Ornith's prompt passes differ from the 2026-10-28 build's on Pascal.** On nine P106-090 cards, wikitext-2, this build
  against `d7a75deb3`: mean KL divergence 0.021653 ± 0.000881, the same top token 93.873 % of the time; this build's
  perplexity is the lower, 5.9938 ± 0.2410 against 6.0735 ± 0.2449. Neither is closer to the model: against
  Ornith's Q8_0 on the same cards (80 chunks of 512), this build's KL divergence is 0.175182 ± 0.002745 and the older
  build's 0.175864 ± 0.002767, a paired difference of −0.00068 ± 0.00069, within noise.
- **Don't use `GGML_CUDA_KEEPALIVE=idle`.** Its 120 s block holds the first launch of each kernel under CUDA's lazy
  module loading: the first request of an agent session waited 506 s for its prompt. `decode` and `request` are not
  affected.
- **`--pipe-depth` on a `qwen4exp` model** (Flash-Next) with the fused delta-net conv (`LLAMA_GDN_FUSE=1`, the default)
  writes a pipe slot's conv history into the first slot's state. Pipelined speculation is off in Flash-Next's preset;
  set `LLAMA_GDN_FUSE=0` if you turn it on.
- **Q4_1_64 files written by v2.0** carry that type's old id and don't load.
- **For people who build from source:** on the fork's upstream-b12128 line (not this release), the card-built mask
  (`LLAMA_GPU_KQ_MASK=1`, off by default) crashes on gemma4's new graph. And under heavy load, `llama-perplexity`
  sometimes exits with the last lines of its log missing; its output files are complete.

## Install

Requirements: Linux with glibc 2.38 or newer (tested on Ubuntu 24.04) and NVIDIA's proprietary driver 580
(`nvidia-driver-580`, not `-open`). No CUDA toolkit: the second asset carries the CUDA runtime libraries.

```bash
sha256sum -c SHA256SUMS
tar xzf {{BIN_ASSET}} && tar xzf {{CUDART_ASSET}}
{{PREFIX}}-{{VERSION}}/llama-server --list-devices
```

Both archives unpack into `{{PREFIX}}-{{VERSION}}/`. `scripts/p106_check.sh` in the repository checks the driver, the
cards, host memory and the model's read speed. Each model's preset, with the files to download:
- [Qwen3.8-Flash-Next](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v3/README.md)
- [Flint](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v3/README.md)
- [Ornith-1.5-122B](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v3/README.md)

From a clone of the repository, one command starts a preset (Flint here; `MODELS` is the folder with the model file):

```bash
. llms-on-crypto-mining-rigs/release/flint-v3/settings.env && env $PRESET_ENV {{PREFIX}}-{{VERSION}}/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

## Assets

| File | Size | SHA-256 |
|---|---|---|
| `{{BIN_ASSET}}` | {{BIN_MB}} MB | `{{BIN_SHA256}}` |
| `{{CUDART_ASSET}}` | {{CUDART_MB}} MB | `{{CUDART_SHA256}}` |

`BUILDINFO.txt` in the first archive records the commit, the build image, and that the binaries contain no AVX
instructions and need glibc 2.38 at most. A third archive carries `test-backend-ops`, llama.cpp's operator tests, for
anyone who wants to check the kernels on their own cards.

## Licences

The binaries are llama.cpp and ggml with this fork's changes, all under the MIT licence (`LICENSE`,
`THIRD_PARTY_NOTICES.md`). The CUDA runtime libraries in the second archive are NVIDIA's, redistributed unmodified under
the NVIDIA CUDA Toolkit End User License Agreement (`NVIDIA_CUDA_RUNTIME_NOTICE.md`). Model files are not part of this
release and keep their own licences.
