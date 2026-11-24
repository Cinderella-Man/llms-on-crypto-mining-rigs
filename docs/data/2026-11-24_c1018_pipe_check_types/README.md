# The GPU pipe check on the release files' types and the Flash-Next preset's switches (C1018, Q1751; 2026-11-24)

What this is: an extension of [bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh), the check
that pipelined decoding gives plain decoding's tokens on the cards. The GPU check of every branch and merge runs it
([C1009](../2026-11-24_c1009_pipe_step_cards/README.md)). It ran random models in F32 and in Q8_0 only, so of Miner
Llama's own Pascal matrix-vector kernels it reached only the Q8_0 ones. The release files use Q2_0 (Swift Flash-Next's
experts), Q4_K, Q5_K, Q6_K and IQ4_XS, and the Flash-Next preset turns on its own kernel switches. Registry row Q1751
in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The check now runs copies in the release files' types and the Flash-Next preset's kernel switches, and the default
  run gained one arm for them.** That arm is `mix-fn-c4@one`:
  - one copy with each tensor role in its release type;
  - the preset's switches on;
  - the prompt decoded four tokens at a time;
  - one card only.

  On an idle machine a build took 116 s with the new defaults (both builds), against 90 and 91 s with the old ones.
  With the mixed copy on three cards as well, it took 154 and 158 s, over the two minutes allowed. The hops between
  cards are tested by the other two arms either way.
- **Each new copy runs the fork's own kernels for its type,** by name in an Nsight Systems profile:
  - the one-token kernel for Q2_0, Q4_K, Q5_K and IQ4_XS;
  - in calls of four tokens, the k-column kernel (Q4_K, Q5_K, IQ4_XS, Q8_0), the two-to-four-token kernels and Q2_0's
    multi-token kernel;
  - with the preset's switches, the read-once kernels for 3 and 4 tokens of Q4_K and Q5_K, in place of the k-column
    kernel.
- **Q6_K is the exception.** Its weights ran on llama.cpp's own kernels in every arm, even with the preset's Q6_K switch
  on, so the check does not cover the fork's Q6_K kernels (registry row Q1783, C1050).
- **The preset's switches act only when a call holds 2 to 8 tokens.** The check decodes every generated token in a call
  of its own, pipelined runs too. Without prompt chunks, `fn` ran exactly `p106`'s kernels with `p106`'s bits. This is
  why the new arms carry `-c4`: their prompt is decoded in calls of 4 tokens.
- **On the tip every check passed:** 1,035 verdicts OK in the first five rounds (450 more in the re-timings) and none
  other than INFO, over five architectures, 36 arms on one card and 9 split over three cards. The tip is
  `pascal-p106-v5` `af00bb13b`.
- **The older build with C932's Flash-Next pipe fault is flagged by every new arm.** That build is `8a9629942`. All six
  arms (the mixed copy and each type, preset switches, four-token prompt calls) show the Flash-Next-type random model
  DIFFERS, on one card and on three: host, fed and recurrent checks, 36 of 36. The tip passes the same 180. The old
  arms flagged this fault too (C1009), so it shows the new arms catch a real fault, not that they catch more.

## The change

- **Model copies by kind** (`<models>` in an arm):
  - `q8`, the Q8_0 copy as before;
  - `<type>`, a copy with every weight in one `llama-quantize` type (`--pure`): `q2_0`, `q4_k`, `q5_k`, `q6_k`, `iq4_xs`
    or any other. Rows too narrow for a K-type or IQ4_XS fall back as in the release files: Q4_K to Q5_0, Q5_K to Q5_1,
    Q6_K to Q8_0, IQ4_XS to IQ4_NL;
  - `mix`, one copy with the release files' types by tensor role (`MIX_ARGS`): routed experts Q2_0, shared experts
    Q4_K, attention Q5_K, linear attention and dense feed-forward IQ4_XS, output Q6_K, the rest Q8_0.

  Only the copies an arm names are made, each copy's tensors by type go into `types.tsv`, and `QUANT_ARGS`' patterns
  (the norms kept F32) win over a copy's own.
- **The setting `fn`:** `p106` (the fork's kernels forced on in a P106's shapes) plus every `GGML_CUDA_MMV*` switch of
  [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) (`FN_SW`, C968's). The one change
  is `GGML_CUDA_MMVF_F32_MT=2` in place of the preset's 1, because 1 acts on compute 6.1 only. When the script sits in
  the repository, it says whether `FN_SW` still matches the preset.
- **The suffix `-c<N>`** (for example `q4_k-fn-c4`): every pipespec run of the arm, plain's too, decodes its prompt in
  calls of N tokens (`llama-pipespec --pipe-chunk N`). That is where the multi-token kernels run. Every run of the arm
  takes the same calls, so the check compares like with like.
- **The suffix `@<layout>,...`** (for example `mix-fn-c4@one`): the arm runs only in those layouts. Without it, an arm
  runs in every layout, as before.
- **The default arms** are `f32-stock q8-p106 mix-fn-c4@one`. Each type's own arm (`q2_0-fn-c4` and so on) is on
  request.
- **The check `profile`** (not in the default `CHECKS`; it needs `NSYS`, an `nsys` binary on the host): it runs the
  host check's runs again under Nsight Systems and writes every kernel the cards ran, with its launch count, into
  `prof-<arch>.tsv`. It records and does not judge. The report's times are deleted unread.

## The run

On a build machine with three RTX 3090s, which run the rig build's compute 6.1 code through its PTX:
- every container labelled, at a low CPU share, without network;
- the cards checked with a CUDA preflight before each round;
- nothing timed but the rounds themselves.

| | |
|---|---|
| **Tip** | Miner Llama's rig tarball of `pascal-p106-v5` at `af00bb13b` (`TESTS=1`, md5 `b51ad2740d5822e7e14af656c413ec22`) |
| **Fault** | the rig tarball of `8a9629942` (C984's, md5 `59ca8336b87367a6522e63c6fcc1381b`), which has C932's Flash-Next pipe fault |
| **Quantizer** | C1009's static CPU `llama-quantize` (the rig tarballs carry none) |
| **Step settings** | five architectures (qwen4exp, qwen35, qwen35moe, qwen3next, flint), seed 1, 64 tokens, checks `host`, `fed`, `recurrent` and `kernels` (`profile` alone in its round) |

| Call | Round | Builds | Arms | Layouts | Exit | Seconds | Verdicts |
|---|---|---|---|---|---|---|---|
| out | defaults | tip as base and head | `f32-stock q8-p106 mix-fn-c4` (the mixed copy in both layouts) | one, split | 0 | 504 | 180 OK, 12 INFO |
| out | reach | tip | `f32-stock` and each of `q8`, `mix`, `q2_0`, `q4_k`, `q5_k`, `q6_k`, `iq4_xs` under `stock`, `p106`, `fn`, `p106-c4`, `fn-c4` (36 arms) | one | 0 | 1,108 | 540 OK, 36 INFO |
| out | split | tip | `f32-stock`, `q8-p106`, `q8-fn-c4`, `mix-fn-c4` and each type's `fn-c4` | split | 0 | 428 | 135 OK, 9 INFO |
| out | prof | tip | reach's 36 arms, `CHECKS=profile` | one | 0 | 4,431 | 180 INFO (kernel lists) |
| out | fault | fault as base, tip as head | `mix-fn-c4` and each type's `fn-c4` | one, split | 1 | 824 | base: 36 DIFFERS (qwen4exp), 144 OK; head: 180 OK; 24 INFO |
| quiet | old | tip as base and head | `f32-stock q8-p106` (the earlier defaults) | one, split | 0 | 183 | 120 OK, 8 INFO |
| quiet | defaults | tip as base and head | `f32-stock q8-p106 mix-fn-c4` (the mixed copy in both layouts) | one, split | 0 | 314 | 180 OK, 12 INFO |
| final | defaults | tip as base and head | `f32-stock q8-p106 mix-fn-c4@one` (the final defaults) | one, split | 0 | 234 | 150 OK, 10 INFO |

- **The first call** (`out`, 16:40–18:42) ran five rounds. Another session's processor-heavy test shared the machine
  during `defaults`, so its 207 and 295 s a build are no measure of the step's time.
- **The second call** (`quiet`, `ROUNDS="old defaults"`, 18:43–18:52) ran with nothing else on the machine, with the
  first call's PTX cache copied in.
- **The third call** (`final`, `ROUNDS=defaults`, 18:53–18:57) ran the same way, after the `@one` default. Another
  session's job was on card 2 by its last line.

Seconds per build ([builds.tsv](builds.tsv)), from a build's first log line to the next:

| Default arms | Build 1 | Build 2 |
|---|---|---|
| `f32-stock q8-p106` (before C1018) | 90 | 91 |
| `f32-stock q8-p106 mix-fn-c4` | 154 | 158 |
| `f32-stock q8-p106 mix-fn-c4@one` (now) | 116 | 116 |

A build's time breaks down as:
- its models, 4 s;
- the one-card arms, side by side on the three cards, 49 s, set by the longest (the mixed copy; 25 s with the old arms);
- the split arms, one after the other, 30 and 33 s. The mixed copy split over three cards added 41 s more.

### Which kernels each setting reaches

From [reach.tsv](reach.tsv): each arm's matrix-product kernels over the five models, against the arm it adds to. In
the type names, the fork's kernels are `mul_mat_vec_q_pascal*` and `kcol::mul_mat_kcol`, llama.cpp's are
`mul_mat_vec_q`, `mul_mat_q` and cuBLAS.

| Copy | `p106` against `stock`: the fork's one-token kernels | `fn-c4` against `p106-c4`: what the preset's switches change in calls of 4 |
|---|---|---|
| q8 | `mul_mat_vec_q_pascal<q8_0>`, `kcol<q8_0>` | nothing |
| q2_0 | `mul_mat_vec_q_pascal<q2_0>` | nothing (Q2_0's multi-token kernel runs in both) |
| q4_k | `mul_mat_vec_q_pascal<q4_K>` (and `<q5_0>` for the fallbacks) | `mul_mat_vec_q_pascal_dmt<q4_K>` in place of `kcol<q4_K>` |
| q5_k | `mul_mat_vec_q_pascal<q5_K>` | `mul_mat_vec_q5_K_pascal_mt<3>` and `<4>` in place of `kcol<q5_K>` |
| q6_k | only the Q8_0 fallbacks' kernels; Q6_K stays on `mul_mat_vec_q<q6_K>` | nothing (Q6_K on `mul_mat_vec_q_moe<q6_K>` and `mul_mat_q<q6_K>` in both) |
| iq4_xs | `mul_mat_vec_q_pascal<iq4_xs>`, `kcol<iq4_nl>` | `mul_mat_vec_q_pascal_mc<iq4_xs>` alone, without `kcol<iq4_xs>` |
| mix | the one-token kernels of Q2_0, Q4_K, Q5_K, IQ4_XS, Q5_0, Q8_0 and `kcol<iq4_nl>`; the Q6_K output on `mul_mat_vec_q<q6_K>` | Q4_K's and Q5_K's read-once kernels and `mul_mat_vec_q_pascal_mc<iq4_xs>` in place of the k-column kernel |

`fn` without chunks ran exactly `p106`'s kernels in every copy. Every kernel and its launch count per arm and model is in
[kernels.tsv](kernels.tsv).

### The plain hashes

[hashes.tsv](hashes.tsv) has the host check's plain logits hash per model and arm, and whether a setting changed it
against the one it adds to.

- **`p106` against `stock`: other logits** in the q8, q2_0, q4_k, q5_k and iq4_xs copies of all five models, so the
  fork's kernels computed them.
  - The q6_k copies: the same logits, state and probe hashes. Only the Q8_0 fallbacks ran other kernels, and they gave
    the same bits.
  - The mix copies: the same logits and probe hashes in all five models. Their KV state hash changed, so the fork's
    kernels did compute other bits. In these copies, though, the layers barely reach the logits: qwen35moe's and
    qwen3next's mix copies give the same logits token by token. Registry row Q1783 (C1050) asks why.
- **`fn` against `p106`: the same** everywhere (the same kernels).
- **`p106-c4` against `p106`: other logits** everywhere but qwen4exp's mix copy, because the prompt goes through other
  kernels.
- **`fn-c4` against `p106-c4`: the same** in 34 of 35 model copies, though Q4_K, Q5_K and IQ4_XS ran other kernels.
  The exception is qwen4exp's iq4_xs copy, where it gave other logits; both settings passed every check.

The plain hashes alone therefore cannot show that the preset's switches act. The kernel names do.

## Files

| File | What it holds |
|---|---|
| [rounds.tsv](rounds.tsv) | per call (`out`, `quiet`, `final`) and round: builds (tags), arms, layouts, checks, exit status, seconds, verdict counts |
| [builds.tsv](builds.tsv) | per call, round and build, the seconds from the build's first log line to the next (its models and every arm) |
| [verdicts.tsv](verdicts.tsv) | every verdict of the first call's rounds but the profile lines: round, tree, arm, layout, architecture, check, verdict, detail, plain tokens and logits hashes |
| [hashes.tsv](hashes.tsv) | reach round: per architecture and copy, the plain logits hash under each setting and whether a setting changed it against the one it adds to |
| [kernels.tsv](kernels.tsv) | prof round: per arm and architecture, every matrix-product kernel the host check's runs launched and its launch count (types by name) |
| [reach.tsv](reach.tsv) | prof round: per arm, its matrix-product kernels (name and type), the ones the arm it adds to does not run, and the ones it no longer runs |
| [types.tsv](types.tsv) | each copy's tensors by type, as `llama-quantize` wrote them |
| [models.tsv](models.tsv) | the md5 of every model file per round and build of the first call |
| [run/run.sh](run/run.sh) | the rounds |
| [run/tables.py](run/tables.py) | the tables from the calls' work folders (`python3 -I run/tables.py <work> <data folder> [<more work folders>]`) |

**The scripts that ran:**
- The first two calls ran `gpu_pipe_check.sh` at md5 `db8cc16e95ca91d9689e346c8cbcf56f`, without the `@<layout>`
  suffix. Its default arms were `f32-stock q8-p106 mix-fn-c4`.
- The third call ran the committed script, md5 `134f3bd6e9295706f9cf125f306d771e`.
- The committed `run.sh` differs from the ones that ran only in comments and in the `old` round, which the first call's
  copy did not have.

## Commands

On the build machine, with a copy of `bench/pipespec/` holding the changed script (`PIPESPEC`), the two rig tarballs'
`bin/` (`TIP`, `FAULT`), a static `llama-quantize` (`QUANT`, as in C1009), an `nsys` (`NSYS`) and the CUDA preflight:

```bash
WORK=<work dir> PIPESPEC=<copy of bench/pipespec> TIP=<tip tarball>/bin FAULT=<fault tarball>/bin \
  QUANT=<llama-quantize> NSYS=<nsight-systems>/bin/nsys GPUS=0,1,2 MOUNTS='<folders holding the tarballs and quantizer>' \
  DOCKER_ARGS='--label <label> --cpu-shares=256' PREFLIGHT=<cuda_preflight.sh> \
  setsid -f bash run/run.sh > <work dir>/driver.log 2>&1 < /dev/null
# the re-timings, each in a work folder of its own once nothing else runs (the first call's jitcache/ copied in)
WORK=<quiet dir> ROUNDS='old defaults' <the same variables> setsid -f bash run/run.sh > <quiet dir>/driver.log 2>&1 < /dev/null
WORK=<final dir> ROUNDS=defaults <the same variables> setsid -f bash run/run.sh > <final dir>/driver.log 2>&1 < /dev/null
python3 -I run/tables.py <work dir> <this folder> <quiet dir> <final dir>
```

In a GPU check, the defaults need nothing new but a `QUANT` that writes Q2_0 (Miner Llama's):

```bash
WORK=<dir> BUILDS='base=<base>/bin head=<branch>/bin' GPUS=0,1,2 QUANT=<llama-quantize> \
  bash bench/pipespec/gpu_pipe_check.sh
# every release type on its own, and the kernel names
ARMS='q2_0-fn-c4 q4_k-fn-c4 q5_k-fn-c4 q6_k-fn-c4 iq4_xs-fn-c4' CHECKS='host profile' NSYS=<nsys> <the same>
```

## Provenance and licence

The random models are written by `test-llama-archs` from the build under test and quantized by C1009's
`llama-quantize`; nothing is downloaded. The tables are this project's own (the repository's licence).
