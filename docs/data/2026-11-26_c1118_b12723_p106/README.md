# Upstream b12723 on a P106: `pascal-p106-v5-cand-b12723` against `eee615cdb` (2026-11-26)

Registry: [Q1851](../../../quants/experiments.md). The merge and its build-machine checks:
[2026-11-25_upstream_b12723](../2026-11-25_upstream_b12723/README.md). The last check of this kind, on upstream
b12674: [2026-11-24_c912_p106_check](../2026-11-24_c912_p106_check/README.md).

**Passed.** On a P106 rig the candidate on upstream b12723 runs every release file exactly as the reference build
does, on the v5 presets. The replies are the same bytes, memory on every card is the same to the MiB, and every
evaluation launches the same kernels. The fork's kernel tests give the last check's results case by case, down to
each error value, and the planted faults are still caught. The random test models' logits differ from the reference's
in exactly the 8 models, and by exactly the amounts, the build machine found, all from upstream's new CPU softplus; no
release file uses it. Two Inkling 2 test models' card results moved by a rounding-sized amount, likely from upstream's
rebuilt Inkling 2 graph. `pascal-p106-v5` does not move on this check alone: Flash-Next's coding-agent run comes next.

The candidate is [`pascal-p106-v5-cand-b12723`](../2026-11-25_upstream_b12723/README.md) `caf27fdae`:
`pascal-p106-v5` (`2d839da1d`) with upstream llama.cpp b12723 merged in (49 commits past b12674) and one fork fix for
Flash-Next under upstream's replay switch. Upstream's three new defaults that would change how a release file runs
stay off behind switches (`LLAMA_RS_REPLAY`, `GGML_CUDA_NORM_ROPE_MROPE`, `GGML_CUDA_NORM_ROPE_BS`), and every switch
was unset here. Its rig build is `llama.cpp-20261125-caf27fdae-pascal-p106-v5-cand-b12723.tar.gz`, 1,630,376,772 B, md5
`779a6b2473045c77a643a9871ffbc96f` (`llama-server` `664f7fecc154b80a90f830ba5e76f378`, `test-backend-ops`
`935e1a36627eaf8623179f5b968f09b4`, `test-llama-archs` `6ce60c67c4b4372b004f53e7c48581e6`, `llama-completion`
`f7b4662fd500c690710c2d8da49e8110`), copied to the rig with its md5 checked at both ends.

**The reference is `eee615cdb`, not `pascal-p106-v5` itself.** `pascal-p106-v5` stands at `2d839da1d`, which is
`eee615cdb` (the b12674 candidate this rig checked last time) plus seven merges of the fork's own finished branches.
That build's tarball was not reachable for this run, while `eee615cdb`'s tree was already on the rig:
`llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674` (`llama-server` `2727faf28b5b34dfe42399a9d431da67`,
`test-backend-ops` `7e6731c4d42a98cfd6a90cf4bdeb0cc5`, `test-llama-archs` `4e5a143af3fe0e4252e323f983973545`,
`llama-completion` `46c24218006b8fb067d62c1ca23c713f`). On the build machine `2d839da1d` gave `eee615cdb`'s greedy
replies, memory per card and logits ([2026-11-25_land_v5_batch3](../2026-11-25_land_v5_batch3/README.md)), and its
seven merges add switches that are off by default or read only by tests: the log flush at exit, ctest's fixed seed,
the host profile's anchors (`GGML_CUDA_PROFILE_HOST_ANCHORS`, read only by the host profile, not the light profile used
here), the `TOPK_MOE_TAIL` test's tie rule (in `test-backend-ops` only), the attention test fault's factor (read only
with the fault on), the 64-row tile for 64-column prompt products (`GGML_CUDA_MMQ_J64_I64`, 0 by default) and the
NVML probe skipped where no card reads it. So a difference here was checked against those seven first, then named
upstream's.

One rig (nine P106-090, the 4 GB host), 18:10–20:41. Every memory and core offset 0, the 75 W power limit, 1,911 /
4,004 MHz maximum clocks, no Xid; no clock, power or offset setting was changed. Nothing is timed. Every switch a release
line does not set was unset.

## Part (2): the kernel suite on one card

[run/rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh) of the earlier checks, unchanged, on card 0,
18:11:50–18:50:06 (38 min), the candidate's binaries. The record sets are R153's
([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md#on-a-p106-r153-q1463)): references computed on a
build machine's CPU, replayed here by [scripts/replay_checks.sh](../../../scripts/replay_checks.sh), so the card
computes and the stored CPU results judge it.

| Check | Result | The last check (b12674, the same sets) |
|---|---|---|
| `v4kern`: C796's three kernel suites and all three together, 47 runs ([replays.tsv](replays.tsv)) | **47 of 47 as expected**: 16,240 recorded case replays; the 41 passing runs pass every recorded case, and the 6 planted faults fail 42, 5, 193, 42, 5 and 193 of them (`q6kmt`, `dmtkt`, `ro8`, then all three kernels on) | the same counts |
| `ro8kt`: C794's K tiles at 5–8 tokens, 16 runs | **16 of 16 as expected**: 3,232 recorded case replays; the planted fault fails 34 | the same |
| Every run's per-case table (case, verdict, the five error values) against the last check's ([suite-vs-last.tsv](suite-vs-last.tsv)) | **63 of 63 tables the same bytes**: every case gives the last check's error values to the printed digit | — |
| Cases the candidate's `test-backend-ops` lists that the sets do not hold ([unrecorded.tsv](unrecorded.tsv)) | the same 16 `IQ4_XS` `MUL_MAT` cases as last time (upstream's #31998), in the same runs; unjudged, counted apart | 16 |
| `test-llama-archs -b CUDA0` plain, the fork's paths forced on (`=2`), and the card-built attention mask checked ([archs.tsv](archs.tsv)) | **159 of 159 tests pass three times**, 0 FAIL lines; 1,634 card masks identical to the host's | 159 of 159, 1,634 masks |
| The random models' CPU logits against `eee615cdb`'s (`archs-off`, `-b CUDA0 -s 1`; [logits.tsv](logits.tsv), [logits-vs-other-machine.tsv](logits-vs-other-machine.tsv)) | **144 of 152 the same bits, 8 differ: the build machine's 8, each with its nmse, largest difference and count of differing values to the printed digit** | — |

**The 16 unjudged cases** are the ones Q1771 recorded on a build machine into new copies of the sets; those copies
were not on the rig yet, so this check replayed R153's sets as the last one did.

**The 8 random models that differ are upstream's
[#32655](https://github.com/ggml-org/llama.cpp/pull/32655)** (the CPU's softplus through `log1pf`), which feeds the
CPU selective scan of the Mamba-family models: `mamba`, `mamba2`, `jamba`, `falcon-h1`, `nemotron_h`,
`nemotron_h_moe`, `metis16` and `granitehybrid`. The build machine's trace build with #32655 and the Inkling 2 graph
changes reverted gave the base's logits on all 152 files. None of the seven fork merges between `eee615cdb` and
`2d839da1d` touches the CPU logits (the build machine found 152 of 152 the same). The CUDA softplus is unchanged, and
no release file is a Mamba-family model.

**Two models' card results moved although their CPU logits did not: `inkling2` and `inkling2_loop`.** On each build the
card agrees with that build's own CPU for every model (largest card-against-CPU nmse 1.42e-07 on both, `gemma4`). Of the
144 models with the same CPU logits on both builds, 142 give the same card-against-CPU nmse; `inkling2` gives 2.76e-10
on `eee615cdb` and 2.74e-10 on the candidate, `inkling2_loop` 6.29e-10 and 6.23e-10. Upstream's three Inkling 2
changes ([#32948](https://github.com/ggml-org/llama.cpp/pull/32948), fewer ops in `mix()` and `control()`;
[#32855](https://github.com/ggml-org/llama.cpp/pull/32855), the weighted sum with the scores as the first operand;
[#32940](https://github.com/ggml-org/llama.cpp/pull/32940), the clip through `ggml_l2_norm`) rebuild that
architecture's graph and touch only `src/models/inkling2.cpp`; none of the seven fork merges touches a model's graph.
A rebuilt graph runs other kernels on the card, which add in another order, while this build's CPU gives the same bits.
That is the likely cause; it was not traced here, since a rig runs only what needs a P106 (Q1931). No release file is
an Inkling 2 model.

## Part (3): the release files on both builds

The last check's [run/cand_load.sh](../2026-11-24_c912_p106_check/run/cand_load.sh), unchanged, all nine cards,
18:50:53–20:41:10, after part (2), driven from a machine that only runs ssh, with every ssh call under a 30-minute
timeout ([run/ssh](run/ssh), first on `PATH`). Each file runs on its **v5 release line** as it stands today
([release/flash-next-v5](../../../release/flash-next-v5/settings.env),
[flint-v5](../../../release/flint-v5/settings.env), [ornith-v5](../../../release/ornith-v5/settings.env); Flint Pascal
Q5 GGUF on Flint's line without `-ts`). Per file, both builds in turn (the candidate first): one llama-server under the
RAM watchdog; 20 s after it answers, each card's memory and the host's MemAvailable; three requests (`/completion`,
temperature 0, 256 tokens, no prompt cache): a coding request, a prose paragraph, and a 321-token code review prompt
that Flint's and Flash-Next's lines read in 64-token micro-batches; the fork's profiler in its light mode on both
servers for part (4). Then [rig_greedy.sh](../2026-11-05_ladder5/run/rig_greedy.sh) decodes the coding request with
`llama-completion`, one token a pass, at `-c 4096`, every switch at its default, on the candidate, and compares the
text with v4.0's from R153 part (3).

The files: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (UkisAI's Swift-1.5 GSQ-RCO Q2_0) with its MTP draft; Flint Pascal
Q8 GGUF (`p065`); Flint Pascal Q5 GGUF (K2h-r16); Ornith-1.5-122B Pascal N2P GGUF (`valid1`).

**Every reply is the same bytes on both builds** ([reply-agree.tsv](reply-agree.tsv),
[load-replies.tsv](load-replies.tsv), [greedy-vs-v4.tsv](greedy-vs-v4.tsv)):

| File | Server, coding request | Server, prose | Server, long prompt (321 tokens) | Drafted and accepted (coding; prose; long) | `rig_greedy.sh` against v4.0's text |
|---|---|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | the same, 743 characters | the same, 1,214 | the same, 1,072 | 242 and 174; 278 and 162; 317 and 149, on both builds | the same, md5 `4384ebc02147be1ee6f38d1ce4a43f21`, 745 B |
| Flint Pascal Q8 GGUF | the same, 813 | the same, 1,187 | the same, 1,219 | — | the same, md5 `c5b2d8f20654728edfe38c37458b4c5c`, 972 B |
| Flint Pascal Q5 GGUF | the same, 1,288 | the same, 1,335 | the same, 1,268 | — | the same, md5 `d72ca1b968a8c1eb2b1ebeea22458ff9`, 1,113 B |
| Ornith-1.5-122B Pascal N2P GGUF | the same, 740 | the same, 1,305 | the same, 977 | — | the same, md5 `35f987c059c7d2f521404e964995cd2d`, 974 B |

- Flash-Next's and both Flint files' server replies are also the last check's bytes. Flash-Next's line has changed
  since in three settings (the sparse attention's split count and its switch point, the prompt taper's `cuts=1`); on
  these short requests its text and draft counts are unchanged.
- Ornith's server replies differ from the last check's because its line changed after that run: the 8-bit attention
  cache with pull request 9's attention kernels, and their 8-bit key scoring (`GGML_CUDA_FATTN_TILE_PASCAL_KDP=1`). On
  today's line both builds give the same bytes. `rig_greedy.sh` runs every switch at its default, so its texts stay
  v4.0's.

**Every file loads with the same memory on every card to the MiB** ([load-memory.tsv](load-memory.tsv)):

| File | Load, s (candidate, reference) | MiB per card 20 s after it answered, cards 0–8 (both builds) | Sum, MiB | MemAvailable 20 s after, kB (candidate, reference) | Lowest during the load, kB (candidate, reference) |
|---|---|---|---:|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 317, 312 | 5,007 5,059 4,911 5,075 4,175 4,927 4,345 4,939 5,545 | 43,983 | 803,172, 823,996 | 292,900, 299,380 |
| Flint Pascal Q8 GGUF | 600, 596 | 5,383 5,923 5,081 5,591 5,923 5,081 5,545 5,293 4,793 | 48,613 | 1,402,904, 1,397,176 | 1,188,972, 1,174,504 |
| Flint Pascal Q5 GGUF | 437, 432 | 4,981 4,763 3,641 4,707 3,631 4,683 3,595 4,141 4,055 | 38,197 | 1,415,200, 1,418,824 | 1,210,456, 1,214,128 |
| Ornith-1.5-122B Pascal N2P GGUF | 689, 687 | 5,561 5,547 5,507 5,547 5,199 5,307 5,469 5,513 4,963 | 48,613 | 1,478,548, 1,509,744 | 1,324,136, 1,316,356 |

- Flash-Next's and both Flint files' figures are also the last check's. Ornith's are lower than the last check's
  (49,139 MiB then): its line changed since.
- After the three requests each card holds a little more, again the same on both builds (the TSV).
- MemAvailable is one reading per load, and the candidate always loaded first, so the two builds did not start from
  the same page cache. The watchdog's floor is 60 MB; it stopped no load.

## Part (4): launch groups per evaluation

Both builds' servers in part (3) ran with the fork's profiler in its light mode (`GGML_CUDA_PROFILE_MODE=graph`): one
pair of CUDA events around each whole evaluation, CUDA graphs, fusions and kernels left as they are, and one line per
evaluation and card with its launch groups. A fusion that stops matching after a merge shows only as more groups. The
same requests on the same tokens give the same evaluations, so the two builds' lines compare one for one
([launch-groups.tsv](launch-groups.tsv), [eval-pairs.tsv](eval-pairs.tsv)).

**Every evaluation read on both builds launches the same groups, on every card:**

| File | Evaluations per card (each build; not ready) | (graph, launch groups), every evaluation read on both builds | Launch groups of one decoded token, cards 0–8 (both builds) |
|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 287 on cards 0–7, 1,405 on card 8 (the draft's graphs); 3 not ready on every card, the same on both builds | the same | 311 286 266 288 230 267 249 267 on cards 0–7; card 8 runs several graphs (89, 39, 202, 85 groups, and 145 and 183 a few times) |
| Flint Pascal Q8 GGUF | 1,723; none not ready on card 0, up to three on cards 1–8 | the same | 81 79 67 73 79 67 73 79 78 |
| Flint Pascal Q5 GGUF | 1,687; none not ready on card 0, three or four on cards 1–8, the same on both builds | the same | 53 37 32 43 32 43 32 37 59 |
| Ornith-1.5-122B Pascal N2P GGUF | 769; none not ready | the same | 204 204 204 204 170 170 170 170 161 |

- Flash-Next's and both Flint files' decoded-token figures are the last check's. Flash-Next's fused routed launch is
  kept (its line leaves `GGML_CUDA_MMID_GLU_UNFUSE` unset).
- Ornith's decoded token launches other counts than in the last check (199 194 199 194 165 165 165 160 156 then): its
  line now runs the 8-bit cache through pull request 9's attention kernels. Both builds launch the same.
- **"Not ready" evaluations:** the light profiler writes an evaluation whose events had not completed when it read
  them as not ready. Every one is a pass of the long prompt: Flash-Next's evaluations 180 to 182 (879 to 881 on card 8);
  Flint's 64-token passes, which the nine cards work on as a pipeline, so a card is often still busy when the host reads
  it (Flint Pascal Q8 GGUF's evaluations 1,129 to 1,132, Flint Pascal Q5 GGUF's 1,038 to 1,041). On Flint Pascal Q8
  GGUF a few of them were ready on one build and not on the other, so cards 1 and 2 show "differs" in
  [launch-groups.tsv](launch-groups.tsv) by those evaluations alone. Matched by index ([eval-pairs.tsv](eval-pairs.tsv)),
  every evaluation read on both builds has the same graph and launch groups, and every card has the same number of
  evaluations on both builds.
- The decoded-token columns count an evaluation as one token's when its card time is under 60 ms
  (`scripts/prof_summary.py`'s rule), so an evaluation near that line can count on one build only; the pair column
  compares every evaluation regardless of time.

## What the watch list asked for

The [merge's watch list](../2026-11-25_upstream_b12723/watch.md) named two things to see first on a P106.

| Watch | What this check shows | What it leaves |
|---|---|---|
| The gated delta net kernel at Flash-Next's verify (#31669's `replay` template argument; the candidate keeps the rollback snapshots by default) | Flash-Next's server replies run the MTP draft and its four-token verify passes: the same bytes, the same drafted and accepted counts and the same launch groups on both builds, so the snapshot path computes and launches as before | its time: one reading per load with the profiler on is no measure. Flash-Next's coding-agent run on the candidate (rig 1 row 12d) times the verify step; upstream's replay form is Q1852 |
| `mul_mat_vec_f`'s 64-bit offsets (#32965) on Flash-Next's F32 shared-expert gate at 4–8 tokens (`GGML_CUDA_MMVF_F32_MT=1`) | the same: every verify pass runs it, with the same replies and launch groups. The build machine's machine-code comparison lists every `mul_mat_vec_f` kernel as changed (440 kernels) | the step time, in the same agent run |

One more reading, not a result: the server's first log line came earlier on the candidate on every file, 0.312 against
0.320 s (Flash-Next), 0.317 against 0.358 s (Flint Pascal Q8 GGUF), 0.291 against 0.339 s (Flint Pascal Q5 GGUF),
0.274 against 0.383 s (Ornith) ([load-start.tsv](load-start.tsv)). The candidate carries the fork's skip of upstream's
NVML probe on cards without fast BF16 (`GGML_CUDA_NVML_PROBE`, one of the seven merges between the two builds), which
the last check's readings pointed to; its test runs here print the skip. One reading each; the load itself took 317
against 312, 600 against 596, 437 against 432 and 689 against 687 s.

## Not checked here

- **Speed.** Nothing is timed: the profiler ran on both servers, and each figure is one reading. Flash-Next's
  coding-agent run on the candidate (rig 1 row 12d) is the speed gate before `pascal-p106-v5` moves; Ornith's (rig 2
  row 26) is a compatibility check.
- **The 16 `IQ4_XS` cases against stored CPU results**: Q1771's sets, once on the rig.
- **Why `inkling2`'s and `inkling2_loop`'s card results moved** (Q1931).
- **The switches on**, upstream's three new ones among them: each has its own rows (Q1852 to Q1854).

## Files

| File | What |
|---|---|
| [replays.tsv](replays.tsv) | every replayed run of `v4kern` and `ro8kt`: what it must do, cases passed, FAIL lines, exit status, verdict, unrecorded cases |
| [suite-vs-last.tsv](suite-vs-last.tsv) | per set: the run tables compared with the last check's, how many differ, and whether the summary and the unrecorded cases are the same |
| [unrecorded.tsv](unrecorded.tsv) | the 16 cases the candidate lists that the record sets do not hold, with the runs that selected each |
| [archs.tsv](archs.tsv) | the three `test-llama-archs -b CUDA0` runs: summary, rows OK, FAIL lines, exit status, card-mask lines |
| [logits.tsv](logits.tsv) | per random model: the candidate's CPU logits against `eee615cdb`'s (nmse, largest difference, values differing) and each build's card-against-CPU nmse |
| [logits-vs-other-machine.tsv](logits-vs-other-machine.tsv) | per random model: the candidate-against-reference nmse and differing values on this rig and on the build machine (the first build of the candidate against `2d839da1d`) |
| [load-memory.tsv](load-memory.tsv) | per file and build: seconds to load, MiB per card 20 s after it answered and after the requests, MemAvailable then and its lowest over the load |
| [load-replies.tsv](load-replies.tsv) | per file, build and request: prompt tokens, tokens generated, md5 of the text, drafted and accepted tokens, the server's generation speed (a diagnostic, profiled), the first characters |
| [reply-agree.tsv](reply-agree.tsv) | per file and request: the candidate's reply against the reference's (the same bytes or not, common prefix, lengths) |
| [greedy-vs-v4.tsv](greedy-vs-v4.tsv) | `rig_greedy.sh`'s text on the candidate against v4.0's from R153 part (3) |
| [launch-groups.tsv](launch-groups.tsv) | per file and card: evaluations read on each build, whether the (graph, launch groups) pairs and their counts are the same, the decoded tokens' launch groups on each build |
| [eval-pairs.tsv](eval-pairs.tsv) | per file and card: evaluations matched by index, those read on both builds and how many of them have the same graph and launch groups, the not-ready evaluations on each build and what the other build read for them |
| [load-start.tsv](load-start.tsv) | per file and build: the server log's time of its first line and of the model load's start, and the seconds to load |
| [run/compare_suites.sh](run/compare_suites.sh) | `suite-vs-last.tsv`'s comparison of two suite runs' per-case tables |
| [run/ssh](run/ssh) | the wrapper that put every ssh call of the load step under a timeout |

The rest ran unchanged from earlier folders: part (2)
[rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh), parts (3) and (4)
[cand_load.sh](../2026-11-24_c912_p106_check/run/cand_load.sh), and the tables
[cand_tables.py](../2026-11-17_v5_cand_p106_check/run/cand_tables.py),
[load_tables.py](../2026-11-24_c912_p106_check/run/load_tables.py),
[eval_pairs.py](../2026-11-24_c912_p106_check/run/eval_pairs.py) and
[extra_tables.py](../2026-11-24_c912_p106_check/run/extra_tables.py). The raw logs (consoles, server logs, card logs,
profiles) are not published: they name the machine.

## Commands

```bash
NEW=llama.cpp-20261125-caf27fdae-pascal-p106-v5-cand-b12723
REF=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674   # already on the rig
# the tarball to the rig (md5 on both ends), then unpacked
md5sum $NEW.tar.gz && cat $NEW.tar.gz | ssh $RIG "cat > /opt/llama-builds/$NEW.tar.gz"
ssh $RIG "cd /opt/llama-builds && md5sum $NEW.tar.gz && tar xzf $NEW.tar.gz && rm $NEW.tar.gz \
  && md5sum $NEW/bin/llama-server $NEW/bin/test-backend-ops $NEW/bin/test-llama-archs $NEW/bin/llama-completion"

# part (2) on the rig, card 0: rig_suite.sh with this repository's scripts it calls copied to $R/repo
# (scripts/replay_checks.sh, docs/data/2026-11-03_upstream_rebase/run/check.sh, docs/data/2026-11-04_moe_skip/run/gpu_checks.sh)
ssh $RIG "R=$R REPO=$R/repo SETS=<folder with v4kern/ and ro8kt/> NEW=/opt/llama-builds/$NEW REF=/opt/llama-builds/$REF \
  STEPS='v4kern ro8kt archs off' setsid -f nohup timeout 5400 bash $R/rig_suite.sh > $R/out/console.txt 2>&1 < /dev/null"

# parts (3) and (4), from the repository root on a machine that only runs ssh, started when part (2) wrote $R/out/done
PATH=docs/data/2026-11-26_c1118_b12723_p106/run:$PATH RIG=$RIG NEW=$NEW REF=$REF REL=v5 \
  FN=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  FLINT_Q5=$MODELS/flint-pascal-Q5.gguf FLINT_Q8=$MODELS/flint-pascal-Q8.gguf \
  ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf RIGDIR=<rig folder> OUT=<local folder> \
  LABEL_NEW=cand LABEL_REF=ref PROFILE=1 GREEDY_BUILDS=$NEW \
  GREEDY_EXPECT='flashnext=4384ebc02147be1ee6f38d1ce4a43f21 flintq8=c5b2d8f20654728edfe38c37458b4c5c flintq5=d72ca1b968a8c1eb2b1ebeea22458ff9 ornith=35f987c059c7d2f521404e964995cd2d' \
  timeout <seconds left before the rig's stop> bash docs/data/2026-11-24_c912_p106_check/run/cand_load.sh

# the tables (standard library only)
D=docs/data/2026-11-26_c1118_b12723_p106; C=docs/data/2026-11-24_c912_p106_check/run
python3 -I $C/load_tables.py <cand_load.sh's OUT> $D/          # also writes greedy.tsv, which greedy-vs-v4.tsv replaces
python3 -I docs/data/2026-11-17_v5_cand_p106_check/run/cand_tables.py <rig_suite.sh's out/> <cand_load.sh's OUT> $D/ cand ref
python3 -I $C/eval_pairs.py <cand_load.sh's OUT> $D/ cand ref
python3 -I $C/extra_tables.py <cand_load.sh's OUT> <rig_suite.sh's out/> <the build machine's test-llama-archs compare log> $D/
bash $D/run/compare_suites.sh <the last check's rig_suite.sh out/> <rig_suite.sh's out/>   # suite-vs-last.tsv
```

`GREEDY_EXPECT` holds v4.0's texts of R153 part (3) (the same `rig_greedy.sh` command, files and split). The build
machine's compare log is its `test-llama-archs -b CPU -s 1` run of the candidate's first build against `2d839da1d`'s
files ([2026-11-25_upstream_b12723](../2026-11-25_upstream_b12723/README.md); the final build gave the same logits).

Provenance: Miner Llama's fork and upstream llama.cpp (MIT); the release files as published on the project's model
pages; the random models are the ones `test-llama-archs` generates.
