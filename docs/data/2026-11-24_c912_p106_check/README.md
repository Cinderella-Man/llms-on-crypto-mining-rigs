# Upstream b12674 on a P106: `pascal-p106-v5-cand-b12674` against `pascal-p106-v5` (2026-11-24)

Registry: [Q1649](../../../quants/experiments.md). The merge and its build-machine checks:
[2026-11-24_c912_upstream_b12674](../2026-11-24_c912_upstream_b12674/README.md).

**Passed.** On a P106 rig the candidate on upstream b12674 runs every release file exactly as `pascal-p106-v5`
(`af00bb13b`, upstream b12483) does, on the v5 presets. The replies are the same bytes, memory on every card is the
same to the MiB, and every evaluation launches the same kernels. The fork's kernel tests give the last check's results
on every recorded case, and the planted faults are still caught. The random test models' logits differ from
`af00bb13b`'s in exactly the 120 models and by exactly the amounts the build machine found, all from upstream applying
weights' scale tensors by itself, which the release files do not have. Upstream's tests now list 16 kernel cases the
fork's recordings lack, so those ran unjudged here. `pascal-p106-v5` does not move on this check alone: one
coding-agent run each on Flash-Next and Ornith comes first.

The candidate is [`pascal-p106-v5-cand-b12674`](../2026-11-24_c912_upstream_b12674/README.md) `eee615cdb`:
`pascal-p106-v5` with upstream llama.cpp b12674 merged in (191 commits past b12483) and one fork fix. Its rig build is
`llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz`, 1,403,235,928 B, md5
`575ef3c702f69f14d7d3022c1287c38f` (`llama-server` `2727faf28b5b34dfe42399a9d431da67`, `test-backend-ops`
`7e6731c4d42a98cfd6a90cf4bdeb0cc5`, `test-llama-archs` `4e5a143af3fe0e4252e323f983973545`, `llama-completion`
`46c24218006b8fb067d62c1ca23c713f`). The reference is `pascal-p106-v5`'s own rig build,
`llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz`, 1,385,271,121 B, md5
`b51ad2740d5822e7e14af656c413ec22` (`llama-server` `f5cca34ea6bd4f3712cd6193bf0605be`, `test-llama-archs`
`bf171d94e7000d14d1933e594eb0d2de`). So a difference here names upstream b12483..b12674 or the candidate's three fork
commits. Both tarballs were copied to the rig with their md5 checked at both ends.

One rig (nine P106-090, the 4 GB host), 14:48–17:16. Every offset 0, the 75 W power limit, 1,911 / 4,004 MHz maximum
clocks, no Xid; no clock, power or offset setting was changed. Earlier the same day a memory-clock test had run on this
boot and the cards' clocks had stepped down, so the cards ran somewhat slower than stock. That does not matter here:
both builds ran in the same boot, and nothing is timed. Every switch a release line does not set was unset.

## Part (2): the kernel suite on one card

[run/rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh) of the earlier checks, unchanged, on card 0,
14:48–15:25 (36 min), the candidate's binaries. The record sets are R153's
([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md#on-a-p106-r153-q1463)): references computed on a
build machine's CPU with v4-rc1's `test-backend-ops`, replayed here by
[scripts/replay_checks.sh](../../../scripts/replay_checks.sh), so the card computes and the stored CPU results judge
it.

| Check | Result | The last check (b12483, the same sets) |
|---|---|---|
| `v4kern`: C796's three kernel suites and all three together, 47 runs ([replays.tsv](replays.tsv)) | **47 of 47 as expected**; every recorded case gives the last check's result, run by run: 16,240 recorded case replays; the 41 passing runs pass every recorded case, and the 6 planted faults fail 42, 5, 193, 42, 5 and 193 of them (`q6kmt`, `dmtkt`, `ro8`, then all three kernels on) | the same counts |
| `ro8kt`: C794's K tiles at 5–8 tokens, 16 runs | **16 of 16 as expected**: 3,232 recorded case replays; the planted fault fails 34 | the same |
| Cases the candidate's `test-backend-ops` lists that the sets do not hold | **16 new `IQ4_XS` `MUL_MAT` cases** (m = 4,096 with k = 1,024, and m = 1,023 with k = 4,096, at 1 to 8 columns), in every run of the sets that selects `IQ4_XS` (10 in each `dmtkt` run, 8 in each `ro8` and `ro8kt` run, 16 in each `kern-all` run). They ran on the card but have no stored result, so they were not judged; the runs count them apart, as the replay tool does by default | 0 |
| `test-llama-archs -b CUDA0` plain, the fork's paths forced on (`=2`), and the card-built attention mask checked ([archs.tsv](archs.tsv)) | **159 of 159 tests pass three times**, 0 FAIL lines; 1,634 card masks identical to the host's | 155 of 155, 1,590 masks |
| The random models' logits against `af00bb13b`'s (`archs-off`, `-b CUDA0 -s 1`; [logits.tsv](logits.tsv)) | **28 of 148 the same bits, 120 differ: the same 120 models, with the same nmse to the digit, as on the build machine**; 4 new models (written, nothing to compare) | — |

**The 16 unjudged cases are upstream's.** Upstream's
[#31998](https://github.com/ggml-org/llama.cpp/pull/31998) (a SYCL change) added `IQ4_NL` and `IQ4_XS` to an existing
block of multi-column matrix-vector cases in `test-backend-ops`, so b12674 lists 16 `IQ4_XS` shapes that R153's sets
were recorded without. On the build machine the same cases pass against the CPU with the fork's Pascal kernels forced
(`MUL_MAT` 5,161 of 5,161 there). Judging them on a P106 needs their CPU results recorded into the sets on a build
machine (Q1771); a rig's own processor is not used for references.

**The archs part checks the candidate against itself, as planned.** Upstream's
[#32669](https://github.com/ggml-org/llama.cpp/pull/32669) applies every weight's `<name>.scale` tensor
automatically. `test-llama-archs` builds each random model with every optional tensor filled with random values, so
120 of the 148 architectures now apply random scales that `af00bb13b` left out, and their logits change. The
[build machine's trace](../2026-11-24_c912_upstream_b12674/README.md) tied all 120 to #32669. Here:
- all three card runs pass every one of the candidate's 159 tests, the card against the candidate's own CPU;
- the candidate's CPU logits differ from `af00bb13b`'s in exactly the build machine's 120 models, and each model's
  nmse and count of differing values is the build machine's number to the printed digit, so the files the candidate
  writes on this rig are the ones it wrote there;
- on each build the card agrees with that build's own CPU for every model (largest card-against-CPU nmse 1.42e-07 on
  both, `gemma4`).

The 4 new models are upstream's new architectures (`domynedge`, `inkling2_loop`, `moshi`, `zaya1`). The test count
rises by 4 too (159 against 155); `kokoro` and `tiny-voca` are new rows that the test skips on a card.

**One model's card result changed although its CPU logits did not: `cohere2`.** Of the 28 models with the same CPU
logits on both builds, 27 give the same card-against-CPU nmse on both. `cohere2` gives 2.80e-14 on `af00bb13b` and
2.86e-14 on the candidate. `cohere2` normalises every block with a layer norm without a bias, the pattern upstream's
new CUDA fusion of `NORM` and `MUL` ([#32454](https://github.com/ggml-org/llama.cpp/pull/32454)) fuses. That is the
likely cause but was not traced here: a rig runs only what needs a P106, and the question is not one (Q1772). The
release files use RMS norms.

## Part (3): the release files on both builds

[run/cand_load.sh](run/cand_load.sh), all nine cards, 15:25–17:16, after part (2). It is the earlier checks'
[cand_load.sh](../2026-11-17_v5_cand_p106_check/run/cand_load.sh) with two changes:
- each file runs on its **v5 release line** ([release/flash-next-v5](../../../release/flash-next-v5/settings.env),
  [flint-v5](../../../release/flint-v5/settings.env), [ornith-v5](../../../release/ornith-v5/settings.env); Flint Pascal
  Q5 GGUF on Flint's line without `-ts`), not v4.0's;
- each load gets a third greedy request, a 321-token code review prompt (`long`). Flint's and Flash-Next's lines read
  a prompt of up to 600 and 1,400 tokens in 64-token micro-batches (`--ubatch-taper`), so this request runs the
  64-column prompt kernels b12674 changed: upstream's 128-row tiles for 64-column passes on Pascal
  ([#31352](https://github.com/ggml-org/llama.cpp/pull/31352)) and the 64-bit output offsets in every `mul_mat_q`
  ([#32374](https://github.com/ggml-org/llama.cpp/pull/32374)). Ornith's line reads it in one 321-token pass. The
  coding and prose requests are a single micro-batch each.

Per file, both builds in turn (the candidate first): one llama-server under the RAM watchdog; 20 s after it answers,
each card's memory and the host's MemAvailable; the three requests (`/completion`, temperature 0, 256 tokens, no prompt
cache); the fork's profiler in its light mode on both servers for part (4). Then
[rig_greedy.sh](../2026-11-05_ladder5/run/rig_greedy.sh) decodes the coding request with `llama-completion`, one token
a pass, at `-c 4096`, every switch at its default, on the candidate only, and compares the text with v4.0's from R153
part (3) (it would have run the reference too for a file whose text differed).

The files: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (UkisAI's Swift-1.5 GSQ-RCO Q2_0) with its MTP draft; Flint Pascal
Q8 GGUF (`p065`); Flint Pascal Q5 GGUF (K2h-r16); Ornith-1.5-122B Pascal N2P GGUF (`valid1`).

**Every reply is the same bytes on both builds** ([reply-agree.tsv](reply-agree.tsv),
[load-replies.tsv](load-replies.tsv), [greedy-vs-v4.tsv](greedy-vs-v4.tsv)):

| File | Server, coding request | Server, prose | Server, long prompt (321 tokens) | Drafted and accepted (coding; prose; long) | `rig_greedy.sh` against v4.0's text |
|---|---|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | the same, 743 characters | the same, 1,214 | the same, 1,072 | 242 and 174; 278 and 162; 317 and 149, on both builds | the same, md5 `4384ebc02147be1ee6f38d1ce4a43f21`, 745 B |
| Flint Pascal Q8 GGUF | the same, 813 | the same, 1,187 | the same, 1,219 | — | the same, md5 `c5b2d8f20654728edfe38c37458b4c5c`, 972 B |
| Flint Pascal Q5 GGUF | the same, 1,288 | the same, 1,335 | the same, 1,268 | — | the same, md5 `d72ca1b968a8c1eb2b1ebeea22458ff9`, 1,113 B |
| Ornith-1.5-122B Pascal N2P GGUF | the same, 728 | the same, 1,257 | the same, 1,035 | — | the same, md5 `35f987c059c7d2f521404e964995cd2d`, 974 B |

The coding and prose replies are also the md5s of the last check, which ran v4.0's lines, and Flash-Next drafted and
accepted the same counts: on these requests the v5 lines give v4.0's text.

**Every file loads with the same memory on every card to the MiB** ([load-memory.tsv](load-memory.tsv)):

| File | Load, s (candidate, reference) | MiB per card 20 s after it answered, cards 0–8 (both builds) | Sum, MiB | MemAvailable 20 s after, kB (candidate, reference) | Lowest during the load, kB (candidate, reference) |
|---|---|---|---:|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 312, 313 | 5,007 5,059 4,911 5,075 4,175 4,927 4,345 4,939 5,545 | 43,983 | 857,592, 836,472 | 320,852, 378,524 |
| Flint Pascal Q8 GGUF | 591, 591 | 5,383 5,923 5,081 5,591 5,923 5,081 5,545 5,293 4,793 | 48,613 | 1,400,040, 1,417,468 | 1,182,220, 1,219,768 |
| Flint Pascal Q5 GGUF | 437, 437 | 4,981 4,763 3,641 4,707 3,631 4,683 3,595 4,141 4,055 | 38,197 | 1,403,820, 1,449,188 | 1,203,052, 1,234,308 |
| Ornith-1.5-122B Pascal N2P GGUF | 692, 687 | 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 | 49,139 | 1,485,476, 1,507,716 | 1,327,948, 1,361,224 |

- Every MiB figure is also the last check's, on v4.0's lines.
- After the three requests each card holds a little more, again the same on both builds (the TSV).
- MemAvailable is one reading per load, and the candidate always loaded first, so the two builds did not start from
  the same page cache. The watchdog's floor is 60 MB; it stopped no load.

## Part (4): launch groups per evaluation

Both builds' servers in part (3) ran with the fork's profiler in its light mode (`GGML_CUDA_PROFILE_MODE=graph`): one
pair of CUDA events around each whole evaluation, CUDA graphs, fusions and kernels left as they are, and one line per
evaluation and card with its launch groups. A fusion that stops matching after a merge shows only as more groups. The
same requests on the same tokens give the same evaluations, so the two builds' lines compare one for one
([launch-groups.tsv](launch-groups.tsv)).

**Every evaluation read on both builds launches the same groups, on every card:**

| File | Evaluations per card (each build; not ready) | (graph, launch groups) pairs, every card | Launch groups of one decoded token, cards 0–8 (both builds) |
|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 287 on cards 0–7, 1,405 on card 8 (the draft's graphs); 3 not ready on every card | the same | 311 286 266 288 230 267 249 267 on cards 0–7; card 8 runs several graphs (89, 39, 202, 85 groups, and 145 and 183 a few times) |
| Flint Pascal Q8 GGUF | 1,723; none not ready on card 0, a few on cards 1–8 (below) | the same for every evaluation read on both builds | 81 79 67 73 79 67 73 79 78 |
| Flint Pascal Q5 GGUF | 1,687; none not ready on card 0, a few on cards 1–8 (below) | the same for every evaluation read on both builds | 53 37 32 43 32 43 32 37 59 |
| Ornith-1.5-122B Pascal N2P GGUF | 769; none not ready | the same | 199 194 199 194 165 165 165 160 156 |

- **Flash-Next's fused routed launch is kept.** Upstream's
  [#32239](https://github.com/ggml-org/llama.cpp/pull/32239) turns off quantized matrix-vector fusion on every Pascal
  card; the candidate keeps the fork's default (fused on compute 6.1 unless `GGML_CUDA_MMID_GLU_UNFUSE=2`), which is
  upstream's rule only with `=2`. Flash-Next's line leaves the switch unset, and its decoded token launches 311 286
  266 288 230 267 249 267 groups on cards 0–7 on both builds, the figures of the last two checks: an unfused launch
  would add groups. Ornith's line sets `=2` (its measured gain) on both builds.
- **"Not ready" evaluations:** the light profiler writes an evaluation whose events had not completed when it read
  them as not ready. Every one is a pass of the long prompt (its prompt graph): Flash-Next's evaluations 180 to 182
  (879 to 881 on card 8), the same on both builds; Flint's 64-token passes, which the nine cards work on as a
  pipeline, so a card is often still busy when the host reads it (Flint Pascal Q8 GGUF's evaluations 1,129 to 1,132,
  Flint Pascal Q5 GGUF's 1,038 to 1,041). On Flint a few of them were ready on one build and not on the other, so the
  per-card sets of pairs in [launch-groups.tsv](launch-groups.tsv) differ by those evaluations alone. Matched by index
  ([eval-pairs.tsv](eval-pairs.tsv)), every evaluation read on both builds has the same graph and launch groups, and
  every card has the same number of evaluations on both builds.
- **Ornith's decoded token launches more groups than in the last check** (199 against 175 on card 0): its v5 line sets
  `GGML_CUDA_MMID_GLU_UNFUSE=2`, so its expert matrix-vector products run unfused. Both builds launch the same.
- The decoded-token columns count an evaluation as one token's when its card time is under 60 ms
  (`scripts/prof_summary.py`'s rule), so an evaluation near that line can count on one build only; the pair column
  compares every evaluation regardless of time.

## What the watch list asked for

The [merge's watch list](../2026-11-24_c912_upstream_b12674/watch.md) named three things to see first on a P106.

| Watch | What this check shows | What it leaves |
|---|---|---|
| The prompt kernels of 64-token micro-batches (#31352's 128-row tiles; #32374's 64-bit offsets) | the long prompt's 64-token passes give the same replies and launch the same groups on both builds, on Flint (both files) and Flash-Next | their speed: one reading per load with the profiler on is no measure (the server's prompt times moved either way between builds: 1,420 against 2,197 ms on Flint Pascal Q8 GGUF, 2,338 against 1,451 ms on Flint Pascal Q5 GGUF). Q1757 times them |
| Each load's start (#30672's NVML probe) | every load took the same seconds on both builds (the table above); the server's first log line came later on the candidate on every file: 0.418 against 0.238 s (Flash-Next), 0.291 against 0.217 s (Flint Pascal Q8 GGUF), 0.398 against 0.249 s (Flint Pascal Q5 GGUF), 0.358 against 0.279 s (Ornith) ([load-start.tsv](load-start.tsv)) | whether the probe costs anything measurable: one reading per load. Q1758 measures it |
| Flash-Next's fused routed launch (#32239) | kept: the same launch groups as `af00bb13b` and as the last two checks | the default itself is the owner's call (Q1613) |

## Not checked here

- **Speed.** Nothing is timed: the profiler ran on both servers, the cards had stepped down, and each figure is one
  reading. One coding-agent run each on Flash-Next and Ornith on the candidate is the speed gate before
  `pascal-p106-v5` moves.
- **The 16 new `IQ4_XS` cases on a P106 against stored CPU results** (Q1771).
- **Why `cohere2`'s card result moved** (Q1772).
- **The switches on.** With a switch set the candidate runs the same switch code as `af00bb13b`; each switch has its
  own rows.

## Files

| File | What |
|---|---|
| [replays.tsv](replays.tsv) | every replayed run of `v4kern` and `ro8kt`: what it must do, cases passed, FAIL lines, exit status, verdict, unrecorded cases |
| [unrecorded.tsv](unrecorded.tsv) | the 16 cases the candidate lists that the record sets do not hold, with the runs that selected each |
| [archs.tsv](archs.tsv) | the three `test-llama-archs -b CUDA0` runs: summary, rows OK, FAIL lines, exit status, card-mask lines |
| [logits.tsv](logits.tsv) | per random model: the candidate's CPU logits against `af00bb13b`'s (nmse, largest difference, values differing) and each build's card-against-CPU nmse |
| [logits-vs-other-machine.tsv](logits-vs-other-machine.tsv) | per random model: the candidate-against-`af00bb13b` nmse and differing values on this rig and on the build machine |
| [load-memory.tsv](load-memory.tsv) | per file and build: seconds to load, MiB per card 20 s after it answered and after the requests, MemAvailable then and its lowest over the load |
| [load-replies.tsv](load-replies.tsv) | per file, build and request: prompt tokens, tokens generated, md5 of the text, drafted and accepted tokens, the server's generation speed (a diagnostic, profiled), the first characters |
| [reply-agree.tsv](reply-agree.tsv) | per file and request: the candidate's reply against the reference's (the same bytes or not, common prefix, lengths) |
| [greedy-vs-v4.tsv](greedy-vs-v4.tsv) | `rig_greedy.sh`'s text on the candidate against v4.0's from R153 part (3) |
| [launch-groups.tsv](launch-groups.tsv) | per file and card: evaluations read on each build, whether the (graph, launch groups) pairs and their counts are the same, the decoded tokens' launch groups on each build |
| [eval-pairs.tsv](eval-pairs.tsv) | per file and card: evaluations matched by index, those read on both builds and how many of them have the same graph and launch groups, the not-ready evaluations on each build and what the other build read for them |
| [load-start.tsv](load-start.tsv) | per file and build: the server log's time of its first line and of the model load's start, and the seconds to load |
| [run/cand_load.sh](run/cand_load.sh) | parts (3) and (4), from a machine that only runs ssh |
| [run/load_tables.py](run/load_tables.py) | the memory and reply tables |
| [run/eval_pairs.py](run/eval_pairs.py) | `eval-pairs.tsv` |
| [run/extra_tables.py](run/extra_tables.py) | `load-start.tsv`, `unrecorded.tsv` and `logits-vs-other-machine.tsv` |

Part (2) ran the earlier checks' [run/rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh) unchanged, and
[cand_tables.py](../2026-11-17_v5_cand_p106_check/run/cand_tables.py) made the other tables. The raw logs (consoles,
server logs, card logs, profiles) are not published: they name the machine.

## Commands

```bash
NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674
REF=llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2
# each tarball from the build machine to the rig (md5 on both ends), then unpacked
ssh $BUILD "cat $WORK/$NEW.tar.gz" | ssh $RIG "cat > /opt/llama-builds/$NEW.tar.gz"
ssh $RIG "cd /opt/llama-builds && md5sum $NEW.tar.gz && tar xzf $NEW.tar.gz && rm $NEW.tar.gz \
  && md5sum $NEW/bin/llama-server $NEW/bin/test-backend-ops $NEW/bin/test-llama-archs"

# part (2) on the rig, card 0: rig_suite.sh with this repository's scripts it calls copied to $R/repo
# (scripts/replay_checks.sh, docs/data/2026-11-03_upstream_rebase/run/check.sh, docs/data/2026-11-04_moe_skip/run/gpu_checks.sh)
ssh $RIG "R=$R REPO=$R/repo SETS=<folder with v4kern/ and ro8kt/> NEW=/opt/llama-builds/$NEW REF=/opt/llama-builds/$REF \
  STEPS='v4kern ro8kt archs off' setsid -f nohup timeout 14400 bash $R/rig_suite.sh > $R/out/console.txt 2>&1 < /dev/null"

# parts (3) and (4), from the repository root on a machine that only runs ssh, started when part (2) wrote $R/out/done
RIG=$RIG NEW=$NEW REF=$REF REL=v5 \
  FN=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  FLINT_Q5=$MODELS/flint-pascal-Q5.gguf FLINT_Q8=$MODELS/flint-pascal-Q8.gguf \
  ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf RIGDIR=<rig folder> OUT=<local folder> \
  LABEL_NEW=cand LABEL_REF=ref PROFILE=1 GREEDY_BUILDS=$NEW \
  GREEDY_EXPECT='flashnext=4384ebc02147be1ee6f38d1ce4a43f21 flintq8=c5b2d8f20654728edfe38c37458b4c5c flintq5=d72ca1b968a8c1eb2b1ebeea22458ff9 ornith=35f987c059c7d2f521404e964995cd2d' \
  timeout 14400 bash docs/data/2026-11-24_c912_p106_check/run/cand_load.sh

# the tables (standard library only)
D=docs/data/2026-11-24_c912_p106_check
python3 -I $D/run/load_tables.py <cand_load.sh's OUT> $D/
python3 -I docs/data/2026-11-17_v5_cand_p106_check/run/cand_tables.py <rig_suite.sh's out/> <cand_load.sh's OUT> $D/ cand ref
python3 -I $D/run/eval_pairs.py <cand_load.sh's OUT> $D/ cand ref
python3 -I $D/run/extra_tables.py <cand_load.sh's OUT> <rig_suite.sh's out/> <the build machine's test-llama-archs compare log> $D/
```

`GREEDY_EXPECT` holds v4.0's texts of R153 part (3) (the same `rig_greedy.sh` command, files and split).
`load_tables.py` also writes `greedy.tsv`, which `greedy-vs-v4.tsv` replaces here.

Provenance: Miner Llama's fork and upstream llama.cpp (MIT); the release files as published on the project's model
pages; the random models are the ones `test-llama-archs` generates.
