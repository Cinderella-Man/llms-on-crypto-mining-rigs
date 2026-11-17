# The v5 candidate on a P106: `pascal-p106-v5-cand-b12410` against v4.0 (2026-11-17)

**Passed, so `pascal-p106-v5` now holds upstream b12410.** On a P106 rig the candidate runs every release file
exactly as v4.0 does. The replies are the same bytes, memory on every card is the same to the MiB, and every
evaluation launches the same kernels. The fork's kernel tests pass with the same counts as v4.0's check, and the
planted faults are still caught. One random test model (qwen2vl) gives different numbers. That comes from upstream's
fix to that model (#32241), not from a kernel, and no release file uses it. On 2026-11-17 at 20:56 `pascal-p106-v5`
was fast-forwarded from `a112712e3` to the candidate `1f1110c67`.

The candidate is [`pascal-p106-v5-cand-b12410`](../2026-11-17_v5_cand_b12410/README.md) `1f1110c67`: the v5 line's
three switches (C861, C865, C866, each off by default) with upstream llama.cpp b12410 merged in
([C854](../2026-11-17_c854_upstream/README.md)). Its rig build is
`llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410.tar.gz`, md5 `5bdcfc21c6b4b0f59a9f7a1d8fed2c08`. The
reference is v4.0 itself: the rig build v4.0 shipped, `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(tarball md5 `ac5f32d66ec286eeef4cf28f5d484816`, `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`), built from
`eb312b7bc`, whose tree the tag `miner-llama-v4.0` (`eb82916d3`) holds unchanged.

One rig (nine P106-090, the 4 GB host), every card at stock: memory and core clock offsets 0 on all nine (36 NVML
readings, [bench/mem_offset](../../../bench/mem_offset/)'s `nvml_memoc.py probe`), the 75 W power limit, 1,911 / 4,004
MHz maximum clocks, no Xid. No clock, power or offset setting was changed. Nothing here is timed.

## Part (2): the kernel suite on one card

[run/rig_suite.sh](run/rig_suite.sh), card 0, 18:30–19:05, the candidate's binaries, every switch a run does not set
at its default. The record sets are R153's
([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md#on-a-p106-r153-q1463)): references computed on a
build machine's CPU with v4-rc1's `test-backend-ops`, replayed here by
[scripts/replay_checks.sh](../../../scripts/replay_checks.sh), so the card computes and the stored CPU results judge
it.

**Upstream's attention rename does not touch these sets.** b12410 appends `,mask_heads=<n>` to every attention case's
name (#32217), so a set recorded before the merge misses its attention cases. Both sets here hold only `MUL_MAT`
cases (the fork's matrix-vector kernels at 1–8 tokens), whose names b12410 keeps: every selected case was found in
its record (0 unrecorded in every run), so nothing had to be recorded again or counted apart.

| Check | Result | R153 (v4-rc1, the same sets) |
|---|---|---|
| `v4kern`: C796's three kernel suites (Q6_K, IQ4_XS/Q4_K and the read-once kernel at 5–8 tokens) and all three together, 47 runs ([replays.tsv](replays.tsv)) | **47 of 47 as expected**: 16,240 case replays; every case of the 41 passing runs passes; the 6 planted faults fail cases (`q6kmt` 42, `dmtkt` 5, `ro8` 193, and with all three kernels on 42, 5, 193) | 47 of 47, 16,240, the same failed-case counts |
| `ro8kt`: C794's K tiles at 5–8 tokens, 16 runs | **16 of 16 as expected**: 3,232 case replays, 202 of 202 in every passing run (the fault with the tiles off among them); the planted fault fails 34 | 16 of 16, 3,232, 34 |
| `test-llama-archs -b CUDA0` plain, the fork's paths forced on (`=2`), and the card-built attention mask checked (C662's [check.sh](../2026-11-03_upstream_rebase/run/check.sh) `NATIVE=1`; [archs.tsv](archs.tsv)) | **154 of 154 tests pass three times**, 0 FAIL lines; 1,579 card masks identical to the host's. The 154th is upstream's new Inkling 2 (#32217) | 153 of 153 three times, 1,579 masks |
| The random models' logits against v4.0's: v4.0's `test-llama-archs` writes each model's logits, the candidate's compares (the -moeskip check's `archs-off` run, `-b CUDA0 -s 1`; [logits.tsv](logits.tsv)) | **145 of 146 the same bits.** qwen2vl differs in all 32,768 values (nmse 1.099e-01): upstream's own fix to that model, below. Inkling 2 is new (written, nothing to compare) | — |

**The logits these files hold are the CPU's.** `LLAMA_ARCHS_LOGITS` writes the CPU backend's logits of each random
model, and the card is checked against the CPU inside each run (the nmse column of the test's table). So the
comparison says the two builds' CPU paths agree; for the card, each model's card-against-CPU error is the same number
on both builds for all 145 (`card_vs_cpu_nmse_*` in [logits.tsv](logits.tsv)).

**qwen2vl: upstream #32241, a fix to that model's graph.** b12410's `qwen2vl : apply FFN weight scales`
([#32241](https://github.com/ggml-org/llama.cpp/pull/32241)) passes the per-tensor FFN weight scales
(`ffn_up_s`, `ffn_gate_s`, `ffn_down_s`) to qwen2vl's feed-forward block, which loaded them and then ignored them
(upstream: garbage output on NVFP4 files). The random model has those scales: a model built without a file
(`llama_model_init_from_user`, as `test-llama-archs` builds them) gets every optional tensor the loader asks for, and
the test fills it with random values. So v4.0's qwen2vl ignores random scales that the candidate applies, and every
logit changes. It is a graph change, not a kernel change: on each build the card agrees with that build's own CPU
(nmse 6.32e-14 on v4.0, 3.00e-14 on the candidate), and no other model changes. None of the release files is a
qwen2vl model.

## Part (3): the release files on both builds

[run/cand_load.sh](run/cand_load.sh), all nine cards, 19:06–20:54, after part (2). It is R153 part (3)'s
[r153_load.sh](../2026-11-13_c831_upstream/run/r153_load.sh) on the **v4.0 release lines**
([release/flash-next-v4](../../../release/flash-next-v4/settings.env), [flint-v4](../../../release/flint-v4/settings.env),
[ornith-v4](../../../release/ornith-v4/settings.env); Flint Pascal Q5 GGUF on Flint's line without `-ts`, as the
preset's README says), every switch the line does not set at its default, so all three v5 switches are unset. Per file,
both builds in turn (the candidate first):
- one llama-server under the RAM watchdog (`bench/agent_check/rig_server.sh`); 20 s after it answers, the memory of
  each card and the host's MemAvailable;
- two greedy requests (`/completion`, temperature 0, 256 tokens, no prompt cache): a coding request and a prose
  paragraph;
- then [rig_greedy.sh](../2026-11-05_ladder5/run/rig_greedy.sh) decodes the coding request with `llama-completion`,
  one token a pass, at `-c 4096` on R153's split with every switch at its default. It ran on the candidate only: its
  text is compared with v4.0's text from R153 part (3), the same command, files and split on this rig (the script would
  have run v4.0 again for a file whose text differed).

The files: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table as the third
shard) with its MTP draft; Flint Pascal Q8 GGUF (`p065`); Flint Pascal Q5 GGUF (K2h-r16); Ornith-1.5-122B Pascal N2P
GGUF (`valid1`).

**Every reply is the same bytes on both builds** ([reply-agree.tsv](reply-agree.tsv),
[load-replies.tsv](load-replies.tsv), [greedy-vs-v4.tsv](greedy-vs-v4.tsv)):

| File | Server, coding request | Server, prose | Drafted and accepted (coding; prose) | `rig_greedy.sh` against v4.0's text |
|---|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | the same, 743 characters | the same, 1,214 | 242 and 174; 278 and 162, on both builds | the same, md5 `4384ebc02147be1ee6f38d1ce4a43f21`, 745 B |
| Flint Pascal Q8 GGUF | the same, 813 | the same, 1,187 | — | the same, md5 `c5b2d8f20654728edfe38c37458b4c5c`, 972 B |
| Flint Pascal Q5 GGUF | the same, 1,288 | the same, 1,335 | — | the same, md5 `d72ca1b968a8c1eb2b1ebeea22458ff9`, 1,113 B |
| Ornith-1.5-122B Pascal N2P GGUF | the same, 728 | the same, 1,257 | — | the same, md5 `35f987c059c7d2f521404e964995cd2d`, 974 B |

The replies are sane text: Flash-Next and Ornith begin their reasoning about the request, Flint continues the raw
prompt (`/completion` applies no chat template).

**Every file loads with the same memory on every card to the MiB** ([load-memory.tsv](load-memory.tsv)):

| File | Load, s (candidate, v4.0) | MiB per card 20 s after it answered, cards 0–8 (both builds) | Sum, MiB | MemAvailable 20 s after, kB (candidate, v4.0) | Lowest during the load, kB (candidate, v4.0) |
|---|---|---|---:|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 317, 313 | 5,007 5,059 4,911 5,075 4,175 4,927 4,345 4,939 5,545 | 43,983 | 856,612, 881,328 | 428,760, 463,420 |
| Flint Pascal Q8 GGUF | 591, 592 | 5,383 5,923 5,081 5,591 5,923 5,081 5,545 5,293 4,793 | 48,613 | 1,437,288, 1,452,812 | 1,234,896, 1,245,368 |
| Flint Pascal Q5 GGUF | 435, 432 | 4,981 4,763 3,641 4,707 3,631 4,683 3,595 4,141 4,055 | 38,197 | 1,436,016, 1,446,088 | 1,232,920, 1,235,804 |
| Ornith-1.5-122B Pascal N2P GGUF | 688, 685 | 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 | 49,139 | 1,520,720, 1,528,456 | 1,376,384, 1,362,292 |

- After the two requests each card holds a little more, again the same on both builds (the TSV).
- MemAvailable is one reading per load, and the candidate always loaded first, so the two builds did not start from
  the same page cache (as in R153). The lowest reading, 428,760 kB (Flash-Next on the candidate, while it loads), is
  far above the watchdog's 60 MB floor; the watchdog stopped no load.
- The most swap in use during a load was about the same on both builds (kB, candidate then v4.0): Flash-Next 30,580
  and 30,544, Flint Q8 308,440 and 311,244, Flint Q5 310,096 and 312,988, Ornith 352,844 and 352,612.
- The server logs hold no error on either build; Flint's line prints the usual fit-step warning (`-ngl 99` is set by
  hand) on both.

The load times are a diagnostic, not a speed result: the servers ran with the profiler of part (4).

## Part (4): launch groups per evaluation

A fusion that stops matching after a merge passes every test and shows only as more kernel launches per token. So
both builds' servers in part (3) ran with the fork's profiler in its light mode (`GGML_CUDA_PROFILE_MODE=graph`), which
puts one pair of CUDA events around each whole evaluation and leaves CUDA graphs, fusions and kernels as they are. It
writes one line per evaluation and card with its launch groups (one launch group is one kernel or one fused group).
The same two requests on the same tokens give the same evaluations, so the two builds' lines can be compared one for
one ([launch-groups.tsv](launch-groups.tsv)).

**Every evaluation on every card launches the same groups on both builds**, with none left unread:

| File | Evaluations per card (each build) | (graph, launch groups) pairs, every card | Launch groups of one decoded token, cards 0–8 (both builds) |
|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 175 on cards 0–7, 868 on card 8 (the draft's graphs) | the same | 311 286 266 288 230 267 249 267 on cards 0–7; card 8 runs several graphs (89, 39, 202, 85 groups) |
| Flint Pascal Q8 GGUF | 1,116 | the same | 81 79 67 73 79 67 73 79 78 |
| Flint Pascal Q5 GGUF | 1,025 | the same | 53 37 32 43 32 43 32 37 59 |
| Ornith-1.5-122B Pascal N2P GGUF | 512 | the same | 175 170 175 170 145 145 145 140 140 |

The table's decoded-token columns count an evaluation as one token's when its GPU time is under 60 ms
(`scripts/prof_summary.py`'s rule), so an evaluation near that line can count on one build only: on a few cards one
more evaluation counts on one build (Flash-Next's card 0, a 311-group one; Flint's, prompt evaluations of 156 to 187
groups; the TSV). The pair column compares every
evaluation regardless of time, and it is the same everywhere.

This runs on the server's release lines rather than the planned `llama-completion` decode at `-n 64`: the same
check (launch groups per evaluation, per card, both builds), on the line the release runs, without eight more loads.

## Not checked here

- **Speed.** Nothing is timed; the first agent check on `pascal-p106-v5` measures it.
- **The switches on.** Each of the three v5 switches has its own rig row on the switch merge's build
  ([2026-11-17_v5_merge_switches](../2026-11-17_v5_merge_switches/README.md)); with a switch set, the candidate runs
  the same switch code.
- **C662's whole `test-backend-ops`** and the record sets past R153's two (gpu_checks.sh's `experts` and `dense`
  parts): R153 left them out for rig time, and the candidate's machine code with every switch unset is C854's build's
  ([2026-11-17_v5_cand_b12410](../2026-11-17_v5_cand_b12410/README.md)).

## Files

| File | What |
|---|---|
| [replays.tsv](replays.tsv) | every replayed run of `v4kern` and `ro8kt`: what it must do, cases passed, FAIL lines, exit status, verdict, unrecorded cases |
| [archs.tsv](archs.tsv) | the three `test-llama-archs -b CUDA0` runs: summary, rows OK, FAIL lines, exit status, card-mask lines |
| [logits.tsv](logits.tsv) | per random model: the candidate's CPU logits against v4.0's (nmse, largest difference, values differing) and each build's card-against-CPU nmse |
| [load-memory.tsv](load-memory.tsv) | per file and build: seconds to load, MiB per card 20 s after it answered and after the requests, MemAvailable then and its lowest over the load |
| [load-replies.tsv](load-replies.tsv) | per file, build and request: tokens, md5 of the text, drafted and accepted tokens, the server's generation speed (a diagnostic, profiled), the first characters |
| [reply-agree.tsv](reply-agree.tsv) | per file and request: the candidate's reply against v4.0's (the same bytes or not, common prefix, lengths) |
| [greedy-vs-v4.tsv](greedy-vs-v4.tsv) | `rig_greedy.sh`'s text on the candidate against v4.0's from R153 part (3) |
| [launch-groups.tsv](launch-groups.tsv) | per file and card: evaluations read on each build, whether the (graph, launch groups) pairs and their counts are the same, the decoded tokens' launch groups on each build |
| [run/rig_suite.sh](run/rig_suite.sh) | part (2) on the rig, card 0 |
| [run/cand_load.sh](run/cand_load.sh) | parts (3) and (4), from a machine that only runs ssh |
| [run/cand_tables.py](run/cand_tables.py) | the tables above but the memory and reply ones, which [r153_load_tables.py](../2026-11-13_c831_upstream/run/r153_load_tables.py) makes |

The raw logs (consoles, server logs, card logs, profiles) are not published: they name the machine.

## Commands

```bash
# the candidate's rig build to the rig (from a machine that reaches both; md5 on both ends), unpacked next to v4.0's
ssh <build machine> "cat <folder>/llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410.tar.gz" \
  | ssh $RIG "cat > /opt/llama-builds/llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410.tar.gz"
ssh $RIG "cd /opt/llama-builds && md5sum <tarball> && tar xzf <tarball> && rm <tarball>"

# part (2) on the rig, card 0: run/rig_suite.sh with this repository's scripts it calls copied to $R/repo, R153's sets
ssh $RIG "R=$R REPO=$R/repo SETS=<folder with v4kern/ and ro8kt/> NEW=/opt/llama-builds/<candidate> \
  REF=/opt/llama-builds/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 STEPS='v4kern ro8kt archs off' \
  setsid -f nohup bash $R/rig_suite.sh > $R/out/console.txt 2>&1 < /dev/null"

# parts (3) and (4), from the repository root on a machine that only runs ssh (here started when part (2) ended)
RIG=$RIG NEW=<candidate folder> REF=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 \
  FN=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  FLINT_Q5=$MODELS/flint-pascal-Q5.gguf FLINT_Q8=$MODELS/flint-pascal-Q8.gguf \
  ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf RIGDIR=<rig folder> OUT=<local folder> \
  LABEL_NEW=cand PROFILE=1 GREEDY_BUILDS=<candidate folder> \
  GREEDY_EXPECT='flashnext=4384ebc02147be1ee6f38d1ce4a43f21 flintq8=c5b2d8f20654728edfe38c37458b4c5c flintq5=d72ca1b968a8c1eb2b1ebeea22458ff9 ornith=35f987c059c7d2f521404e964995cd2d' \
  bash docs/data/2026-11-17_v5_cand_p106_check/run/cand_load.sh

# the tables (standard library only)
python3 -I docs/data/2026-11-13_c831_upstream/run/r153_load_tables.py <OUT> docs/data/2026-11-17_v5_cand_p106_check/
python3 -I docs/data/2026-11-17_v5_cand_p106_check/run/cand_tables.py <rig_suite.sh's out/> <OUT> docs/data/2026-11-17_v5_cand_p106_check/
```

`GREEDY_EXPECT` holds v4.0's texts of R153 part (3) (the same `rig_greedy.sh` command, files and split), so the greedy
step runs the candidate only and runs v4.0 again for a file whose text differs. The release files are the ones
[R153 part (3)](../2026-11-13_c831_upstream/README.md#part-3-the-release-files-load-on-v4-rc2) loaded.
