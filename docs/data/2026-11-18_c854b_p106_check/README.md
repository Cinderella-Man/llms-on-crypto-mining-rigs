# Upstream b12483 on a P106: `pascal-p106-v5-cand-b12483` against `pascal-p106-v5` (2026-11-18)

**Passed.** On a P106 rig the candidate on upstream b12483 runs every release file exactly as `pascal-p106-v5`
(`1f1110c67`, upstream b12410) does. The replies are the same bytes, memory on every card is the same to the MiB, and
every evaluation launches the same kernels. The fork's kernel tests give the same table as the last check, and the
planted faults are still caught. Two random test models give different numbers: qwen3next and qwen4exp. Both come
from upstream fixes that now apply an optional weight scale these models' graphs used to ignore (#32393, #32476),
not from a kernel. Flash-Next, the one release file of either type, runs the same graph and gives the same replies.
`pascal-p106-v5` is not moved by this check alone: upstream rewrote the server's slot scheduling between the two tags,
so one agent run each on Ornith and Flash-Next comes first.

**Part (5), Ornith's agent run (14:00–14:40):** 12 of 12 tasks, generation as v4.0's, and the new scheduler read
fewer prompt tokens a task than v4.0 did, not more. Prompts were 2.5 % slower on the same requests, a loss by the
earlier runs' rule; but the v4.0 runs it is compared with ran on the other rig, and that rig alone has moved as much
between days, so the engine's share needs a reference on this rig.

The candidate is [`pascal-p106-v5-cand-b12483`](../2026-11-17_c854b_upstream_newest/README.md) `602691153`:
`pascal-p106-v5` with upstream llama.cpp b12483 merged in and upstream's open #32434 (the slot scheduler's fix for
recurrent and hybrid models, and LoRA) carried as one commit. Its rig build is
`llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483.tar.gz`, md5 `4e2cf20dbcb2a144be88d1eee1ec976d`
(`llama-server` `86607661d5a19d1a90d19ee4db10c90b`, `test-backend-ops` `624c544cb27a89e50ccb321331c8fc72`,
`test-llama-archs` `373ee35accf4a4327903ad5a12c39dea`, `llama-completion` `cf3c298e23857d701f9c555509319f1d`). The
reference is `pascal-p106-v5`'s own rig build, `llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410` (tarball md5
`5bdcfc21c6b4b0f59a9f7a1d8fed2c08`, `llama-server` `5b134f43f3a3e885bfba02538ffebcbe`), which passed the
[same check against v4.0](../2026-11-17_v5_cand_p106_check/README.md) on 2026-11-17. So a difference here names
upstream b12410..b12483 or #32434.

One rig (nine P106-090, the 4 GB host), 11:16–13:45, every card at stock: the 75 W power limit, 1,911 / 4,004 MHz
maximum clocks, no Xid, no clock, power or offset setting changed. Every switch was unset. Nothing here is timed.

## Part (2): the kernel suite on one card

[run/rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh) of the last check, unchanged, on card 0,
11:18–11:55, the candidate's binaries. The record sets are R153's
([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md#on-a-p106-r153-q1463)): references computed on a
build machine's CPU with v4-rc1's `test-backend-ops`, replayed here by
[scripts/replay_checks.sh](../../../scripts/replay_checks.sh), so the card computes and the stored CPU results judge
it. Both sets hold only `MUL_MAT` cases, and every selected case was found in its record (0 unrecorded), so nothing
needed a new CPU recording.

| Check | Result | The last check (`1f1110c67`, the same sets) |
|---|---|---|
| `v4kern`: C796's three kernel suites and all three together, 47 runs ([replays.tsv](replays.tsv)) | **47 of 47 as expected**, 16,240 case replays; the 6 planted faults fail 42, 5, 193, 42, 5 and 193 cases | the same table, row for row |
| `ro8kt`: C794's K tiles at 5–8 tokens, 16 runs | **16 of 16 as expected**, 3,232 case replays; the planted fault fails 34 | the same |
| `test-llama-archs -b CUDA0` plain, the fork's paths forced on (`=2`), and the card-built attention mask checked ([archs.tsv](archs.tsv)) | **155 of 155 tests pass three times**, 0 FAIL lines; 1,590 card masks identical to the host's | 154 of 154, 1,579 masks |
| The random models' logits against `1f1110c67`'s (`archs-off`, `-b CUDA0 -s 1`; [logits.tsv](logits.tsv)) | **145 of 147 the same bits**; qwen3next and qwen4exp differ (below). zonos3 is new (written, nothing to compare) | — |

**The new test is zonos3**, upstream's zonos3-mini TTS model
([#32020](https://github.com/ggml-org/llama.cpp/pull/32020)); its 11 mask checks make the 1,590.

**The logits these files hold are the CPU's** (`LLAMA_ARCHS_LOGITS` writes the CPU backend's logits), and the card is
checked against the CPU inside each run. For the 145 models with the same logits, each model's card-against-CPU error
is the same number on both builds.

**qwen3next: upstream #32393, a fix to that model's graph.** b12483's `qwen3next : apply output projection weight
scale` ([#32393](https://github.com/ggml-org/llama.cpp/pull/32393)) passes the linear-attention layers' optional
`ssm_out.scale` to their output projection, as qwen35 and qwen35moe already did; the loader asked for that tensor at
b12410 too, and qwen3next's graph ignored it. A model built without a file (`test-llama-archs`) gets every optional
tensor, filled with a normal draw, so the candidate multiplies each linear-attention output by a random scalar that
the reference left out. The logits move little (nmse 5.890e-10, largest difference 2.229e-05, 32,687 of 32,768
values), and the card agrees with that build's own CPU on both (nmse 2.65e-14 on `1f1110c67`, 2.78e-14 on the
candidate). qwen35 and qwen35moe, which share the gated delta net, the hybrid memory and the scaled output projection,
give the same bits on both builds. A file without `ssm_out.scale` runs the same graph as before: Swift Qwen3.8-Flash-Next
Pascal Q2 GGUF, a qwen3next model, gives the same replies and the same launch groups below.

**qwen4exp: upstream #32476.** `convert : fix FP8 qwen4exp checkpoints`
([#32476](https://github.com/ggml-org/llama.cpp/pull/32476)) adds an optional per-tensor scale for the per-layer
token embedding table, multiplied in after the gather. The random model has it, so every logit changes (nmse
3.877e-01); on each build the card agrees with its own CPU (4.06e-14 and 2.94e-14). No release file is a qwen4exp
model.

## Part (3): the release files on both builds

[run/cand_load.sh](../2026-11-17_v5_cand_p106_check/run/cand_load.sh) of the last check, unchanged, all nine cards,
11:56–13:45, after part (2): each file on its **v4.0 release line**
([release/flash-next-v4](../../../release/flash-next-v4/settings.env), [flint-v4](../../../release/flint-v4/settings.env),
[ornith-v4](../../../release/ornith-v4/settings.env); Flint Pascal Q5 GGUF on Flint's line without `-ts`), every switch
the line does not set at its default. Per file, the candidate first, then the reference: one llama-server under the
RAM watchdog, its memory per card 20 s after it answered, two greedy requests (`/completion`, temperature 0, 256
tokens, no prompt cache: a coding request and a prose paragraph), and the profiler's light mode on both servers for
part (4). Then [rig_greedy.sh](../2026-11-05_ladder5/run/rig_greedy.sh) decodes the coding request with
`llama-completion`, one token a pass, at `-c 4096`, on the candidate only, compared with v4.0's text from R153 part (3).

The files: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (UkisAI's Swift-1.5 GSQ-RCO Q2_0, its n-gram table as the third
shard) with its MTP draft; Flint Pascal Q8 GGUF (`p065`); Flint Pascal Q5 GGUF (K2h-r16); Ornith-1.5-122B Pascal N2P
GGUF (`valid1`).

**Every reply is the same bytes on both builds**, and the same as on the last check, where `1f1110c67` matched v4.0
([reply-agree.tsv](reply-agree.tsv), [load-replies.tsv](load-replies.tsv), [greedy-vs-v4.tsv](greedy-vs-v4.tsv)):

| File | Server, coding request | Server, prose | Drafted and accepted (coding; prose) | `rig_greedy.sh` against v4.0's text |
|---|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | the same, 743 characters | the same, 1,214 | 242 and 174; 278 and 162, on both builds | the same, md5 `4384ebc02147be1ee6f38d1ce4a43f21`, 745 B |
| Flint Pascal Q8 GGUF | the same, 813 | the same, 1,187 | — | the same, md5 `c5b2d8f20654728edfe38c37458b4c5c`, 972 B |
| Flint Pascal Q5 GGUF | the same, 1,288 | the same, 1,335 | — | the same, md5 `d72ca1b968a8c1eb2b1ebeea22458ff9`, 1,113 B |
| Ornith-1.5-122B Pascal N2P GGUF | the same, 728 | the same, 1,257 | — | the same, md5 `35f987c059c7d2f521404e964995cd2d`, 974 B |

Flint's graph passes the file's selection bias (`ffn_exp_probs_b`) to the MoE router, the one place the one real
kernel the merge changed can run (`topk_moe_cuda<N, true>`, two register-zeroing moves in the other order; the
[build's kernel comparison](../2026-11-17_c854b_upstream_newest/README.md)). Both Flint files give the same bytes and
the same launch groups on both builds.

**Every file loads with the same memory on every card to the MiB** ([load-memory.tsv](load-memory.tsv)):

| File | Load, s (candidate, reference) | MiB per card 20 s after it answered, cards 0–8 (both builds) | Sum, MiB | MemAvailable 20 s after, kB (candidate, reference) | Lowest during the load, kB (candidate, reference) |
|---|---|---|---:|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 320, 314 | 5,007 5,059 4,911 5,075 4,175 4,927 4,345 4,939 5,545 | 43,983 | 874,828, 861,908 | 445,056, 436,532 |
| Flint Pascal Q8 GGUF | 594, 594 | 5,383 5,923 5,081 5,591 5,923 5,081 5,545 5,293 4,793 | 48,613 | 1,426,276, 1,440,316 | 1,211,348, 1,232,320 |
| Flint Pascal Q5 GGUF | 434, 431 | 4,981 4,763 3,641 4,707 3,631 4,683 3,595 4,141 4,055 | 38,197 | 1,429,092, 1,425,676 | 1,220,872, 1,238,740 |
| Ornith-1.5-122B Pascal N2P GGUF | 688, 686 | 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 | 49,139 | 1,507,000, 1,493,876 | 1,349,360, 1,359,480 |

- After the two requests each card holds a little more, again the same on both builds (the TSV). Every MiB figure is
  the last check's too.
- MemAvailable is one reading per load, and the candidate always loaded first, so the two builds did not start from
  the same page cache. The lowest reading, 436,532 kB, is far above the watchdog's 60 MB floor; the watchdog stopped no
  load.
- The load times are a diagnostic, not a speed result: the servers ran with the profiler of part (4).

## Part (4): launch groups per evaluation

Both builds' servers in part (3) ran with the fork's profiler in its light mode (`GGML_CUDA_PROFILE_MODE=graph`): one
pair of CUDA events around each whole evaluation, CUDA graphs, fusions and kernels left as they are, and one line per
evaluation and card with its launch groups. A fusion that stops matching after a merge shows only as more groups. The
same two requests on the same tokens give the same evaluations, so the two builds' lines compare one for one
([launch-groups.tsv](launch-groups.tsv)).

**Every evaluation on every card launches the same groups on both builds**, with none left unread:

| File | Evaluations per card (each build) | (graph, launch groups) pairs, every card | Launch groups of one decoded token, cards 0–8 (both builds) |
|---|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | 175 on cards 0–7, 868 on card 8 (the draft's graphs) | the same | 311 286 266 288 230 267 249 267 on cards 0–7; card 8 runs several graphs (89, 39, 202, 85 groups) |
| Flint Pascal Q8 GGUF | 1,116 | the same | 81 79 67 73 79 67 73 79 78 |
| Flint Pascal Q5 GGUF | 1,025 | the same | 53 37 32 43 32 43 32 37 59 |
| Ornith-1.5-122B Pascal N2P GGUF | 512 | the same | 175 170 175 170 145 145 145 140 140 |

The decoded-token figures are the last check's, so they are v4.0's too. The table's decoded-token columns count an
evaluation as one token's when its GPU time is under 60 ms (`scripts/prof_summary.py`'s rule), so an evaluation near
that line can count on one build only. On Flint's cards that happened a few times: an 81-group evaluation on Flint
Pascal Q8 GGUF's card 0, a 110-group one on Flint Pascal Q5 GGUF's card 8, and prompt evaluations of 156 to 188 groups
on a few others (the TSV). The pair column compares every evaluation regardless of time, and it is the same
everywhere.

## Part (5): the agent check on Ornith (2026-11-18)

**12 of 12 tasks passed, generation is v4.0's, and the new slot scheduler read less, not more.** Prompts were 2.5 %
slower on the 21 requests that read the same prompt in all three runs, and a task at equal work took 2.17 minutes
against v4.0's 2.11 and 2.13. By the rule of the earlier agent runs that is a **loss**. The two v4.0 runs it is
compared with ran on the other rig two days earlier, though, and the rule's band (two runs back to back on one rig)
is narrower than the error of such a comparison: on the second rig alone, v3.0's prompts on the same requests moved
1.8 % from one day to another, and this rig's cards ran 0.6 % slower while busy. A v5 reference on this rig is needed
to tell the engine from the rig.

The run is the agent check of [R166](../2026-11-16_r166_ornith_stack_no_taper/README.md)
([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and JavaScript alternating, a
12-minute cap a task, a 50-minute budget, the makers' sampling at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at 49,152 tokens, one arm, one load: **`o4c2`, the released
v4.0 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)) with the candidate's
`llama-server`** (md5 `86607661d5a19d1a90d19ee4db10c90b`). It is R166a's `o4sna` with only the build changed: the
started command (521 bytes, md5 `056afa770d54ddc800cf930eee9a723e`) is R166a's byte for byte apart from the build
folder's name, and the server's `switches:` line is R166's, `6 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1
GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`. It ran on
the same rig as parts (2) to (4), with the machine that drives the rig as the agent's client: the server started at
13:59:53 and was up after 683 s (this rig's loads of the file in part (3): 688 and 686 s), the tasks ran
14:11:18–14:39:27. Every card at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no Xid). The references
are R166a `o4sna` and R166b `o4snb`, the same preset on v4.0 (`eb82916d3`) on the second rig, 2026-11-16.

| | v4.0 `o4sna` (second rig) | v4.0 `o4snb` (second rig) | **candidate `o4c2` (this rig)** | class by the rule (against the mean; the spread of the two v4.0 runs) |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** | |
| minutes a task at equal work (`ka`'s work) | 2.11 | 2.13 | **2.17** | loss (+2.4 %; 0.02) |
| the 21 requests read on the same prompt in all three runs, prompt s | 211.4 | 210.9 | **216.4** | loss (+2.5 %; 0.5) |
| of them, the 12 first turns, prompt s | 115.8 | 115.3 | **119.2** | loss (+3.2 %; 0.5) |
| prompt tok/s, the whole session | 103.0 | 105.9 | **96.2** | loss (−7.9 %; 2.9); a different mix of prompt sizes |
| generation tok/s | 16.52 | 16.13 | **16.30** | negligible (−0.2 %; 0.39) |
| first token median s | 6.69 | 6.06 | **7.56** | loss (+18.6 %; 0.63); follows the turn mix |
| first token p90 s | 11.29 | 11.32 | **11.11** | gain (−1.7 %; 0.03) |
| first token on a task's first turn, median s | 8.43 | 8.45 | **8.58** | loss (+1.7 %; 0.02) |
| prompt tokens read a task | 3,405 | 4,130 | **3,259** | less work |
| reply tokens re-read (turns; their prompt s of all prompt s) | 4,831 (2; 30.6 of 396.6) | 12,332 (3; 71.9 of 467.8) | **2,143 (1; 13.2 of 406.6)** | |
| tokens generated a task | 1,818 | 2,300 | **1,689** | |
| minutes, all 12 tasks; turns | 29.1; 63 | 36.8; 73 | **28.0; 63** | different work |
| tool calls / failed / malformed | 63 / 2 / 0 | 73 / 2 / 0 | **63 / 0 / 0** | |
| checkpoints restored / created / erased; full re-processing | 13 / 0 / 0; 0 | 14 / 0 / 0; 0 | **12 / 0 / 0; 0** | |
| graph reallocations | 0 in 63 requests | 0 in 73 | **0 in 63** | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,888.2 | 1,888.2 | **1,876.9** | |
| lowest MemAvailable MiB | 1,225 | 1,228 | **1,199** | |

Each value is one run ([ornith_agent.tsv](ornith_agent.tsv), [ornith_same_prompts.md](ornith_same_prompts.md),
[ornith_work.md](ornith_work.md), [ornith_clocks.tsv](ornith_clocks.tsv), [ornith_host.tsv](ornith_host.tsv),
[ornith_reallocs.tsv](ornith_reallocs.tsv), [o4c2_gonogo.md](o4c2_gonogo.md); R166's rows from its folder). The class
is R166's rule: a gain or a loss when the difference from the two v4.0 runs' mean is larger than their spread (the
gap between them), negligible otherwise. Minutes a task at equal work price `ka`'s work (v3.0's 2026-11-06 run on
the second rig: 2.52) at each run's own prompt and generation speeds.

- **The work: the new slot scheduler loses no cache on Ornith's hybrid layers.** The agent read 3,259 prompt tokens a
  task (3,405 and 4,130 on v4.0) and read a reply again on one turn (Python `list-ops`, turn 3: 2,143 tokens, 13.2 s),
  where v4.0 did on two and three. The server restored a checkpoint 12 times, created and erased none and re-read no
  prompt in full; the report's cache hit is 85.8 % (85.5 % and 84.4 %; [o4c2_report_table.md](o4c2_report_table.md)).
  The work depends on what the model wrote at temperature 0.6, so fewer tokens read is no gain in itself; what the
  run shows is that the rewritten reuse does not read more.
- **Prompts on the same work: +2.5 %.** 21 requests read the same number of new tokens on the same cached prefix in
  all three runs: every task's first turn and nine second turns ([ornith_same_prompts.md](ornith_same_prompts.md)).
  They took 216.4 s against 211.4 and 210.9 s. 19 of the 21 are slower than in both v4.0 runs, most by 0.5 to 3 %;
  the first turn of JavaScript `triangle` (430 tokens) is the largest, 7.95 s against 7.11 and 6.75 s. The session's prompt tok/s (96.2 against 103.0 and
  105.9) also counts a different mix of prompt sizes: after the third turn this run's requests read fewer new tokens
  (a median 31 at turn 4, against 128 and 218), and a short prompt reads fewer tokens a second.
- **First token:** the median over all turns, 7.56 against 6.69 and 6.06 s, follows the turns' sizes (turn 3 read a
  median 449 new tokens, against 367 and 262; [o4c2_compare.md](o4c2_compare.md)). A task's first turn, the same
  request in every run, took 8.58 against 8.43 and 8.45 s; the p90 is 11.11 against 11.29 and 11.32 s.
- **Generation: negligible**, 16.30 against 16.52 and 16.13 tok/s.
- **The rule's band is narrower than this comparison's error.** R166's two runs ran back to back on one rig and agree
  to 0.2 % on the matched prompts (211.4 and 210.9 s). This comparison crosses rigs and days, and two measured
  differences of that kind are as large as the 2.5 %:
  - v3.0 on the second rig, with the same file, line and build each time, took 376.9 s (2026-11-06, `ka`), 370.4 s
    (2026-11-14, R160c `o3a`) and 371.6 s (2026-11-16, R160f `o3b`) on the 22 requests the three runs share: 1.8 %
    between days on one rig ([ornith_same_prompts_v3.md](ornith_same_prompts_v3.md)). The first rig's only published
    Ornith agent run on the v3 line, 2026-11-05, took 378.1 s on the same 22 requests, but on the build before the
    v3.0 tag and without the keep-alive, so it does not price the rig alone.
  - This rig's cards ran at a lower clock while busy: busy medians of 1,847 1,847 1,885 1,885 1,860 1,911 1,873 1,911
    1,873 MHz (cards 0–8), against 1,860 1,898 1,885 1,873 1,911 1,898 1,873 1,898 1,898 in both R166 runs; 1,876.9
    against 1,888.2 on the mean (−0.6 %). Both rigs run the same 75 W limit and stock clocks; under the limit each card
    boosts as far as its own chip allows ([ornith_clocks.tsv](ornith_clocks.tsv)).

  So the 2.5 % may be the rig, the day or the engine. The same launcher with `pascal-p106-v5`'s own build
  (`1f1110c67`, part (2)'s reference) on this rig, next to this run, tells them apart.
- **Host** ([ornith_host.tsv](ornith_host.tsv)): lowest MemAvailable 1,199 MiB at 14:39:36, during the last task
  (1,225 and 1,228 on v4.0), 3,105 MiB at the end; swap 30 MiB at the start and 341 MiB at its highest at 14:09:39,
  during the load (338 and 337). The RAM watchdog stopped nothing. This run's card log is `scripts/card_log.py`'s,
  which counts 75 MiB on each idle card that `nvidia-smi` (R166's card log) reads as 0 MiB. Its card peaks, 5,830
  5,846 5,776 5,846 5,454 5,562 5,724 5,796 5,202 MiB, are 5,755 5,771 5,701 5,771 5,379 5,487 5,649 5,721 5,127 less
  that 75, against R166b's 5,755 5,771 5,701 5,771 5,381 5,489 5,649 5,721 5,127. The report's memory gate
  ([o4c2_gonogo.md](o4c2_gonogo.md)) reads the client's log shared with R160's and R166's runs: NO-GO on its 718 MB
  low (R160d's) and its 341 MB swap, as for R166.
- **Graph reallocations:** 0 in 63 requests, in every prompt-size bucket
  ([o4c2_requests_reallocs.tsv](o4c2_requests_reallocs.tsv)).

`pascal-p106-v5` stays `1f1110c67`.

## Not checked here

- **The server's new slot scheduling under an agent on Swift Qwen3.8-Flash-Next Pascal Q2 GGUF.** Upstream's #32154
  and the carried #32434 change which cached state a turn reuses. Ornith's agent run is part (5); Flash-Next's, with
  its draft and the work counts, comes before `pascal-p106-v5` moves.
- **Speed on the same rig.** Parts (2) to (4) time nothing; part (5) is timed against references on another rig.
- **The switches on.** With a switch set the candidate runs the same switch code as `1f1110c67`; each switch has its
  own rig rows.
- **C662's whole `test-backend-ops`** and the record sets past R153's two, as in the last check. No case needed a new
  CPU recording, so none was left out for that reason.

## Files

| File | What |
|---|---|
| [replays.tsv](replays.tsv) | every replayed run of `v4kern` and `ro8kt`: what it must do, cases passed, FAIL lines, exit status, verdict, unrecorded cases |
| [archs.tsv](archs.tsv) | the three `test-llama-archs -b CUDA0` runs: summary, rows OK, FAIL lines, exit status, card-mask lines |
| [logits.tsv](logits.tsv) | per random model: the candidate's CPU logits against `1f1110c67`'s (nmse, largest difference, values differing) and each build's card-against-CPU nmse |
| [load-memory.tsv](load-memory.tsv) | per file and build: seconds to load, MiB per card 20 s after it answered and after the requests, MemAvailable then and its lowest over the load |
| [load-replies.tsv](load-replies.tsv) | per file, build and request: tokens, md5 of the text, drafted and accepted tokens, the server's generation speed (a diagnostic, profiled), the first characters |
| [reply-agree.tsv](reply-agree.tsv) | per file and request: the candidate's reply against the reference's (the same bytes or not, common prefix, lengths) |
| [greedy-vs-v4.tsv](greedy-vs-v4.tsv) | `rig_greedy.sh`'s text on the candidate against v4.0's from R153 part (3) |
| [launch-groups.tsv](launch-groups.tsv) | per file and card: evaluations read on each build, whether the (graph, launch groups) pairs and their counts are the same, the decoded tokens' launch groups on each build |
| [ornith_agent.tsv](ornith_agent.tsv) | part (5), per arm (R160's and R166's runs, and `o4c2`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` |
| [ornith_same_prompts.md](ornith_same_prompts.md) | the 21 requests read on the same prompt (task, turn, new tokens, cached tokens) in `o4c2`, `o4sna` and `o4snb`: each run's prompt ms; and each run's generation over all its requests |
| [ornith_same_prompts_v3.md](ornith_same_prompts_v3.md) | the same for v3.0's four Ornith runs: the first rig's of 2026-11-05 and the second rig's `ka`, `o3a`, `o3b` |
| [ornith_work.md](ornith_work.md), [ornith_work.tsv](ornith_work.tsv) | `report.py work` on the three runs' `turns.csv`: prompt tokens read, tokens generated, replies re-read, the longest first-token wait |
| [ornith_clocks.tsv](ornith_clocks.tsv) | each card's median SM clock while busy, and its share of busy samples below 1,800 MHz, over each run's tasks |
| [ornith_host.tsv](ornith_host.tsv) | the three runs' lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end, each card's peak memory (`o4c2`'s from the NVML card log, 75 MiB over `nvidia-smi`'s on an idle card) |
| [ornith_reallocs.tsv](ornith_reallocs.tsv), [o4c2_requests_reallocs.tsv](o4c2_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size (the three runs), and per request (`o4c2`) |
| [o4c2_compare.md](o4c2_compare.md), [o4c2_report_table.md](o4c2_report_table.md), [o4c2_gonogo.md](o4c2_gonogo.md) | the agent check's report on `o4c2`: per task, per arm and first token per turn; per arm over the client's attempts (R160's, R166's and two other runs); the gates and each server log's counts |
| [agent/o4c2/](agent/o4c2/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/ornith_agent.sh](run/ornith_agent.sh) | part (5)'s run and report |
| [run/ornith_tables.sh](run/ornith_tables.sh) | part (5)'s tables, with [run/same_prompts.py](run/same_prompts.py), [run/busy_clocks.py](run/busy_clocks.py) and [run/host_cards.py](run/host_cards.py) (R166's `host.py` for the NVML card log) |

In the tables `cand` is the candidate and `ref` is `pascal-p106-v5`'s build `1f1110c67`. The scripts are the last
check's, byte for byte: [rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh) (md5
`e8b260a425fc1ce9c0444054f7c82e21`), [cand_load.sh](../2026-11-17_v5_cand_p106_check/run/cand_load.sh)
(`9bab37202d25212029fa08463fb73437`) and [cand_tables.py](../2026-11-17_v5_cand_p106_check/run/cand_tables.py), with
[r153_load_tables.py](../2026-11-13_c831_upstream/run/r153_load_tables.py) for the memory and reply tables. The raw
logs (consoles, server logs, card logs, profiles) are not published: they name the machine.

## Commands

```bash
NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483
REF=llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410
# the candidate's rig build to the rig (md5 on both ends), unpacked next to the reference
ssh $RIG "cat > /opt/llama-builds/$NEW.tar.gz" < $NEW.tar.gz
ssh $RIG "cd /opt/llama-builds && md5sum $NEW.tar.gz && tar xzf $NEW.tar.gz && rm $NEW.tar.gz \
  && md5sum $NEW/bin/llama-server $NEW/bin/test-backend-ops"

# part (2) on the rig, card 0: rig_suite.sh with this repository's scripts it calls copied to $R/repo
# (scripts/replay_checks.sh, docs/data/2026-11-03_upstream_rebase/run/check.sh, docs/data/2026-11-04_moe_skip/run/gpu_checks.sh)
ssh $RIG "R=$R REPO=$R/repo SETS=<folder with v4kern/ and ro8kt/> NEW=/opt/llama-builds/$NEW REF=/opt/llama-builds/$REF \
  STEPS='v4kern ro8kt archs off' setsid -f nohup timeout 14400 bash $R/rig_suite.sh > $R/out/console.txt 2>&1 < /dev/null"

# parts (3) and (4), from the repository root on a machine that only runs ssh, started when part (2) wrote $R/out/done
RIG=$RIG NEW=$NEW REF=$REF \
  FN=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  FLINT_Q5=$MODELS/flint-pascal-Q5.gguf FLINT_Q8=$MODELS/flint-pascal-Q8.gguf \
  ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf RIGDIR=<rig folder> OUT=<local folder> \
  LABEL_NEW=cand2 LABEL_REF=v5 PROFILE=1 GREEDY_BUILDS=$NEW \
  GREEDY_EXPECT='flashnext=4384ebc02147be1ee6f38d1ce4a43f21 flintq8=c5b2d8f20654728edfe38c37458b4c5c flintq5=d72ca1b968a8c1eb2b1ebeea22458ff9 ornith=35f987c059c7d2f521404e964995cd2d' \
  timeout 14400 bash docs/data/2026-11-17_v5_cand_p106_check/run/cand_load.sh

# the tables (standard library only), on a folder that names each model's loads load-cand and load-ref
# (links to load-cand2 and load-v5) next to its greedy/
python3 -I docs/data/2026-11-13_c831_upstream/run/r153_load_tables.py <links> docs/data/2026-11-18_c854b_p106_check/
python3 -I docs/data/2026-11-17_v5_cand_p106_check/run/cand_tables.py <rig_suite.sh's out/> <links> docs/data/2026-11-18_c854b_p106_check/ cand ref
```

`GREEDY_EXPECT` holds v4.0's texts of R153 part (3) (the same `rig_greedy.sh` command, files and split), so the greedy
step runs the candidate only and would have run the reference too for a file whose text differed; none did.
`r153_load_tables.py` also writes `greedy.tsv`, which `greedy-vs-v4.tsv` replaces here.

Part (5), from the repository root on the machine that drives the rig (here also the client), with the rig clean (no
llama process, 0 MiB on every card):

```bash
# the run (~10 min load, 12 tasks), then its report; CLIENT_OUT was the client folder R160's and R166's runs used
RIG=$RIG CLIENT=local MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf BIN=/opt/llama-builds/$NEW/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4c2 \
  bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh
RIG=$RIG CLIENT=local MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> CLIENT_OUT=<the same> \
  OUT=<out>/agent STEPS="report collect" bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh

# the tables, on a folder holding R166b's table rows plus o4c2's and the three runs' server and host logs
# (run/ornith_tables.sh lists them)
bash docs/data/2026-11-18_c854b_p106_check/run/ornith_tables.sh <tables folder>
```

`run/ornith_agent.sh` is the harness call the run and its report used, with the machine-specific values taken from
the environment.
