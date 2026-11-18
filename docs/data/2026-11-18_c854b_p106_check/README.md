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

**Part (6), Flash-Next's agent run (14:55–15:33):** the speed is v4.0's. A task at equal work took 2.40 minutes
against 2.43 and 2.37, and generation ran at 27.00 tok/s against 26.67 and 27.61. The first turns read the same
prompts in 82.28 s against 81.84 and 85.11 s, and one of the two v4.0 runs ran on this rig. 11 of 12 tasks passed:
JavaScript `complex-numbers` ran to the 12-minute cap, a path this model has taken on that task before. The server
read a previous reply again on 7 turns, against 2 and 1 in v4.0's runs. That cost little here, but its cause is not
known yet.

**Part (7), the same-rig reference on Ornith (15:55–16:32):** `pascal-p106-v5`'s own build, with part (5)'s launcher
on the same rig, read the 21 matched requests in 216.38 s; the candidate took 216.37 s. So part (5)'s 2.5 % was the
rig, not the engine. Generation and the first turns are within the spread of v4.0's two runs. Leaving out each
session's first request, which reads a whole prompt on cards that have just loaded, the candidate took 0.7 % longer on
the other 20, more than v4.0's two runs differ there. One run of each build cannot settle a gap that small.

**Part (8), the candidate's second Ornith run (16:57–17:28):** the same launcher on the candidate again, on the same
rig. On part (7)'s 20 requests with a cached prefix it took 187.15 s, against 187.20 s on `pascal-p106-v5`'s build,
so part (7)'s 0.7 % did not come back. The candidate's own two runs differ by 0.7 % there (188.55 and 187.15 s). 12
of 12 tasks passed.

**Part (9), the second run of `pascal-p106-v5`'s build and the verdict (17:50–18:19): negligible, so both models
support moving `pascal-p106-v5` to the candidate.** On part (7)'s 20 warm requests `1f1110c67` took 187.20 and 187.26 s,
the candidate 188.55 and 187.15 s: the means are 0.33 % apart, inside the candidate's own 0.74 % between its runs. The
21 requests took 215.74 s on `1f1110c67` and 215.70 s on the candidate, as means of two. Generation at the same mix of
contexts is 16.35 and 16.39 tok/s against 16.38 and 16.35. All four runs passed 12 of 12.

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
  (`1f1110c67`, part (2)'s reference) on this rig, next to this run, tells them apart: [part (7)](#part-7-the-same-rig-reference-on-ornith-2026-11-18).
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

## Part (6): the agent check on Flash-Next (2026-11-18)

**The speed is v4.0's, 11 of 12 tasks passed, and the server read a previous reply again on more turns.** A task at
equal work took 2.40 minutes against v4.0's 2.43 and 2.37. Generation ran at 27.00 tok/s against 26.67 and 27.61.
The 11 first turns that read the same prompt in all three runs took 82.28 s against 81.84 and 85.11 s. All three are
**negligible** by the rule of the earlier agent runs. One of the two v4.0 runs ran on this rig. JavaScript
`complex-numbers` ran to the 12-minute cap after 34 turns; this model has taken that long path on that task before.
The server read a previous reply again as prompt on 7 turns, against 2 and 1 in v4.0's runs. That cost 51.6 of the
run's 413.9 prompt seconds, against 31.0 of 394.2 and 6.9 of 280.6. Why the prompts stopped matching inside those
replies is not known yet.

The run is R156's agent check ([R156](../2026-11-13_r156_agent_v4r/README.md),
[bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and JavaScript alternating, a
12-minute cap a task, a 40-minute budget) on the maker's Swift GSQ-RCO Q2_0
(`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram table in the third shard) with the MTP draft
(`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens, one arm, one load: the harness's arm `v4r`, **the released
v4.0 preset ([release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)) with the candidate's
`llama-server`** (md5 `86607661d5a19d1a90d19ee4db10c90b`). The tables call it `v4rc2`, to keep it apart from R156a's
`v4r`. Its started command (1,689 bytes, md5 `4e11746b52bcc136ef186ffe9c161aa0`) is R156b's byte for byte apart from
the build folder's name, and the server's `switches:` line names R156's 39 switches with the same values. It ran on
the same rig as parts (2) to (5), with the machine that drives the rig as the agent's client. The server started at
14:54:57 and was up after 311 s (R156b on this rig: 315 s), and the tasks ran 15:00:11–15:32:58. Every card was at
stock before and after (75 W, 1,911 / 4,004 MHz maximum; the card log's start line reads core and memory offsets of 0
on all nine), with no Xid.

The references are R156's two runs of the same preset on v4.0 (`eb82916d3`), 2026-11-13: **R156a `v4r` on the second
rig and R156b `v4r2` on this rig**. Both had the 7-GPU quant machine's CPU as the agent's client; this run's client is
the machine that drives the rig. The client changes the time outside the server, not the server's own prompt and
decode times. The proxy saw the first token a median 0.07 s after the server's prefill (p90 0.17 s), against 0.06 s
(p90 0.16 s) in R156.

| | v4.0 `v4r` (R156a, second rig) | v4.0 `v4r2` (R156b, this rig) | **candidate `v4rc2` (this rig)** | class by the rule (against the mean; the spread of the two v4.0 runs) |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **11 / 12** | `complex-numbers` at the cap |
| minutes a task at equal work (R90 `v3`'s work) | 2.43 | 2.37 | **2.40** | negligible (0.0 %; 0.06) |
| the same without `complex-numbers` | 2.46 | 2.46 | **2.44** | −0.8 %; the two v4.0 runs agree to 0.00 |
| the 11 first turns read on the same prompt in all three runs, prompt s | 81.84 | 85.11 | **82.28** | negligible (−1.4 %; 3.27) |
| prompt tok/s, the whole session | 133.9 | 144.6 | **139.2** | negligible (−0.0 %; 10.7); a different mix of prompt sizes |
| the same without `complex-numbers` | 140.7 | 140.8 | **146.1** | +3.8 %; agree to 0.1 |
| generation tok/s | 26.67 | 27.61 | **27.00** | negligible (−0.5 %; 0.94) |
| the same without `complex-numbers` | 27.59 | 27.58 | **27.99** | +1.5 %; agree to 0.01 |
| first token median s | 2.42 | 2.48 | **2.52** | loss (+2.9 %; 0.06); follows the turn mix |
| first token p90 s | 9.00 | 8.65 | **9.54** | loss (+8.1 %; 0.35); follows the turn mix |
| first token on a task's first turn, median s | 6.68 | 6.65 | **6.69** | negligible (+0.4 %; 0.03) |
| prompt tokens read a task | 4,400 | 3,381 | **4,802** | |
| the same without `complex-numbers` | 3,481 | 3,268 | **3,516** | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 4,863 (2; 31.0 of 394.2) | 761 (1; 6.9 of 280.6) | **4,854 (7; 51.6 of 413.9)** | |
| draft acceptance (accepted of drafted tokens) | 0.749 | 0.759 | **0.740** | |
| minutes, all 12 tasks; turns | 33.4; 95 | 30.6; 69 | **32.6; 92** | different work |
| tool calls / failed / malformed | 90 / 3 / 0 | 65 / 5 / 0 | **90 / 3 / 0** | |
| checkpoints restored / erased; full re-processing | 12 / 1; 1 | 12 / 0; 0 | **17 / 2; 1** | |
| graph reallocations | 0 in 95 requests | 0 in 68 | **0 in 91** | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,886.8 | 1,885.3 | **1,878.3** | |
| lowest MemAvailable MiB; swap highest MiB | 312; 21 | 419; 404 | **387; 464** | |

Each value is one run ([flash_agent.tsv](flash_agent.tsv), [flash_agent_nocx.tsv](flash_agent_nocx.tsv),
[flash_same_prompts.md](flash_same_prompts.md), [flash_work.md](flash_work.md),
[flash_work_nocx.tsv](flash_work_nocx.tsv), [flash_clocks.tsv](flash_clocks.tsv), [flash_host.tsv](flash_host.tsv),
[flash_gonogo.md](flash_gonogo.md); R156's session figures from [its folder](../2026-11-13_r156_agent_v4r/agent.tsv)).
The class is R156's and R166's rule: a gain or a loss when the difference from the two v4.0 runs' mean is larger than
their spread (the gap between them), negligible otherwise. Where the two v4.0 runs agree to 0.00 or 0.01, that gap
measures nothing, so those rows get no class. One task that takes a different path moves those figures by more. Minutes a task at equal work price R90 `v3`'s work (v3.0
on this rig, 2026-11-11: 3.08) at each run's own prompt and generation speeds. The table's R90 row is R156's
published row, figure for figure.

- **The task at the cap.** `complex-numbers` ran 34 turns and 34 tool calls (2 failed) and hit the 12-minute cap with
  its tests failing. On the v4 line this task has taken the long path before: in R156's client folder `v4c2` ran it to
  the cap in 35 turns, `v4c32` passed it in 37 turns (10.8 minutes) and R156a in 29 (9.7 minutes), and v3.0's `v3b`
  failed it at the cap. The other 11 tasks passed. JavaScript `variable-length-quantity` took 6.8 minutes in 6 turns,
  writing long replies; R156a took 9.6 minutes on it and R156b hit the cap there.
- **Prompts on the same work: negligible.** 11 requests read the same number of new tokens on the same cached prefix
  in all three runs: every task's first turn except `pig-latin`'s ([flash_same_prompts.md](flash_same_prompts.md)).
  They took 82.28 s here, against 81.84 s on the second rig and 85.11 s on this rig; R156b's first turn of JavaScript
  `variable-length-quantity` took 5.39 s against 3.66 and 3.69 s. `pig-latin`'s first turn is not in the set. Here
  and in R156a it read its whole 2,324-token prompt (12.34 and 12.69 s), because the long `complex-numbers` task
  before it had pushed the shared system prefix's checkpoint out of the four kept on the cards. The server's log
  says "no checkpoint below 1553". R156b read 775 tokens on the cached prefix.
- **R156's per-request tables are rebuilt from its server logs** ([run/server_turns.py](run/server_turns.py),
  [agent/r156a_v4r/](agent/r156a_v4r/), [agent/r156b_v4r2/](agent/r156b_v4r2/)). The client's tables of those runs
  are not on the machine that drove this one. The server log holds every request in order, and the client log gives
  each task's turns. On R90's and this run's logs the rebuild gives every request's prompt and decode tokens and
  milliseconds exactly as the client's tables, and the cached count within one token: the release line counts the
  last generated token on some requests only. On R156 it gives the published prompt tokens a task (4,400 and 3,381)
  and generation (26.67 and 27.61 tok/s). The matched-prompt table therefore matches cached counts within one token
  (`CACHE_TOL=1`).
- **Generation: negligible**, 27.00 against 26.67 and 27.61 tok/s, and 27.99 against 27.59 and 27.58 without
  `complex-numbers`. The draft's acceptance is 0.740 against 0.749 and 0.759.
- **First token follows the turn mix.** The turns after the first read a median of 267 new tokens, against 215 and
  216.5, and their server prefill took a median 2.23 s, against 2.16 and 2.08 s. A task's first turn is the same
  request in every run, and it took 6.69 s against 6.68 and 6.65 s.
- **The work, and the reply re-reads.** The agent read 4,802 prompt tokens a task, against 4,400 and 3,381; without
  `complex-numbers`, 3,516 against 3,481 and 3,268 ([flash_work.md](flash_work.md),
  [flash_work_nocx.tsv](flash_work_nocx.tsv)). On 7 turns the server read the previous reply again:
  - `affine-cipher` turn 2 (a reply of 71 tokens);
  - `variable-length-quantity` (Python) turn 3 (800 tokens);
  - `grade-school` turn 4 (150 tokens);
  - `complex-numbers` turns 5, 11 and 21 (1,436, 455 and 322 tokens);
  - `pig-latin` turn 3 (1,620 tokens).

  R156a re-read on 2 turns (`complex-numbers` turn 26 and `pig-latin` turn 3), and R156b on 1 (`pig-latin` turn 3).
  The earlier published Flash-Next runs re-read on 0 to 2 turns each (2026-11-03's `mtp`, `mtp2`, `nodraft`,
  baseline and pipelined runs, and R90's two arms). On each of the 7, the new prompt stopped matching the cached
  tokens inside the previous reply. The server's logged prefix match (`f_keep`, rounded to a thousandth) puts that
  point 23, 362, 132, 496, 92, 209 and 189 tokens into the replies listed above, in that order: inside the reasoning
  on some, inside the tool call on others. On this hybrid model the server can only go back to a checkpoint, so it
  restored the one at the previous prompt's end (4 tokens short) and read the whole reply again.
  The 7 turns re-read 4,854 tokens in 51.6 s; R156a's 2 turns re-read 4,863 tokens in 31.0 s. `pig-latin` turn 3
  re-reads in every run, and 3 of the 7 are in `complex-numbers`' long path. The other 3 are turns that re-read in
  neither R156 run. Three things could cause it: the candidate's chat handling rendering a reply back into the
  history differently from the tokens the model wrote, this run's client, or the replies themselves. Which one is
  open: C884 (Q1598) renders this run's recorded requests with both builds and compares the tokens at each of those
  points.
- **Clocks** ([flash_clocks.tsv](flash_clocks.tsv)): this rig's busy medians 1,847 1,847 1,885 1,898 1,860 1,911 1,873
  1,911 1,873 MHz (cards 0–8), against 1,860 1,847 1,898 1,898 1,873 1,911 1,885 1,911 1,885 in R156b on the same
  rig; 1,878.3 against 1,885.3 on the mean (−0.4 %).
- **Host** ([flash_host.tsv](flash_host.tsv)): lowest MemAvailable 387 MiB at 15:03:04, during the first task (419
  and 312 MiB), and 3,054 MiB at the end. Swap went from 30 MiB at the start to 464 MiB at its highest, at 15:08:13
  (36 → 404 and 21 → 21 MiB). The RAM watchdog stopped nothing. This run's card log is `scripts/card_log.py`'s, which
  counts 75 MiB on each idle card that `nvidia-smi` (R156's card log) reads as 0 MiB. Less those 75 MiB, its card
  peaks (5,117 5,161 5,019 5,175 4,275 5,035 4,429 5,047 5,625 MiB) are R156a's to 2 MiB. The report's memory gate
  ([flash_gonogo.md](flash_gonogo.md)) is NO-GO on the 387 MB low and the 464 MB of swap, as R156's was (276 MB and
  713 MB in its shared folder).
- **Graph reallocations:** 0 in 91 requests, in every prompt-size bucket ([flash_reallocs.tsv](flash_reallocs.tsv)).

`pascal-p106-v5` stays `1f1110c67`.

## Part (7): the same-rig reference on Ornith (2026-11-18)

**On the same rig the candidate reads Ornith's prompts as fast as `pascal-p106-v5`'s own build, so part (5)'s 2.5 %
was the rig.** Part (5)'s launcher with `pascal-p106-v5`'s build `1f1110c67` read the 21 matched requests in
216.38 s; the candidate read them in 216.37 s. Generation (16.30 tok/s on the candidate against 16.52) and the first
turns are within the spread of v4.0's two runs. One small difference remains. A session's first request reads its
whole prompt on cards that have just loaded, and it varies more than the band on its own. Without it, the candidate
took 188.55 s on the other 20 against 187.20 s (+0.7 %), more than the 0.15 % between v4.0's two back-to-back runs on
the same 20. With one run of each build, that is at the edge of what a run can show.

The run (R206, Q1597) is part (5)'s, arm **`o4v5a`**: [run/ornith_agent.sh](run/ornith_agent.sh) with `ARM=o4v5a` and
`BIN` = `pascal-p106-v5`'s build `1f1110c67` (`llama-server` md5 `5b134f43f3a3e885bfba02538ffebcbe`, part (2)'s
reference). The started command (521 bytes, md5 `8c07491e57c50b7c1eb8f1d5c35cd646`) is `o4c2`'s byte for byte apart
from the build folder's name. The server's `switches:` line names the same six settings (this build reads 888
`LLAMA_` and `GGML_` names, the candidate 889). It ran on the same rig 1 h 15 min after part (5)'s run, with part
(6)'s Flash-Next run in between: the server started at 15:55:02 and was up after 683 s (part (5): 683 s), the tasks
ran 16:06:27–16:31:32. Every card at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no Xid; the card log
reads offset 0 on every card). From 16:01 to 16:23 the client machine also ran the client of another rig's agent
check. The prompt and generation times are the rig's server's own, so the matched requests are not affected; the
minutes and the first-token times, which the client measures, shared that machine from the first task to 16:23.

| | **`1f1110c67` `o4v5a` (this rig)** | candidate `o4c2` (this rig, part (5)) | v4.0 `o4sna` (second rig) | v4.0 `o4snb` (second rig) | the candidate against `1f1110c67` by the rule (the spread: `o4snb` against `o4sna`) |
|---|---|---|---|---|---|
| passed | **12 / 12** | 12 / 12 | 12 / 12 | 12 / 12 | |
| the 21 requests read on the same prompt in all four runs, prompt s | **216.38** | 216.37 | 211.35 | 210.86 | negligible (0.0 %; 0.49) |
| of them, the 12 first turns | **119.59** | 119.20 | 115.83 | 115.29 | negligible (−0.3 %; 0.54) |
| of them, the 20 with a cached prefix (all but a session's first request) | **187.20** | 188.55 | 184.62 | 184.35 | loss (+0.7 %; 0.27) |
| a session's first request (Python `affine-cipher`, 2,742 tokens, nothing cached), prompt s | **29.19** | 27.82 | 26.73 | 26.51 | |
| of the 21, read slower / faster than the reference column; median of the ratios | | 15 / 6; +0.21 % | | 12 / 9 against `o4sna`; +0.10 % | |
| minutes a task at equal work (`ka`'s work) | **2.13** | 2.17 | 2.11 | 2.13 | loss (+1.9 %; 0.02); each session's prompt mix |
| prompt tok/s, the whole session | **99.3** | 96.2 | 103.0 | 105.9 | a different mix of prompt sizes |
| generation tok/s | **16.52** | 16.30 | 16.52 | 16.13 | negligible (−1.3 %; 0.39) |
| first token median s | **6.56** | 7.56 | 6.69 | 6.06 | follows the turn mix |
| first token p90 s | **11.10** | 11.11 | 11.29 | 11.32 | negligible (+0.1 %; 0.03) |
| first token on a task's first turn, median s | **8.63** | 8.58 | 8.43 | 8.45 | gain (−0.6 %; 0.02) |
| prompt tokens read a task | **3,114** | 3,259 | 3,405 | 4,130 | |
| reply tokens re-read (turns; their prompt s of all prompt s) | **1,996 (1; 12.1 of 376.1)** | 2,143 (1; 13.2 of 406.6) | 4,831 (2; 30.6 of 396.6) | 12,332 (3; 71.9 of 467.8) | |
| tokens generated a task | **1,501** | 1,689 | 1,818 | 2,300 | |
| minutes, all 12 tasks; turns | **24.9; 61** | 28.0; 63 | 29.1; 63 | 36.8; 73 | different work |
| tool calls / failed / malformed | **61 / 2 / 0** | 63 / 0 / 0 | 63 / 2 / 0 | 73 / 2 / 0 | |
| checkpoints restored / created / erased; full re-processing | **12 / 0 / 0; 0** | 12 / 0 / 0; 0 | 13 / 0 / 0; 0 | 14 / 0 / 0; 0 | |
| graph reallocations | **0 in 61 requests** | 0 in 63 | 0 in 63 | 0 in 73 | |
| busy SM clock, mean of the nine cards' medians, MHz | **1,878.3** | 1,876.9 | 1,888.2 | 1,888.2 | |
| lowest MemAvailable MiB | **1,228** | 1,199 | 1,225 | 1,228 | |

Each value is one run ([r206_agent.tsv](r206_agent.tsv), [r206_same_prompts4.md](r206_same_prompts4.md),
[r206_paired4.md](r206_paired4.md), [r206_paired_r166.md](r206_paired_r166.md), [r206_work.md](r206_work.md),
[r206_clocks.tsv](r206_clocks.tsv), [r206_host.tsv](r206_host.tsv), [r206_reallocs.tsv](r206_reallocs.tsv),
[o4v5a_gonogo.md](o4v5a_gonogo.md); the other runs' rows as in part (5)). The class is part (5)'s rule with
`1f1110c67`'s run as the reference: a gain or a loss when the candidate's difference from it is larger than the gap
between v4.0's two back-to-back runs, negligible otherwise.

- **The rig, not the engine.** `1f1110c67` on this rig takes 2.5 % longer on the 21 requests than v4.0 on the second
  rig (216.38 against 211.35 and 210.86 s), as the candidate does (216.37 s). With every switch off `1f1110c67`
  launches v4.0's kernels in every evaluation ([the 2026-11-17 check](../2026-11-17_v5_cand_p106_check/README.md)),
  so the gap is the first rig's and the day's, not an engine's. Its cards
  ran as in part (5): busy medians 1,847 1,847 1,885 1,898 1,860 1,911 1,873 1,911 1,873 MHz (cards 0–8), the
  candidate's run's on eight cards and 1,898 against 1,885 MHz on card 3 ([r206_clocks.tsv](r206_clocks.tsv)).
- **A session's first request varies more than the band.** The 21 include each session's first request (Python
  `affine-cipher`, turn 1: 2,742 tokens, nothing in the cache, the first request after the load). It took 29.19 s
  here, 27.82 s on the candidate and 26.73 and 26.51 s on v4.0, a spread larger than the 0.49 s band of the 21
  together, so it decides their total. On the other 20, the candidate took 188.55 s against 187.20 s (+0.7 %),
  slower on 15 of them, the median request by 0.22 %. v4.0's two runs differ by 0.15 % there (184.62 and 184.35 s;
  12 of 20 slower, median +0.10 %; [r206_paired_r166.md](r206_paired_r166.md)). By the rule that is a loss, est. 0.7 %
  on prompts that start from a cached prefix, but it rests on one run of each build. A second run of each, in
  alternating order on this rig, tells (Q1599).
- **The two runs on this rig alone** match on 25 requests (the 21 and four later turns): 252.01 s on `1f1110c67`
  against 252.03 s on the candidate, the 24 with a cached prefix 222.82 against 224.22 s (+0.6 %;
  [r206_paired.md](r206_paired.md), [r206_same_prompts.md](r206_same_prompts.md)).
- **Generation: negligible,** 16.30 tok/s on the candidate against 16.52 (−1.3 %; v4.0's two runs differ by 0.39
  tok/s, 2.4 %). The candidate's run generated more tokens a task (1,689 against 1,501), and longer replies decode at
  deeper contexts.
- **Minutes a task at equal work** price `ka`'s work at each session's own prompt and generation speeds, and the
  session's prompt tok/s counts its mix of prompt sizes (96.2 against 99.3, while the matched requests take the same
  time). The +1.9 % follows that mix and the generation speed; it is a loss by the rule, as in part (5).
- **The work:** 3,114 prompt tokens read a task, one reply read again (Python `list-ops`, turn 3, 1,996 tokens,
  12.1 s, the turn the candidate's run and `o4snb` also re-read), 12 checkpoint restores, none created or erased, no
  full re-processing; the report's cache hit is 85.7 % (the candidate's 85.8 %;
  [o4v5a_report_table.md](o4v5a_report_table.md)). The candidate's slot scheduler reads no more than its own base.
  Two tool calls in `affine-cipher` returned an error to the agent (the report's "failed"); none were malformed.
- **Host** ([r206_host.tsv](r206_host.tsv)): lowest MemAvailable 1,228 MiB at 16:31:38, during the last task, and
  3,177 MiB at the end; swap 30 MiB at the start and 341 MiB at its highest at 16:02:51, during the load (the
  candidate's: 30 and 341). The RAM watchdog stopped nothing. The card peaks in the NVML card log (75 MiB over
  `nvidia-smi`'s on an idle card) are 5,808 5,828 5,754 5,828 5,436 5,546 5,706 5,782 5,188 MiB, and the candidate's
  run's 5,830 5,846 5,776 5,846 5,454 5,562 5,724 5,796 5,202: 22 18 22 18 18 16 18 14 14 MiB more, on every card.
  At load the two builds take the same MiB on every card (part (3)). The report's memory gate
  ([o4v5a_gonogo.md](o4v5a_gonogo.md)) reads the client's log shared with R160's and R166's runs: NO-GO on its 718 MB
  low (R160d's) and its 342 MB swap, as before.
- **Graph reallocations:** 0 in 61 requests, in every prompt-size bucket ([r206_reallocs.tsv](r206_reallocs.tsv)).

This run is also the first control run of Q1593 (the v4 preset on `1f1110c67`, two runs); part (5)'s `o4c2` is the
same preset on the candidate, so whichever tree Q1593 runs on has one control run already.

`pascal-p106-v5` stays `1f1110c67`.

## Part (8): the candidate's second run on Ornith (2026-11-18)

**On the same rig the candidate's second run reads part (7)'s 20 warm requests in the time `pascal-p106-v5`'s own
build took, so part (7)'s 0.7 % did not come back.** It took 187.15 s on them, against 187.20 s on `1f1110c67` and
188.55 s in the candidate's first run. The candidate's two runs differ by 0.7 % there, where v4.0's two back-to-back
runs differ by 0.15 %: the first run was the slow one. One run of `1f1110c67` stands against two of the candidate, so
the second run of `1f1110c67` (Q1599's other half, part (9)) gives the verdict.

The run (R207a, the first of Q1599's two) is part (5)'s, arm **`o4c3`**: [run/ornith_agent.sh](run/ornith_agent.sh)
with `ARM=o4c3` and `BIN` = the candidate's `llama-server` (md5 `86607661d5a19d1a90d19ee4db10c90b`, checked on the rig
before the run). The started command (521 bytes, md5 `056afa770d54ddc800cf930eee9a723e`) is part (5)'s byte for
byte, and the server's `switches:` line names the same six settings (889 names read). It ran on the same rig 14 min
after part (7)'s run ended, with nothing between them: the server started at 16:46:27 and was up after 681 s (parts
(5) and (7): 683 s), the tasks ran 16:57:49–17:27:33. Every card was at stock before and after (75 W, 1,911 / 4,004 MHz
maximum, no Xid; the card log reads offset 0 on every card). From 16:42 to 17:12:42 the client machine also ran the
client of another rig's agent check. The prompt and generation times are the rig's server's own, so the matched
requests are not affected. The minutes and the first-token times, which the client measures, shared that machine
until 17:12:42.

| | **candidate `o4c3` (this run)** | candidate `o4c2` (part (5)) | `1f1110c67` `o4v5a` (part (7)) | v4.0 `o4sna` (second rig) | v4.0 `o4snb` (second rig) |
|---|---|---|---|---|---|
| passed | **12 / 12** | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| part (7)'s 21 requests, prompt s | **215.02** | 216.37 | 216.38 | 211.35 | 210.86 |
| of them, the 12 first turns | **118.02** | 119.20 | 119.59 | 115.83 | 115.29 |
| of them, the 20 with a cached prefix (all but a session's first request) | **187.15** | 188.55 | 187.20 | 184.62 | 184.35 |
| a session's first request (Python `affine-cipher`, 2,742 tokens, nothing cached), prompt s | **27.87** | 27.82 | 29.19 | 26.73 | 26.51 |
| the 19 with a cached prefix read on the same prompt in all five runs, prompt s | **177.21** | 178.46 | 177.24 | 174.91 | 174.63 |
| the 22 with a cached prefix, the cached tokens within one, prompt s | **210.25** | 211.69 | 210.29 | 207.21 | 206.96 |
| minutes a task at equal work (`ka`'s work) | **2.21** | 2.17 | 2.13 | 2.11 | 2.13 |
| prompt tok/s, the whole session | **92.0** | 96.2 | 99.3 | 103.0 | 105.9 |
| generation tok/s | **16.19** | 16.30 | 16.52 | 16.52 | 16.13 |
| first token median s | **5.03** | 7.56 | 6.56 | 6.69 | 6.06 |
| first token p90 s | **10.34** | 11.11 | 11.10 | 11.29 | 11.32 |
| first token on a task's first turn, median s | **8.55** | 8.58 | 8.63 | 8.43 | 8.45 |
| prompt tokens read a task | **3,146** | 3,259 | 3,114 | 3,405 | 4,130 |
| reply tokens re-read (turns; their prompt s of all prompt s) | **1,371 (2; 17.4 of 410.2)** | 2,143 (1; 13.2 of 406.6) | 1,996 (1; 12.1 of 376.1) | 4,831 (2; 30.6 of 396.6) | 12,332 (3; 71.9 of 467.8) |
| tokens generated a task | **1,801** | 1,689 | 1,501 | 1,818 | 2,300 |
| minutes, all 12 tasks; turns | **29.5; 72** | 28.0; 63 | 24.9; 61 | 29.1; 63 | 36.8; 73 |
| tool calls / failed / malformed | **72 / 1 / 0** | 63 / 0 / 0 | 61 / 2 / 0 | 63 / 2 / 0 | 73 / 2 / 0 |
| checkpoints restored / created / erased; full re-processing | **13 / 0 / 0; 0** | 12 / 0 / 0; 0 | 12 / 0 / 0; 0 | 13 / 0 / 0; 0 | 14 / 0 / 0; 0 |
| graph reallocations | **0 in 72 requests** | 0 in 63 | 0 in 61 | 0 in 63 | 0 in 73 |
| busy SM clock, mean of the nine cards' medians, MHz | **1,878.3** | 1,876.9 | 1,878.3 | 1,888.2 | 1,888.2 |
| lowest MemAvailable MiB | **1,224** | 1,199 | 1,228 | 1,225 | 1,228 |

Each value is one run ([r207_agent.tsv](r207_agent.tsv), [r207_r206set.md](r207_r206set.md),
[r207_paired5.md](r207_paired5.md), [r207_paired5_tol1.md](r207_paired5_tol1.md), [r207_work.md](r207_work.md),
[r207_clocks.tsv](r207_clocks.tsv), [r207_host.tsv](r207_host.tsv), [r207_reallocs.tsv](r207_reallocs.tsv),
[o4c3_gonogo.md](o4c3_gonogo.md); the other runs' rows as in parts (5) and (7)).

- **Part (7)'s 21 requests.** This run read 20 of them on the same prompt as the other four runs. The 21st, Python
  `affine-cipher`'s turn 2, read the same 1,185 new tokens with one token more from the cache (2,811 against 2,810), so
  [r207_r206set.md](r207_r206set.md) takes this run's time for it from the table that matches the cached tokens
  within one ([r207_paired5_tol1.md](r207_paired5_tol1.md)). On the 20 with a cached prefix the candidate took 187.15 s
  against 187.20 s on `1f1110c67` (−0.03 %), and 188.55 s in its first run (this run −0.74 % against it).
- **The same on the sets matched in all five runs.** On the 19 with a cached prefix that read the same prompt exactly
  in all five runs, this run took 177.21 s against 177.24 s on `1f1110c67`: 10 requests slower, 9 faster, the median
  request +0.02 % ([r207_paired5.md](r207_paired5.md)). The candidate's first run took 178.46 s there (14 slower,
  5 faster, median +0.21 %; [r207_paired5_o4c2.md](r207_paired5_o4c2.md)), and v4.0's two runs 174.91 and 174.63 s
  (0.16 % apart; [r207_paired5_r166.md](r207_paired5_r166.md)). With the cached tokens matched within one, the 22
  with a cached prefix took 210.25 s against 210.29 s (11 slower, 11 faster), the candidate's first run 211.69 s
  ([r207_paired5_tol1.md](r207_paired5_tol1.md), [r207_paired5_tol1_o4c2.md](r207_paired5_tol1_o4c2.md)). The
  candidate's two runs differ by 0.7 % on each of these sets ([r207_paired5_cand.md](r207_paired5_cand.md),
  [r207_paired5_tol1_cand.md](r207_paired5_tol1_cand.md)), more than v4.0's two do.
- **The three runs on this rig alone** match on 22 requests: 229.52 s on `1f1110c67`, 229.41 s and 228.18 s on the
  candidate's two runs; the 21 with a cached prefix 200.34, 201.60 and 200.32 s ([r207_paired.md](r207_paired.md),
  [r207_paired_o4c2.md](r207_paired_o4c2.md), [r207_same_prompts.md](r207_same_prompts.md)).
- **The first turns** took 118.02 s against 119.59 s on `1f1110c67` (−1.3 %; 8 of 12 faster) and 119.20 s in the
  candidate's first run. A session's first request took 27.87 s (27.82 s in the candidate's first run, 29.19 s on
  `1f1110c67`).
- **Generation:** 16.19 tok/s, against 16.30 in the candidate's first run and 16.52 on `1f1110c67`. v4.0's two runs
  differ by 0.39 tok/s (16.52 and 16.13). This run generated the most tokens a task of the three runs on this rig
  (1,801 against 1,689 and 1,501).
- **Minutes a task at equal work** (2.21 against 2.17 and 2.13) price `ka`'s work at each session's prompt and
  generation speeds, and the session's prompt tok/s counts its mix of prompt sizes (92.0: 72 turns, more of them short
  later turns), while the matched requests take the same time. The first-token median (5.03 s) follows the same mix;
  on a task's first turn it is 8.55 s against 8.58 and 8.63 s.
- **The work:** 3,146 prompt tokens read a task. Two replies were read again (Python `pig-latin` turn 4, 996 tokens,
  9.60 s; Python `variable-length-quantity` turn 6, 375 tokens, 7.79 s). The server restored 13 checkpoints, created or
  erased none and re-processed nothing in full; the report's cache hit is 86.8 %
  ([o4c3_report_table.md](o4c3_report_table.md)). One tool call in `affine-cipher` returned an error to the agent;
  none were malformed. Graph reallocations: 0 in 72 requests ([r207_reallocs.tsv](r207_reallocs.tsv)).
- **Clocks:** busy medians 1,847 1,847 1,885 1,898 1,860 1,911 1,873 1,911 1,873 MHz on cards 0–8, part (7)'s card
  for card ([r207_clocks.tsv](r207_clocks.tsv)).
- **Host** ([r207_host.tsv](r207_host.tsv)): lowest MemAvailable 1,224 MiB at 17:27:38, as the server stopped after
  the last task, and 3,181 MiB at 17:27:58; swap 31 MiB at the start and 342 MiB at its highest at 16:55:02, during
  the load. The RAM watchdog stopped nothing. The card peaks in the NVML card log are 5,830 5,846 5,776 5,846 5,454
  5,562 5,724 5,796 5,202 MiB: the candidate's first run's to the MiB on every card, so its 22 18 22 18 18 16 18 14
  14 MiB above `1f1110c67`'s peaks on cards 0–8 (part (7)) come back. The report's memory gate ([o4c3_gonogo.md](o4c3_gonogo.md)) reads the
  client's log shared with R160's and R166's runs: NO-GO on its 718 MB low and its 342 MB swap, as before.

`pascal-p106-v5` stays `1f1110c67`.

## Part (9): `pascal-p106-v5`'s second run on Ornith, and the verdict (2026-11-18)

**Negligible. With two runs of each build on the same rig, the candidate reads Ornith's prompts and generates as fast
as `pascal-p106-v5`'s own build.** On part (7)'s 20 requests with a cached prefix, `1f1110c67`'s two runs took 187.20
and 187.26 s and the candidate's two 188.55 and 187.15 s. The two builds' means are 0.33 % apart, less than the
candidate's own two runs differ (0.74 %). Of the four runs only the candidate's first was slower, and it was the day's
first agent run on this rig. The three runs after it, in alternating order, took 187.20, 187.15 and 187.26 s. At the
same mix of contexts, generation is 16.35 and 16.39 tok/s on `1f1110c67` and 16.38 and 16.35 on the candidate.

The run (R207b, the second of Q1599's two) is part (7)'s, arm **`o4v5b`**: [run/ornith_agent.sh](run/ornith_agent.sh)
with `ARM=o4v5b` and `BIN` = `pascal-p106-v5`'s build `1f1110c67` (`llama-server` md5
`5b134f43f3a3e885bfba02538ffebcbe`, checked on the rig before the run). The started command (521 bytes, md5
`8c07491e57c50b7c1eb8f1d5c35cd646`) is part (7)'s byte for byte, and the server's `switches:` line names the same six
settings (888 names read). It ran on the same rig 12 min after part (8)'s run ended, with nothing between them: the
server started at 17:39:29 and was up after 682 s (parts (5), (7) and (8): 683, 683 and 681 s), and the tasks ran
17:50:54–18:19:11. Every card was at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no Xid; the card log
reads offset 0 on every card). From 17:37:39 to 18:03 the client machine also ran the client of another rig's agent
check. The prompt and generation times are the rig's server's own, so the matched requests are not affected. The
minutes and the first-token times, which the client measures, shared that machine until 18:03.

The four runs on this rig, in the order they ran: `o4c2` 14:11, `o4v5a` 16:06, `o4c3` 16:57, `o4v5b` 17:50 (part (6)'s
Flash-Next run 14:55–15:33 between the first two). The class compares the two builds' means against each build's own
two runs (the second against the first): a gain or a loss when the means differ by more than both builds' own runs do,
negligible otherwise.

| | `1f1110c67` `o4v5a` (part (7)) | **`1f1110c67` `o4v5b` (this run)** | candidate `o4c2` (part (5)) | candidate `o4c3` (part (8)) | the candidate's mean against `1f1110c67`'s; each build's own two runs | class |
|---|---|---|---|---|---|---|
| passed | 12 / 12 | **12 / 12** | 12 / 12 | 12 / 12 | | the same |
| part (7)'s 21 requests, prompt s | 216.38 | **215.10** | 216.37 | 215.02 | 215.70 against 215.74 s (−0.02 %); 0.59 % and 0.62 % | negligible |
| of them, the 12 first turns | 119.59 | **117.85** | 119.20 | 118.02 | 118.61 against 118.72 s (−0.09 %); 1.46 % and 0.99 % | negligible |
| of them, the 20 with a cached prefix (all but a session's first request) | 187.20 | **187.26** | 188.55 | 187.15 | 187.85 against 187.23 s (+0.33 %); 0.03 % and 0.74 % | negligible |
| the 19 with a cached prefix read on the same prompt in all four runs, prompt s | 177.24 | **177.28** | 178.46 | 177.21 | 177.84 against 177.26 s (+0.32 %); 0.02 % and 0.70 % | negligible |
| the 22 with a cached prefix, the cached tokens within one, prompt s | 206.33 | **206.46** | 207.74 | 206.49 | 207.12 against 206.40 s (+0.35 %); 0.06 % and 0.60 % | negligible |
| minutes a task at equal work (`ka`'s work) | 2.13 | **2.14** | 2.17 | 2.21 | 2.19 against 2.135 (+2.6 %); 0.5 % and 1.8 % | loss by the rule; each session's mix |
| generation tok/s, the whole session | 16.52 | **16.40** | 16.30 | 16.19 | 16.245 against 16.46 (−1.3 %); 0.7 % and 0.7 % | loss by the rule; the replies' depth |
| generation tok/s at the same mix of contexts | 16.35 | **16.39** | 16.38 | 16.35 | 16.365 against 16.37 (−0.03 %); 0.2 % and 0.2 % | negligible |
| first token on a task's first turn, median s | 8.63 | **8.49** | 8.58 | 8.55 | 8.565 against 8.56 s (+0.06 %); 1.6 % and 0.3 % | negligible |
| prompt tok/s, the whole session | 99.3 | **100.0** | 96.2 | 92.0 | | a different mix of prompt sizes |
| first token median s | 6.56 | **7.23** | 7.56 | 5.03 | | follows the turn mix |
| first token p90 s | 11.10 | **11.72** | 11.11 | 10.34 | | |
| prompt tokens read a task | 3,114 | **3,121** | 3,259 | 3,146 | | |
| tokens generated a task; of them past 8,000 tokens of context | 1,501; 619 | **1,738; 386** | 1,689; 4,320 | 1,801; 5,042 | | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 1,996 (1; 12.1 of 376.1) | **2,002 (2; 20.3 of 374.7)** | 2,143 (1; 13.2 of 406.6) | 1,371 (2; 17.4 of 410.2) | | |
| minutes, all 12 tasks; turns | 24.9; 61 | **28.1; 60** | 28.0; 63 | 29.5; 72 | | different work |
| tool calls / failed / malformed | 61 / 2 / 0 | **60 / 1 / 0** | 63 / 0 / 0 | 72 / 1 / 0 | | |
| checkpoints restored / created / erased; full re-processing | 12 / 0 / 0; 0 | **13 / 0 / 0; 0** | 12 / 0 / 0; 0 | 13 / 0 / 0; 0 | | |
| graph reallocations | 0 in 61 requests | **0 in 60** | 0 in 63 | 0 in 72 | | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,878.3 | **1,876.9** | 1,876.9 | 1,878.3 | | |
| lowest MemAvailable MiB | 1,228 | **1,182** | 1,199 | 1,224 | | |

Each value is one run ([r207b_agent.tsv](r207b_agent.tsv), [r207b_r206set.md](r207b_r206set.md),
[r207b_pairs.md](r207b_pairs.md), [r207b_pairs_tol1.md](r207b_pairs_tol1.md), [r207b_decode.md](r207b_decode.md),
[r207b_work.md](r207b_work.md), [r207b_clocks.tsv](r207b_clocks.tsv), [r207b_host.tsv](r207b_host.tsv),
[r207b_reallocs.tsv](r207b_reallocs.tsv), [o4v5b_gonogo.md](o4v5b_gonogo.md); the other runs' rows as in parts (5),
(7) and (8)). The means are of the two runs' values; a spread is the second run against the first.

- **Part (7)'s 21 requests.** This run read all 21 on the same prompt within one cached token, as part (8)'s run did
  ([r207b_r206set.md](r207b_r206set.md) takes `o4v5b`'s and `o4c3`'s times from the tables that match the cached
  tokens within one). On the 20 with a cached prefix `1f1110c67` took 187.26 s against its first run's 187.20 s
  (+0.03 %). Its two runs are 0.03 % apart and the candidate's 0.74 %.
- **Request by request** on the 20 requests read on the same prompt in all four runs (19 with a cached prefix;
  [r207b_pairs.md](r207b_pairs.md), each pair's rows in `r207b_paired_*.md`): the candidate's first run against
  `1f1110c67`'s two, 1.0068 and 1.0067 (14 and 10 of 19 slower); its second run 0.9998 and 0.9997 (10 and 8 of 19
  slower); `1f1110c67`'s second run against its first 1.0002 (11 slower, 8 faster); the candidate's second against its
  first 0.9930 (7 slower, 12 faster). With the cached tokens within one, on 22 with a cached prefix, the same: 1.0068
  and 1.0062, 1.0008 and 1.0001, 1.0006 and 0.9940 ([r207b_pairs_tol1.md](r207b_pairs_tol1.md)). Only the candidate's
  first run is off the other three, by the same 0.7 % against each.
- **Generation at the same context** ([r207b_decode.md](r207b_decode.md), from the published `turns.csv`:
  [run/decode_by_context.py](run/decode_by_context.py)). Each request is put in the bucket of its middle context (its
  start plus half the tokens it generated). Bucket by bucket the four runs decode at the same speed: at 4,500 to 6,000
  tokens 60.91 and 60.30 ms a token on `1f1110c67`, 60.51 and 60.52 on the candidate; at 6,000 to 8,000 tokens 62.34
  and 62.59, 61.88 and 62.39. The candidate's runs generated 4,320 and 5,042 tokens past 8,000 tokens of context, where
  every run decodes slowest (64.84 and 65.52 ms a token on
  `1f1110c67`, 66.08 and 66.18 on the candidate, whose requests there reach deeper), against 619 and 386 on `1f1110c67`. Weighted by the tokens all
  four runs generated in each bucket, the session speeds become 16.35, 16.39, 16.38 and 16.35 tok/s. So the session's
  −1.3 % is where the replies sat, not the engine.
- **Minutes a task at equal work** price `ka`'s work at each session's prompt tok/s and generation tok/s. Both are
  session averages: the candidate's sessions read more short later turns (prompt tok/s 96.2 and 92.0 against 99.3 and
  100.0, while the same prompts take the same time) and generated deeper in the context. The +2.6 % follows those two
  mixes; on the same work, the matched prompts and the generation at the same context, the builds are equal.
- **The first turns** took 117.85 s, against 119.59 s in `1f1110c67`'s first run and 119.20 and 118.02 s on the
  candidate. A session's first request took 27.83 s here, against 29.19 s in `1f1110c67`'s first run and 27.82 and 27.87 s
  on the candidate. The first token on a task's first
  turn: 8.49 s against 8.63, 8.58 and 8.55 s.
- **The work:** 3,121 prompt tokens read a task. Two replies were read again (Python `pig-latin` turn 4, 1,448 tokens,
  10.85 s; Python `scale-generator` turn 3, 554 tokens, 9.47 s). The server restored 13 checkpoints, created or erased
  none and re-processed nothing in full; the report's cache hit is 85.5 % ([o4v5b_report_table.md](o4v5b_report_table.md)).
  One tool call in `affine-cipher` returned an error to the agent; none were malformed. Graph reallocations: 0 in 60
  requests ([r207b_reallocs.tsv](r207b_reallocs.tsv)).
- **Clocks:** busy medians 1,847 1,847 1,885 1,885 1,860 1,911 1,873 1,911 1,873 MHz on cards 0–8, the candidate's
  first run's card for card; the other two runs read 1,898 MHz on card 3 ([r207b_clocks.tsv](r207b_clocks.tsv)).
- **Host** ([r207b_host.tsv](r207b_host.tsv)): lowest MemAvailable 1,182 MiB at 18:17:13, during the last task but one,
  and 3,102 MiB at the end of the log; swap 31 MiB at the start and 342 MiB at its highest at 17:47:22, during the load.
  The RAM watchdog stopped nothing. The card peaks in the NVML card log are 5,808 5,828 5,754 5,828 5,438 5,546 5,706
  5,782 5,190 MiB: part (7)'s on seven cards and 2 MiB above them on cards 4 and 8. The candidate's two runs peak 22 18
  22 18 16 16 18 14 12 MiB above this run on cards 0–8. The report's memory gate ([o4v5b_gonogo.md](o4v5b_gonogo.md))
  reads the client's log shared with R160's and R166's runs: NO-GO on its 718 MB low and its 343 MB swap, as before.

**With part (6), both models support moving `pascal-p106-v5` to the candidate `602691153`.** On Ornith, every measure
of the same work is within the spread of each build's own two runs. On Flash-Next, the candidate's run was within
v4.0's two runs on equal-work minutes, generation and the first turns. The only differences are session averages
that follow each session's mix of prompts and reply depths.

`pascal-p106-v5` stays `1f1110c67` until the owner moves it.

## Not checked here

- **Why Flash-Next's prompts stopped matching inside 7 replies** (part (6)): C884 (Q1598).
- **Why the candidate's card peaks in the agent run are above `1f1110c67`'s on every card** (parts (7) to (9)): the
  two builds load to the same MiB, both candidate runs peak at the same MiB on every card, and `1f1110c67`'s two runs
  within 2 MiB of each other; the peaks were not traced to an allocation.
- **Why the candidate's first Ornith run read the requests with a cached prefix 0.7 % slower than the three runs after
  it** (part (9)): it was the day's first agent run on this rig, after the morning's kernel suite and loads; no run
  timed the rig at that hour on another build (Q1600, parked).
- **Speed on the same rig for Flash-Next against `1f1110c67`.** Parts (2) to (4) time nothing. Part (6) has one v4.0
  reference on this rig; part (7) is the same-rig reference for Ornith only.
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
| [run/ornith_agent.sh](run/ornith_agent.sh) | parts (5), (7), (8) and (9)'s run and report (`ARM` names the arm: `o4c2`, the default, `o4v5a`, `o4c3` and `o4v5b`) |
| [run/ornith_tables.sh](run/ornith_tables.sh) | part (5)'s tables, with [run/same_prompts.py](run/same_prompts.py), [run/busy_clocks.py](run/busy_clocks.py) and [run/host_cards.py](run/host_cards.py) (R166's `host.py` for the NVML card log) |
| [flash_agent.tsv](flash_agent.tsv), [flash_agent_nocx.tsv](flash_agent_nocx.tsv) | part (6), `agent_tables.py` on R90's `v3` and `v4rc2`: passed, turns, prompt and generation tok/s, first token median, p90 and first turn, draft acceptance, minutes, minutes a task at equal work against R90's `v3`; all 12 tasks, and the 11 without `complex-numbers` |
| [flash_same_prompts.md](flash_same_prompts.md) | the 11 requests read on the same prompt (task, turn, new tokens, cached tokens within one) in `v4rc2`, R156a `v4r` and R156b `v4r2`: each run's prompt ms; and each run's generation over all its requests |
| [flash_work.md](flash_work.md), [flash_work.tsv](flash_work.tsv), [flash_work_nocx.tsv](flash_work_nocx.tsv) | `report.py work` on the three runs' per-request tables (prompt tokens read, tokens generated, replies re-read and each re-read turn), and the prompt tokens read and generated without `complex-numbers` |
| [flash_clocks.tsv](flash_clocks.tsv) | each card's median SM clock while busy, and its share of busy samples below 1,800 MHz, over each run's tasks |
| [flash_host.tsv](flash_host.tsv) | the three runs' lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end, each card's peak memory (`v4rc2`'s from the NVML card log, 75 MiB over `nvidia-smi`'s on an idle card) |
| [flash_reallocs.tsv](flash_reallocs.tsv) | `v4rc2`'s graph reallocations and prompt seconds per prompt size |
| [flash_compare.md](flash_compare.md), [flash_report_table.md](flash_report_table.md), [flash_gonogo.md](flash_gonogo.md) | the agent check's report on the run (the harness's arm name `v4r`): per task, per arm and first token per turn; the attempts; the gates and the server log's counts |
| [agent/v4rc2/](agent/v4rc2/), [agent/r90v3/](agent/r90v3/) | `turns.csv` and `attempts.csv` of this run (arm renamed from `v4r`) and of R90's `v3` (2026-11-11, the equal-work reference) |
| [agent/r156a_v4r/](agent/r156a_v4r/), [agent/r156b_v4r2/](agent/r156b_v4r2/) | R156a's and R156b's per-request tables rebuilt from their server logs by `run/server_turns.py` (the client's columns empty, the cached count within one token) |
| [run/flash_agent.sh](run/flash_agent.sh) | part (6)'s run and report |
| [run/flash_tables.sh](run/flash_tables.sh) | part (6)'s tables, with [run/server_turns.py](run/server_turns.py) and [run/same_prompts.py](run/same_prompts.py) (`CACHE_TOL`) |
| [r206_agent.tsv](r206_agent.tsv) | part (7), `agent_tables.py` on R160's and R166's runs, `o4c2` and `o4v5a`: the columns of `ornith_agent.tsv` |
| [r206_same_prompts4.md](r206_same_prompts4.md), [r206_same_prompts.md](r206_same_prompts.md) | the requests read on the same prompt in `o4v5a`, `o4c2`, `o4sna` and `o4snb` (21), and in this rig's two runs alone (25): each run's prompt ms; and each run's generation over all its requests |
| [r206_paired4.md](r206_paired4.md), [r206_paired.md](r206_paired.md), [r206_paired_r166.md](r206_paired_r166.md) | request by request, the candidate against `1f1110c67` on the 21 and on the 25, and `o4snb` against `o4sna` on the 21: each request's ratio, then the totals, the requests read slower and faster and the median ratio, for all, the first turns and the requests with a cached prefix |
| [r206_work.md](r206_work.md), [r206_work.tsv](r206_work.tsv) | `report.py work` on the four runs' `turns.csv` |
| [r206_clocks.tsv](r206_clocks.tsv) | each card's median SM clock while busy, and its share of busy samples below 1,800 MHz, over `o4v5a`'s and `o4c2`'s tasks |
| [r206_host.tsv](r206_host.tsv) | `o4v5a`'s lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end, each card's peak memory (NVML card log) |
| [r206_reallocs.tsv](r206_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, `o4v5a` and `o4c2` |
| [o4v5a_compare.md](o4v5a_compare.md), [o4v5a_report_table.md](o4v5a_report_table.md), [o4v5a_gonogo.md](o4v5a_gonogo.md) | the agent check's report on `o4v5a`: per task, per arm and first token per turn; per arm over the client's attempts; the gates and each server log's counts |
| [agent/o4v5a/](agent/o4v5a/) | `turns.csv` and `attempts.csv` of `o4v5a` |
| [run/r206_tables.sh](run/r206_tables.sh) | part (7)'s tables, with [run/paired_prompts.py](run/paired_prompts.py) |
| [r207_agent.tsv](r207_agent.tsv) | part (8), `agent_tables.py` on R160's and R166's runs, `o4c2`, `o4v5a` and `o4c3`: the columns of `ornith_agent.tsv` |
| [r207_r206set.md](r207_r206set.md) | part (7)'s 21 requests: `o4v5a`'s, `o4c2`'s and `o4c3`'s seconds (`o4c3`'s from the one-token table), for all, the first turns and the 20 with a cached prefix |
| [r207_same_prompts5.md](r207_same_prompts5.md), [r207_same_prompts.md](r207_same_prompts.md) | the requests read on the same prompt in `o4v5a`, `o4c2`, `o4c3`, `o4sna` and `o4snb` (20), and in this rig's three runs alone (22): each run's prompt ms; and each run's generation over all its requests |
| [r207_paired5.md](r207_paired5.md), [r207_paired5_o4c2.md](r207_paired5_o4c2.md), [r207_paired5_cand.md](r207_paired5_cand.md), [r207_paired5_r166.md](r207_paired5_r166.md) | request by request on the 20 matched in all five runs: `o4c3` against `o4v5a`, `o4c2` against `o4v5a`, `o4c3` against `o4c2`, `o4snb` against `o4sna`; each request's ratio, then the totals, the requests read slower and faster and the median ratio, for all, the first turns and the requests with a cached prefix |
| [r207_paired5_tol1.md](r207_paired5_tol1.md), [r207_paired5_tol1_o4c2.md](r207_paired5_tol1_o4c2.md), [r207_paired5_tol1_cand.md](r207_paired5_tol1_cand.md), [r207_paired5_tol1_r166.md](r207_paired5_tol1_r166.md) | the same four pairs with the cached tokens matched within one (`CACHE_TOL=1`): 23 requests |
| [r207_paired.md](r207_paired.md), [r207_paired_o4c2.md](r207_paired_o4c2.md), [r207_paired_cand.md](r207_paired_cand.md) | the same on the 22 matched in this rig's three runs: `o4c3` and `o4c2` against `o4v5a`, `o4c3` against `o4c2` |
| [r207_work.md](r207_work.md), [r207_work.tsv](r207_work.tsv) | `report.py work` on the five runs' `turns.csv` |
| [r207_clocks.tsv](r207_clocks.tsv) | each card's median SM clock while busy, and its share of busy samples below 1,800 MHz, over `o4c3`'s, `o4v5a`'s and `o4c2`'s tasks |
| [r207_host.tsv](r207_host.tsv) | `o4c3`'s lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end of the log, each card's peak memory (NVML card log) |
| [r207_reallocs.tsv](r207_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, `o4c3`, `o4v5a` and `o4c2` |
| [o4c3_compare.md](o4c3_compare.md), [o4c3_report_table.md](o4c3_report_table.md), [o4c3_gonogo.md](o4c3_gonogo.md) | the agent check's report on `o4c3`: per task, per arm and first token per turn; per arm over the client's attempts; the gates and each server log's counts |
| [agent/o4c3/](agent/o4c3/) | `turns.csv` and `attempts.csv` of `o4c3` |
| [run/r207_tables.sh](run/r207_tables.sh) | part (8)'s tables, with [run/paired_prompts.py](run/paired_prompts.py) (`CACHE_TOL`) |
| [r207b_agent.tsv](r207b_agent.tsv) | part (9), `agent_tables.py` on R160's and R166's runs, `o4c2`, `o4v5a`, `o4c3` and `o4v5b`: the columns of `ornith_agent.tsv` |
| [r207b_r206set.md](r207b_r206set.md) | part (7)'s 21 requests: each of the four rig-1 runs' seconds (`o4c3`'s and `o4v5b`'s from the one-token tables), each build's mean of two, the candidate's mean against `1f1110c67`'s and each build's second run against its first, for all, the first turns and the 20 with a cached prefix |
| [r207b_pairs.md](r207b_pairs.md), [r207b_pairs_tol1.md](r207b_pairs_tol1.md) | the totals of the six pairs below on the requests matched in all four rig-1 runs (20, and 23 with the cached tokens within one): all, the first turns and those with a cached prefix, the requests read slower and faster and the median ratio |
| `r207b_paired_v5a_c2.md`, `r207b_paired_v5a_c3.md`, `r207b_paired_v5b_c2.md`, `r207b_paired_v5b_c3.md`, `r207b_paired_v5.md`, `r207b_paired_cand.md`, each with a `_tol1` twin | request by request: each candidate run against each `1f1110c67` run, `o4v5b` against `o4v5a` and `o4c3` against `o4c2`; exactly, and with the cached tokens within one |
| [r207b_same_prompts4.md](r207b_same_prompts4.md), [r207b_same_prompts6.md](r207b_same_prompts6.md) | the requests read on the same prompt in the four rig-1 runs (20), and in those and v4.0's two (20): each run's prompt ms; and each run's generation over all its requests |
| [r207b_decode.md](r207b_decode.md) | the four rig-1 runs' generation per middle-context bucket (requests, tokens, decode s, tok/s, ms a token), and each run's tok/s at the same mix of contexts |
| [r207b_work.md](r207b_work.md), [r207b_work.tsv](r207b_work.tsv) | `report.py work` on the six runs' `turns.csv` |
| [r207b_clocks.tsv](r207b_clocks.tsv) | each card's median SM clock while busy, and its share of busy samples below 1,800 MHz, over the four rig-1 runs' tasks |
| [r207b_host.tsv](r207b_host.tsv) | `o4v5b`'s lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end of the log, each card's peak memory (NVML card log) |
| [r207b_reallocs.tsv](r207b_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, the four rig-1 runs |
| [o4v5b_compare.md](o4v5b_compare.md), [o4v5b_report_table.md](o4v5b_report_table.md), [o4v5b_gonogo.md](o4v5b_gonogo.md) | the agent check's report on `o4v5b`: per task, per arm and first token per turn; per arm over the client's attempts; the gates and each server log's counts |
| [agent/o4v5b/](agent/o4v5b/) | `turns.csv` and `attempts.csv` of `o4v5b` |
| [run/r207b_tables.sh](run/r207b_tables.sh) | part (9)'s tables, with [run/paired_prompts.py](run/paired_prompts.py) and [run/decode_by_context.py](run/decode_by_context.py) |

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

Part (6), the same way (here the rig's paths are written as `$MODELS` and `/opt/llama-builds`, as in R156):

```bash
# the run (~5 min load, 12 tasks) and its report; r156.sh's agent and report steps, arm v4r, the candidate as NEW
RIG=$RIG CLIENT=local MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf NEW=$NEW REF=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 \
  RIGDIR=<rig folder> CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/c854-flash OUT=<out> \
  bash docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh

# R156's per-request tables from their server logs and client logs (R156's raw logs, not published)
python3 -I docs/data/2026-11-18_c854b_p106_check/run/server_turns.py r156a-v4r <R156a>/server-v4r.log <R156a>/v4r.log \
  > docs/data/2026-11-18_c854b_p106_check/agent/r156a_v4r/turns.csv
python3 -I docs/data/2026-11-18_c854b_p106_check/run/server_turns.py r156b-v4r2 <R156b>/server-v4r2.log <R156b>/v4r2.log \
  > docs/data/2026-11-18_c854b_p106_check/agent/r156b_v4r2/turns.csv

# the tables, on a folder holding the server logs and host logs run/flash_tables.sh lists
bash docs/data/2026-11-18_c854b_p106_check/run/flash_tables.sh <tables folder>
```

`agent/v4rc2/` is the run's `res/turns.csv` and `res/attempts.csv` with the arm renamed from `v4r` (and
`agent/r90v3/` R90's `v3` rows, renamed from `v3`). The prefix-match points in part (6) are each re-read turn's
`selected slot by LCP similarity ... f_keep` line in the server log times the cached tokens before it, less the
previous prompt's end.

Part (7), as part (5) with one more name and `pascal-p106-v5`'s build, from the repository root on the machine that
drives the rig, with the rig clean:

```bash
REF=llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410
RIG=$RIG CLIENT=local ARM=o4v5a MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf BIN=/opt/llama-builds/$REF/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4v5a \
  bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4v5a MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> OUT=<out>/agent STEPS="report collect" bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh

# the tables, on a folder holding part (5)'s table rows plus o4v5a's, both runs' server logs and o4v5a's host logs
# (run/r206_tables.sh lists them)
bash docs/data/2026-11-18_c854b_p106_check/run/r206_tables.sh <tables folder>
```

Part (8), as part (5) with the arm's name, from the repository root on the machine that drives the rig, with the rig
clean:

```bash
RIG=$RIG CLIENT=local ARM=o4c3 MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf BIN=/opt/llama-builds/$NEW/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4c3 \
  bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4c3 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> OUT=<out>/agent STEPS="report collect" bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh

# the tables, on a folder holding part (7)'s table rows plus o4c3's, the three runs' server logs and card logs and
# o4c3's host log (run/r207_tables.sh lists them); the window is o4c3's tasks, from the client's first request
O4C3_FROM="2026-11-18 16:57:49" O4C3_TO="2026-11-18 17:27:33" \
  bash docs/data/2026-11-18_c854b_p106_check/run/r207_tables.sh <tables folder>
```

Part (9), as part (7) with the arm's name, from the repository root on the machine that drives the rig, with the rig
clean:

```bash
REF=llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410
RIG=$RIG CLIENT=local ARM=o4v5b MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf BIN=/opt/llama-builds/$REF/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4v5b \
  bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4v5b MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> OUT=<out>/agent STEPS="report collect" bash docs/data/2026-11-18_c854b_p106_check/run/ornith_agent.sh

# the tables, on a folder holding part (8)'s table rows plus o4v5b's, the four runs' server logs and card logs and
# o4v5b's host log (run/r207b_tables.sh lists them); the window is o4v5b's tasks, from the client's first request
O4V5B_FROM="2026-11-18 17:50:54" O4V5B_TO="2026-11-18 18:19:11" \
  bash docs/data/2026-11-18_c854b_p106_check/run/r207b_tables.sh <tables folder>
```
