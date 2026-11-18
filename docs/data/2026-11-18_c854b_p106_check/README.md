# Upstream b12483 on a P106: `pascal-p106-v5-cand-b12483` against `pascal-p106-v5` (2026-11-18)

**Passed.** On a P106 rig the candidate on upstream b12483 runs every release file exactly as `pascal-p106-v5`
(`1f1110c67`, upstream b12410) does. The replies are the same bytes, memory on every card is the same to the MiB, and
every evaluation launches the same kernels. The fork's kernel tests give the same table as the last check, and the
planted faults are still caught. Two random test models give different numbers: qwen3next and qwen4exp. Both come
from upstream fixes that now apply an optional weight scale these models' graphs used to ignore (#32393, #32476),
not from a kernel. Flash-Next, the one release file of either type, runs the same graph and gives the same replies.
`pascal-p106-v5` is not moved by this check alone: upstream rewrote the server's slot scheduling between the two tags,
so one agent run each on Ornith and Flash-Next comes first.

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

## Not checked here

- **The server's new slot scheduling under an agent.** Upstream's #32154 and the carried #32434 change which cached
  state a turn reuses. Two greedy requests without a prompt cache do not exercise that; one agent-check run each on
  Ornith-1.5-122B Pascal N2P GGUF and Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, with the work counts (prompt tokens read
  a task), does, and comes before `pascal-p106-v5` moves.
- **Speed.** Nothing is timed.
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
