# `--mtp-device` and memory, on three newer cards (C954, 2026-11-23)

What this is: the GPU half of the check of [C910](../2026-11-21_c910_mtp_device_mem/README.md) (registry
[Q1647](../../../quants/experiments.md)), point 8 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). `--mtp-device N` puts a model's MTP layer (the layer that drafts the next
token) on card N together with a copy of the output head, so that drafting does not hop to the head's card and back.
C910's branch (`pascal-p106-v5-mtp-device-mem` `f94aa552b`, two commits on `pascal-p106-v5` `3dfad876f`) fixes three
things that mattered on a rig's 4 GB host: the memory fit did not count the copy, the copy was never freed, and making
it held the whole head in host memory for a moment. With the switch unset nothing changes. The branch's processor
check passed; here it runs on a build machine's three RTX 3090s, through the rig build's compute 6.1 PTX (the same
binaries a P106 rig runs), with the real model. Correctness and memory only: nothing here is timed.

## Result

**In plain words:** it passes. With the branch, the memory planner counts the head's copy on the chosen card, the
copy is listed as one of the model's buffers, the card gets all of it back when the model is freed, and loading never
holds the head in host memory. The base build, with the same switch, leaves the copy out of the plan, keeps 502 MiB
of card 0 after every free, and holds the whole head (500 MiB) in host memory for about 0.4 s in every load. Every
reply is the same token for token, with the switch and without, on both builds.

The output head of Ornith-1.5-122B Pascal N2P GGUF is 500.2 MiB (Q5_0). Numbers are one run each, MiB per card from
`nvidia-smi`.

| Pass criterion (Q1687 point 8) | Result |
|---|---|
| the copy counted by the fit and shown as a model buffer | **yes.** Before each load the fit projects card 0 at 17,014 MiB with `--mtp-device 0` on the branch, 500 MiB more than the base's 16,514 with the same line (cards 1 and 2 the same on both); the loader prints the copy as one more `CUDA0 model buffer size = 500.17 MiB` line beside `MTP output head copied to CUDA0, 500.2 MiB` ([fit.tsv](fit.tsv), [buffers.tsv](buffers.tsv)). Card 0 once up is 17,335 MiB on both builds, so the base's plan was 500 MiB short on that card |
| no head-sized rise in host RSS | **yes.** The branch with `--mtp-device 0`: the server's `RssAnon` peaks at 550 MiB in load 1, the unset arms' value, and its largest short-lived buffer is the 145 MiB one every arm's first load has; load 2 none. The base with `--mtp-device 0`: a 500.2 MiB buffer in each load, about 0.4 s long (`RssAnon` 330 → 825 → 325 MiB in load 1, 852 → 1,352 → 852 in load 2) ([rss.tsv](rss.tsv)) |
| card memory back after the free | **yes.** The branch with `--mtp-device 0`: 329, 413 and 327 MiB on cards 0, 1, 2 asleep after free 1 and after free 2, the unset arms' values (the CUDA contexts the process keeps). The base with `--mtp-device 0`: card 0 at 831 after free 1 and 1,333 after free 2, one 502 MiB copy left behind per load ([cards.tsv](cards.tsv)) |
| the same replies with and without | **yes.** All 48 replies (4 arms × 2 loads × 6 requests) the same token ids request by request, and the same drafted and accepted counts ([replies.tsv](replies.tsv), [compare.tsv](compare.tsv)) |

**The fit, card by card** ([fit.tsv](fit.tsv); projected use before load 1, MiB):

| Arm | Card 0 | Card 1 | Card 2 | Card 0 once up |
|---|---:|---:|---:|---:|
| `new-unset` | 15,159 | 16,317 | 16,703 | 15,475 |
| `new-dev0` | 17,014 | 16,317 | 15,349 | 17,335 |
| `base-dev0` | 16,514 | 16,317 | 15,349 | 17,335 |
| `base-unset` | 15,159 | 16,317 | 16,703 | 15,475 |

`--mtp-device 0` moves the MTP layer (layer 48), its cache and its compute buffer from card 2 to card 0 on both builds
(card 2's projection 1,354 MiB lower, card 0's 1,355 higher on the base); the branch adds the copy on top. Between a
projection and the card's use once up the unset arms leave 316 MiB and the branch with the switch 321 (the CUDA
context and the like, which the fit's margin covers); the base with the switch 821, the copy included.

**Each load and free** ([cards.tsv](cards.tsv), card 0 MiB):

| Arm | Up after load 1 | Asleep after free 1 | Up after load 2 | Asleep after free 2 |
|---|---:|---:|---:|---:|
| `new-unset` | 15,475 | 329 | 15,497 | 329 |
| `new-dev0` | 17,335 | 329 | 17,357 | 329 |
| `base-dev0` | 17,335 | **831** | **17,859** | **1,333** |
| `base-unset` | 15,475 | 329 | 15,497 | 329 |

On the base, load 2's fit saw the 502 MiB already lost (card 0 free 23,346 MiB against 23,848 for the branch) but
still left the copy out of its projection. Cards 1 and 2 return to 413 and 327 MiB after every free in every arm.

**Host memory** ([rss.tsv](rss.tsv)): the server's `RssAnon` (its own memory; the model file's mapped pages, up to
46,275 MiB of `RssFile`, are not in it) at a load's start, its largest value during the load, and once the server
answered:

| Arm | Load 1: start, peak, up | Load 2: start, peak, up (after the waking request) |
|---|---|---|
| `new-unset` | 97, 550, 550 | 801, 850, 850 |
| `new-dev0` | 97, 550, 550 | 801, 903, 903 |
| `base-dev0` | 97, **824**, 550 | 851, **1,352**, 852 |
| `base-unset` | 98, 550, 550 | 801, 849, 849 |

The base's peaks are the head buffer: it rises by 495 and 492 MiB over 261 ms (the value read every 5 ms), holds for
about 145 ms while the head crosses the host, and falls by 500.2 MiB. The branch's copy goes through a buffer of at most
16 MiB, which leaves no step of its own in the samples. The 145 MiB that every arm's first load takes and returns within
a second, about 0.5 s before the server answers, comes with the setting unset too, so it is not the head. Load 2's
level once up is read after the one-token request that woke the server, so it moves by tens of MiB with when that
request's allocations land; after the requests all four arms sit at 908 to 915 MiB.

**Which kernels ran.** The server's switch line lists the forced Pascal kernels as set (`GGML_CUDA_MMVQ_PASCAL=2`,
`GGML_CUDA_MMVF_PASCAL=2` and the P106 shapes), so the replies ran on the fork's Pascal matrix-vector kernels as a
P106 runs them. The copy's placement and memory do not depend on them.

## How it ran

**Loading and freeing twice in one process.** llama-server's sleep mode (`--sleep-idle-seconds 15`) frees the model
and its contexts after 15 idle seconds and loads them again, in the same process, on the next request. Each arm is one
server process: load 1, the requests, sleep (free 1), a one-token request that wakes it (load 2), the requests again,
sleep (free 2).

**The line.** Ornith-1.5-122B Pascal N2P GGUF with its v5 preset ([release/ornith-v5](../../../release/ornith-v5/settings.env)),
the layer split changed to the three cards (`-ts 1,1,1`, which puts the output head on the last card), drafting one
token a step with the MTP layer in its file (`--spec-type draft-mtp --spec-draft-n-max 1`, as C899's MTP arms and
C910's rig check run it), and the memory fit on at log verbosity 4 instead of the preset's `-fit off`, so that the
server log shows the fit's projected memory per card before each load (on these cards the fit changes nothing, so the
load is the preset's). The fork's Pascal matrix-vector kernels turn themselves on only on compute 6.1 cards, so every
arm forces them on with a P106's launch shapes (`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2
GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`), as [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) did: the replies ran
on the fork's kernels. Where the head's copy sits and what it costs do not depend on them.

| Arm | Build | `--mtp-device` |
|---|---|---|
| `new-unset` | the branch `f94aa552b` | unset: the MTP layer on the last card, no copy |
| `new-dev0` | the branch | `0`: the MTP layer and the head's copy on the first card |
| `base-dev0` | the base `3dfad876f` | `0` |
| `base-unset` | the base | unset |

**What each arm records.** The server log (the fit's projection per card, the loader's "model buffer size" lines and
its "MTP output head copied" line, for each load); every card's memory use from `nvidia-smi` before the start, once
up after each load and asleep after each free; the server process's resident memory from `/proc` every 5 ms
([run/rss_sampler.py](run/rss_sampler.py)): `RssAnon` is the process's own memory, where a head-sized host buffer
shows, while `RssFile` counts the model file's mapped pages, which a load maps and unmaps by the gigabyte; and the
replies: [C953's three greedy requests](../2026-11-23_c953_ornith_kcol_gpu/run/replies.py) of 256 tokens (a coding
request, a prose paragraph and an Elixir program) once and once more with `n_probs` 2, after each load.

## Files

| File | What it holds |
|---|---|
| [fit.tsv](fit.tsv) | the memory fit's projection before each load: per card its total, projected use and free memory, and the free-memory target |
| [buffers.tsv](buffers.tsv) | each load's "model buffer size" lines and its "MTP output head copied" line |
| [cards.tsv](cards.tsv) | every card's memory use (MiB) at each phase of each arm |
| [rss.tsv](rss.tsv) | per load: the server's `RssAnon` at the load's start, its largest value during the load and when, and once the server answered; the largest `RssFile` |
| [replies.tsv](replies.tsv) | every request: arm, cycle, round, prompt, tokens, md5 of the token ids and of the text, drafted, accepted, stop |
| [compare.tsv](compare.tsv) | per request, each arm's md5 of the token ids against `new-unset`'s first cycle |
| [run/run.sh](run/run.sh) | the driver |
| [run/rss_sampler.py](run/rss_sampler.py) | the resident-memory sampler |
| [run/tables.py](run/tables.py) | the tables from the driver's output |

## Commands

On the build machine, with the branch's rig tarball (`llama.cpp-20261121-f94aa552b-pascal-p106-v5-mtp-device-mem.tar.gz`,
md5 `3f26fdb33d6d9465040a048ff6a5560f`, built by C910 with `TESTS=1`) and the base's
(`llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`)
unpacked, this repository copied to `$REPO`, and the cards checked with a short CUDA allocation and copy on each first:

```bash
env WORK=$WORK NEW=$NEW_TREE/bin REF=$BASE_TREE/bin REPO=$REPO ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
    MOUNTS="<the folders holding the builds and the model>" DOCKER_ARGS="--cpu-shares=256" \
    setsid nohup bash $REPO/docs/data/2026-11-23_c954_mtp_device_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
# after it wrote $WORK/DONE, on the output copied back:
python3 -I run/tables.py $WORK/load .
```

Every container runs `nvidia/cuda:12.9.2-devel-ubuntu24.04` with the PTX JIT's cache in `$WORK/jitcache`, on all three
cards. The sampler and the requests run on the host.

## Provenance and licence

The model file is Ornith-1.5-122B Pascal N2P GGUF (valid1), the release file of
[release/ornith-v5](../../../release/ornith-v5/settings.env). The prompts are this project's own. The tables are
derived from the runs' logs, memory readings and replies and published under this repository's licence.
