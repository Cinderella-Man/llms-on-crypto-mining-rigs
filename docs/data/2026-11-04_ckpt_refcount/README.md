# Reference-counted context checkpoints, and a memory fit that counts the drafter (C154, C155; 2026-11-04)

Registry rows [Q704 and Q706](../../../quants/experiments.md), ideas 6 and 12 of
[the November news note](../../research/notes/flash_next_news_2026-11.md). Two host-side upstream changes, ported onto
Miner Llama's `pascal-p106-v3-b12076-ladder2` as `pascal-p106-v3-b12076-ckptrc`:

- **[#30137](https://github.com/ggml-org/llama.cpp/pull/30137) (open): reference-counted checkpoints.** llama-server's
  context checkpoints (saved recurrent states, 112.6 MiB each for Flash-Next on the rig) are deep-copied whenever a
  prompt is copied: into the prompt cache (`--cache-ram`), and into child slots (several completions of one request).
  The PR keeps them as shared pointers to checkpoints that never change once taken, so a copy of a prompt costs no
  checkpoint data, and it stops zero-filling state buffers that are written whole right after. Its author measured
  1,846 → 1,621 MiB of peak memory with two checkpoints and an 8 GB prompt cache.
- **[#31610](https://github.com/ggml-org/llama.cpp/pull/31610) and [#31569](https://github.com/ggml-org/llama.cpp/pull/31569)
  (open): the memory fit and the drafter.** `-fit on` measures a draft model on its own, and drafts that read the
  target's context (DFlash, EAGLE-3, Gemma 4's assistant) fail that measurement, so the fit leaves them out. #31610
  measures the draft with a no-alloc context of the main model beside it. #31569 counts the NextN layers that the
  loader counts for `-ngl` even when they are not loaded (no MTP), so the fitted `-ngl` and `-ts` match the loader.

## The switches

| Switch | Default | What it does |
|---|---|---|
| `--ctx-checkpoints-share` (env `LLAMA_ARG_CTX_CHECKPOINTS_SHARE`) | off | #30137's sharing: the prompt cache's entries and child slots share the slot's host checkpoints instead of copying them. Off, every copy of a prompt copies the checkpoints' data, as before |
| `LLAMA_FIT_DRAFT_CTX` | on (`0` = as before) | #31610: the fit measures a draft with the main model's context, and a draft that reads the target's embeddings or head (`LLAMA_DRAFT_SHARE`, [Q564](../2026-11-04_draft_mem/README.md)) without those tensors, as it loads |
| `LLAMA_FIT_NEXTN_SKIP` | on (`0` = as before) | #31569: the fitted `-ngl` and `-ts` include the NextN layers that are not loaded, on the output layer's device |

Always on (they change no state): checkpoint and prompt-cache buffers are no longer zero-filled before they are
written (`common_state_data`), and a prompt-cache save that copies less than the whole state is dropped instead of
kept.

How the port fits the fork's own checkpoint code ([`-ckpt`](../2026-11-04_ckpt_cards/README.md)):

- A checkpoint in a device store (`--ctx-checkpoints-vram`) is never shared: copies of a prompt take host copies of
  those, or none (child slots), as before. Its move to the host, in the background or at once
  (`--ctx-checkpoints-vram-sync`), replaces it with a new host checkpoint instead of changing it in place, so a
  checkpoint someone else holds never changes.
- `--ctx-checkpoints-spread` and its smallest-gap eviction work on the shared list unchanged; erasing a checkpoint from
  a slot drops only the slot's reference.
- The prompt cache counts a shared checkpoint in each entry that holds it, as upstream does (Q741).

## Result: the same states, and the memory it saves needs a prompt cache

Checked on the CPU in a container without a GPU ([run/checks.sh](run/checks.sh)): random-weight models from
`test-llama-archs` (qwen4exp, Flash-Next's architecture; qwen35; mamba) and, for checkpoints of a real size,
Qwen3.6-35B-A3B Q4_K_M (Gated DeltaNet + attention like Flash-Next; one checkpoint is 62.8 MiB).

- **`test-llama-context`**, every case on the three random models, C22's device-store cases `store` and `store_p`
  included: 54 OK, 6 SKIP, 0 FAIL, as on `-ladder2` ([context_cases.tsv](context_cases.tsv)).
- **C22's six server arms** ([server_check.sh](../2026-11-04_ckpt_cards/run/server_check.sh): host checkpoints, all
  on the devices, two on the devices with background and immediate moves, the spread, spread + devices) pass on this
  build, and again with `--ctx-checkpoints-share` on every arm ([server_check_verdicts.txt](server_check_verdicts.txt)).
- **[server_mem.sh](run/server_mem.sh)** sends eight requests to each server: a 3,000-token prompt, its continuation,
  the history edited at token 2,200, the original again, a continuation of the edit, then a second 3,000-token
  conversation (with a prompt cache, the first one's state and checkpoints go into it), back to the first (restored
  from the cache, the second saved), and back to the second. Every switch-on arm gives its switch-off arm's tokens and
  top-3 probabilities exactly, over all eight requests, and the switch-off arms give `-ladder2`'s
  ([mem_verdict.txt](mem_verdict.txt), [mem_requests.tsv](mem_requests.tsv)).

Peak resident memory (VmHWM, MiB) per server, the random qwen4exp on the rig build (checkpoints of 0.17 MiB) and
Qwen3.6-35B-A3B on a native CPU build (`-lm none`, so the weights are counted too):

| Arms | Prompt cache, checkpoints | Random qwen4exp: off / on / `-ladder2` | Qwen3.6-35B-A3B: off / on | Off − on |
|---|---|---|---|---|
| `c0` | `--cache-ram 0`, 4 | 241.0 / 241.1 / 240.9 | 20,188.7 / 20,188.4 | +0.3 |
| `cr` | 4096 MiB, 4 | 251.0 / 251.5 / 251.2 | 20,767.6 / 20,579.9 | **+187.7** (3 × 62.8) |
| `cr2` | 4096 MiB, 2 | 250.6 / 250.9 / — | 20,642.9 / 20,516.3 | **+126.6** (2 × 62.8) |
| `crv` | 4096 MiB, 4, `--ctx-checkpoints-vram 2` | 251.4 / 251.0 / — | — | — |
| `crs` | 4096 MiB, 4, `--ctx-checkpoints-spread 512` | 251.4 / 250.8 / — | — | — |

- **With a prompt cache the switch saves the copies of the checkpoints at each save.** On Qwen3.6-35B-A3B the peak falls by the
  saved checkpoints' size, at the saves themselves: when the second conversation's state goes into the cache (P7, two
  checkpoints) the switch-off server peaks 124.8 MiB higher, when the first one's goes in (P8, three checkpoints)
  187.7 MiB higher. The resident memory after the requests is the same (20,523.1 MiB in both `cr` arms): the copy
  arm's extra copies are freed when the slot drops its own, as upstream reports (peak, not steady memory).
- **With `--cache-ram 0` nothing copies a checkpoint**, so the switch changes nothing: the c0 arms are equal within
  noise. The rig's lines run `--cache-ram 0` and one completion per request
  ([bench/agent_check/rig_lib.sh](../../../bench/agent_check/rig_lib.sh)), so the news note's estimate of −225 to
  −450 MiB of peak host memory does not reach them; on the rig the change is expected to give the same host memory,
  less the zero-fill of each new 112.6 MiB checkpoint (a memory pass per save). [run/rig_share_arm.sh](run/rig_share_arm.sh)
  checks that on the agent check.
- On the random model the differences (±0.6 MiB) are below what the test can resolve: its checkpoints are 0.17 MiB.

## Result: the fit now counts a DFlash draft

[fit_check.sh](run/fit_check.sh) on the CPU: the fit's projection of host memory (with no GPU the fit has no device to
split over, so it adds everything up), from the files' metadata only. Flash-Next is Swift-1.5 GSQ-RCO Q2_0 at 64k
context, `-b 2048 -ub 512`; DFlash2 is the drafter of [Q561](../2026-11-03_flash_next_dflash2/README.md) (Q8_0),
the MTP draft the separate MTP file of [Q226](../2026-11-03_flash_next_mtp/README.md). MiB ([fit.tsv](fit.tsv)):

| Case | Draft | Switches | Projected (main + draft) | The draft's own share |
|---|---|---|---|---|
| `s_main` | none | — | 65,709 (63,456 weights + 2,033 context + 219 compute) | — |
| `s_dflash0` | DFlash2 | `LLAMA_FIT_DRAFT_CTX=0` (as before) | 66,047 | **0: "dflash requires ctx_other", fitted without it** |
| `s_dflash_old` | DFlash2 | `-ladder2` | 66,047 | 0, the same failure |
| `s_dflash` | DFlash2 | default | 66,768 | **721** (611 + 45 + 64) |
| `s_mtp0`, `s_mtp`, `s_mtp_old` | MTP | either, and `-ladder2` | 68,973 | 2,926 (2,562 + 161 + 203) |
| `s_mtp_share` | MTP | `LLAMA_DRAFT_SHARE=all` | 67,832 | 1,784 (1,420 + 161 + 203): the 1,142 MiB of embeddings and head it reads from the target are not counted |

- With any draft the main context grows by 338 MiB (2,033 → 2,371: speculation keeps more recurrent states), on
  every build; that part was always counted.
- **The DFlash2 draft's 721 MiB were missing from every fit before**, which is what #31610 fixes. On the rig they sit
  on the card that holds the draft (`-devd`), so a split fitted with a DFlash draft had ~0.7 GB less on that card than
  the fit said. The MTP draft needs no target context, so its measurement was right before and is unchanged; Q213's
  numbers with the MTP draft stand.
- The random qwen4exp drafting for itself (`draft-simple`): 65 MiB more with or without the switch; with
  `LLAMA_DRAFT_SHARE=all` 1 MiB less, and the log reads "reading from the source model: token embeddings output head".
- `LLAMA_FIT_NEXTN_SKIP` changes only the layer split over devices, which a machine without a GPU does not reach, and
  only for a file that carries NextN layers without loading them. The Swift file's MTP head is a separate file (an MTP
  context on the Swift file alone fails: "model doesn't contain MTP layers"), so on it the switch is expected to change
  nothing; the rig's fit probe confirms that (below).

## On the rig (next)

- **The agent check with the switch** ([run/rig_share_arm.sh](run/rig_share_arm.sh)): the checkpoint line of the
  checkpoints-on-the-cards check (its `ckv` arm, `--ctx-checkpoints-vram 4`) with `--ctx-checkpoints-share`, on this
  build; then the host memory at its low point from `mem_guard.sh`'s log (MemAvailable, MemAvailable minus CmaFree,
  swap). Expected: the same as without the switch.
- **The fit per card with DFlash2** ([scripts/fit_probe.sh](../../../scripts/fit_probe.sh), whose configs now take a
  fifth field of environment variables): the release split with and without the DFlash2 draft, each with
  `LLAMA_FIT_DRAFT_CTX=0` and without; and `-fit on` without `-ts`, with and without `LLAMA_FIT_NEXTN_SKIP=0` (expected
  equal on the Swift file).

## Files

| File | What |
|---|---|
| [context_cases.tsv](context_cases.tsv) | `test-llama-context` per model and case: verdict and skip reason |
| [server_check_verdicts.txt](server_check_verdicts.txt) | C22's server arms on this build, without and with `--ctx-checkpoints-share` |
| [mem.tsv](mem.tsv) | `server_mem.sh` per model, arm and request: VmRSS, VmHWM and RssAnon (kB) after the request |
| [mem_requests.tsv](mem_requests.tsv) | per model, arm and request: prompt tokens processed and reused, where the reuse came from, checkpoints created, the generated tokens |
| [mem_verdict.txt](mem_verdict.txt) | the comparisons of `server_mem.sh` |
| [fit.tsv](fit.tsv), [fit_lines.txt](fit_lines.txt) | `fit_check.sh`: the projection per case, and the fit's log lines (breakdowns, the draft's failure) |
| [run/chain.sh](run/chain.sh) | the rig build of the branch, a native CPU build of the same tree, then `checks.sh` in a container without a GPU |
| [run/checks.sh](run/checks.sh) | the random models, `test-llama-context`, C22's `server_check.sh` twice, `server_mem.sh` on both models, `fit_check.sh` |
| [run/server_mem.sh](run/server_mem.sh) | the server arms and their memory |
| [run/fit_check.sh](run/fit_check.sh) | the fit cases |
| [run/rig_share_arm.sh](run/rig_share_arm.sh) | for the rig: the agent-check arm with the switch, and its host memory at the low point |

## Commands

The build machine needs Docker and a llama.cpp clone with the fork as remote `fork` in `$WORK/src`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)). `OLD_BIN` is the `bin/` of the
`-ladder2` rig build; `MOUNTS` the folders the models and `OLD_BIN` live in.

```bash
WORK=<work dir> REPO=<this repository> BRANCH=pascal-p106-v3-b12076-ckptrc JOBS=6 THREADS=8 \
  OLD_BIN=<-ladder2 build>/bin BIG_MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf> \
  MAIN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf> DRAFT=<Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf> \
  MTP=<mtp-Qwen3.8-Flash-Next-draft.gguf> MOUNTS='<model and build folders>' DOCKER_ARGS='--label <your label>' \
  bash docs/data/2026-11-04_ckpt_refcount/run/chain.sh
cat $WORK/checks/mem/verdict.txt $WORK/checks/mem_big/verdict.txt $WORK/checks/fit/fit.tsv
```

On the rig, from the machine that drives the agent check (`agent_check.sh`'s environment, e.g.
[s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example)), with this build's `llama-server` as `REL_BIN`:

```bash
LINE='<the release line> --ctx-checkpoints-vram 4' ARMS=ckvs bash docs/data/2026-11-04_ckpt_refcount/run/rig_share_arm.sh <out>
# ARMS="ckvr ckvs" adds the same line without the switch on this build (+40 min); read <out>/host_memory.tsv
```

## Provenance

The build: `pascal-p106-v3-b12076-ckptrc` `3e49981fb` (6 commits on `-ladder2` `a330672c1`: #31569's three and
#31610's one cherry-picked, one conflict on the draft's measuring call resolved as both; the fit switches; the #30137
port), rig tarball md5 `1223019d520eb3d290101ecb7b5316bc` (sm_61, no AVX). Models: random weights generated by
`test-llama-archs -s 1`; Qwen3.6-35B-A3B Q4_K_M; Swift-1.5 GSQ-RCO Q2_0 (Swift Open License 1.0 and qwen-community-1.0),
the DFlash2 drafter (qwen-community-1.0) and the MTP draft (Apache-2.0), of which the fit reads only the metadata.
The tables hold memory sizes, token counts and the token ids generated for random-token prompts; no weights or text.
