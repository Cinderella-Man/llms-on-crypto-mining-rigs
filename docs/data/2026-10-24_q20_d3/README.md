# Q20/D3: the staircase head in the engine, its exactness checks and the first served rig run (2026-10-24/25)

D3 put the staircase drafter's head into Miner Llama's engine (`--pipe-draft`, `--pipe-guess staircase`, and with
`--pipe-store` behind the copy lookup, the request's continuations and the answer store). The exactness set, run on a
CUDA machine (never the rig), checks the C++ head against PyTorch's `drafter.Drafter.forward` and pipelined decoding with
the head against plain decoding. The served rig run on rc4 (`460160e93`, K2h-r16, depth 3) measured tokens per second
and guess counts per request. The write-up is [docs/drafter.md](../../drafter.md) (§6 "Served on the rig", steps 7–9,
changelog); the registry row is Q20 in [quants/experiments.md](../../../quants/experiments.md). The heads on the rig
(R2-9, R2-gen) and the store were built from the drafter's first data round, which was later excluded on licence grounds
([2026-10-24_q20_drafter](../2026-10-24_q20_drafter/README.md)): these numbers measure the method, not a releasable
head. The three rig tests that were to follow (`run/rig_d3_*.sh`) never ran.

## Results

Served on the rig, medians of the 8 requests (chat, code, edit at 512 tokens and a tool request at 1,536, each at T 0
and with the makers' sampling), tok/s at depth 3:

| Split (as llama assigned it) | Head | No guessing | Copy + continuations + store | Store, then the head | Copy, then the head |
|---|---|---|---|---|---|
| nine, `-ts 6,6,6,6,6,6,6,5,2` (7/6/6/6/6/6/6/5/1, a 7-row window) | R2-9 | 33.4 | 48.6 | 50.4 | 44.4 |
| nine, `-ts 6,6,6,6,5,5,5,5,5` (7/6/6/6/5/5/5/5/4, 6 rows) | R2-9 | 33.4 | 48.6 | 50.3 | – |
| six, llama's own split (9/8/8/9/8/7, 4 rows) | R2-gen | 34.3 | 49.0 | 49.5 | – |

- **The head's guesses on the release split** (`d3e2e-R2-9-9l/summary.tsv`, summed over the 8 requests): behind the copy
  lookup alone 2,752 of 5,117 right (53.8 %); behind the store 801 of 1,702 (47.1 %).
- **One guess** on one P106 (`llama-pipe-drafter-check --bench 300`, median): 6.20 ms at 2,048 tokens of context and
  7.29 ms at 8,192 for the release split (an 8-row window, as the check derives it from `--split 6,6,6,6,6,6,6,5,2`);
  5.34 / 5.90 ms for the even split (6 rows); 5.30 / 5.86 ms for six cards (4 rows).
- **Identity on the rig:** all 80 answers equal (T 0 against no guessing, T 1 among the pipelined configurations).

The exactness set (`run/`), all correctness only:

| Check | Script | Result |
|---|---|---|
| C++ head against `Drafter.forward`, random weights, heads A, A fused, B, C, embedding table on the card and in host memory | `cases.sh`, `check.sh` step 1 | largest logit difference 0.26–0.29 % of the logits' standard deviation with f16 weights (3.2 % for C), 5.7 % with q8_0; the same top id at every step |
| A real head on real Q51 data | `cases3.sh` with `REAL_HEAD` (`check_export.py real`) | f16 0.43 % of the sd; all q8_0 0.16 rms logit error; q8_0 with an f16 output head 0.035 rms (the export default) |
| Pipelined decoding with the head against plain decoding, depths 0–4, T 0 and T 1 on the card | `check.sh` step 2 | equal in tokens, per-token logits hashes, KV state and probe in all 8 runs |
| Plain decoding with and without the head loaded; llama-server with the head against no guessing | `combo34.sh` | identical; 7 of 7 requests at depths 3 and 5 |
| Windows derived from the split, 3–12 cards and a spare card | `cases2.sh`, `numcheck2.sh` | 14 of 14 |
| The same after the window correction for tokens older than the window | `cases3.sh`, `numcheck3.sh`, `check3.sh` | 14 of 14; one near-tie flips on the CPU (top two logits 0.0005 apart, `case_top.py`); identity with the depth-general head on 3 virtual devices and with K2h-r16 alone |
| llama-server, staircase with the store | `check_ss2.sh`, `check_ss2b.sh` | found a crash in `launch()` when long copy or store runs left fewer pending commits than the flush loop took; after the fix (`460160e93`) 10 of 10 and 13 of 13 identical (512-token rewrites included) |

## Files

| File | What |
|---|---|
| `d3e2e-R2-9-9l/summary.tsv` | the release split (nine cards, light last card), R2-9: per request tok/s and speed-up for no guessing (`d0`), copy + continuations + store (`d3s`), store then the head (`d3ds`), copy then the head (`d3d`); guesses right / wrong / from the store and the head's right / guesses (`d…/…`); identity |
| `d3e2e-R2-9-9e/summary.tsv` | nine cards, even split, R2-9 (`d0`, `d3s`, `d3ds`) |
| `d3e2e-R2-gen-6c/summary.tsv` | six cards, R2-gen (`d0`, `d3s`, `d3ds`) |
| `run/build.sh`, `run/rigbuild.sh` | the test build (sm_70, sm_86 and compute_61 PTX) and the rig tarball of the drafter branch |
| `run/cases.sh`, `cases2.sh`, `cases3.sh` | random-weight heads and reference cases from `bench/staircase_drafter/check_export.py` |
| `run/check.sh`, `numcheck2.sh`, `numcheck3.sh`, `check3.sh`, `combo34.sh` | the exactness set above |
| `run/check_ss.sh`, `check_ss2.sh`, `check_ss2b.sh`, `check_ss4.sh` | llama-server with the store and the head (`check_ss.sh` and `check_ss4.sh` are the single-card variants) |
| `run/export_heads.sh` | trained heads into the engine's side-file format (`export_gguf.py`) with md5s |
| `run/case_top.py` | the top logits per step of a reference case, to tell a near-tie from a real mismatch |
| `run/rig_d3_sources.sh` | rig test, never run: what each guess source is worth (copy alone, + continuations, + store, copy then the head, all) |
| `run/rig_d3_q8.sh` | rig test, never run: Flint Q8_0 at 262k with the head behind the store, the head's card picked from free memory |
| `run/rig_d3_place.sh` | rig test, never run: the head on the light last card, on a spare card, and on card 0 with the table in host memory |

The `run/` jobs use a check directory `D` laid out as: `src/` (the fork tree), `out/build/bin` and `out/cuda12-libs`
(`build.sh`'s build and the CUDA runtime libraries for the container), `py/` (a copy of
[bench/staircase_drafter](../../../bench/staircase_drafter/README.md)), `pipespec/` (a copy of
[bench/pipespec](../../../bench/pipespec/)), `cases/`, `cases3/`, `heads/`, `check/` (logs) and `rc4ss/` (the rc4 tree
with the staircase and the store, built the same way). `MODELS` holds the Flint GGUFs (`flint-imatU-Q4_0-p4648b.gguf`,
`flint-q113-K2h-r16.gguf`), `TRAIN` the drafter's training folder (`heads/`, `vocab.npy`). The rig tests take
`RIG` (the rig's ssh alias, default `rig`), the build tarball `T` in `SRC`, the head files in `HEADS`, the store
`STORE_SRC`, and the no-guessing answers of an earlier run as `REF9` / `REF6`.

## Made on 2026-10-24 and 2026-10-25 with

The served run, per split, through [bench/pipespec/rig_server.sh](../../../bench/pipespec/) (one server per
configuration), and the guess cost through the fork's `llama-pipe-drafter-check`:

```bash
ssh "$RIG" "env BIN=$B MODEL=/mnt/models/flint/flint-q113-K2h-r16.gguf TAG=d3e2e-R2-9-9l GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 DEPTHS='0 3s 3ds' GUESS=cont CTX=8192 MAX_TOKENS=512 TOOL_MAX_TOKENS=1536 SEED=1 \
    STORE=<store>.pstore DRAFT=/opt/llama-builds/q20/R2-9.gguf EXTRA_ARGS='-lv 4' bash /tmp/pipespec/rig_server.sh"
# then DEPTHS='3d' APPEND=1 for the head alone; nine even: TS=6,6,6,6,5,5,5,5,5; six cards: GPUS=0,1,2,3,4,5, no TS
ssh "$RIG" "CUDA_VISIBLE_DEVICES=0 $B/llama-pipe-drafter-check /opt/llama-builds/q20/R2-9.gguf --bench 300 \
    --split 6,6,6,6,6,6,6,5,2 --commits 1 --ctx 8192"
```

The requests are [bench/pipespec](../../../bench/pipespec/)'s prompts (this repository's own texts). The tables hold
counts and speeds only; the answers stay out.
