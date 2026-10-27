# Q20: the staircase drafter's first data round and its licence audit (2026-10-24/25)

The first data round of the staircase drafter (Q20: the D1 pilot, M1–M3) generated 74.67 M answer tokens of Flint
answers, dumped their hidden states and trained the heads R1 and R2-*. On 2026-10-25 an audit of the prompt sources
excluded the whole round on licence grounds: it mixed in ShareGPT, UltraChat, Magicoder, NuminaMath-CoT and
LLM-written tool-call sets, and every head and store built from it is not for release. The method, results and the
audit's table are in [docs/drafter.md](../../drafter.md) (M0–M3, §6, changelog), the exact training commands in
[release/flint-v2.1/reproduce/drafter.md](../../../release/flint-v2.1/reproduce/drafter.md), and the registry row Q20 in
[quants/experiments.md](../../../quants/experiments.md). This folder holds the audit's code, which applies to any
answer set built with [bench/staircase_drafter](../../../bench/staircase_drafter/README.md); no prompt or answer text.

## Results of the audit

| Step | Script | Result |
|---|---|---|
| ShareGPT's share | `run/count_sharegpt.py` | 8,865 answers to ShareGPT prompts, 12.62 M of 74.67 M answer tokens (16.9 %) |
| Licences of the third-party packages behind the edit and agent prompts | `run/sitepkg_licences.py` | each package's licence from its installed METADATA |
| The per-package ruling | `run/strict_sitepkgs.py` | [strict-sitepkgs.md](../2026-10-24_q161_datastore/strict/strict-sitepkgs.md): 6,481 answers, 6.70 M tokens in; 244 answers, 0.29 M out (tqdm, pip's vendored certifi and resolvelib, two vendored Arrow headers) |
| Q51 hold-out | `run/q51_overlap.py` | 2,180 prompts share a 12-word span with a Q51 prompt (1,544 answered) |
| The clean lists | `run/count_strict.py` | core (my own edit and agent prompts on clean trees) 15,222 answers, 14.14 M tokens; core + OpenR1-Math 17,745, 17.91 M; the audit table of docs/drafter.md M3 |
| The core alone, with dump coverage | `run/core_ids.py` | the core per writer and how much of it was dumped |
| Training weights | [rec_weights.py](../../../bench/staircase_drafter/rec_weights.py) | edit, agent and code answers drawn twice per epoch (`train.py --rec-weights`) |
| Files Q51 is built from | [q51_sources.py](../../../bench/staircase_drafter/q51_sources.py) | the `{"source": name}` list that `clean_prompts.py --q51` and `check_sources.py --q51-sources` keep out of training prompts |
| GPU top-k in the dump | `run/gtcheck_eval.py` | same top-1 everywhere and top-20 overlap 1.0 against the host path, at 0.89× its speed: KEEP |

**Storage of the round:** the first dump (2,618 answers, 3.08 M answer tokens, 17-layer grid) took 117 GB; one later
dump wrote 171 GB in 50 minutes (57 MB/s of features and 13 MB/s of six-card pack). The round's feature dumps and
packs, with their archive copies, came to 4.4 TB (2.8 TB + 1.6 TB) on two HDDs.

## Files

| File | What |
|---|---|
| `run/count_sharegpt.py` | answers and answer tokens to ShareGPT prompts, per writer |
| `run/sitepkg_licences.py` | per site-packages package: answers, answer tokens, distribution and licence from METADATA |
| `run/strict_sitepkgs.py` | applies the per-package ruling, writes `strict-sitepkgs.md`, extends `strict-ids.txt` and writes `strict-ids-notool.txt` |
| `run/q51_overlap.py` | ids of prompts sharing a 12-word span with Q51 (all messages joined), counts per file and category |
| `run/count_strict.py` | every source rule in one place: `strict-ids-core.txt` and `strict-ids.txt`, answers and tokens per group, writer and category, and how many are dumped |
| `run/core_ids.py` | the earlier core list from `strict-ids.txt` (edit, agent and code), per writer, with dump coverage |
| `run/gtcheck_eval.py` | GPU-top-k dump against the default path on Q51: top-1, top-20 overlap, log-prob gap, throughput; ADOPT or KEEP |

The answer files are `gen-<label>.jsonl`: `A`–`C2s` Flint Q8_0, `E1`–`F2` Flint G3 (llama.cpp), `FV3` UD-Q4_K_XL,
`V1`/`V2` the makers' NVFP4 GGUF in vLLM.

## Made on 2026-10-25 with

CPU only, Python 3.12, in the drafter's work folder `W` (prompt files `prompts*.jsonl`, answer files, dump maps):

```bash
python3 -I bench/staircase_drafter/q51_sources.py $W/q51-sources.jsonl
python3 -I run/count_sharegpt.py $W
docker run --rm -v $W:/w -v $PWD/run:/run:ro python:3.12-slim \
    sh -c 'pip install -q huggingface_hub pyarrow && python3 -I /run/sitepkg_licences.py /w'
python3 -I run/strict_sitepkgs.py $W             # strict-sitepkgs.md, strict-ids.txt, strict-ids-notool.txt
python3 -I run/q51_overlap.py $W                 # q51-overlap-ids.txt
python3 -I run/count_strict.py $W                # strict-ids-core.txt, strict-ids.txt (final rules)
python3 -I bench/staircase_drafter/rec_weights.py $W/strict-ids.txt $W/rec-weights-code.txt \
    $W/prompts.jsonl $W/prompts-ext.jsonl $W/prompts-ext2.jsonl $W/prompts-tools.jsonl
python3 -I run/gtcheck_eval.py $W/test           # gt-a/gt-b dumps of Q51 and their logs
```

`count_strict.py` is the final version of the rules (it applies the package ruling itself); `strict_sitepkgs.py` and
`core_ids.py` are the earlier steps that wrote the lists the answer stores of
[Q161](../2026-10-24_q161_datastore/README.md) were built from. The prompt sets named here are third-party data sets;
the scripts read only their `source` field and token counts. The scripts are MIT, as the rest of this repository.
