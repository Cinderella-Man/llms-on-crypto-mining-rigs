# test-llama-archs on qwen4exp, seed 1, CPU: base `602691153` and new `c9c3134d4`

Each table is the new tree's. "Base" says whether the base tree's output of the same test is the same bytes
after its first line (a timestamp).

## `--mtp-kv-only`

Summary: all 5 test(s) passed, rc=0. Base: the same bytes.

| Model arch. | Config | KV, prompt | Drafts | KV, end | Nodes off/on | Full off/on | Rows off/on | Catch-up | Copies | Bytes | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | f16 fa | identical | identical | identical | 1458/531 | - | - | - | - | - | OK |
| qwen4exp | f16 | identical | identical | identical | 1503/549 | - | - | - | - | - | OK |
| qwen4exp | q8_0 fa | identical | identical | identical | 1566/585 | - | - | - | - | - | OK |
| qwen4exp | f16 fa top-k | identical | identical | identical | 1647/531 | - | - | - | - | - | OK |
| qwen4exp | f16 fa dist | identical | identical | identical | 2043/2043 | - | - | - | - | - | OK |

## `--mtp-kv-only with LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1`

Summary: all 5 test(s) passed, rc=0. Base: the same bytes.

| Model arch. | Config | KV, prompt | Drafts | KV, end | Nodes off/on | Full off/on | Rows off/on | Catch-up | Copies | Bytes | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | f16 fa | identical | identical | identical | 1629/657 | 1/1 | 574/574 | 9/9 | 18/9 | 31200/30376 | OK |
| qwen4exp | f16 | identical | identical | identical | 1674/675 | 1/1 | 574/574 | 9/9 | 18/9 | 31200/30376 | OK |
| qwen4exp | q8_0 fa | identical | identical | identical | 1737/711 | 1/1 | 574/574 | 9/9 | 18/9 | 31200/30376 | OK |
| qwen4exp | f16 fa top-k | identical | identical | identical | 1818/657 | 1/1 | 574/574 | 9/9 | 18/9 | 31200/30376 | OK |
| qwen4exp | f16 fa dist | identical | identical | identical | 2214/2214 | 1/1 | 574/574 | 9/9 | 18/18 | 31200/31200 | OK |

## `--mtp-graph-keep`

Summary: all 8 test(s) passed, rc=0. Base: the same bytes.

| Model arch. | Config | Prompt | Drafts | End | Short | Built, prompt | Built, steps | Compute KiB | Status |
|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | f16 fa | identical | identical | identical | off | 7/7 | 32/4 of 64 | 1411/2278 | OK |
| qwen4exp | f16 | identical | identical | identical | off | 7/7 | 32/4 of 64 | 1447/2339 | OK |
| qwen4exp | q8_0 fa | identical | identical | identical | off | 7/7 | 32/4 of 64 | 1411/2278 | OK |
| qwen4exp | f16 fa host | identical | identical | identical | off | 7/7 | 32/4 of 64 | 1411/2278 | OK |
| qwen4exp | f16 fa kv-only | identical | identical | identical | off | 7/7 | 32/4 of 64 | 1411/1713 | OK |
| qwen4exp | f16 fa short | identical | identical | identical | on | 7/7 | 32/5 of 64 | 1411/2358 | OK |
| qwen4exp | f16 fa carry | identical | identical | identical | off | 7/7 | 48/5 of 64 | 1411/2358 | OK |
| qwen4exp | f16 fa kv+sl+carry | identical | identical | identical | on | 7/7 | 48/7 of 64 | 1411/1953 | OK |

## `--mtp-replan`

Summary: all 4 test(s) passed, rc=0. Base: the same bytes.

| Model arch. | Config | Drafts | End | Short | Re-plans base p/s/t | Re-plans keep p/s/t | Re-plans plan p/s/t | Growths b/k/p | Plans | Compute KiB k/p | Re-plans p+c p/s/t | Re-plans w+c p/s/t | Re-plans whole p/s/t | Whole p+c/w+c/w | Growths c/wc/w | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | f16 short | identical | identical | on | 9/111/0 | 1/28/0 | 0/0/0 | 0/29/0 | 6 | 6188/6593 | 9/0/0 | 1/0/0 | 0/0/0 | 0/1/0 | 0/0/0 | OK keep's first: nodes 0 -> 193 |
| qwen4exp | f16 short kpool | identical | identical | on | 9/111/0 | 1/28/0 | 0/0/0 | 0/29/0 | 6 | 6132/6522 | 9/0/0 | 1/0/0 | 0/0/0 | 0/1/0 | 0/0/0 | OK keep's first: nodes 0 -> 202 |
| qwen4exp | f16 | identical | identical | off | 0/0/0 | 1/24/0 | 0/0/0 | 0/25/0 | 5 | 5873/6211 | 9/0/0 | 1/0/0 | 0/0/0 | 0/1/0 | 0/0/0 | OK keep's first: nodes 0 -> 183 |
| qwen4exp | q8_0 short | identical | identical | on | 9/111/0 | 1/28/0 | 0/0/0 | 0/29/0 | 6 | 6188/6593 | 9/0/0 | 1/0/0 | 0/0/0 | 0/1/0 | 0/0/0 | OK keep's first: nodes 0 -> 205 |

## `--mtp-catchup-overlap`

Summary: all 5 test(s) passed, rc=0. Base: the same bytes.

| Model arch. | Config | Checkpoint | Prompt | Drafts | End | Ckpt spec | Waited off/on | Status |
|---|---|---|---|---|---|---|---|---|
| qwen4exp | f16 fa | identical | identical | identical | identical | identical | 0/4 | OK |
| qwen4exp | f16 | identical | identical | identical | identical | identical | 0/4 | OK |
| qwen4exp | q8_0 fa | identical | identical | identical | identical | identical | 0/4 | OK |
| qwen4exp | f16 fa kv-only | identical | identical | identical | identical | identical | 0/4 | OK |
| qwen4exp | f16 fa host | identical | identical | identical | identical | identical | 0/4 | OK |

## `--mtp-plain-attn`

Summary: all 5 test(s) passed, rc=0. Base: not in the base tree (the new test).

| Model arch. | Config | Idx off/on | KV, prompt | Drafts | KV, end | Dev off/on | K-pool in | QSA ops | Status |
|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | qsa fa | 1/1 | identical | identical | identical | 0/0 | 84/57 | 60/24 | OK |
| qwen4exp | qsa fa sw | 1/1 | identical | identical | identical | 1/1 | 24/15 | 180/126 | OK |
| qwen4exp | plain fa | 0/0 | identical | identical | identical | 0/0 | 0/0 | 0/0 | OK |
| qwen4exp | plain | 0/0 | identical | identical | identical | 0/0 | 0/0 | 0/0 | OK |
| qwen4exp | plain fa sw | 0/0 | identical | identical | identical | 0/0 | 0/0 | 0/0 | OK |

## Every architecture (`test-llama-archs -s 1`)

New: Summary: all 153 test(s) passed, rc=0. Base: Summary: all 153 test(s) passed; its output the same bytes.
qwen4exp on the CPU: NMSE against the CPU OK (0.00e+00), round trip OK, mixed batch OK (2.38e-16).

## `--mtp-plain-attn` on the base's code (`602691153` with the new test file only)

rc=134 (134: aborted). The rows it printed; the third stops on the assertion:

| Model arch. | Config | Idx off/on | KV, prompt | Drafts | KV, end | Dev off/on | K-pool in | QSA ops | Status |
|---|---|---|---|---|---|---|---|---|---|
| qwen4exp | qsa fa | 1/1 | identical | identical | identical | 0/0 | 84/57 | 60/24 | OK |
| qwen4exp | qsa fa sw | 1/1 | identical | identical | identical | 1/1 | 24/15 | 180/126 | OK |
| qwen4exp | plain fa | ../../../src/ggml/src/ggml-backend.cpp:347: GGML_ASSERT(buffer) failed |
