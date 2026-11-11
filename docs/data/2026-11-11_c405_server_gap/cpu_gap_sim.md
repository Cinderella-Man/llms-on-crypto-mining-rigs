# /c/smoke/on-sim.trace.jsonl

- lines: 89 (28 without a draft, 61 generation steps); drafted 180, accepted 118; drafting passes beyond the drafted tokens 0 (in 0 steps); line writes: median 45 us
- typical steps: 32 of 61 (the commonest of 1 mark sequences among the steps that drafted 3 tokens, 0 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.post → s.shift | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.shift → d.sync | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.sync → d.sync.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end → s.sel | 0.010 | 0.011 | 0.009 | 0.014 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.drf.in → d.dec | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec → d.dec.bal | 0.003 | 0.003 | 0.002 | 0.004 |  |
| d.dec.bal → d.dec.mem | 0.008 | 0.009 | 0.007 | 0.010 |  |
| d.dec.mem → d.dec.out | 0.000 | 0.001 | 0.000 | 0.001 |  |
| d.dec.out → d.ub.apply | 0.006 | 0.006 | 0.005 | 0.007 |  |
| d.ub.apply → d.ub.build | 0.115 | 0.119 | 0.111 | 0.139 |  |
| d.ub.build → d.ub.in | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.in → d.ub.comp | 0.314 | 0.325 | 0.306 | 0.350 |  |
| d.ub.comp → d.ub.cp | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.ub.cp → d.dec.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end → d.sync(6) | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6) → d.sync.end(6) | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6) → d.dec#2 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec#2 → d.dec.bal#2 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.006 | 0.006 | 0.006 | 0.007 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.003 | 0.003 | 0.003 | 0.003 |  |
| d.ub.apply#2 → d.ub.reuse | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.ub.reuse → d.ub.in#2 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.302 | 0.309 | 0.295 | 0.335 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#2 → d.sync(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6)#2 → d.dec#3 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec#3 → d.dec.bal#3 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.295 | 0.298 | 0.284 | 0.312 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.dec.end#3 → d.sync(6)#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6)#3 → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.drf.out → s.drf.end | 0.017 | 0.017 | 0.016 | 0.018 | the yield's end: the queue's worker thread stops |
| s.drf.end → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end#2 → v.sync | 0.029 | 0.029 | 0.024 | 0.034 |  |
| v.sync → v.sync.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| v.sync.end → s.ckpt | 0.008 | 0.009 | 0.007 | 0.012 |  |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.000 | the generation checkpoint check |
| s.ckgen → s.bat | 0.001 | 0.001 | 0.000 | 0.001 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.001 | 0.001 | 0.000 | 0.001 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.001 | 0.001 | 0.001 | 0.002 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.000 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.000 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.000 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.002 | 0.002 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.001 | 0.001 | 0.001 | 0.002 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.003 | 0.003 | 0.003 | 0.004 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.010 | 0.011 | 0.008 | 0.013 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.000 | 0.000 | 0.000 | 0.001 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.008 | 0.008 | 0.006 | 0.010 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.003 | 0.003 | 0.002 | 0.004 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.003 | 0.003 | 0.002 | 0.004 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 0.518 | 0.526 | 0.509 | 0.561 |  |
| v.ub.comp → v.ub.cp | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.ub.cp → v.dec.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.dec.end → s.dec.ret | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.ret → v.sync#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| v.sync#2 → v.sync.end#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync.end#2 → s.dec.out | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.dec.out → s.dec.end | 0.016 | 0.016 | 0.015 | 0.017 |  |
| s.dec.end → s.mpd | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.cu.in → d.dec#4 | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.dec#4 → d.dec.bal#4 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.008 | 0.008 | 0.007 | 0.009 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.006 | 0.006 | 0.005 | 0.008 |  |
| d.ub.apply#4 → d.ub.build#2 | 0.122 | 0.126 | 0.115 | 0.141 |  |
| d.ub.build#2 → d.ub.in#4 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.468 | 0.482 | 0.461 | 0.514 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#4 → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.016 | 0.016 | 0.015 | 0.017 |  |
| s.cu.end → s.pd | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.acc → s.acc.cl | 0.002 | 0.003 | 0.002 | 0.004 |  |
| s.acc.cl → v.sync(24) | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync(24) → v.sync.end(24) | 0.007 | 0.007 | 0.006 | 0.008 |  |
| v.sync.end(24) → s.acc.smp | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.acc.smp → s.acc.spec | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.acc.spec → s.rm | 0.012 | 0.012 | 0.011 | 0.012 |  |
| s.rm → s.tok | 0.002 | 0.002 | 0.001 | 0.003 |  |
| s.tok → s.tok#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.tok#2 → s.tok#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.tok#3 → s.tok#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.tok#4 → s.tg | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.tg → s.upd.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| **the line (s.upd → s.upd.end)** | 2.389 | 2.441 | 2.333 | 2.665 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| d.sync.end(6)#3 → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.drf.out → s.drf.end | 0.017 | 0.017 | 0.016 | 0.018 | the yield's end: the queue's worker thread stops |
| s.drf.end → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end#2 → v.sync | 0.029 | 0.029 | 0.024 | 0.034 |  |
| v.sync → v.sync.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| v.sync.end → s.ckpt | 0.008 | 0.009 | 0.007 | 0.012 |  |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.000 | the generation checkpoint check |
| s.ckgen → s.bat | 0.001 | 0.001 | 0.000 | 0.001 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.001 | 0.001 | 0.000 | 0.001 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.001 | 0.001 | 0.001 | 0.002 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.000 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.000 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.000 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.002 | 0.002 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.001 | 0.001 | 0.001 | 0.002 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.003 | 0.003 | 0.003 | 0.004 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.010 | 0.011 | 0.008 | 0.013 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.000 | 0.000 | 0.000 | 0.001 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.008 | 0.008 | 0.006 | 0.010 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.003 | 0.003 | 0.002 | 0.004 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.003 | 0.003 | 0.002 | 0.004 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 0.087 | 0.091 | 0.081 | 0.108 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.052 | 0.053 | 0.048 | 0.063 | the line's write, the task loop |

check: OK (0 problems)
