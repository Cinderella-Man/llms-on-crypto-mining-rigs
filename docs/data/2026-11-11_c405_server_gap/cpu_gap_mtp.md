# /c/smoke/on-mtp.trace.jsonl

- lines: 211 (9 without a draft, 202 generation steps); drafted 597, accepted 0; drafting passes beyond the drafted tokens 9 (in 6 steps); line writes: median 50 us
- typical steps: 187 of 202 (the commonest of 1 mark sequences among the steps that drafted 3 tokens, 3 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.002 | 0.002 | 0.001 | 0.003 |  |
| s.post → s.shift | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.shift → s.sel | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.drf.in → mtp.drf | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.002 | 0.002 | 0.002 | 0.003 |  |
| mtp.dp → d.dec | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec → d.dec.bal | 0.004 | 0.004 | 0.004 | 0.005 |  |
| d.dec.bal → d.dec.mem | 0.009 | 0.009 | 0.008 | 0.010 |  |
| d.dec.mem → d.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.out → d.ub.apply | 0.006 | 0.007 | 0.005 | 0.008 |  |
| d.ub.apply → d.ub.build | 0.080 | 0.085 | 0.078 | 0.100 |  |
| d.ub.build → d.ub.in | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.ub.in → d.ub.comp | 0.190 | 0.198 | 0.179 | 0.231 |  |
| d.ub.comp → d.ub.cp | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.ub.cp → d.dec.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end → mtp.dp.ret | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.dp.ret → d.sync(6) | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync(6) → d.sync.end(6) | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6) → mtp.dp.smp | 0.001 | 0.001 | 0.001 | 0.002 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync → d.sync.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end → mtp.dp.row | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| mtp.dp#2 → d.dec#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec#2 → d.dec.bal#2 | 0.004 | 0.004 | 0.003 | 0.005 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.007 | 0.008 | 0.007 | 0.009 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.ub.apply#2 → d.ub.reuse | 0.004 | 0.004 | 0.004 | 0.005 |  |
| d.ub.reuse → d.ub.in#2 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.184 | 0.190 | 0.174 | 0.218 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.000 | 0.000 | 0.000 | 0.001 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync#2 → d.sync.end#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp.row#2 → mtp.dp#3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| mtp.dp#3 → d.dec#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec#3 → d.dec.bal#3 | 0.003 | 0.003 | 0.002 | 0.004 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.006 | 0.006 | 0.005 | 0.007 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.003 | 0.003 | 0.002 | 0.004 |  |
| d.ub.apply#3 → d.ub.reuse#2 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.reuse#2 → d.ub.in#3 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.ub.in#3 → d.ub.comp#3 | 0.182 | 0.186 | 0.170 | 0.214 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#3 → mtp.dp.ret#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.dp.ret#3 → d.sync(6)#3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync(6)#3 → d.sync.end(6)#3 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync.end(6)#3 → mtp.dp.smp#3 | 0.000 | 0.000 | 0.000 | 0.001 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#3 → d.sync#3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync#3 → d.sync.end#3 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| d.sync.end#3 → mtp.dp.row#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.000 | 0.000 | 0.000 | 0.001 | back in the server |
| s.drf.out → s.drf.end | 0.016 | 0.016 | 0.015 | 0.017 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.007 | 0.007 | 0.006 | 0.008 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.000 | the generation checkpoint check |
| s.ckgen → s.bat | 0.000 | 0.000 | 0.000 | 0.001 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.000 | 0.000 | 0.000 | 0.001 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.001 | 0.001 | 0.001 | 0.002 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.000 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.002 | 0.002 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.001 | 0.001 | 0.001 | 0.002 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.004 | 0.004 | 0.003 | 0.005 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.009 | 0.010 | 0.008 | 0.012 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.001 | 0.000 | 0.001 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.011 | 0.011 | 0.009 | 0.015 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.003 | 0.003 | 0.003 | 0.004 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.003 | 0.003 | 0.002 | 0.004 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 0.834 | 0.869 | 0.799 | 1.005 |  |
| v.ub.comp → v.ub.cp | 0.002 | 0.002 | 0.001 | 0.003 |  |
| v.ub.cp → v.dec.end | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.dec.end → s.dec.ret | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.dec.ret → v.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 0.000 | 0.000 | 0.000 | 0.000 |  |
| v.sync.end → s.dec.out | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.dec.out → s.dec.end | 0.016 | 0.016 | 0.015 | 0.017 |  |
| s.dec.end → s.mpd | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.cu.in → mtp.cu | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.cu → v.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync#2 → v.sync.end#2 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| v.sync.end#2 → mtp.cu.dec | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.cu.dec → d.dec#4 | 0.002 | 0.002 | 0.002 | 0.003 |  |
| d.dec#4 → d.dec.bal#4 | 0.008 | 0.008 | 0.007 | 0.009 |  |
| d.dec.bal#4 → d.dec.mem#4 | 0.009 | 0.010 | 0.009 | 0.011 |  |
| d.dec.mem#4 → d.dec.out#4 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.out#4 → d.ub.apply#4 | 0.008 | 0.008 | 0.006 | 0.010 |  |
| d.ub.apply#4 → d.ub.build#2 | 0.087 | 0.092 | 0.084 | 0.108 |  |
| d.ub.build#2 → d.ub.in#4 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.ub.in#4 → d.ub.comp#4 | 0.347 | 0.367 | 0.334 | 0.417 |  |
| d.ub.comp#4 → d.ub.cp#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.ub.cp#4 → d.dec.end#4 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.dec.end#4 → mtp.cu.ret | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.cu.ret → v.sync(4) | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync(4) → v.sync.end(4) | 0.001 | 0.001 | 0.001 | 0.002 |  |
| v.sync.end(4) → mtp.cu.end | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.cu.end → s.cu.out | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.015 | 0.015 | 0.014 | 0.016 |  |
| s.cu.end → s.pd | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.pd → s.acc | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.acc → s.acc.cl | 0.004 | 0.004 | 0.003 | 0.006 |  |
| s.acc.cl → v.sync(6) | 0.000 | 0.000 | 0.000 | 0.001 |  |
| v.sync(6) → v.sync.end(6) | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync.end(6) → s.acc.smp | 0.002 | 0.003 | 0.002 | 0.003 |  |
| s.acc.smp → s.acc.spec | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.acc.spec → s.rm | 0.015 | 0.016 | 0.014 | 0.018 |  |
| s.rm → s.tok | 0.001 | 0.002 | 0.001 | 0.002 |  |
| s.tok → s.tg | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.tg → s.upd.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| **the line (s.upd → s.upd.end)** | 2.138 | 2.232 | 2.079 | 2.556 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| d.sync.end#3 → mtp.dp.row#3 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp.row#3 → mtp.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.000 | 0.000 | 0.000 | 0.001 | back in the server |
| s.drf.out → s.drf.end | 0.016 | 0.016 | 0.015 | 0.017 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.007 | 0.007 | 0.006 | 0.008 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.000 | the generation checkpoint check |
| s.ckgen → s.bat | 0.000 | 0.000 | 0.000 | 0.001 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.000 | 0.000 | 0.000 | 0.001 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.001 | 0.001 | 0.001 | 0.002 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.000 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.002 | 0.002 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.001 | 0.001 | 0.001 | 0.002 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.004 | 0.004 | 0.003 | 0.005 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.009 | 0.010 | 0.008 | 0.012 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.001 | 0.001 | 0.000 | 0.001 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.011 | 0.011 | 0.009 | 0.015 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.003 | 0.003 | 0.003 | 0.004 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.003 | 0.003 | 0.002 | 0.004 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 0.060 | 0.061 | 0.055 | 0.070 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.052 | 0.054 | 0.049 | 0.066 | the line's write, the task loop |

check: OK (0 problems)
