# <OUT>/st-w3h1.jsonl

- lines: 421 (8 without a draft, 413 generation steps); drafted 821, accepted 637; drafting passes beyond the drafted tokens 5 (in 5 steps); line writes: median 208 us
- typical steps: 280 of 413 (the commonest of 4 mark sequences among the steps that drafted 2 tokens, 2 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.006 | 0.007 | 0.005 | 0.008 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.shift → s.sel | 0.026 | 0.025 | 0.004 | 0.028 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.010 | 0.010 | 0.009 | 0.010 |  |
| mtp.dp → d.dec | 0.006 | 0.006 | 0.006 | 0.007 |  |
| d.dec → d.dec.bal | 0.012 | 0.012 | 0.011 | 0.013 |  |
| d.dec.bal → d.dec.mem | 0.029 | 0.029 | 0.025 | 0.031 |  |
| d.dec.mem → d.dec.out | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply → d.ub.build | 2.868 | 2.249 | 1.383 | 3.036 |  |
| d.ub.build → d.ub.in | 0.074 | 0.048 | 0.016 | 0.076 |  |
| d.ub.in → d.ub.comp | 0.784 | 0.930 | 0.701 | 0.802 |  |
| d.ub.comp → d.ub.cp | 0.025 | 0.026 | 0.024 | 0.027 |  |
| d.ub.cp → d.dec.end | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end → mtp.dp.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.dp.ret → d.sync(6) | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.sync(6) → d.sync.end(6) | 4.838 | 4.507 | 4.066 | 4.908 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.006 | 0.005 | 0.006 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.004 | 0.004 | 0.003 | 0.004 |  |
| mtp.dp.row → mtp.dp#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| mtp.dp#2 → d.dec#2 | 0.006 | 0.006 | 0.006 | 0.007 |  |
| d.dec#2 → d.dec.bal#2 | 0.012 | 0.012 | 0.012 | 0.013 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.031 | 0.030 | 0.028 | 0.033 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply#2 → d.ub.reuse | 0.034 | 0.034 | 0.033 | 0.035 |  |
| d.ub.reuse → d.ub.in#2 | 0.066 | 0.041 | 0.010 | 0.069 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.718 | 0.816 | 0.652 | 0.780 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.023 | 0.024 | 0.022 | 0.024 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end#2 → mtp.dp.ret#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.ret#2 → d.sync(6)#2 | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.sync(6)#2 → d.sync.end(6)#2 | 5.061 | 4.672 | 4.213 | 5.088 |  |
| d.sync.end(6)#2 → mtp.dp.smp#2 | 0.004 | 0.004 | 0.003 | 0.004 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp#2 → d.sync#2 | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync#2 → d.sync.end#2 | 0.015 | 0.015 | 0.014 | 0.015 |  |
| d.sync.end#2 → mtp.dp.row#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.row#2 → mtp.drf.end | 0.001 | 0.001 | 0.001 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.095 | 0.096 | 0.092 | 0.097 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.004 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.015 | 0.015 | 0.013 | 0.016 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.039 | 0.039 | 0.034 | 0.042 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.008 | 0.008 | 0.007 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.004 | 0.004 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.040 | 0.040 | 0.036 | 0.042 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.076 | 0.048 | 0.013 | 0.078 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 5.122 | 4.429 | 2.971 | 5.309 |  |
| v.ub.comp → v.ub.cp | 0.040 | 0.042 | 0.039 | 0.042 |  |
| v.ub.cp → v.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| v.dec.end → s.dec.ret | 0.005 | 0.005 | 0.004 | 0.006 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 141.952 | 134.777 | 124.641 | 144.163 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.102 | 0.103 | 0.099 | 0.107 |  |
| mtp.cu.dec → d.dec#3 | 0.006 | 0.006 | 0.005 | 0.007 |  |
| d.dec#3 → d.dec.bal#3 | 0.017 | 0.018 | 0.017 | 0.018 |  |
| d.dec.bal#3 → d.dec.mem#3 | 0.036 | 0.037 | 0.032 | 0.039 |  |
| d.dec.mem#3 → d.dec.out#3 | 0.007 | 0.007 | 0.006 | 0.007 |  |
| d.dec.out#3 → d.ub.apply#3 | 0.005 | 0.005 | 0.004 | 0.006 |  |
| d.ub.apply#3 → d.ub.build#2 | 1.586 | 1.581 | 1.485 | 1.644 |  |
| d.ub.build#2 → d.ub.in#3 | 0.074 | 0.047 | 0.013 | 0.076 |  |
| d.ub.in#3 → d.ub.comp#3 | 1.027 | 0.915 | 0.771 | 1.044 |  |
| d.ub.comp#3 → d.ub.cp#3 | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.cp#3 → d.dec.end#3 | 0.001 | 0.002 | 0.001 | 0.002 |  |
| d.dec.end#3 → mtp.cu.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu.ret → mtp.cu.end | 0.190 | 0.188 | 0.177 | 0.196 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.002 | 0.003 |  |
| s.acc → s.acc.cl | 0.013 | 0.013 | 0.011 | 0.014 |  |
| s.acc.cl → v.sync(18) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(18) → v.sync.end(18) | 0.309 | 0.313 | 0.306 | 0.324 |  |
| v.sync.end(18) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.181 | 0.184 | 0.180 | 0.193 |  |
| s.rm → s.tok | 0.005 | 0.158 | 0.005 | 0.007 |  |
| s.tok → s.tok#2 | 0.001 | 0.175 | 0.001 | 0.003 |  |
| s.tok#2 → s.tok#3 | 0.001 | 0.112 | 0.000 | 0.002 |  |
| s.tok#3 → s.tg | 0.000 | 0.023 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 166.334 | 156.988 | 142.889 | 168.268 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.304 | 0.306 | 0.298 | 0.315 | the drafted id's way back, the host's wake |
| d.sync.end#2 → mtp.dp.row#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| mtp.dp.row#2 → mtp.drf.end | 0.001 | 0.001 | 0.001 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.001 | 0.001 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.000 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.095 | 0.096 | 0.092 | 0.097 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.001 | 0.001 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.004 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.015 | 0.015 | 0.013 | 0.016 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.039 | 0.039 | 0.034 | 0.042 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.008 | 0.008 | 0.007 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.004 | 0.004 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.040 | 0.040 | 0.036 | 0.042 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.076 | 0.048 | 0.013 | 0.078 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 0.299 | 0.272 | 0.219 | 0.315 |  |
| v.ub.in → first main card's graph_compute entry | 0.499 | 0.453 | 0.237 | 0.508 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 1.101 | 1.031 | 0.763 | 1.128 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.225 | 0.221 | 0.201 | 0.234 | the line's write, the task loop |
