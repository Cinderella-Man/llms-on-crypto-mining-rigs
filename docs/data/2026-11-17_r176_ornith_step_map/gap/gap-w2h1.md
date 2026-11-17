# <OUT>/st-w2h1.jsonl

- lines: 567 (10 without a draft, 557 generation steps); drafted 557, accepted 491; drafting passes beyond the drafted tokens 0 (in 0 steps); line writes: median 165 us
- typical steps: 482 of 557 (the commonest of 4 mark sequences among the steps that drafted 1 tokens, 1 drafting passes); the first 3 after each prompt left out

## Waterfall (ms between consecutive marks; the rows add up to the line)

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd → s.post | 0.006 | 0.006 | 0.006 | 0.007 |  |
| s.post → s.shift | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.shift → s.sel | 0.023 | 0.023 | 0.004 | 0.028 |  |
| s.sel → s.drf | 0.000 | 0.000 | 0.000 | 0.000 |  |
| s.drf → s.drf.in | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.drf.in → mtp.drf | 0.001 | 0.001 | 0.000 | 0.001 |  |
| mtp.drf → mtp.dp | 0.010 | 0.010 | 0.009 | 0.011 |  |
| mtp.dp → d.dec | 0.006 | 0.007 | 0.006 | 0.007 |  |
| d.dec → d.dec.bal | 0.012 | 0.012 | 0.011 | 0.013 |  |
| d.dec.bal → d.dec.mem | 0.029 | 0.028 | 0.025 | 0.031 |  |
| d.dec.mem → d.dec.out | 0.004 | 0.004 | 0.003 | 0.004 |  |
| d.dec.out → d.ub.apply | 0.003 | 0.003 | 0.002 | 0.003 |  |
| d.ub.apply → d.ub.build | 1.550 | 1.700 | 1.300 | 2.103 |  |
| d.ub.build → d.ub.in | 0.040 | 0.046 | 0.016 | 0.075 |  |
| d.ub.in → d.ub.comp | 0.779 | 0.889 | 0.698 | 0.799 |  |
| d.ub.comp → d.ub.cp | 0.024 | 0.025 | 0.023 | 0.025 |  |
| d.ub.cp → d.dec.end | 0.001 | 0.001 | 0.000 | 0.001 |  |
| d.dec.end → mtp.dp.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.dp.ret → d.sync(6) | 0.001 | 0.001 | 0.001 | 0.002 |  |
| d.sync(6) → d.sync.end(6) | 4.441 | 4.521 | 4.090 | 4.966 |  |
| d.sync.end(6) → mtp.dp.smp | 0.006 | 0.006 | 0.005 | 0.006 | the drafted id and candidates read, the draft's host sampler |
| mtp.dp.smp → d.sync | 0.000 | 0.000 | 0.000 | 0.001 |  |
| d.sync → d.sync.end | 0.015 | 0.015 | 0.015 | 0.016 |  |
| d.sync.end → mtp.dp.row | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.row → mtp.drf.end | 0.001 | 0.001 | 0.000 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.002 | 0.002 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.001 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.095 | 0.095 | 0.092 | 0.098 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.003 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.014 | 0.014 | 0.013 | 0.015 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.035 | 0.036 | 0.032 | 0.038 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.008 | 0.008 | 0.008 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.004 | 0.004 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.041 | 0.042 | 0.040 | 0.043 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.023 | 0.043 | 0.012 | 0.073 | the graph's inputs set (host side) |
| v.ub.in → v.ub.comp | 4.170 | 3.630 | 2.670 | 4.235 |  |
| v.ub.comp → v.ub.cp | 0.030 | 0.030 | 0.029 | 0.031 |  |
| v.ub.cp → v.dec.end | 0.003 | 0.003 | 0.003 | 0.004 |  |
| v.dec.end → s.dec.ret | 0.005 | 0.005 | 0.005 | 0.006 |  |
| s.dec.ret → v.sync | 0.001 | 0.001 | 0.000 | 0.001 |  |
| v.sync → v.sync.end | 97.132 | 101.535 | 95.189 | 107.917 |  |
| v.sync.end → s.dec.out | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.dec.out → s.dec.end | 0.003 | 0.003 | 0.002 | 0.004 |  |
| s.dec.end → s.mpd | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.mpd → s.cu.in | 0.001 | 0.001 | 0.000 | 0.001 |  |
| s.cu.in → mtp.cu | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu → mtp.cu.dec | 0.100 | 0.102 | 0.097 | 0.106 |  |
| mtp.cu.dec → d.dec#2 | 0.005 | 0.006 | 0.005 | 0.006 |  |
| d.dec#2 → d.dec.bal#2 | 0.016 | 0.016 | 0.015 | 0.017 |  |
| d.dec.bal#2 → d.dec.mem#2 | 0.034 | 0.035 | 0.030 | 0.037 |  |
| d.dec.mem#2 → d.dec.out#2 | 0.007 | 0.007 | 0.006 | 0.007 |  |
| d.dec.out#2 → d.ub.apply#2 | 0.004 | 0.004 | 0.003 | 0.005 |  |
| d.ub.apply#2 → d.ub.build#2 | 1.606 | 1.599 | 1.504 | 1.677 |  |
| d.ub.build#2 → d.ub.in#2 | 0.039 | 0.044 | 0.012 | 0.073 |  |
| d.ub.in#2 → d.ub.comp#2 | 0.862 | 0.798 | 0.698 | 0.891 |  |
| d.ub.comp#2 → d.ub.cp#2 | 0.003 | 0.003 | 0.003 | 0.004 |  |
| d.ub.cp#2 → d.dec.end#2 | 0.002 | 0.002 | 0.001 | 0.002 |  |
| d.dec.end#2 → mtp.cu.ret | 0.002 | 0.002 | 0.001 | 0.002 |  |
| mtp.cu.ret → mtp.cu.end | 0.131 | 0.129 | 0.117 | 0.139 |  |
| mtp.cu.end → s.cu.out | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.cu.out → s.cu.end | 0.000 | 0.000 | 0.000 | 0.001 |  |
| s.cu.end → s.pd | 0.001 | 0.001 | 0.001 | 0.001 |  |
| s.pd → s.acc | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc → s.acc.cl | 0.013 | 0.013 | 0.011 | 0.015 |  |
| s.acc.cl → v.sync(12) | 0.002 | 0.002 | 0.001 | 0.002 |  |
| v.sync(12) → v.sync.end(12) | 0.227 | 0.230 | 0.223 | 0.242 |  |
| v.sync.end(12) → s.acc.smp | 0.001 | 0.001 | 0.001 | 0.002 |  |
| s.acc.smp → s.acc.spec | 0.002 | 0.002 | 0.001 | 0.002 |  |
| s.acc.spec → s.rm | 0.181 | 0.185 | 0.180 | 0.193 |  |
| s.rm → s.tok | 0.006 | 0.107 | 0.005 | 0.007 |  |
| s.tok → s.tok#2 | 0.001 | 0.068 | 0.001 | 0.002 |  |
| s.tok#2 → s.tg | 0.000 | 0.007 | 0.000 | 0.001 |  |
| s.tg → s.upd.end | 0.001 | 0.001 | 0.001 | 0.002 |  |
| **the line (s.upd → s.upd.end)** | 121.937 | 116.142 | 107.707 | 123.843 |  |

## The gap: the last drafting pass's wait → the verify pass's inputs set

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| draft card's work end → d.sync.end (last pass) | 0.306 | 0.307 | 0.301 | 0.314 | the drafted id's way back, the host's wake |
| d.sync.end → mtp.dp.row | 0.004 | 0.004 | 0.003 | 0.005 |  |
| mtp.dp.row → mtp.drf.end | 0.001 | 0.001 | 0.000 | 0.001 | the drafter's end (shortlist and carry reset, length check) |
| mtp.drf.end → s.drf.out | 0.002 | 0.002 | 0.001 | 0.002 | back in the server |
| s.drf.out → s.drf.end | 0.000 | 0.000 | 0.000 | 0.001 | the yield's end: the queue's worker thread stops |
| s.drf.end → s.ckpt | 0.095 | 0.095 | 0.092 | 0.098 | draft bookkeeping, the draft context's seq_rm, checkpoints |
| s.ckpt → s.ckgen | 0.000 | 0.000 | 0.000 | 0.001 | the generation checkpoint check |
| s.ckgen → s.bat | 0.002 | 0.002 | 0.001 | 0.002 | the sampled and drafted tokens into the batch |
| s.bat → s.pre.end | 0.002 | 0.002 | 0.001 | 0.002 | the prompt loop over the slots, the rest of pre_decode |
| s.pre.end → s.slots.end | 0.006 | 0.006 | 0.005 | 0.006 | TTS and realtime slots, LoRA, embeddings flag, taper |
| s.slots.end → s.render | 0.000 | 0.000 | 0.000 | 0.001 | the batch rendered for llama_process |
| s.render → s.dec | 0.000 | 0.000 | 0.000 | 0.001 | into decode() |
| s.dec → s.dec.y | 0.000 | 0.000 | 0.000 | 0.001 | metrics, output rows, screened head, head bound, taper hint |
| s.dec.y → s.dec.in | 0.001 | 0.001 | 0.001 | 0.002 | the yield's start: the queue's worker thread woken |
| s.dec.in → v.dec | 0.004 | 0.004 | 0.003 | 0.005 | llama_process into llama_context::decode |
| v.dec → v.dec.bal | 0.014 | 0.014 | 0.013 | 0.015 | pipe drain, batch checks, the batch allocator |
| v.dec.bal → v.dec.mem | 0.035 | 0.036 | 0.032 | 0.038 | shortlist, scheduler reserve, memory update, the memory slot (init_batch) |
| v.dec.mem → v.dec.out | 0.008 | 0.008 | 0.008 | 0.009 | output buffer, the samplers' begin, the cross-context waits |
| v.dec.out → v.ub.apply | 0.004 | 0.004 | 0.004 | 0.005 | the ubatch, the memory context applied |
| v.ub.apply → v.ub.reuse | 0.041 | 0.042 | 0.040 | 0.043 | graph parameters, the graph reused |
| v.ub.reuse → v.ub.in | 0.023 | 0.043 | 0.012 | 0.073 | the graph's inputs set (host side) |
| **host: last d.sync.end → v.ub.in** | 0.291 | 0.265 | 0.226 | 0.302 |  |
| v.ub.in → first main card's graph_compute entry | 0.339 | 0.337 | 0.164 | 0.350 | the scheduler's input copies for that card |
| **the gap, as S37 counts it** | 0.935 | 0.909 | 0.696 | 0.961 |  |

## Between lines

| from → to | median | mean | p10 | p90 | what runs there |
|---|---:|---:|---:|---:|---:|
| s.upd.end → next line's s.upd | 0.175 | 0.168 | 0.137 | 0.187 | the line's write, the task loop |
