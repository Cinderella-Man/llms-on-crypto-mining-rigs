bench trace: 30 main passes for 5 tests x (1 + 5); device patterns [((0, 1, 2, 3, 4, 5, 6, 7, 8), 30)]
draft trace: 511 passes from line 2967, 104 main passes; draft device patterns [((4, 8), 407)]; drafts per step [(4, 101), (1, 3)]

| part | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | verify in the draft request (median) | draft passes per step (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| routed experts (Q2_0) | 27.2 | 29.9 | 44.3 | 58.3 | 70.7 | 45.1 | 2.9 |
| shared expert | 4.4 | 5.7 | 7.3 | 9.6 | 9.2 | 7.7 | 0.3 |
| BF16 hyper-connections | 44.6 | 63.3 | 83.4 | 101.8 | 123.0 | 77.2 | 2.9 |
| router (F32) | 2.5 | 3.4 | 4.5 | 3.2 | 6.0 | 2.7 | 0.2 |
| attention and recurrent projections | 24.5 | 38.1 | 50.9 | 69.2 | 67.9 | 52.4 | 1.7 |
| recurrent state (delta net, conv) | 4.3 | 4.8 | 5.3 | 5.8 | 6.2 | 9.8 | 0.0 |
| flash attention | 0.7 | 0.9 | 1.2 | 1.2 | 1.7 | 0.9 | 0.2 |
| output layer | 3.5 | 3.5 | 3.6 | 3.6 | 3.6 | 10.4 | 18.5 |
| token embeddings | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| other ops without weights | 26.3 | 30.5 | 30.9 | 31.1 | 30.8 | 32.1 | 2.8 |
| other weights | 1.6 | 2.0 | 2.7 | 3.2 | 3.9 | 3.0 | 1.3 |
| **GPU total** | **139.6** | **182.0** | **234.1** | **286.9** | **323.0** | **241.5** | **30.9** |
| wall, profiler on | 178.7 | 220.2 | 279.8 | 339.2 | 381.9 |  |  |
| wall, profiler off | 136.0 | 185.8 | 244.6 | 305.4 | 359.9 |  |  |

drafting request: 101 verify passes; step wall time (profiled, host clock) median 327.0 ms, mean 427.4; prompt pass 2642.3 ms GPU
reply timings: {'prompt_n': 79, 'prompt_ms': 8080.728, 'prompt_per_token_ms': 102.28769620253165, 'prompt_per_second': 9.776346883597617, 'predicted_n': 312, 'predicted_ms': 43018.428, 'predicted_per_token_ms': 138.3229196141479, 'predicted_per_second': 7.229459895652162, 'draft_n': 303, 'draft_n_accepted': 210}
