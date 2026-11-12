# C811 checks (build machine, CPU only, 2026-11-12)

Every line of the checks driver (run/checks.sh) in order: the first run 17:36-17:56, the merged `pascal-p106-v4` (`e27947287`) 17:56:48-17:58:09.

| row | check | verdict | detail |
|---|---|---|---|
| start | pascal-p106-v4-b12246-c811 | OK | tip cfc657de5 (C811: GGML_CUDA_KEEPALIVE_EAGER, no first kernel launch waits for a keep-alive block (Q1417, from FN85)); files changed against the base:  6 files changed, 792 insertions(+), 16 deletions(-) |
| rig | build | OK | rc=0; llama.cpp-20261112-cfc657de5-pascal-p106-v4-b12246-c811.tar.gz md5 fcba7b406d934d102b6f7936e1982738, 741524473 B; commit: cfc657de5125bc72165633b92d94c7fee94aed4f (2026-11-12 17:33:42 +0100);avx instructions in llama-bench: 0;highest glibc symbol: GLIBC_2.38;; test-backend-ops md5 56514e791bb5ddc087b2a3949d4c1c50; test-cuda-keepalive md5 62c42e4562ad1ccbf2596543a0d2ed3b; llama-server md5 b9e64b34c0619b501c0efedc6370e360; 34 warning lines |
| names | switch list | OK | 861 names in the build's list; GGML_CUDA_KEEPALIVE_EAGER in it: 1, lines of bin/llama-server holding it: 2; holding ggml_backend_cuda_keepalive_counts: 2 |
| sass | llama-server sass against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| sass | llama-server ptx against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| cpu | build | OK | rc=0; commit cfc657de5125bc72165633b92d94c7fee94aed4f C811: GGML_CUDA_KEEPALIVE_EAGER, no first kernel launch waits for a keep-alive block (Q1417,; static, every target; 3 warning lines, 0 error lines |
| ka | test-cuda-keepalive run 1 | OK | rc=0; test-cuda-keepalive: 162 checks, 0 failed; 0 failed checks named |
| ka | test-cuda-keepalive run 2 | OK | rc=0; test-cuda-keepalive: 162 checks, 0 failed; 0 failed checks named |
| ka | test-cuda-keepalive run 3 | OK | rc=0; test-cuda-keepalive: 162 checks, 0 failed; 0 failed checks named |
| ka | test-cuda-keepalive gpu without a card | OK | rc=1 (1 expected); gpu: no CUDA device |
| off | base writes | OK | rc=0; the base's (`d097d6d88`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| off | tip | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager0 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=0; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager1 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager2 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=2; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| c565 static | c811 tree | OK | commit cfc657de5125bc72165633b92d94c7fee94aed4f  |
| c565 static | c811 build | OK | rc=0; BUILD_SHARED_LIBS=OFF; all targets; 0 error lines |
| c565 static | c811 ctest | OK | 100% tests passed, 0 tests failed out of 70; not run: ^(test-conversion-fp8/test-download-model/test-eval-callback-download-model/test-llama-context-model/test-thread-safety/test-eval-callback/test-arg-parser)$ |
| c565 static | c811 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 static | c811 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 static | c811 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 static | c811 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 static | c811 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 static | c811 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 static | c811 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c811 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c811 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 static | c811 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 static | c811 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 static | c811 hand thread-safety | OK | test-thread-safety rc=0: 0.00.028.391 I All threads finished without errors. |
| c565 static | c811 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 static | c811 hand eval-callback | OK | llama-eval-callback rc=0:     sum = 4.585421 |
| c565 shared | c811 tree | OK | commit cfc657de5125bc72165633b92d94c7fee94aed4f  |
| c565 shared | c811 build | OK | rc=0; BUILD_SHARED_LIBS=ON; all targets; 0 error lines |
| c565 shared | c811 ctest | OK | 100% tests passed, 0 tests failed out of 70; not run: ^(test-conversion-fp8/test-download-model/test-eval-callback-download-model/test-llama-context-model/test-thread-safety/test-eval-callback/test-arg-parser)$ |
| c565 shared | c811 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 shared | c811 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 shared | c811 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 shared | c811 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 shared | c811 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 shared | c811 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 shared | c811 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c811 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c811 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 shared | c811 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 shared | c811 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 shared | c811 hand thread-safety | OK | test-thread-safety rc=0: 0.00.032.893 I All threads finished without errors. |
| c565 shared | c811 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 shared | c811 hand eval-callback | OK | llama-eval-callback rc=0:     sum = 1.732148 |
| trim | CPU tree | OK | removed (the binaries in cpu/bin stay); 7.6G in the work folder; 356 GB free |
| start | pascal-p106-v4-b12246-c811 | OK | tip cfc657de5 (C811: GGML_CUDA_KEEPALIVE_EAGER, no first kernel launch waits for a keep-alive block (Q1417, from FN85)); files changed against the base:  6 files changed, 792 insertions(+), 16 deletions(-) |
| merged | pascal-p106-v4 | OK | e27947287 (Merge -c811 (C811: GGML_CUDA_KEEPALIVE_EAGER, no first kernel launch waits for a keep-alive block; off by default) into ); its tree is not cfc657de5's; commits since the base: 8 |
| merged | cpu build | OK | rc=0; e27947287; static, every target; 3 warning lines, 0 error lines |
| merged | test-cuda-keepalive | OK | rc=0; test-cuda-keepalive: 162 checks, 0 failed |
| moff | base writes | OK | rc=0; the base's (`d097d6d88`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| moff | merged | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| moff | merged-eager1 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
