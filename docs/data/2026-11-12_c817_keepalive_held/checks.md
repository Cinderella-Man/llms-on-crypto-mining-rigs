# C817 checks (build machine, CPU only, 2026-11-12)

Every line of the checks driver (run/checks.sh) in order: the run on the branch tip `5f8dd491b` 21:31:15-22:14:44, the merged `pascal-p106-v4` (`e572eb1e6`) 22:15:18-22:18:37. The base's rig tarball (`b9a1c1fa7`) was built in a first run on `8f6916133`, 20:48-21:15: its line is the last table.

| row | check | verdict | detail |
|---|---|---|---|
| start | pascal-p106-v4-b12246-c817 | OK | tip 5f8dd491b (C817: test-cuda-keepalive: the held-start case checks that the held block never ran, not that the first block did (a sto); files changed against the base:  3 files changed, 381 insertions(+), 28 deletions(-) |
| rig | build | OK | rc=0; llama.cpp-20261112-5f8dd491b-pascal-p106-v4-b12246-c817.tar.gz md5 8b651102279353d33c35260c1b825d56, 741714237 B; commit: 5f8dd491b317496b01d12d7b8addabd9a1c44072 (2026-11-12 21:30:59 +0100);avx instructions in llama-bench: 0;highest glibc symbol: GLIBC_2.38;; test-backend-ops md5 804407bec644c1e46d01101e654c76d3; test-cuda-keepalive md5 719c24782620821a45253b9cd405b02a; llama-server md5 fbc82b14f36576b89ae611bffac5b1ac; 34 warning lines |
| names | switch list | OK | 867 names in the build's list; GGML_CUDA_KEEPALIVE_EAGER in it: 1, lines of bin/llama-server holding it: 2; holding ggml_backend_cuda_keepalive_counts: 2 |
| sass | llama-server sass against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| sass | llama-server ptx against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| cpu | build | OK | rc=0; commit 5f8dd491b317496b01d12d7b8addabd9a1c44072 C817: test-cuda-keepalive: the held-start case checks that the held block never ran, not tha; static, every target; 3 warning lines, 0 error lines |
| ka | test-cuda-keepalive run 1 | OK | rc=0; test-cuda-keepalive: 237 checks, 0 failed; 0 failed checks named; test_held_stop: C811's controller: the stop at a held start took 300.1 ms (the evaluation it is held behind: 300 ms);test_held_stop: C817: the stop at a held start took 0.0 ms (the evaluation it is held behind: 300 ms);test_held_chunk: C811's controller: each card queued at 0.0, 60.4, 120.6, 181.0 ms (all in 181.1 ms; 60 ms a card);test_held_chunk: C817: each card queued at 0.0, 0.3, 0.4, 0.6 ms (all in 0.6 ms; 60 ms a card);test_held_threads: 484 runs, 303 ended before they began, 0 ended by the timeout; |
| ka | test-cuda-keepalive run 2 | OK | rc=0; test-cuda-keepalive: 237 checks, 0 failed; 0 failed checks named; test_held_stop: C811's controller: the stop at a held start took 300.0 ms (the evaluation it is held behind: 300 ms);test_held_stop: C817: the stop at a held start took 0.0 ms (the evaluation it is held behind: 300 ms);test_held_chunk: C811's controller: each card queued at 0.0, 60.2, 120.3, 180.5 ms (all in 180.5 ms; 60 ms a card);test_held_chunk: C817: each card queued at 0.0, 0.3, 0.3, 0.5 ms (all in 0.5 ms; 60 ms a card);test_held_threads: 468 runs, 297 ended before they began, 0 ended by the timeout; |
| ka | test-cuda-keepalive run 3 | OK | rc=0; test-cuda-keepalive: 237 checks, 0 failed; 0 failed checks named; test_held_stop: C811's controller: the stop at a held start took 300.1 ms (the evaluation it is held behind: 300 ms);test_held_stop: C817: the stop at a held start took 0.0 ms (the evaluation it is held behind: 300 ms);test_held_chunk: C811's controller: each card queued at 0.0, 60.2, 120.4, 180.5 ms (all in 180.6 ms; 60 ms a card);test_held_chunk: C817: each card queued at 0.0, 0.2, 0.3, 0.5 ms (all in 0.5 ms; 60 ms a card);test_held_threads: 468 runs, 297 ended before they began, 0 ended by the timeout; |
| ka | test-cuda-keepalive gpu without a card | OK | rc=1 (1 expected); gpu: no CUDA device |
| off | base writes | OK | rc=0; C811's static CPU build (`cfc657de5`, the base's CPU results): 146 logits files; Summary: all 151 test(s) passed |
| off | tip | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager0 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=0; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager1 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | eager2 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=2; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| c565 static | c817 tree | OK | commit 5f8dd491b317496b01d12d7b8addabd9a1c44072  |
| c565 static | c817 build | OK | rc=0; BUILD_SHARED_LIBS=OFF; all targets; 0 error lines |
| c565 static | c817 ctest | OK | 100% tests passed, 0 tests failed out of 71; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 static | c817 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 static | c817 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 static | c817 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 static | c817 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 static | c817 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 static | c817 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 static | c817 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c817 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c817 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 static | c817 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 static | c817 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 static | c817 hand thread-safety | OK | test-thread-safety rc=0: 0.00.078.058 I All threads finished without errors. |
| c565 static | c817 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 static | c817 hand eval-callback | OK | llama-eval-callback rc=0:     sum = 2.950592 |
| c565 shared | c817 tree | OK | commit 5f8dd491b317496b01d12d7b8addabd9a1c44072  |
| c565 shared | c817 build | OK | rc=0; BUILD_SHARED_LIBS=ON; all targets; 0 error lines |
| c565 shared | c817 ctest | OK | 100% tests passed, 0 tests failed out of 71; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 shared | c817 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 shared | c817 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 shared | c817 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 shared | c817 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 shared | c817 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 shared | c817 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 shared | c817 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c817 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c817 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 shared | c817 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 shared | c817 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 shared | c817 hand thread-safety | OK | test-thread-safety rc=0: 0.00.031.947 I All threads finished without errors. |
| c565 shared | c817 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 shared | c817 hand eval-callback | OK | llama-eval-callback rc=0:             [      0.0981,       0.0561,      -0.0205,    ...,  |
| trim | CPU tree | OK | removed (the binaries in cpu/bin stay); 9.3G in the work folder; 532 GB free |
| start | pascal-p106-v4-b12246-c817 | OK | tip 5f8dd491b (C817: test-cuda-keepalive: the held-start case checks that the held block never ran, not that the first block did (a sto); files changed against the base:  3 files changed, 381 insertions(+), 28 deletions(-) |
| merged | pascal-p106-v4 | OK | e572eb1e6 (Merge -c817 (C817: at GGML_CUDA_KEEPALIVE_EAGER 1 and 2, a stop on a card whose keep-alive block has not begun (its star); its tree is not 5f8dd491b's; commits since the base: 6 |
| merged | cpu build | OK | rc=0; e572eb1e6; static, every target; 3 warning lines, 0 error lines |
| merged | test-cuda-keepalive | OK | rc=0; test-cuda-keepalive: 237 checks, 0 failed |
| moff | base writes | OK | rc=0; C811's static CPU build (`cfc657de5`, the base's CPU results): 146 logits files; Summary: all 151 test(s) passed |
| moff | merged | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| moff | merged-eager1 | SAME | rc=0; GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |

The first run on `8f6916133` (its base line; the rest of that run is superseded by the run above):

| row | check | verdict | detail |
|---|---|---|---|
| base | build | OK | rc=0; llama.cpp-20261112-b9a1c1fa7.tar.gz md5 262917966d1c0a597f2b5e5a2adc1d91, 741689679 B; commit: b9a1c1fa7b9b98dfe919374897afe6fde60b7219 (2026-11-12 20:02:14 +0100);avx instructions in llama-bench: 0;highest glibc symbol: GLIBC_2.38;; llama-server md5 8c4a34475448e9a76fa26531d8bc9297; 34 warning lines |
