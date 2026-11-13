# C826 checks (build machine, CPU only, 2026-11-13)

Every line of the checks driver (run/checks.sh) in order: the run on the branch tip `b248a54e2` 09:23:51-10:19:28 (the PTX step runs C779's ptx_names.sh), the merged `pascal-p106-v4` (`36ef48889`) 10:20:07-10:38:36. Base: the `-c822` rig tarball (`cc97bb6b6`, whose tree is `2144cf760`'s) for the GPU code; a static CPU build of `65ac9a47d` (to which `2144cf760` adds CUDA host code only) for the logits.

| row | check | verdict | detail |
|---|---|---|---|
| start | pascal-p106-v4-b12246-c826 | OK | tip b248a54e2 (CUDA: GGML_CUDA_GRAPHS_PATCH compares sources by kind, keeps unchanged memsets and memcpys, names refused updates (C826,); files changed against the base:  4 files changed, 552 insertions(+), 94 deletions(-) |
| cu | ggml-cuda.cu alone | OK | rc=0; 0 warning lines, 0 of them naming ggml-cuda.cu or graph-patch.h;  |
| rig | build | OK | rc=0; llama.cpp-20261113-b248a54e2-pascal-p106-v4-b12246-c826.tar.gz md5 c2cc27fa673fcc30a6c79e95fb9430b8, 678730034 B; commit: b248a54e2745ced88c6c118e3f36e1b0adeab327 (2026-11-13 09:21:54 +0100);avx instructions in llama-bench: 0;highest glibc symbol: GLIBC_2.38;; test-backend-ops md5 fa45c59032e2a640fa5b973dbc6f8944; llama-server md5 f8b9d0a70038b44c483b1aa7b5bfb2cd; 34 warning lines |
| names | switch list | OK | 870 names in the build's list; GGML_CUDA_GRAPHS_PATCH, GGML_CUDA_GRAPHS_PATCH_WINDOW and GGML_CUDA_GRAPHS_PATCH_MAX in it: 3 of 3; lines of bin/llama-server holding GGML_CUDA_GRAPHS_PATCH: 7 |
| sass | llama-server sass against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| sass | llama-server ptx against the base | DIFFERS | 8540 functions (8540 in the base's); dumps: files differ; merged 8540 functions: base 8530; listed 10; NEW: _Z15flash_attn_tileILi128ELi128ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi1ELi2ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi2ELi1ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi1ELi2ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi2ELi1ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi64ELi64ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi64ELi64ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code; |
| ptx | llama-server PTX, internal names' numbers masked | SAME | lines: new 10670256, base 10670256;differing lines: 70;masked md5: new 80551b469e1173101aab20a1fb7ebee5, base 80551b469e1173101aab20a1fb7ebee5;; names: 14 _cu_1fadf18e_476420;14 _cu_1fadf18e_481720;7 _cu_1fadf18e_497520;14 _cu_8a22d1c2_482320;14 _cu_8a22d1c2_487420;7 _cu_8a22d1c2_503420; |
| cpu | build | OK | rc=0; commit b248a54e2745ced88c6c118e3f36e1b0adeab327 CUDA: GGML_CUDA_GRAPHS_PATCH compares sources by kind, keeps unchanged memsets and memcpys, ; static, every target; 3 warning lines, 0 error lines |
| gp | test-cuda-graph-patch run 1 | OK | rc=0; controls: window 2 made other executables in 34816 evaluations of 322 runs; templates with the recording's parameters in 2465 evaluations of 71 runs test-cuda-graph-patch: 24545 checks, 0 failed ; 0 failed checks named |
| gp | test-cuda-graph-patch run 2 | OK | rc=0; controls: window 2 made other executables in 34816 evaluations of 322 runs; templates with the recording's parameters in 2465 evaluations of 71 runs test-cuda-graph-patch: 24545 checks, 0 failed ; 0 failed checks named |
| gp | the two runs' output | SAME | the model's runs are seeded: the same counts both times |
| gp | test-cuda-graph-force | OK | rc=0; test-cuda-graph-force: 6070 checks, 0 failed |
| off | base writes | OK | rc=0; the base's (`65ac9a47d`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| off | tip | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | force | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | patch | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | patch-w0 | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1 GGML_CUDA_GRAPHS_PATCH_WINDOW=0 GGML_CUDA_GRAPHS_PATCH_MAX=100; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| c565 static | c826 tree | OK | commit b248a54e2745ced88c6c118e3f36e1b0adeab327  |
| c565 static | c826 build | OK | rc=0; BUILD_SHARED_LIBS=OFF; all targets; 0 error lines |
| c565 static | c826 ctest | OK | 100% tests passed, 0 tests failed out of 72; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 static | c826 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 static | c826 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 static | c826 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 static | c826 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 static | c826 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 static | c826 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 static | c826 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c826 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c826 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 static | c826 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 static | c826 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 static | c826 hand thread-safety | OK | test-thread-safety rc=0: 0.00.029.306 I All threads finished without errors. |
| c565 static | c826 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 static | c826 hand eval-callback | OK | llama-eval-callback rc=0:     sum = -1.205612 |
| c565 shared | c826 tree | OK | commit b248a54e2745ced88c6c118e3f36e1b0adeab327  |
| c565 shared | c826 build | OK | rc=0; BUILD_SHARED_LIBS=ON; all targets; 0 error lines |
| c565 shared | c826 ctest | OK | 100% tests passed, 0 tests failed out of 72; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 shared | c826 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 shared | c826 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 shared | c826 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 shared | c826 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 shared | c826 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 shared | c826 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 shared | c826 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c826 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c826 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 shared | c826 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 shared | c826 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 shared | c826 hand thread-safety | OK | test-thread-safety rc=0: 0.00.060.583 I All threads finished without errors. |
| c565 shared | c826 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 shared | c826 hand eval-callback | OK | llama-eval-callback rc=0:     sum = 2.349647 |
| trim | CPU tree | OK | removed (the binaries in cpu/bin stay); 5.9G in the work folder |
| start | pascal-p106-v4-b12246-c826 | OK | tip b248a54e2 (CUDA: GGML_CUDA_GRAPHS_PATCH compares sources by kind, keeps unchanged memsets and memcpys, names refused updates (C826,); files changed against the base:  4 files changed, 552 insertions(+), 94 deletions(-) |
| merged | pascal-p106-v4 | OK | 36ef48889 (Merge -c826 (C826: GGML_CUDA_GRAPHS_PATCH compares a node's sources by kind (type, op, flags, buffer, extra), so the sch); its tree is not b248a54e2's; commits since the base: 6 |
| merged | cpu build | OK | rc=0; 36ef48889; static, every target; 3 warning lines, 0 error lines |
| merged | test-cuda-graph-patch | OK | rc=0; test-cuda-graph-patch: 24545 checks, 0 failed |
| moff | base writes | OK | rc=0; the base's (`65ac9a47d`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| moff | merged | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| moff | merged-patch | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
