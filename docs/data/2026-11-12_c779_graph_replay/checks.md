# C779 checks (build machine, CPU only, 2026-11-12)

Every line of the checks driver (run/checks.sh) in order: the run on the branch tip `771cafa97` 19:31:40-20:01:52, the merged `pascal-p106-v4` (`b9a1c1fa7`) 20:02:34-20:04:08. The PTX comparison with internal names masked (run/ptx_names.sh, 19:48-19:52) is the last table.

| row | check | verdict | detail |
|---|---|---|---|
| start | pascal-p106-v4-b12246-c779 | OK | tip 771cafa97 (C779: GGML_CUDA_GRAPHS_PATCH_MAX defaults to 100 (a share fallback instantiates anew in patch mode, so it can cost more ); files changed against the base:  5 files changed, 1154 insertions(+), 8 deletions(-) |
| cu | ggml-cuda.cu alone | OK | rc=0; 0 warning lines;  |
| rig | build | OK | rc=0; llama.cpp-20261112-771cafa97-pascal-p106-v4-b12246-c779.tar.gz md5 ceb3d5db5b748910e06e00f26211d2bd, 678532453 B; commit: 771cafa97c281981460e73c9534bcf05773048ae (2026-11-12 19:24:44 +0100);avx instructions in llama-bench: 0;highest glibc symbol: GLIBC_2.38;; test-backend-ops md5 eed2e5641686cd7fb7aa2a1a89fc2d4a; llama-server md5 e7ce845e00b467b34798598794f22d1c; 34 warning lines |
| names | switch list | OK | 865 names in the build's list; GGML_CUDA_GRAPHS_PATCH, GGML_CUDA_GRAPHS_PATCH_WINDOW and GGML_CUDA_GRAPHS_PATCH_MAX in it: 3 of 3; lines of bin/llama-server holding GGML_CUDA_GRAPHS_PATCH: 4 |
| sass | llama-server sass against the base | SAME | 8540 functions (8540 in the base's); dumps: the same file; merged 8540 functions: base 8540; listed 0;  |
| sass | llama-server ptx against the base | DIFFERS | 8540 functions (8540 in the base's); dumps: files differ; merged 8540 functions: base 8530; listed 10; NEW: _Z15flash_attn_tileILi128ELi128ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi1ELi2ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi128ELi128ELi2ELi1ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi1ELi2ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi256ELi256ELi2ELi1ELb1EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi64ELi64ELi1ELi2ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code;NEW: _Z15flash_attn_tileILi64ELi64ELi2ELi1ELb0EEvPKcS1_S1_S1_S1_PKiS3_PfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil: no branch changed it; not the base's code; |
| cpu | build | OK | rc=0; commit 771cafa97c281981460e73c9534bcf05773048ae C779: GGML_CUDA_GRAPHS_PATCH_MAX defaults to 100 (a share fallback instantiates anew in patc; static, every target; 3 warning lines, 0 error lines |
| gp | test-cuda-graph-patch run 1 | OK | rc=0; random runs: 2000 x 300 evaluations (600000); window 5: 70068 patched (41.2 groups each), 525190 full captures, every executable a fresh capture's; not_ready 6669 structure 6492 share 222925 skip 0 pool 239435 fuse 49254 nodes 415 api 0; the control (window 2) made other executables in 30018 evaluations of 277 runs test-cuda-graph-patch: 21013 checks, 0 failed ; 0 failed checks named |
| gp | test-cuda-graph-patch run 2 | OK | rc=0; random runs: 2000 x 300 evaluations (600000); window 5: 70068 patched (41.2 groups each), 525190 full captures, every executable a fresh capture's; not_ready 6669 structure 6492 share 222925 skip 0 pool 239435 fuse 49254 nodes 415 api 0; the control (window 2) made other executables in 30018 evaluations of 277 runs test-cuda-graph-patch: 21013 checks, 0 failed ; 0 failed checks named |
| gp | the two runs' output | SAME | the model's runs are seeded: the same counts both times |
| gp | test-cuda-graph-force | OK | rc=0; test-cuda-graph-force: 6070 checks, 0 failed |
| off | base writes | OK | rc=0; the base's (`cfc657de5`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| off | tip | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | force | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | patch | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| off | patch-w0 | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1 GGML_CUDA_GRAPHS_PATCH_WINDOW=0 GGML_CUDA_GRAPHS_PATCH_MAX=100; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| c565 static | c779 tree | OK | commit 771cafa97c281981460e73c9534bcf05773048ae  |
| c565 static | c779 build | OK | rc=0; BUILD_SHARED_LIBS=OFF; all targets; 0 error lines |
| c565 static | c779 ctest | OK | 100% tests passed, 0 tests failed out of 71; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 static | c779 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 static | c779 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 static | c779 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 static | c779 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 static | c779 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 static | c779 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 static | c779 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c779 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 static | c779 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 static | c779 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 static | c779 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 static | c779 hand thread-safety | OK | test-thread-safety rc=0: 0.00.030.535 I All threads finished without errors. |
| c565 static | c779 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 static | c779 hand eval-callback | OK | llama-eval-callback rc=0:     sum = -1.504189 |
| c565 shared | c779 tree | OK | commit 771cafa97c281981460e73c9534bcf05773048ae  |
| c565 shared | c779 build | OK | rc=0; BUILD_SHARED_LIBS=ON; all targets; 0 error lines |
| c565 shared | c779 ctest | OK | 100% tests passed, 0 tests failed out of 71; not run: ^(test-conversion-fp8\|test-download-model\|test-eval-callback-download-model\|test-llama-context-model\|test-thread-safety\|test-eval-callback\|test-arg-parser)$ |
| c565 shared | c779 net | OK | 100% tests passed, 0 tests failed out of 1; test-arg-parser, with network |
| c565 shared | c779 archs plain | OK | Summary: all 151 test(s) passed; plain rc=0 |
| c565 shared | c779 archs write | OK | Summary: all 151 test(s) passed; write rc=0 |
| c565 shared | c779 archs compare | OK | Summary: all 151 test(s) passed; compare rc=0 |
| c565 shared | c779 archs files | OK | 146 files written; 146 compared, differ: none; compared while writing (a shared file): none |
| c565 shared | c779 hand hop-cast | OK | test-hop-cast rc=0: 54 checks, 0 failed |
| c565 shared | c779 hand route-once-x1 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c779 hand route-once-x30 | OK | test-route-once rc=0: 223 checks, 0 failed |
| c565 shared | c779 hand bytes-switches | OK | test-bytes-switches rc=0: 535 checks, 0 failed |
| c565 shared | c779 hand bytes-zero-tail | OK | test-bytes-switches rc=0: 536 checks, 0 failed |
| c565 shared | c779 hand moe-ids | OK | test-moe-ids rc=0: 13 checks, 0 failed |
| c565 shared | c779 hand thread-safety | OK | test-thread-safety rc=0: 0.00.033.311 I All threads finished without errors. |
| c565 shared | c779 hand context-model | OK | test-llama-context rc=0: failed   : 0 |
| c565 shared | c779 hand eval-callback | OK | llama-eval-callback rc=0:     sum = -2.884993 |
| trim | CPU tree | OK | removed (the binaries in cpu/bin stay); 7.0G in the work folder |
| start | pascal-p106-v4-b12246-c779 | OK | tip 771cafa97 (C779: GGML_CUDA_GRAPHS_PATCH_MAX defaults to 100 (a share fallback instantiates anew in patch mode, so it can cost more ); files changed against the base:  5 files changed, 1154 insertions(+), 8 deletions(-) |
| merged | pascal-p106-v4 | OK | b9a1c1fa7 (Merge -c779 (C779: GGML_CUDA_GRAPHS_PATCH=1, with GGML_CUDA_GRAPHS_FORCE=1 a forced graph whose properties changed in a ); its tree is not 771cafa97's; commits since the base: 14 |
| merged | cpu build | OK | rc=0; b9a1c1fa7; static, every target; 3 warning lines, 0 error lines |
| merged | test-cuda-graph-patch | OK | rc=0; test-cuda-graph-patch: 21013 checks, 0 failed |
| moff | base writes | OK | rc=0; the base's (`cfc657de5`) static CPU build: 146 logits files; Summary: all 151 test(s) passed |
| moff | merged | SAME | rc=0; no switch; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |
| moff | merged-patch | SAME | rc=0; GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1; against the base's 146 files: 146 compared, 0 differ: none; written by this build only: 0; Summary: all 151 test(s) passed |

## The PTX with internal names masked (run/ptx_names.sh)

```
lines: new 10670256, base 10670256
differing lines: 70
their internal names (with the number after the hash):
     14 _cu_8a22d1c2_482320
     14 _cu_8a22d1c2_487420
      7 _cu_8a22d1c2_503420
     14 _cu_b8226404_479820
     14 _cu_b8226404_485820
      7 _cu_b8226404_501120
masked md5: new 80551b469e1173101aab20a1fb7ebee5, base 80551b469e1173101aab20a1fb7ebee5
verdict: SAME
```
