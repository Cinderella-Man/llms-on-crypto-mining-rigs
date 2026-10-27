# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
# It builds llama-tool-sampling-bench in a llama.cpp tree at /tmp/rq/src/llama.cpp prepared as in
# bench/tool_sampling/README.md (the folder copied to examples/tool-sampling and added to examples/CMakeLists.txt).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q68
cd /tmp/rq/src/llama.cpp
cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=Release > $OUTD/q68/cmake.log 2>&1
cmake --build build-cpu -j 2 --target llama-tool-sampling-bench > $OUTD/q68/build.log 2>&1 || { tail -30 $OUTD/q68/build.log; exit 1; }
lscpu | grep -E "Model name|Flags" | cut -c1-200 > $OUTD/q68/cpu.txt
./build-cpu/bin/llama-tool-sampling-bench $S4L0 > $OUTD/q68/bench.txt 2>&1
tail -40 $OUTD/q68/bench.txt
