# Built inside a llama.cpp tree: copy this folder to examples/tool-sampling and add `add_subdirectory(tool-sampling)` to
# examples/CMakeLists.txt.
set(TARGET llama-tool-sampling-bench)
add_executable(${TARGET} tool_sampling_bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
