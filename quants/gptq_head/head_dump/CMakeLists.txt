# Built inside a llama.cpp tree: copy this folder to examples/head-dump and add `add_subdirectory(head-dump)` to
# examples/CMakeLists.txt.
set(TARGET llama-head-dump)
add_executable(${TARGET} head_dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
