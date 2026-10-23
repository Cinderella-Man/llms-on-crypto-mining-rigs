# Built inside a llama.cpp tree: copy this folder to examples/logit-dump and add `add_subdirectory(logit-dump)` to
# examples/CMakeLists.txt.
set(TARGET llama-logit-dump)
add_executable(${TARGET} logit_dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
