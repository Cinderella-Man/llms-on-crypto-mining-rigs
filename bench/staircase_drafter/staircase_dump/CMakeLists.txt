# Built inside a llama.cpp tree: copy this folder to examples/staircase-dump and add
# `add_subdirectory(staircase-dump)` to examples/CMakeLists.txt.
set(TARGET llama-staircase-dump)
add_executable(${TARGET} staircase_dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
