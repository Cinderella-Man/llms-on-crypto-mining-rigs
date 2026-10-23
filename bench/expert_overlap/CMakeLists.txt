# Built inside a llama.cpp tree: copy this folder to examples/route-dump and add `add_subdirectory(route-dump)` to
# examples/CMakeLists.txt.
set(TARGET llama-route-dump)
add_executable(${TARGET} route_dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
