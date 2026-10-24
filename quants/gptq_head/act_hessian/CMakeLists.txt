# Built inside a llama.cpp tree: copy this folder to examples/act-hessian and add `add_subdirectory(act-hessian)` to
# examples/CMakeLists.txt.
set(TARGET llama-act-hessian)
add_executable(${TARGET} act_hessian.cpp)
install(TARGETS ${TARGET} RUNTIME)
find_package(OpenMP REQUIRED)
target_link_libraries(${TARGET} PRIVATE llama-common llama OpenMP::OpenMP_CXX ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
