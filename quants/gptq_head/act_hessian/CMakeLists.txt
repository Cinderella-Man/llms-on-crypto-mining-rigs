# Built inside a llama.cpp tree: copy this folder to examples/act-hessian and add `add_subdirectory(act-hessian)` to
# examples/CMakeLists.txt.
set(TARGET llama-act-hessian)
add_executable(${TARGET} act_hessian.cpp)
install(TARGETS ${TARGET} RUNTIME)
find_package(OpenMP REQUIRED)
target_link_libraries(${TARGET} PRIVATE llama-common llama OpenMP::OpenMP_CXX ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# -DHESS_BLAS_LIB=<static OpenBLAS built with OpenMP, e.g. .../openblas-openmp/libopenblas.a> -DHESS_BLAS_INCLUDE=<dir of
# its cblas.h>: Gram matrices with cblas_ssyrk instead of plain loops
set(HESS_BLAS_LIB "" CACHE FILEPATH "static OpenBLAS (OpenMP build) for llama-act-hessian")
set(HESS_BLAS_INCLUDE "" CACHE PATH "directory of that OpenBLAS's cblas.h")
if (HESS_BLAS_LIB)
    target_compile_definitions(${TARGET} PRIVATE HESS_BLAS)
    target_include_directories(${TARGET} PRIVATE ${HESS_BLAS_INCLUDE})
    target_link_libraries(${TARGET} PRIVATE ${HESS_BLAS_LIB})
endif()
