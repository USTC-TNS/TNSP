cmake_minimum_required(VERSION 3.8)
project(TAT-test-cuda VERSION 0.2.19 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(CUDAToolkit)
link_libraries(CUDA::cublas CUDA::cudart)

set(TAT_MATH_LIBRARIES /opt/openblas-0.3.17/lib/libopenblas.so.0)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/TAT)
add_executable(run main.cpp)
target_compile_definitions(run PUBLIC -DTAT_USE_TIMER -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1)
target_link_libraries(run TAT)
