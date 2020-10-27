cmake_minimum_required(VERSION 3.13)

project(TAT VERSION 0.0.7 LANGUAGES CXX)

# 常设置的参数有
# CMAKE_BUILD_TYPE
# TAT_NO_MPI, TAT_FORCE_BLAS_DYNAMIC, TAT_PYTHON_MODULE
# BLA_VENDOR
# PYBIND11_PYTHON_VERSION, PYTHON_EXECUTABLE

if(NOT DEFINED TAT_PYTHON_MODULE)
   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(TAT_PYTHON_MODULE TAT)
   else()
      set(TAT_PYTHON_MODULE TAT_${CMAKE_BUILD_TYPE})
   endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
   set(TAT_COMPILER_NAME "GCC")
else()
   set(TAT_COMPILER_NAME "${CMAKE_CXX_COMPILER_ID}")
endif()

add_compile_definitions(TAT_VERSION="${PROJECT_VERSION}")
add_compile_definitions(TAT_PYTHON_MODULE=${TAT_PYTHON_MODULE})
if(CMAKE_BUILD_TYPE STREQUAL "")
   add_compile_definitions(TAT_BUILD_TYPE="Default")
else()
   add_compile_definitions(TAT_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
endif()
add_compile_definitions(TAT_COMPILER_INFORMATION="${TAT_COMPILER_NAME} ${CMAKE_CXX_COMPILER_VERSION} on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# linux一般没问题, clang, gcc当然是都没问题的 我只使用过openmpi
# blas, lapack和mkl都没有问题
# windows的情况如下
# vs 2019 可以找到intel mpi, 使用clang和msvc都是可以的, 但是mkl需要手动设置MKLROOT的环境变量
# clion 无法找到intel mpi, 其他与vs 2019一致
# 自己的pc上linux中无法使用intel的编译器进行编译, 似乎与版本有关系, 在学校超算上配合合适的gcc版本是可以编译的
# windows中的intel情况懒得测试了

# 设置为c++17, 大多数超算上目前都有支持c++17的编译器, 故如此, c++20的话部分不支持, 所以本库也不使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 尝试启用mpi, 如果无法启用, 也没关系
if(NOT TAT_NO_MPI)
   if(CMAKE_CXX_COMPILER MATCHES "mpi")
      message("-- Using mpi compiler directly")
      add_compile_definitions(TAT_USE_MPI)
   else()
      find_package(MPI QUIET)
      if(MPI_FOUND)
         include_directories(${MPI_INCLUDE_PATH})
         link_libraries(${MPI_LIBRARIES})
         message("-- Using mpi by linking mpi libraries")
         add_compile_definitions(TAT_USE_MPI)
      else()
         message("-- No mpi support")
      endif()
   endif()
else()
   message("-- Disable mpi support manually")
endif()

# msvc必须加个utf8的参数
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 链接blas和lapack, 尽量使用静态链接, 如果是emscripten则链接emlapack, 需要自行放入emscripten目录下
if(EMSCRIPTEN)
   message("-- Use emscripten blas and lapack")
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/liblapack.a)
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/libblas.a)
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/libf2c.a)
else()
   if (NOT TAT_FORCE_BLAS_DYNAMIC)
      message("-- Try to find static blas and lapack")
      set(BLA_STATIC ON)
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
   endif()
   if (NOT (LAPACK_FOUND AND BLAS_FOUND))
      message("-- Try to find dynamic libraries")
      set(BLA_STATIC OFF)
      find_package(BLAS REQUIRED)
      find_package(LAPACK REQUIRED)
   endif()
   link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
   # 检查是否使用了mkl
   if(BLAS_LIBRARIES MATCHES mkl)
      message("-- Using mkl")
      add_compile_definitions(TAT_USE_MKL_TRANSPOSE)
   else()
      message("-- Not using mkl")
   endif()
endif()

# examples中的各个程序
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/examples/*.cpp)
foreach(FILE ${CPP_SRC})
   get_filename_component(NAME ${FILE} NAME_WE)
   add_executable(${NAME} ${FILE})
endforeach(FILE)

# 以及他们的测试
enable_testing()
add_test(simple_test ${PROJECT_BINARY_DIR}/simple_test ${PROJECT_SOURCE_DIR}/examples/simple_test.out)
add_test(diag ${PROJECT_BINARY_DIR}/diag ${PROJECT_SOURCE_DIR}/examples/diag.out)
add_test(mps ${PROJECT_BINARY_DIR}/mps 4 4 100 0.1 10 ${PROJECT_SOURCE_DIR}/examples/mps.out)
add_test(fermi-diag ${PROJECT_BINARY_DIR}/fermi-diag ${PROJECT_SOURCE_DIR}/examples/fermi-diag.out)
add_test(OBC ${PROJECT_BINARY_DIR}/OBC ${PROJECT_SOURCE_DIR}/examples/OBC.out)

# 尝试产生本库的python接口, 先寻找系统中的pybind11, 否则尝试使用本目录下自行放置的pybind11
if(NOT EMSCRIPTEN)
   find_package(pybind11 QUIET)
   if(pybind11_FOUND)
      message("-- Enable python(system)")
      pybind11_add_module(${TAT_PYTHON_MODULE} ${PROJECT_SOURCE_DIR}/python/TAT.cpp)
   elseif(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
      add_subdirectory(pybind11)
      message("-- Enable python(local)")
      pybind11_add_module(${TAT_PYTHON_MODULE} ${PROJECT_SOURCE_DIR}/python/TAT.cpp)
   else()
      message("-- Disable python")
   endif()
endif()
