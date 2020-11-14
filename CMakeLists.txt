cmake_minimum_required(VERSION 3.13)

project(TAT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

string(REGEX REPLACE "-O[23]" "-Ofast" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-O[23]" "-Ofast" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "-O[23]" "-Ofast" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}")
string(REGEX REPLACE "-O[23]" "-Ofast" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

# linux一般没问题, clang, gcc当然是都没问题的 我只使用过openmpi
# blas, lapack和mkl都没有问题
# windows的情况如下
# vs 2019 可以找到intel mpi, 使用clang和msvc都是可以的, 但是mkl需要手动设置MKLROOT的环境变量
# clion 无法找到intel mpi, 其他与vs 2019一致
# 自己的pc上linux中无法使用intel的编译器进行编译, 似乎与版本有关系, 在学校超算上配合合适的gcc版本是可以编译的
# windows中的intel情况懒得测试了

# 设置TAT library
add_library(TAT INTERFACE)
target_include_directories(TAT INTERFACE ${PROJECT_SOURCE_DIR}/include/)

# 设置为c++17, 大多数超算上目前都有支持c++17的编译器, 故如此, c++20的话部分不支持, 所以本库也不使用
# 可以打开gnu dialect，但是作为测试保证在c++17标准下可以编译
target_compile_features(TAT INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)

# 常设置的参数有
# CMAKE_BUILD_TYPE, CMAKE_CXX_FLAGS
# TAT_USE_MPI, TAT_PREFER_BLAS_STATIC, TAT_PYTHON_MODULE, TAT_FORCE_VERSION
# BLA_VENDOR
# PYBIND11_PYTHON_VERSION, PYTHON_EXECUTABLE
option(TAT_USE_MPI "Use mpi for TAT" ON)
option(TAT_PREFER_BLAS_STATIC "Prefer to static blas library for TAT" ON)
set(TAT_PYTHON_MODULE TAT CACHE STRING "Set python binding module name")
set(TAT_FORCE_VERSION ${PROJECT_VERSION} CACHE STRING "Force set TAT version")

target_compile_definitions(TAT INTERFACE TAT_VERSION="${TAT_FORCE_VERSION}")
target_compile_definitions(TAT INTERFACE TAT_PYTHON_MODULE=${TAT_PYTHON_MODULE})
target_compile_definitions(TAT INTERFACE TAT_BUILD_TYPE="$<IF:$<STREQUAL:${CMAKE_BUILD_TYPE},>,Default,${CMAKE_BUILD_TYPE}>")
target_compile_definitions(TAT INTERFACE TAT_COMPILER_INFORMATION="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# msvc必须加个utf8的参数
target_compile_options(TAT INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 尝试启用mpi, 如果无法启用, 也没关系
if(CMAKE_CXX_COMPILER MATCHES "mpi")
   message("-- Using mpi compiler directly")
   target_compile_definitions(TAT INTERFACE TAT_USE_MPI)
   if(NOT TAT_USE_MPI)
      message("-- Ignore TAT_USE_MPI=OFF")
   endif()
else()
   if(TAT_USE_MPI)
         find_package(MPI QUIET)
         if(MPI_FOUND)
            target_include_directories(TAT INTERFACE ${MPI_INCLUDE_PATH})
            target_link_libraries(TAT INTERFACE ${MPI_LIBRARIES})
            message("-- Using mpi by linking mpi libraries")
            target_compile_definitions(TAT INTERFACE TAT_USE_MPI)
         else()
            message("-- No mpi support since mpi not found")
         endif()
   else()
      message("-- Disable mpi support since TAT_USE_MPI=OFF")
   endif()
endif()

# 链接blas和lapack, 尽量使用静态链接, 如果是emscripten则链接emlapack, 需要自行放入emscripten目录下
if(EMSCRIPTEN)
   message("-- Use emscripten blas and lapack")
   target_link_libraries(TAT INTERFACE ${PROJECT_SOURCE_DIR}/emscripten/liblapack.a)
   target_link_libraries(TAT INTERFACE ${PROJECT_SOURCE_DIR}/emscripten/libblas.a)
   target_link_libraries(TAT INTERFACE ${PROJECT_SOURCE_DIR}/emscripten/libf2c.a)
else()
   if(TAT_PREFER_BLAS_STATIC)
      message("-- Try to find static blas and lapack")
      set(BLA_STATIC ON)
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
   endif()
   if(NOT (LAPACK_FOUND AND BLAS_FOUND))
      message("-- Try to find dynamic libraries")
      set(BLA_STATIC OFF)
      find_package(BLAS REQUIRED)
      find_package(LAPACK REQUIRED)
   endif()
   target_link_libraries(TAT INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
   # 检查是否使用了mkl
   if(BLAS_LIBRARIES MATCHES mkl)
      message("-- Using mkl")
      target_compile_definitions(TAT INTERFACE TAT_USE_MKL_TRANSPOSE)
   else()
      message("-- Not using mkl")
   endif()
endif()

get_directory_property(TAT_HAS_PARENT PARENT_DIRECTORY)

# 尝试产生本库的python接口, 先寻找系统中的pybind11, 否则尝试使用本目录下自行放置的pybind11
set(TAT_BUILD_PYTAT OFF)
if(NOT EMSCRIPTEN)
   find_package(pybind11 QUIET)
   if(pybind11_FOUND)
      message("-- Enable python(system)")
      set(TAT_BUILD_PYTAT ON)
   elseif(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
      add_subdirectory(pybind11)
      message("-- Enable python(local)")
      set(TAT_BUILD_PYTAT ON)
   else()
      message("-- Disable python since pybind11 not found, try install pybind11 or put it into TAT directory")
   endif()
endif()
if(TAT_BUILD_PYTAT)
   pybind11_add_module(PyTAT ${PROJECT_SOURCE_DIR}/python/TAT.cpp)
   target_link_libraries(PyTAT PRIVATE TAT)
   set_target_properties(PyTAT PROPERTIES OUTPUT_NAME ${TAT_PYTHON_MODULE})
   if(TAT_HAS_PARENT)
      set_target_properties(PyTAT PROPERTIES EXCLUDE_FROM_ALL ON)
   endif()
endif()

if(NOT TAT_HAS_PARENT)
   # examples中的各个程序
   link_libraries(TAT)

   file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/examples/*.cpp ${PROJECT_SOURCE_DIR}/test/*.cpp)
   foreach(FILE ${CPP_SRC})
      get_filename_component(NAME ${FILE} NAME_WE)
      add_executable(${NAME} ${FILE})
   endforeach(FILE)

   # 以及他们的测试
   enable_testing()
   add_test(simple_test ${PROJECT_BINARY_DIR}/simple_test ${PROJECT_SOURCE_DIR}/test/simple_test.out)
   add_test(diag ${PROJECT_BINARY_DIR}/diag ${PROJECT_SOURCE_DIR}/examples/diag.out)
   add_test(mps ${PROJECT_BINARY_DIR}/mps 4 4 100 0.1 10 ${PROJECT_SOURCE_DIR}/examples/mps.out)
   add_test(fermi-diag ${PROJECT_BINARY_DIR}/fermi-diag ${PROJECT_SOURCE_DIR}/examples/fermi-diag.out)
   add_test(OBC ${PROJECT_BINARY_DIR}/OBC ${PROJECT_SOURCE_DIR}/examples/OBC.out)
endif()
