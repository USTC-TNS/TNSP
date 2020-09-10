cmake_minimum_required(VERSION 3.13)

project(TAT VERSION 0.0.5 LANGUAGES CXX)

# linux一般没问题, clang, gcc当然是都没问题的 我只使用过openmpi
# blas, lapack和mkl都没有问题
# windows的情况如下
# vs 2019 可以找到intel mpi, 使用clang和msvc都是可以的, 但是mkl需要手动设置MKLROOT的环境变量
# clion 无法找到intel mpi, 其他与vs 2019一致
# linux中无法使用intel的编译器进行编译, windows中懒得测试去了

# 设置为c++17, 大多数超算上目前都有支持c++17的编译器, 故如此, c++20的话部分不支持, 所以本库也不使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 尝试启用mpi, 如果无法启用, 也没关系
if(CMAKE_CXX_COMPILER MATCHES "mpi")
   message("-- Using mpi compiler directly")
   add_compile_definitions(TAT_USE_MPI)
else()
   find_package(MPI)
   if(MPI_FOUND)
      include_directories(${MPI_INCLUDE_PATH})
      link_libraries(${MPI_LIBRARIES})
      message("-- Using mpi by linking mpi libraries")
      add_compile_definitions(TAT_USE_MPI)
   else()
      message("-- No mpi support")
   endif()
endif()

# msvc必须加个utf8的参数
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 链接blas和lapack, 尽量使用静态链接, 如果是emscripten则链接emlapack, 需要自行放入emscripten目录下
if(EMSCRIPTEN)
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/libf2c.a)
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/libblas.a)
   link_libraries(${PROJECT_SOURCE_DIR}/emscripten/liblapack.a)
else()
   set(BLA_STATIC ON)
   find_package(BLAS)
   find_package(LAPACK)
   if (NOT (LAPACK_FOUND AND BLAS_FOUND))
      set(BLA_STATIC OFF)
      find_package(BLAS REQUIRED)
      find_package(LAPACK REQUIRED)
   endif()
   link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/examples/*.cpp)
foreach(FILE ${CPP_SRC})
   get_filename_component(NAME ${FILE} NAME_WE)
   add_executable(${NAME} ${FILE})
endforeach(FILE)

enable_testing()
add_test(simple_test ${PROJECT_BINARY_DIR}/simple_test ${PROJECT_SOURCE_DIR}/examples/simple_test.out)
add_test(diag ${PROJECT_BINARY_DIR}/diag ${PROJECT_SOURCE_DIR}/examples/diag.out)
add_test(mps ${PROJECT_BINARY_DIR}/mps 4 4 100 0.1 10 ${PROJECT_SOURCE_DIR}/examples/mps.out)
add_test(fermi-diag ${PROJECT_BINARY_DIR}/fermi-diag ${PROJECT_SOURCE_DIR}/examples/fermi-diag.out)
add_test(OBC ${PROJECT_BINARY_DIR}/OBC ${PROJECT_SOURCE_DIR}/examples/OBC.out)

# 尝试产生本库的python接口, 先寻找系统中的pybind11, 否则尝试使用本目录下自行放置的pybind11
find_package(pybind11)
if(pybind11_FOUND)
   message("-- Enable python(system)")
   pybind11_add_module(TAT ${PROJECT_SOURCE_DIR}/python/TAT.cpp)
elseif(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
   add_subdirectory(pybind11)
   message("-- Enable python(local)")
   pybind11_add_module(TAT ${PROJECT_SOURCE_DIR}/python/TAT.cpp)
else()
   message("-- Disable python")
endif()
