cmake_minimum_required (VERSION 3.8)

project (TAT VERSION 0.0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable (simple_test "simple_test.cpp")

add_test(simple_test ${PROJECT_BINARY_DIR}/simple_test ${PROJECT_SOURCE_DIR}/simple_test.out)
