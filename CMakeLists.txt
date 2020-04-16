cmake_minimum_required(VERSION 3.8)

project(TAT VERSION 0.0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/*.cpp)
foreach(FILE ${CPP_SRC})
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME} ${FILE})
endforeach(FILE)

add_test(simple_test ${PROJECT_BINARY_DIR}/simple_test ${PROJECT_SOURCE_DIR}/simple_test.out)
add_test(diag ${PROJECT_BINARY_DIR}/diag ${PROJECT_SOURCE_DIR}/diag.out)
add_test(mps ${PROJECT_BINARY_DIR}/mps 4 4 100 0.1 10 ${PROJECT_SOURCE_DIR}/mps.out)
add_test(fermi-diag ${PROJECT_BINARY_DIR}/fermi-diag ${PROJECT_SOURCE_DIR}/fermi-diag.out)
