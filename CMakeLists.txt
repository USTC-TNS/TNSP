cmake_minimum_required(VERSION 3.5)
# set(CMAKE_VERBOSE_MAKEFILE ON)
project (TAT VERSION 0.0.3 LANGUAGES CXX)

# Version
message("-- Version is ${TAT_VERSION}")
add_definitions("-DTAT_VERSION=\"v${TAT_VERSION}\"")

# Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Math Library
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(MKL)
if(MKL_FOUND)
  add_definitions("-DTAT_USE_MKL")
else()
  find_package(CBLAS)
  find_package(LAPACKE)
endif()

# Dependence
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/rang/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/args)

if(MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIRS})
  link_directories(${MKL_LIBRARY_DIRS})
  link_libraries(-Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
else()
  include_directories(${CBLAS_INCLUDE_DIRS})
  link_directories(${CBLAS_LIBRARY_DIRS})
  link_libraries(${CBLAS_LIBRARIES})
  include_directories(${LAPACKE_INCLUDE_DIRS})
  link_directories(${LAPACKE_LIBRARY_DIRS})
  link_libraries(${LAPACKE_LIBRARIES})
endif()

# Style
add_custom_target(style COMMAND clang-format -style=file -i ${PROJECT_SOURCE_DIR}/example/*/*.hpp ${PROJECT_SOURCE_DIR}/example/*.cpp ${PROJECT_SOURCE_DIR}/include/*.hpp)
add_custom_target(doc COMMAND cd ${PROJECT_SOURCE_DIR} && doxygen)
add_custom_target(speed COMMAND time ${PROJECT_BINARY_DIR}/Heisenberg_MPS_SU -L100 -D12 1>&-)

# Target
file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/example/*.cpp)
enable_testing()
foreach(FILE ${CPP_SRC})
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME} ${FILE})
  add_test(NAME ${NAME} COMMAND ${PROJECT_SOURCE_DIR}/check_output ${PROJECT_BINARY_DIR}/${NAME} ${PROJECT_SOURCE_DIR}/diff/${NAME}.out)
endforeach(FILE)
