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

# Binary Size
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# Optimize and Warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fwhole-program")
endif()

# Find Math Library
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(MKL)
if(MKL_FOUND)
  add_definitions("-DTAT_USE_MKL")
else()
  find_package(CBLAS)
  find_package(LAPACKE)
endif()
find_package(OpenMP)

# Include Path
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/hptt/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/rang/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/args)
if(MKL_FOUND)
  include_directories(${MKL_INCLUDE_DIRS})
else()
  include_directories(${CBLAS_INCLUDE_DIRS})
  include_directories(${LAPACKE_INCLUDE_DIRS})
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Link Path
link_directories(${PROJECT_SOURCE_DIR}/deps/hptt/lib)
if(MKL_FOUND)
  link_directories(${MKL_LIBRARY_DIRS})
else()
  link_directories(${CBLAS_LIBRARY_DIRS})
  link_directories(${LAPACKE_LIBRARY_DIRS})
endif()

# Link
link_libraries(-Wl,--start-group -lhptt ${OpenMP_CXX_LIBRARIES} -Wl,--end-group)
if(MKL_FOUND)
  link_libraries(-Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
else()
  link_libraries(${CBLAS_LIBRARIES} ${LAPACKE_LIBRARIES})
endif()

# Style
add_custom_target(style COMMAND clang-format -style=file -i ${PROJECT_SOURCE_DIR}/example/*/*.hpp ${PROJECT_SOURCE_DIR}/example/*.cpp ${PROJECT_SOURCE_DIR}/include/TAT.hpp)
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
