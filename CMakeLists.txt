cmake_minimum_required(VERSION 3.5)
# set(CMAKE_VERBOSE_MAKEFILE ON)
project (TAT)

# Version
execute_process(COMMAND git -C ${PROJECT_SOURCE_DIR} describe --tags OUTPUT_VARIABLE TAT_VERSION_ORIGIN RESULT_VARIABLE TAT_VERSION_ERROR)
if(NOT ${TAT_VERSION_ERROR})
  string(STRIP ${TAT_VERSION_ORIGIN} TAT_VERSION)
else()
  set(TAT_VERSION unknown)
endif()
message("-- Version is ${TAT_VERSION}")
add_definitions("-DTAT_VERSION=\"${TAT_VERSION}\"")

# Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Binary Size
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # TODO: CL Option to get small binary size here
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

# Cl specified
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # TODO: several patch to let cl could compile
  add_definitions("-D__restrict__=__restrict")
  add_definitions("-D_Complex=complex")
  set(ENV{EXTREME} 1)
endif()

# Optimize and Warning
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # TODO: Cl option is different with gcc
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fwhole-program")
endif()

# Find Math Library
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(MKL)
find_package(CBLAS)
find_package(LAPACKE)
if(MKL_FOUND)
  add_definitions("-DTAT_USE_MKL")
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
add_custom_target(style COMMAND clang-format -style=file -i ${PROJECT_SOURCE_DIR}/example/*.cpp ${PROJECT_SOURCE_DIR}/include/TAT.hpp)
add_custom_target(doc COMMAND cd ${PROJECT_SOURCE_DIR} && doxygen)

# Target
if(WIN32)
  set(EXTNAME)
else()
  set(EXTNAME .out)
endif()
file(GLOB CPP_SRC ${PROJECT_SOURCE_DIR}/example/*.cpp)
foreach(FILE ${CPP_SRC})
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME}${EXTNAME} ${FILE})
endforeach(FILE)
