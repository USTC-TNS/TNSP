cmake_minimum_required(VERSION 3.12)
set(CMAKE_VERBOSE_MAKEFILE ON)
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
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Binary Size
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # CL Option to get small binary size here
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

# Cl specified
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # several patch to let cl could compile
  add_definitions("-D__restrict__=__restrict")
  add_definitions("-D_Complex=complex")
  set(ENV{EXTREME} 1)
endif()

# Optimize and Warning
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # Cl option is different with gcc
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -O0")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fwhole-program")
endif()

# Find MKL
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(MKL)

# Include Path
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/hptt/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/rang/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/args)
include_directories(${MKL_INCLUDE_DIRS})

# Link Path
link_directories(${PROJECT_SOURCE_DIR}/deps/hptt/lib)
link_directories(${MKL_LIBRARY_DIRS})

# Extreme
if(DEFINED ENV{EXTREME})
  add_definitions("-DTAT_EXTREME")
endif()

# Link
if(DEFINED ENV{STATIC})
  link_libraries(-static-libgcc -static-libstdc++)
  link_libraries(-Wl,-Bstatic -ljemalloc_pic -Wl,-Bdynamic)
  if(NOT DEFINED ENV{EXTREME})
    link_libraries(-Wl,-Bstatic -lhptt -Wl,-Bdynamic)
  endif()
  link_libraries(-Wl,-Bstatic -Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group -Wl,-Bdynamic)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
else()
  link_libraries(-ljemalloc)
  if(NOT DEFINED ENV{EXTREME})
    link_libraries(-lhptt)
  endif()
  link_libraries(-Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
endif()

# Style
add_custom_target(style COMMAND astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type `find ${PROJECT_SOURCE_DIR}/example -type f` `find ${PROJECT_SOURCE_DIR}/include -type f`)

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
