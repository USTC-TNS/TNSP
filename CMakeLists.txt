cmake_minimum_required(VERSION 3.14)
set(CMAKE_VERBOSE_MAKEFILE ON)
project (TAT)

execute_process(COMMAND git describe --tags OUTPUT_VARIABLE TAT_VERSION_ORIGIN)
string(STRIP ${TAT_VERSION_ORIGIN} TAT_VERSION)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/hptt/include")
include_directories("${PROJECT_SOURCE_DIR}/rang/include")
include_directories("${PROJECT_SOURCE_DIR}/args")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -DTAT_VERSION=\\\"${TAT_VERSION}\\\"")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fwhole-program")
endif()

#set(BUILD_SHARED_LIBS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(MKL)

if(DEFINED ENV{STATIC})
  link_libraries(-static-libgcc -static-libstdc++)
  link_libraries(-Wl,-Bstatic -ljemalloc_pic -Wl,-Bdynamic)
  link_libraries(${PROJECT_SOURCE_DIR}/hptt/lib/libhptt.a)
  link_libraries(-Wl,-Bstatic -Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group -Wl,-Bdynamic)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
else()
  link_libraries(-ljemalloc)
  link_libraries(${PROJECT_SOURCE_DIR}/hptt/lib/libhptt.so)
  link_libraries(-Wl,--start-group ${INT_LIB} ${SEQ_LIB} ${COR_LIB} -Wl,--end-group)
  link_libraries(${THR_LIB} ${MAT_LIB} ${LDL_LIB})
endif()

add_custom_target(style COMMAND astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type *.cpp `find ./include -type f`)

file(GLOB CPP_SRC *.cpp)

foreach(FILE ${CPP_SRC})
  get_filename_component(NAME ${FILE} NAME_WE)
  add_executable(${NAME}.out ${FILE})
endforeach(FILE)
