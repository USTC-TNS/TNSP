cmake_minimum_required(VERSION 3.13)

# cmake variables:
# TAT_INCLUDE_PATH
# CMAKE_BUILD_TYPE, CMAKE_CXX_FLAGS
# TAT_USE_MPI, TAT_PYTHON_MODULE, TAT_FORCE_VERSION
# TAT_MATH_LIBRARIES, BLA_VENDOR, BLA_STATIC
# PYBIND11_PYTHON_VERSION, PYTHON_EXECUTABLE

add_library(TAT INTERFACE)

# Find TAT_INCLUDE_PATH
find_path(TAT_INCLUDE_PATH NAMES "TAT/TAT.hpp" HINTS ${PROJECT_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/include/ REQUIRED)
message("-- TAT headers found at ${TAT_INCLUDE_PATH}")
target_include_directories(TAT INTERFACE ${TAT_INCLUDE_PATH})

# std=c++17
target_compile_features(TAT INTERFACE cxx_std_17)

option(TAT_USE_MPI "Use mpi for TAT" ON)
set(TAT_PYTHON_MODULE TAT CACHE STRING "Set python binding module name")
set(TAT_FORCE_VERSION 0.2.4 CACHE STRING "Force set TAT version")

# These macros used for record compiling information
target_compile_definitions(TAT INTERFACE TAT_VERSION="${TAT_FORCE_VERSION}")
target_compile_definitions(TAT INTERFACE TAT_BUILD_TYPE="$<IF:$<STREQUAL:${CMAKE_BUILD_TYPE},>,Default,${CMAKE_BUILD_TYPE}>")
target_compile_definitions(TAT INTERFACE TAT_COMPILER_INFORMATION="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# msvc need utf-8 and __cplusplus macro
target_compile_options(TAT INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
target_compile_options(TAT INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>")

# try to enable mpi
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

# link lapack and blas
if(DEFINED TAT_MATH_LIBRARIES)
   message("-- Use customed math libraries")
   target_link_libraries(TAT INTERFACE ${TAT_MATH_LIBRARIES})
else()
   if(EMSCRIPTEN)
      message("-- Use emscripten blas and lapack")
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/liblapack.a)
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/libblas.a)
      target_link_libraries(TAT INTERFACE ${CMAKE_SOURCE_DIR}/emscripten/libf2c.a)
   else()
      find_package(BLAS REQUIRED)
      find_package(LAPACK REQUIRED)
      target_link_libraries(TAT INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
      # check whether using mkl
      if(BLAS_LIBRARIES MATCHES mkl)
         message("-- Using mkl")
         target_compile_definitions(TAT INTERFACE TAT_USE_MKL_TRANSPOSE)
         target_compile_definitions(TAT INTERFACE TAT_USE_MKL_GEMM_BATCH)
      else()
         message("-- Not using mkl")
      endif()
   endif()
endif()

# PyTAT
set(TAT_BUILD_PYTAT OFF)
if(NOT EMSCRIPTEN)
   if(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
      add_subdirectory(pybind11)
      message("-- Enable python(local)")
      set(TAT_BUILD_PYTAT ON)
   else()
      find_package(pybind11 QUIET)
      if(pybind11_FOUND)
         message("-- Enable python(system)")
         set(TAT_BUILD_PYTAT ON)
      else()
         message("-- Disable python since pybind11 not found, try install pybind11 or put it into TAT directory")
      endif()
   endif()
   if(NOT EXISTS ${PROJECT_SOURCE_DIR}/PyTAT/PyTAT.cpp)
      set(TAT_BUILD_PYTAT OFF)
   endif()
endif()
if(TAT_BUILD_PYTAT)
   file(GLOB PYTAT_SRC ${PROJECT_SOURCE_DIR}/PyTAT/generated_code/*.cpp ${PROJECT_SOURCE_DIR}/PyTAT/*.cpp)
   pybind11_add_module(PyTAT ${PYTAT_SRC} EXCLUDE_FROM_ALL)
   target_link_libraries(PyTAT PRIVATE TAT)
   set_target_properties(PyTAT PROPERTIES OUTPUT_NAME ${TAT_PYTHON_MODULE})
   target_compile_definitions(PyTAT PRIVATE TAT_PYTHON_MODULE=${TAT_PYTHON_MODULE})
endif()
