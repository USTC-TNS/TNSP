cmake_minimum_required(VERSION 3.8)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")
project(PyTAT LANGUAGES CXX)

# Include TAT target
if(NOT TARGET TAT)
  set(TAT_BUILD_TEST OFF CACHE BOOL "Override TAT option")
  set(TAT_USE_MPI OFF CACHE BOOL "Override TAT option")
  add_subdirectory(TAT EXCLUDE_FROM_ALL)
endif()
if(DEFINED SKBUILD_PROJECT_VERSION)
  target_compile_definitions(TAT INTERFACE TAT_VERSION=\"${SKBUILD_PROJECT_VERSION}\")
endif()

option(TAT_BUILD_PYTAT "Build PyTAT" ON)
if(TAT_BUILD_PYTAT)
  # load Pybind11
  if(EXISTS ${CMAKE_SOURCE_DIR}/pybind11)
    # Try to find in the top level directory
    add_subdirectory(${CMAKE_SOURCE_DIR}/pybind11 pybind11)
    message(STATUS "Enable python(local)")
  else()
    # Find in the system path
    find_package(pybind11 REQUIRED)
    message(STATUS "Enable python(system)")
  endif()

  # Define PyTAT
  file(GLOB PYTAT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  set(Symmetries No BoseZ2 BoseU1 FermiU1 FermiU1BoseZ2 FermiU1BoseU1 FermiZ2 FermiU1FermiU1)
  set(Shorts S D C Z)
  set(Scalars float double std::complex<float> std::complex<double>)
  foreach(Sym IN LISTS Symmetries)
    foreach(value RANGE 0 3)
      list(GET Shorts ${value} Short)
      list(GET Scalars ${value} Scalar)
      set(part_name dealing_tensor_${Sym}_${Short})
      set(FUNCTION_NAME dealing_Tensor_${Sym}_${Short})
      set(SCALAR_NAME ${Scalar})
      set(SYMMETRY_NAME ${Sym}Symmetry)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dealing_tensor.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/PyTAT-generated/${part_name}.cpp @ONLY)
      list(APPEND PYTAT_SRC ${CMAKE_CURRENT_BINARY_DIR}/PyTAT-generated/${part_name}.cpp)
    endforeach()
  endforeach()
  pybind11_add_module(PyTAT ${PYTAT_SRC})
  target_link_libraries(PyTAT PRIVATE TAT)
  target_include_directories(PyTAT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(PyTAT PROPERTIES OUTPUT_NAME TAT)

  # Install PyTAT
  install(TARGETS PyTAT LIBRARY DESTINATION .)
endif()
