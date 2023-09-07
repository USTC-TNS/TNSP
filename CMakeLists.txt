cmake_minimum_required(VERSION 3.8)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")
project(TAT LANGUAGES CXX)

# Check if it is main project
set(TAT_MAIN_PROJECT OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(TAT_MAIN_PROJECT ON)
endif()

# Add some misc config for cmake if it is main project
if(TAT_MAIN_PROJECT)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(PROJECT_DESCRIPTION "TAT is a header-only c++ tensor library with support for Abelian symmetry tensor and fermi tensor.")
  set(PROJECT_HOMEPAGE_URL "https://github.com/USTC-TNS/TAT")
endif()

# common used cmake variables:
# CMAKE_BUILD_TYPE, CMAKE_CXX_FLAGS
# TAT_USE_MPI, TAT_MATH_LIBRARIES, BLA_VENDOR, BLA_STATIC
# PYBIND11_PYTHON_VERSION, PYTHON_EXECUTABLE
# TAT_BUILD_PYTAT, TAT_BUILD_TETRAUX, TAT_BUILD_TEST

# Add TAT target
set(TAT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-build.cmake" @ONLY)
include("${CMAKE_CURRENT_BINARY_DIR}/TAT-config-build.cmake")

# Install TAT
if(TAT_MAIN_PROJECT)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/TAT TYPE INCLUDE)
  set(TAT_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-install.cmake" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-install.cmake" DESTINATION lib/cmake/TAT RENAME TAT-config.cmake)

  set(PKG_CONFIG_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/TAT.pc")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT.pc.in" "${PKG_CONFIG_FILE_NAME}" @ONLY)
  install(FILES "${PKG_CONFIG_FILE_NAME}" DESTINATION lib/pkgconfig)

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.rst DESTINATION share/licenses/TAT)
endif()

# Pybind11
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
  add_subdirectory(pybind11)
  set(TAT_PYBIND11_FOUND ON)
  message(STATUS "Enable python(local)")
else()
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    message(STATUS "Enable python(system)")
    set(TAT_PYBIND11_FOUND ON)
  else()
    message(STATUS "Disable python since pybind11 not found, try install pybind11 or put it into TAT directory")
    set(TAT_PYBIND11_FOUND OFF)
  endif()
endif()

# PyTAT
option(TAT_BUILD_PYTAT "Build PyTAT" ON)
if(TAT_BUILD_PYTAT AND TAT_PYBIND11_FOUND)
  file(GLOB PYTAT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/PyTAT/*.cpp)
  set(Symmetries No Z2 U1 Fermi FermiZ2 FermiU1 Parity FermiFermi)
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
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PyTAT/dealing_tensor.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/PyTAT-generated/${part_name}.cpp @ONLY)
      list(APPEND PYTAT_SRC ${CMAKE_CURRENT_BINARY_DIR}/PyTAT-generated/${part_name}.cpp)
    endforeach()
  endforeach()
  pybind11_add_module(PyTAT ${PYTAT_SRC})
  target_link_libraries(PyTAT PRIVATE TAT)
  target_include_directories(PyTAT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PyTAT)
  set_target_properties(PyTAT PROPERTIES OUTPUT_NAME TAT)
endif()

# tetraux
option(TAT_BUILD_TETRAUX "Build tetraux" ON)
if(TAT_BUILD_TETRAUX AND TAT_PYBIND11_FOUND)
  pybind11_add_module(tetraux ${CMAKE_CURRENT_SOURCE_DIR}/tetraux/tetraux.cpp)
  target_compile_features(tetraux PRIVATE cxx_std_17)
endif()

# Add test
option(TAT_BUILD_TEST "Build test of TAT" ON)
if(TAT_BUILD_TEST)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    add_custom_target(test_executables)
    enable_testing()
    file(GLOB CPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    foreach(FILE ${CPP_SRC})
      get_filename_component(NAME ${FILE} NAME_WE)
      add_executable(${NAME} ${FILE})
      target_link_libraries(${NAME} TAT GTest::gtest_main)
      gtest_discover_tests(${NAME})
      add_dependencies(test_executables ${NAME})
    endforeach(FILE)
  else()
    message(STATUS "GTest not found, do not build test, even TAT_BUILD_TEST=ON")
  endif()
endif()
