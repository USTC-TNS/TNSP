cmake_minimum_required(VERSION 3.8)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")
project(PyTAT)

add_subdirectory(parent)
if(DEFINED SKBUILD_PROJECT_VERSION)
  target_compile_definitions(TAT INTERFACE TAT_VERSION=\"${SKBUILD_PROJECT_VERSION}\")
endif()
install(TARGETS PyTAT LIBRARY DESTINATION .)
