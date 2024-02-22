cmake_minimum_required(VERSION 3.8)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")
project(TAT LANGUAGES CXX)

# Add TAT target
set(TAT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-build.cmake" @ONLY)
include("${CMAKE_CURRENT_BINARY_DIR}/TAT-config-build.cmake")

# Install TAT with cmake
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/TAT TYPE INCLUDE)
set(TAT_INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-install.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TAT-config-install.cmake" DESTINATION lib/cmake/TAT RENAME TAT-config.cmake)

# Install TAT with pkg-config
set(PKG_CONFIG_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/TAT.pc")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/TAT.pc.in" "${PKG_CONFIG_FILE_NAME}" @ONLY)
install(FILES "${PKG_CONFIG_FILE_NAME}" DESTINATION lib/pkgconfig)

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
      target_include_directories(${NAME} PRIVATE ${GTEST_INCLUDE_DIRS})
      target_link_libraries(${NAME} PRIVATE TAT ${GTEST_BOTH_LIBRARIES})
      gtest_discover_tests(${NAME})
      add_dependencies(test_executables ${NAME})
    endforeach()
  else()
    message(STATUS "GTest not found, do not build test, even if TAT_BUILD_TEST=ON")
  endif()
endif()
