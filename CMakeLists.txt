cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GaussGen)
option(EIGEN3_INCLUDE_DIR "Path to the Eigen3 include directory")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if (NOT EIGEN3_INCLUDE_DIR)
  if (DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "EIGEN3_INCLUDE_DIR option is not set")
  endif()
endif()
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT REQUIRED COMPONENTS RIO Net Core)
include(${ROOT_USE_FILE})
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_executable(gaussgen-test ${SOURCES})
target_include_directories(gaussgen-test
  PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gaussgen-test
  ${ROOT_LIBRARIES})

add_library(gaussgen SHARED ${SOURCES})
target_include_directories(gaussgen
  PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  PULBIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gaussgen
  ${ROOT_LIBRARIES})
set_target_properties(gaussgen
  PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

install(TARGETS gaussgen-test DESTINATION bin)
install(TARGETS gaussgen
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/gaussgen)
