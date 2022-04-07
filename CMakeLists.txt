cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GaussGen)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3 REQUIRED NO_MODULE)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT REQUIRED COMPONENTS RIO Net Core)
include(${ROOT_USE_FILE})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/env-gaussgen.sh.in env-gaussgen.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rootlogon.C.in rootlogon.C)

include(GNUInstallDirs)
add_subdirectory(library)
add_subdirectory(test)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gaussgenConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/gaussgenConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/gaussgen/cmake"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gaussgenConfigVersion.cmake"
  VERSION "${GAUSSGEN_CORE_VERSION_MAJOR}.${GAUSSGEN_CORE_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion)

export(TARGETS gaussgen FILE gaussgenTargets.cmake)
export(PACKAGE gaussgen)
install(EXPORT gaussgenTargets DESTINATION lib/cmake)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/gaussgenConfig.cmake
  DESTINATION lib/cmake)
install(FILES ${PROJECT_BINARY_DIR}/env-gaussgen.sh DESTINATION bin)
install(FILES ${PROJECT_BINARY_DIR}/rootlogon.C DESTINATION share/gaussgen)
