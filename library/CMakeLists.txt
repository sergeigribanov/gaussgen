file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(gaussgen SHARED ${SOURCES})
target_include_directories(gaussgen
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(gaussgen PUBLIC Eigen3::Eigen ${ROOT_LIBRARIES})
set_target_properties(gaussgen
  PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

install(TARGETS gaussgen
  EXPORT gaussgenTargets
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include)
