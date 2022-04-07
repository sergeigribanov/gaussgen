file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(gaussgen-test ${SOURCES})
target_link_libraries(gaussgen-test PUBLIC gaussgen)

