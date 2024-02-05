# Mission

find_package(jsoncpp REQUIRED)
add_library(mission STATIC mission.cpp)
target_link_libraries(mission PUBLIC re2::re2 ggml)
target_link_libraries(mission PUBLIC jsoncpp_lib)
target_include_directories(mission PUBLIC .)

add_executable(pipeline pipeline.cpp)
target_link_libraries(pipeline PRIVATE mission common)

