# Mission
#set(CMAKE_CXX_FLAGS "-g") # 在debug时使用，在完成debug后应当注释掉。

find_package(jsoncpp REQUIRED)
add_library(mission STATIC mission.cpp)
target_link_libraries(mission PUBLIC re2::re2 ggml)
target_link_libraries(mission PUBLIC jsoncpp_lib)
target_include_directories(mission PUBLIC .)

#add_executable(pipeline pipeline.cpp) # debug时使用
add_library(pipeline STATIC pipeline.cpp) # pip install .时使用
target_link_libraries(pipeline PUBLIC mission common)
target_compile_features(pipeline PRIVATE cxx_std_17)

