#set(CMAKE_CXX_FLAGS "-g") # 在debug时使用，在完成debug后应当注释掉。
set(TARGET main)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
