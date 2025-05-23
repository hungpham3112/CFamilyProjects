cmake_minimum_required(VERSION 3.10)

project(LearnCompilation)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(LearnCompilation script/learning.cpp)

add_custom_command(TARGET LearnCompilation
    POST_BUILD
    COMMAND $<TARGET_FILE:LearnCompilation>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)