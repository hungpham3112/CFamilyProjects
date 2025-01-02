cmake_minimum_required(VERSION 3.10)
project(day_10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(main src/main.cpp src/rectangle.cpp)

target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/include())