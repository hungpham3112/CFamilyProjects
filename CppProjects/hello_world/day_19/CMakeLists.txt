cmake_minimum_required(VERSION 3.10)
project(LearnOOP VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main src/cat.cpp src/main.cpp)

target_include_directories(main PUBLIC "${PROJECT_SOURCE_DIR}/include")