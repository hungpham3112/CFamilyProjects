cmake_minimum_required(VERSION 3.15..3.27 FATAL_ERROR)

project(main)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/miniforge3/envs/DL-verification/lib/python3.8/site-packages/torch")

find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)

set(CUTLASS_INCLUDE_DIRS "/cutlass/cutlass-3.3.0/include/" "/cutlass/tools/util/include/" "/cutlass/include/")
set(CUTBLAS_INCLUDE_DIRS "/usr/local/cuda-12.3/targets/x86_64-linux/include/")
set(Python3_INCLUDE_DIRS "/root/miniforge3/envs/DL-verification/include/python3.8/")

include_directories(${TORCH_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} %{CUTBLAS_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# List your source files
set(SOURCES main.cpp)

# Use add_executable with the source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the target with Torch libraries
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Python3::Python)

# Set the C++ standard for the target
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

