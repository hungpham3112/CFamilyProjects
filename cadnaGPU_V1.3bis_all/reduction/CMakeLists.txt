cmake_minimum_required(VERSION 3.15..3.27 FATAL_ERROR)

project(test_reduction)

if(NOT DEFINED FILENAME)
    set(FILENAME "reduction_cadna.cu")
endif()

option(FILENAME "Specify the source file" "reduction_cadna.cu")

set(CMAKE_PREFIX_PATH "$ENV{HOME}/miniforge3/envs/School/lib/python3.11/site-packages/torch")

find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)

set(HELPER_CUDA "$ENV{HOME}/cadnaGPU_V1.3bis_all/cuda-samples/Common/")
set(CMAKE_CXX_COMPILER "clang++")
set(CUTLASS_INCLUDE_DIRS "/cutlass/include/")
set(Python3_INCLUDE_DIRS "$ENV{HOME}/miniforge3/envs/School/include/python3.11/")
set(CADNA 
    "$ENV{HOME}/cadnaGPU_V1.3bis_all/include/" 
    "$ENV{HOME}/cadnaGPU_V1.3bis_all/C/"
    "$ENV{HOME}/cadnaGPU_V1.3bis_all/Cgpu/")

set(REDUCTION_COMMON
    "$ENV{HOME}/CFamilyProjects/cadnaGPU_V1.3bis_all/cuda-samples/Common/")

include_directories(${TORCH_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${CADNA} ${HELPER_CUDA} ${REDUCTION_COMMON})
if(CMAKE_CXX_COMPILER)
    message(STATUS "MY_VARIABLE is set: ${CMAKE_CXX_COMPILER}")
else()
    message(STATUS "MY_VARIABLE is not set")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# List your source files
set(SOURCES ${FILENAME})

link_directories($ENV{HOME}/cadnaGPU_V1.3bis_all/lib/)

# Use add_executable with the source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the target with Torch libraries
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} Python3::Python cadnaC)

# Set the C++ standard for the target
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

