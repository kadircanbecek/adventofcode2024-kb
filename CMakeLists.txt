cmake_minimum_required(VERSION 3.18)

# First, explicitly tell CMake where to find CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# The key difference is that we need to set CUDA architectures BEFORE the project() command
# We're setting multiple architectures to ensure compatibility
set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75")

# Now we can declare our project
project(CudaTestProject CUDA CXX)

# Print diagnostic information to help us understand our CUDA setup
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Set C++ standard for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include our common CUDA settings
include(cmake/cuda_common.cmake)

# Add our subdirectories
add_subdirectory(common)
add_subdirectory(day01)