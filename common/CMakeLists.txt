# cuda_test_project/common/CMakeLists.txt
add_library(aoc_common INTERFACE
    cuda_utils.cuh
)

target_include_directories(aoc_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})