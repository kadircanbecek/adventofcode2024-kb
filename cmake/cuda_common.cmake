set(CMAKE_CUDA_ARCHITECTURES "52;60;70;75;80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

function(add_daily_challenge DAY_NUMBER)
    set(target_name "day${DAY_NUMBER}")
    add_executable(${target_name} main.cu)
    target_link_libraries(${target_name} PRIVATE aoc_common)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"  # Use the architectures we defined
    )
endfunction()