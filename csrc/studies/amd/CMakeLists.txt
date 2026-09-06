cmake_minimum_required(VERSION 3.21)
project(amd_gemm_study LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(day01_vector_add src/day01/vector_add.hip)
add_executable(day01_reduction src/day01/reduction.hip)

target_compile_options(day01_vector_add PRIVATE -Wall -Wextra)
target_compile_options(day01_reduction PRIVATE -Wall -Wextra)
