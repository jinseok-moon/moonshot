cmake_minimum_required(VERSION 3.21)

project(moonshot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG)
endif()

option(ENABLE_CUDA "Build CUDA examples" ON)

find_package(OpenMP QUIET)

# ---- CUDA -------------------------------------------------------------------
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(src/cuda)
endif()

message(STATUS "Backends enabled: CUDA=${ENABLE_CUDA}")
