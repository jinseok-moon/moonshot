# CUDA-specific common interface target
add_library(cuda_common INTERFACE)
target_compile_features(cuda_common INTERFACE cxx_std_20)
target_include_directories(cuda_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_SOURCE_DIR}/src/common)
target_link_libraries(cuda_common INTERFACE
  CUDA::cudart
  CUDA::cublas)

add_custom_target(cuda_examples)

function(cuda_example_add_executable NAME)
  add_executable(${NAME} ${ARGN})
  target_link_libraries(${NAME} PRIVATE cuda_common)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(${NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif()
  set_property(TARGET ${NAME} PROPERTY FOLDER "cuda-samples")
  set_property(TARGET ${NAME} PROPERTY CUDA_ARCHITECTURES "80;86;89")
  add_dependencies(cuda_examples ${NAME})
endfunction()

# Discover immediate child example directories
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child}
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
    add_subdirectory(${child})
  endif()
endforeach()
