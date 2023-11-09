project(gpt C CUDA)
cmake_minimum_required(VERSION 3.11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "-w")
add_executable(cpu_main main.c)
target_link_libraries(cpu_main m)
find_package(CUDA 11.7 REQUIRED)
enable_language(CUDA)

message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
add_executable(gpu_main main.cu)

