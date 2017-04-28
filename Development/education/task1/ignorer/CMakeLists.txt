cmake_minimum_required(VERSION 2.8.4)
project(ignorer)

set(CMAKE_CXX_STANDARD 14)

include_directories(${task1_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp)
add_executable(ignorer ${SOURCE_FILES})

target_link_libraries(ignorer ${OpenCL_LIBRARY})

file(COPY ignorerKernel.cl DESTINATION .)
