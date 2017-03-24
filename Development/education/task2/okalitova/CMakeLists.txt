cmake_minimum_required(VERSION 3.3)

set(MODULE_NAME okalitova)

project(${MODULE_NAME})

set(CMAKE_CXX_STANDARD 14)

# set include directories
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${task1_SOURCE_DIR}/include)

# set source files
set(SOURCE_FILES main.cpp)
add_executable(${MODULE_NAME} ${SOURCE_FILES})

# linking
target_link_libraries(${MODULE_NAME} ${OpenCL_LIBRARY})

# copy kernel file to bin directory
file(COPY matrixMultiplication.cl DESTINATION .)