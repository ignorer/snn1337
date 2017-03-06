set(MODULE_NAME zuev-stepan)

project(${MODULE_NAME})

set(CMAKE_CXX_STANDARD 14)

# set include directories
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${task2_SOURCE_DIR}/include)

# set source files
set(SOURCE_FILES main.cpp)
add_executable(${MODULE_NAME} ${SOURCE_FILES})

# linking
target_link_libraries(${MODULE_NAME} ${OpenCL_LIBRARY})

# copy kernel file to bin directory
configure_file(kernel.cl kernel.cl) # Workaround to copy dile everytime
file(COPY kernel.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
