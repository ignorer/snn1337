project(FullyConnected)

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp)
add_executable(FullyConnected ${SOURCE_FILES})

target_link_libraries(FullyConnected ${OpenCL_LIBRARY})

#file(COPY kernel.cl DESTINATION .)