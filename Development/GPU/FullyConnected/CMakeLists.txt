project(FullyConnected)

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp FullyConnectedNN.cpp Layer.cpp)
set(HEADER_FILES Layer.h FullyConnectedNN.h)
add_executable(FullyConnected ${SOURCE_FILES})

target_link_libraries(FullyConnected ${OpenCL_LIBRARY})

# copy kernel to working directory
file(COPY neuron.cl DESTINATION .)
# copy resources to working directory
file(COPY network_digits DESTINATION .)
file(COPY network_xor DESTINATION .)
