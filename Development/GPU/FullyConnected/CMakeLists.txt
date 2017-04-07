project(FullyConnected)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp FullyConnectedNN.cpp Layer.cpp ClStructHolder.cpp)
set(HEADER_FILES Layer.h FullyConnectedNN.h ClStructHolder.h)
add_executable(FullyConnected ${SOURCE_FILES})

target_link_libraries(FullyConnected ${OpenCL_LIBRARY})

# copy kernel to working directory
file(COPY neuron.cl DESTINATION .)
# copy resources to working directory
file(COPY network_digits DESTINATION .)
file(COPY network_xor DESTINATION .)
file(COPY input_digits DESTINATION .)
file(COPY output_digits DESTINATION .)