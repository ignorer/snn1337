project(Perceptron)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp FullyConnectedNN.cpp Layer.cpp ClStructHolder.cpp InputReader.cpp)
set(HEADER_FILES Layer.h FullyConnectedNN.h ClStructHolder.h InputReader.h)
add_executable(Perceptron ${SOURCE_FILES})

target_link_libraries(Perceptron ${OpenCL_LIBRARY})

# copy kernel to working directory
file(COPY batchedNeuron.cl DESTINATION .)
# copy resources to working directory
file(COPY network_mnist DESTINATION .)
file(COPY mnist DESTINATION .)
