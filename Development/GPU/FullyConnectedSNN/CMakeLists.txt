project(FullyConnectedSNN)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES main.cpp FullyConnectedNN.cpp Layer.cpp Neuron.cpp NetworkInitializer.cpp InputReader.cpp)
set(HEADER_FILES FullyConnectedNN.h Layer.h Neuron.h NetworkInitializer.h InputReader.h)
add_executable(FullyConnectedSNN ${SOURCE_FILES})

target_link_libraries(FullyConnectedSNN ${OpenCL_LIBRARY})

# copy resources to working directory
file(COPY res/network DESTINATION ./res)
file(COPY res/network_test DESTINATION ./res)
file(COPY mnist DESTINATION .)