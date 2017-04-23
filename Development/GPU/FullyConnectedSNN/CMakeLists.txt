project(FullyConnectedSNN)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GPU_SOURCE_DIR}/include)

set(SOURCE_FILES FullyConnectedNN.cpp Layer.cpp Neuron.cpp NetworkInitializer.cpp InputReader.cpp ClStructHolder.cpp main.cpp)
set(HEADER_FILES FullyConnectedNN.h Layer.h Neuron.h NetworkInitializer.h InputReader.h ClStructHolder.h)
add_executable(FullyConnectedSNN ${SOURCE_FILES})

target_link_libraries(FullyConnectedSNN ${OpenCL_LIBRARY})
set_property(TARGET FullyConnectedSNN PROPERTY CXX_STANDARD 14)

# copy resources to working directory
file(COPY neuron.cl DESTINATION .)
file(COPY res/network DESTINATION ./res)
file(COPY res/network_test DESTINATION ./res)
file(COPY mnist DESTINATION .)